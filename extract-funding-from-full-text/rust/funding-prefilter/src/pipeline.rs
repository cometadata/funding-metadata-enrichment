//! End-to-end pipeline orchestration.
//!
//! Walks an input directory of parquet shards, applies the two-stage regex
//! gate ([`crate::gate_document`]) to every `status == "ok"` row in parallel
//! via rayon, and writes one candidate parquet per input shard (see
//! [`crate::parquet_io::write_candidates`]).
//!
//! Key properties:
//! * **Resumability** — a shard whose output file already exists and is
//!   non-empty is skipped unless `--force` is set. The 0-row "empty candidate"
//!   file written by `write_candidates` therefore distinguishes "processed,
//!   no hits" from "never ran".
//! * **Error isolation** — each shard runs inside a `catch_unwind` panic
//!   boundary and its own `Result` match arm. One corrupt shard increments
//!   `shards_errored` but does not kill sibling workers.
//! * **Progress** — a background thread emits one `[progress] ...` line per
//!   `progress_interval_secs` to stderr. It shuts down cleanly via an
//!   `AtomicBool` flag polled once per second so the main thread doesn't wait
//!   a full interval on exit.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::gate_document;
use crate::parquet_io::{read_shard_rows, write_candidates, CandidateRow};
use crate::patterns::Patterns;

/// User-facing configuration for a pipeline run.
pub struct PipelineConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub patterns_path: PathBuf,
    pub threads: usize,
    pub force: bool,
    pub progress_interval_secs: u64,
}

/// Terminal counters reported back to the caller (and CLI).
pub struct PipelineStats {
    pub shards_total: usize,
    pub shards_done: usize,
    /// Shards that were already processed (output file existed, non-empty,
    /// `--force` not set).
    pub shards_skipped: usize,
    pub shards_errored: usize,
    pub rows_read: u64,
    pub candidates_kept: u64,
    pub elapsed_secs: f64,
}

/// Outcome of a single shard's worker — used by the parent `for_each` to bump
/// the right atomic counter.
enum ShardOutcome {
    Processed,
    Skipped,
}

/// Walk `input` for `*.parquet` files, returning them sorted by path for
/// deterministic dispatch order.
pub(crate) fn enumerate_shards(input: &Path) -> Result<Vec<PathBuf>> {
    let mut shards: Vec<PathBuf> = WalkDir::new(input)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.path().to_path_buf())
        .filter(|path| {
            path.extension()
                .map(|ext| ext == "parquet")
                .unwrap_or(false)
        })
        .collect();

    if shards.is_empty() {
        return Err(anyhow!(
            "no *.parquet shards found under {}",
            input.display()
        ));
    }

    shards.sort();
    Ok(shards)
}

/// Process a single shard: read rows, apply the gate, write the candidate
/// parquet. Returns `Skipped` if the output already exists and `--force` is
/// off, `Processed` otherwise.
fn process_shard(
    shard_path: &Path,
    cfg: &PipelineConfig,
    patterns: &Patterns,
    rows_counter: &AtomicU64,
    cand_counter: &AtomicU64,
) -> Result<ShardOutcome> {
    // Preserve the shard's position relative to `--input` inside `--output`,
    // so e.g. `input/2001/shard.parquet` -> `output/2001/shard.parquet`.
    let rel = shard_path
        .strip_prefix(&cfg.input)
        .with_context(|| {
            format!(
                "shard path {} is not inside input root {}",
                shard_path.display(),
                cfg.input.display()
            )
        })?;
    let out = cfg.output.join(rel);

    // Resumability gate: non-empty output file means "already processed".
    // A 0-byte file would indicate a partial write from a crashed run — we
    // deliberately re-process that by requiring `len() > 0`.
    if !cfg.force {
        if let Ok(meta) = std::fs::metadata(&out) {
            if meta.len() > 0 {
                return Ok(ShardOutcome::Skipped);
            }
        }
    }

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("creating output parent directory {}", parent.display())
        })?;
    }

    // Stable shard identifier stamped into every candidate row; we use the
    // file stem (e.g. `arXiv_src_9104_001`) so the merge step can trace each
    // id back to its source shard without replaying the full path.
    let shard_id = shard_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            anyhow!(
                "cannot derive shard_id from non-unicode file stem of {}",
                shard_path.display()
            )
        })?
        .to_string();

    let mut candidates: Vec<CandidateRow> = Vec::new();
    let iter = read_shard_rows(shard_path)
        .with_context(|| format!("opening shard {}", shard_path.display()))?;
    for row_res in iter {
        let row = row_res?;
        rows_counter.fetch_add(1, Ordering::Relaxed);
        if row.status != "ok" {
            continue;
        }
        let (is_candidate, num_matched) = gate_document(&row.text, patterns);
        if is_candidate {
            candidates.push(CandidateRow {
                arxiv_id: row.arxiv_id,
                num_matched_paragraphs: num_matched,
            });
            cand_counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    write_candidates(&out, &shard_id, &candidates)
        .with_context(|| format!("writing candidates to {}", out.display()))?;

    Ok(ShardOutcome::Processed)
}

/// Background reporter. Ticks every `interval_secs`, polling `done` once per
/// second so shutdown is responsive even with a long interval.
pub(crate) fn spawn_progress_thread(
    interval_secs: u64,
    done: Arc<AtomicBool>,
    start: Instant,
    shards_total: usize,
    shards_done: Arc<AtomicUsize>,
    shards_skipped: Arc<AtomicUsize>,
    shards_errored: Arc<AtomicUsize>,
    rows_read: Arc<AtomicU64>,
    candidates_kept: Arc<AtomicU64>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Wait `interval_secs`, but check `done` every second so we can exit
        // quickly when the main thread signals completion.
        loop {
            for _ in 0..interval_secs {
                if done.load(Ordering::Relaxed) {
                    return;
                }
                thread::sleep(Duration::from_secs(1));
            }
            if done.load(Ordering::Relaxed) {
                return;
            }
            let d = shards_done.load(Ordering::Relaxed);
            let s = shards_skipped.load(Ordering::Relaxed);
            let e = shards_errored.load(Ordering::Relaxed);
            let r = rows_read.load(Ordering::Relaxed);
            let c = candidates_kept.load(Ordering::Relaxed);
            let elapsed = start.elapsed().as_secs_f64();
            let finished = (d + s + e) as f64;
            let rate = if elapsed > 0.0 { finished / elapsed } else { 0.0 };
            eprintln!(
                "[progress] shards={}/{} (skipped={} errored={}) rows={} candidates={} elapsed={:.1}s rate={:.2} shards/s",
                d + s + e,
                shards_total,
                s,
                e,
                r,
                c,
                elapsed,
                rate
            );
        }
    })
}

/// Run the pipeline end-to-end. See the module doc for semantics.
pub fn run(config: &PipelineConfig) -> Result<PipelineStats> {
    let shards = enumerate_shards(&config.input)?;
    let shards_total = shards.len();

    let patterns = Arc::new(
        Patterns::load(&config.patterns_path).with_context(|| {
            format!(
                "loading patterns from {}",
                config.patterns_path.display()
            )
        })?,
    );

    let rows_read = Arc::new(AtomicU64::new(0));
    let candidates_kept = Arc::new(AtomicU64::new(0));
    let shards_done = Arc::new(AtomicUsize::new(0));
    let shards_skipped = Arc::new(AtomicUsize::new(0));
    let shards_errored = Arc::new(AtomicUsize::new(0));
    let done_flag = Arc::new(AtomicBool::new(false));

    let start = Instant::now();

    let progress_handle = spawn_progress_thread(
        config.progress_interval_secs.max(1),
        Arc::clone(&done_flag),
        start,
        shards_total,
        Arc::clone(&shards_done),
        Arc::clone(&shards_skipped),
        Arc::clone(&shards_errored),
        Arc::clone(&rows_read),
        Arc::clone(&candidates_kept),
    );

    // Build a dedicated thread pool so callers with other rayon workloads
    // don't have their global pool sized by us, and so `--threads` is
    // authoritative.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads.max(1))
        .build()
        .context("building rayon thread pool")?;

    pool.install(|| {
        shards.par_iter().for_each(|shard_path| {
            // Each shard is wrapped in `catch_unwind` so a panic inside any
            // parquet/regex call escalates to a single errored shard rather
            // than aborting the whole process (rayon propagates panics by
            // default). `AssertUnwindSafe` is justified here because the
            // shared state is all atomics + `Arc<Patterns>` which are already
            // unwind-safe in effect.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                process_shard(
                    shard_path,
                    config,
                    &patterns,
                    &rows_read,
                    &candidates_kept,
                )
            }));
            match result {
                Ok(Ok(ShardOutcome::Processed)) => {
                    shards_done.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Ok(ShardOutcome::Skipped)) => {
                    shards_skipped.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Err(e)) => {
                    eprintln!(
                        "[error] shard {:?}: {:#}",
                        shard_path, e
                    );
                    shards_errored.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    eprintln!("[panic] shard {:?}", shard_path);
                    shards_errored.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    });

    // Signal the reporter, then join so its last line (if any) lands before
    // our final summary. A panic in the progress thread is logged but not
    // fatal.
    done_flag.store(true, Ordering::Relaxed);
    if let Err(e) = progress_handle.join() {
        eprintln!("[warn] progress thread panicked: {:?}", e);
    }

    let elapsed_secs = start.elapsed().as_secs_f64();

    let stats = PipelineStats {
        shards_total,
        shards_done: shards_done.load(Ordering::Relaxed),
        shards_skipped: shards_skipped.load(Ordering::Relaxed),
        shards_errored: shards_errored.load(Ordering::Relaxed),
        rows_read: rows_read.load(Ordering::Relaxed),
        candidates_kept: candidates_kept.load(Ordering::Relaxed),
        elapsed_secs,
    };

    eprintln!(
        "[done] shards_total={} done={} skipped={} errored={} rows={} candidates={} elapsed={:.2}s",
        stats.shards_total,
        stats.shards_done,
        stats.shards_skipped,
        stats.shards_errored,
        stats.rows_read,
        stats.candidates_kept,
        stats.elapsed_secs,
    );

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet_io::read_candidate_arxiv_ids;

    use std::fs::File;
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::sync::Arc;

    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    /// Path to the real YAML so the tests exercise the production gate
    /// (rather than a hand-rolled toy list that might mask integration bugs).
    fn real_patterns_path() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../funding_statement_extractor/configs/patterns/funding_patterns.yaml");
        path
    }

    /// Build a 3-row input parquet with a known mix of rows:
    /// * row 0: `status=ok`, funding text (should be kept)
    /// * row 1: `status=ok`, non-funding text (should be read, not kept)
    /// * row 2: `status=error`, funding text (should be skipped by status)
    fn write_test_shard(path: &Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("arxiv_id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
        ]));
        let arxiv = Arc::new(StringArray::from(vec![
            "2401.00001",
            "2401.00002",
            "2401.00003",
        ])) as Arc<dyn Array>;
        let text = Arc::new(StringArray::from(vec![
            "This work was supported by the NSF under grant AST-0000001.",
            "The quick brown fox jumps over the lazy dog.",
            "This work was supported by the NSF but extraction failed.",
        ])) as Arc<dyn Array>;
        let status = Arc::new(StringArray::from(vec!["ok", "ok", "error"])) as Arc<dyn Array>;

        let batch = RecordBatch::try_new(schema.clone(), vec![arxiv, text, status]).unwrap();
        let file = File::create(path).unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Build a config for a test with a single-shard input and a fresh output.
    fn make_config(input: &Path, output: &Path) -> PipelineConfig {
        PipelineConfig {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            patterns_path: real_patterns_path(),
            threads: 1,
            force: false,
            progress_interval_secs: 3600,
        }
    }

    #[test]
    fn pipeline_processes_small_shard_with_candidates() {
        let input_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();
        let shard_path = input_dir.path().join("testshard.parquet");
        write_test_shard(&shard_path);

        let cfg = make_config(input_dir.path(), output_dir.path());
        let stats = run(&cfg).unwrap();

        assert_eq!(stats.shards_total, 1);
        assert_eq!(stats.shards_done, 1);
        assert_eq!(stats.shards_skipped, 0);
        assert_eq!(stats.shards_errored, 0);
        assert_eq!(stats.rows_read, 3);
        assert_eq!(stats.candidates_kept, 1);

        let out_path = output_dir.path().join("testshard.parquet");
        assert!(out_path.exists(), "output parquet must be written");
        let ids = read_candidate_arxiv_ids(&out_path).unwrap();
        assert_eq!(ids, vec!["2401.00001".to_string()]);
    }

    #[test]
    fn pipeline_skips_already_processed_shard() {
        let input_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();
        let shard_path = input_dir.path().join("testshard.parquet");
        write_test_shard(&shard_path);

        // First run actually processes.
        let cfg = make_config(input_dir.path(), output_dir.path());
        let first = run(&cfg).unwrap();
        assert_eq!(first.shards_done, 1);
        assert_eq!(first.rows_read, 3);

        // Second run (same config, no --force) must skip.
        let second = run(&cfg).unwrap();
        assert_eq!(second.shards_total, 1);
        assert_eq!(second.shards_skipped, 1);
        assert_eq!(second.shards_done, 0);
        assert_eq!(second.shards_errored, 0);
        assert_eq!(
            second.rows_read, 0,
            "skipped shards must not touch the input parquet"
        );
        assert_eq!(second.candidates_kept, 0);
    }

    #[test]
    fn pipeline_force_reprocesses() {
        let input_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();
        let shard_path = input_dir.path().join("testshard.parquet");
        write_test_shard(&shard_path);

        let cfg = make_config(input_dir.path(), output_dir.path());
        run(&cfg).unwrap();

        let cfg_force = PipelineConfig {
            force: true,
            ..make_config(input_dir.path(), output_dir.path())
        };
        let forced = run(&cfg_force).unwrap();
        assert_eq!(forced.shards_done, 1);
        assert_eq!(forced.shards_skipped, 0);
        assert_eq!(forced.rows_read, 3);
        assert_eq!(forced.candidates_kept, 1);
    }

    #[test]
    fn pipeline_errors_isolated() {
        let input_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();

        // A valid shard and a corrupt one — the corrupt shard must surface
        // as `shards_errored == 1` without breaking the valid one.
        let valid_shard = input_dir.path().join("good.parquet");
        write_test_shard(&valid_shard);

        let broken_shard = input_dir.path().join("broken.parquet");
        let mut f = File::create(&broken_shard).unwrap();
        f.write_all(b"not a parquet").unwrap();
        f.sync_all().unwrap();

        let cfg = make_config(input_dir.path(), output_dir.path());
        let stats = run(&cfg).unwrap();
        assert_eq!(stats.shards_total, 2);
        assert_eq!(stats.shards_errored, 1, "broken shard must be isolated");
        assert_eq!(
            stats.shards_done, 1,
            "valid shard must still process despite sibling failure"
        );
        assert_eq!(stats.shards_skipped, 0);

        // The valid shard's output must still be readable and contain the
        // expected row.
        let good_out = output_dir.path().join("good.parquet");
        assert!(good_out.exists(), "valid shard output must exist");
        let ids = read_candidate_arxiv_ids(&good_out).unwrap();
        assert_eq!(ids, vec!["2401.00001".to_string()]);
    }
}
