//! End-to-end orchestration for the `materialize` subcommand.
//!
//! Given the original input shards (full schema, e.g. 13 columns including
//! `text` and `status`) plus the per-shard candidate parquets produced by
//! [`crate::pipeline::run`], emit new parquets that contain ONLY the candidate
//! rows but preserve every input column. The downstream ColBERT extractor
//! consumes these so it can run extraction without re-scanning the whole
//! corpus.
//!
//! Mirrors `pipeline::run`'s orchestration shape — sorted shard walk, parallel
//! per-shard workers behind a dedicated rayon pool, panic isolation,
//! resumability via existing-non-empty-output check, background progress
//! thread (reused via `pub(crate)`). The novel piece is per-shard work: open
//! the input parquet WITHOUT column projection so the full schema flows
//! through, build a boolean mask over the `arxiv_id` column against the
//! candidate id set, and stream filtered batches through an `ArrowWriter`
//! parameterized by the input schema.
//!
//! Empty filtered batches are skipped, but `ArrowWriter::close()` is still
//! called — so a shard with no surviving rows produces a 0-row parquet with
//! the input schema, satisfying the "output exists + non-empty-file"
//! resumability marker (the parquet footer is non-zero bytes).

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use arrow::compute::filter_record_batch;
use arrow_array::{Array, BooleanArray, LargeStringArray, RecordBatch, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use crate::parquet_io::read_candidate_arxiv_ids;
use crate::pipeline::{enumerate_shards, spawn_progress_thread, PipelineStats};

/// User-facing configuration for a materialize run.
pub struct MaterializeConfig {
    /// Original full-schema shards root.
    pub input: PathBuf,
    /// Per-shard candidate parquets root (the `--output` of `pipeline::run`).
    pub candidates: PathBuf,
    /// Destination root for the materialized full-row parquets. Mirrors the
    /// input tree.
    pub output: PathBuf,
    pub threads: usize,
    pub force: bool,
    pub progress_interval_secs: u64,
}

/// Outcome of a single shard's worker — used by the parent `for_each` to bump
/// the right atomic counter. `Skipped` covers both "already processed
/// (resumability)" and "no candidate file present"; the eprintln in the latter
/// case carries the disambiguation.
enum ShardOutcome {
    Processed,
    Skipped,
}

/// Process a single shard: stream the input parquet through a boolean filter
/// against the candidate id set, write the surviving rows out under the same
/// input schema. Returns `Skipped` if the candidate file is missing OR the
/// output already exists and `--force` is off; `Processed` otherwise.
fn process_shard(
    shard_path: &Path,
    cfg: &MaterializeConfig,
    rows_counter: &AtomicU64,
    kept_counter: &AtomicU64,
) -> Result<ShardOutcome> {
    // Preserve the shard's position relative to `--input` inside `--candidates`
    // and `--output`, mirroring the pipeline's path convention.
    let rel = shard_path
        .strip_prefix(&cfg.input)
        .with_context(|| {
            format!(
                "shard path {} is not inside input root {}",
                shard_path.display(),
                cfg.input.display()
            )
        })?;
    let candidate_path = cfg.candidates.join(rel);
    let out_path = cfg.output.join(rel);

    // No candidate file means `pipeline::run` was never executed for this
    // shard. Skip + log so the operator can correlate against `run`'s outputs.
    if !candidate_path.exists() {
        eprintln!(
            "[skip] shard={} reason=no_candidate_file path={}",
            rel.display(),
            candidate_path.display()
        );
        return Ok(ShardOutcome::Skipped);
    }

    // Resumability gate — same check as `pipeline::process_shard`. A 0-byte
    // file would indicate a partial write from a crashed run; the `len() > 0`
    // requirement deliberately re-processes that.
    if !cfg.force {
        if let Ok(meta) = std::fs::metadata(&out_path) {
            if meta.len() > 0 {
                return Ok(ShardOutcome::Skipped);
            }
        }
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("creating output parent directory {}", parent.display())
        })?;
    }

    let ids: HashSet<String> = read_candidate_arxiv_ids(&candidate_path)
        .with_context(|| {
            format!(
                "reading candidate ids from {}",
                candidate_path.display()
            )
        })?
        .into_iter()
        .collect();

    // Open the input shard with NO column projection so the full schema (text,
    // status, num_tex_files, …) flows through unchanged. Capture the schema
    // before consuming the builder via `.build()`.
    let file = File::open(shard_path)
        .with_context(|| format!("opening input shard {}", shard_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).with_context(|| {
        format!("reading parquet metadata from {}", shard_path.display())
    })?;
    let schema = builder.schema().clone();
    let reader = builder
        .with_batch_size(2048)
        .build()
        .with_context(|| {
            format!("building parquet reader for {}", shard_path.display())
        })?;

    let out_file = File::create(&out_path)
        .with_context(|| format!("creating output parquet {}", out_path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(props))
        .with_context(|| format!("opening ArrowWriter for {}", out_path.display()))?;

    for batch_res in reader {
        let batch = batch_res.with_context(|| {
            format!("reading batch from input shard {}", shard_path.display())
        })?;
        let mask = build_arxiv_id_mask(&batch, &ids, shard_path)?;
        let filtered = filter_record_batch(&batch, &mask)
            .with_context(|| format!("filtering batch from {}", shard_path.display()))?;

        rows_counter.fetch_add(batch.num_rows() as u64, Ordering::Relaxed);
        kept_counter.fetch_add(filtered.num_rows() as u64, Ordering::Relaxed);

        // Skip empty filtered batches — `close()` still emits a 0-row parquet
        // with the schema, which is the resumability marker for "shard
        // processed, no surviving rows".
        if filtered.num_rows() > 0 {
            writer.write(&filtered).with_context(|| {
                format!("writing filtered batch to {}", out_path.display())
            })?;
        }
    }

    writer
        .close()
        .with_context(|| format!("closing ArrowWriter for {}", out_path.display()))?;

    Ok(ShardOutcome::Processed)
}

/// Build a `BooleanArray` of length `batch.num_rows()` where each entry is
/// `true` iff the row's `arxiv_id` is in `ids`. Defensive against both `Utf8`
/// (`StringArray`) and `LargeUtf8` (`LargeStringArray`) flavors of the column —
/// real shards in `cometadata/arxiv-latex-extract-full-text` use `LargeUtf8`
/// for some string columns, so a single-type downcast would silently fail.
/// Null arxiv_ids → `false` (cannot match).
fn build_arxiv_id_mask(
    batch: &RecordBatch,
    ids: &HashSet<String>,
    path: &Path,
) -> Result<BooleanArray> {
    let arxiv_idx = batch.schema().index_of("arxiv_id").map_err(|_| {
        anyhow!(
            "input parquet at {} is missing required column \"arxiv_id\"",
            path.display()
        )
    })?;
    let col = batch.column(arxiv_idx);
    let mut mask: Vec<bool> = Vec::with_capacity(batch.num_rows());
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        for i in 0..arr.len() {
            mask.push(if arr.is_null(i) {
                false
            } else {
                ids.contains(arr.value(i))
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        for i in 0..arr.len() {
            mask.push(if arr.is_null(i) {
                false
            } else {
                ids.contains(arr.value(i))
            });
        }
    } else {
        return Err(anyhow!(
            "column `arxiv_id` in {} has unsupported type {:?} (expected Utf8 or LargeUtf8)",
            path.display(),
            col.data_type()
        ));
    }
    Ok(BooleanArray::from(mask))
}

/// Run the materialize pipeline end-to-end. See the module doc for semantics.
pub fn run(config: &MaterializeConfig) -> Result<PipelineStats> {
    let shards = enumerate_shards(&config.input)?;
    let shards_total = shards.len();

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

    // Dedicated thread pool — keeps `--threads` authoritative and avoids
    // sharing rayon's global pool with whatever else the caller is running.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads.max(1))
        .build()
        .context("building rayon thread pool")?;

    pool.install(|| {
        shards.par_iter().for_each(|shard_path| {
            // Each shard is wrapped in `catch_unwind` so a panic inside any
            // parquet/filter call escalates to a single errored shard rather
            // than aborting the whole process. Same justification as
            // `pipeline::run`: shared state is atomics + immutable config.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                process_shard(shard_path, config, &rows_read, &candidates_kept)
            }));
            match result {
                Ok(Ok(ShardOutcome::Processed)) => {
                    shards_done.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Ok(ShardOutcome::Skipped)) => {
                    shards_skipped.fetch_add(1, Ordering::Relaxed);
                }
                Ok(Err(e)) => {
                    eprintln!("[error] shard {:?}: {:#}", shard_path, e);
                    shards_errored.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    eprintln!("[panic] shard {:?}", shard_path);
                    shards_errored.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    });

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

    use std::sync::Arc;

    use arrow_array::{Array, RecordBatch, StringArray, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use crate::parquet_io::{write_candidates, CandidateRow};

    /// Build an input parquet with a 4-column schema mirroring (a subset of)
    /// the real shard layout — `arxiv_id`, `text`, `status`, `num_tex_files` —
    /// so the test can verify both string and numeric non-projected columns
    /// survive untouched through the materialize filter.
    fn write_full_input_shard(path: &Path, rows: &[(&str, &str, &str, u32)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("arxiv_id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
            Field::new("num_tex_files", DataType::UInt32, false),
        ]));
        let arxiv = Arc::new(StringArray::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )) as Arc<dyn Array>;
        let text = Arc::new(StringArray::from(
            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )) as Arc<dyn Array>;
        let status = Arc::new(StringArray::from(
            rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        )) as Arc<dyn Array>;
        let num_tex = Arc::new(UInt32Array::from(
            rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        )) as Arc<dyn Array>;

        let batch =
            RecordBatch::try_new(schema.clone(), vec![arxiv, text, status, num_tex]).unwrap();
        let file = File::create(path).unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Build a candidate parquet shaped exactly like `pipeline::run` would
    /// emit — `(arxiv_id, shard_id, num_matched_paragraphs)`. Reuses the
    /// production `write_candidates` so the schema/encoding stays in lockstep.
    fn write_candidate_shard(path: &Path, ids: &[&str]) {
        let rows: Vec<CandidateRow> = ids
            .iter()
            .map(|id| CandidateRow {
                arxiv_id: id.to_string(),
                num_matched_paragraphs: 1,
            })
            .collect();
        write_candidates(path, "shard-test", &rows).unwrap();
    }

    fn make_config(input: &Path, candidates: &Path, output: &Path) -> MaterializeConfig {
        MaterializeConfig {
            input: input.to_path_buf(),
            candidates: candidates.to_path_buf(),
            output: output.to_path_buf(),
            threads: 1,
            force: false,
            progress_interval_secs: 3600,
        }
    }

    /// Read a materialized parquet into parallel Vecs for assertion.
    fn read_materialized(path: &Path) -> (Vec<String>, Vec<String>, Vec<String>, Vec<u32>) {
        let file = File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let mut arxiv_ids = Vec::new();
        let mut texts = Vec::new();
        let mut statuses = Vec::new();
        let mut nums = Vec::new();
        for batch_res in reader {
            let batch = batch_res.unwrap();
            let arxiv_arr = batch
                .column_by_name("arxiv_id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let text_arr = batch
                .column_by_name("text")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let status_arr = batch
                .column_by_name("status")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let num_arr = batch
                .column_by_name("num_tex_files")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                arxiv_ids.push(arxiv_arr.value(i).to_string());
                texts.push(text_arr.value(i).to_string());
                statuses.push(status_arr.value(i).to_string());
                nums.push(num_arr.value(i));
            }
        }
        (arxiv_ids, texts, statuses, nums)
    }

    /// Schema of a parquet on disk.
    fn read_schema(path: &Path) -> Arc<Schema> {
        let file = File::open(path).unwrap();
        ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .schema()
            .clone()
    }

    #[test]
    fn materialize_keeps_only_candidate_rows() {
        let input_dir = TempDir::new().unwrap();
        let candidates_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();

        let input_shard = input_dir.path().join("testshard.parquet");
        write_full_input_shard(
            &input_shard,
            &[
                ("A", "text-a", "ok", 1),
                ("B", "text-b", "ok", 2),
                ("C", "text-c", "error", 3),
                ("D", "text-d", "ok", 4),
            ],
        );

        let candidate_shard = candidates_dir.path().join("testshard.parquet");
        write_candidate_shard(&candidate_shard, &["B", "D"]);

        let cfg = make_config(input_dir.path(), candidates_dir.path(), output_dir.path());
        let stats = run(&cfg).unwrap();

        assert_eq!(stats.shards_total, 1);
        assert_eq!(stats.shards_done, 1);
        assert_eq!(stats.shards_skipped, 0);
        assert_eq!(stats.shards_errored, 0);
        assert_eq!(stats.rows_read, 4);
        assert_eq!(stats.candidates_kept, 2);

        let out_path = output_dir.path().join("testshard.parquet");
        assert!(out_path.exists(), "materialized shard must be written");

        let (arxiv_ids, texts, statuses, nums) = read_materialized(&out_path);
        assert_eq!(arxiv_ids, vec!["B".to_string(), "D".to_string()]);
        assert_eq!(texts, vec!["text-b".to_string(), "text-d".to_string()]);
        assert_eq!(statuses, vec!["ok".to_string(), "ok".to_string()]);
        assert_eq!(nums, vec![2, 4]);
    }

    #[test]
    fn materialize_empty_candidate_set_writes_zero_row_file() {
        let input_dir = TempDir::new().unwrap();
        let candidates_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();

        let input_shard = input_dir.path().join("testshard.parquet");
        write_full_input_shard(
            &input_shard,
            &[
                ("A", "text-a", "ok", 1),
                ("B", "text-b", "ok", 2),
                ("C", "text-c", "error", 3),
                ("D", "text-d", "ok", 4),
            ],
        );

        let candidate_shard = candidates_dir.path().join("testshard.parquet");
        write_candidate_shard(&candidate_shard, &[]);

        let cfg = make_config(input_dir.path(), candidates_dir.path(), output_dir.path());
        let stats = run(&cfg).unwrap();

        assert_eq!(stats.shards_total, 1);
        assert_eq!(stats.shards_done, 1);
        assert_eq!(stats.rows_read, 4);
        assert_eq!(stats.candidates_kept, 0);

        let out_path = output_dir.path().join("testshard.parquet");
        assert!(out_path.exists(), "0-row materialized shard must still be written");
        let len = std::fs::metadata(&out_path).unwrap().len();
        assert!(
            len > 0,
            "materialized shard must have a non-zero file size (parquet footer) so the resumability marker fires"
        );

        // Reader must succeed with 0 rows.
        let (arxiv_ids, texts, statuses, nums) = read_materialized(&out_path);
        assert!(arxiv_ids.is_empty());
        assert!(texts.is_empty());
        assert!(statuses.is_empty());
        assert!(nums.is_empty());

        // Schema must equal the input schema field-for-field.
        let in_schema = read_schema(&input_shard);
        let out_schema = read_schema(&out_path);
        assert_eq!(out_schema.fields().len(), in_schema.fields().len());
        for (a, b) in out_schema.fields().iter().zip(in_schema.fields().iter()) {
            assert_eq!(a.name(), b.name(), "schema field name preserved");
            assert_eq!(a.data_type(), b.data_type(), "schema field type preserved");
        }
    }

    #[test]
    fn materialize_skips_when_candidate_missing() {
        let input_dir = TempDir::new().unwrap();
        let candidates_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();

        let input_shard = input_dir.path().join("testshard.parquet");
        write_full_input_shard(
            &input_shard,
            &[("A", "text-a", "ok", 1)],
        );

        // Intentionally do NOT write a candidate file under candidates_dir.

        let cfg = make_config(input_dir.path(), candidates_dir.path(), output_dir.path());
        let stats = run(&cfg).unwrap();

        assert_eq!(stats.shards_total, 1);
        assert_eq!(stats.shards_skipped, 1);
        assert_eq!(stats.shards_done, 0);
        assert_eq!(stats.shards_errored, 0);
        assert_eq!(stats.rows_read, 0, "missing-candidate skip must not touch the input");
        assert_eq!(stats.candidates_kept, 0);

        let out_path = output_dir.path().join("testshard.parquet");
        assert!(
            !out_path.exists(),
            "materialize must not write any output when the candidate file is missing"
        );
    }

    #[test]
    fn materialize_resumability() {
        let input_dir = TempDir::new().unwrap();
        let candidates_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();

        let input_shard = input_dir.path().join("testshard.parquet");
        write_full_input_shard(
            &input_shard,
            &[
                ("A", "text-a", "ok", 1),
                ("B", "text-b", "ok", 2),
                ("C", "text-c", "error", 3),
                ("D", "text-d", "ok", 4),
            ],
        );
        let candidate_shard = candidates_dir.path().join("testshard.parquet");
        write_candidate_shard(&candidate_shard, &["B", "D"]);

        let cfg = make_config(input_dir.path(), candidates_dir.path(), output_dir.path());
        let first = run(&cfg).unwrap();
        assert_eq!(first.shards_done, 1);
        assert_eq!(first.rows_read, 4);
        assert_eq!(first.candidates_kept, 2);

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
    fn materialize_force_reprocesses() {
        let input_dir = TempDir::new().unwrap();
        let candidates_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();

        let input_shard = input_dir.path().join("testshard.parquet");
        write_full_input_shard(
            &input_shard,
            &[
                ("A", "text-a", "ok", 1),
                ("B", "text-b", "ok", 2),
                ("C", "text-c", "error", 3),
                ("D", "text-d", "ok", 4),
            ],
        );
        let candidate_shard = candidates_dir.path().join("testshard.parquet");
        write_candidate_shard(&candidate_shard, &["B", "D"]);

        let cfg = make_config(input_dir.path(), candidates_dir.path(), output_dir.path());
        run(&cfg).unwrap();

        let cfg_force = MaterializeConfig {
            force: true,
            ..make_config(input_dir.path(), candidates_dir.path(), output_dir.path())
        };
        let forced = run(&cfg_force).unwrap();
        assert_eq!(forced.shards_done, 1);
        assert_eq!(forced.shards_skipped, 0);
        assert_eq!(forced.rows_read, 4);
        assert_eq!(forced.candidates_kept, 2);
    }
}
