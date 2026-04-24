//! `funding-prefilter` CLI.
//!
//! Two subcommands:
//! * `run` (default) — walk an input directory of parquet shards, apply the
//!   two-stage regex gate, and write per-shard candidate parquets.
//! * `merge` — concatenate per-shard candidate parquets into a single parquet
//!   plus a sorted, deduplicated `arxiv_id` text file for downstream jobs.
//!
//! The default `--patterns` path is baked in at build time via
//! `CARGO_MANIFEST_DIR` so the installed binary always resolves back to the
//! sibling Python config, regardless of the caller's CWD. Pass `--patterns` to
//! override.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, Subcommand};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use walkdir::WalkDir;

use funding_prefilter::pipeline::{run as run_pipeline, PipelineConfig};

#[derive(Parser)]
#[command(
    name = "funding-prefilter",
    version,
    about = "Tier-2 funding-statement prefilter for local parquet mirrors",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    // Shortcut: if no subcommand given, these flags are interpreted as `run`.
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand)]
enum Command {
    /// Run the prefilter over a directory of parquet shards (default).
    Run(RunArgs),
    /// Merge per-shard candidate parquets into a single parquet + text file.
    Merge(MergeArgs),
}

#[derive(clap::Args, Default, Clone)]
struct RunArgs {
    /// Input directory (recursively scanned for *.parquet).
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output directory (shard outputs mirror the input tree).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to the funding patterns YAML. Defaults to the sibling Python config.
    #[arg(long)]
    patterns: Option<PathBuf>,

    /// Worker thread count. Defaults to num_cpus::get().
    #[arg(short, long)]
    threads: Option<usize>,

    /// Reprocess shards even if output already exists.
    #[arg(long)]
    force: bool,

    /// Progress reporting interval in seconds.
    #[arg(long, default_value_t = 5)]
    progress_interval: u64,
}

#[derive(clap::Args)]
struct MergeArgs {
    /// Directory of per-shard candidate parquets (the `--output` of `run`).
    #[arg(short, long)]
    input: PathBuf,

    /// Output parquet file (concatenation of all shard candidate rows).
    #[arg(short, long)]
    output: PathBuf,

    /// Output text file with one arxiv_id per line.
    #[arg(long)]
    ids: PathBuf,
}

/// Default patterns YAML location, resolved at build time so the installed
/// binary always finds the repo's shared config regardless of the caller's
/// CWD. Override with `--patterns` when running against a different YAML.
fn default_patterns_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../funding_statement_extractor/configs/patterns/funding_patterns.yaml")
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let cmd = match cli.command {
        Some(c) => c,
        None => Command::Run(cli.run),
    };

    let result = match cmd {
        Command::Run(args) => run_cmd(args),
        Command::Merge(args) => merge_cmd(args),
    };

    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

/// Handle `funding-prefilter run`. Missing `--input` / `--output` exits with
/// status 2 (clap-style "usage error") rather than propagating an anyhow
/// error, so scripts can distinguish user errors from runtime failures.
fn run_cmd(args: RunArgs) -> Result<ExitCode> {
    let input = match args.input {
        Some(p) => p,
        None => {
            eprintln!("error: --input is required for `run`");
            return Ok(ExitCode::from(2));
        }
    };
    let output = match args.output {
        Some(p) => p,
        None => {
            eprintln!("error: --output is required for `run`");
            return Ok(ExitCode::from(2));
        }
    };

    let patterns_path = args.patterns.unwrap_or_else(default_patterns_path);
    let threads = args.threads.unwrap_or_else(num_cpus::get);

    let cfg = PipelineConfig {
        input,
        output,
        patterns_path,
        threads,
        force: args.force,
        progress_interval_secs: args.progress_interval,
    };

    let stats = run_pipeline(&cfg).context("pipeline run failed")?;

    // Single-line machine-parseable summary on stdout. The human-oriented
    // `[done] ...` line on stderr is emitted by `pipeline::run`.
    println!(
        "done: shards_total={} done={} skipped={} errored={} rows={} candidates={} elapsed={:.2}s",
        stats.shards_total,
        stats.shards_done,
        stats.shards_skipped,
        stats.shards_errored,
        stats.rows_read,
        stats.candidates_kept,
        stats.elapsed_secs,
    );

    Ok(ExitCode::SUCCESS)
}

/// Arrow/parquet schema for candidate shards — must stay in sync with
/// `parquet_io::write_candidates`. We rebuild it here rather than re-exporting
/// so `parquet_io` can keep its helper private.
fn candidate_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("arxiv_id", DataType::Utf8, false),
        Field::new("shard_id", DataType::Utf8, false),
        Field::new("num_matched_paragraphs", DataType::UInt32, false),
    ]))
}

/// Handle `funding-prefilter merge`. Walks `args.input` recursively for
/// `*.parquet`, re-reads all three columns of each shard (candidate parquets
/// are tiny, so the cost of reading instead of projecting is negligible),
/// concatenates into a single parquet at `args.output`, and writes a sorted
/// deduplicated id list to `args.ids`.
///
/// Deduplication is applied to the ids file only — the parquet keeps every
/// row so the `shard_id` / `num_matched_paragraphs` provenance survives for
/// anyone who needs it later.
fn merge_cmd(args: MergeArgs) -> Result<ExitCode> {
    // Deterministic shard order makes the merged parquet byte-reproducible
    // across reruns, which is useful for downstream diffs.
    let mut shards: Vec<PathBuf> = WalkDir::new(&args.input)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.path().to_path_buf())
        .filter(|p| p.extension().map(|e| e == "parquet").unwrap_or(false))
        .collect();
    shards.sort();

    // Accumulators for the concatenated parquet. We keep three parallel Vecs
    // rather than `Vec<(String, String, u32)>` to avoid an extra copy when we
    // build the final `StringArray` / `UInt32Array` at the end — the arrow
    // constructors take flat slices directly.
    let mut arxiv_ids: Vec<String> = Vec::new();
    let mut shard_ids: Vec<String> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();

    // Sorted, deduplicated set used to write the ids file. BTreeSet gives us
    // the sort-on-insert property for free.
    let mut unique_ids: BTreeSet<String> = BTreeSet::new();

    for shard_path in &shards {
        let file = File::open(shard_path)
            .with_context(|| format!("opening candidate parquet {}", shard_path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).with_context(|| {
            format!(
                "reading candidate parquet metadata from {}",
                shard_path.display()
            )
        })?;
        let reader = builder.with_batch_size(2048).build().with_context(|| {
            format!(
                "building reader for candidate parquet {}",
                shard_path.display()
            )
        })?;

        for batch_res in reader {
            let batch = batch_res.with_context(|| {
                format!(
                    "reading batch from candidate parquet {}",
                    shard_path.display()
                )
            })?;

            // Resolve columns by name instead of fixed index so we're tolerant
            // to schema column ordering (the writer always emits the same
            // order, but a future refactor might reorder the struct).
            let arxiv_arr = column_as_string(&batch, "arxiv_id", shard_path)?;
            let shard_arr = column_as_string(&batch, "shard_id", shard_path)?;
            let count_arr = column_as_u32(&batch, "num_matched_paragraphs", shard_path)?;

            for i in 0..batch.num_rows() {
                if arxiv_arr.is_null(i) {
                    anyhow::bail!(
                        "null `arxiv_id` at row {} in {}",
                        i,
                        shard_path.display()
                    );
                }
                if shard_arr.is_null(i) {
                    anyhow::bail!(
                        "null `shard_id` at row {} in {}",
                        i,
                        shard_path.display()
                    );
                }
                if count_arr.is_null(i) {
                    anyhow::bail!(
                        "null `num_matched_paragraphs` at row {} in {}",
                        i,
                        shard_path.display()
                    );
                }

                let arxiv = arxiv_arr.value(i).to_string();
                unique_ids.insert(arxiv.clone());
                arxiv_ids.push(arxiv);
                shard_ids.push(shard_arr.value(i).to_string());
                counts.push(count_arr.value(i));
            }
        }
    }

    let rows_total = arxiv_ids.len();
    let unique_count = unique_ids.len();

    // --- Write concatenated parquet.
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating parent directory {}", parent.display())
            })?;
        }
    }

    let schema = candidate_schema();
    let arxiv_col = Arc::new(StringArray::from(arxiv_ids));
    let shard_col = Arc::new(StringArray::from(shard_ids));
    let count_col = Arc::new(UInt32Array::from(counts));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![arxiv_col, shard_col, count_col],
    )
    .context("building merged RecordBatch")?;

    let out_file = File::create(&args.output)
        .with_context(|| format!("creating output parquet {}", args.output.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(out_file, schema, Some(props))
        .with_context(|| format!("opening ArrowWriter for {}", args.output.display()))?;
    if batch.num_rows() > 0 {
        writer
            .write(&batch)
            .with_context(|| format!("writing merged batch to {}", args.output.display()))?;
    }
    writer
        .close()
        .with_context(|| format!("closing ArrowWriter for {}", args.output.display()))?;

    // --- Write sorted, deduplicated ids file.
    if let Some(parent) = args.ids.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating parent directory {}", parent.display())
            })?;
        }
    }

    let ids_file = File::create(&args.ids)
        .with_context(|| format!("creating ids file {}", args.ids.display()))?;
    let mut ids_writer = BufWriter::new(ids_file);
    for id in &unique_ids {
        writeln!(ids_writer, "{id}")
            .with_context(|| format!("writing to ids file {}", args.ids.display()))?;
    }
    ids_writer
        .flush()
        .with_context(|| format!("flushing ids file {}", args.ids.display()))?;

    println!(
        "merge: shards={} rows={} unique_ids={} output={} ids={}",
        shards.len(),
        rows_total,
        unique_count,
        args.output.display(),
        args.ids.display(),
    );

    Ok(ExitCode::SUCCESS)
}

/// Downcast a named column to `StringArray`, returning a clear error if the
/// column is missing or the wrong type. Inlined here rather than exported
/// from `parquet_io` because merge is the only second consumer.
fn column_as_string<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &std::path::Path,
) -> Result<&'a StringArray> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| anyhow::anyhow!("column {:?} missing in {}", name, path.display()))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "column {:?} is not a utf8 StringArray in {}",
                name,
                path.display()
            )
        })
}

fn column_as_u32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &std::path::Path,
) -> Result<&'a UInt32Array> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| anyhow::anyhow!("column {:?} missing in {}", name, path.display()))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "column {:?} is not a UInt32Array in {}",
                name,
                path.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use funding_prefilter::parquet_io::{write_candidates, CandidateRow};
    use tempfile::TempDir;

    /// End-to-end smoke test of `merge_cmd` — build two candidate parquets
    /// with overlapping ids, run merge, then re-read the outputs to assert
    /// (a) the parquet preserves every row (no dedup there), and (b) the ids
    /// file is sorted + deduplicated.
    #[test]
    fn merge_cmd_concatenates_and_dedups_ids() {
        let input_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();

        // Two shards with one overlapping arxiv_id (2401.00002).
        let shard_a = input_dir.path().join("a.parquet");
        write_candidates(
            &shard_a,
            "shard-a",
            &[
                CandidateRow {
                    arxiv_id: "2401.00002".to_string(),
                    num_matched_paragraphs: 1,
                },
                CandidateRow {
                    arxiv_id: "2401.00001".to_string(),
                    num_matched_paragraphs: 3,
                },
            ],
        )
        .unwrap();

        let shard_b = input_dir.path().join("b.parquet");
        write_candidates(
            &shard_b,
            "shard-b",
            &[
                CandidateRow {
                    arxiv_id: "2401.00002".to_string(),
                    num_matched_paragraphs: 2,
                },
                CandidateRow {
                    arxiv_id: "2401.00003".to_string(),
                    num_matched_paragraphs: 4,
                },
            ],
        )
        .unwrap();

        let out_parquet = output_dir.path().join("merged.parquet");
        let out_ids = output_dir.path().join("ids.txt");
        let args = MergeArgs {
            input: input_dir.path().to_path_buf(),
            output: out_parquet.clone(),
            ids: out_ids.clone(),
        };

        let code = merge_cmd(args).unwrap();
        assert_eq!(code, ExitCode::SUCCESS);

        // Parquet must contain all 4 rows (no dedup there).
        assert!(out_parquet.exists());
        let file = File::open(&out_parquet).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let mut total_rows = 0usize;
        for batch in reader {
            total_rows += batch.unwrap().num_rows();
        }
        assert_eq!(total_rows, 4, "merge parquet must keep every shard row");

        // IDs file must be sorted + deduplicated.
        let ids_text = std::fs::read_to_string(&out_ids).unwrap();
        let ids: Vec<&str> = ids_text.lines().collect();
        assert_eq!(ids, vec!["2401.00001", "2401.00002", "2401.00003"]);
    }

    /// If the input directory has no `*.parquet` files the merge should still
    /// succeed with 0 rows — downstream tooling relies on the output files
    /// existing so it can `ls` them unconditionally.
    #[test]
    fn merge_cmd_empty_input_produces_empty_outputs() {
        let input_dir = TempDir::new().unwrap();
        let output_dir = TempDir::new().unwrap();
        let out_parquet = output_dir.path().join("merged.parquet");
        let out_ids = output_dir.path().join("ids.txt");
        let args = MergeArgs {
            input: input_dir.path().to_path_buf(),
            output: out_parquet.clone(),
            ids: out_ids.clone(),
        };

        let code = merge_cmd(args).unwrap();
        assert_eq!(code, ExitCode::SUCCESS);
        assert!(out_parquet.exists(), "empty merge must still create parquet");
        assert!(out_ids.exists(), "empty merge must still create ids file");
        let ids_text = std::fs::read_to_string(&out_ids).unwrap();
        assert!(ids_text.is_empty(), "empty input -> empty ids file");
    }
}
