# funding_statement_extractor_cascade — HF Jobs scripts

PEP 723 self-contained scripts run via `hf jobs uv run`. Each script's docstring
holds its exact invocation; this file documents `extract_funding_cascade_job.py`,
the extraction worker dispatched in bulk by
`hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py`.

Other scripts here:

- `benchmark_hf_job.py` — F1 / precision-recall benchmark of the cascade against
  the gold `funding_statements` column of the bench split.
- `profile_on_h200.py` — single-document per-stage latency profiler.

Recommended image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`. Flavor:
`a100-large` for extraction. Tests live in `tests/`.

## What the worker does

`extract_funding_cascade_job.py` streams one or more input parquet files from a
Hub dataset, runs the 3-stage cascade over every row, stages the predictions as
a local parquet per (input file x text column), and pushes them to
`predictions/<text_column>/<input_file>.parquet` in the output dataset repo.
The per-column path means parallel single-column jobs writing to the same
output repo never overwrite each other. Commits are batched (see
`--commit-batch-size`) and a
preflight commit probe runs before any GPU work so the worker exits cleanly if
the Hub is rejecting writes.

For each input row the worker reads the text column, skips rows whose `status`
column is set to anything other than `ok` (or null), skips rows with empty text,
and runs the cascade. Output rows carry the cascade result plus provenance
metadata.

## Worker setup — important

The PEP 723 header pins the package as a `git+https://` dependency:

```
"funding-statement-extractor-cascade @ git+https://github.com/cometadata/funding-metadata-enrichment.git@funding-statement-extractor-cascade#subdirectory=funding_statement_extractor_cascade"
```

The header points at the `funding_statement_extractor_cascade` subdirectory of
the `cometadata/funding-metadata-enrichment` repository. It currently pins the
`funding-statement-extractor-cascade` branch so the job can be tested before
that branch is merged; switch the ref back to `@main` in both
`extract_funding_cascade_job.py` and `profile_on_h200.py` once it lands on
`main`.

## CLI contract

| Flag | Required | Default | Purpose |
|---|---|---|---|
| `--input-repo` | yes | — | HF dataset repo id holding the input parquets. |
| `--input-files` | yes | — | Comma-separated in-repo parquet paths. |
| `--output-repo` | yes | — | HF dataset repo id to push predictions to. |
| `--job-tag` | yes | — | Free-form tag for log correlation with the orchestrator. |
| `--text-column` | no | `vlm_markdown` | Comma-separated text column(s) to run; one output parquet per column at `predictions/<text_column>/`, each row also tagged with `text_source`. |
| `--id-column` | no | `doc_id` | Column with the per-row identifier. |
| `--skip-cleanup` | no | off | Short-circuit Stage 3 (Qwen cleaner); emit the raw stage-2 span. |
| `--dtype` | no | `bf16` | Model dtype: `auto`, `fp32`, `fp16`, `bf16`. |
| `--commit-batch-size` | no | `25` | Number of staged input files per Hub commit. |
| `--skip-preflight` | no | off | Skip the preflight commit probe. |
| `--allow-cpu` | no | off | Skip the CUDA probe; for local smoke tests only. |

## Stdout contract

The worker prints machine-parsed lines on stdout that the orchestrator consumes:

- `[done file=<path> rows=<n> elapsed_s=<seconds>]` — emitted once per input
  file *after* its batch is committed to the Hub. The orchestrator parses this
  to mark the file done, record its row count, and update its EMA estimate.
- `[file_summary file=<path> text_column=<col> processed=<n> skipped_status=<n> skipped_empty=<n>]`
  — emitted after each (file x text column) is staged, giving a count of rows
  processed versus rows skipped because of their `status` value or empty text.

The worker also logs `[start file=...]`, `[staged file=...]`, and `[fail file=...]`
lines, but only `[done ...]` and `[file_summary ...]` are part of the contract.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success — all input files processed and committed. |
| 2 | CUDA unavailable after a 120s probe (no CPU fallback). |
| 3 | The final batch commit failed. |
| 4 | Too many consecutive batch-commit failures (2). |
| 5 | The preflight commit check failed before any GPU work. |

## Output schema

Each `predictions/<text_column>/<input_file>.parquet` file has the following columns:

| Column | Type | Notes |
|---|---|---|
| `doc_id` | string | Per-row id (the `--id-column` value, or the row index as fallback). |
| `arxiv_id` | string | Copied from the input `--id-column`. |
| `shard_id` | string | Copied from the input `shard_id` column, if present. |
| `input_file` | string | The in-repo path of the source parquet. |
| `text_source` | string | The input text column this prediction was run on (e.g. `vlm_markdown`). |
| `row_idx` | int64 | Zero-based row index within the source parquet. |
| `funding_statement` | string | Cleaned answer; `""` if no funding found. |
| `rough_span` | string | Pre-cleanup stage-2 output; `""` if none. |
| `chunk_prob` | float64 | Max stage-1 probability across surviving chunks. |
| `no_answer_prob` | float64 | Min stage-2 no-answer probability. |
| `text_length` | int64 | Character length of the input text. |
| `latency_ms` | float64 | Wall-clock cascade latency for the row. |
| `skip_cleanup` | bool | Whether Stage 3 was skipped for this run. |
| `error` | string | `"TypeName: message"` if the row failed, else null. |

The schema is flat — the cascade returns at most one statement per document, so
there is no list wrapping.

## Example invocation

```bash
hf jobs uv run --flavor a100-large \
  --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  hf_jobs/jobs/funding_statement_extractor_cascade/extract_funding_cascade_job.py \
  --input-repo cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test \
  --input-files test-00000-of-00001.parquet \
  --output-repo USER/cascade-bench-out \
  --job-tag cascade-bench-001
```

The bench dataset's `test` split has 596 documents. Its text column is
`vlm_markdown`, its id column is `doc_id`, and the gold column is
`funding_statements` — the worker defaults already match (`--text-column`
defaults to `vlm_markdown`, `--id-column` to `doc_id`).
