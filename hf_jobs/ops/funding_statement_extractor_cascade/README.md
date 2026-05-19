# funding_statement_extractor_cascade — local ops scripts

Run locally with the project venv at the repo root (`.venv/`).

- `orchestrate_extractions.py` — local orchestrator that drives A100 HF Jobs
  over a VLM-markdown corpus with manifest-backed resume, retry, and EMA-based
  rebalancing.
- `profile_extraction.py` — per-stage profiler over a cached document sample;
  `_profile_utils.py` is the shared accumulator.
- `diff_predictions.py` — writes a cascade-vs-gold diff (JSONL) for regression
  checks against the bench split.

Tests live in `tests/`.

## What the orchestrator does

`orchestrate_extractions.py` submits and supervises HF Jobs that run
`hf_jobs/jobs/funding_statement_extractor_cascade/extract_funding_cascade_job.py`.
It:

- **Lists inputs** under `--input-repo` / `--input-subdir` with one indexed Hub
  tree call, recording each parquet's byte size.
- **Maintains a parquet manifest** (`--manifest`) so a killed run resumes
  exactly where it left off. On startup it reloads the manifest, merges in any
  newly discovered input files, and reconciles rows against output parquets
  already present on the Hub (matched by basename, so flat and year-sharded
  layouts both work).
- **Batches jobs** by estimated wall-clock: it greedily packs pending files into
  a batch up to `--target-seconds`, capped at `--max-files-per-batch` files.
- **Estimates runtime with an EMA.** The first batches use
  `--seed-seconds-per-byte`; after each `[done ...]` event the seconds-per-byte
  estimate is updated (alpha 0.3) and pending `est_seconds` are recomputed.
- **Submits jobs 429-aware.** `submit_a100_job` retries on 429/5xx/transient
  network errors. The worker itself handles 429s on the commit path.
- **Polls and unsticks jobs.** It tails job logs for `[done ...]` events;
  a job silent for `--stuck-min` minutes while `RUNNING`, or stuck `SCHEDULING`
  longer than `--stuck-min`, is cancelled and its files released. Infra
  cancellations release files back to `pending` without charging a retry
  attempt; genuine `ERROR`/`CANCELED` terminal states bump `attempts` and mark
  the file `failed` once `--max-attempts` is reached.
- **Writes a run summary** (status counts, total rows, worker seconds, estimated
  cost) and pushes a summary JSON, a manifest snapshot, and a generated
  `README.md` to the output repo when the run completes.

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--input-repo` | `cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test` | HF dataset repo to read input parquets from. |
| `--input-subdir` | `""` | Optional in-repo subdirectory to scope the listing. |
| `--output-repo` | `""` | HF dataset repo to push predictions and run metadata to. |
| `--manifest` | `manifests/cascade-bench-2026-05-19.parquet` | Local manifest parquet path. |
| `--max-in-flight` | `1` | Maximum concurrent HF Jobs. |
| `--target-seconds` | `1800` | Target wall-clock per job batch (30 min). |
| `--max-files-per-batch` | `5` | Maximum input files per batch. |
| `--max-attempts` | `2` | Attempts before a file is marked `failed`. |
| `--stuck-min` | `15` | Minutes of log silence before cancelling a `RUNNING` job. |
| `--poll-interval-s` | `60` | Seconds between poll cycles. |
| `--seed-seconds-per-byte` | `9e-7` | Initial runtime estimate; replaced by the EMA after the first jobs complete. |
| `--worker-script` | `hf_jobs/jobs/funding_statement_extractor_cascade/extract_funding_cascade_job.py` | Path to the PEP 723 worker. |
| `--job-timeout` | `2h` | HF Job timeout. |
| `--dry-run` | off | Don't submit jobs; mark batches done immediately. |

The job image is `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` on the
`a100-large` flavor.

## Dry run

`--dry-run` exercises the full state machine — manifest seed/merge,
reconciliation, batch picking, the assigned->done transition, the EMA update,
and the release path — without submitting any HF Job or spending GPU time. It
synthesizes `submit`/`poll` functions that immediately report each batched file
as done with a row count and elapsed time derived from its byte size. It does
not need an `HF_TOKEN`. Use it to validate orchestrator changes before a real
run:

```bash
python hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py \
  --output-repo USER/cascade-bench-out --dry-run
```

## Manifest schema

The manifest parquet has one `ManifestRow` per input file:

| Field | Type | Notes |
|---|---|---|
| `input_file` | string | In-repo path of the source parquet. |
| `size_bytes` | int64 | File size, used for the runtime estimate. |
| `est_seconds` | float64 | Estimated wall-clock; seeded then EMA-refined. |
| `status` | string | `pending` / `assigned` / `done` / `failed`. |
| `attempts` | int64 | Number of real attempts charged against the file. |
| `job_id` | string | HF Job id while `assigned`, else null. |
| `assigned_at` | float64 | Epoch seconds the file was assigned to a job. |
| `completed_at` | float64 | Epoch seconds the file reached `done`. |
| `output_path` | string | In-repo path of the prediction parquet. |
| `last_error` | string | Reason text from the last release/failure. |
| `worker_elapsed_s` | float64 | Worker-reported elapsed seconds for the file. |
| `row_count` | int64 | Worker-reported row count; filled when done. |

## Example end-to-end run

```bash
HF_TOKEN=hf_... python hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py \
  --output-repo USER/cascade-bench-out
```

This points at the default bench corpus
`cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test`, whose
`test` split is a single parquet of 596 documents. With one input file the
orchestrator's batching, EMA rebalancing, and resume logic have little to do —
the run mostly demonstrates parity with the v1 orchestrator. The orchestrator
becomes load-bearing only when pointed at a larger multi-shard VLM-markdown
corpus, where the manifest, retry budget, and EMA-driven batch sizing actually
matter.
