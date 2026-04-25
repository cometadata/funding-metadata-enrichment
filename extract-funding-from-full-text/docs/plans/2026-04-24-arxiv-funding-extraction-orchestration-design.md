# ArXiv funding-statement extraction — HF Jobs orchestration design

**Date:** 2026-04-24
**Author:** Adam (with Claude)
**Status:** Approved, ready for implementation plan

## Goal

Run the Tier 2 funding-statement extractor over every parquet in
`cometadata/arxiv-latex-extract-full-text/results-2026-04-24/` using 8 concurrent
H200 HF Jobs, writing one output parquet per input parquet to
`cometadata/arxiv-funding-statement-extractions`. Resume cleanly across
orchestrator crashes, job failures, and HF scheduling errors.

## Non-goals

- Autoscaling beyond 8 parallel jobs.
- Mid-file checkpointing (one input parquet = one work unit).
- Tuning extractor config — Tier 2 defaults from `CLAUDE.md` are baked in.
- Carrying input text into outputs (rejoinable downstream by `arxiv_id` /
  `(input_file, row_idx)`).

## Architecture

Two scripts, one shared local manifest:

```
scripts/
  extract_funding_job.py         # worker — runs inside H200 HF Job (PEP-723)
  orchestrate_extractions.py     # local poller — runs on laptop
manifests/
  arxiv-extractions-2026-04-24.parquet         # state, orchestrator-only writer
  arxiv-extractions-2026-04-24-jobs.parquet    # append-only job log (forensics)
  arxiv-extractions-2026-04-24-summary.json    # final report
```

**Single source of truth:** the local manifest parquet. The output repo is the
secondary truth — presence of `predictions/<filename>.parquet` confirms work
landed. On startup, orchestrator reconciles manifest against output repo to
recover from a crash mid-run.

## Manifest schema

One row per input parquet file. Rewritten in full each loop iteration via
write-temp-then-rename (small file, atomic).

| column              | type        | notes |
|---------------------|-------------|-------|
| `input_file`        | str         | e.g. `results-2026-04-24/shard-00042.parquet` (PK) |
| `row_count`         | int64       | from parquet footer at orchestrator startup |
| `est_seconds`       | float       | `row_count * seconds_per_row`; rescaled via EMA |
| `status`            | str         | `pending` / `assigned` / `done` / `failed` |
| `attempts`          | int         | 0..2 |
| `job_id`            | str         | current/last HF job id, null if never assigned |
| `assigned_at`       | timestamp   | when current job was submitted |
| `completed_at`      | timestamp   | when output landed (status=done) |
| `output_path`       | str         | `predictions/<filename>` in output repo |
| `last_error`        | str         | truncated error from last failure |
| `worker_elapsed_s`  | float       | from worker `[done ...]` line; feeds rebalancing |

### State machine

```
pending  --(orchestrator submits job)--->  assigned
assigned --(output parquet seen on hub)--> done
assigned --(job exits, output missing)---> pending  [if attempts < 2]
                                       \-> failed   [if attempts == 2]
assigned --(stuck >15min no log)-------->  pending  [orchestrator cancels job, attempts++]
```

## Worker — `scripts/extract_funding_job.py`

PEP-723 header identical to `benchmark_hf_job.py` (same deps, same `torch>=2.5,<2.7`
pin, same CUDA-init retry probe — abort if no GPU after 120s).

### Argv

```
--input-repo cometadata/arxiv-latex-extract-full-text
--input-files file1.parquet,file2.parquet,...    # comma-separated paths within repo
--output-repo cometadata/arxiv-funding-statement-extractions
--job-tag <orchestrator-uuid>                     # echoed in logs for correlation
--colbert-model lightonai/GTE-ModernColBERT-v1
--batch-size 512 --dtype bf16
--text-column text --id-column arxiv_id           # configurable input schema
```

Tier 2 baked in (no flags): `enable_paragraph_prefilter=True`,
`regex_match_score_floor=11.0`, `top_k=5`, `threshold=10.0`.

### Per-input-file loop

```python
for input_file in args.input_files:
    t0 = time.perf_counter()
    ds = load_dataset(args.input_repo, data_files=input_file,
                      split="train", streaming=True)

    def docs_iter():
        for row_idx, row in enumerate(ds):
            yield DocPayload(
                doc_id=row[args.id_column],
                text=row[args.text_column],
                metadata={
                    "row_idx": row_idx,
                    "input_file": input_file,
                    "arxiv_id": row.get("arxiv_id"),
                    # ...other carry-through fields determined at startup
                },
            )

    output_rows = []
    for result in extract_funding_statements_batch(
        documents=docs_iter(), queries=queries,
        enable_paragraph_prefilter=True, regex_match_score_floor=11.0,
        dtype="bf16", paragraphs_per_batch=4096, encode_batch_size=512,
    ):
        output_rows.append(_make_output_row(result))

    out_path = f"predictions/{Path(input_file).name}"
    _push_parquet_to_hub(output_rows, args.output_repo, out_path)
    elapsed = time.perf_counter() - t0
    print(f"[done file={input_file} rows={len(output_rows)} elapsed_s={elapsed:.1f}",
          flush=True)
```

### Output row schema

| column                | type |
|-----------------------|------|
| `arxiv_id`            | str  |
| `doc_id`              | str  |
| `input_file`          | str  |
| `row_idx`             | int64 |
| `predicted_statements`| list<str> |
| `predicted_details`   | list<struct{statement: str, score: float32, query: str, paragraph_idx: int32}> |
| `text_length`         | int64 |
| `latency_ms`          | float32 |
| `error`               | str (null on success) |

Zero-prediction rows are kept (downstream coverage analysis depends on it).

### Push semantics

`HfApi.upload_file` per output parquet (single atomic hub commit). Faster +
simpler than `Dataset.push_to_hub` for one-file commits.

### Worker exit codes

- `0` — all input files processed and pushed.
- `2` — no GPU after 120s probe.
- `3` — exception during processing.

## Orchestrator — `scripts/orchestrate_extractions.py`

Runs locally in `.venv` (no PEP-723).

### Startup

1. Load manifest if present, else seed it: `HfFileSystem().ls(...)` the input
   directory, read each parquet's footer via
   `pyarrow.parquet.ParquetFile(fs.open(path)).metadata.num_rows` (footer-only,
   no data download). Populate one row per file with `status=pending`,
   `est_seconds = row_count * 0.045`.
2. **Reconcile against output repo:** list `predictions/*.parquet` in
   `cometadata/arxiv-funding-statement-extractions`. For any manifest row whose
   `output_path` exists on hub but `status != done`, mark `done`. Recovers from
   "orchestrator crashed after job pushed."
3. **Reconcile assigned rows:** for each `assigned` row, `hf jobs inspect <job_id>`.
   If finished, treat as fresh completion event in the loop below.

### Main loop (every 60s)

```
for each in_flight job:
    status = hf jobs inspect job_id
    new_log_lines = hf jobs logs job_id (tail since last poll)
    parse "[done file=X rows=N elapsed_s=T]" lines
        -> mark those files done, record worker_elapsed_s, update EMA
    if no new log lines for >15min:
        hf jobs cancel; release unfinished files (attempts++)
    if job finished:
        for each assigned file: if output exists -> done, else release (attempts++)
        if any released file hits attempts == 2: mark failed
        remove job from in_flight

while len(in_flight) < 8 and pending files exist:
    batch = pick_next_batch(pending_files, target_seconds=5400)  # 90 min
    job_id = submit_h200_job(worker_script, --input-files=batch)
    mark batch rows assigned, job_id, assigned_at=now
    in_flight[job_id] = batch

if no pending and no in_flight: break
sleep 60s
```

### Batch picking

Greedy bin-packing: sort pending files by `est_seconds` descending, accumulate
until next file would exceed `target_seconds=5400` (90 min). Always include at
least one file (single huge file gets its own job). Cap at 50 files per batch
to keep argv reasonable.

### Rebalancing

On every `[done ...]` log line, update global `seconds_per_row` EMA
(`alpha=0.3`). Recompute `est_seconds` for remaining `pending` rows. Subsequent
batches use the updated estimate. First batch uses seed `0.045 s/row` from the
Tier 2 H200 baseline.

### Submission

Reuses the retry-with-backoff pattern from `benchmark_hf_job.py`: 60s sleep on
429, up to 10 attempts. Image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`,
flavor `h200`, timeout `2h`, secrets `HF_TOKEN`.

## Failure handling

| failure                      | detection                                        | action |
|------------------------------|--------------------------------------------------|--------|
| Submit 429                   | `api.run_job()` raises with "429"                | sleep 60s, retry up to 10x; files stay `pending` |
| Submit hard fail (10x)       | retry exhausted                                  | leave files `pending`; retried next loop |
| No GPU (worker exits 2)      | `hf jobs inspect` exit_code=2, no `[done ...]`   | release files (attempts++) |
| Worker exception (exit 3)    | finished, files lack `[done ...]` line + output  | release unfinished files (attempts++) |
| Hub push failure inside worker | `[done ...]` logged but output missing on hub  | reconcile catches it; file released (attempts++) |
| Timeout at 2h                | job state `error`/`timeout`                      | partial completions kept, rest released |
| Stuck job (no logs 15m)      | log tail shows no new lines                      | `hf jobs cancel`; release all assigned files (attempts++) |
| Per-doc extraction error     | `result.error != None` from batch engine         | written to output row's `error` field; **does not fail the file** |
| Permanent file failure       | `attempts == 2` after release                    | mark `failed`; orchestrator continues |

### Recovery scenarios

- **Orchestrator Ctrl-C / crash:** re-run; reconcile picks up outputs that
  landed during the gap. Orphan `assigned` jobs continue running on hub; their
  full log history is replayed via `hf jobs logs`. Idempotent.
- **Re-run after "complete":** zero `pending` files → exits immediately. To
  force re-extraction, manually flip status to `pending` and clear `attempts`.
- **Adding new input files later:** seed step *merges* newly-listed files into
  the manifest as `pending`, never overwrites.

## Output dataset

**Repo:** `cometadata/arxiv-funding-statement-extractions` (private until
flipped public).

**Layout:**

```
predictions/
  shard-00000.parquet           # mirrors input filenames 1:1
  shard-00001.parquet
  ...
run_metadata/
  manifest-snapshot.parquet     # final manifest, pushed at end-of-run
  summary.json                  # final report
README.md                       # auto-written: source, date, config, counts
```

**Final summary** (written locally to
`manifests/arxiv-extractions-2026-04-24-summary.json` and mirrored to
`run_metadata/summary.json` in the output repo):

- Counts by status (`done`, `failed`).
- Total wall time and total HF Job seconds consumed.
- Estimated cost @ $5/hr.
- List of `failed` files with `last_error`.
- Throughput stats from `worker_elapsed_s` series.

**README** auto-generated: source repo + commit SHA, run date, extractor git
ref (`statement-only-extraction`), Tier 2 config, model, throughput, cost,
count of failed input files.

## Open implementation details

- **Input column names** — `--text-column` / `--id-column` are configurable;
  defaults assumed to be `text` / `arxiv_id`. Verify against one parquet from
  `cometadata/arxiv-latex-extract-full-text/results-2026-04-24/` at
  implementation time. Orchestrator should fetch the first parquet's schema at
  startup and emit a clear error if configured columns are missing — *before*
  launching any jobs.
- **Carry-through fields** — beyond `arxiv_id`, `doc_id`, `input_file`,
  `row_idx`: TBD based on actual input schema. Locked in at implementation time
  alongside column-name verification.
