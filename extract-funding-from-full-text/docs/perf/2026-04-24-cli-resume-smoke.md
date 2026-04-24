# CLI batch-engine smoke + resume kill-test (Task 16)

**Date:** 2026-04-24
**HEAD at test:** 2085ef7 (CLI: route through batch engine; add --legacy-engine for one release)
**Platform:** macOS (Darwin 24.5.0), CPU path (no CUDA; Tier 2 prefilter on)

## Corpus

- Source: `cometadata/arxiv-funding-statement-extraction`, `data/test.jsonl`
- Trimmed to first 500 rows to keep wall time under ~10 min on Mac CPU
  (plan suggested ≥1000; kill+resume semantics don't depend on corpus size)
- Saved to `/tmp/funding_test.parquet`, columns: `file, statements, text, found, category, augmentation`

## Module entry correction

Plan's example used `python -m funding_statement_extractor.statements.cli`, but that
module has no `if __name__ == "__main__"` block, so it exits silently with code 0.
The real entry point is `funding_statement_extractor.cli.main` (which calls
`statements_cli.add_arguments` + `statements_cli.run`). All runs below use the
correct entry. Task 15's smoke-test note should be updated to reflect this, and
Task 16's plan snippet likewise.

## Run 1 (original, killed with SIGKILL)

Command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -u -m funding_statement_extractor.cli.main \
    -i /tmp/funding_test.parquet -o /tmp/funding_test_out.json \
    --enable-paragraph-prefilter --batch-size 50 --workers 4 \
    --paragraphs-per-batch 1024 --encode-batch-size 64 \
    --parquet-text-column text --parquet-batch-size 64
```

- Launched PID: 72290
- Checkpoint flushes every `--batch-size 50` completions. First flush landed at
  ~t=135 s (model load + first 50 docs). By the time I could capture and act,
  the checkpoint had already advanced to 300 docs.
- `kill -9 72290` issued at checkpoint count = **300 / 500** (60 %).
- Verified no stragglers: `pkill -9 -f 'python.*funding_statement_extractor'`
  cleared remaining multiprocessing workers (SIGKILL on the parent orphans them
  cleanly once the pool's parent pipe closes).

Post-kill file sizes:
- `/tmp/funding_test_out.json` — 133 779 bytes, 300 results
- `/tmp/funding_test_out.json.checkpoint` — 70 209 bytes, 300 entries

## Run 2 (resume)

Same command + `--resume`. PID 72658.

Log:

```
Resuming from checkpoint: 300 documents already processed
Discovered 500 parquet rows to process
Processed 50/200 documents, saved checkpoint
Processed 100/200 documents, saved checkpoint
Processed 150/200 documents, saved checkpoint
Processed 200/200 documents, saved checkpoint
```

Note the `/200` denominator — the batch engine correctly subtracted the 300 already-processed
from the 500 discovered (see `_prepare_run` in `funding_statement_extractor/statements/cli.py`
at the `total_documents = max(discovered_documents - len(processed_lookup), 0)` line).

Resume wall time: **111 s** (model load + 200 remaining docs ≈ 1.8 docs/sec).

Final summary printed by CLI:

```
Total documents processed: 500
Documents with funding: 296
Total statements: 441
```

## Completeness + uniqueness checks

```python
import json
data = json.load(open("/tmp/funding_test_out.json"))
results = data["results"]
assert len(results) == 500                       # pass: 500
assert len(results) == len(set(results.keys()))  # pass: 500 unique
cp = json.load(open("/tmp/funding_test_out.json.checkpoint"))
assert cp["total_processed"] == 500              # pass
```

All three pass. No duplicates, no missing rows.

Note: the parquet rows do not carry unique ids (the `file` column has duplicates —
324 distinct values in 500 rows, presumably because the test split keeps
augmentation variants under the same source file). The CLI synthesizes ids of
the form `funding_test-row-<N>` for parquet rows lacking a unique id column, so
all 500 rows land as distinct keys. This is correct and expected given the
loader's fallback behaviour.

## Flag adaptations

- `--allow-cpu`: not a CLI flag (only in `benchmark_hf_job.py`). Omitted. CLI
  happily runs on CPU without a GPU guard.
- No other flags needed adaptation. All documented flags
  (`--enable-paragraph-prefilter`, `--batch-size`, `--workers`,
  `--paragraphs-per-batch`, `--encode-batch-size`, `--parquet-text-column`,
  `--parquet-batch-size`, `--resume`) worked as expected.

## Wall time accounting

Original run to kill:      ~140 s (model load + 300 docs)
Resume run to completion:   111 s (model load + 200 docs)
Total wall: ~4.2 min active + orchestration overhead.
