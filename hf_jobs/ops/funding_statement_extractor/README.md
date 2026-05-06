# funding_statement_extractor — local ops scripts

Run locally with the project venv at `funding_statement_extractor/.venv/`.

- `orchestrate_extractions.py` — local pool of concurrent A100 jobs over the 17.5k-file `cometadata/arxiv-latex-extract-full-text/results-2026-04-24/` corpus. Manifest-backed resume, retry, EMA-based rebalancing. Default `--worker-script` is `hf_jobs/funding_statement_extractor/extract_funding_job.py`.
- `build_funded_only_split.py` — re-packs prediction parquets into deterministic 10k-row shards under `funded_only/shard-NNNNN.parquet` on the same Hub dataset, keeping only rows where `predicted_statements` is non-empty.
- `profile_extraction.py` — per-phase profiler over a cached 50-doc sample. Monkey-patches the extractor to attribute per-doc wall-clock to each pipeline phase. `_profile_utils.py` is the shared accumulator.
- `diff_predictions.py` — byte-identity gate that compares `per_doc_predictions` between two profile JSONs. Exits non-zero on any mismatch. Used as the regression check after refactors.

Tests live in `tests/`. Run from the repo root:

    funding_statement_extractor/.venv/bin/pytest \
        ops/funding_statement_extractor/tests \
        hf_jobs/funding_statement_extractor/tests
