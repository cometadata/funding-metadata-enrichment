# funding_statement_extractor — HF Jobs scripts

PEP 723 self-contained scripts run via `hf jobs uv run`. Each one's docstring has the exact invocation; this file is just a map.

- `benchmark_hf_job.py` — baseline F1 / precision-recall / throughput benchmark over `cometadata/arxiv-funding-statement-extraction`. Pushes predictions and metrics back to the Hub.
- `extract_funding_job.py` — Tier-2 extraction worker. Runs the ColBERT extractor over a list of input parquets and pushes per-shard predictions to the output dataset. Dispatched in bulk by `ops/funding_statement_extractor/orchestrate_extractions.py`.
- `profile_on_h200.py` — per-phase profiler, with all profiling utilities bundled inline so the script is fetchable and runnable by HF Jobs without the rest of the repo.

Recommended image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`. Flavor depends on the script (`a100-large` for extraction, `h100x1` / `h200x1` for benchmark / profiler).

Tests live in `tests/`.
