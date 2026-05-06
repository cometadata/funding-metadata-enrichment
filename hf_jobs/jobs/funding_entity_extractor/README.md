# funding_entity_extractor — HF Jobs scripts

PEP 723 self-contained scripts run via `hf jobs uv run`.

- `benchmark_hf_job.py` — high-throughput benchmark for the `cometadata/funding-extraction-llama-3.1-8b-instruct-artifact-data-mix-grpo-mixed-reward` LoRA. Spawns `funding-extract serve` as a subprocess, runs async extraction over a parquet shard from `cometadata/arxiv-funding-statement-extractions`, then pushes predictions and metrics to the Hub. See script docstring for the exact `hf jobs uv run` invocation.
