# funding-entity-extractor

High-throughput online vLLM utility for extracting structured funder/award metadata from funding statements, using the LoRA at `cometadata/funding-extraction-llama-3.1-8b-instruct-artifact-data-mix-grpo-mixed-reward`.

## Install

```bash
# Client only (lightweight; for hosts that talk to a remote vLLM server)
uv pip install -e .

# Server + client (GPU host running vLLM)
uv pip install -e ".[serve]"

# Development
uv pip install -e ".[dev,serve]"
```

## Serve

Start vLLM with the funding-extraction LoRA preloaded:

```bash
funding-extract serve \
  --port 8000 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096
```

Pass-through args after `--`:

```bash
funding-extract serve --port 8000 -- --enforce-eager --quantization fp8
```

## Run extraction

```bash
funding-extract run \
  --input shard.parquet \
  --output shard.extracted.parquet \
  --vllm-url http://localhost:8000 \
  --concurrency 256
```

Input parquet is expected to have a `predicted_statements: list<string>` column (rows where this list is empty are filtered out). Output preserves all input columns and appends:
- `extracted_funders` — `list<list<struct<funder_name, awards: list<struct<award_ids, funding_scheme, award_title>>>>>` (parallel-indexed with `predicted_statements`; `null` entries on parse failure)
- `extraction_raw` — verbatim model output per statement
- `extraction_error` — `null` on success, else `"ParseError: …"` or `"HTTPError: …"`
- `extraction_latency_ms` — per-statement model latency

Re-running with the same `--output` resumes: rows already present (matched by `arxiv_id, row_idx`) are skipped on the input side.

## Python API

```python
import asyncio
from funding_entity_extractor import extract_statements

async def main():
    results = await extract_statements(
        ["This work was supported by NSF grant DMS-1613002."],
        vllm_url="http://localhost:8000",
        concurrency=256,
    )
    for r in results:
        print(r.funders, r.error)

asyncio.run(main())
```

## Benchmark

`benchmark/benchmark_hf_job.py` is a PEP 723 uv script designed for `hf jobs uv run`. See its docstring for invocation.
