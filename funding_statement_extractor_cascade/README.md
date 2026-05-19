# Funding Statement Extractor Cascade

A 3-stage cascade that takes the VLM-converted markdown of an arXiv paper and
returns a single cleaned funding-acknowledgment statement (or `""` if the paper
has none). Stage 1 (ModernBERT chunk classifier) finds the windows of the
document most likely to contain a funding statement; Stage 2 (ModernBERT span
head) extracts a rough character span from the surviving chunks; Stage 3
(Qwen3-4B + LoRA cleaner) rewrites that rough span into the exact statement as
it should appear in a database, fixing LaTeX markers, hyphenated line breaks,
and abnormal whitespace without paraphrasing.

## Benchmark numbers

Measured on the 596-document `test` split of
[`cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test`](https://huggingface.co/datasets/cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test):

| Metric | F1 |
|---|---|
| best@0.85 (lenient match) | 0.953 |
| strict tsr@0.95 (token-set-ratio >= 0.95) | 0.725 |

## Models

The cascade wraps three Hugging Face models (all CC0):

| Stage | Model | Role |
|---|---|---|
| 1 | [`cometadata/funding-chunk-classifier-modernbert-base`](https://huggingface.co/cometadata/funding-chunk-classifier-modernbert-base) | Binary classifier scoring each 8192-token chunk for the presence of a funding statement. |
| 2 | [`cometadata/funding-extraction-modernbert-base-spanhead`](https://huggingface.co/cometadata/funding-extraction-modernbert-base-spanhead) | Span head emitting start/end logits plus a no-answer probability over a surviving chunk. |
| 3 | [`cometadata/funding-cleaning-qwen3-4b-lora`](https://huggingface.co/cometadata/funding-cleaning-qwen3-4b-lora) | LoRA adapter over [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) that cleans the rough span into the final statement. |

The two ModernBERT stages share the [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
backbone. Weights are downloaded from the Hub on first run.

## Installation

```bash
git clone https://github.com/cometadata/funding-metadata-enrichment.git
cd funding-metadata-enrichment/funding_statement_extractor_cascade

uv venv
source .venv/bin/activate

uv pip install -e .
```

## Quick Start

```bash
# Single markdown file -> JSON
funding-statement-extractor-cascade -i paper.md -o results.json

# A directory of markdown files (module form)
python -m funding_statement_extractor_cascade -i docs/

# A parquet dataset; columns auto-detect, preferring vlm_markdown / arxiv_id
funding-statement-extractor-cascade -i bench.parquet \
  --parquet-text-column vlm_markdown --parquet-id-column arxiv_id
```

## Python API

```python
from funding_statement_extractor_cascade.cascade.loader import load_models
from funding_statement_extractor_cascade.cascade.pipeline import extract_funding
from funding_statement_extractor_cascade.cascade.types import CascadeConfig

cfg = CascadeConfig()
models = load_models(cfg)
result = extract_funding(open("paper.md").read(), models, cfg)
print(result.funding_statement or "(no funding statement)")
```

`extract_funding` returns a `CascadeResult` with the cleaned `funding_statement`,
the pre-cleanup `rough_span`, the stage-1 `chunk_prob`, the stage-2
`no_answer_prob`, a `latency_ms`, and an optional `error` string. To run many
documents, use `extract_funding_batch` from
`funding_statement_extractor_cascade.cascade.batch`, which iterates
`DocPayload` objects and yields a `BatchResult` per document, isolating per-doc
exceptions into `CascadeResult.error` so the stream never aborts mid-run.

## CLI Options

### Required

- `-i, --input` — input markdown file, directory, or parquet dataset.
- `-o, --output` — output JSON file (default: `funding_results.json`).

### Input source

- `--input-format` — force `markdown` or `parquet` instead of auto-detecting.
- `--parquet-text-column` — column with markdown text (default: auto-detect, preferring `vlm_markdown`).
- `--parquet-id-column` — column with a unique id (default: auto-detect, preferring `arxiv_id`).
- `--parquet-batch-size` — rows per parquet read batch (default: 64).

### Cascade engine

- `--skip-cleanup` — skip Stage 3 (Qwen cleaner) and return the raw stage-2 span verbatim.
- `--chunk-keep-threshold` — minimum stage-1 sigmoid probability for a chunk to survive (default: 0.4).
- `--chunk-top-k` — maximum number of surviving chunks fed to Stage 2 (default: 2).
- `--span-no-answer-threshold` — stage-2 no-answer probability above which a chunk is discarded (default: 0.5).
- `--span-max-length` — maximum span length in tokens for the stage-2 end search (default: 300).
- `--cleaner-context-chars` — characters of surrounding context passed to the cleaner on each side of the rough span (default: 400).
- `--cleaner-max-new-tokens` — maximum new tokens the cleaner may generate (default: 512).
- `--dtype` — model dtype: `auto`, `fp32`, `fp16`, `bf16` (default: `bf16`).
- `--device` — torch device for all three models (default: `cuda`).

### Runtime

- `--batch-size` — documents between checkpoint flushes (default: 50).
- `--resume` — resume from a previous checkpoint.
- `--force` — reprocess all documents, ignoring the checkpoint.
- `--retry-failed` — re-process documents marked failed in the checkpoint.
- `--checkpoint-file` — custom checkpoint file path (default: `<output>.checkpoint`).
- `-v, --verbose` — enable verbose logging.

## Output Format

The CLI writes a JSON file shaped as follows:

```json
{
  "timestamp": "2026-05-19T12:00:00",
  "parameters": {
    "input_path": "paper.md",
    "input_format": "markdown",
    "skip_cleanup": false,
    "dtype": "bf16"
  },
  "results": {
    "paper.md": {
      "funding_statement": "This work was supported by NSF grant 12345.",
      "rough_span": "This work was supported by NSF grant 12345.",
      "chunk_prob": 0.97,
      "no_answer_prob": 0.02,
      "latency_ms": 812.4,
      "error": null
    }
  },
  "summary": {
    "total_files": 1,
    "files_with_funding": 1
  }
}
```

Each entry under `results` is keyed by document id. `funding_statement` is the
cleaned answer (`""` if no funding statement was found); `rough_span` is the
pre-cleanup stage-2 output; `chunk_prob` and `no_answer_prob` are diagnostics
(either may be `null` when the cascade short-circuits early); `error` is `null`
on success or a `"TypeName: message"` string when a document failed.

### `--skip-cleanup` semantics

With `--skip-cleanup`, Stage 3 is not loaded or run. `funding_statement` is set
to the raw stage-2 span verbatim, identical to `rough_span`. This is faster and
needs far less VRAM (no Qwen model), at the cost of the LaTeX/whitespace
normalization the cleaner provides.

## Dependencies & GPU

- A CUDA GPU is required; the cascade has no CPU fallback path for production use.
- An A100-large (80GB) is recommended. The full cascade fits comfortably:
  Qwen3-4B in bf16 is roughly 8GB, the two ModernBERT-base models total well
  under 0.5GB, and the KV cache is under 1GB — about 9GB VRAM total.
- With `--skip-cleanup` only the two ModernBERT stages load, needing roughly 0.5GB.
- `transformers >= 4.49` is required: the cleaner is loaded with the `dtype=`
  kwarg (which replaced `torch_dtype=` in 4.49). Older transformers silently
  ignores `dtype`, leaving Qwen in fp32 and risking OOM. The package pins
  `transformers>=4.49,<6`.

## Package Structure

```
funding_statement_extractor_cascade/
├── pyproject.toml
├── README.md
├── LICENSE                                     # MIT
├── .gitignore
├── docs/plans/2026-05-19-funding-statement-extractor-cascade.md
├── tests/                                      # GPU-free, network-free (mocked models)
│   ├── conftest.py
│   ├── cascade_fakes.py
│   ├── test_chunking.py
│   ├── test_cascade_types.py
│   ├── test_cascade_pipeline.py
│   ├── test_cascade_batch.py
│   ├── test_loader.py
│   ├── test_input_sources.py
│   ├── test_checkpointing.py
│   ├── test_settings.py
│   ├── test_vendored_models.py
│   ├── test_smoke_imports.py
│   └── test_cli.py
├── funding_statement_extractor_cascade/
│   ├── __init__.py                             # public API exports
│   ├── __main__.py                             # python -m funding_statement_extractor_cascade
│   ├── exceptions.py
│   ├── cli/
│   │   └── main.py                             # argparse entry point
│   ├── config/
│   │   └── settings.py                         # ApplicationConfig + validate()
│   ├── io/
│   │   ├── loaders.py                          # markdown / dir / parquet loaders
│   │   └── checkpointing.py                    # CheckpointRepository
│   ├── processing/
│   │   └── chunking.py                         # build_chunks(token_count, max_tok, stride)
│   ├── models/
│   │   ├── chunk_classifier/modeling.py        # vendored ChunkClassifier
│   │   └── span_head/modeling.py               # vendored SpanHead
│   └── cascade/
│       ├── types.py                            # CascadeConfig / CascadeResult / DocPayload / BatchResult
│       ├── loader.py                           # load_models(cfg) -> CascadeModels
│       ├── pipeline.py                         # extract_funding() — the 3-stage core
│       ├── batch.py                            # extract_funding_batch() -> Iterator[BatchResult]
│       └── cli.py                              # add_arguments() + run()
└── hf_jobs/
    ├── jobs/
    │   └── funding_statement_extractor_cascade/
    │       ├── README.md
    │       ├── extract_funding_cascade_job.py  # PEP 723 HF Job worker
    │       ├── benchmark_hf_job.py             # F1 vs gold on the bench split
    │       ├── profile_on_h200.py              # single-doc latency profiler
    │       └── tests/
    └── ops/
        └── funding_statement_extractor_cascade/
            ├── README.md
            ├── orchestrate_extractions.py      # local job orchestrator
            ├── _profile_utils.py
            ├── profile_extraction.py
            ├── diff_predictions.py             # cascade-vs-gold diff
            └── tests/
```

## License

MIT. See [LICENSE](LICENSE).
