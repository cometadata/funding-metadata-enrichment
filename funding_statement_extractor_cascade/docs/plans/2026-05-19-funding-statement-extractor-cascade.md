# Funding Statement Extractor Cascade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a 3-stage cascade (ModernBERT chunk classifier → ModernBERT span head → Qwen3-4B+LoRA cleaner) that takes VLM-converted markdown of an arXiv paper and outputs the cleaned funding-acknowledgment statement (or `""` if none), wrapped in a standalone Python package and a full HF Job + local orchestrator pipeline.

**Architecture:** A new standalone repo at `/Users/adambuttrick/Downloads/parth_pipeline/` mirroring the v1 layout at `/Users/adambuttrick/Documents/GitHub/funding-metadata-enrichment/` (package + `hf_jobs/{jobs,ops}/<name>/`). The cascade is a pure-Python module wrapping three HF models; the worker is a PEP 723 inline-deps script targeting `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` on `a100-large`; the local orchestrator uses a parquet manifest + EMA + 429-aware retry — identical control flow to v1, only the engine inside the worker differs. The cascade is **independent** of v1; both ship side by side.

**Tech Stack:** Python 3.12, PyTorch ≥ 2.4, transformers ≥ 4.49 (Qwen3 chat template + `dtype=` kwarg), peft ≥ 0.14 (LoRA adapter for cleaner), accelerate, huggingface_hub ≥ 0.26, pyarrow, pydantic, pytest. All package tests run without GPU and without network (mocked models + monkeypatched HfApi/pyarrow).

**Reference codebase (v1):** `/Users/adambuttrick/Documents/GitHub/funding-metadata-enrichment/` — every task below names the closest v1 analog so you can crib patterns. Where this cascade diverges from v1, the divergence is called out explicitly.

---

## Repo layout (final state)

```
parth_pipeline/
├── pyproject.toml                            # mirrors v1 funding_statement_extractor/pyproject.toml
├── README.md                                 # rewritten for cascade; v1 README is the structural template
├── LICENSE                                   # MIT (match v1)
├── .gitignore                                # standard Python
├── docs/plans/2026-05-19-funding-statement-extractor-cascade.md   # this file
├── tests/
│   ├── __init__.py
│   ├── conftest.py                           # sys.path injection; mirrors v1 tests/test_input_sources.py top
│   ├── test_chunking.py                      # processing/chunking.py unit tests (no model)
│   ├── test_cascade_types.py                 # CascadeConfig / CascadeResult dataclass defaults
│   ├── test_cascade_pipeline.py              # 3-stage cascade with mocked models
│   ├── test_cascade_batch.py                 # streaming batch iterator + per-doc error isolation
│   ├── test_input_sources.py                 # loaders.py — markdown / dir / parquet (copy v1 pattern)
│   ├── test_checkpointing.py                 # CheckpointRepository round-trip
│   ├── test_settings.py                      # ApplicationConfig.validate()
│   └── test_cli.py                           # argparse + run() with fake extractor
├── funding_statement_extractor_cascade/
│   ├── __init__.py                           # public API exports
│   ├── __main__.py                           # `python -m funding_statement_extractor_cascade` → cli/main.py:main
│   ├── exceptions.py                         # mirrors v1 funding_statement_extractor/exceptions.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py                           # mirrors v1 funding_statement_extractor/cli/main.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                       # mirrors v1 funding_statement_extractor/config/settings.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── loaders.py                        # mirrors v1 funding_statement_extractor/io/loaders.py (text col defaults to vlm_markdown)
│   │   └── checkpointing.py                  # COPY v1 funding_statement_extractor/io/checkpointing.py verbatim
│   ├── processing/
│   │   ├── __init__.py
│   │   └── chunking.py                       # NEW (no v1 analog): build_chunks(token_count, max_tok, stride)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chunk_classifier/
│   │   │   ├── __init__.py
│   │   │   └── modeling.py                   # VENDORED from cometadata/funding-chunk-classifier-modernbert-base
│   │   └── span_head/
│   │       ├── __init__.py
│   │       └── modeling.py                   # VENDORED from cometadata/funding-extraction-modernbert-base-spanhead
│   └── cascade/
│       ├── __init__.py
│       ├── types.py                          # CascadeConfig + CascadeResult + DocPayload + BatchResult
│       ├── loader.py                         # load_models(cfg) → CascadeModels
│       ├── pipeline.py                       # extract_funding(text, models, cfg) — the 3-stage core
│       ├── batch.py                          # extract_funding_batch(documents, models, cfg) → Iterator[BatchResult]
│       └── cli.py                            # add_arguments() + run() (mirrors v1 statements/cli.py — but cascade-shaped)
└── hf_jobs/
    ├── jobs/
    │   └── funding_statement_extractor_cascade/
    │       ├── README.md
    │       ├── extract_funding_cascade_job.py   # mirrors v1 jobs/funding_statement_extractor/extract_funding_job.py
    │       ├── benchmark_hf_job.py              # mirrors v1 jobs/.../benchmark_hf_job.py (F1 vs gold on bench split)
    │       ├── profile_on_h200.py               # mirrors v1 jobs/.../profile_on_h200.py
    │       └── tests/
    │           ├── conftest.py
    │           └── test_extract_funding_cascade_job.py
    └── ops/
        └── funding_statement_extractor_cascade/
            ├── README.md
            ├── orchestrate_extractions.py       # mirrors v1 ops/.../orchestrate_extractions.py
            ├── _profile_utils.py                # mirrors v1
            ├── profile_extraction.py            # mirrors v1
            ├── diff_predictions.py              # mirrors v1 (cascade vs gold; v1's compared cascade vs v1)
            └── tests/
                ├── conftest.py
                └── test_orchestrate_extractions.py
```

---

## Core data structures (full spec)

### `cascade/types.py`

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CascadeConfig:
    # HF repos (CC0)
    chunk_repo: str = "cometadata/funding-chunk-classifier-modernbert-base"
    span_repo:  str = "cometadata/funding-extraction-modernbert-base-spanhead"
    clean_repo: str = "cometadata/funding-cleaning-qwen3-4b-lora"
    mb_base:    str = "answerdotai/ModernBERT-base"
    qwen_base:  str = "Qwen/Qwen3-4B-Instruct-2507"

    # Stage 1 — chunk classifier
    max_tok: int = 8192
    stride:  int = 4096
    chunk_keep_threshold: float = 0.4
    chunk_top_k: int = 2

    # Stage 2 — span head
    span_no_answer_threshold: float = 0.5
    span_max_length: int = 300

    # Stage 3 — cleaner
    cleaner_context_chars: int = 400
    cleaner_max_new_tokens: int = 512
    skip_cleanup: bool = False

    # Runtime
    device: str = "cuda"
    dtype:  str = "bf16"   # one of: auto, fp32, fp16, bf16


@dataclass
class CascadeResult:
    funding_statement: str            # "" if none
    rough_span: str                   # "" if none
    chunk_prob: Optional[float]       # max stage-1 prob across surviving chunks; None if no chunks
    no_answer_prob: Optional[float]   # min stage-2 no_answer; None if no chunk fed to spanner
    latency_ms: float
    error: Optional[str] = None


@dataclass
class DocPayload:
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    doc_id: str
    result: CascadeResult
    enqueue_ts: float
    yield_ts: float
    metadata: Optional[Dict[str, Any]] = None
```

### `cascade/loader.py`

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CascadeModels:
    mb_tok: object        # transformers PreTrainedTokenizerFast
    chunker: object       # ChunkClassifier (vendored)
    spanner: object       # SpanHead (vendored)
    qwen_tok: object      # transformers PreTrainedTokenizerFast
    cleaner: object       # peft.PeftModel-wrapped Qwen3-4B (or None when skip_cleanup=True)
```

### HF Job `OUTPUT_SCHEMA` (single-statement, with diagnostics)

```python
import pyarrow as pa

OUTPUT_SCHEMA = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("arxiv_id", pa.string()),
    pa.field("shard_id", pa.string()),
    pa.field("input_file", pa.string()),
    pa.field("row_idx", pa.int64()),
    pa.field("funding_statement", pa.string()),   # cleaned answer; "" if none
    pa.field("rough_span", pa.string()),           # stage-2 raw; "" if none
    pa.field("chunk_prob", pa.float64()),
    pa.field("no_answer_prob", pa.float64()),
    pa.field("text_length", pa.int64()),
    pa.field("latency_ms", pa.float64()),
    pa.field("skip_cleanup", pa.bool_()),
    pa.field("error", pa.string()),
])
```

**Divergence from v1:** v1 had `predicted_statements: list<string>` + `predicted_details: list<struct>` because the ColBERT extractor returns N candidate statements per doc. The cascade always returns ≤ 1, so the schema is flat — no list wrapping. This is honest to the model's contract.

### `SYSTEM_CLEAN` (verbatim from training; load-bearing)

```python
SYSTEM_CLEAN = (
    "You are a funding statement cleaner. Given a rough extracted funding "
    "statement and its surrounding context from an academic paper, output "
    "the exact funding statement as it should appear in a database. Clean "
    "up LaTeX markers ($^{N}$, \\textsuperscript), hyphenated line breaks, "
    "and abnormal whitespace, but DO NOT paraphrase. If the rough span is "
    "not actually a funding statement, output the single word: NONE"
)
```

---

## Known risks (read before starting)

1. **Qwen3-4B-Instruct-2507 + transformers version** — The reference snippet calls `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16, device_map=DEVICE)`. The `dtype=` kwarg (replacing `torch_dtype=`) landed in transformers 4.49. Pin `transformers>=4.49,<6` in **both** the package `pyproject.toml` and the PEP 723 worker. Older transformers silently ignores `dtype`, leaving the model in fp32 → OOM on A100-large.
2. **GPU memory** — On A100-large (80GB) all three models fit comfortably: Qwen3-4B bf16 ≈ 8GB, two ModernBERT-base ≈ 0.4GB total, plus KV cache ≪ 1GB. Don't ship with `device_map="auto"` — it can offload pieces to CPU. Use `device_map=DEVICE` (the string `"cuda"` or `"cuda:0"`).
3. **Vendored modeling.py drift** — The two `modeling.py` files are fetched once and committed into the repo. If upstream updates them (rare, but the architecture spec could change), the cascade silently breaks. Mitigation: pin model revisions in `CascadeConfig` (one future step; out of scope here) and add a vendoring docstring at the top of each `modeling.py` recording the commit SHA the copy came from.
4. **PEFT wrapping overhead** — `PeftModel.from_pretrained(...).eval()` keeps the LoRA adapter wrapped at inference time. `merge_and_unload()` would be ~10% faster but might subtly change outputs. **Don't merge** — match the reference snippet exactly. F1 numbers were measured with the wrapped form.
5. **`apply_chat_template` `add_generation_prompt=True`** — Qwen3 requires this; the reference snippet uses it. Don't drop it during refactor.
6. **Span end indexing** — `e = s + int(e_log[s : s + 300].argmax())` — when `s + 300 > seq_len`, Python slicing harmlessly returns a shorter slice. The guard `0 <= s <= e < len(c_offsets)` catches degenerate cases. Both behaviours must be preserved.
7. **Char-offset reconstruction** — `enc["offset_mapping"]` (with `add_special_tokens=False`) gives offsets into the **full original** doc text. Slicing `offsets[st:en]` gives the offsets for the chunk while the values themselves still point into the full doc — reconstructing the rough span via `vlm_markdown[char_s:char_e]` is correct.
8. **Bench dataset structure** — The default input corpus `cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test` has 597 docs in its `test` split. Confirm the column is named `vlm_markdown` and the id column is `arxiv_id` before launching a sweep (Task 0.5 below has a one-liner to check).

---

## Implementation order at a glance

| Phase | Tasks | Commit boundary |
|---|---|---|
| 0 — bootstrap | 0.1 – 0.5 | repo skeleton green-on-pytest (no logic) |
| 1 — vendor models | 1.1 – 1.3 | modeling.py files importable |
| 2 — types + config + exceptions | 2.1 – 2.3 | CascadeConfig/Result + ApplicationConfig |
| 3 — chunking | 3.1 | pure chunking unit-tested |
| 4 — cascade pipeline | 4.0 – 4.6 | 3-stage `extract_funding()` works under mocks |
| 5 — batch + I/O | 5.1 – 5.3 | streaming iterator + loaders + checkpoint |
| 6 — CLI | 6.1 – 6.2 | `funding-statement-extractor-cascade -i x.md` runs end-to-end with mocks |
| 7 — HF Job worker | 7.1 – 7.4 | PEP 723 worker with mocked HfApi tests green |
| 8 — local orchestrator | 8.1 – 8.5 | dry-run orchestrator green |
| 9 — bench / profile / diff | 9.1 – 9.5 | analytics scripts importable + smoke-tested |
| 10 — docs | 10.1 – 10.3 | three READMEs |

Each task below is a single TDD cycle (write failing test → impl → green → commit). Tasks are sized for 2–5 min of focused work each.

---

## Conventions

- **TDD per file:** Write failing test → run, see it fail → write minimum implementation → run, see it pass → commit. Use `pytest -xvs` while iterating; `pytest -q` for the final pass.
- **Commit messages:** Conventional commits — `feat:`, `test:`, `chore:`, `docs:`, `refactor:`. Each task ends with one commit.
- **No emojis** anywhere.
- **No docstrings beyond one line** unless the WHY is non-obvious.
- **No `# Type` imports of v1 modules** — the cascade does not depend on the v1 package.
- **Path discipline:** Cascade tests use `Path(__file__).resolve().parents[1]` `sys.path` injection (v1 pattern from `tests/test_input_sources.py:6-8`).
- **Mocked models everywhere:** No test downloads a real model or hits CUDA. Stage 1/2 mocks return canned logits as plain tensors; the cleaner mock returns a canned string.

---

## Phase 0 — Repo bootstrap

### Task 0.1: Initialize git + .gitignore + LICENSE

**Files:**
- Create: `/Users/adambuttrick/Downloads/parth_pipeline/.gitignore`
- Create: `/Users/adambuttrick/Downloads/parth_pipeline/LICENSE`

**Step 1: Initialize the repo**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && git init -b main
```
Expected: `Initialized empty Git repository in .../parth_pipeline/.git/`

**Step 2: Write `.gitignore`** (standard Python; no v1 file to copy — write fresh)

```
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.venv/
.env
.pytest_cache/
.coverage
htmlcov/
.DS_Store
*.pt
*.bin
*.safetensors
/tmp/
/scratch/
```

**Step 3: Write `LICENSE`** — copy `/Users/adambuttrick/Documents/GitHub/funding-metadata-enrichment/funding_statement_extractor/LICENSE` verbatim (MIT).

**Step 4: Commit**

```bash
git add .gitignore LICENSE && git commit -m "chore: initial gitignore and license"
```

---

### Task 0.2: Write `pyproject.toml`

**Files:**
- Create: `/Users/adambuttrick/Downloads/parth_pipeline/pyproject.toml`

**Mirrors:** v1 `funding_statement_extractor/pyproject.toml`. Diverges in: package name, dependency set (drops `pylate`; adds `transformers`, `peft`, `accelerate`, `huggingface_hub`, `rapidfuzz`).

**Step 1: Write the file**

```toml
[project]
name = "funding-statement-extractor-cascade"
version = "0.1.0"
description = "Cascade for extracting cleaned funding-acknowledgment statements from VLM-converted arXiv markdown."
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "pyarrow>=12.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "torch>=2.4.0",
    "transformers>=4.49,<6",
    "peft>=0.14.0",
    "accelerate>=1.0",
    "huggingface_hub>=0.26.0",
    "rapidfuzz>=3.0",
    "datasets>=3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
]

[project.scripts]
funding-statement-extractor-cascade = "funding_statement_extractor_cascade.cli.main:main"

[tool.setuptools.packages.find]
include = ["funding_statement_extractor_cascade*"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

**Step 2: Commit**

```bash
git add pyproject.toml && git commit -m "chore: pyproject.toml with cascade dependencies"
```

---

### Task 0.3: Create package + tests skeleton

**Files:**
- Create empty `__init__.py` in:
  - `funding_statement_extractor_cascade/__init__.py`
  - `funding_statement_extractor_cascade/cli/__init__.py`
  - `funding_statement_extractor_cascade/config/__init__.py`
  - `funding_statement_extractor_cascade/io/__init__.py`
  - `funding_statement_extractor_cascade/processing/__init__.py`
  - `funding_statement_extractor_cascade/models/__init__.py`
  - `funding_statement_extractor_cascade/models/chunk_classifier/__init__.py`
  - `funding_statement_extractor_cascade/models/span_head/__init__.py`
  - `funding_statement_extractor_cascade/cascade/__init__.py`
  - `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create directories + empty `__init__.py` files**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  mkdir -p funding_statement_extractor_cascade/{cli,config,io,processing,models/chunk_classifier,models/span_head,cascade} \
           tests && \
  for d in \
    funding_statement_extractor_cascade \
    funding_statement_extractor_cascade/cli \
    funding_statement_extractor_cascade/config \
    funding_statement_extractor_cascade/io \
    funding_statement_extractor_cascade/processing \
    funding_statement_extractor_cascade/models \
    funding_statement_extractor_cascade/models/chunk_classifier \
    funding_statement_extractor_cascade/models/span_head \
    funding_statement_extractor_cascade/cascade \
    tests ; do touch "$d/__init__.py" ; done
```

**Step 2: Write `tests/conftest.py`** — sys.path injection so tests can import the package without installation (mirrors v1 `tests/test_input_sources.py:6-8`):

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

**Step 3: Run pytest to confirm green-on-no-tests baseline**

Run: `cd /Users/adambuttrick/Downloads/parth_pipeline && uv venv && source .venv/bin/activate && uv pip install -e .[dev] && pytest -q`

(Per @feedback_python_env_uv — use uv for the venv, not brew/system pip.)

Expected: `no tests ran in 0.0Xs` (exit code 5 from pytest is fine for "no tests").

**Step 4: Commit**

```bash
git add funding_statement_extractor_cascade tests && git commit -m "chore: package and test directory skeleton"
```

---

### Task 0.4: `exceptions.py` + smoke import test

**Files:**
- Create: `funding_statement_extractor_cascade/exceptions.py`
- Create: `tests/test_smoke_imports.py`

**Mirrors:** v1 `funding_statement_extractor/exceptions.py` (one-class-per-line, all subclass `FundingStatementExtractorCascadeError`).

**Step 1: Write failing test** — `tests/test_smoke_imports.py`:

```python
def test_package_imports():
    import funding_statement_extractor_cascade
    from funding_statement_extractor_cascade import exceptions

    assert issubclass(exceptions.ConfigurationError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.DocumentLoadError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.ModelLoadError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.ExtractionError, exceptions.FundingStatementExtractorCascadeError)
    assert issubclass(exceptions.CheckpointError, exceptions.FundingStatementExtractorCascadeError)
```

**Step 2: Run test → FAIL** with `ModuleNotFoundError: No module named 'funding_statement_extractor_cascade.exceptions'`.

Run: `pytest tests/test_smoke_imports.py -xvs`

**Step 3: Write `exceptions.py`:**

```python
class FundingStatementExtractorCascadeError(Exception):
    pass


class ConfigurationError(FundingStatementExtractorCascadeError):
    pass


class DocumentLoadError(FundingStatementExtractorCascadeError):
    pass


class ModelLoadError(FundingStatementExtractorCascadeError):
    pass


class ExtractionError(FundingStatementExtractorCascadeError):
    pass


class CheckpointError(FundingStatementExtractorCascadeError):
    pass


class ProcessingError(FundingStatementExtractorCascadeError):
    pass
```

**Step 4: Run test → PASS.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/exceptions.py tests/test_smoke_imports.py && \
  git commit -m "feat: exceptions module with smoke import test"
```

---

### Task 0.5: Sanity-check the input corpus schema

**Files:** none (one-shot diagnostic)

**Purpose:** Before building anything that consumes the bench corpus, confirm the dataset has `vlm_markdown` (text) and `arxiv_id` (id) columns in its `test` split.

**Step 1: Run a one-liner with `datasets` (this requires network):**

```bash
python -c "from datasets import load_dataset; d=load_dataset('cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test', split='test'); print('rows=', len(d)); print('cols=', d.column_names)"
```

Expected: `rows= 597` and a list including `vlm_markdown` and `arxiv_id`. If column names differ, **stop and update `CascadeConfig` and worker defaults** before continuing.

If network is unavailable, skip this task and revisit before Task 7.1. No commit needed.

---

## Phase 1 — Vendor modeling.py from the two ModernBERT repos

> **Background:** Both custom ModernBERT models ship a `modeling.py` with the `ChunkClassifier` and `SpanHead` class definitions. The reference snippet fetches them via `hf_hub_download` + `importlib`. We're vendoring instead (per the brainstorming decision) so the package is self-contained and type-checkable.
>
> **Vendoring note:** At the top of each vendored `modeling.py` add a comment recording the upstream repo, the file's commit SHA, and the date pulled. If the upstream architecture changes, that SHA tells you what state was tested.

### Task 1.1: Vendor `chunk_classifier/modeling.py`

**Files:**
- Create: `funding_statement_extractor_cascade/models/chunk_classifier/modeling.py`

**Step 1: Download the file** (requires network)

```bash
mkdir -p /tmp/vendor && \
  python -c "from huggingface_hub import hf_hub_download; \
             p=hf_hub_download('cometadata/funding-chunk-classifier-modernbert-base','modeling.py'); \
             import shutil; shutil.copy(p,'/tmp/vendor/chunk_modeling.py'); \
             print('saved to', '/tmp/vendor/chunk_modeling.py')"
```

**Step 2: Capture the upstream commit SHA**

```bash
python -c "from huggingface_hub import HfApi; \
           commits = HfApi().list_repo_commits('cometadata/funding-chunk-classifier-modernbert-base'); \
           print(next(iter(commits)).commit_id)"
```

Record the SHA (e.g. `abc1234…`) — you'll paste it into the header comment.

**Step 3: Copy + prepend provenance header**

```bash
{ echo "# Vendored from https://huggingface.co/cometadata/funding-chunk-classifier-modernbert-base"; \
  echo "# Source: modeling.py at commit <PASTE-SHA-HERE> on 2026-05-19"; \
  echo "# If upstream changes the ChunkClassifier architecture, re-vendor and update this header."; \
  echo; \
  cat /tmp/vendor/chunk_modeling.py; } > /Users/adambuttrick/Downloads/parth_pipeline/funding_statement_extractor_cascade/models/chunk_classifier/modeling.py
```

**Step 4: Verify the file imports without error**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  python -c "from funding_statement_extractor_cascade.models.chunk_classifier.modeling import ChunkClassifier; print(ChunkClassifier)"
```

Expected: `<class 'funding_statement_extractor_cascade.models.chunk_classifier.modeling.ChunkClassifier'>`. **Do not instantiate** — the constructor downloads weights.

If the import fails, inspect `modeling.py` for relative imports or missing deps and either pin the missing dep in `pyproject.toml` or rewrite the import to use the fully qualified path.

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/models/chunk_classifier/modeling.py && \
  git commit -m "feat: vendor ChunkClassifier modeling.py from cometadata HF repo"
```

---

### Task 1.2: Vendor `span_head/modeling.py`

**Files:**
- Create: `funding_statement_extractor_cascade/models/span_head/modeling.py`

Repeat Task 1.1 verbatim against `cometadata/funding-extraction-modernbert-base-spanhead`. Header points to the spanhead repo + its commit SHA. Final verification:

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  python -c "from funding_statement_extractor_cascade.models.span_head.modeling import SpanHead; print(SpanHead)"
```

**Commit:**

```bash
git add funding_statement_extractor_cascade/models/span_head/modeling.py && \
  git commit -m "feat: vendor SpanHead modeling.py from cometadata HF repo"
```

---

### Task 1.3: Vendored-class smoke test

**Files:**
- Create: `tests/test_vendored_models.py`

**Step 1: Write the test** — imports only; no instantiation (constructors download weights).

```python
def test_chunk_classifier_class_is_importable():
    from funding_statement_extractor_cascade.models.chunk_classifier.modeling import ChunkClassifier
    assert ChunkClassifier.__name__ == "ChunkClassifier"


def test_span_head_class_is_importable():
    from funding_statement_extractor_cascade.models.span_head.modeling import SpanHead
    assert SpanHead.__name__ == "SpanHead"
```

**Step 2: Run** `pytest tests/test_vendored_models.py -xvs` — expected: 2 passed.

**Step 3: Commit**

```bash
git add tests/test_vendored_models.py && \
  git commit -m "test: smoke imports for vendored ChunkClassifier and SpanHead"
```

---

## Phase 2 — Types, config, settings

### Task 2.1: `cascade/types.py` — data classes

**Files:**
- Create: `funding_statement_extractor_cascade/cascade/types.py`
- Create: `tests/test_cascade_types.py`

**Mirrors:** v1 `statements/models.py` for the dataclass pattern. Diverges: cascade uses plain `@dataclass`, not `BaseModel`, because we don't need JSON serialization at this layer (the HF Job worker builds parquet rows directly).

**Step 1: Write failing test** — `tests/test_cascade_types.py`:

```python
from funding_statement_extractor_cascade.cascade.types import (
    BatchResult,
    CascadeConfig,
    CascadeResult,
    DocPayload,
)


def test_cascade_config_defaults_match_reference_snippet():
    cfg = CascadeConfig()
    assert cfg.chunk_repo == "cometadata/funding-chunk-classifier-modernbert-base"
    assert cfg.span_repo == "cometadata/funding-extraction-modernbert-base-spanhead"
    assert cfg.clean_repo == "cometadata/funding-cleaning-qwen3-4b-lora"
    assert cfg.mb_base == "answerdotai/ModernBERT-base"
    assert cfg.qwen_base == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.max_tok == 8192
    assert cfg.stride == 4096
    assert cfg.chunk_keep_threshold == 0.4
    assert cfg.chunk_top_k == 2
    assert cfg.span_no_answer_threshold == 0.5
    assert cfg.span_max_length == 300
    assert cfg.cleaner_context_chars == 400
    assert cfg.cleaner_max_new_tokens == 512
    assert cfg.skip_cleanup is False
    assert cfg.device == "cuda"
    assert cfg.dtype == "bf16"


def test_cascade_config_is_frozen():
    import dataclasses
    cfg = CascadeConfig()
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        cfg.max_tok = 1024  # type: ignore[misc]


def test_cascade_result_defaults():
    r = CascadeResult(funding_statement="", rough_span="", chunk_prob=None,
                      no_answer_prob=None, latency_ms=0.0)
    assert r.error is None
    assert r.funding_statement == ""


def test_doc_payload_metadata_optional():
    d = DocPayload(doc_id="x", text="hello")
    assert d.metadata is None


def test_batch_result_carries_through_metadata():
    r = CascadeResult(funding_statement="x", rough_span="x", chunk_prob=0.9,
                      no_answer_prob=0.1, latency_ms=12.3)
    b = BatchResult(doc_id="d1", result=r, enqueue_ts=0.0, yield_ts=0.1,
                    metadata={"input_file": "f.parquet"})
    assert b.metadata["input_file"] == "f.parquet"
    assert b.result.funding_statement == "x"
```

**Step 2: Run → FAIL** (`ModuleNotFoundError`).

**Step 3: Write `cascade/types.py`** — exactly the spec from the Data Structures section above. Don't add fields the test doesn't require; YAGNI.

**Step 4: Run → 5 passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/types.py tests/test_cascade_types.py && \
  git commit -m "feat: CascadeConfig, CascadeResult, DocPayload, BatchResult dataclasses"
```

---

### Task 2.2: `config/settings.py` — `ApplicationConfig` hierarchy

**Files:**
- Create: `funding_statement_extractor_cascade/config/settings.py`
- Create: `tests/test_settings.py`

**Mirrors:** v1 `config/settings.py` (one dataclass per concern: input, output, runtime, processing). Diverges: drops `ExtractionSettings.colbert_model` / `threshold` / `top_k`; replaces with `CascadeRuntimeSettings` carrying the cascade-specific overrides that can come from CLI flags.

**Step 1: Write failing test** — `tests/test_settings.py`:

```python
from pathlib import Path

import pytest

from funding_statement_extractor_cascade.config.settings import (
    ApplicationConfig,
    CascadeRuntimeSettings,
    InputSettings,
    OutputSettings,
    ProcessingSettings,
    RuntimeSettings,
)
from funding_statement_extractor_cascade.exceptions import ConfigurationError


def _make_app(tmp_path, **overrides):
    src = tmp_path / "in.md"
    src.write_text("x")
    return ApplicationConfig(
        input=InputSettings(path=src),
        output=OutputSettings(output_path=tmp_path / "out.json",
                              checkpoint_path=tmp_path / "out.json.checkpoint"),
        cascade=CascadeRuntimeSettings(),
        processing=ProcessingSettings(),
        runtime=RuntimeSettings(),
    )


def test_default_runtime_workers_is_none(tmp_path):
    cfg = _make_app(tmp_path)
    assert cfg.runtime.workers is None


def test_default_cascade_settings_align_with_cascade_config(tmp_path):
    cfg = _make_app(tmp_path)
    assert cfg.cascade.skip_cleanup is False
    assert cfg.cascade.dtype == "bf16"
    assert cfg.cascade.chunk_keep_threshold == 0.4


def test_validate_passes_with_existing_input(tmp_path):
    cfg = _make_app(tmp_path)
    cfg.validate()  # no raise


def test_validate_rejects_missing_input(tmp_path):
    src = tmp_path / "missing.md"  # not created
    cfg = ApplicationConfig(
        input=InputSettings(path=src),
        output=OutputSettings(output_path=tmp_path / "o.json",
                              checkpoint_path=tmp_path / "c.json"),
        cascade=CascadeRuntimeSettings(),
        processing=ProcessingSettings(),
        runtime=RuntimeSettings(),
    )
    with pytest.raises(ConfigurationError):
        cfg.validate()
```

**Step 2: Run → FAIL.**

**Step 3: Write `config/settings.py`:**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from funding_statement_extractor_cascade.exceptions import ConfigurationError


@dataclass
class InputSettings:
    path: Path
    input_format: Optional[str] = None
    parquet_text_column: Optional[str] = None
    parquet_id_column: Optional[str] = None
    parquet_batch_size: int = 64


@dataclass
class OutputSettings:
    output_path: Path
    checkpoint_path: Path


@dataclass
class CascadeRuntimeSettings:
    """Cascade hyperparameters and engine knobs that can be overridden from the CLI.

    Keeps CascadeConfig (cascade/types.py) immutable + opinionated; this layer
    is the mutable view used by the app / CLI.
    """
    chunk_repo: str = "cometadata/funding-chunk-classifier-modernbert-base"
    span_repo: str = "cometadata/funding-extraction-modernbert-base-spanhead"
    clean_repo: str = "cometadata/funding-cleaning-qwen3-4b-lora"
    max_tok: int = 8192
    stride: int = 4096
    chunk_keep_threshold: float = 0.4
    chunk_top_k: int = 2
    span_no_answer_threshold: float = 0.5
    span_max_length: int = 300
    cleaner_context_chars: int = 400
    cleaner_max_new_tokens: int = 512
    skip_cleanup: bool = False
    device: str = "cuda"
    dtype: str = "bf16"


@dataclass
class ProcessingSettings:
    skip_extraction: bool = False


@dataclass
class RuntimeSettings:
    batch_size: int = 50            # flush interval
    workers: Optional[int] = None    # reserved for multiprocess input prep
    resume: bool = False
    force: bool = False
    verbose: bool = False
    retry_failed: bool = False


@dataclass
class ApplicationConfig:
    input: InputSettings
    output: OutputSettings
    cascade: CascadeRuntimeSettings
    processing: ProcessingSettings
    runtime: RuntimeSettings

    def validate(self) -> None:
        if not self.input.path.exists():
            raise ConfigurationError(f"Input path {self.input.path} does not exist.")
        if self.input.input_format and self.input.input_format not in {"markdown", "parquet"}:
            raise ConfigurationError("input_format must be one of: markdown, parquet.")
        if self.cascade.chunk_keep_threshold < 0 or self.cascade.chunk_keep_threshold > 1:
            raise ConfigurationError("chunk_keep_threshold must be in [0, 1].")
        if self.cascade.span_no_answer_threshold < 0 or self.cascade.span_no_answer_threshold > 1:
            raise ConfigurationError("span_no_answer_threshold must be in [0, 1].")
        if self.cascade.max_tok < self.cascade.stride:
            raise ConfigurationError("max_tok must be >= stride.")
        if self.cascade.dtype not in {"auto", "fp32", "fp16", "bf16"}:
            raise ConfigurationError("dtype must be one of: auto, fp32, fp16, bf16.")
```

**Step 4: Run → 4 passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/config/settings.py tests/test_settings.py && \
  git commit -m "feat: ApplicationConfig hierarchy with CascadeRuntimeSettings"
```

---

### Task 2.3: Public re-exports in package `__init__.py`

**Files:**
- Modify: `funding_statement_extractor_cascade/__init__.py`
- Modify: `tests/test_smoke_imports.py`

**Step 1: Extend the smoke test** to assert top-level re-exports:

```python
def test_top_level_public_api():
    import funding_statement_extractor_cascade as pkg

    assert hasattr(pkg, "CascadeConfig")
    assert hasattr(pkg, "CascadeResult")
    assert hasattr(pkg, "DocPayload")
    assert hasattr(pkg, "BatchResult")
    assert hasattr(pkg, "__version__")
```

**Step 2: Write `__init__.py`:**

```python
from funding_statement_extractor_cascade.cascade.types import (
    BatchResult,
    CascadeConfig,
    CascadeResult,
    DocPayload,
)

__version__ = "0.1.0"

__all__ = ["BatchResult", "CascadeConfig", "CascadeResult", "DocPayload", "__version__"]
```

**Step 3: Run → pass.**

**Step 4: Commit**

```bash
git add funding_statement_extractor_cascade/__init__.py tests/test_smoke_imports.py && \
  git commit -m "feat: re-export public cascade types from package root"
```

---

## Phase 3 — Chunking helper

> **Why a separate module:** The chunking logic is pure, deterministic, and side-effect-free — perfect for unit tests with zero infrastructure. Lifting it out of the pipeline file also lets us test the edge cases (length exactly `MAX_TOK`, short docs, the `if en == len(ids): break` termination) without ever loading a model.

### Task 3.1: `processing/chunking.py` — `build_chunks`

**Files:**
- Create: `funding_statement_extractor_cascade/processing/chunking.py`
- Create: `tests/test_chunking.py`

**Step 1: Write failing test:**

```python
from funding_statement_extractor_cascade.processing.chunking import build_chunks


def test_short_doc_yields_single_full_chunk():
    chunks = build_chunks(token_count=100, max_tok=8192, stride=4096)
    assert chunks == [(0, 100)]


def test_doc_exactly_max_tok_yields_single_chunk():
    chunks = build_chunks(token_count=8192, max_tok=8192, stride=4096)
    assert chunks == [(0, 8192)]


def test_long_doc_overlaps_at_stride():
    # 12000 tokens with max_tok=8192, stride=4096:
    #   [0, 8192], [4096, 12000]  — second chunk hits the end, loop breaks
    chunks = build_chunks(token_count=12000, max_tok=8192, stride=4096)
    assert chunks == [(0, 8192), (4096, 12000)]


def test_very_long_doc_produces_full_overlap_grid():
    # 20000 tokens: chunks at 0, 4096, 8192, 12288 — last chunk clamps and breaks
    chunks = build_chunks(token_count=20000, max_tok=8192, stride=4096)
    assert chunks == [(0, 8192), (4096, 12288), (8192, 16384), (12288, 20000)]


def test_doc_one_token_past_max_yields_two_chunks():
    chunks = build_chunks(token_count=8193, max_tok=8192, stride=4096)
    assert chunks == [(0, 8192), (4096, 8193)]
```

**Step 2: Run → FAIL** (ImportError).

**Step 3: Write `processing/chunking.py`:**

```python
from typing import List, Tuple


def build_chunks(*, token_count: int, max_tok: int, stride: int) -> List[Tuple[int, int]]:
    """Reproduce the chunking loop from the reference snippet exactly.

    Short docs (<= max_tok) get a single [0, token_count] chunk; longer docs
    get overlapping (start, start + max_tok) windows stepped by `stride`. The
    loop terminates the first time a window reaches token_count.
    """
    if token_count <= max_tok:
        return [(0, token_count)]

    chunks: List[Tuple[int, int]] = []
    for st in range(0, token_count, stride):
        en = min(st + max_tok, token_count)
        chunks.append((st, en))
        if en == token_count:
            break
    return chunks
```

**Step 4: Run → 5 passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/processing/chunking.py tests/test_chunking.py && \
  git commit -m "feat: chunk-windowing helper with edge-case tests"
```

---

## Phase 4 — Cascade pipeline (3 stages, mocked-model tests)

> **Test strategy:** Stages 1 and 2 take real `torch.Tensor` logits as output; we build fake `ChunkClassifier` / `SpanHead` callables that return canned tensors so the assertions exercise the decoding logic only. The cleaner mock is a callable that captures the prompt and returns a canned id sequence. **No CUDA, no real models, no network.**
>
> The pipeline functions accept the models as an explicit `CascadeModels` dataclass argument — pure dependency injection. This is what makes mocking trivial.
>
> **Important faithfulness constraints:**
> - Use `torch.amp.autocast("cuda", dtype=torch.bfloat16)` exactly as in the reference. In tests, monkeypatch this so it's a no-op on CPU (see Task 4.0).
> - `torch.no_grad()` wraps everything.
> - `enc = mb_tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)` — all three kwargs are load-bearing.
> - The `cleaner.generate(...)` call must use `do_sample=False`, `max_new_tokens=cfg.cleaner_max_new_tokens`, `pad_token_id=qwen_tok.eos_token_id`.

### Task 4.0: Test helpers — fake tokenizer + autocast no-op

**Files:**
- Create: `tests/cascade_fakes.py`

**Step 1: Write the fakes** — used by every Phase 4 test:

```python
"""Test doubles for the cascade pipeline. No CUDA, no real models, no network."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch


@contextmanager
def autocast_noop(*args, **kwargs):
    """Replacement for torch.amp.autocast — does nothing on CPU."""
    yield


class FakeMbTokenizer:
    """Mimics the ModernBERT tokenizer's __call__ contract used by pipeline.py.

    Always returns 1 token per character so offset bookkeeping is trivial:
    the offset_mapping is [(0,1),(1,2),...,(n-1,n)] for an n-char input.
    """

    def __call__(self, text: str, *, add_special_tokens=False,
                 return_offsets_mapping=True, truncation=False):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        assert truncation is False
        n = len(text)
        return {
            "input_ids": list(range(n)),
            "offset_mapping": [(i, i + 1) for i in range(n)],
        }


@dataclass
class FakeQwenTokenizer:
    """Minimal stand-in for the Qwen tokenizer's chat-template + decode methods."""
    eos_token_id: int = 0
    captured_messages: List[List[dict]] = None

    def __post_init__(self):
        self.captured_messages = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_tensors):
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        self.captured_messages.append(messages)
        # Pretend the rendered prompt is 10 tokens of zeros.
        return torch.zeros((1, 10), dtype=torch.long)

    def decode(self, ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        # The fake cleaner stuffs the desired output into ids as raw bytes.
        return getattr(self, "_canned_output", "")


class FakeChunker:
    """Returns a fixed logit per call, sequenced from `scalars` (one per chunk)."""

    def __init__(self, scalars: List[float]):
        self._scalars = list(scalars)
        self.calls = 0

    def __call__(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        s = self._scalars[self.calls]
        self.calls += 1
        return torch.tensor(s)


class FakeSpanner:
    """Returns canned (start_logits, end_logits, no_answer_logit) per call."""

    def __init__(self, calls: List[Tuple[List[float], List[float], float]]):
        self._calls = list(calls)
        self.idx = 0

    def __call__(self, ids: torch.Tensor, attn: torch.Tensor):
        s_log, e_log, na = self._calls[self.idx]
        self.idx += 1
        return (
            torch.tensor(s_log).unsqueeze(0),
            torch.tensor(e_log).unsqueeze(0),
            torch.tensor(na),
        )


class FakeCleaner:
    """Records the prompt ids and returns a tensor whose decoded value is `canned`."""

    def __init__(self, canned: str, qwen_tok: FakeQwenTokenizer):
        self._canned = canned
        self._qwen_tok = qwen_tok
        self.last_prompt = None

    def generate(self, prompt_ids: torch.Tensor, *, max_new_tokens: int,
                 do_sample: bool, pad_token_id: int) -> torch.Tensor:
        assert do_sample is False
        self.last_prompt = prompt_ids
        # Set up the fake tokenizer to return our canned string from decode().
        self._qwen_tok._canned_output = self._canned
        # Return prompt + a single synthetic generated token (the tokenizer ignores it).
        return torch.cat([prompt_ids, torch.tensor([[1]])], dim=1)
```

**Step 2: No test for the fakes themselves** — they're test infrastructure; their correctness is exercised by the cascade tests in 4.1–4.5.

**Step 3: Commit**

```bash
git add tests/cascade_fakes.py && \
  git commit -m "test: fake tokenizer, chunker, spanner, cleaner for cascade tests"
```

---

### Task 4.1: `cascade/loader.py` — placeholder + import

**Files:**
- Create: `funding_statement_extractor_cascade/cascade/loader.py`

**Why now:** The pipeline needs a `CascadeModels` dataclass to type its model bundle. We define the class now and stub the loader; the real loader lands in Task 4.6.

**Step 1: Write the file**

```python
"""Model loader for the cascade. Real loading lands in Task 4.6; this is the type."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CascadeModels:
    mb_tok: object        # transformers tokenizer
    chunker: object       # vendored ChunkClassifier
    spanner: object       # vendored SpanHead
    qwen_tok: Optional[object]      # None when skip_cleanup=True
    cleaner: Optional[object]       # None when skip_cleanup=True
```

**Step 2: No test required** — Task 4.5 exercises CascadeModels through the pipeline tests.

**Step 3: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/loader.py && \
  git commit -m "feat: CascadeModels dataclass (loader stub; real load lands later)"
```

---

### Task 4.2: Stage 1 (chunk classifier) decoding

**Files:**
- Create: `funding_statement_extractor_cascade/cascade/pipeline.py`
- Create: `tests/test_cascade_pipeline.py`

**Step 1: Write failing test** — minimal Stage-1 surface:

```python
from unittest.mock import patch

import torch

from funding_statement_extractor_cascade.cascade.loader import CascadeModels
from funding_statement_extractor_cascade.cascade.pipeline import _run_stage1
from funding_statement_extractor_cascade.cascade.types import CascadeConfig

from tests.cascade_fakes import FakeChunker, FakeMbTokenizer, autocast_noop


def _models(chunker):
    return CascadeModels(mb_tok=FakeMbTokenizer(), chunker=chunker,
                         spanner=None, qwen_tok=None, cleaner=None)


def test_stage1_short_doc_uses_single_chunk():
    text = "x" * 100  # tokenizer = 1 token per char
    chunker = FakeChunker(scalars=[5.0])  # sigmoid(5.0) ~= 0.993 → above 0.4
    cfg = CascadeConfig()
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        ids, offsets, surviving = _run_stage1(text, _models(chunker), cfg)
    assert chunker.calls == 1
    assert len(surviving) == 1
    assert surviving[0][1:] == (0, 100)  # (prob, st, en) — st/en
    assert surviving[0][0] > 0.99


def test_stage1_filters_below_threshold():
    text = "x" * 100
    chunker = FakeChunker(scalars=[-5.0])  # sigmoid(-5.0) ~ 0.007 → below 0.4
    cfg = CascadeConfig()
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        _, _, surviving = _run_stage1(text, _models(chunker), cfg)
    assert surviving == []


def test_stage1_returns_top_2_above_threshold():
    text = "x" * (8192 + 4096 + 2000)  # 3 chunks worth
    # chunk_probs after sigmoid roughly: [0.99, 0.27, 0.88]
    chunker = FakeChunker(scalars=[5.0, -1.0, 2.0])
    cfg = CascadeConfig()
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        _, _, surviving = _run_stage1(text, _models(chunker), cfg)
    # Top-2 surviving, sorted by descending prob
    assert len(surviving) == 2
    probs = [p for (p, _, _) in surviving]
    assert probs == sorted(probs, reverse=True)
    assert all(p >= 0.4 for p in probs)
```

**Step 2: Run → FAIL (no pipeline module).**

**Step 3: Implement `_run_stage1` in `pipeline.py`:**

```python
from typing import List, Tuple

import torch

from funding_statement_extractor_cascade.cascade.loader import CascadeModels
from funding_statement_extractor_cascade.cascade.types import CascadeConfig
from funding_statement_extractor_cascade.processing.chunking import build_chunks


def _run_stage1(text: str, models: CascadeModels, cfg: CascadeConfig
                ) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[float, int, int]]]:
    """Returns (ids, offsets, surviving_chunks).

    surviving_chunks = top-k (by prob desc) chunks with sigmoid(logit) >= threshold.
    """
    enc = models.mb_tok(text, add_special_tokens=False,
                         return_offsets_mapping=True, truncation=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]

    chunks = build_chunks(token_count=len(ids), max_tok=cfg.max_tok, stride=cfg.stride)
    chunk_probs: List[Tuple[float, int, int]] = []
    for st, en in chunks:
        c_ids = torch.tensor(ids[st:en]).unsqueeze(0).to(cfg.device)
        c_attn = torch.ones_like(c_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logit = models.chunker(c_ids, c_attn).float()
        chunk_probs.append((torch.sigmoid(logit).item(), st, en))

    chunk_probs.sort(key=lambda x: -x[0])
    surviving = [c for c in chunk_probs[: cfg.chunk_top_k]
                 if c[0] >= cfg.chunk_keep_threshold]
    return ids, offsets, surviving
```

**Note:** The fake's `.to(cfg.device)` is harmless because fake tensors are CPU. In production this moves to GPU; the device string from cfg.device is honored.

**Step 4: Run → 3 passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/pipeline.py tests/test_cascade_pipeline.py && \
  git commit -m "feat: cascade Stage 1 — ModernBERT chunk classifier with top-k filter"
```

---

### Task 4.3: Stage 2 (span head) decoding

**Files:**
- Modify: `funding_statement_extractor_cascade/cascade/pipeline.py`
- Modify: `tests/test_cascade_pipeline.py`

**Step 1: Append failing tests** (the Stage 2 surface):

```python
from tests.cascade_fakes import FakeSpanner


def _models_2(chunker, spanner):
    return CascadeModels(mb_tok=FakeMbTokenizer(), chunker=chunker,
                         spanner=spanner, qwen_tok=None, cleaner=None)


def test_stage2_skips_chunks_above_no_answer_threshold():
    text = "x" * 100
    chunker = FakeChunker(scalars=[5.0])
    # no_answer logit = 5.0 → sigmoid ~0.993 → above 0.5 → skip
    spanner = FakeSpanner(calls=[([0.1] * 100, [0.1] * 100, 5.0)])
    cfg = CascadeConfig()
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        _, _, surv = _run_stage1(text, _models_2(chunker, spanner), cfg)
        from funding_statement_extractor_cascade.cascade.pipeline import _run_stage2
        best = _run_stage2(text, [], _models_2(chunker, spanner), cfg) if not surv else \
               _run_stage2(text, surv, _models_2(chunker, spanner), cfg)
    # offset_mapping for `text`
    assert best is None


def test_stage2_returns_lowest_no_answer_chunk():
    text = "abcdefghij" * 10   # 100 chars/tokens
    chunker = FakeChunker(scalars=[5.0, 5.0])
    # First chunk: no_answer = -5.0 (sigmoid ~0.007), start argmax @ 2, end argmax @ 4 (relative)
    # Second chunk: no_answer = -2.0 (sigmoid ~0.12)
    s1 = [0.0] * 100; s1[2] = 5.0
    e1 = [0.0] * 100; e1[4] = 5.0
    s2 = [0.0] * 100; s2[10] = 5.0
    e2 = [0.0] * 100; e2[12] = 5.0
    spanner = FakeSpanner(calls=[(s1, e1, -5.0), (s2, e2, -2.0)])
    cfg = CascadeConfig()
    # Skip stage1; provide synthetic surviving with two chunks
    surv = [(0.99, 0, 100), (0.95, 0, 100)]   # both span full text in our fake
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        from funding_statement_extractor_cascade.cascade.pipeline import _run_stage2
        # Stage 2 needs offsets; build them as 1:1 char map of length 100.
        offsets = [(i, i + 1) for i in range(100)]
        ids = list(range(100))
        best = _run_stage2(text, surv, _models_2(chunker, spanner), cfg,
                            ids=ids, offsets=offsets)
    # Lowest no_answer = first chunk's -5.0
    assert best is not None
    no_a, rough_span, c_s, c_e = best
    assert no_a < 0.01
    # span = text[c_offsets[2][0] : c_offsets[4][1]] = text[2:5]
    assert rough_span == text[2:5]


def test_stage2_returns_none_when_all_skipped():
    text = "x" * 100
    chunker = FakeChunker(scalars=[])
    spanner = FakeSpanner(calls=[([0.0] * 100, [0.0] * 100, 5.0)])
    cfg = CascadeConfig()
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        from funding_statement_extractor_cascade.cascade.pipeline import _run_stage2
        ids = list(range(100))
        offsets = [(i, i + 1) for i in range(100)]
        best = _run_stage2(text, [(0.9, 0, 100)],
                            _models_2(chunker, spanner), cfg, ids=ids, offsets=offsets)
    assert best is None
```

**Step 2: Run → FAIL (no `_run_stage2`).**

**Step 3: Append `_run_stage2` to `pipeline.py`:**

```python
from typing import Optional


def _run_stage2(
    text: str,
    surviving: List[Tuple[float, int, int]],
    models: CascadeModels,
    cfg: CascadeConfig,
    *,
    ids: List[int],
    offsets: List[Tuple[int, int]],
) -> Optional[Tuple[float, str, int, int]]:
    """Returns (no_answer_prob, rough_span, char_start, char_end) for the best chunk, or None."""
    best: Optional[Tuple[float, str, int, int]] = None
    for _, st, en in surviving:
        c_offsets = offsets[st:en]
        c_ids = torch.tensor(ids[st:en]).unsqueeze(0).to(cfg.device)
        c_attn = torch.ones_like(c_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            s_log, e_log, na_log = models.spanner(c_ids, c_attn)
        s_log = s_log.squeeze(0).float().cpu()
        e_log = e_log.squeeze(0).float().cpu()
        no_a = torch.sigmoid(na_log).item()
        if no_a >= cfg.span_no_answer_threshold:
            continue
        s = int(s_log.argmax())
        e = s + int(e_log[s : s + cfg.span_max_length].argmax())
        if not (0 <= s <= e < len(c_offsets)):
            continue
        char_s = c_offsets[s][0]
        char_e = c_offsets[e][1]
        span_text = text[char_s:char_e].strip()
        if best is None or no_a < best[0]:
            best = (no_a, span_text, char_s, char_e)
    return best
```

**Step 4: Run → 3 new tests passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/pipeline.py tests/test_cascade_pipeline.py && \
  git commit -m "feat: cascade Stage 2 — span head with lowest-no-answer selection"
```

---

### Task 4.4: Stage 3 (cleaner) prompt + decode

**Files:**
- Modify: `funding_statement_extractor_cascade/cascade/pipeline.py`
- Modify: `tests/test_cascade_pipeline.py`

**Step 1: Append failing tests:**

```python
from tests.cascade_fakes import FakeCleaner, FakeQwenTokenizer

from funding_statement_extractor_cascade.cascade.pipeline import (
    SYSTEM_CLEAN,
    _run_stage3,
)


def _models_with_cleaner(canned: str):
    qwen = FakeQwenTokenizer()
    return CascadeModels(
        mb_tok=FakeMbTokenizer(),
        chunker=FakeChunker(scalars=[]),
        spanner=FakeSpanner(calls=[]),
        qwen_tok=qwen,
        cleaner=FakeCleaner(canned, qwen),
    )


def test_stage3_passes_rough_in_load_bearing_format():
    text = "intro paragraph. " + "ROUGH_FUNDING_SPAN" + ". closing paragraph."
    char_s = text.index("ROUGH_FUNDING_SPAN")
    char_e = char_s + len("ROUGH_FUNDING_SPAN")
    cfg = CascadeConfig()
    models = _models_with_cleaner("Cleaned statement.")
    out = _run_stage3(text, "ROUGH_FUNDING_SPAN", char_s, char_e, models, cfg)
    assert out == "Cleaned statement."
    # Inspect the prompt the cleaner saw
    msgs = models.qwen_tok.captured_messages[-1]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_CLEAN
    assert msgs[1]["role"] == "user"
    assert "<ROUGH>ROUGH_FUNDING_SPAN</ROUGH>" in msgs[1]["content"]


def test_stage3_returns_empty_on_NONE():
    text = "x" * 1000
    models = _models_with_cleaner("NONE")
    out = _run_stage3(text, "rough", 100, 110, models, CascadeConfig())
    assert out == ""


def test_stage3_returns_empty_on_lowercase_none():
    text = "x" * 1000
    models = _models_with_cleaner("none")  # any case maps to empty
    out = _run_stage3(text, "rough", 100, 110, models, CascadeConfig())
    assert out == ""


def test_stage3_strips_decoded_output():
    text = "x" * 100
    models = _models_with_cleaner("  trimmed result  ")
    out = _run_stage3(text, "rough", 10, 20, models, CascadeConfig())
    assert out == "trimmed result"


def test_stage3_context_window_respects_doc_boundaries():
    # rough span sits near the start; left context should clamp at 0, not raise
    text = "rough" + ("y" * 1000)
    models = _models_with_cleaner("ok")
    out = _run_stage3(text, "rough", 0, 5, models, CascadeConfig())
    assert out == "ok"
    user_msg = models.qwen_tok.captured_messages[-1][1]["content"]
    assert user_msg.startswith("<ROUGH>")
```

**Step 2: Run → FAIL.**

**Step 3: Append to `pipeline.py`:**

```python
SYSTEM_CLEAN = (
    "You are a funding statement cleaner. Given a rough extracted funding "
    "statement and its surrounding context from an academic paper, output "
    "the exact funding statement as it should appear in a database. Clean "
    "up LaTeX markers ($^{N}$, \\textsuperscript), hyphenated line breaks, "
    "and abnormal whitespace, but DO NOT paraphrase. If the rough span is "
    "not actually a funding statement, output the single word: NONE"
)


def _run_stage3(text: str, rough_span: str, char_s: int, char_e: int,
                models: CascadeModels, cfg: CascadeConfig) -> str:
    ctx_l = text[max(0, char_s - cfg.cleaner_context_chars): char_s]
    ctx_r = text[char_e: char_e + cfg.cleaner_context_chars]
    user = f"{ctx_l}<ROUGH>{rough_span}</ROUGH>{ctx_r}"
    messages = [
        {"role": "system", "content": SYSTEM_CLEAN},
        {"role": "user", "content": user},
    ]
    prompt_ids = models.qwen_tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(cfg.device)
    out = models.cleaner.generate(
        prompt_ids,
        max_new_tokens=cfg.cleaner_max_new_tokens,
        do_sample=False,
        pad_token_id=models.qwen_tok.eos_token_id,
    )
    decoded = models.qwen_tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()
    if decoded.upper() == "NONE":
        return ""
    return decoded
```

**Step 4: Run → 5 new tests passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/pipeline.py tests/test_cascade_pipeline.py && \
  git commit -m "feat: cascade Stage 3 — Qwen+LoRA cleaner with verbatim SYSTEM_CLEAN"
```

---

### Task 4.5: End-to-end `extract_funding` + skip_cleanup path

**Files:**
- Modify: `funding_statement_extractor_cascade/cascade/pipeline.py`
- Modify: `tests/test_cascade_pipeline.py`

**Step 1: Append failing tests:**

```python
from funding_statement_extractor_cascade.cascade.pipeline import extract_funding


def _all_three(canned: str, chunker_scalars, spanner_calls):
    qwen = FakeQwenTokenizer()
    return CascadeModels(
        mb_tok=FakeMbTokenizer(),
        chunker=FakeChunker(scalars=chunker_scalars),
        spanner=FakeSpanner(calls=spanner_calls),
        qwen_tok=qwen,
        cleaner=FakeCleaner(canned, qwen),
    )


def test_e2e_returns_empty_when_no_chunk_survives_stage1():
    text = "x" * 100
    models = _all_three("ignored", chunker_scalars=[-5.0], spanner_calls=[])
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, CascadeConfig())
    assert r.funding_statement == ""
    assert r.rough_span == ""
    assert r.error is None


def test_e2e_returns_empty_when_stage2_no_answer_high():
    text = "x" * 100
    models = _all_three("ignored",
                        chunker_scalars=[5.0],
                        spanner_calls=[([0.0] * 100, [0.0] * 100, 5.0)])
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, CascadeConfig())
    assert r.funding_statement == ""
    assert r.rough_span == ""


def test_e2e_full_path_returns_cleaned():
    text = "intro " + ("z" * 40) + " end"   # 50 chars
    s_log = [0.0] * 50; s_log[6] = 5.0       # start = 6
    e_log = [0.0] * 50; e_log[10] = 5.0      # end relative = 10 → abs end = 16
    models = _all_three("cleaned funding!",
                        chunker_scalars=[5.0],
                        spanner_calls=[(s_log, e_log, -5.0)])
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, CascadeConfig())
    assert r.funding_statement == "cleaned funding!"
    # rough_span sits inside the z-block
    assert r.rough_span and r.rough_span.startswith("z")
    assert 0.0 <= r.chunk_prob <= 1.0
    assert 0.0 <= r.no_answer_prob <= 1.0
    assert r.latency_ms >= 0.0


def test_e2e_skip_cleanup_returns_rough_span():
    text = "intro " + ("z" * 40) + " end"
    s_log = [0.0] * 50; s_log[6] = 5.0
    e_log = [0.0] * 50; e_log[10] = 5.0
    models = _all_three("would-be-cleaned",
                        chunker_scalars=[5.0],
                        spanner_calls=[(s_log, e_log, -5.0)])
    cfg = CascadeConfig(skip_cleanup=True)
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, cfg)
    # funding_statement == rough_span when skip_cleanup=True
    assert r.funding_statement == r.rough_span
    assert r.funding_statement.startswith("z")
    # The cleaner was never called
    assert models.qwen_tok.captured_messages == []


def test_e2e_returns_error_on_exception():
    text = "x" * 100

    class BrokenChunker:
        def __call__(self, *a, **kw):
            raise RuntimeError("boom")

    models = CascadeModels(mb_tok=FakeMbTokenizer(), chunker=BrokenChunker(),
                            spanner=None, qwen_tok=None, cleaner=None)
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, CascadeConfig())
    assert r.funding_statement == ""
    assert r.error is not None
    assert "RuntimeError" in r.error
```

**Step 2: Run → FAIL.**

**Step 3: Append the public `extract_funding` to `pipeline.py`:**

```python
import time

from funding_statement_extractor_cascade.cascade.types import CascadeResult


@torch.no_grad()
def extract_funding(text: str, models: CascadeModels, cfg: CascadeConfig) -> CascadeResult:
    t0 = time.perf_counter()
    try:
        ids, offsets, surviving = _run_stage1(text, models, cfg)
        if not surviving:
            return CascadeResult(
                funding_statement="", rough_span="",
                chunk_prob=None, no_answer_prob=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        chunk_prob = max(p for p, _, _ in surviving)
        best = _run_stage2(text, surviving, models, cfg, ids=ids, offsets=offsets)
        if best is None:
            return CascadeResult(
                funding_statement="", rough_span="",
                chunk_prob=chunk_prob, no_answer_prob=None,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        no_a, rough_span, char_s, char_e = best

        if cfg.skip_cleanup:
            funding_statement = rough_span
        else:
            funding_statement = _run_stage3(text, rough_span, char_s, char_e, models, cfg)

        return CascadeResult(
            funding_statement=funding_statement,
            rough_span=rough_span,
            chunk_prob=chunk_prob,
            no_answer_prob=no_a,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
    except Exception as exc:
        return CascadeResult(
            funding_statement="", rough_span="",
            chunk_prob=None, no_answer_prob=None,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )
```

**Step 4: Run → 5 new tests passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/pipeline.py tests/test_cascade_pipeline.py && \
  git commit -m "feat: cascade extract_funding orchestrator with skip_cleanup + error capture"
```

---

### Task 4.6: Real `load_models` factory

**Files:**
- Modify: `funding_statement_extractor_cascade/cascade/loader.py`
- Create: `tests/test_loader.py`

> **Note:** `load_models` actually hits the network and downloads weights. We test only its *control flow* — that it honors `skip_cleanup` by returning a `CascadeModels` with `cleaner=None`. The full instantiation path is exercised by the HF Job worker and the benchmark scripts on real hardware.

**Step 1: Write failing test:**

```python
from unittest.mock import MagicMock, patch

from funding_statement_extractor_cascade.cascade.loader import CascadeModels, load_models
from funding_statement_extractor_cascade.cascade.types import CascadeConfig


@patch("funding_statement_extractor_cascade.cascade.loader.PeftModel")
@patch("funding_statement_extractor_cascade.cascade.loader.AutoModelForCausalLM")
@patch("funding_statement_extractor_cascade.cascade.loader.AutoTokenizer")
@patch("funding_statement_extractor_cascade.cascade.loader.SpanHead")
@patch("funding_statement_extractor_cascade.cascade.loader.ChunkClassifier")
@patch("funding_statement_extractor_cascade.cascade.loader.hf_hub_download")
@patch("funding_statement_extractor_cascade.cascade.loader.torch")
def test_load_models_full_cascade(torch_m, hub, chunk_cls, span_cls,
                                  auto_tok, auto_lm, peft):
    torch_m.load.return_value = {}
    chunk_cls.return_value.to.return_value.eval.return_value = MagicMock(name="chunker")
    span_cls.return_value.to.return_value.eval.return_value = MagicMock(name="spanner")
    auto_lm.from_pretrained.return_value = MagicMock(name="qwen_base")
    peft.from_pretrained.return_value.eval.return_value = MagicMock(name="peft_cleaner")
    auto_tok.from_pretrained.side_effect = [MagicMock(name="mb_tok"),
                                            MagicMock(name="qwen_tok")]

    cfg = CascadeConfig()
    models = load_models(cfg)
    assert isinstance(models, CascadeModels)
    assert models.mb_tok is not None
    assert models.chunker is not None
    assert models.spanner is not None
    assert models.qwen_tok is not None
    assert models.cleaner is not None


@patch("funding_statement_extractor_cascade.cascade.loader.PeftModel")
@patch("funding_statement_extractor_cascade.cascade.loader.AutoModelForCausalLM")
@patch("funding_statement_extractor_cascade.cascade.loader.AutoTokenizer")
@patch("funding_statement_extractor_cascade.cascade.loader.SpanHead")
@patch("funding_statement_extractor_cascade.cascade.loader.ChunkClassifier")
@patch("funding_statement_extractor_cascade.cascade.loader.hf_hub_download")
@patch("funding_statement_extractor_cascade.cascade.loader.torch")
def test_load_models_skip_cleanup_leaves_cleaner_none(torch_m, hub, chunk_cls,
                                                       span_cls, auto_tok,
                                                       auto_lm, peft):
    torch_m.load.return_value = {}
    chunk_cls.return_value.to.return_value.eval.return_value = MagicMock()
    span_cls.return_value.to.return_value.eval.return_value = MagicMock()
    auto_tok.from_pretrained.return_value = MagicMock()

    cfg = CascadeConfig(skip_cleanup=True)
    models = load_models(cfg)
    assert models.cleaner is None
    assert models.qwen_tok is None
    auto_lm.from_pretrained.assert_not_called()
    peft.from_pretrained.assert_not_called()
```

**Step 2: Run → FAIL.**

**Step 3: Replace `loader.py` with the real implementation:**

```python
"""Model loader for the cascade. Wired so tests can monkeypatch heavyweight imports."""
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from funding_statement_extractor_cascade.cascade.types import CascadeConfig
from funding_statement_extractor_cascade.models.chunk_classifier.modeling import ChunkClassifier
from funding_statement_extractor_cascade.models.span_head.modeling import SpanHead


_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class CascadeModels:
    mb_tok: object
    chunker: object
    spanner: object
    qwen_tok: Optional[object]
    cleaner: Optional[object]


def _resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return torch.bfloat16
    return _DTYPE_MAP[dtype_str]


def load_models(cfg: CascadeConfig) -> CascadeModels:
    mb_tok = AutoTokenizer.from_pretrained(cfg.mb_base)

    chunker = ChunkClassifier(cfg.mb_base).to(cfg.device).eval()
    chunker.load_state_dict(torch.load(
        hf_hub_download(cfg.chunk_repo, "pytorch_model.bin"),
        map_location=cfg.device, weights_only=True,
    ))

    spanner = SpanHead(cfg.mb_base).to(cfg.device).eval()
    spanner.load_state_dict(torch.load(
        hf_hub_download(cfg.span_repo, "pytorch_model.bin"),
        map_location=cfg.device, weights_only=True,
    ))

    if cfg.skip_cleanup:
        return CascadeModels(mb_tok=mb_tok, chunker=chunker, spanner=spanner,
                             qwen_tok=None, cleaner=None)

    qwen_tok = AutoTokenizer.from_pretrained(cfg.qwen_base)
    cleaner_base = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_base, dtype=_resolve_dtype(cfg.dtype), device_map=cfg.device,
    )
    cleaner = PeftModel.from_pretrained(cleaner_base, cfg.clean_repo).eval()

    return CascadeModels(mb_tok=mb_tok, chunker=chunker, spanner=spanner,
                          qwen_tok=qwen_tok, cleaner=cleaner)
```

**Step 4: Run → 2 passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/loader.py tests/test_loader.py && \
  git commit -m "feat: load_models with skip_cleanup short-circuit"
```

---

## Phase 5 — Batch iterator + I/O

### Task 5.1: `cascade/batch.py` — streaming `extract_funding_batch`

**Files:**
- Create: `funding_statement_extractor_cascade/cascade/batch.py`
- Create: `tests/test_cascade_batch.py`

**Mirrors:** v1 `statements/batch_extraction.py` shape (DocPayload → generator of BatchResult). Diverges: no multiprocessing pipeline — the cascade is sequential per doc because each doc's GPU passes are too small to benefit from inter-doc parallelism, and the cleaner is generation-bound.

**Step 1: Write failing tests:**

```python
from unittest.mock import patch

from funding_statement_extractor_cascade.cascade.batch import extract_funding_batch
from funding_statement_extractor_cascade.cascade.types import (
    CascadeConfig,
    CascadeResult,
    DocPayload,
)


def _fake_extractor(call_log):
    def _extract(text, models, cfg):
        call_log.append(text)
        return CascadeResult(funding_statement=f"r:{text}", rough_span="x",
                              chunk_prob=0.9, no_answer_prob=0.1, latency_ms=1.0)
    return _extract


def test_batch_yields_one_result_per_doc():
    calls = []
    docs = [DocPayload(doc_id="a", text="x"),
            DocPayload(doc_id="b", text="y")]
    with patch("funding_statement_extractor_cascade.cascade.batch.extract_funding",
               side_effect=_fake_extractor(calls)):
        results = list(extract_funding_batch(documents=docs, models=None, cfg=CascadeConfig()))
    assert [r.doc_id for r in results] == ["a", "b"]
    assert [r.result.funding_statement for r in results] == ["r:x", "r:y"]


def test_batch_passes_through_metadata():
    docs = [DocPayload(doc_id="a", text="x",
                       metadata={"input_file": "f.parquet", "row_idx": 7})]
    with patch("funding_statement_extractor_cascade.cascade.batch.extract_funding",
               side_effect=_fake_extractor([])):
        result = list(extract_funding_batch(documents=docs, models=None, cfg=CascadeConfig()))[0]
    assert result.metadata == {"input_file": "f.parquet", "row_idx": 7}


def test_batch_isolates_per_doc_exception():
    def _broken(text, models, cfg):
        if text == "boom":
            raise RuntimeError("boom!")
        return CascadeResult(funding_statement="ok", rough_span="ok",
                              chunk_prob=0.9, no_answer_prob=0.1, latency_ms=1.0)

    docs = [
        DocPayload(doc_id="d1", text="ok"),
        DocPayload(doc_id="d2", text="boom"),
        DocPayload(doc_id="d3", text="ok"),
    ]
    with patch("funding_statement_extractor_cascade.cascade.batch.extract_funding",
               side_effect=_broken):
        results = list(extract_funding_batch(documents=docs, models=None, cfg=CascadeConfig()))
    assert [r.doc_id for r in results] == ["d1", "d2", "d3"]
    assert results[1].result.funding_statement == ""
    assert results[1].result.error is not None
    assert "RuntimeError" in results[1].result.error
```

**Step 2: Run → FAIL.**

**Step 3: Write `cascade/batch.py`:**

```python
import time
from typing import Iterable, Iterator

from funding_statement_extractor_cascade.cascade.loader import CascadeModels
from funding_statement_extractor_cascade.cascade.pipeline import extract_funding
from funding_statement_extractor_cascade.cascade.types import (
    BatchResult,
    CascadeConfig,
    CascadeResult,
    DocPayload,
)


def extract_funding_batch(
    documents: Iterable[DocPayload],
    *,
    models: CascadeModels,
    cfg: CascadeConfig,
) -> Iterator[BatchResult]:
    """Iterate docs sequentially through the cascade, yielding one BatchResult each.

    Per-doc exceptions are captured as `CascadeResult.error` so the iterator
    never terminates mid-stream.
    """
    for doc in documents:
        enqueue_ts = time.monotonic()
        try:
            result = extract_funding(doc.text, models, cfg)
        except Exception as exc:
            result = CascadeResult(
                funding_statement="", rough_span="",
                chunk_prob=None, no_answer_prob=None,
                latency_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        yield BatchResult(
            doc_id=doc.doc_id,
            result=result,
            enqueue_ts=enqueue_ts,
            yield_ts=time.monotonic(),
            metadata=doc.metadata,
        )
```

**Step 4: Run → 3 passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/batch.py tests/test_cascade_batch.py && \
  git commit -m "feat: streaming extract_funding_batch with per-doc error isolation"
```

---

### Task 5.2: `io/loaders.py` — markdown / dir / parquet input sources

**Files:**
- Create: `funding_statement_extractor_cascade/io/loaders.py`
- Create: `tests/test_input_sources.py`

**Mirrors:** v1 `io/loaders.py` almost line-for-line. Diverges: parquet fallback text columns prepend `vlm_markdown` ahead of `markdown` (cascade was trained on VLM-converted text); id fallback prepends `arxiv_id` ahead of `source_id`.

**Step 1: Write failing test** — copy the v1 test pattern from `funding-metadata-enrichment/funding_statement_extractor/tests/test_input_sources.py` lines 1–158, with these adjustments:
- import path `funding_statement_extractor_cascade.io.loaders`
- in `test_stream_parquet_documents_infers_text_column`, change `fallback_text_columns=['vlm_markdown', 'markdown']` and assert that auto-selected column is `vlm_markdown` when both are present
- add one new test:

```python
def test_stream_parquet_prefers_vlm_markdown_over_markdown(tmp_path, monkeypatch):
    rows = [{"vlm_markdown": "VLM text",
              "markdown": "plain text",
              "arxiv_id": "2401.00001"}]
    batches = [FakeBatch(rows, ["vlm_markdown", "markdown", "arxiv_id"])]
    dataset = FakeDataset(batches)
    install_fake_pyarrow(monkeypatch, dataset)
    p = tmp_path / "x.parquet"
    p.write_text("placeholder")
    it, _ = stream_parquet_documents(
        p, text_column=None, id_column=None, batch_size=1,
        fallback_text_columns=["vlm_markdown", "markdown"],
        fallback_id_columns=["arxiv_id"],
    )
    docs = list(it)
    assert docs[0].content == "VLM text"
    assert docs[0].document_id == "2401.00001"
```

**Step 2: Run → FAIL.**

**Step 3: Implement `io/loaders.py`** — copy v1 `io/loaders.py` verbatim (lines 1–204 from `funding-metadata-enrichment/funding_statement_extractor/funding_statement_extractor/io/loaders.py`), then change two import paths:
- `from funding_statement_extractor.exceptions import DocumentLoadError` → `from funding_statement_extractor_cascade.exceptions import DocumentLoadError`

No other changes — the resolver logic (`_resolve_column`) already handles ordered fallback lists, so prepending `vlm_markdown` happens at the *caller* (cli.py — Task 6.1) not the loader.

**Step 4: Run → all loader tests pass.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/io/loaders.py tests/test_input_sources.py && \
  git commit -m "feat: markdown/dir/parquet loaders (cascade prefers vlm_markdown)"
```

---

### Task 5.3: `io/checkpointing.py` — verbatim copy + tests

**Files:**
- Create: `funding_statement_extractor_cascade/io/checkpointing.py`
- Create: `tests/test_checkpointing.py`

**Mirrors:** v1 `io/checkpointing.py` verbatim. The checkpoint format does not depend on the engine.

**Step 1: Write failing tests:**

```python
from funding_statement_extractor_cascade.io.checkpointing import CheckpointRepository, get_file_hash


def test_hash_is_deterministic():
    assert get_file_hash("a.md") == get_file_hash("a.md")
    assert get_file_hash("a.md") != get_file_hash("b.md")


def test_repo_records_and_round_trips(tmp_path):
    p = tmp_path / "ck.json"
    repo = CheckpointRepository(p)
    repo.record("h1", {"path": "a.md", "status": "ok"})
    repo.save()

    fresh = CheckpointRepository(p)
    data = fresh.load(resume=True)
    assert data["processed_files"]["h1"]["status"] == "ok"
    assert fresh.is_processed("h1")
    assert not fresh.is_processed("nope")


def test_save_is_atomic(tmp_path):
    p = tmp_path / "ck.json"
    repo = CheckpointRepository(p)
    repo.save()
    assert not (tmp_path / "ck.json.tmp").exists()
    assert p.exists()
```

**Step 2: Run → FAIL.**

**Step 3: Copy v1 `io/checkpointing.py` verbatim** (lines 1–52 from `funding-metadata-enrichment/funding_statement_extractor/funding_statement_extractor/io/checkpointing.py`), then change one import:
- `from funding_statement_extractor.exceptions import CheckpointError` → `from funding_statement_extractor_cascade.exceptions import CheckpointError`

**Step 4: Run → 3 passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/io/checkpointing.py tests/test_checkpointing.py && \
  git commit -m "feat: CheckpointRepository (verbatim port from v1)"
```

---

## Phase 6 — CLI

> **Strategy:** v1 has a heavier CLI (multiprocessing pool, two engines, normalization). The cascade is simpler — one engine, sequential per-doc — so the CLI is smaller. We mirror v1's two-file split (`cli/main.py` = top-level argparse + logging; `cascade/cli.py` = arg definitions + run loop) so the structure is recognizable.

### Task 6.1: `cascade/cli.py` — `add_arguments` + `run`

**Files:**
- Create: `funding_statement_extractor_cascade/cascade/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write failing tests:**

```python
import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from funding_statement_extractor_cascade.cascade.cli import (
    add_arguments,
    build_config,
    run,
)
from funding_statement_extractor_cascade.cascade.types import (
    BatchResult,
    CascadeResult,
)


def _args(input_path):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(["-i", str(input_path)])


def test_required_input(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("hi")
    args = _args(md)
    assert args.input == str(md)
    assert args.output == "funding_results.json"
    assert args.skip_cleanup is False
    assert args.chunk_keep_threshold == 0.4
    assert args.dtype == "bf16"


def test_build_config_propagates_cascade_overrides(tmp_path):
    md = tmp_path / "x.md"; md.write_text("hi")
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", str(md), "--skip-cleanup",
                              "--chunk-keep-threshold", "0.7",
                              "--dtype", "fp16"])
    cfg = build_config(args)
    assert cfg.cascade.skip_cleanup is True
    assert cfg.cascade.chunk_keep_threshold == 0.7
    assert cfg.cascade.dtype == "fp16"


def test_run_processes_single_markdown_file(tmp_path):
    md = tmp_path / "x.md"; md.write_text("paper text")
    out = tmp_path / "out.json"

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", str(md), "-o", str(out)])

    fake_result = CascadeResult(funding_statement="Funded by NSF.",
                                 rough_span="Funded by NSF.",
                                 chunk_prob=0.9, no_answer_prob=0.05,
                                 latency_ms=10.0)

    def _fake_batch(documents, *, models, cfg):
        for d in documents:
            yield BatchResult(doc_id=d.doc_id, result=fake_result,
                              enqueue_ts=0.0, yield_ts=0.01, metadata=d.metadata)

    with patch("funding_statement_extractor_cascade.cascade.cli.load_models",
               return_value=SimpleNamespace()), \
         patch("funding_statement_extractor_cascade.cascade.cli.extract_funding_batch",
               side_effect=_fake_batch):
        run(args)

    import json
    data = json.loads(out.read_text())
    assert data["results"]["x.md"]["funding_statement"] == "Funded by NSF."
    assert data["summary"]["total_files"] == 1
    assert data["summary"]["files_with_funding"] == 1
```

**Step 2: Run → FAIL.**

**Step 3: Write `cascade/cli.py`:**

```python
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

from funding_statement_extractor_cascade.cascade.batch import extract_funding_batch
from funding_statement_extractor_cascade.cascade.loader import load_models
from funding_statement_extractor_cascade.cascade.types import DocPayload
from funding_statement_extractor_cascade.config.settings import (
    ApplicationConfig,
    CascadeRuntimeSettings,
    InputSettings,
    OutputSettings,
    ProcessingSettings,
    RuntimeSettings,
)
from funding_statement_extractor_cascade.exceptions import ConfigurationError
from funding_statement_extractor_cascade.io.checkpointing import (
    CheckpointRepository,
    get_file_hash,
)
from funding_statement_extractor_cascade.io.loaders import (
    DocumentPayload,
    build_markdown_documents,
    determine_input_format,
    stream_parquet_documents,
)

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", required=True,
                        help="Input markdown file, directory, or parquet dataset")
    parser.add_argument("-o", "--output", default="funding_results.json",
                        help="Output JSON file (default: funding_results.json)")
    parser.add_argument("--input-format", choices=["markdown", "parquet"])
    parser.add_argument("--parquet-text-column",
                        help='Column with markdown text (default: auto-detect, preferring "vlm_markdown")')
    parser.add_argument("--parquet-id-column",
                        help='Column with a unique id (default: auto-detect, preferring "arxiv_id")')
    parser.add_argument("--parquet-batch-size", type=int, default=64)

    # Cascade engine knobs
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Skip Stage 3 (Qwen cleaner); return raw stage-2 span verbatim.")
    parser.add_argument("--chunk-keep-threshold", type=float, default=0.4)
    parser.add_argument("--chunk-top-k", type=int, default=2)
    parser.add_argument("--span-no-answer-threshold", type=float, default=0.5)
    parser.add_argument("--span-max-length", type=int, default=300)
    parser.add_argument("--cleaner-context-chars", type=int, default=400)
    parser.add_argument("--cleaner-max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")

    # Runtime
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Documents between checkpoint flushes (default: 50)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--checkpoint-file")


def build_config(args: argparse.Namespace) -> ApplicationConfig:
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = (Path(args.checkpoint_file) if args.checkpoint_file
                       else Path(f"{args.output}.checkpoint"))
    return ApplicationConfig(
        input=InputSettings(
            path=input_path,
            input_format=args.input_format,
            parquet_text_column=args.parquet_text_column,
            parquet_id_column=args.parquet_id_column,
            parquet_batch_size=args.parquet_batch_size,
        ),
        output=OutputSettings(output_path=output_path, checkpoint_path=checkpoint_path),
        cascade=CascadeRuntimeSettings(
            chunk_keep_threshold=args.chunk_keep_threshold,
            chunk_top_k=args.chunk_top_k,
            span_no_answer_threshold=args.span_no_answer_threshold,
            span_max_length=args.span_max_length,
            cleaner_context_chars=args.cleaner_context_chars,
            cleaner_max_new_tokens=args.cleaner_max_new_tokens,
            skip_cleanup=args.skip_cleanup,
            device=args.device,
            dtype=args.dtype,
        ),
        processing=ProcessingSettings(),
        runtime=RuntimeSettings(batch_size=args.batch_size,
                                resume=args.resume,
                                force=args.force,
                                retry_failed=args.retry_failed,
                                verbose=args.verbose),
    )


def _docs_iter(cfg: ApplicationConfig, checkpoint_repo: CheckpointRepository
                ) -> Iterator[Tuple[DocumentPayload, str]]:
    fmt = determine_input_format(cfg.input.path, cfg.input.input_format)
    cfg.input.input_format = fmt

    processed = checkpoint_repo.data.get("processed_files", {})
    if cfg.runtime.force:
        processed = {}

    if fmt == "parquet":
        text_fallbacks = ([] if cfg.input.parquet_text_column
                          else ["vlm_markdown", "markdown", "content", "text", "body"])
        id_fallbacks = ([] if cfg.input.parquet_id_column
                        else ["arxiv_id", "source_id", "doc_id", "document_id", "id", "file_name"])
        stream, _ = stream_parquet_documents(
            input_path=cfg.input.path,
            text_column=cfg.input.parquet_text_column,
            id_column=cfg.input.parquet_id_column,
            batch_size=cfg.input.parquet_batch_size,
            fallback_text_columns=text_fallbacks,
            fallback_id_columns=id_fallbacks,
        )
        for doc in stream:
            h = get_file_hash(doc.checkpoint_key)
            if cfg.runtime.force or h not in processed:
                yield doc, h
        return

    for doc in build_markdown_documents(cfg.input.path):
        h = get_file_hash(doc.checkpoint_key)
        if cfg.runtime.force or h not in processed:
            yield doc, h


def _build_payloads(docs_iter, checkpoint_repo, verbose):
    for document, doc_hash in docs_iter:
        try:
            text = document.load_text()
        except Exception as exc:
            if verbose:
                print(f"  warn: failed to read {document.document_id}: {exc}")
            checkpoint_repo.record(doc_hash, {
                "path": document.file_path or document.document_id,
                "document_id": document.document_id,
                "processed_at": datetime.now().isoformat(),
                "status": "failed",
                "error": f"load_text: {type(exc).__name__}: {exc}",
            })
            continue
        yield DocPayload(
            doc_id=document.document_id,
            text=text,
            metadata={"doc_hash": doc_hash,
                      "path": document.file_path or document.document_id},
        )


def run(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    try:
        cfg.validate()
    except ConfigurationError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    from funding_statement_extractor_cascade.cascade.types import CascadeConfig
    cascade_cfg = CascadeConfig(
        chunk_keep_threshold=cfg.cascade.chunk_keep_threshold,
        chunk_top_k=cfg.cascade.chunk_top_k,
        span_no_answer_threshold=cfg.cascade.span_no_answer_threshold,
        span_max_length=cfg.cascade.span_max_length,
        cleaner_context_chars=cfg.cascade.cleaner_context_chars,
        cleaner_max_new_tokens=cfg.cascade.cleaner_max_new_tokens,
        skip_cleanup=cfg.cascade.skip_cleanup,
        device=cfg.cascade.device,
        dtype=cfg.cascade.dtype,
    )

    models = load_models(cascade_cfg)

    checkpoint_repo = CheckpointRepository(cfg.output.checkpoint_path)
    if cfg.runtime.resume and not cfg.runtime.force:
        checkpoint_repo.load(resume=True)

    results: dict = {}
    payloads = _build_payloads(_docs_iter(cfg, checkpoint_repo), checkpoint_repo,
                                cfg.runtime.verbose)
    processed = 0
    for batch in extract_funding_batch(documents=payloads, models=models, cfg=cascade_cfg):
        results[batch.doc_id] = {
            "funding_statement": batch.result.funding_statement,
            "rough_span": batch.result.rough_span,
            "chunk_prob": batch.result.chunk_prob,
            "no_answer_prob": batch.result.no_answer_prob,
            "latency_ms": batch.result.latency_ms,
            "error": batch.result.error,
        }
        h = (batch.metadata or {}).get("doc_hash") or get_file_hash(batch.doc_id)
        checkpoint_repo.record(h, {
            "path": (batch.metadata or {}).get("path", batch.doc_id),
            "document_id": batch.doc_id,
            "processed_at": datetime.now().isoformat(),
            "found_funding": bool(batch.result.funding_statement),
            "status": "failed" if batch.result.error else "ok",
            "error": batch.result.error,
        })
        processed += 1
        if processed % max(1, cfg.runtime.batch_size) == 0:
            _flush(cfg, results, checkpoint_repo)

    _flush(cfg, results, checkpoint_repo)


def _flush(cfg: ApplicationConfig, results: dict, repo: CheckpointRepository) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "input_path": str(cfg.input.path),
            "input_format": cfg.input.input_format,
            "skip_cleanup": cfg.cascade.skip_cleanup,
            "dtype": cfg.cascade.dtype,
        },
        "results": results,
        "summary": {
            "total_files": len(results),
            "files_with_funding": sum(1 for r in results.values() if r["funding_statement"]),
        },
    }
    cfg.output.output_path.write_text(json.dumps(payload, indent=2))
    repo.save()
```

**Step 4: Run → 3 passed.**

**Step 5: Commit**

```bash
git add funding_statement_extractor_cascade/cascade/cli.py tests/test_cli.py && \
  git commit -m "feat: cascade CLI with skip_cleanup + dtype + checkpoint flush"
```

---

### Task 6.2: `cli/main.py` + `__main__.py` — top-level entry

**Files:**
- Create: `funding_statement_extractor_cascade/cli/main.py`
- Create: `funding_statement_extractor_cascade/__main__.py`

**Mirrors:** v1 `cli/main.py` exactly.

**Step 1: Write `cli/main.py`:**

```python
import argparse
import logging

from funding_statement_extractor_cascade.cascade import cli as cascade_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="funding-statement-extractor-cascade",
        description="Extract cleaned funding statements from VLM-converted arXiv markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s -i paper.md -o results.json
  %(prog)s -i docs/ --skip-cleanup --dtype bf16
  %(prog)s -i bench.parquet --parquet-text-column vlm_markdown --parquet-id-column arxiv_id
""",
    )
    cascade_cli.add_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    cascade_cli.run(args)


if __name__ == "__main__":
    main()
```

**Step 2: Write `__main__.py`:**

```python
from funding_statement_extractor_cascade.cli.main import main

main()
```

**Step 3: Smoke test:**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  python -m funding_statement_extractor_cascade --help
```

Expected: argparse help text with cascade flags listed. (Don't call without `-i` — it'll error out, which is the correct behavior.)

**Step 4: Commit**

```bash
git add funding_statement_extractor_cascade/cli/main.py funding_statement_extractor_cascade/__main__.py && \
  git commit -m "feat: top-level CLI entry and python -m support"
```

---

## Phase 7 — HF Job worker

> **Reference:** v1 `hf_jobs/jobs/funding_statement_extractor/extract_funding_job.py` (full 466 lines) and its `tests/test_extract_funding_job.py` (lines 1–415). Almost every helper here has a direct v1 analog — `parse_args`, `make_output_row`, `stage_parquet_locally`, `_parse_retry_hint_seconds`, `preflight_commit_check`, `commit_staged_batch`, `emit_done_lines`, `main`. The shape and error handling are identical; only `OUTPUT_SCHEMA`, the call into the extractor, and the model bootstrap differ.

### Task 7.1: HF Job scaffolding directory

**Files:**
- Create empty `__init__.py` in:
  - `hf_jobs/__init__.py`
  - `hf_jobs/jobs/__init__.py`
  - `hf_jobs/jobs/funding_statement_extractor_cascade/__init__.py`
  - `hf_jobs/jobs/funding_statement_extractor_cascade/tests/__init__.py`
  - `hf_jobs/ops/__init__.py`
  - `hf_jobs/ops/funding_statement_extractor_cascade/__init__.py`
  - `hf_jobs/ops/funding_statement_extractor_cascade/tests/__init__.py`
- Create: `hf_jobs/jobs/funding_statement_extractor_cascade/tests/conftest.py`
- Create: `hf_jobs/ops/funding_statement_extractor_cascade/tests/conftest.py`

**Step 1: Create the directories**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  mkdir -p hf_jobs/{jobs,ops}/funding_statement_extractor_cascade/tests && \
  for d in \
    hf_jobs \
    hf_jobs/jobs \
    hf_jobs/jobs/funding_statement_extractor_cascade \
    hf_jobs/jobs/funding_statement_extractor_cascade/tests \
    hf_jobs/ops \
    hf_jobs/ops/funding_statement_extractor_cascade \
    hf_jobs/ops/funding_statement_extractor_cascade/tests ; do touch "$d/__init__.py" ; done
```

**Step 2: Write both `conftest.py` files** with the same content — sys.path injection so tests can `import extract_funding_cascade_job` (the worker is a flat script, not a package):

```python
import sys
from pathlib import Path

JOB_DIR = Path(__file__).resolve().parents[1]
if str(JOB_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_DIR))
```

**Step 3: Commit**

```bash
git add hf_jobs && git commit -m "chore: hf_jobs directory skeleton"
```

---

### Task 7.2: PEP 723 worker — `parse_args` + `make_output_row` + `stage_parquet_locally`

**Files:**
- Create: `hf_jobs/jobs/funding_statement_extractor_cascade/extract_funding_cascade_job.py`
- Create: `hf_jobs/jobs/funding_statement_extractor_cascade/tests/test_extract_funding_cascade_job.py`

**Step 1: Write failing tests:**

```python
import pyarrow.parquet as pq

from extract_funding_cascade_job import (
    OUTPUT_SCHEMA,
    make_output_row,
    parse_args,
    stage_parquet_locally,
)


def test_parse_args_minimum_defaults():
    args = parse_args([
        "--input-repo", "cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test",
        "--input-files", "test-00000-of-00001.parquet",
        "--output-repo", "user/cascade-bench-out",
        "--job-tag", "cascade-bench-001",
    ])
    assert args.text_column == "vlm_markdown"
    assert args.id_column == "arxiv_id"
    assert args.dtype == "bf16"
    assert args.skip_cleanup is False
    assert args.commit_batch_size == 25
    assert args.skip_preflight is False


def test_parse_args_input_files_strips_whitespace():
    args = parse_args([
        "--input-repo", "x", "--input-files", " a.parquet , b.parquet ",
        "--output-repo", "y", "--job-tag", "z",
    ])
    assert args.input_files == ["a.parquet", "b.parquet"]


def test_parse_args_skip_cleanup_flag():
    args = parse_args([
        "--input-repo", "x", "--input-files", "a.parquet",
        "--output-repo", "y", "--job-tag", "z", "--skip-cleanup",
    ])
    assert args.skip_cleanup is True


def test_make_output_row_full():
    from types import SimpleNamespace
    cascade_result = SimpleNamespace(
        funding_statement="This work was funded by NSF.",
        rough_span="funded by NSF",
        chunk_prob=0.93, no_answer_prob=0.04, latency_ms=421.5,
        error=None,
    )
    batch = SimpleNamespace(
        doc_id="2401.00001", result=cascade_result,
        enqueue_ts=100.0, yield_ts=100.42,
        metadata={"arxiv_id": "2401.00001",
                  "input_file": "test-00000-of-00001.parquet",
                  "row_idx": 12, "shard_id": "shard-0", "text_length": 8421},
    )
    row = make_output_row(batch, skip_cleanup=False)
    assert row["arxiv_id"] == "2401.00001"
    assert row["funding_statement"] == "This work was funded by NSF."
    assert row["rough_span"] == "funded by NSF"
    assert row["chunk_prob"] == 0.93
    assert row["no_answer_prob"] == 0.04
    assert row["text_length"] == 8421
    assert row["latency_ms"] == 421.5
    assert row["skip_cleanup"] is False
    assert row["error"] is None


def test_make_output_row_empty_when_no_funding():
    from types import SimpleNamespace
    batch = SimpleNamespace(
        doc_id="2401.00002",
        result=SimpleNamespace(funding_statement="", rough_span="",
                               chunk_prob=None, no_answer_prob=None,
                               latency_ms=10.0, error=None),
        enqueue_ts=0.0, yield_ts=0.01,
        metadata={"arxiv_id": "2401.00002", "input_file": "x.parquet",
                  "row_idx": 0, "text_length": 100},
    )
    row = make_output_row(batch, skip_cleanup=False)
    assert row["funding_statement"] == ""
    assert row["chunk_prob"] is None
    assert row["no_answer_prob"] is None
    assert row["error"] is None


def test_stage_parquet_locally_empty_uses_explicit_schema(tmp_path):
    local = stage_parquet_locally(
        [], path_in_repo="predictions/empty.parquet", staging_dir=tmp_path,
    )
    assert local.exists()
    tbl = pq.read_table(local)
    assert tbl.num_rows == 0
    expected = {"doc_id", "arxiv_id", "shard_id", "input_file", "row_idx",
                "funding_statement", "rough_span", "chunk_prob",
                "no_answer_prob", "text_length", "latency_ms", "skip_cleanup",
                "error"}
    assert set(tbl.column_names) == expected


def test_stage_parquet_locally_writes_rows(tmp_path):
    rows = [{
        "doc_id": "x", "arxiv_id": "x", "shard_id": "s",
        "input_file": "f.parquet", "row_idx": 0,
        "funding_statement": "Funded.", "rough_span": "Funded",
        "chunk_prob": 0.9, "no_answer_prob": 0.1,
        "text_length": 100, "latency_ms": 1.0, "skip_cleanup": False, "error": None,
    }]
    local = stage_parquet_locally(
        rows, path_in_repo="predictions/f.parquet", staging_dir=tmp_path,
    )
    tbl = pq.read_table(local)
    assert tbl.num_rows == 1
    assert tbl.column("funding_statement").to_pylist() == ["Funded."]
```

**Step 2: Run → FAIL (script does not exist).**

**Step 3: Write `extract_funding_cascade_job.py`** — PEP 723 header + the helpers. (We grow `main` later in Task 7.4.)

```python
#!/usr/bin/env python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "funding-statement-extractor-cascade @ git+https://github.com/USER/parth_pipeline.git@main",
#     "datasets>=3.0.0",
#     "huggingface-hub>=0.26.0",
#     "transformers>=4.49,<6",
#     "peft>=0.14.0",
#     "accelerate>=1.0",
#     "torch>=2.4.0,<2.8.0",
#     "rapidfuzz>=3.0",
#     "pyarrow>=12.0.0",
# ]
# ///
"""HF Job worker: run the funding-statement cascade over input parquets and push results.

Designed to run on `a100-large` via `hf jobs uv run` on
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi

logger = logging.getLogger("extract_funding_cascade_job")


OUTPUT_SCHEMA = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("arxiv_id", pa.string()),
    pa.field("shard_id", pa.string()),
    pa.field("input_file", pa.string()),
    pa.field("row_idx", pa.int64()),
    pa.field("funding_statement", pa.string()),
    pa.field("rough_span", pa.string()),
    pa.field("chunk_prob", pa.float64()),
    pa.field("no_answer_prob", pa.float64()),
    pa.field("text_length", pa.int64()),
    pa.field("latency_ms", pa.float64()),
    pa.field("skip_cleanup", pa.bool_()),
    pa.field("error", pa.string()),
])


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the funding-statement cascade over input parquets and push results to hub.",
    )
    p.add_argument("--input-repo", required=True,
                    help="HF dataset repo id (default corpus: cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test).")
    p.add_argument("--input-files", required=True,
                    type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    help="Comma-separated in-repo parquet paths.")
    p.add_argument("--output-repo", required=True)
    p.add_argument("--job-tag", required=True,
                    help="Free-form tag for log correlation with the orchestrator.")
    p.add_argument("--text-column", default="vlm_markdown")
    p.add_argument("--id-column", default="arxiv_id")
    p.add_argument("--skip-cleanup", action="store_true",
                    help="Short-circuit Stage 3 (Qwen cleaner); emit raw stage-2 span.")
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--commit-batch-size", type=int, default=25)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--allow-cpu", action="store_true",
                    help="Skip CUDA probe; for local smoke tests only.")
    return p.parse_args(argv)


def make_output_row(batch_result, *, skip_cleanup: bool):
    meta = batch_result.metadata or {}
    r = batch_result.result
    return {
        "doc_id": batch_result.doc_id,
        "arxiv_id": meta.get("arxiv_id"),
        "shard_id": meta.get("shard_id"),
        "input_file": meta.get("input_file"),
        "row_idx": meta.get("row_idx"),
        "funding_statement": r.funding_statement,
        "rough_span": r.rough_span,
        "chunk_prob": r.chunk_prob,
        "no_answer_prob": r.no_answer_prob,
        "text_length": meta.get("text_length", 0),
        "latency_ms": r.latency_ms,
        "skip_cleanup": skip_cleanup,
        "error": r.error,
    }


def stage_parquet_locally(rows, *, path_in_repo, staging_dir):
    if rows is None:
        rows = []
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    local_path = staging_dir / Path(path_in_repo).name
    pq.write_table(table, local_path, compression="zstd")
    return local_path
```

**Step 4: Run → 7 passed.**

**Step 5: Commit**

```bash
git add hf_jobs/jobs/funding_statement_extractor_cascade && \
  git commit -m "feat: HF Job worker — argparse + OUTPUT_SCHEMA + stage_parquet_locally"
```

---

### Task 7.3: Worker — preflight + commit batching + 429 retry

**Files:**
- Modify: `hf_jobs/jobs/funding_statement_extractor_cascade/extract_funding_cascade_job.py`
- Modify: `hf_jobs/jobs/funding_statement_extractor_cascade/tests/test_extract_funding_cascade_job.py`

**Mirrors:** v1 `extract_funding_job.py:131-256` (preflight, retry hint parsing, commit_staged_batch, emit_done_lines). These helpers are not engine-specific; we **port them verbatim**.

**Step 1: Append failing tests** — copy v1 `tests/test_extract_funding_job.py:219-407` essentially verbatim, changing only the import path to `extract_funding_cascade_job`. Specifically port:
- `test_commit_staged_batch_single_commit`
- `test_commit_staged_batch_retries_on_429`
- `test_commit_staged_batch_empty_is_noop`
- `test_parse_retry_hint_seconds_variants`
- `test_preflight_commit_check_succeeds_first_try`
- `test_preflight_commit_check_retries_429_and_succeeds`
- `test_preflight_commit_check_raises_after_exhaustion`
- `test_preflight_commit_check_reraises_non_429`
- `test_batched_commit_groups_files`

Use the v1 file at `/Users/adambuttrick/Documents/GitHub/funding-metadata-enrichment/hf_jobs/jobs/funding_statement_extractor/tests/test_extract_funding_job.py:219-407` as the source.

**Step 2: Run → FAIL.**

**Step 3: Port `extract_funding_job.py:131-256` into `extract_funding_cascade_job.py`** — `_parse_retry_hint_seconds`, `preflight_commit_check`, `commit_staged_batch`, `emit_done_lines`. Copy verbatim from v1 lines 143–256. They are engine-agnostic.

**Step 4: Run → all ported tests pass.**

**Step 5: Commit**

```bash
git add hf_jobs/jobs/funding_statement_extractor_cascade && \
  git commit -m "feat: worker preflight + 429-retry batched commits (ported from v1)"
```

---

### Task 7.4: Worker — `main()` orchestration

**Files:**
- Modify: `hf_jobs/jobs/funding_statement_extractor_cascade/extract_funding_cascade_job.py`
- Modify: `hf_jobs/jobs/funding_statement_extractor_cascade/tests/test_extract_funding_cascade_job.py`

**Step 1: Write failing test** — exercise `main()` with mocked `load_dataset`, `load_models`, `extract_funding_batch`, `HfApi`:

```python
def test_main_end_to_end_with_mocks(tmp_path, monkeypatch):
    import extract_funding_cascade_job as mod
    from types import SimpleNamespace

    fake_rows = [
        {"vlm_markdown": "paper A text", "arxiv_id": "2401.00001",
         "status": "ok", "shard_id": "s0"},
        {"vlm_markdown": "paper B text", "arxiv_id": "2401.00002",
         "status": "ok", "shard_id": "s0"},
    ]

    def fake_load_dataset(repo, data_files, split, streaming):
        return iter(fake_rows)

    def fake_load_models(cfg):
        return SimpleNamespace()

    def fake_extract_batch(documents, *, models, cfg):
        for d in documents:
            result = SimpleNamespace(
                funding_statement=f"funding-{d.doc_id}", rough_span="rs",
                chunk_prob=0.9, no_answer_prob=0.05,
                latency_ms=10.0, error=None,
            )
            yield SimpleNamespace(doc_id=d.doc_id, result=result,
                                  enqueue_ts=0.0, yield_ts=0.01,
                                  metadata=d.metadata)

    monkeypatch.setattr(mod, "_load_dataset", fake_load_dataset, raising=False)
    monkeypatch.setattr(mod, "_load_models", fake_load_models, raising=False)
    monkeypatch.setattr(mod, "_extract_funding_batch", fake_extract_batch, raising=False)

    commits = []
    class FakeApi:
        def upload_file(self, **kw): pass
        def create_commit(self, **kw): commits.append(kw)

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    monkeypatch.setenv("EXTRACT_FUNDING_STAGING", str(tmp_path))

    rc = mod.main([
        "--input-repo", "ds", "--input-files", "test-00000.parquet",
        "--output-repo", "out-repo", "--job-tag", "tag-1",
        "--allow-cpu", "--skip-preflight",
    ])
    assert rc == 0
    # Exactly one final commit
    assert len(commits) == 1
    staged = (tmp_path / "test-00000.parquet")
    assert staged.exists()
    import pyarrow.parquet as pq
    tbl = pq.read_table(staged)
    assert tbl.num_rows == 2
    assert "funding-2401.00001" in tbl.column("funding_statement").to_pylist()
```

**Step 2: Run → FAIL.**

**Step 3: Append `main()` to `extract_funding_cascade_job.py`** — mirrors v1 `extract_funding_job.py:275-461` with cascade-specific substitutions:

```python
import os


def _load_dataset(repo, data_files, split, streaming):
    """Thin indirection so tests can monkeypatch without importing `datasets`."""
    from datasets import load_dataset
    return load_dataset(repo, data_files=data_files, split=split, streaming=streaming)


def _load_models(cfg):
    from funding_statement_extractor_cascade.cascade.loader import load_models
    return load_models(cfg)


def _extract_funding_batch(documents, *, models, cfg):
    from funding_statement_extractor_cascade.cascade.batch import extract_funding_batch
    return extract_funding_batch(documents, models=models, cfg=cfg)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stdout)
    logger.info("job_tag=%s input_files=%d skip_cleanup=%s",
                 args.job_tag, len(args.input_files), args.skip_cleanup)

    import torch

    if args.allow_cpu:
        logger.info("torch=%s cuda=%s (allow-cpu=True)",
                     torch.__version__, torch.cuda.is_available())
    else:
        cuda_ok = False
        for attempt in range(120):
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    cuda_ok = True
                    break
            except Exception as exc:
                logger.warning("cuda probe attempt %d failed: %s", attempt, exc)
            time.sleep(1)
        if not cuda_ok:
            logger.error("CUDA unavailable after 120s — aborting (no CPU fallback).")
            return 2
        logger.info("torch=%s cuda device=%s",
                     torch.__version__, torch.cuda.get_device_name(0))

    from funding_statement_extractor_cascade.cascade.types import CascadeConfig, DocPayload
    cascade_cfg = CascadeConfig(
        skip_cleanup=args.skip_cleanup,
        dtype=args.dtype,
        device="cpu" if args.allow_cpu else "cuda",
    )
    models = _load_models(cascade_cfg)

    staging = Path(os.environ.get("EXTRACT_FUNDING_STAGING",
                                    "/tmp/extract_funding_cascade_outputs"))
    if not args.skip_preflight:
        try:
            preflight_commit_check(args.output_repo, job_tag=args.job_tag,
                                    staging_dir=staging)
        except Exception as exc:
            logger.error("preflight check failed; aborting before GPU work: %s", exc)
            return 5

    staged_buffer: list[dict] = []
    batch_index = 0
    consecutive_commit_failures = 0
    MAX_CONSECUTIVE_COMMIT_FAILURES = 2

    for input_file in args.input_files:
        t0 = time.perf_counter()
        logger.info("[start file=%s]", input_file)
        try:
            ds = _load_dataset(args.input_repo, input_file, "train", True)
            counters = {"processed": 0, "skipped_status": 0, "skipped_empty": 0}

            def docs_iter():
                for row_idx, row in enumerate(ds):
                    if row.get("status") not in (None, "ok"):
                        counters["skipped_status"] += 1
                        continue
                    text = row.get(args.text_column)
                    if not text:
                        counters["skipped_empty"] += 1
                        continue
                    counters["processed"] += 1
                    yield DocPayload(
                        doc_id=str(row.get(args.id_column) or row_idx),
                        text=text,
                        metadata={
                            "row_idx": row_idx,
                            "input_file": input_file,
                            "arxiv_id": row.get(args.id_column),
                            "shard_id": row.get("shard_id"),
                            "text_length": len(text),
                        },
                    )

            output_rows = []
            for batch in _extract_funding_batch(docs_iter(), models=models,
                                                  cfg=cascade_cfg):
                output_rows.append(make_output_row(batch, skip_cleanup=args.skip_cleanup))

            out_path = f"predictions/{Path(input_file).name}"
            local_path = stage_parquet_locally(
                output_rows, path_in_repo=out_path, staging_dir=staging,
            )
            elapsed = time.perf_counter() - t0
            staged_buffer.append({
                "input_file": input_file,
                "path_in_repo": out_path,
                "local_path": local_path,
                "rows": len(output_rows),
                "elapsed_s": elapsed,
            })
            logger.info("[staged file=%s rows=%d local=%s buffered=%d]",
                         input_file, len(output_rows), local_path, len(staged_buffer))
            print(
                f"[file_summary file={input_file} processed={counters['processed']} "
                f"skipped_status={counters['skipped_status']} "
                f"skipped_empty={counters['skipped_empty']}]",
                flush=True,
            )

            if len(staged_buffer) >= args.commit_batch_size:
                batch_index += 1
                try:
                    commit_staged_batch(staged_buffer, repo_id=args.output_repo,
                                          batch_index=batch_index)
                except Exception as commit_exc:
                    consecutive_commit_failures += 1
                    logger.exception("batch commit %d failed (%d/%d): %s",
                                       batch_index, consecutive_commit_failures,
                                       MAX_CONSECUTIVE_COMMIT_FAILURES, commit_exc)
                    if consecutive_commit_failures >= MAX_CONSECUTIVE_COMMIT_FAILURES:
                        logger.error("aborting after %d consecutive commit failures",
                                       consecutive_commit_failures)
                        return 4
                else:
                    emit_done_lines(staged_buffer)
                    staged_buffer = []
                    consecutive_commit_failures = 0
        except Exception as exc:
            logger.exception("[fail file=%s] %s", input_file, exc)

    if staged_buffer:
        batch_index += 1
        try:
            commit_staged_batch(staged_buffer, repo_id=args.output_repo,
                                  batch_index=batch_index)
            emit_done_lines(staged_buffer)
        except Exception as exc:
            logger.exception("final batch commit failed: %s", exc)
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run → test passes.**

**Step 5: Commit**

```bash
git add hf_jobs/jobs/funding_statement_extractor_cascade && \
  git commit -m "feat: worker main() with CUDA probe + staged commit batching"
```

---

## Phase 8 — Local orchestrator

> **Reference:** v1 `hf_jobs/ops/funding_statement_extractor/orchestrate_extractions.py` (lines 1–end) and `tests/test_orchestrate_extractions.py` (lines 1–435). The orchestrator is engine-agnostic; we port it nearly verbatim with cascade-specific defaults (smaller batches; bench-corpus paths; new image/flavor invariants stay the same).
>
> **Bench corpus reality check:** The bench split has a single parquet (~597 rows). Defaults for `--max-files-per-batch`, `--target-seconds`, and `--max-in-flight` are smaller than v1's.

### Task 8.1: Manifest + parse_done_line

**Files:**
- Create: `hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py`
- Create: `hf_jobs/ops/funding_statement_extractor_cascade/tests/test_orchestrate_extractions.py`

**Step 1: Write failing tests** — port v1's `test_orchestrate_extractions.py:33-115` (manifest round-trip, atomic write, parse_done_line variants, `read_row_count_local`). Use the same `_row` helper.

**Step 2: Run → FAIL.**

**Step 3: Port v1 `orchestrate_extractions.py:1-117`** — `ManifestRow`, `Manifest`, `write_manifest`, `read_manifest`, `parse_done_line`, `read_row_count_local`. Verbatim. Only `_DONE_RE` is shared with the worker contract from Task 7.3.

**Step 4: Run → tests pass.**

**Step 5: Commit**

```bash
git add hf_jobs/ops/funding_statement_extractor_cascade && \
  git commit -m "feat: orchestrator manifest + parse_done_line (ported from v1)"
```

---

### Task 8.2: Batch picking + EMA + listing + reconcile

**Files:**
- Modify: `hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py`
- Modify: `hf_jobs/ops/funding_statement_extractor_cascade/tests/test_orchestrate_extractions.py`

**Step 1: Append failing tests** — port v1's `test_pick_batch_*`, `test_update_ema`, `test_list_input_files_with_sizes`, `test_seed_merges_new_files_preserves_existing`, `test_reconcile_marks_done_when_output_exists` (v1 file lines 60–186).

**Step 2: Run → FAIL.**

**Step 3: Port v1 `orchestrate_extractions.py:78-209`** — `pick_next_batch`, `update_ema`, `list_input_files_with_sizes`, `seed_or_merge_manifest`, `reconcile_against_outputs`, `list_existing_outputs`. Verbatim.

**Step 4: Run → tests pass.**

**Step 5: Commit**

```bash
git add hf_jobs/ops/funding_statement_extractor_cascade && \
  git commit -m "feat: orchestrator pick_next_batch + EMA + hub listing helpers"
```

---

### Task 8.3: Job submission + polling

**Files:**
- Modify: `hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py`
- Modify: `hf_jobs/ops/funding_statement_extractor_cascade/tests/test_orchestrate_extractions.py`

**Step 1: Append failing tests** — port v1's `test_submit_job_retries_429`, `test_submit_job_gives_up_after_max_retries`, `test_submit_job_retries_503`, `test_poll_job_state_*` (v1 file lines 183–435).

**Step 2: Run → FAIL.**

**Step 3: Port v1 `orchestrate_extractions.py:212-303`** — `IMAGE`, `FLAVOR`, `submit_a100_job`, `JobState`, `poll_job_state`. Verbatim — the bash invocation is the same regardless of engine.

**Step 4: Run → tests pass.**

**Step 5: Commit**

```bash
git add hf_jobs/ops/funding_statement_extractor_cascade && \
  git commit -m "feat: orchestrator submit_a100_job + poll_job_state (ported from v1)"
```

---

### Task 8.4: Run summary + README rendering

**Files:**
- Modify: `hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py`
- Modify: `hf_jobs/ops/funding_statement_extractor_cascade/tests/test_orchestrate_extractions.py`

**Step 1: Append failing test** — port v1's `test_write_run_summary_renders_local_json` (v1 file lines 347–410), adjusting the asserted README content to cascade-specific text:

```python
def test_write_run_summary_renders_cascade_readme(tmp_path, monkeypatch):
    # ... same as v1 ...
    readme_text = readme_path.read_text()
    assert "cascade" in readme_text.lower()
    assert "A100-large bf16" in readme_text
    assert "funding-cleaning-qwen3-4b-lora" in readme_text
```

**Step 2: Run → FAIL.**

**Step 3: Port v1 `orchestrate_extractions.py:317-390`** — `_render_readme`, `write_run_summary`. Update `_render_readme` to reflect cascade specifics:

```python
def _render_readme(summary):
    return f"""# arxiv-funding-statement-cascade-extractions

Funding-statement extractions via the 3-stage cascade.

- Engine: `funding_statement_extractor_cascade`
- Stage 1: `cometadata/funding-chunk-classifier-modernbert-base` (threshold 0.4, top-2)
- Stage 2: `cometadata/funding-extraction-modernbert-base-spanhead` (no-answer threshold 0.5)
- Stage 3: `cometadata/funding-cleaning-qwen3-4b-lora` (greedy, max 512 new tokens)
- Hardware: A100-large bf16
- Files processed: {summary['n_files']}
- Status: {summary['status_counts']}
- Total rows: {summary['total_rows']:,}
- Total worker seconds: {summary['total_worker_seconds']:.0f}
- Est cost @ $5/hr: ${summary['estimated_cost_usd']:.2f}
- Completed: {summary['completed_at']}

## Schema
Per row in `predictions/*.parquet`:
- `doc_id`, `arxiv_id`, `shard_id`, `input_file`, `row_idx`
- `funding_statement: string`  — cleaned answer; `""` if no funding found
- `rough_span: string`         — pre-cleanup stage-2 output
- `chunk_prob: float64`, `no_answer_prob: float64`
- `text_length: int64`, `latency_ms: float64`, `skip_cleanup: bool`, `error: string`
"""
```

**Step 4: Run → test passes.**

**Step 5: Commit**

```bash
git add hf_jobs/ops/funding_statement_extractor_cascade && \
  git commit -m "feat: orchestrator write_run_summary with cascade README rendering"
```

---

### Task 8.5: `parse_orch_args` + `main` + dry-run

**Files:**
- Modify: `hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py`
- Modify: `hf_jobs/ops/funding_statement_extractor_cascade/tests/test_orchestrate_extractions.py`

**Step 1: Append failing test** — port v1's `test_dry_run_exercises_state_machine` (v1 file lines 316–345), adjusting the default values:

```python
def test_dry_run_against_bench_corpus(tmp_path, monkeypatch):
    import orchestrate_extractions as mod

    listing = [("test-00000-of-00001.parquet", 1_000_000)]  # bench is 1 file
    monkeypatch.setattr(mod, "list_input_files_with_sizes",
                          lambda repo, sub: listing)
    monkeypatch.setattr(mod, "list_existing_outputs", lambda repo: set())

    manifest_path = tmp_path / "bench.parquet"
    rc = mod.main([
        "--manifest", str(manifest_path),
        "--input-repo", "cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test",
        "--input-subdir", "",
        "--output-repo", "user/cascade-bench-out",
        "--max-in-flight", "1",
        "--max-files-per-batch", "1",
        "--target-seconds", "9999",
        "--seed-seconds-per-byte", "1e-6",
        "--dry-run",
    ])
    assert rc == 0
    rows = mod.read_manifest(manifest_path)
    assert rows[0].status == "done"
    assert rows[0].job_id.startswith("dry-")
```

**Step 2: Port v1 `orchestrate_extractions.py:393-end`** — `parse_orch_args` (with cascade defaults below) and `main()` with the dry-run state machine.

`parse_orch_args` defaults to change:
- `--input-repo` default: `cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test`
- `--input-subdir` default: `""` (bench data is at root, not in a subdir)
- `--output-repo` default: leave empty (user must specify; bench is private to them)
- `--manifest` default: `manifests/cascade-bench-2026-05-19.parquet`
- `--max-in-flight` default: `1` (bench is one file)
- `--target-seconds` default: `1800` (30 min — cascade is slower per-doc than ColBERT)
- `--max-files-per-batch` default: `5`
- `--worker-script` default: `hf_jobs/jobs/funding_statement_extractor_cascade/extract_funding_cascade_job.py`

If `--input-subdir` is `""`, `list_input_files_with_sizes` should call `list_repo_tree(path_in_repo="", recursive=True)` — add a guard:

```python
def list_input_files_with_sizes(repo_id, subdir):
    api = HfApi()
    out = []
    kwargs = dict(repo_id=repo_id, repo_type="dataset", recursive=True)
    if subdir:
        kwargs["path_in_repo"] = subdir
    for entry in api.list_repo_tree(**kwargs):
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if path is None or size is None: continue
        if not path.endswith(".parquet"): continue
        out.append((path, int(size)))
    out.sort(key=lambda t: t[0])
    return out
```

**Step 3: Run → test passes.**

**Step 4: Commit**

```bash
git add hf_jobs/ops/funding_statement_extractor_cascade && \
  git commit -m "feat: orchestrator parse_orch_args + main with cascade bench-corpus defaults"
```

---

## Phase 9 — Bench / profile / diff scripts

> **Reference:** v1 `hf_jobs/jobs/funding_statement_extractor/benchmark_hf_job.py`, `profile_on_h200.py`, and v1 `hf_jobs/ops/funding_statement_extractor/{diff_predictions,_profile_utils,profile_extraction}.py`. These scripts are all thin tools — each does one thing.
>
> **Scope reminder:** The bench corpus has gold funding statements in a column (likely `funding_statement` or `gold_funding`). Confirm the column name in Task 0.5 before writing `benchmark_hf_job.py`. Per the spec: end-to-end bench gives best@0.85 F1 = 0.953, strict tsr@0.95 F1 = 0.725.

### Task 9.1: `benchmark_hf_job.py` — local F1 vs gold

**Files:**
- Create: `hf_jobs/jobs/funding_statement_extractor_cascade/benchmark_hf_job.py`
- Create: `hf_jobs/jobs/funding_statement_extractor_cascade/tests/test_benchmark.py`

**Step 1: Write failing test** — exercise the metric helpers, not the model:

```python
import pytest

from benchmark_hf_job import (
    best_match_token_f1,
    char_overlap_ratio,
    score_predictions,
    strict_token_set_recall,
)


@pytest.mark.parametrize("pred,gold,want", [
    ("a b c", "a b c", 1.0),
    ("a b", "a b c", pytest.approx(0.8)),     # F1 of {a,b} vs {a,b,c}
    ("", "a b", 0.0),
    ("", "", 1.0),                              # empty/empty = perfect match
])
def test_best_match_token_f1(pred, gold, want):
    assert best_match_token_f1(pred, gold) == want


def test_strict_token_set_recall_full_match():
    assert strict_token_set_recall("a b c", "a b c") == 1.0


def test_strict_token_set_recall_partial():
    # gold tokens {a,b,c}, pred tokens {a,b} → recall=2/3
    assert strict_token_set_recall("a b", "a b c") == pytest.approx(2/3)


def test_score_predictions_aggregates_correctly():
    preds = [{"arxiv_id": "1", "funding_statement": "Funded by NSF."},
              {"arxiv_id": "2", "funding_statement": ""}]
    golds = [{"arxiv_id": "1", "gold_funding": "Funded by NSF grant 123."},
              {"arxiv_id": "2", "gold_funding": ""}]
    summary = score_predictions(preds, golds, id_col="arxiv_id",
                                  pred_col="funding_statement",
                                  gold_col="gold_funding")
    assert summary["n"] == 2
    assert "best_at_0.85_f1" in summary
    assert "strict_tsr_at_0.95_f1" in summary
```

**Step 2: Run → FAIL.**

**Step 3: Implement `benchmark_hf_job.py`** — standalone script (no PEP 723 needed; it runs locally). Mirrors v1 `benchmark_hf_job.py` shape but with cascade output schema.

```python
"""Benchmark cascade predictions against the gold funding statements on the bench split.

Usage:
    python benchmark_hf_job.py --predictions-repo user/cascade-bench-out \
                                --gold-repo cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test \
                                --output bench-summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Dict, List, Optional


_TOK_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(s: str) -> List[str]:
    return _TOK_RE.findall((s or "").lower())


def best_match_token_f1(pred: str, gold: str) -> float:
    pt, gt = Counter(_tokens(pred)), Counter(_tokens(gold))
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    overlap = sum((pt & gt).values())
    if overlap == 0:
        return 0.0
    p = overlap / sum(pt.values())
    r = overlap / sum(gt.values())
    return 2 * p * r / (p + r)


def strict_token_set_recall(pred: str, gold: str) -> float:
    pt, gt = set(_tokens(pred)), set(_tokens(gold))
    if not gt:
        return 1.0 if not pt else 0.0
    return len(pt & gt) / len(gt)


def char_overlap_ratio(pred: str, gold: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    common = sum(1 for ch in pred if ch in gold)
    return common / max(len(gold), 1)


def score_predictions(predictions, golds, *, id_col: str, pred_col: str, gold_col: str) -> Dict:
    by_gold = {g[id_col]: g.get(gold_col, "") for g in golds}
    f1s, tsrs = [], []
    for p in predictions:
        gid = p[id_col]
        if gid not in by_gold:
            continue
        gold = by_gold[gid]
        pred = p.get(pred_col, "") or ""
        f1s.append(best_match_token_f1(pred, gold))
        tsrs.append(strict_token_set_recall(pred, gold))
    n = len(f1s)

    def at_threshold(values, thresh):
        return sum(v >= thresh for v in values) / max(n, 1)

    return {
        "n": n,
        "mean_best_match_f1": sum(f1s) / max(n, 1),
        "best_at_0.85_f1": at_threshold(f1s, 0.85),
        "mean_strict_tsr": sum(tsrs) / max(n, 1),
        "strict_tsr_at_0.95_f1": at_threshold(tsrs, 0.95),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-repo", required=True)
    p.add_argument("--gold-repo",
                    default="cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test")
    p.add_argument("--gold-split", default="test")
    p.add_argument("--id-col", default="arxiv_id")
    p.add_argument("--pred-col", default="funding_statement")
    p.add_argument("--gold-col", default="funding_statement")  # confirm in 0.5
    p.add_argument("--output", default="bench-summary.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset
    preds = list(load_dataset(args.predictions_repo, split="train"))
    golds = list(load_dataset(args.gold_repo, split=args.gold_split))
    summary = score_predictions(preds, golds, id_col=args.id_col,
                                  pred_col=args.pred_col, gold_col=args.gold_col)
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Step 4: Run → tests pass.**

**Step 5: Commit**

```bash
git add hf_jobs/jobs/funding_statement_extractor_cascade/benchmark_hf_job.py \
        hf_jobs/jobs/funding_statement_extractor_cascade/tests/test_benchmark.py && \
  git commit -m "feat: benchmark script computing best@0.85 + strict tsr@0.95 F1"
```

---

### Task 9.2: `profile_on_h200.py` — single-doc latency profile

**Files:**
- Create: `hf_jobs/jobs/funding_statement_extractor_cascade/profile_on_h200.py`

**Mirrors:** v1 `profile_on_h200.py` shape (PEP 723, single-doc timing, JSON summary). Diverges: prints per-stage timings (stage1_ms, stage2_ms, stage3_ms) instead of just total.

**Step 1: Write the script** (no unit test — it's a one-shot benchmark tool):

```python
#!/usr/bin/env python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "funding-statement-extractor-cascade @ git+https://github.com/USER/parth_pipeline.git@main",
#     "datasets>=3.0.0",
#     "huggingface-hub>=0.26.0",
#     "transformers>=4.49,<6",
#     "peft>=0.14.0",
#     "accelerate>=1.0",
#     "torch>=2.4.0,<2.8.0",
#     "pyarrow>=12.0.0",
# ]
# ///
"""Latency profile of the cascade on a single H200/A100. Prints per-stage times."""
from __future__ import annotations

import argparse
import json
import time

from datasets import load_dataset

from funding_statement_extractor_cascade.cascade.loader import load_models
from funding_statement_extractor_cascade.cascade.pipeline import extract_funding
from funding_statement_extractor_cascade.cascade.types import CascadeConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-repo",
                    default="cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test")
    p.add_argument("--split", default="test")
    p.add_argument("--text-column", default="vlm_markdown")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--skip-cleanup", action="store_true")
    p.add_argument("--output", default="profile-summary.json")
    args = p.parse_args()

    cfg = CascadeConfig(skip_cleanup=args.skip_cleanup)
    models = load_models(cfg)

    ds = load_dataset(args.input_repo, split=args.split)
    latencies = []
    for i, row in enumerate(ds):
        if i >= args.n: break
        text = row[args.text_column] or ""
        t0 = time.perf_counter()
        r = extract_funding(text, models, cfg)
        wall = (time.perf_counter() - t0) * 1000.0
        latencies.append({
            "doc_id": row.get("arxiv_id"),
            "text_chars": len(text),
            "wall_ms": wall,
            "extract_ms": r.latency_ms,
            "had_funding": bool(r.funding_statement),
        })

    summary = {
        "n": len(latencies),
        "skip_cleanup": args.skip_cleanup,
        "mean_wall_ms": sum(x["wall_ms"] for x in latencies) / max(len(latencies), 1),
        "p50_wall_ms": sorted(x["wall_ms"] for x in latencies)[len(latencies) // 2],
        "max_wall_ms": max((x["wall_ms"] for x in latencies), default=0),
        "n_with_funding": sum(1 for x in latencies if x["had_funding"]),
        "latencies": latencies,
    }
    with open(args.output, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "latencies"}, indent=2))


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add hf_jobs/jobs/funding_statement_extractor_cascade/profile_on_h200.py && \
  git commit -m "feat: profile_on_h200 single-doc latency benchmark"
```

---

### Task 9.3: `profile_extraction.py` + `_profile_utils.py` — local equivalents

**Files:**
- Create: `hf_jobs/ops/funding_statement_extractor_cascade/_profile_utils.py`
- Create: `hf_jobs/ops/funding_statement_extractor_cascade/profile_extraction.py`

**Mirrors:** v1 `_profile_utils.py` + `profile_extraction.py`. **Port verbatim** (engine-agnostic — both files measure throughput statistics over a parquet of predictions/outputs). Change two imports: cascade column names (`funding_statement` not `predicted_statements`).

The `_profile_utils.py` from v1 has helpers for parquet listing, sizes, ratios — adopt as-is.

The `profile_extraction.py` runs the cascade locally over a small sample. Replace v1's `extract_funding_statements_batch` call with `extract_funding_batch`.

**Step 1: Port the files** with the imports and column substitutions above.

**Step 2: Commit**

```bash
git add hf_jobs/ops/funding_statement_extractor_cascade/{_profile_utils,profile_extraction}.py && \
  git commit -m "feat: local profile_extraction + _profile_utils (ported from v1)"
```

---

### Task 9.4: `diff_predictions.py` — side-by-side cascade vs gold

**Files:**
- Create: `hf_jobs/ops/funding_statement_extractor_cascade/diff_predictions.py`

**Mirrors:** v1 `diff_predictions.py` (cascade vs gold instead of cascade vs v1). Produces a JSONL file with one row per doc containing `{arxiv_id, gold, prediction, best_f1, strict_tsr, agree_at_0.85}`.

**Step 1: Write the script** (~80 lines; reuse helpers from `benchmark_hf_job.py`):

```python
"""Side-by-side diff of cascade predictions vs gold for the bench corpus.

Outputs JSONL: arxiv_id, gold, prediction, best_match_f1, strict_tsr, agree.
"""
from __future__ import annotations

import argparse
import json
import sys

# Import the metric helpers from the benchmark script (sibling file)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]
                    / "jobs" / "funding_statement_extractor_cascade"))
from benchmark_hf_job import best_match_token_f1, strict_token_set_recall  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-repo", required=True)
    p.add_argument("--gold-repo",
                    default="cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test")
    p.add_argument("--gold-split", default="test")
    p.add_argument("--id-col", default="arxiv_id")
    p.add_argument("--pred-col", default="funding_statement")
    p.add_argument("--gold-col", default="funding_statement")
    p.add_argument("--output", default="diff.jsonl")
    p.add_argument("--threshold", type=float, default=0.85)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset
    preds = list(load_dataset(args.predictions_repo, split="train"))
    golds = list(load_dataset(args.gold_repo, split=args.gold_split))

    by_id = {g[args.id_col]: g for g in golds}
    with open(args.output, "w") as fh:
        for p in preds:
            gid = p[args.id_col]
            g = by_id.get(gid)
            if g is None:
                continue
            gold = g.get(args.gold_col, "") or ""
            pred = p.get(args.pred_col, "") or ""
            f1 = best_match_token_f1(pred, gold)
            tsr = strict_token_set_recall(pred, gold)
            fh.write(json.dumps({
                "arxiv_id": gid,
                "gold": gold,
                "prediction": pred,
                "best_match_f1": f1,
                "strict_tsr": tsr,
                "agree": f1 >= args.threshold,
            }) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Commit**

```bash
git add hf_jobs/ops/funding_statement_extractor_cascade/diff_predictions.py && \
  git commit -m "feat: diff_predictions cascade-vs-gold JSONL writer"
```

---

### Task 9.5: Phase-9 smoke tests

**Files:**
- Modify: `hf_jobs/jobs/funding_statement_extractor_cascade/tests/test_benchmark.py`

**Step 1: Add an integration smoke test** that exercises `score_predictions` against the F1 numbers from the spec (sanity check that our metric implementation matches the reported scale):

```python
def test_score_predictions_matches_reported_quality_on_synthetic_set():
    # Construct 20 synthetic gold/pred pairs where 19 match best@0.85 and 1 doesn't.
    preds = [{"arxiv_id": str(i), "funding_statement": f"Funded by grant {i} NSF NIH"}
              for i in range(20)]
    golds = [{"arxiv_id": str(i), "gold_funding": f"Funded by grant {i} NSF NIH"}
              for i in range(19)]
    golds.append({"arxiv_id": "19", "gold_funding": "Completely unrelated text here"})
    summary = score_predictions(preds, golds, id_col="arxiv_id",
                                  pred_col="funding_statement",
                                  gold_col="gold_funding")
    assert summary["best_at_0.85_f1"] == pytest.approx(19/20)
```

**Step 2: Run → pass.**

**Step 3: Commit**

```bash
git add hf_jobs/jobs/funding_statement_extractor_cascade/tests/test_benchmark.py && \
  git commit -m "test: benchmark scoring matches reported scale on synthetic data"
```

---

## Phase 10 — Documentation

### Task 10.1: Package `README.md`

**Files:**
- Create: `/Users/adambuttrick/Downloads/parth_pipeline/README.md`

**Mirrors:** v1 `funding_statement_extractor/README.md` structure (Installation / Quick Start / CLI Options / Output Format). Content is cascade-specific.

**Step 1: Write the README.** Must cover:
- One-paragraph description: 3-stage cascade, VLM-markdown input, single cleaned statement output.
- **Benchmark numbers** stated up front: best@0.85 F1 = 0.953, strict tsr@0.95 F1 = 0.725 on the 597-doc bench split.
- The three models with HF links + one line each on what they do.
- Install via `uv venv && uv pip install -e .` (per @feedback_python_env_uv).
- Quick Start: `funding-statement-extractor-cascade -i paper.md -o results.json` and `python -m funding_statement_extractor_cascade -i docs/`.
- Python API example:
  ```python
  from funding_statement_extractor_cascade.cascade.loader import load_models
  from funding_statement_extractor_cascade.cascade.pipeline import extract_funding
  from funding_statement_extractor_cascade.cascade.types import CascadeConfig

  cfg = CascadeConfig()
  models = load_models(cfg)
  result = extract_funding(open("paper.md").read(), models, cfg)
  print(result.funding_statement or "(no funding statement)")
  ```
- Full CLI options table (every flag from `add_arguments`).
- Output JSON schema (the `_flush` payload shape).
- The `--skip-cleanup` semantics: returns raw stage-2 span verbatim.
- A "Dependencies & GPU" section noting: needs a CUDA GPU; A100-large recommended; transformers ≥ 4.49 pin; ~9GB VRAM with cleaner, ~0.5GB without.
- A "Package Structure" tree (copy the layout from this plan's top).

**Step 2: Commit**

```bash
git add README.md && git commit -m "docs: package README with cascade usage and benchmark numbers"
```

---

### Task 10.2: HF Job worker `README.md`

**Files:**
- Create: `hf_jobs/jobs/funding_statement_extractor_cascade/README.md`

**Mirrors:** v1 `hf_jobs/jobs/funding_statement_extractor/README.md`.

**Step 1: Write it.** Must cover:
- What the worker does: streams input parquets, runs the cascade, pushes `predictions/*.parquet`.
- The CLI contract (`--input-repo --input-files --output-repo --job-tag --text-column --id-column --skip-cleanup --dtype --commit-batch-size --skip-preflight --allow-cpu`).
- The `[done file=...]` / `[file_summary ...]` stdout contract the orchestrator parses.
- The exit codes (0 ok, 2 no CUDA, 3 final commit failed, 4 too many commit failures, 5 preflight failed).
- The `OUTPUT_SCHEMA` table.
- An example `hf jobs uv run` invocation:
  ```bash
  hf jobs uv run --flavor a100-large \
    --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    hf_jobs/jobs/funding_statement_extractor_cascade/extract_funding_cascade_job.py \
    --input-repo cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test \
    --input-files test-00000-of-00001.parquet \
    --output-repo USER/cascade-bench-out \
    --job-tag cascade-bench-001
  ```
- A note that the PEP 723 `git+https://` dependency URL must be updated to the real repo URL before launch.

**Step 2: Commit**

```bash
git add hf_jobs/jobs/funding_statement_extractor_cascade/README.md && \
  git commit -m "docs: HF Job worker README"
```

---

### Task 10.3: Orchestrator `README.md`

**Files:**
- Create: `hf_jobs/ops/funding_statement_extractor_cascade/README.md`

**Mirrors:** v1 `hf_jobs/ops/funding_statement_extractor/README.md`.

**Step 1: Write it.** Must cover:
- What the orchestrator does: manifest-backed resume, EMA rebalancing, 429-aware retry, batched job submission over the bench corpus.
- `parse_orch_args` flags + their cascade defaults.
- `--dry-run` usage for testing the state machine without spending GPU.
- The manifest schema (ManifestRow fields).
- An example end-to-end run:
  ```bash
  HF_TOKEN=hf_... python hf_jobs/ops/funding_statement_extractor_cascade/orchestrate_extractions.py \
    --output-repo USER/cascade-bench-out
  ```
- A note that since the bench corpus is one parquet, the orchestrator mostly demonstrates parity with v1; it becomes load-bearing only when pointed at a larger VLM-markdown corpus.

**Step 2: Commit**

```bash
git add hf_jobs/ops/funding_statement_extractor_cascade/README.md && \
  git commit -m "docs: orchestrator README"
```

---

## Final verification

After Task 10.3, run the full suite and confirm everything is green:

**Step 1: Package tests**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  source .venv/bin/activate && pytest -q
```
Expected: all tests pass, zero failures, zero errors. No test should require network or CUDA.

**Step 2: HF Job worker tests**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  pytest hf_jobs/jobs/funding_statement_extractor_cascade/tests -q
```
Expected: all pass.

**Step 3: Orchestrator tests**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  pytest hf_jobs/ops/funding_statement_extractor_cascade/tests -q
```
Expected: all pass.

**Step 4: CLI smoke**

```bash
cd /Users/adambuttrick/Downloads/parth_pipeline && \
  python -m funding_statement_extractor_cascade --help
```
Expected: help text renders without error.

**Step 5: Final commit if anything was touched during verification** — otherwise the repo is already clean.

---

## Verification checklist (definition of done)

- [ ] `pytest -q` green across all three test roots, no network, no CUDA.
- [ ] `extract_funding()` reproduces the reference snippet's logic exactly (chunk threshold 0.4, top-2; span no-answer 0.5; span max-length 300; greedy decode; `NONE` → `""`).
- [ ] `--skip-cleanup` returns the raw stage-2 span and never calls the cleaner.
- [ ] Vendored `modeling.py` files import cleanly and carry a provenance header with the upstream commit SHA.
- [ ] HF Job worker emits `[done file=...]` lines only after a successful commit (orchestrator contract).
- [ ] `OUTPUT_SCHEMA` is the flat single-statement shape (no list wrapping) — divergence from v1 documented in the README.
- [ ] `transformers>=4.49` pinned in both `pyproject.toml` and the PEP 723 header.
- [ ] Three READMEs written (package, worker, orchestrator).
- [ ] No dependency on the v1 `funding_statement_extractor` package anywhere.

---

## Risk register (re-check before launching a real sweep)

| Risk | Surfaces in | Mitigation |
|---|---|---|
| Qwen3 fp32 OOM | Task 4.6, 7.4 | `transformers>=4.49` + `dtype=bf16`; verified by `load_models` test mocking `dtype` resolution |
| Bench column names differ from `vlm_markdown`/`arxiv_id`/`funding_statement` | Task 0.5, 9.1 | One-shot dataset probe in Task 0.5 — **must run before Phase 7/9** |
| Vendored `modeling.py` drift | Task 1.1, 1.2 | Provenance header with commit SHA; re-vendor if upstream changes |
| `device_map="auto"` CPU offload | Task 4.6 | Use `device_map=cfg.device` (a fixed string), never `"auto"` |
| PEP 723 `git+https://` URL is a placeholder | Task 7.2, 9.2 | README note + must be set to the real repo URL before `hf jobs uv run` |
| Cascade slower per-doc than v1 ColBERT | Task 8.5 | Orchestrator `--target-seconds` default lowered to 1800; revisit after first profiling run |

---

## Notes on v1 reuse (port vs mirror vs new)

- **Ported verbatim (engine-agnostic):** `io/checkpointing.py`, the worker's `_parse_retry_hint_seconds` / `preflight_commit_check` / `commit_staged_batch` / `emit_done_lines`, and the orchestrator's `ManifestRow` / `write_manifest` / `read_manifest` / `parse_done_line` / `pick_next_batch` / `update_ema` / `list_input_files_with_sizes` / `seed_or_merge_manifest` / `reconcile_against_outputs` / `list_existing_outputs` / `submit_a100_job` / `poll_job_state` / `write_run_summary`.
- **Mirrored with edits:** `io/loaders.py` (vlm_markdown fallback order), `config/settings.py` (CascadeRuntimeSettings replaces ExtractionSettings), `cli/main.py` + `cascade/cli.py` (smaller — one engine), the worker's `parse_args` / `make_output_row` / `stage_parquet_locally` / `main` (cascade schema + cascade call), `_render_readme` / `parse_orch_args` (cascade defaults).
- **New (no v1 analog):** `cascade/types.py`, `cascade/loader.py`, `cascade/pipeline.py` (the 3-stage core), `cascade/batch.py` (sequential, not the v1 multiprocessing pipeline), `processing/chunking.py`, `models/*/modeling.py` (vendored), `tests/cascade_fakes.py`, `benchmark_hf_job.py` (cascade was new; v1's benchmarked ColBERT).
- **Deliberately NOT ported:** v1's `build_funded_only_split.py` (we're bench-only — no corpus pre-split needed), v1's `processing/normalization.py` (the cleaner LoRA does normalization), v1's `statements/post_filter.py` / `config/loader.py` / queries+patterns YAML (ColBERT-specific — the cascade has no queries/patterns), v1's `legacy_engine` ProcessPoolExecutor path.

---
