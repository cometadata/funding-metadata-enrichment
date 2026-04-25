# ArXiv funding-statement extraction orchestration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a worker script (`scripts/extract_funding_job.py`) that runs the Tier 2 funding-statement extractor inside an HF Job over a list of input parquet files, plus a local orchestrator (`scripts/orchestrate_extractions.py`) that manages 8 concurrent H200 jobs over the full `cometadata/arxiv-latex-extract-full-text/results-2026-04-24/` corpus with resume, retry, and rebalancing.

**Architecture:** Two scripts, one local manifest parquet. Worker is stateless, processes its assigned input files one at a time, pushes one output parquet per input file to `cometadata/arxiv-funding-statement-extractions`, and logs `[done file=X rows=N elapsed_s=T]` lines that the orchestrator parses. Orchestrator owns the manifest, polls `hf jobs inspect` + `hf jobs logs` every 60s, and refills the in-flight pool up to 8. See `docs/plans/2026-04-24-arxiv-funding-extraction-orchestration-design.md` for the full design.

**Tech Stack:** Python 3.12, `huggingface_hub`, `datasets` (streaming), `pyarrow`, `pylate`, the in-tree `funding_statement_extractor` package. Worker uses PEP-723 inline deps via `hf jobs uv run` on the `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` image (h200 flavor). Orchestrator runs locally in `.venv`.

**Reference files (read first):**
- `docs/plans/2026-04-24-arxiv-funding-extraction-orchestration-design.md` — design.
- `scripts/benchmark_hf_job.py` — template for worker (PEP-723 header, CUDA probe, dtype patch, batch engine usage, `_apply_dtype_patch`, `_gpu_mem_summary`).
- `funding_statement_extractor/statements/batch_extraction.py` — `extract_funding_statements_batch()` API and `DocPayload`/`Result` shapes.
- `funding_statement_extractor/config/loader.py` — `load_queries()`.
- `CLAUDE.md` (gitignored) — Tier 2 defaults, HF Jobs invariants (torch pin, CUDA retry probe, image).

---

## Task 0: Verify input dataset schema

Before writing any code, confirm the column names and row schema of one input parquet so all downstream code uses the right defaults.

**Files:** none (exploratory).

**Step 1: Read one parquet's schema and a sample row.**

Run from repo root:

```bash
.venv/bin/python3 - <<'PYEOF'
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
fs = HfFileSystem()
base = "datasets/cometadata/arxiv-latex-extract-full-text/results-2026-04-24"
files = sorted(fs.ls(base, detail=False))
print(f"n_files={len(files)}")
print(f"first_5={files[:5]}")
with fs.open(files[0]) as f:
    pf = pq.ParquetFile(f)
    print(f"schema:\n{pf.schema_arrow}")
    print(f"num_rows={pf.metadata.num_rows}")
    tbl = pf.read_row_group(0, columns=None)
    print(f"first_row_keys={list(tbl.column_names)}")
    row0 = {c: tbl[c][0].as_py() for c in tbl.column_names if c != 'text'}
    print(f"first_row_no_text={row0}")
    text = tbl['text'][0].as_py() if 'text' in tbl.column_names else None
    print(f"text_present={text is not None} text_len={len(text) if text else 0}")
PYEOF
```

**Step 2: Record findings.**

In the implementation, hard-code the defaults observed:
- `--text-column` default
- `--id-column` default
- Carry-through fields (e.g. `arxiv_id`, plus any others worth keeping like `title`, `categories`, `update_date`)
- Total file count + average row count (informs orchestrator sanity)

If the column names differ from the assumed `text` / `arxiv_id`, update Tasks 2, 8, 13, 14 below to match. **Do not proceed past Task 0 without this confirmation.**

**Step 3: Commit a short notes file.**

```bash
mkdir -p docs/plans/notes
.venv/bin/python3 -c "..." > docs/plans/notes/input-schema-2026-04-24.txt  # paste the output
git add docs/plans/notes/input-schema-2026-04-24.txt
git commit -m "docs: record input parquet schema for 2026-04-24 corpus"
```

---

## Task 1: Worker — output-row builder (TDD)

Pure function that maps a `BatchResult` from `extract_funding_statements_batch()` plus carry-through metadata to an output dict matching the schema in the design.

**Files:**
- Create: `scripts/extract_funding_job.py` (initial skeleton with just the helper + a module docstring)
- Test: `tests/scripts/test_extract_funding_job.py`

**Step 1: Write the failing test.**

```python
# tests/scripts/test_extract_funding_job.py
from types import SimpleNamespace
from scripts.extract_funding_job import make_output_row


def _fake_statement(statement, score, query, paragraph_idx):
    return SimpleNamespace(statement=statement, score=score,
                           query=query, paragraph_idx=paragraph_idx)


def test_make_output_row_with_predictions():
    result = SimpleNamespace(
        doc_id="2401.00001",
        statements=[
            _fake_statement("Funded by NSF grant 12345.", 42.0, "who funded this work", 7),
        ],
        error=None,
        metadata={
            "arxiv_id": "2401.00001",
            "input_file": "results-2026-04-24/shard-00000.parquet",
            "row_idx": 3,
            "text_length": 8421,
        },
        enqueue_ts=100.0,
        yield_ts=100.5,
    )
    row = make_output_row(result)
    assert row["arxiv_id"] == "2401.00001"
    assert row["doc_id"] == "2401.00001"
    assert row["input_file"] == "results-2026-04-24/shard-00000.parquet"
    assert row["row_idx"] == 3
    assert row["predicted_statements"] == ["Funded by NSF grant 12345."]
    assert row["predicted_details"][0]["score"] == 42.0
    assert row["predicted_details"][0]["paragraph_idx"] == 7
    assert row["text_length"] == 8421
    assert row["latency_ms"] == 500.0
    assert row["error"] is None


def test_make_output_row_zero_predictions_kept():
    result = SimpleNamespace(
        doc_id="2401.00002", statements=[], error=None,
        metadata={"arxiv_id": "2401.00002", "input_file": "x.parquet",
                  "row_idx": 0, "text_length": 100},
        enqueue_ts=0.0, yield_ts=0.1,
    )
    row = make_output_row(result)
    assert row["predicted_statements"] == []
    assert row["predicted_details"] == []
    assert row["error"] is None
```

**Step 2: Run, verify it fails.**

```bash
.venv/bin/pytest tests/scripts/test_extract_funding_job.py -v
```
Expected: FAIL — module not found.

**Step 3: Implement.**

Create `scripts/extract_funding_job.py` with PEP-723 header copied from `benchmark_hf_job.py:1-14` (same deps, same torch pin) and:

```python
def make_output_row(result):
    meta = result.metadata or {}
    return {
        "arxiv_id": meta.get("arxiv_id"),
        "doc_id": result.doc_id,
        "input_file": meta.get("input_file"),
        "row_idx": meta.get("row_idx"),
        "predicted_statements": [s.statement for s in result.statements],
        "predicted_details": [
            {
                "statement": s.statement,
                "score": float(s.score),
                "query": s.query,
                "paragraph_idx": s.paragraph_idx,
            }
            for s in result.statements
        ],
        "text_length": meta.get("text_length", 0),
        "latency_ms": (result.yield_ts - result.enqueue_ts) * 1000.0,
        "error": result.error,
    }
```

Make sure the PEP-723 block is intact and `from __future__ import annotations` follows the closing `# ///` line.

**Step 4: Run, verify pass.**

```bash
.venv/bin/pytest tests/scripts/test_extract_funding_job.py -v
```
Expected: 2 passed.

**Step 5: Commit.**

```bash
git add scripts/extract_funding_job.py tests/scripts/test_extract_funding_job.py
git commit -m "feat(extract-job): output-row builder with predictions and metadata carry-through"
```

---

## Task 2: Worker — argv parsing

**Files:**
- Modify: `scripts/extract_funding_job.py` (add `parse_args`)
- Modify: `tests/scripts/test_extract_funding_job.py` (add tests)

**Step 1: Write the failing test.**

```python
from scripts.extract_funding_job import parse_args


def test_parse_args_minimum():
    args = parse_args([
        "--input-repo", "cometadata/arxiv-latex-extract-full-text",
        "--input-files", "a.parquet,b.parquet",
        "--output-repo", "cometadata/arxiv-funding-statement-extractions",
        "--job-tag", "abc123",
    ])
    assert args.input_repo == "cometadata/arxiv-latex-extract-full-text"
    assert args.input_files == ["a.parquet", "b.parquet"]
    assert args.output_repo == "cometadata/arxiv-funding-statement-extractions"
    assert args.job_tag == "abc123"
    assert args.text_column == "text"
    assert args.id_column == "arxiv_id"  # or whatever Task 0 confirmed
    assert args.dtype == "bf16"
    assert args.batch_size == 512
    assert args.colbert_model == "lightonai/GTE-ModernColBERT-v1"


def test_parse_args_input_files_strips_whitespace():
    args = parse_args([
        "--input-repo", "x", "--input-files", " a.parquet , b.parquet ",
        "--output-repo", "y", "--job-tag", "z",
    ])
    assert args.input_files == ["a.parquet", "b.parquet"]
```

**Step 2: Run, verify fails.**

**Step 3: Implement** in `scripts/extract_funding_job.py`:

```python
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input-repo", required=True)
    p.add_argument("--input-files", required=True,
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    p.add_argument("--output-repo", required=True)
    p.add_argument("--job-tag", required=True)
    p.add_argument("--text-column", default="text")
    p.add_argument("--id-column", default="arxiv_id")  # adjust per Task 0
    p.add_argument("--colbert-model", default="lightonai/GTE-ModernColBERT-v1")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Skip CUDA probe; for local smoke tests only.")
    return p.parse_args(argv)
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

```bash
git commit -am "feat(extract-job): CLI argv with Tier-2 defaults baked in"
```

---

## Task 3: Worker — hub push helper (TDD with mocked HfApi)

**Files:**
- Modify: `scripts/extract_funding_job.py`
- Modify: `tests/scripts/test_extract_funding_job.py`

**Step 1: Write the failing test.**

```python
def test_push_parquet_writes_temp_then_uploads(tmp_path, monkeypatch):
    from scripts import extract_funding_job as mod

    captured = {}

    class FakeApi:
        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            import pyarrow.parquet as pq
            tbl = pq.read_table(path_or_fileobj)
            captured["rows"] = tbl.num_rows
            captured["cols"] = tbl.column_names
            captured["path_in_repo"] = path_in_repo
            captured["repo_id"] = repo_id

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())

    rows = [
        {"arxiv_id": "x", "doc_id": "x", "input_file": "f.parquet", "row_idx": 0,
         "predicted_statements": ["A"], "predicted_details": [
             {"statement": "A", "score": 1.0, "query": "q", "paragraph_idx": 0}
         ],
         "text_length": 10, "latency_ms": 1.0, "error": None},
    ]
    mod.push_parquet_to_hub(rows, repo_id="org/repo", path_in_repo="predictions/f.parquet",
                            staging_dir=tmp_path)
    assert captured["rows"] == 1
    assert captured["repo_id"] == "org/repo"
    assert captured["path_in_repo"] == "predictions/f.parquet"
    assert "predicted_statements" in captured["cols"]
    assert "predicted_details" in captured["cols"]
```

**Step 2: Run, verify fails.**

**Step 3: Implement.**

```python
from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq

def push_parquet_to_hub(rows, *, repo_id, path_in_repo, staging_dir):
    if not rows:
        # Still upload an empty parquet so the manifest sees the marker
        rows = []
    table = pa.Table.from_pylist(rows)
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    local_path = staging_dir / Path(path_in_repo).name
    pq.write_table(table, local_path, compression="zstd")
    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"add {path_in_repo}",
    )
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

```bash
git commit -am "feat(extract-job): push_parquet_to_hub helper with zstd compression"
```

---

## Task 4: Worker — assemble main() and per-file loop

This task wires Tasks 1–3 plus the CUDA probe and dtype patch from `benchmark_hf_job.py:293-307, 686-717` into a working `main()`. No test (covered by Task 5 smoke test).

**Files:**
- Modify: `scripts/extract_funding_job.py`

**Step 1: Copy the CUDA-init retry probe** from `benchmark_hf_job.py:686-717` verbatim (it implements the invariant from `feedback_hf_jobs_cuda_retry_probe.md`).

**Step 2: Copy `_apply_dtype_patch`** from `benchmark_hf_job.py:293-307` verbatim.

**Step 3: Write `main()`:**

```python
def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stdout)
    logger.info("job_tag=%s input_files=%d", args.job_tag, len(args.input_files))

    import torch
    if not args.allow_cpu:
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
            logger.error("CUDA not available after 120s — aborting (refusing CPU fallback).")
            return 2
        logger.info("torch=%s device=%s", torch.__version__, torch.cuda.get_device_name(0))

    _apply_dtype_patch(args.dtype)

    from datasets import load_dataset
    from funding_statement_extractor.config.loader import load_queries
    from funding_statement_extractor.statements.batch_extraction import (
        DocPayload, extract_funding_statements_batch,
    )

    queries = load_queries()
    logger.info("loaded %d queries", len(queries))

    staging = Path("/tmp/extract_funding_outputs")

    for input_file in args.input_files:
        t0 = time.perf_counter()
        logger.info("[start file=%s]", input_file)
        try:
            ds = load_dataset(args.input_repo, data_files=input_file,
                              split="train", streaming=True)

            def docs_iter():
                for row_idx, row in enumerate(ds):
                    text = row.get(args.text_column)
                    if not text:
                        continue
                    yield DocPayload(
                        doc_id=str(row.get(args.id_column) or row_idx),
                        text=text,
                        metadata={
                            "row_idx": row_idx,
                            "input_file": input_file,
                            "arxiv_id": row.get("arxiv_id"),
                            "text_length": len(text),
                            # Add any other Task 0-confirmed carry-through fields here
                        },
                    )

            output_rows = []
            for result in extract_funding_statements_batch(
                documents=docs_iter(), queries=queries,
                model_name=args.colbert_model,
                top_k=5, threshold=10.0,
                enable_paragraph_prefilter=True,
                regex_match_score_floor=11.0,
                paragraphs_per_batch=4096,
                encode_batch_size=args.batch_size,
                dtype=args.dtype,
            ):
                output_rows.append(make_output_row(result))

            out_path = f"predictions/{Path(input_file).name}"
            push_parquet_to_hub(output_rows, repo_id=args.output_repo,
                                path_in_repo=out_path, staging_dir=staging)
            elapsed = time.perf_counter() - t0
            # CRITICAL: this exact format is parsed by the orchestrator
            print(f"[done file={input_file} rows={len(output_rows)} elapsed_s={elapsed:.1f}]",
                  flush=True)
        except Exception as exc:
            logger.exception("[fail file=%s] %s", input_file, exc)
            # Continue to next file rather than failing the whole job —
            # the orchestrator will retry the unfinished file.

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Manually invoke `--help`** to confirm argparse is wired:

```bash
.venv/bin/python3 scripts/extract_funding_job.py --help
```
Expected: prints usage, no errors.

**Step 5: Commit.**

```bash
git commit -am "feat(extract-job): main() with CUDA probe, per-file streaming, done-line logging"
```

---

## Task 5: Worker — local CPU smoke test

Verify the worker end-to-end on a tiny input without launching an HF Job. Requires `--allow-cpu`.

**Files:** none (one-off run + manual verification).

**Step 1: Find the smallest input parquet** from Task 0's listing. Pick one with <50 rows if possible, else use `head` slicing in a temp file.

**Step 2: Create a throwaway test repo on hub.**

```bash
hf repo create test-funding-extraction-smoke --type dataset --private
```

Note the full repo id (e.g. `<your-user>/test-funding-extraction-smoke`).

**Step 3: Run the worker locally on CPU.**

```bash
.venv/bin/python3 scripts/extract_funding_job.py \
    --input-repo cometadata/arxiv-latex-extract-full-text \
    --input-files results-2026-04-24/<smallest-file>.parquet \
    --output-repo <your-user>/test-funding-extraction-smoke \
    --job-tag local-smoke \
    --dtype fp32 --batch-size 32 --allow-cpu
```

This will be slow (Mac CPU). Cap input rows by editing `docs_iter()` temporarily to `if row_idx >= 5: break` if needed for impatience — revert before commit.

Expected output:
- `[start file=...]` log
- A `[done file=... rows=... elapsed_s=...]` line
- Exit code 0

**Step 4: Verify on hub.**

```bash
.venv/bin/python3 - <<'PYEOF'
from datasets import load_dataset
ds = load_dataset("<your-user>/test-funding-extraction-smoke",
                  data_files="predictions/*.parquet", split="train")
print(ds)
print(ds[0])
PYEOF
```

Expected: rows printed with the schema from Task 1.

**Step 5: Delete the smoke repo (optional cleanup).**

```bash
hf repo delete <your-user>/test-funding-extraction-smoke --type dataset -y
```

No commit (no code change). If anything broke, fix in a follow-up commit.

---

## Task 6: Worker — single H200 smoke test

Same as Task 5 but inside an actual H200 HF Job. Catches issues with the PEP-723 deps, image, CUDA, and hub push under real conditions.

**Files:** none (one-off run).

**Step 1: Re-create the smoke test repo** (same as Task 5 step 2) if you deleted it.

**Step 2: Submit the job** using the `HfApi.run_job` snippet from `CLAUDE.md`. Save this as `/tmp/submit_smoke.py`:

```python
from huggingface_hub import HfApi
import base64, os, time

token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
api = HfApi()
SCRIPT = "scripts/extract_funding_job.py"
b64 = base64.b64encode(open(SCRIPT, "rb").read()).decode()
INPUT_FILE = "results-2026-04-24/<small-file>.parquet"
OUTPUT_REPO = "<your-user>/test-funding-extraction-smoke"
cmd = ["bash", "-c",
    "set -euxo pipefail && apt-get update -qq && apt-get install -y -qq git && "
    "pip install --quiet --root-user-action=ignore uv && "
    f"echo {b64} | base64 -d > /tmp/p.py && rm -rf /root/.cache/uv/environments-v2 && "
    "uv run /tmp/p.py "
    f"--input-repo cometadata/arxiv-latex-extract-full-text "
    f"--input-files {INPUT_FILE} "
    f"--output-repo {OUTPUT_REPO} "
    "--job-tag h200-smoke"]
for attempt in range(10):
    try:
        job = api.run_job(
            image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
            command=cmd, secrets={"HF_TOKEN": token},
            flavor="h200", timeout="30m")
        print(job.id); break
    except Exception as e:
        if "429" in str(e): time.sleep(60); continue
        raise
```

Run: `.venv/bin/python3 /tmp/submit_smoke.py` — captures the job_id.

**Step 3: Tail the job log** until completion:

```bash
hf jobs logs <job_id> --follow
```

Look for: cold-start trace, CUDA probe success, `[start file=...]`, `[done file=... rows=N elapsed_s=T]`, exit 0.

**Step 4: Verify output landed** (same as Task 5 step 4 — load via `datasets`).

**Step 5: Record observed `seconds_per_row`** = `elapsed_s / N`. This calibrates the orchestrator's seed estimate (Task 13). Note in `docs/plans/notes/throughput-h200-tier2.txt` and commit.

```bash
git add docs/plans/notes/throughput-h200-tier2.txt
git commit -m "docs: record h200 Tier-2 seconds-per-row from smoke job"
```

---

## Task 7: Orchestrator — manifest read/write helpers (TDD)

**Files:**
- Create: `scripts/orchestrate_extractions.py`
- Test: `tests/scripts/test_orchestrate_extractions.py`

**Step 1: Write the failing test.**

```python
# tests/scripts/test_orchestrate_extractions.py
import pyarrow.parquet as pq
from scripts.orchestrate_extractions import (
    Manifest, ManifestRow, write_manifest, read_manifest,
)


def test_roundtrip_manifest(tmp_path):
    rows = [
        ManifestRow(input_file="a.parquet", row_count=100, est_seconds=4.5,
                    status="pending", attempts=0, job_id=None,
                    assigned_at=None, completed_at=None,
                    output_path=None, last_error=None, worker_elapsed_s=None),
        ManifestRow(input_file="b.parquet", row_count=200, est_seconds=9.0,
                    status="done", attempts=1, job_id="job-xyz",
                    assigned_at=1000.0, completed_at=1100.0,
                    output_path="predictions/b.parquet",
                    last_error=None, worker_elapsed_s=99.0),
    ]
    path = tmp_path / "m.parquet"
    write_manifest(rows, path)
    out = read_manifest(path)
    assert len(out) == 2
    assert out[0].input_file == "a.parquet"
    assert out[1].status == "done"
    assert out[1].worker_elapsed_s == 99.0


def test_write_manifest_atomic(tmp_path):
    """Write must use temp+rename so an interrupted write can't corrupt."""
    path = tmp_path / "m.parquet"
    write_manifest([], path)
    # No .tmp file should remain
    assert not (tmp_path / "m.parquet.tmp").exists()
    assert path.exists()
```

**Step 2: Run, verify fails.**

```bash
.venv/bin/pytest tests/scripts/test_orchestrate_extractions.py -v
```

**Step 3: Implement.**

```python
# scripts/orchestrate_extractions.py
from __future__ import annotations
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ManifestRow:
    input_file: str
    row_count: int
    est_seconds: float
    status: str           # pending | assigned | done | failed
    attempts: int
    job_id: Optional[str]
    assigned_at: Optional[float]
    completed_at: Optional[float]
    output_path: Optional[str]
    last_error: Optional[str]
    worker_elapsed_s: Optional[float]


Manifest = List[ManifestRow]


def write_manifest(rows: Manifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist([asdict(r) for r in rows])
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def read_manifest(path: Path) -> Manifest:
    tbl = pq.read_table(path)
    return [ManifestRow(**r) for r in tbl.to_pylist()]
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

```bash
git add scripts/orchestrate_extractions.py tests/scripts/test_orchestrate_extractions.py
git commit -m "feat(orchestrate): manifest dataclass + atomic parquet read/write"
```

---

## Task 8: Orchestrator — pure helpers: batch picker, EMA, log parser (TDD)

Three small pure functions in one task because they're trivial.

**Files:**
- Modify: `scripts/orchestrate_extractions.py`
- Modify: `tests/scripts/test_orchestrate_extractions.py`

**Step 1: Write the failing tests.**

```python
from scripts.orchestrate_extractions import (
    pick_next_batch, update_ema, parse_done_line,
)


def test_pick_batch_packs_until_target():
    rows = [
        ManifestRow("a.parquet", 1000, 50.0, "pending", 0, None, None, None, None, None, None),
        ManifestRow("b.parquet", 2000, 100.0, "pending", 0, None, None, None, None, None, None),
        ManifestRow("c.parquet", 3000, 150.0, "pending", 0, None, None, None, None, None, None),
        ManifestRow("d.parquet", 9999, 999.0, "done", 0, None, None, None, None, None, None),
    ]
    batch = pick_next_batch(rows, target_seconds=200.0, max_files=50)
    # Greedy desc by est_seconds: c=150 -> total 150, then b=100 would push to 250 > 200
    # but we always include at least one, so batch = [c]
    # Then on next call rows still has b and a, so we'd pick b (100) and a (50).
    assert [r.input_file for r in batch] == ["c.parquet"]


def test_pick_batch_handles_huge_single_file():
    rows = [
        ManifestRow("huge.parquet", 100000, 5000.0, "pending", 0, None, None, None, None, None, None),
    ]
    batch = pick_next_batch(rows, target_seconds=1800.0, max_files=50)
    assert [r.input_file for r in batch] == ["huge.parquet"]


def test_pick_batch_skips_non_pending():
    rows = [
        ManifestRow("a.parquet", 100, 5.0, "done", 0, None, None, None, None, None, None),
        ManifestRow("b.parquet", 100, 5.0, "assigned", 0, None, None, None, None, None, None),
        ManifestRow("c.parquet", 100, 5.0, "pending", 0, None, None, None, None, None, None),
    ]
    batch = pick_next_batch(rows, target_seconds=100.0, max_files=50)
    assert [r.input_file for r in batch] == ["c.parquet"]


def test_pick_batch_caps_at_max_files():
    rows = [ManifestRow(f"f{i}.parquet", 1, 0.001, "pending", 0, None, None, None, None, None, None)
            for i in range(100)]
    batch = pick_next_batch(rows, target_seconds=999999.0, max_files=10)
    assert len(batch) == 10


def test_update_ema():
    assert update_ema(prev=None, sample=0.05, alpha=0.3) == 0.05
    # 0.7 * 0.05 + 0.3 * 0.10 = 0.035 + 0.03 = 0.065
    assert abs(update_ema(prev=0.05, sample=0.10, alpha=0.3) - 0.065) < 1e-9


def test_parse_done_line():
    line = "[done file=results-2026-04-24/shard-0.parquet rows=12345 elapsed_s=678.9]"
    parsed = parse_done_line(line)
    assert parsed == {"file": "results-2026-04-24/shard-0.parquet",
                      "rows": 12345, "elapsed_s": 678.9}


def test_parse_done_line_returns_none_on_no_match():
    assert parse_done_line("INFO some other log line") is None
```

**Step 2: Run, verify fails.**

**Step 3: Implement.**

```python
import re

_DONE_RE = re.compile(r"\[done file=(?P<file>\S+) rows=(?P<rows>\d+) elapsed_s=(?P<elapsed>[\d.]+)\]")


def pick_next_batch(rows, *, target_seconds, max_files):
    pending = [r for r in rows if r.status == "pending"]
    pending.sort(key=lambda r: r.est_seconds, reverse=True)
    batch = []
    total = 0.0
    for r in pending:
        if not batch:
            batch.append(r)
            total += r.est_seconds
            continue
        if total + r.est_seconds > target_seconds:
            break
        if len(batch) >= max_files:
            break
        batch.append(r)
        total += r.est_seconds
    return batch


def update_ema(prev, sample, alpha):
    if prev is None:
        return sample
    return (1 - alpha) * prev + alpha * sample


def parse_done_line(line):
    m = _DONE_RE.search(line)
    if not m:
        return None
    return {
        "file": m.group("file"),
        "rows": int(m.group("rows")),
        "elapsed_s": float(m.group("elapsed")),
    }
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

```bash
git commit -am "feat(orchestrate): batch picker, EMA, done-line parser"
```

---

## Task 9: Orchestrator — list inputs + read row counts from parquet footers

**Files:**
- Modify: `scripts/orchestrate_extractions.py`
- Modify: `tests/scripts/test_orchestrate_extractions.py`

**Step 1: Write the failing test** (uses a real local parquet, no hub).

```python
def test_read_row_count_from_footer(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": list(range(123))}), p)
    from scripts.orchestrate_extractions import read_row_count_local
    assert read_row_count_local(p) == 123
```

**Step 2: Run, verify fails.**

**Step 3: Implement two functions.**

```python
def read_row_count_local(path):
    return pq.ParquetFile(str(path)).metadata.num_rows


def list_input_files_with_row_counts(repo_id, subdir):
    """Returns list of (path_within_repo, row_count) tuples for all parquets in repo_id/subdir."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    base = f"datasets/{repo_id}/{subdir}".rstrip("/")
    files = sorted(p for p in fs.ls(base, detail=False) if p.endswith(".parquet"))
    out = []
    for f in files:
        with fs.open(f) as fh:
            rc = pq.ParquetFile(fh).metadata.num_rows
        # Strip the "datasets/<repo_id>/" prefix to get the in-repo path
        in_repo = f.split(f"datasets/{repo_id}/", 1)[1]
        out.append((in_repo, rc))
    return out
```

**Step 4: Run pytest, verify the local test passes.**

**Step 5: Manually smoke-test the hub call** against a tiny known dataset (no test, just verify it works):

```bash
.venv/bin/python3 -c "
from scripts.orchestrate_extractions import list_input_files_with_row_counts
files = list_input_files_with_row_counts('cometadata/arxiv-latex-extract-full-text', 'results-2026-04-24')
print(f'n_files={len(files)} total_rows={sum(rc for _, rc in files):,}')
print(files[:3])
"
```

Expected: prints file count + total rows + a few sample tuples in <60s.

**Step 6: Commit.**

```bash
git commit -am "feat(orchestrate): footer-only row-count reader + input file lister"
```

---

## Task 10: Orchestrator — manifest seed + reconcile against output repo

**Files:**
- Modify: `scripts/orchestrate_extractions.py`
- Modify: `tests/scripts/test_orchestrate_extractions.py`

**Step 1: Write the failing test for seed merge logic.**

```python
def test_seed_merges_new_files_preserves_existing():
    from scripts.orchestrate_extractions import seed_or_merge_manifest

    existing = [
        ManifestRow("a.parquet", 100, 4.5, "done", 1, "j1", 1.0, 2.0,
                    "predictions/a.parquet", None, 90.0),
    ]
    listing = [("a.parquet", 100), ("b.parquet", 200)]
    merged = seed_or_merge_manifest(existing, listing, seconds_per_row=0.045)
    by_file = {r.input_file: r for r in merged}
    assert by_file["a.parquet"].status == "done"  # preserved
    assert by_file["a.parquet"].attempts == 1     # preserved
    assert by_file["b.parquet"].status == "pending"
    assert by_file["b.parquet"].est_seconds == 200 * 0.045


def test_reconcile_marks_done_when_output_exists():
    from scripts.orchestrate_extractions import reconcile_against_outputs

    rows = [
        ManifestRow("a.parquet", 100, 4.5, "assigned", 1, "j1", 1.0, None,
                    None, None, None),
        ManifestRow("b.parquet", 200, 9.0, "pending", 0, None, None, None,
                    None, None, None),
    ]
    # Pretend output repo has predictions/a.parquet
    existing_outputs = {"predictions/a.parquet"}
    out = reconcile_against_outputs(rows, existing_outputs)
    by = {r.input_file: r for r in out}
    assert by["a.parquet"].status == "done"
    assert by["a.parquet"].output_path == "predictions/a.parquet"
    assert by["a.parquet"].completed_at is not None
    assert by["b.parquet"].status == "pending"  # untouched
```

**Step 2: Run, verify fails.**

**Step 3: Implement.**

```python
import time as _time

def seed_or_merge_manifest(existing, listing, *, seconds_per_row):
    by_file = {r.input_file: r for r in existing}
    for input_file, row_count in listing:
        if input_file in by_file:
            continue
        by_file[input_file] = ManifestRow(
            input_file=input_file, row_count=row_count,
            est_seconds=row_count * seconds_per_row,
            status="pending", attempts=0, job_id=None,
            assigned_at=None, completed_at=None,
            output_path=None, last_error=None, worker_elapsed_s=None,
        )
    return [by_file[k] for k in sorted(by_file)]


def reconcile_against_outputs(rows, existing_outputs):
    out = []
    for r in rows:
        candidate = f"predictions/{Path(r.input_file).name}"
        if candidate in existing_outputs and r.status != "done":
            r = ManifestRow(**{**asdict(r), "status": "done",
                               "output_path": candidate,
                               "completed_at": _time.time()})
        out.append(r)
    return out


def list_existing_outputs(repo_id):
    """Returns set of in-repo paths like {'predictions/x.parquet', ...}."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    base = f"datasets/{repo_id}/predictions"
    try:
        files = fs.ls(base, detail=False)
    except FileNotFoundError:
        return set()
    return {f.split(f"datasets/{repo_id}/", 1)[1] for f in files if f.endswith(".parquet")}
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

```bash
git commit -am "feat(orchestrate): manifest seed/merge + reconcile against output repo"
```

---

## Task 11: Orchestrator — job submitter with 429 retry

**Files:**
- Modify: `scripts/orchestrate_extractions.py`
- Modify: `tests/scripts/test_orchestrate_extractions.py`

**Step 1: Write the failing test** (mocks `HfApi.run_job`).

```python
def test_submit_job_retries_429(monkeypatch):
    from scripts import orchestrate_extractions as mod
    calls = {"n": 0, "sleeps": []}

    class FakeJob:
        id = "job-fake"

    class FakeApi:
        def run_job(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("HTTP 429 too many requests")
            return FakeJob()

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(mod.time, "sleep", lambda s: calls["sleeps"].append(s))

    job_id = mod.submit_h200_job(
        script_path="scripts/extract_funding_job.py",
        worker_argv=["--input-repo", "x", "--input-files", "f.parquet",
                     "--output-repo", "y", "--job-tag", "z"],
        token="t", timeout="2h",
    )
    assert job_id == "job-fake"
    assert calls["n"] == 3
    assert calls["sleeps"] == [60, 60]


def test_submit_job_gives_up_after_max_retries(monkeypatch):
    from scripts import orchestrate_extractions as mod

    class FakeApi:
        def run_job(self, **kw):
            raise RuntimeError("HTTP 429")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    import pytest
    with pytest.raises(RuntimeError):
        mod.submit_h200_job(
            script_path="scripts/extract_funding_job.py",
            worker_argv=["--input-repo", "x", "--input-files", "f.parquet",
                         "--output-repo", "y", "--job-tag", "z"],
            token="t", timeout="2h", max_retries=3,
        )
```

**Step 2: Run, verify fails.**

**Step 3: Implement.**

```python
import base64, time
from huggingface_hub import HfApi

IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
FLAVOR = "h200"


def submit_h200_job(*, script_path, worker_argv, token, timeout="2h", max_retries=10):
    b64 = base64.b64encode(Path(script_path).read_bytes()).decode()
    argv_str = " ".join(worker_argv)
    cmd = ["bash", "-c",
           "set -euxo pipefail && apt-get update -qq && apt-get install -y -qq git && "
           "pip install --quiet --root-user-action=ignore uv && "
           f"echo {b64} | base64 -d > /tmp/p.py && rm -rf /root/.cache/uv/environments-v2 && "
           f"uv run /tmp/p.py {argv_str}"]
    api = HfApi()
    last_exc = None
    for attempt in range(max_retries):
        try:
            job = api.run_job(image=IMAGE, command=cmd,
                              secrets={"HF_TOKEN": token},
                              flavor=FLAVOR, timeout=timeout)
            return job.id
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc):
                time.sleep(60)
                continue
            raise
    raise last_exc
```

**Step 4: Run, verify pass.**

**Step 5: Commit.**

```bash
git commit -am "feat(orchestrate): submit_h200_job with 429 backoff"
```

---

## Task 12: Orchestrator — job state poller (status + log tail)

This wraps `hf jobs inspect` and `hf jobs logs`. Use the `huggingface_hub` Python API rather than shelling out to `hf` (cleaner, testable).

**Files:**
- Modify: `scripts/orchestrate_extractions.py`
- Modify: `tests/scripts/test_orchestrate_extractions.py`

**Step 1: Confirm the API.** Open a Python REPL and check what `HfApi` exposes for job inspect/logs:

```bash
.venv/bin/python3 -c "from huggingface_hub import HfApi; help(HfApi.inspect_job)" 2>&1 | head -30
.venv/bin/python3 -c "from huggingface_hub import HfApi; help(HfApi.fetch_job_logs)" 2>&1 | head -30
```

If the methods are named differently in the installed version, adjust the wrapper below. (As of `huggingface_hub>=0.25.0`, methods include `inspect_job` and an iterator-returning `fetch_job_logs`.)

**Step 2: Write the failing test** (mocks the API).

```python
def test_poll_job_state_parses_logs_and_status(monkeypatch):
    from scripts import orchestrate_extractions as mod

    class FakeApi:
        def inspect_job(self, job_id):
            return SimpleNamespace(status=SimpleNamespace(stage="COMPLETED"),
                                   id=job_id)
        def fetch_job_logs(self, job_id):
            yield SimpleNamespace(data="INFO booting...")
            yield SimpleNamespace(data="[done file=a.parquet rows=10 elapsed_s=5.0]")
            yield SimpleNamespace(data="[done file=b.parquet rows=20 elapsed_s=10.0]")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    state = mod.poll_job_state("job-x")
    assert state.stage == "COMPLETED"
    assert state.done_files == [
        {"file": "a.parquet", "rows": 10, "elapsed_s": 5.0},
        {"file": "b.parquet", "rows": 20, "elapsed_s": 10.0},
    ]
    assert state.last_log_ts is not None
```

**Step 3: Run, verify fails.**

**Step 4: Implement.**

```python
@dataclass
class JobState:
    stage: str           # e.g. RUNNING / COMPLETED / ERROR / CANCELED
    done_files: list     # parsed [done ...] events
    last_log_ts: Optional[float]


def poll_job_state(job_id):
    api = HfApi()
    info = api.inspect_job(job_id)
    stage = getattr(info.status, "stage", None) or getattr(info, "status", "UNKNOWN")
    done = []
    last_ts = None
    try:
        for entry in api.fetch_job_logs(job_id):
            line = getattr(entry, "data", "") or ""
            parsed = parse_done_line(line)
            if parsed:
                done.append(parsed)
            last_ts = _time.time()
    except Exception:
        pass  # transient log fetch failures are fine; orchestrator will retry next loop
    return JobState(stage=str(stage), done_files=done, last_log_ts=last_ts)
```

**Step 5: Run, verify pass.**

**Step 6: Commit.**

```bash
git commit -am "feat(orchestrate): poll_job_state via HfApi inspect+fetch_job_logs"
```

---

## Task 13: Orchestrator — main loop

Wires Tasks 7–12. Long function, no per-step TDD (covered by Task 14 dry-run). Aim for clarity over cleverness.

**Files:**
- Modify: `scripts/orchestrate_extractions.py`

**Step 1: Add CLI argv parsing.**

```python
def parse_orch_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input-repo", default="cometadata/arxiv-latex-extract-full-text")
    p.add_argument("--input-subdir", default="results-2026-04-24")
    p.add_argument("--output-repo", default="cometadata/arxiv-funding-statement-extractions")
    p.add_argument("--manifest", default="manifests/arxiv-extractions-2026-04-24.parquet")
    p.add_argument("--max-in-flight", type=int, default=8)
    p.add_argument("--target-seconds", type=int, default=5400)  # 90 min
    p.add_argument("--max-files-per-batch", type=int, default=50)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--stuck-min", type=int, default=15)
    p.add_argument("--poll-interval-s", type=int, default=60)
    p.add_argument("--seed-seconds-per-row", type=float, default=0.045)
    p.add_argument("--dry-run", action="store_true",
                   help="Don't actually submit jobs; print what would happen.")
    return p.parse_args(argv)
```

**Step 2: Write `main()`.** Sketch (fill in details from the design doc):

```python
def main(argv=None) -> int:
    args = parse_orch_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stdout)

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = read_manifest(manifest_path)
        logger.info("loaded manifest with %d rows", len(manifest))
    else:
        manifest = []

    logger.info("listing inputs from %s/%s", args.input_repo, args.input_subdir)
    listing = list_input_files_with_row_counts(args.input_repo, args.input_subdir)
    logger.info("found %d input files (%d total rows)",
                len(listing), sum(rc for _, rc in listing))
    manifest = seed_or_merge_manifest(manifest, listing,
                                      seconds_per_row=args.seed_seconds_per_row)

    logger.info("listing existing outputs in %s", args.output_repo)
    existing = list_existing_outputs(args.output_repo)
    manifest = reconcile_against_outputs(manifest, existing)
    write_manifest(manifest, manifest_path)

    token = os.environ.get("HF_TOKEN") or open(
        os.path.expanduser("~/.cache/huggingface/token")).read().strip()

    in_flight = {}     # job_id -> {"files": [input_file...], "submitted_at": ts, "last_log_ts": ts}
    seconds_per_row_ema = None

    def manifest_by_file():
        return {r.input_file: r for r in manifest}

    while True:
        # --- Poll in-flight jobs ---
        for job_id in list(in_flight):
            state = poll_job_state(job_id)
            mb = manifest_by_file()
            for ev in state.done_files:
                row = mb.get(ev["file"])
                if row and row.status == "assigned" and row.job_id == job_id:
                    row.status = "done"
                    row.completed_at = _time.time()
                    row.output_path = f"predictions/{Path(ev['file']).name}"
                    row.worker_elapsed_s = ev["elapsed_s"]
                    sample = ev["elapsed_s"] / max(ev["rows"], 1)
                    seconds_per_row_ema = update_ema(seconds_per_row_ema, sample, 0.3)
                    logger.info("[done] file=%s rows=%d elapsed=%.1fs ema=%.4fs/row",
                                ev["file"], ev["rows"], ev["elapsed_s"],
                                seconds_per_row_ema)
            if state.last_log_ts:
                in_flight[job_id]["last_log_ts"] = state.last_log_ts

            # Stuck detection
            silent_min = (_time.time() - in_flight[job_id]["last_log_ts"]) / 60
            if silent_min > args.stuck_min and state.stage == "RUNNING":
                logger.warning("job %s silent for %.1fmin — cancelling", job_id, silent_min)
                if not args.dry_run:
                    HfApi().cancel_job(job_id)
                state.stage = "CANCELED"  # fall through to release

            if state.stage in ("COMPLETED", "ERROR", "CANCELED"):
                # Release files that didn't reach 'done'
                for f in in_flight[job_id]["files"]:
                    row = mb[f]
                    if row.status == "assigned" and row.job_id == job_id:
                        row.attempts += 1
                        if row.attempts >= args.max_attempts:
                            row.status = "failed"
                            row.last_error = f"job {job_id} stage={state.stage}"
                            logger.error("[failed] file=%s after %d attempts", f, row.attempts)
                        else:
                            row.status = "pending"
                            row.job_id = None
                            row.assigned_at = None
                            row.last_error = f"job {job_id} stage={state.stage}"
                            logger.warning("[released] file=%s attempts=%d", f, row.attempts)
                del in_flight[job_id]

        # Recompute estimates with EMA
        if seconds_per_row_ema:
            for r in manifest:
                if r.status == "pending":
                    r.est_seconds = r.row_count * seconds_per_row_ema

        # --- Refill ---
        while len(in_flight) < args.max_in_flight:
            batch = pick_next_batch(manifest,
                                    target_seconds=args.target_seconds,
                                    max_files=args.max_files_per_batch)
            if not batch:
                break
            job_tag = f"orch-{int(_time.time())}-{len(in_flight)}"
            input_files_arg = ",".join(r.input_file for r in batch)
            worker_argv = [
                "--input-repo", args.input_repo,
                "--input-files", input_files_arg,
                "--output-repo", args.output_repo,
                "--job-tag", job_tag,
            ]
            if args.dry_run:
                logger.info("[dry-run] would submit batch (%d files, est %.1fs): %s",
                            len(batch), sum(r.est_seconds for r in batch),
                            [r.input_file for r in batch])
                # Mark as assigned with a fake id so the loop can exit
                fake_id = f"dry-{job_tag}"
                for r in batch:
                    r.status = "assigned"; r.job_id = fake_id; r.assigned_at = _time.time()
                in_flight[fake_id] = {"files": [r.input_file for r in batch],
                                      "submitted_at": _time.time(),
                                      "last_log_ts": _time.time()}
                # In dry-run, immediately mark them done so we exit
                for r in batch:
                    r.status = "done"; r.completed_at = _time.time()
                    r.output_path = f"predictions/{Path(r.input_file).name}"
                del in_flight[fake_id]
                continue

            try:
                job_id = submit_h200_job(
                    script_path="scripts/extract_funding_job.py",
                    worker_argv=worker_argv, token=token, timeout="2h")
            except Exception as exc:
                logger.error("submit failed: %s — leaving %d files pending", exc, len(batch))
                break  # try again next loop
            for r in batch:
                r.status = "assigned"; r.job_id = job_id; r.assigned_at = _time.time()
            in_flight[job_id] = {"files": [r.input_file for r in batch],
                                 "submitted_at": _time.time(),
                                 "last_log_ts": _time.time()}
            logger.info("[submitted] job=%s files=%d est_total=%.0fs",
                        job_id, len(batch), sum(r.est_seconds for r in batch))

        write_manifest(manifest, manifest_path)

        n_pending = sum(1 for r in manifest if r.status == "pending")
        n_assigned = sum(1 for r in manifest if r.status == "assigned")
        n_done = sum(1 for r in manifest if r.status == "done")
        n_failed = sum(1 for r in manifest if r.status == "failed")
        logger.info("status: pending=%d assigned=%d done=%d failed=%d in_flight=%d",
                    n_pending, n_assigned, n_done, n_failed, len(in_flight))
        if n_pending == 0 and n_assigned == 0 and not in_flight:
            logger.info("all done")
            break

        time.sleep(args.poll_interval_s)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 3: Verify it imports and `--help` works.**

```bash
.venv/bin/python3 scripts/orchestrate_extractions.py --help
```

**Step 4: Commit.**

```bash
git commit -am "feat(orchestrate): main loop with submit/poll/release/refill/EMA-rebalance"
```

---

## Task 14: Orchestrator — dry-run end-to-end

**Files:** none (one-off run).

**Step 1: Run with `--dry-run`** against the real input dataset:

```bash
rm -f manifests/arxiv-extractions-2026-04-24.parquet  # start fresh
.venv/bin/python3 scripts/orchestrate_extractions.py --dry-run
```

Expected:
- Lists input files (should match Task 0 count)
- Reconciles against output repo (probably 0 matches if untouched)
- Logs `[dry-run] would submit batch ...` lines for every file
- Manifest gets written and shows all rows as `done` (dry-run shortcut)
- Exits 0

**Step 2: Inspect manifest.**

```bash
.venv/bin/python3 -c "
from scripts.orchestrate_extractions import read_manifest
m = read_manifest('manifests/arxiv-extractions-2026-04-24.parquet')
print(f'n={len(m)}')
print(f'statuses: {set(r.status for r in m)}')
print(f'sample: {m[0]}')
"
```

**Step 3: Re-run dry-run (idempotency check).**

```bash
.venv/bin/python3 scripts/orchestrate_extractions.py --dry-run
```

Expected: sees all rows already `done`, exits immediately with `all done`.

**Step 4: Reset and commit.**

```bash
rm manifests/arxiv-extractions-2026-04-24.parquet
# no commit
```

---

## Task 15: Create output dataset repo

**Files:** none.

**Step 1: Create the empty private repo.**

```bash
hf repo create arxiv-funding-statement-extractions --type dataset --private --org cometadata
```

(Skip if it already exists. If you don't have org write access, create under your user and update `--output-repo` defaults.)

**Step 2: Verify.**

```bash
hf repo info cometadata/arxiv-funding-statement-extractions --type dataset
```

No commit (no code change).

---

## Task 16: Live single-batch test

Run the orchestrator for one real H200 job (forced to one small batch) to validate the full loop end-to-end.

**Files:** none (one-off run).

**Step 1: Pick a tiny subset.** Easiest path: edit `seed_or_merge_manifest` to limit to the first 1 input file via a temp env-var hack, OR seed manifest manually:

```bash
.venv/bin/python3 -c "
from scripts.orchestrate_extractions import (
    list_input_files_with_row_counts, seed_or_merge_manifest, write_manifest
)
listing = list_input_files_with_row_counts(
    'cometadata/arxiv-latex-extract-full-text', 'results-2026-04-24')
listing = listing[:1]  # JUST ONE FILE
m = seed_or_merge_manifest([], listing, seconds_per_row=0.045)
write_manifest(m, 'manifests/arxiv-extractions-2026-04-24.parquet')
print(m[0])
"
```

**Step 2: Run orchestrator with `--max-in-flight 1` and let it complete.**

```bash
.venv/bin/python3 scripts/orchestrate_extractions.py --max-in-flight 1
```

Expected:
- Submits one H200 job
- Polls every 60s, eventually sees `[done file=...]` log line
- Marks the row done, writes manifest, exits

This will take ~10-30 minutes depending on the file size. Tail the log; if anything goes wrong, kill with Ctrl-C, fix, and re-run (manifest preserves state).

**Step 3: Verify output landed.**

```bash
.venv/bin/python3 -c "
from huggingface_hub import HfFileSystem
print(HfFileSystem().ls('datasets/cometadata/arxiv-funding-statement-extractions/predictions'))
"
```

**Step 4: Reset manifest for full run.**

```bash
rm manifests/arxiv-extractions-2026-04-24.parquet
```

(Note: don't delete the output parquet — Task 17 will reconcile and skip it on resume.)

No commit.

---

## Task 17: Full launch

**Files:** none (or commit any small fixes uncovered by Task 16).

**Step 1: Launch.**

```bash
nohup .venv/bin/python3 scripts/orchestrate_extractions.py \
    > /tmp/orch.log 2>&1 &
echo $! > /tmp/orch.pid
```

Or run in the foreground in a `tmux` / `screen` session — your call.

**Step 2: Monitor.**

```bash
tail -f /tmp/orch.log
```

Watch for: 8 jobs in flight, EMA stabilizing, `[done] file=...` lines accumulating, no repeated retries on the same file.

**Step 3: Throughput sanity check** after the first ~30 min:

```bash
.venv/bin/python3 -c "
from scripts.orchestrate_extractions import read_manifest
m = read_manifest('manifests/arxiv-extractions-2026-04-24.parquet')
done = [r for r in m if r.status == 'done']
total_rows = sum(r.row_count for r in done)
print(f'done={len(done)}/{len(m)} files, {total_rows:,} rows')
elapsed = [r.worker_elapsed_s for r in done if r.worker_elapsed_s]
if elapsed:
    print(f'mean elapsed/file: {sum(elapsed)/len(elapsed):.0f}s')
"
```

If observed throughput diverges sharply from the Task 6 calibration, kill the orchestrator and tune `--target-seconds` (smaller = shorter jobs, faster turnaround on tail; larger = lower overhead).

**Step 4: Iterate until complete.** Orchestrator self-exits when `n_pending == n_assigned == 0`.

**Step 5: Final sanity.**

```bash
.venv/bin/python3 -c "
from scripts.orchestrate_extractions import read_manifest
m = read_manifest('manifests/arxiv-extractions-2026-04-24.parquet')
from collections import Counter
print(Counter(r.status for r in m))
print('failed files:')
for r in m:
    if r.status == 'failed':
        print(f'  {r.input_file}: {r.last_error}')
"
```

If any `failed`, investigate via `hf jobs logs <last job_id>` for that row and decide whether to manually reset to `pending` for retry.

**Step 6: Commit any fixes/tweaks made during the run.**

---

## Task 18: Final summary + README on output repo

**Files:**
- Modify: `scripts/orchestrate_extractions.py` (add summary writer, called at end of `main()`)

**Step 1: Implement `write_run_summary`** that writes:
- Local: `manifests/arxiv-extractions-2026-04-24-summary.json`
- Hub: `run_metadata/summary.json` and `run_metadata/manifest-snapshot.parquet` to the output repo
- Hub: `README.md` with the auto-generated content from the design doc.

Code sketch:

```python
def write_run_summary(manifest, output_repo, manifest_path):
    from collections import Counter
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(manifest),
        "status_counts": dict(Counter(r.status for r in manifest)),
        "total_rows": sum(r.row_count for r in manifest if r.status == "done"),
        "total_worker_seconds": sum(r.worker_elapsed_s or 0 for r in manifest),
        "estimated_cost_usd": sum(r.worker_elapsed_s or 0 for r in manifest) / 3600.0 * 5.0,
        "failed_files": [
            {"input_file": r.input_file, "last_error": r.last_error}
            for r in manifest if r.status == "failed"
        ],
    }
    summary_path = manifest_path.parent / (manifest_path.stem + "-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    api = HfApi()
    api.upload_file(path_or_fileobj=str(summary_path),
                    path_in_repo="run_metadata/summary.json",
                    repo_id=output_repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(manifest_path),
                    path_in_repo="run_metadata/manifest-snapshot.parquet",
                    repo_id=output_repo, repo_type="dataset")
    readme = _render_readme(summary)
    readme_tmp = manifest_path.parent / "README.md"
    readme_tmp.write_text(readme)
    api.upload_file(path_or_fileobj=str(readme_tmp),
                    path_in_repo="README.md",
                    repo_id=output_repo, repo_type="dataset")


def _render_readme(summary):
    return f"""# arxiv-funding-statement-extractions

Funding-statement extractions over `cometadata/arxiv-latex-extract-full-text/results-2026-04-24/`.

- Extractor: `funding_statement_extractor` @ `statement-only-extraction`
- Model: `lightonai/GTE-ModernColBERT-v1`
- Config: Tier 2 (paragraph prefilter, regex floor 11.0, top_k=5, threshold=10.0)
- Hardware: H200 bf16, batch_size 512
- Files processed: {summary['n_files']}
- Status: {summary['status_counts']}
- Total rows: {summary['total_rows']:,}
- Total worker seconds: {summary['total_worker_seconds']:.0f}
- Est cost @ $5/hr: ${summary['estimated_cost_usd']:.2f}
- Completed: {summary['completed_at']}

## Schema

Per row in `predictions/*.parquet`:
- `arxiv_id`, `doc_id`, `input_file`, `row_idx`
- `predicted_statements`: list[str]
- `predicted_details`: list[struct{{statement, score, query, paragraph_idx}}]
- `text_length`, `latency_ms`, `error`
"""
```

**Step 2: Wire it into `main()`** just before `return 0`:

```python
write_run_summary(manifest, args.output_repo, manifest_path)
```

**Step 3: Run it once** (after Task 17 completes, or manually with a finished manifest):

```bash
.venv/bin/python3 scripts/orchestrate_extractions.py
```

Should detect everything is done, write summary, push to hub, exit.

**Step 4: Verify** the output repo has `README.md`, `run_metadata/summary.json`, `run_metadata/manifest-snapshot.parquet`.

**Step 5: Commit.**

```bash
git commit -am "feat(orchestrate): final run summary + README on output repo"
```

---

## Notes for the executing engineer

- **Frequent commits.** Each task ends with a commit; resist the urge to batch.
- **The CUDA retry probe is non-negotiable** — without it, jobs silently fall back to CPU on H200 (the host-side container can take 30s to expose CUDA). See `~/.claude/projects/.../memory/feedback_hf_jobs_cuda_retry_probe.md`.
- **The torch pin in the worker's PEP-723 header is non-negotiable** — `torch>=2.5.0,<2.7.0` ensures cu124 wheels which the H200 host driver supports. cu128 wheels (torch 2.7+) silently fail and fall back to CPU.
- **Don't shorten timeouts to "just test"** — H200 cold start eats 5-8 min before any real work. A 30 min timeout on a real job is right at the edge.
- **Manifest is the truth.** If you ever want to re-do a file, edit the manifest (set status back to `pending`, clear `attempts`, `job_id`, `last_error`). The orchestrator picks it up on next loop.
- **If `huggingface_hub` API names differ** from `inspect_job` / `fetch_job_logs`, check the installed version and adjust Task 12 accordingly. The CLI commands `hf jobs inspect <id>` and `hf jobs logs <id>` always work as a fallback.
