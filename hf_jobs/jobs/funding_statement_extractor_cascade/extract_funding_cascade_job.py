#!/usr/bin/env python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "funding-statement-extractor-cascade @ git+https://github.com/cometadata/funding-metadata-enrichment.git@funding-statement-extractor-cascade#subdirectory=funding_statement_extractor_cascade",
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
import os
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
    pa.field("text_source", pa.string()),
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
    p.add_argument("--text-column", dest="text_columns", default=["vlm_markdown"],
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="Comma-separated text column(s); one prediction set per "
                        "column, tagged in the output `text_source` field.")
    p.add_argument("--id-column", default="doc_id")
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
        "text_source": meta.get("text_source"),
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


_RETRY_HINT_RE = re.compile(
    r"retry.*?in\s+(?:about\s+)?(\d+)\s+(second|minute|hour)s?", re.IGNORECASE
)


def _parse_retry_hint_seconds(message: str) -> Optional[float]:
    """Extract HF's "retry in X minute/hour" hint from a 429 error message."""
    m = _RETRY_HINT_RE.search(message or "")
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    return n * {"second": 1, "minute": 60, "hour": 3600}[unit]


def preflight_commit_check(repo_id, *, job_tag, max_wait_s=1800.0, staging_dir=None):
    """Probe hub commit availability before doing expensive GPU work.

    Stages a tiny status JSON and tries to commit it. Honours HF's
    "retry in N minute/hour" hint from 429 responses (capped at max_wait_s).
    Raises RuntimeError if the hub remains unwilling after one full retry
    cycle, so the worker can exit cleanly without burning GPU time.
    """
    import json
    from datetime import datetime, timezone
    staging_dir = Path(staging_dir or "/tmp/extract_funding_outputs")
    staging_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_tag": job_tag,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "preflight": True,
    }
    local_path = staging_dir / f"_preflight_{job_tag}.json"
    local_path.write_text(json.dumps(payload))
    api = HfApi()
    last_exc: Optional[Exception] = None
    for attempt in range(8):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"worker_status/{job_tag}.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"worker {job_tag} preflight",
            )
            logger.info("preflight commit succeeded on attempt %d", attempt + 1)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if "429" not in str(exc):
                raise
            hint = _parse_retry_hint_seconds(str(exc))
            wait_s = min(hint or 60.0, max_wait_s)
            logger.warning(
                "preflight hit 429 (attempt %d/8); HF hint=%s, sleeping %.0fs",
                attempt + 1, hint, wait_s,
            )
            time.sleep(wait_s)
    raise RuntimeError(
        "preflight commit still 429 after exhausting retry budget"
    ) from last_exc


def commit_staged_batch(staged, *, repo_id, batch_index, max_retries=5, retry_sleep_s=60.0):
    """Commit a batch of staged files in a single hub commit.

    ``staged`` is a list of dicts with keys ``path_in_repo`` and ``local_path``.
    Retries on HTTP 429 up to ``max_retries`` times with ``retry_sleep_s`` between attempts.
    """
    if not staged:
        return
    operations = [
        CommitOperationAdd(
            path_in_repo=item["path_in_repo"],
            path_or_fileobj=str(item["local_path"]),
        )
        for item in staged
    ]
    msg = f"add {len(staged)} prediction files (batch {batch_index})"
    api = HfApi()
    last_exc = None
    for attempt in range(max_retries):
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=msg,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if "429" in str(exc):
                logger.warning(
                    "create_commit hit 429 (attempt %d/%d); sleeping %.0fs",
                    attempt + 1, max_retries, retry_sleep_s,
                )
                time.sleep(retry_sleep_s)
                continue
            raise
    raise RuntimeError(
        f"create_commit failed after {max_retries} retries on 429"
    ) from last_exc


def emit_done_lines(staged):
    """Print the orchestrator-parsed [done ...] line for each file in a committed batch."""
    for item in staged:
        print(
            f"[done file={item['input_file']} rows={item['rows']} "
            f"elapsed_s={item['elapsed_s']:.1f}]",
            flush=True,
        )


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
    logger.info("job_tag=%s input_files=%d text_columns=%s skip_cleanup=%s",
                args.job_tag, len(args.input_files),
                ",".join(args.text_columns), args.skip_cleanup)

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

            def docs_iter(text_column):
                for row_idx, row in enumerate(ds):
                    if row.get("status") not in (None, "ok"):
                        counters["skipped_status"] += 1
                        continue
                    text = row.get(text_column)
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
                            "text_source": text_column,
                            "arxiv_id": row.get(args.id_column),
                            "shard_id": row.get("shard_id"),
                            "text_length": len(text),
                        },
                    )

            output_rows = []
            for text_column in args.text_columns:
                col_start = counters["processed"]
                logger.info("[start file=%s text_column=%s]", input_file, text_column)
                for batch in _extract_funding_batch(docs_iter(text_column),
                                                    models=models, cfg=cascade_cfg):
                    output_rows.append(
                        make_output_row(batch, skip_cleanup=args.skip_cleanup))
                logger.info("[column_done file=%s text_column=%s rows=%d]",
                            input_file, text_column,
                            counters["processed"] - col_start)

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
