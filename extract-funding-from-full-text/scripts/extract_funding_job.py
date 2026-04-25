#!/usr/bin/env python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "extract-funding-statements-from-full-text @ git+https://github.com/cometadata/funding-metadata-enrichment.git@statement-only-extraction#subdirectory=extract-funding-from-full-text",
#     "datasets>=3.0.0",
#     "huggingface-hub>=0.25.0",
#     "rapidfuzz>=3.0.0",
#     "ftfy>=6.0.0",
#     "sentence-transformers==5.1.1",
#     "pylate==1.4.0",
#     "torch>=2.5.0,<2.7.0",
# ]
# ///
"""Worker script: run Tier-2 funding-statement extraction over input parquets and push results to hub.

Designed to run on an HF Job (a100-large flavor) via `hf jobs uv run` on the
`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` image.
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

logger = logging.getLogger("extract_funding_job")


OUTPUT_SCHEMA = pa.schema([
    pa.field("arxiv_id", pa.string()),
    pa.field("shard_id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("input_file", pa.string()),
    pa.field("row_idx", pa.int64()),
    pa.field("predicted_statements", pa.list_(pa.string())),
    pa.field("predicted_details", pa.list_(pa.struct([
        pa.field("statement", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("query", pa.string()),
        pa.field("paragraph_idx", pa.int64()),
    ]))),
    pa.field("text_length", pa.int64()),
    pa.field("latency_ms", pa.float64()),
    pa.field("error", pa.string()),
])


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run Tier-2 funding-statement extraction over a list of input parquets and push results to hub.",
    )
    p.add_argument("--input-repo", required=True,
                   help="HF dataset repo id containing the input parquets.")
    p.add_argument("--input-files", required=True,
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="Comma-separated list of in-repo parquet paths to process.")
    p.add_argument("--output-repo", required=True,
                   help="HF dataset repo id to push prediction parquets to.")
    p.add_argument("--job-tag", required=True,
                   help="Free-form tag for log correlation with the orchestrator.")
    p.add_argument("--text-column", default="text",
                   help="Name of the text column to extract from (default: text).")
    p.add_argument("--id-column", default="arxiv_id",
                   help="Name of the primary id column (default: arxiv_id).")
    p.add_argument("--colbert-model", default="lightonai/GTE-ModernColBERT-v1")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Skip CUDA probe; for local smoke tests only.")
    p.add_argument("--commit-batch-size", type=int, default=25,
                   help="Number of staged prediction files to combine into a single hub commit (default: 25).")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the preflight commit check before starting GPU work.")
    return p.parse_args(argv)


def make_output_row(result):
    meta = result.metadata or {}
    return {
        "arxiv_id": meta.get("arxiv_id"),
        "shard_id": meta.get("shard_id"),
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


def push_parquet_to_hub(rows, *, repo_id, path_in_repo, staging_dir):
    """DEPRECATED: one-commit-per-file path. Retained for backward compat / tests.

    New code should use ``stage_parquet_locally`` + ``commit_staged_batch`` to
    avoid the hub's 320 commits/hour rate limit.
    """
    if rows is None:
        rows = []
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
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


def stage_parquet_locally(rows, *, path_in_repo, staging_dir):
    """Write rows to a local parquet file using OUTPUT_SCHEMA. Returns local Path."""
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


def _apply_dtype_patch(dtype_str: str) -> None:
    if dtype_str == "auto" or dtype_str == "fp32":
        return
    import torch
    from pylate import models as pl_models

    target = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
    orig_init = pl_models.ColBERT.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.to(target)
        logger.info("ColBERT weights cast to %s", target)

    pl_models.ColBERT.__init__ = patched_init


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stdout)
    logger.info("job_tag=%s input_files=%d", args.job_tag, len(args.input_files))

    import torch

    if args.allow_cpu:
        logger.info(
            "torch=%s cuda=%s device=%s (allow-cpu=True; skipping CUDA-required probe)",
            torch.__version__,
            torch.cuda.is_available(),
            "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        )
    else:
        cuda_ok = False
        for attempt in range(120):
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    cuda_ok = True
                    break
            except Exception as exc:  # noqa: BLE001
                logger.warning("cuda probe attempt %d failed: %s", attempt, exc)
            time.sleep(1)
        if not cuda_ok:
            logger.error(
                "CUDA not available after 120s wait — refusing to fall back to CPU "
                "(would take ~14h on a100-large host CPU). Aborting."
            )
            return 2

        logger.info(
            "torch=%s cuda=%s device=%s",
            torch.__version__,
            torch.cuda.is_available(),
            torch.cuda.get_device_name(0),
        )

    _apply_dtype_patch(args.dtype)

    from datasets import load_dataset
    from funding_statement_extractor.config.loader import load_queries
    from funding_statement_extractor.statements.batch_extraction import (
        DocPayload,
        extract_funding_statements_batch,
    )

    queries = load_queries()
    logger.info("loaded %d queries", len(queries))

    staging = Path("/tmp/extract_funding_outputs")
    if not args.skip_preflight:
        try:
            preflight_commit_check(args.output_repo, job_tag=args.job_tag,
                                   staging_dir=staging)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "preflight commit check failed; aborting before GPU work: %s", exc,
            )
            return 5

    staged_buffer: list[dict] = []
    batch_index = 0
    consecutive_commit_failures = 0
    MAX_CONSECUTIVE_COMMIT_FAILURES = 2

    for input_file in args.input_files:
        t0 = time.perf_counter()
        logger.info("[start file=%s]", input_file)
        try:
            ds = load_dataset(args.input_repo, data_files=input_file,
                              split="train", streaming=True)

            counters = {"processed": 0, "skipped_status": 0, "skipped_empty": 0}

            def docs_iter():
                for row_idx, row in enumerate(ds):
                    if row.get("status") != "ok":
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
                            "arxiv_id": row.get("arxiv_id"),
                            "shard_id": row.get("shard_id"),
                            "text_length": len(text),
                        },
                    )

            output_rows = []
            for result in extract_funding_statements_batch(
                documents=docs_iter(),
                queries=queries,
                model_name=args.colbert_model,
                top_k=5,
                threshold=10.0,
                enable_paragraph_prefilter=True,
                regex_match_score_floor=11.0,
                paragraphs_per_batch=4096,
                encode_batch_size=args.batch_size,
                dtype=args.dtype,
            ):
                output_rows.append(make_output_row(result))

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
                logger.info("committing batch %d (%d files)", batch_index, len(staged_buffer))
                try:
                    commit_staged_batch(
                        staged_buffer, repo_id=args.output_repo, batch_index=batch_index,
                    )
                except Exception as commit_exc:  # noqa: BLE001
                    consecutive_commit_failures += 1
                    logger.exception(
                        "batch commit %d failed (consecutive=%d/%d): %s",
                        batch_index, consecutive_commit_failures,
                        MAX_CONSECUTIVE_COMMIT_FAILURES, commit_exc,
                    )
                    if consecutive_commit_failures >= MAX_CONSECUTIVE_COMMIT_FAILURES:
                        logger.error(
                            "aborting worker after %d consecutive commit failures "
                            "(%d files in buffer never persisted) — orchestrator "
                            "will resubmit unfinished files",
                            consecutive_commit_failures, len(staged_buffer),
                        )
                        return 4
                    # Keep buffer; next batch boundary retries with more files.
                else:
                    # Only emit [done ...] AFTER successful commit — orchestrator
                    # treats these as proof the file is on hub.
                    emit_done_lines(staged_buffer)
                    staged_buffer = []
                    consecutive_commit_failures = 0
        except Exception as exc:  # noqa: BLE001
            logger.exception("[fail file=%s] %s", input_file, exc)
            # Continue to next file rather than failing the whole job —
            # the orchestrator will retry the unfinished file.

    # Flush any remaining staged files in a final commit.
    if staged_buffer:
        batch_index += 1
        logger.info("committing final batch %d (%d files)", batch_index, len(staged_buffer))
        try:
            commit_staged_batch(
                staged_buffer, repo_id=args.output_repo, batch_index=batch_index,
            )
            emit_done_lines(staged_buffer)
            staged_buffer = []
        except Exception as exc:  # noqa: BLE001
            logger.exception("final batch commit failed: %s", exc)
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
