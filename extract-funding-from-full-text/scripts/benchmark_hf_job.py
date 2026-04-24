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
"""
Baseline benchmark for the statement-only ColBERT extractor.

Runs the unmodified extractor over `cometadata/arxiv-funding-statement-extraction`
(train or test split, folding `data/` and `arxiv-latex-extract/`) and reports
precision/recall/F1 per degradation tier plus throughput, then pushes predictions
and metrics back to Hub.

Designed to run via:

    hf jobs uv run --flavor h100x1 \
        --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
        --secrets HF_TOKEN=$(hf auth token) --timeout 1h \
        <url-or-path-to-this-script> -- --split train --run-name train-baseline-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("benchmark_hf_job")


# ---------------------------------------------------------------------------
# Inlined evaluation helpers (from evals/evaluate_funding_extraction/evaluate_funding_extraction.py)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    import ftfy

    text = ftfy.fix_text(text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("⋆", "").replace("•", "").replace("*", "")
    text = " ".join(text.split())
    text = re.sub(r"\\([a-z])", r"\1", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    na = normalize_text(a)
    nb = normalize_text(b)
    scores = [
        fuzz.partial_ratio(na, nb),
        fuzz.partial_ratio(nb, na),
        fuzz.token_sort_ratio(na, nb),
        fuzz.token_set_ratio(na, nb),
    ]
    return max(scores) / 100.0


def best_score(target: str, candidates: Iterable[str]) -> float:
    best = 0.0
    for cand in candidates:
        score = similarity(target, cand)
        if score > best:
            best = score
    return best


def evaluate_bucket(
    gold_per_doc: List[List[str]],
    pred_per_doc: List[List[str]],
    threshold: float,
) -> Dict[str, float]:
    gold_total = 0
    pred_total = 0
    matched_gold = 0
    matched_pred = 0

    for gold_stmts, pred_stmts in zip(gold_per_doc, pred_per_doc):
        gold_total += len(gold_stmts)
        pred_total += len(pred_stmts)

        for gold_stmt in gold_stmts:
            if best_score(gold_stmt, pred_stmts) >= threshold:
                matched_gold += 1

        for pred_stmt in pred_stmts:
            if best_score(pred_stmt, gold_stmts) >= threshold:
                matched_pred += 1

    precision = matched_pred / pred_total if pred_total else 0.0
    recall = matched_gold / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    def f_beta(beta: float) -> float:
        if precision == 0.0 and recall == 0.0:
            return 0.0
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    return {
        "gold_statements": gold_total,
        "predicted_statements": pred_total,
        "gold_statements_matched": matched_gold,
        "predicted_statements_matched": matched_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0_5": f_beta(0.5),
        "f1_5": f_beta(1.5),
    }


# ---------------------------------------------------------------------------
# Row bucketing
# ---------------------------------------------------------------------------

def bucket_for(row: Dict[str, Any]) -> str:
    aug = row.get("augmentation")
    cats = row.get("category") or []
    if aug is None:
        return "clean"
    if "clean_relocated" in cats:
        return "clean_relocated"
    return aug.get("tier", "unknown")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline benchmark for the statement-only ColBERT funding extractor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="cometadata/arxiv-funding-statement-extraction")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--subdirs",
        default="data,arxiv-latex-extract",
        help="Comma-separated JSONL sub-dirs to fold together into one split.",
    )
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--colbert-model", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Paragraph-encode batch size (H100/H200-appropriate default; library default is 32).",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="bf16",
        help="Cast ColBERT weights to this dtype after load. 'auto' = keep default fp32. bf16 = ~2-3x faster on H100/H200 at negligible accuracy loss.",
    )
    parser.add_argument("--enable-pattern-rescue", action="store_true")
    parser.add_argument("--enable-post-filter", action="store_true")
    parser.add_argument("--enable-paragraph-prefilter", action="store_true")
    parser.add_argument("--queries-file", default=None)

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="rapidfuzz threshold for counting a prediction as matching gold.",
    )

    parser.add_argument(
        "--push-to-hub",
        default="adambuttrick/arxiv-funding-statement-retrieval-extractions",
    )
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--private", action="store_true")

    parser.add_argument("--output-dir", default="./baseline_results")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--log-every", type=int, default=50)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_merged_split(dataset: str, split: str, subdirs: List[str]):
    from datasets import concatenate_datasets, load_dataset

    parts = []
    for sub in subdirs:
        logger.info("loading %s/%s.jsonl from %s", sub, split, dataset)
        part = load_dataset(
            dataset,
            data_files=f"{sub}/{split}.jsonl",
            split="train",
        )
        logger.info("  -> %d rows", len(part))
        parts.append(part)
    if len(parts) == 1:
        return parts[0]
    merged = concatenate_datasets(parts)
    logger.info("merged -> %d rows", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

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


def _gpu_mem_summary() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "gpu=cpu"
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        return (
            f"gpu_mem alloc={alloc_gb:.2f}GB peak={peak_gb:.2f}GB "
            f"reserved={reserved_gb:.2f}GB total={total_gb:.1f}GB"
        )
    except Exception as exc:
        return f"gpu_mem probe failed: {exc}"


def run_extraction(args: argparse.Namespace, ds) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], float]:
    _apply_dtype_patch(args.dtype)

    from funding_statement_extractor.config.loader import load_queries
    from funding_statement_extractor.statements.extraction import SemanticExtractionService

    try:
        import torch

        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("gpu=%s total_memory=%.1fGB", torch.cuda.get_device_name(0), total_gb)
    except Exception as exc:
        logger.warning("torch probe failed: %s", exc)

    service = SemanticExtractionService()
    queries = load_queries(queries_file=args.queries_file)
    logger.info("loaded %d semantic queries", len(queries))

    predictions_rows: List[Dict[str, Any]] = []
    latencies_by_bucket: Dict[str, List[float]] = defaultdict(list)

    n = len(ds)
    t_start = time.perf_counter()
    for i, row in enumerate(ds):
        text = row["text"]
        t0 = time.perf_counter()
        statements = service.extract_funding_statements(
            queries=queries,
            content=text,
            model_name=args.colbert_model,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            enable_paragraph_prefilter=args.enable_paragraph_prefilter,
        )
        if args.enable_pattern_rescue:
            rescued = service.rescue_by_patterns(text, statements)
            statements = list(statements) + list(rescued)
        if args.enable_post_filter and statements:
            from funding_statement_extractor.statements.post_filter import apply_post_filter

            statements = apply_post_filter(
                statements,
                high_confidence_threshold=30.0,
                low_confidence_threshold=10.0,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        b = bucket_for(row)
        warmup = i == 0
        latencies_by_bucket[b].append((dt_ms, warmup))

        predictions_rows.append(
            {
                "file": row["file"],
                "bucket": b,
                "category": row["category"],
                "found": bool(row.get("found")),
                "gold_statements": list(row.get("statements") or []),
                "predicted_statements": [s.statement for s in statements],
                "predicted_details": [
                    {
                        "statement": s.statement,
                        "score": float(s.score),
                        "query": s.query,
                        "paragraph_idx": s.paragraph_idx,
                    }
                    for s in statements
                ],
                "text_length": len(text),
                "latency_ms": dt_ms,
                "warmup": warmup,
            }
        )

        if i % args.log_every == 0 or i == n - 1:
            logger.info(
                "[%d/%d] bucket=%s n_pred=%d dt=%.0fms %s",
                i + 1,
                n,
                b,
                len(statements),
                dt_ms,
                _gpu_mem_summary(),
            )

    total_seconds = time.perf_counter() - t_start
    return predictions_rows, dict(latencies_by_bucket), total_seconds


# ---------------------------------------------------------------------------
# Metrics assembly
# ---------------------------------------------------------------------------

def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def compute_metrics(
    predictions_rows: List[Dict[str, Any]],
    latencies_by_bucket: Dict[str, List[Tuple[float, bool]]],
    total_seconds: float,
    similarity_threshold: float,
) -> List[Dict[str, Any]]:
    buckets_in_order = ["overall", "clean", "clean_relocated", "mild", "medium", "heavy", "combined"]
    buckets_present = set(r["bucket"] for r in predictions_rows)
    buckets = [b for b in buckets_in_order if b == "overall" or b in buckets_present]

    rows_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in predictions_rows:
        rows_by_bucket[r["bucket"]].append(r)
    rows_by_bucket["overall"] = list(predictions_rows)

    overall_lats_tagged: List[Tuple[float, bool]] = [
        entry for entries in latencies_by_bucket.values() for entry in entries
    ]

    metrics_rows: List[Dict[str, Any]] = []
    for b in buckets:
        rows = rows_by_bucket.get(b, [])
        if not rows:
            continue
        gold = [r["gold_statements"] for r in rows]
        pred = [r["predicted_statements"] for r in rows]
        stats = evaluate_bucket(gold, pred, similarity_threshold)

        if b == "overall":
            lats_tagged = overall_lats_tagged
            docs_per_sec = (len(rows) / total_seconds) if total_seconds > 0 else 0.0
        else:
            lats_tagged = latencies_by_bucket.get(b, [])
            bucket_sec = sum(lat for lat, _warmup in lats_tagged) / 1000.0
            docs_per_sec = (len(rows) / bucket_sec) if bucket_sec > 0 else 0.0

        post_warmup_lats = [lat for lat, warmup in lats_tagged if not warmup]
        mean_lat = statistics.fmean(post_warmup_lats) if post_warmup_lats else 0.0
        p50 = _percentile(post_warmup_lats, 0.5)
        p95 = _percentile(post_warmup_lats, 0.95)

        metrics_rows.append(
            {
                "bucket": b,
                "n_documents": len(rows),
                **stats,
                "mean_latency_ms": mean_lat,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "docs_per_sec": docs_per_sec,
                "similarity_threshold": similarity_threshold,
            }
        )
    return metrics_rows


def build_run_config(args: argparse.Namespace, n_rows: int, total_seconds: float) -> Dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            max_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            gpu_name = "cpu"
            max_gpu_mem_mb = 0.0
    except Exception:
        gpu_name = "unknown"
        max_gpu_mem_mb = 0.0

    return {
        "run_name": args.run_name,
        "colbert_model": args.colbert_model,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "enable_pattern_rescue": args.enable_pattern_rescue,
        "enable_post_filter": args.enable_post_filter,
        "enable_paragraph_prefilter": args.enable_paragraph_prefilter,
        "dtype": args.dtype,
        "similarity_threshold": args.similarity_threshold,
        "dataset": args.dataset,
        "split": args.split,
        "subdirs": args.subdirs,
        "n_rows": n_rows,
        "total_seconds": total_seconds,
        "overall_docs_per_sec": n_rows / total_seconds if total_seconds > 0 else 0.0,
        "gpu_name": gpu_name,
        "max_gpu_memory_mb": max_gpu_mem_mb,
        "git_ref": "statement-only-extraction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def push_results(
    predictions_rows: List[Dict[str, Any]],
    metrics_rows: List[Dict[str, Any]],
    run_config: Dict[str, Any],
    repo_id: str,
    run_name: str,
    private: bool,
) -> None:
    from datasets import Dataset

    logger.info("pushing predictions (n=%d) to %s", len(predictions_rows), repo_id)
    Dataset.from_list(predictions_rows).push_to_hub(
        repo_id,
        config_name=f"predictions-{run_name}",
        private=private,
    )

    combined_metrics = list(metrics_rows) + [{"bucket": "__run_config__", **run_config}]
    logger.info("pushing metrics (n=%d) to %s", len(combined_metrics), repo_id)
    Dataset.from_list(combined_metrics).push_to_hub(
        repo_id,
        config_name=f"metrics-{run_name}",
        private=private,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    if args.run_name is None:
        args.run_name = f"{args.split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info("run_name=%s", args.run_name)

    import torch

    cuda_ok = False
    for attempt in range(30):
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                cuda_ok = True
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("cuda probe attempt %d failed: %s", attempt, exc)
        time.sleep(1)
    if not cuda_ok:
        logger.error(
            "CUDA not available after 30s wait — refusing to fall back to CPU "
            "(would take ~14h on h200 host CPU). Aborting."
        )
        return 2

    logger.info(
        "torch=%s cuda=%s device=%s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0),
    )

    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    ds = load_merged_split(args.dataset, args.split, subdirs)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        logger.info("capped to %d rows via --max-samples", len(ds))

    predictions_rows, latencies_by_bucket, total_seconds = run_extraction(args, ds)
    metrics_rows = compute_metrics(
        predictions_rows,
        latencies_by_bucket,
        total_seconds,
        args.similarity_threshold,
    )
    run_config = build_run_config(args, n_rows=len(predictions_rows), total_seconds=total_seconds)

    logger.info("=== summary ===")
    logger.info(
        "rows=%d wall=%.1fs throughput=%.2f docs/sec gpu=%s peak_mem=%.1fMB",
        run_config["n_rows"],
        run_config["total_seconds"],
        run_config["overall_docs_per_sec"],
        run_config["gpu_name"],
        run_config["max_gpu_memory_mb"],
    )
    for row in metrics_rows:
        logger.info(
            "%16s  n=%-6d  P=%.3f  R=%.3f  F1=%.3f  p50=%.0fms  p95=%.0fms",
            row["bucket"],
            row["n_documents"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["p50_latency_ms"],
            row["p95_latency_ms"],
        )

    out_dir = Path(args.output_dir) / args.run_name
    write_jsonl(out_dir / "predictions.jsonl", predictions_rows)
    write_jsonl(out_dir / "metrics.jsonl", metrics_rows)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)
    logger.info("local artifacts written to %s", out_dir)

    if not args.no_push:
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            logger.warning("no HF_TOKEN in env; push_to_hub will fail unless cli-cached token is picked up")
        push_results(
            predictions_rows,
            metrics_rows,
            run_config,
            args.push_to_hub,
            args.run_name,
            args.private,
        )
    else:
        logger.info("--no-push set; skipping hub push")

    return 0


if __name__ == "__main__":
    sys.exit(main())
