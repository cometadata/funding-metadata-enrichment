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
    return p.parse_args(argv)


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
