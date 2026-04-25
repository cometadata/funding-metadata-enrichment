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
