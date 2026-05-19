#!/usr/bin/env python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "funding-statement-extractor-cascade @ git+https://github.com/cometadata/funding-metadata-enrichment.git@main#subdirectory=funding_statement_extractor_cascade",
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
            "doc_id": row.get("doc_id"),
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
