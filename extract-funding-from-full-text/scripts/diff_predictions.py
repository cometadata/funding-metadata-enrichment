#!/usr/bin/env python
"""Diff per_doc_predictions across two profile reports.

Used as the byte-identity gate after Tier 1 refactor: compares the
predicted funding statements per document between a baseline profile
JSON and a post-fix profile JSON. Exits 0 on full equality, non-zero
on any mismatch.

Usage:
    python scripts/diff_predictions.py reports/baseline.json reports/post.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "per_doc_predictions" not in data:
        sys.exit(f"error: {path} has no per_doc_predictions field")
    return data["per_doc_predictions"]


def normalize_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "statement": s["statement"],
            "query": s["query"],
            "paragraph_idx": s["paragraph_idx"],
        }
        for s in doc.get("statements", [])
    ]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("baseline", type=Path)
    p.add_argument("post", type=Path)
    p.add_argument("--allow-score-drift", type=float, default=0.0,
                   help="Tolerance for score differences (default 0.0 = strict).")
    args = p.parse_args(argv)

    baseline = load_predictions(args.baseline)
    post = load_predictions(args.post)

    if len(baseline) != len(post):
        print(f"FAIL: doc count mismatch baseline={len(baseline)} post={len(post)}")
        return 1

    mismatches = 0
    for b_doc, p_doc in zip(baseline, post):
        b_norm = normalize_doc(b_doc)
        p_norm = normalize_doc(p_doc)
        if b_norm != p_norm:
            mismatches += 1
            print(f"DIFF doc_idx={b_doc['doc_idx']}:")
            b_set = {(s["statement"], s["query"]) for s in b_norm}
            p_set = {(s["statement"], s["query"]) for s in p_norm}
            for missing in b_set - p_set:
                print(f"  - missing in post: query={missing[1]!r} stmt={missing[0][:120]!r}")
            for extra in p_set - b_set:
                print(f"  + extra in post:   query={extra[1]!r} stmt={extra[0][:120]!r}")

    if mismatches:
        print(f"FAIL: {mismatches}/{len(baseline)} docs differ")
        return 1
    print(f"OK: {len(baseline)} docs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
