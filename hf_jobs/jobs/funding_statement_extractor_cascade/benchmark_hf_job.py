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
from typing import Dict, List


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
    p.add_argument("--id-col", default="doc_id")
    p.add_argument("--pred-col", default="funding_statement")
    p.add_argument("--gold-col", default="funding_statements")
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
