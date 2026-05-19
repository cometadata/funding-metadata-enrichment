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
    p.add_argument("--id-col", default="doc_id")
    p.add_argument("--pred-col", default="funding_statement")
    p.add_argument("--gold-col", default="funding_statements")
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
