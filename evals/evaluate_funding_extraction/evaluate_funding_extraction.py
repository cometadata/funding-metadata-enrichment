import re
import json
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ftfy
from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("⋆", "").replace("•", "").replace("*", "")
    text = " ".join(text.split())
    text = re.sub(r"\\([a-z])", r"\1", text)
    return text.strip()


def canonicalize_filename(filename: str) -> str:
    name = Path(filename).name
    stem = name[:-3] if name.endswith(".md") else name
    if "-" in stem:
        stem = stem.replace("-", ".", 1)
    return f"{stem}.md"


def load_gold(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    preds: Dict[str, List[str]] = {}
    for filename, payload in data.get("results", {}).items():
        statements: List[str] = []
        for item in payload.get("funding_statements", []):
            if isinstance(item, dict):
                text = item.get("statement")
                if text:
                    statements.append(text)
            elif isinstance(item, str):
                statements.append(item)
        preds[canonicalize_filename(filename)] = statements
    return preds


def similarity(a: str, b: str) -> float:
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


def filter_predictions_to_gold(
    predictions: Dict[str, List[str]], gold_filenames: Iterable[str]
) -> Tuple[Dict[str, List[str]], int, int, int]:
    gold_set = set(gold_filenames)
    total_preds_in_file = sum(len(v) for v in predictions.values())

    filtered: Dict[str, List[str]] = {
        fname: stmts for fname, stmts in predictions.items() if fname in gold_set
    }

    total_preds_used = sum(len(v) for v in filtered.values())
    total_preds_discarded = total_preds_in_file - total_preds_used
    return filtered, total_preds_in_file, total_preds_used, total_preds_discarded


def evaluate_split(
    gold_entries: List[Dict],
    predictions: Dict[str, List[str]],
    threshold: float,
) -> Dict[str, float]:
    gold_total = 0
    pred_total = 0
    matched_gold = 0
    matched_pred = 0

    for entry in gold_entries:
        filename = canonicalize_filename(entry["filename"])
        gold_statements = entry.get("statements", [])
        pred_statements = predictions.get(filename, [])

        gold_total += len(gold_statements)
        pred_total += len(pred_statements)

        for gold_stmt in gold_statements:
            if best_score(gold_stmt, pred_statements) >= threshold:
                matched_gold += 1

        for pred_stmt in pred_statements:
            if best_score(pred_stmt, gold_statements) >= threshold:
                matched_pred += 1

    precision = matched_pred / pred_total if pred_total else 0.0
    recall = matched_gold / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    def f_beta(beta: float) -> float:
        if precision == 0.0 and recall == 0.0:
            return 0.0
        beta2 = beta * beta
        return (1 + beta2) * precision * recall / (beta2 * precision + recall) if (beta2 * precision + recall) else 0.0

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


def print_summary(
    label: str,
    stats: Dict[str, float],
    threshold: float,
    preds_in_file: int,
    preds_used: int,
    preds_discarded: int,
) -> None:
    print(f"{label} @ threshold {threshold:.2f}")
    print(f"  Gold statements:            {stats['gold_statements']}")
    print(f"  Predicted statements (all): {preds_in_file}")
    print(f"  Predicted statements used:  {preds_used}")
    print(f"  Predicted statements ignored (ID mismatch): {preds_discarded}")
    print(f"  Gold matched:               {stats['gold_statements_matched']}")
    print(f"  Predictions matched:        {stats['predicted_statements_matched']}")
    print(f"  Precision:                  {stats['precision']:.4f}")
    print(f"  Recall:                     {stats['recall']:.4f}")
    print(f"  F1:                         {stats['f1']:.4f}")
    print(f"  F0.5:                       {stats['f0_5']:.4f}")
    print(f"  F1.5:                       {stats['f1_5']:.4f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate funding statement extraction against a single gold JSON file."
    )
    parser.add_argument(
        "-p",
        "--predictions",
        required=True,
        type=Path,
        help="Path to predictions JSON (e.g., new_funding_statements.json).",
    )
    parser.add_argument(
        "-g",
        "--gold",
        required=True,
        type=Path,
        help="Path to comparison/gold JSON file (e.g., train.json or test.json).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=True,
        help="Similarity threshold (0.0-1.0) for counting a match.",
    )
    parser.add_argument(
        "-o",
        "--output-json",
        required=True,
        type=Path,
        help="Path to write JSON summary of results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_raw = load_predictions(args.predictions)
    gold_entries = load_gold(args.gold)
    gold_filenames = [canonicalize_filename(e["filename"]) for e in gold_entries]

    (
        predictions_filtered,
        preds_in_file,
        preds_used,
        preds_discarded,
    ) = filter_predictions_to_gold(predictions_raw, gold_filenames)

    stats = evaluate_split(gold_entries, predictions_filtered, args.threshold)

    print_summary(
        "Evaluation",
        stats,
        args.threshold,
        preds_in_file,
        preds_used,
        preds_discarded,
    )

    output = {
        "threshold": args.threshold,
        "files_in_gold": len(gold_entries),
        "gold_files_with_statements": sum(
            1 for e in gold_entries if e.get("statements")
        ),
        "predicted_statements_in_file": preds_in_file,
        "predicted_statements_used": preds_used,
        "predicted_statements_discarded": preds_discarded,
        "metrics": stats,
    }
    args.output_json.write_text(json.dumps(output, indent=2))
    print(f"Wrote JSON results to {args.output_json}")


if __name__ == "__main__":
    main()
