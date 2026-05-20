# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.0.0",
#     "pyarrow>=12.0.0",
#     "rapidfuzz>=3.0.0",
#     "ftfy>=6.0",
# ]
# ///
"""Evaluate cascade funding-statement predictions against a gold file."""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import ftfy
import pandas as pd
from rapidfuzz import fuzz

GOLD_COLUMN = "funding_statements"
PRED_COLUMN = "funding_statement"
ID_COLUMN = "doc_id"
SOURCE_COLUMN = "text_source"

DEFAULT_FUZZY_THRESHOLD = 85.0
L4_CHUNK_HIT_THRESHOLD = 90.0
L4_COVERAGE_THRESHOLD = 0.8
CHUNK_SIZE = 120

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "none": 99}
EVAL_MODES = ["equality", "containment"]

DATA_SUFFIXES = {".parquet", ".json", ".jsonl"}


def normalize_ws(text: str) -> str:
    return " ".join(ftfy.fix_text(text or "").split())


def normalize_aggressive(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", ftfy.fix_text(text or "").lower())
    return " ".join(cleaned.split())


def chunk_statement(text: str, size: int = CHUNK_SIZE) -> List[str]:
    chunks: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(sentence) <= size:
            if sentence.strip():
                chunks.append(sentence.strip())
        else:
            for i in range(0, len(sentence), size):
                piece = sentence[i : i + size].strip()
                if piece:
                    chunks.append(piece)
    return chunks


def _chunk_coverage(pred_norm: str, gold_norm: str) -> float:
    chunks = chunk_statement(pred_norm)
    if not chunks or not gold_norm:
        return 0.0
    hits = sum(1 for ch in chunks
               if fuzz.partial_ratio(ch, gold_norm) >= L4_CHUNK_HIT_THRESHOLD)
    return hits / len(chunks)


def match_level_equality(pred: str, gold: str, fuzzy_threshold: float) -> str:
    if not pred or not gold:
        return "none"
    if pred == gold:
        return "L0"
    if normalize_ws(pred) == normalize_ws(gold):
        return "L1"
    pred_n, gold_n = normalize_aggressive(pred), normalize_aggressive(gold)
    if pred_n and pred_n == gold_n:
        return "L2"
    if pred_n and gold_n and fuzz.ratio(pred_n, gold_n) >= fuzzy_threshold:
        return "L3"
    if _chunk_coverage(pred_n, gold_n) >= L4_COVERAGE_THRESHOLD:
        return "L4"
    return "none"


def match_level_containment(pred: str, gold: str, fuzzy_threshold: float) -> str:
    if not pred or not gold:
        return "none"
    if pred in gold:
        return "L0"
    pred_w, gold_w = normalize_ws(pred), normalize_ws(gold)
    if pred_w and pred_w in gold_w:
        return "L1"
    pred_n, gold_n = normalize_aggressive(pred), normalize_aggressive(gold)
    if pred_n and pred_n in gold_n:
        return "L2"
    if pred_n and gold_n and fuzz.partial_ratio(pred_n, gold_n) >= fuzzy_threshold:
        return "L3"
    if _chunk_coverage(pred_n, gold_n) >= L4_COVERAGE_THRESHOLD:
        return "L4"
    return "none"


MATCHERS = {"equality": match_level_equality, "containment": match_level_containment}


def best_level(pred: str, gold_entries: List[str], matcher, fuzzy_threshold: float) -> str:
    best = "none"
    for gold in gold_entries:
        level = matcher(pred, gold, fuzzy_threshold)
        if LEVEL_RANK[level] < LEVEL_RANK[best]:
            best = level
    return best


def _read_records(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        return pd.DataFrame.from_records(rows)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = [{ID_COLUMN: k, **v} if isinstance(v, dict) else v
                    for k, v in data.items()]
        return pd.DataFrame.from_records(data)
    raise ValueError(f"unsupported file type: {path} (expected .parquet/.json/.jsonl)")


def load_predictions(path: Path) -> pd.DataFrame:
    frames = []
    if path.is_dir():
        for f in sorted(p for p in path.rglob("*") if p.suffix in DATA_SUFFIXES):
            df = _read_records(f)
            if ID_COLUMN not in df.columns or PRED_COLUMN not in df.columns:
                print(f"  skipping {f} (not a predictions file)")
                continue
            if SOURCE_COLUMN not in df.columns:
                rel_parent = f.parent.relative_to(path)
                df[SOURCE_COLUMN] = rel_parent.name or f.stem
            frames.append(df)
        if not frames:
            raise ValueError(
                f"no predictions files with '{ID_COLUMN}'/'{PRED_COLUMN}' under {path}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = _read_records(path)
        if SOURCE_COLUMN not in df.columns:
            df[SOURCE_COLUMN] = path.stem
    for col in (ID_COLUMN, PRED_COLUMN):
        if col not in df.columns:
            raise ValueError(f"predictions missing required column '{col}'")
    return df


def gold_entries(gold_value) -> List[str]:
    if gold_value is None or isinstance(gold_value, float):
        return []
    if isinstance(gold_value, str):
        return [gold_value] if gold_value else []
    return [str(x) for x in gold_value if x]


def load_gold(path: Path) -> Dict[str, List[str]]:
    df = _read_records(path)
    for col in (ID_COLUMN, GOLD_COLUMN):
        if col not in df.columns:
            raise ValueError(f"gold missing required column '{col}'")
    return {row[ID_COLUMN]: gold_entries(row[GOLD_COLUMN]) for _, row in df.iterrows()}


def score_group(pred_df: pd.DataFrame, gold_by_doc: Dict[str, List[str]],
                fuzzy_threshold: float) -> Dict:
    n_pred = n_gold = 0
    false_positives: List[str] = []
    false_negatives: List[str] = []
    correct = {mode: {level: 0 for level in LEVELS} for mode in EVAL_MODES}
    unmatched = {mode: [] for mode in EVAL_MODES}

    for _, row in pred_df.iterrows():
        pred = row.get(PRED_COLUMN) or ""
        golds = gold_by_doc.get(row[ID_COLUMN], [])
        pred_pos, gold_pos = bool(pred), bool(golds)
        n_pred += int(pred_pos)
        n_gold += int(gold_pos)
        if pred_pos and not gold_pos:
            false_positives.append(row[ID_COLUMN])
        if gold_pos and not pred_pos:
            false_negatives.append(row[ID_COLUMN])
        if not (pred_pos and gold_pos):
            continue
        for mode in EVAL_MODES:
            level = best_level(pred, golds, MATCHERS[mode], fuzzy_threshold)
            if level == "none":
                unmatched[mode].append(row[ID_COLUMN])
            for cut in LEVELS:
                if LEVEL_RANK[level] <= LEVEL_RANK[cut]:
                    correct[mode][cut] += 1

    def metrics(mode: str) -> Dict:
        tiers = {}
        for level in LEVELS:
            hits = correct[mode][level]
            precision = hits / n_pred if n_pred else 0.0
            recall = hits / n_gold if n_gold else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
            tiers[level] = {"correct": hits, "precision": precision,
                            "recall": recall, "f1": f1}
        return tiers

    return {
        "n_predictions": n_pred,
        "n_gold_positive": n_gold,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "false_positive_ids": false_positives,
        "false_negative_ids": false_negatives,
        "equality": metrics("equality"),
        "containment": metrics("containment"),
        "unmatched_ids": {mode: unmatched[mode] for mode in EVAL_MODES},
    }


def evaluate(pred_df: pd.DataFrame, gold_by_doc: Dict[str, List[str]],
             fuzzy_threshold: float) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for source, group in pred_df.groupby(SOURCE_COLUMN, sort=True):
        print(f"  [{source}] scoring {len(group)} predictions ...")
        results[str(source)] = score_group(group, gold_by_doc, fuzzy_threshold)
    return results


def _tier_table(results: Dict[str, Dict], mode: str, metric: str) -> str:
    width = max([17] + [len(s) for s in results])
    header = f"  | {'text_source':<{width}} | " + " | ".join(f"{l:>6}" for l in LEVELS) + " |"
    lines = [header, "  |" + "-" * (len(header) - 3) + "|"]
    for source, res in results.items():
        cells = " | ".join(f"{res[mode][l][metric]:6.4f}" for l in LEVELS)
        lines.append(f"  | {source:<{width}} | {cells} |")
    return "\n".join(lines)


def print_summary(label: str, results: Dict[str, Dict]) -> None:
    print(f"\n{label}")
    for mode in EVAL_MODES:
        for metric in ("precision", "recall", "f1"):
            print(f"\n  {mode} -- {metric} (L0 strict -> L4 lenient)")
            print(_tier_table(results, mode, metric))
    for source, res in results.items():
        print(f"\n  [{source}] predictions={res['n_predictions']} "
              f"gold_positive={res['n_gold_positive']} "
              f"FP={res['false_positives']} FN={res['false_negatives']} "
              f"unmatched(containment)={len(res['unmatched_ids']['containment'])}")


def print_delta(base: Dict[str, Dict], other: Dict[str, Dict]) -> None:
    print("\nF1 delta (compare - predictions)")
    width = max([17] + [len(s) for s in base])
    for mode in EVAL_MODES:
        print(f"\n  {mode}")
        for source in base:
            if source not in other:
                continue
            deltas = " | ".join(
                f"{other[source][mode][l]['f1'] - base[source][mode][l]['f1']:+6.4f}"
                for l in LEVELS
            )
            print(f"  | {source:<{width}} | {deltas} |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cascade funding-statement predictions against gold.",
    )
    parser.add_argument("-p", "--predictions", required=True, type=Path,
                        help="Local predictions file or directory (parquet/JSON/JSONL).")
    parser.add_argument("-g", "--gold", required=True, type=Path,
                        help="Local gold file (parquet/JSON/JSONL) with doc_id + "
                             "funding_statements.")
    parser.add_argument("-c", "--compare", default=None, type=Path,
                        help="Optional second predictions file/dir; reports F1 deltas.")
    parser.add_argument("-t", "--fuzzy-threshold", type=float,
                        default=DEFAULT_FUZZY_THRESHOLD,
                        help="Similarity cutoff (0-100) for the L3 fuzzy tier.")
    parser.add_argument("-o", "--output-json", required=True, type=Path,
                        help="Path to write the JSON metrics summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading gold from {args.gold} ...")
    gold_by_doc = load_gold(args.gold)

    print(f"Scoring predictions from {args.predictions} ...")
    results = evaluate(load_predictions(args.predictions), gold_by_doc,
                       args.fuzzy_threshold)
    print_summary(f"Predictions: {args.predictions}", results)

    output = {
        "predictions": str(args.predictions),
        "gold": str(args.gold),
        "fuzzy_threshold": args.fuzzy_threshold,
        "tiers": LEVELS,
        "results": results,
    }

    if args.compare:
        print(f"\nScoring comparison predictions from {args.compare} ...")
        compare = evaluate(load_predictions(args.compare), gold_by_doc,
                           args.fuzzy_threshold)
        print_summary(f"Comparison: {args.compare}", compare)
        print_delta(results, compare)
        output["compare"] = {"predictions": str(args.compare), "results": compare}

    args.output_json.write_text(json.dumps(output, indent=2))
    print(f"\nWrote JSON results to {args.output_json}")


if __name__ == "__main__":
    main()
