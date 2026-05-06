# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub>=0.34.0,<1.0",
#     "datasets>=4.5.0",
#     "rapidfuzz>=3.0.0",
#     "ftfy>=6.0",
#     "scipy>=1.11.0",
#     "numpy>=1.24.0",
# ]
# ///
"""Evaluate LoRA funding extraction predictions against ground truth.
"""

import argparse
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

import ftfy
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment


def normalize_text(text) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = ftfy.fix_text(text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("\u22c6", "").replace("\u2022", "").replace("*", "")
    text = " ".join(text.split())
    text = re.sub(r"\\([a-z])", r"\1", text)
    return text.strip()


_PAREN_RE = re.compile(r"\(([^)]+)\)")
_TOKEN_SPLIT_RE = re.compile(r"[\s/\-,]+")
_CONTAINMENT_FLOOR = 0.45


def _extract_parenthetical_acronyms(text: str) -> List[str]:
    """Pull acronym-like tokens from parentheses: 'Foo Bar (FB)' -> ['FB']."""
    return [
        m.strip()
        for m in _PAREN_RE.findall(text)
        if m.strip().isupper() and len(m.strip()) <= 10
    ]


def _tokenize(text: str) -> set:
    """Split on whitespace, slashes, hyphens, commas for containment check."""
    return {t.lower() for t in _TOKEN_SPLIT_RE.split(text) if t}


def _token_containment(shorter: str, longer: str) -> float:
    """Fraction of shorter's tokens that appear in longer."""
    short_toks = _tokenize(shorter)
    long_toks = _tokenize(longer)
    if not short_toks:
        return 0.0
    return len(short_toks & long_toks) / len(short_toks)


def similarity_permissive(a: str, b: str) -> float:
    """Original permissive similarity (partial_ratio + token_set_ratio, no length damping)."""
    na = normalize_text(a)
    nb = normalize_text(b)
    scores = [
        fuzz.partial_ratio(na, nb),
        fuzz.partial_ratio(nb, na),
        fuzz.token_sort_ratio(na, nb),
        fuzz.token_set_ratio(na, nb),
    ]
    return max(scores) / 100.0


def similarity(a: str, b: str) -> float:
    """Containment-aware similarity.

    Three tiers:
    1. token_sort_ratio — always trusted as the strict baseline.
    2. Parenthetical acronym match — if the short string matches an
       acronym in parentheses in the long string, return 0.90.
    3. Token containment boost — if every token of the shorter string
       appears in the longer string (splitting on whitespace, slash,
       hyphen, comma) AND the character-length ratio >= CONTAINMENT_FLOOR,
       return 0.85.  This handles "DOE" ↔ "DOE (United States)" and
       "JSPS" ↔ "MEXT/JSPS" without inflating unrelated substrings.
    """
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0

    token_sort = fuzz.token_sort_ratio(na, nb) / 100.0

    # Acronym-in-parentheses: "Foo Bar (FB)" vs "FB" → credit it
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    for acr in _extract_parenthetical_acronyms(longer):
        if fuzz.ratio(shorter.upper(), acr.upper()) >= 90:
            return max(token_sort, 0.90)

    # Token containment: all of shorter's tokens present in longer?
    len_ratio = len(shorter) / len(longer) if len(longer) else 0.0
    containment = _token_containment(shorter, longer)
    if containment >= 1.0 and len_ratio >= _CONTAINMENT_FLOOR:
        return max(token_sort, 0.85)

    return token_sort


def similarity_strict(a: str, b: str) -> float:
    """Strict similarity: only token_sort_ratio (no partial/set)."""
    na = normalize_text(a)
    nb = normalize_text(b)
    return fuzz.token_sort_ratio(na, nb) / 100.0


def normalize_award_id(award_id: str) -> str:
    aid = award_id.strip().upper()
    aid = re.sub(r"[\s\-/]", "", aid)
    return aid


def award_ids_match(pred_id: str, gold_id: str, mode: str = "normalized") -> bool:
    if mode == "exact":
        return pred_id.strip().upper() == gold_id.strip().upper()
    return normalize_award_id(pred_id) == normalize_award_id(gold_id)


def optimal_match(
    gold_items: List[str],
    pred_items: List[str],
    score_fn: Callable[[str, str], float],
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Find optimal 1:1 matching using the Hungarian algorithm."""
    if not gold_items or not pred_items:
        return []

    n_gold = len(gold_items)
    n_pred = len(pred_items)

    scores = np.zeros((n_gold, n_pred))
    for gi, g in enumerate(gold_items):
        for pi, p in enumerate(pred_items):
            s = score_fn(g, p)
            scores[gi, pi] = s if s >= threshold else 0.0

    row_ind, col_ind = linear_sum_assignment(-scores)

    matched = []
    for gi, pi in zip(row_ind, col_ind):
        if scores[gi, pi] >= threshold:
            matched.append((gi, pi, float(scores[gi, pi])))

    return matched


def optimal_match_ids(
    gold_items: List[str],
    pred_items: List[str],
    id_match_mode: str = "normalized",
) -> List[Tuple[int, int, float]]:
    def score_fn(g: str, p: str) -> float:
        return 1.0 if award_ids_match(p, g, mode=id_match_mode) else 0.0
    return optimal_match(gold_items, pred_items, score_fn, threshold=1.0)


@dataclass
class Award:
    award_ids: List[str] = field(default_factory=list)
    funding_scheme: List[str] = field(default_factory=list)
    award_title: List[str] = field(default_factory=list)


@dataclass
class Funder:
    funder_name: Optional[str] = None
    awards: List[Award] = field(default_factory=list)


@dataclass
class LevelMetrics:
    gold_count: int = 0
    pred_count: int = 0
    matched: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    f0_5: float = 0.0
    f1_5: float = 0.0


def _f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def _build_level_metrics(gold_count: int, pred_count: int, matched: int) -> LevelMetrics:
    precision = matched / pred_count if pred_count else 0.0
    recall = matched / gold_count if gold_count else 0.0
    f1 = _f_beta(precision, recall, 1.0)
    return LevelMetrics(
        gold_count=gold_count,
        pred_count=pred_count,
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        f0_5=_f_beta(precision, recall, 0.5),
        f1_5=_f_beta(precision, recall, 1.5),
    )


def _collect_awards(funder: Funder) -> Tuple[List[str], List[str], List[str]]:
    ids, schemes, titles = [], [], []
    for a in funder.awards:
        ids.extend(a.award_ids)
        schemes.extend(a.funding_scheme)
        titles.extend(a.award_title)
    return ids, schemes, titles


def _merge_funders(
    funders: List[Funder], threshold: float, sim_fn: Callable[[str, str], float] = similarity
) -> List[Funder]:
    """Merge funders whose names are similar above *threshold*.

    Uses iterative Hungarian matching: on each round, find the best 1:1
    pairings among remaining groups, merge any pair that meets the
    threshold, and repeat until no more merges occur.  This avoids the
    order-dependence of the previous greedy approach.
    """
    unnamed_awards: List[Award] = []
    groups: List[Tuple[str, List[Award]]] = []

    for f in funders:
        if not f.funder_name:
            unnamed_awards.extend(f.awards)
        else:
            groups.append((f.funder_name, list(f.awards)))

    # Iteratively merge the closest pair(s) until no pair exceeds the threshold
    changed = True
    while changed and len(groups) > 1:
        changed = False
        names = [name for name, _ in groups]
        n = len(names)
        scores = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s = sim_fn(names[i], names[j])
                scores[i, j] = s if s >= threshold else 0.0
                scores[j, i] = scores[i, j]

        # Find optimal 1:1 pairing among groups
        row_ind, col_ind = linear_sum_assignment(-scores)
        merge_pairs = []
        for ri, ci in zip(row_ind, col_ind):
            if ri < ci and scores[ri, ci] >= threshold:
                merge_pairs.append((ri, ci))

        if not merge_pairs:
            break

        # Merge matched pairs (process in reverse so indices stay valid)
        merged_indices: set = set()
        new_groups = []
        for ri, ci in merge_pairs:
            if ri in merged_indices or ci in merged_indices:
                continue
            name_r, awards_r = groups[ri]
            _name_c, awards_c = groups[ci]
            new_groups.append((name_r, awards_r + awards_c))
            merged_indices.add(ri)
            merged_indices.add(ci)

        for idx, g in enumerate(groups):
            if idx not in merged_indices:
                new_groups.append(g)

        groups = new_groups
        changed = True

    result = [Funder(funder_name=name, awards=awards) for name, awards in groups]
    if unnamed_awards:
        result.append(Funder(funder_name=None, awards=unnamed_awards))
    return result


def _evaluate_per_funder(
    gold_funders: List[Funder],
    pred_funders: List[Funder],
    funder_threshold: float,
    threshold: float,
    id_match_mode: str,
    sim_fn: Callable[[str, str], float] = similarity,
) -> Tuple[LevelMetrics, LevelMetrics, LevelMetrics, LevelMetrics]:
    gold_merged = _merge_funders(gold_funders, funder_threshold, sim_fn)
    pred_merged = _merge_funders(pred_funders, funder_threshold, sim_fn)

    gold_named_indices = [i for i, f in enumerate(gold_merged) if f.funder_name]
    pred_named_indices = [i for i, f in enumerate(pred_merged) if f.funder_name]

    gold_named = [gold_merged[i].funder_name for i in gold_named_indices]
    pred_named = [pred_merged[i].funder_name for i in pred_named_indices]

    funder_matches = optimal_match(gold_named, pred_named, sim_fn, funder_threshold)

    funder_matched_count = len(funder_matches)
    funder_metrics = _build_level_metrics(
        len(gold_named), len(pred_named), funder_matched_count
    )

    matched_gold_set: set = set()
    matched_pred_set: set = set()
    paired = []
    for gm_idx, pm_idx, _score in funder_matches:
        gi = gold_named_indices[gm_idx]
        pi = pred_named_indices[pm_idx]
        paired.append((gi, pi))
        matched_gold_set.add(gi)
        matched_pred_set.add(pi)

    unnamed_gold_idx = next(
        (i for i, f in enumerate(gold_merged) if not f.funder_name), None
    )
    unnamed_pred_idx = next(
        (i for i, f in enumerate(pred_merged) if not f.funder_name), None
    )
    if unnamed_gold_idx is not None and unnamed_pred_idx is not None:
        paired.append((unnamed_gold_idx, unnamed_pred_idx))
        matched_gold_set.add(unnamed_gold_idx)
        matched_pred_set.add(unnamed_pred_idx)

    total_id_gold = total_id_pred = total_id_matched = 0
    total_scheme_gold = total_scheme_pred = total_scheme_matched = 0
    total_title_gold = total_title_pred = total_title_matched = 0

    for gi, pi in paired:
        g_ids, g_schemes, g_titles = _collect_awards(gold_merged[gi])
        p_ids, p_schemes, p_titles = _collect_awards(pred_merged[pi])

        id_matches = optimal_match_ids(g_ids, p_ids, id_match_mode)
        total_id_gold += len(g_ids)
        total_id_pred += len(p_ids)
        total_id_matched += len(id_matches)

        scheme_matches = optimal_match(g_schemes, p_schemes, sim_fn, threshold)
        total_scheme_gold += len(g_schemes)
        total_scheme_pred += len(p_schemes)
        total_scheme_matched += len(scheme_matches)

        title_matches = optimal_match(g_titles, p_titles, sim_fn, threshold)
        total_title_gold += len(g_titles)
        total_title_pred += len(p_titles)
        total_title_matched += len(title_matches)

    for gi in range(len(gold_merged)):
        if gi not in matched_gold_set:
            g_ids, g_schemes, g_titles = _collect_awards(gold_merged[gi])
            total_id_gold += len(g_ids)
            total_scheme_gold += len(g_schemes)
            total_title_gold += len(g_titles)

    for pi in range(len(pred_merged)):
        if pi not in matched_pred_set:
            p_ids, p_schemes, p_titles = _collect_awards(pred_merged[pi])
            total_id_pred += len(p_ids)
            total_scheme_pred += len(p_schemes)
            total_title_pred += len(p_titles)

    id_metrics = _build_level_metrics(total_id_gold, total_id_pred, total_id_matched)
    scheme_metrics = _build_level_metrics(total_scheme_gold, total_scheme_pred, total_scheme_matched)
    title_metrics = _build_level_metrics(total_title_gold, total_title_pred, total_title_matched)

    return funder_metrics, id_metrics, scheme_metrics, title_metrics


def compute_hierarchical_reward(
    gold_funders: List[Funder],
    pred_funders: List[Funder],
    funder_threshold: float = 0.8,
    threshold: float = 0.8,
    id_match_mode: str = "normalized",
    weights: Tuple[float, float, float, float] = (0.50, 0.50, 0.0, 0.0),
) -> Tuple[float, dict]:
    """Compute a scalar RL reward from structured funding extraction.

    Uses the same hierarchical matching as _evaluate_per_funder: sub-fields
    (award IDs, schemes, titles) are only scored under Hungarian-matched
    funder pairs.

    Returns:
        (reward, metrics) where reward is in [0, 1] and metrics contains
        per-field F0.5 scores plus the association gap diagnostic.
    """
    w_funder, w_id, w_scheme, w_title = weights

    # Hierarchical evaluation (matches eval script exactly)
    funder_m, id_m, scheme_m, title_m = _evaluate_per_funder(
        gold_funders, pred_funders, funder_threshold, threshold, id_match_mode,
    )

    # Flat award ID evaluation (for association gap diagnostic)
    all_gold_ids: List[str] = []
    all_pred_ids: List[str] = []
    for f in gold_funders:
        for a in f.awards:
            all_gold_ids.extend(a.award_ids)
    for f in pred_funders:
        for a in f.awards:
            all_pred_ids.extend(a.award_ids)
    flat_id_matches = optimal_match_ids(all_gold_ids, all_pred_ids, id_match_mode)
    flat_id_m = _build_level_metrics(len(all_gold_ids), len(all_pred_ids), len(flat_id_matches))

    # Per-field F0.5, treating both-empty as 1.0
    def _f0_5_or_empty(m: LevelMetrics) -> float:
        if m.gold_count == 0 and m.pred_count == 0:
            return 1.0
        return m.f0_5

    funder_f0_5 = _f0_5_or_empty(funder_m)
    id_f0_5 = _f0_5_or_empty(id_m)
    scheme_f0_5 = _f0_5_or_empty(scheme_m)
    title_f0_5 = _f0_5_or_empty(title_m)
    flat_id_f0_5 = _f0_5_or_empty(flat_id_m)

    # When funders exist but optional sub-fields (scheme, title) are
    # absent from both gold and pred, inherit from the product of active
    # field scores instead of a free 1.0.  This prevents empty optional
    # fields from inflating the reward above what funder + award merit.
    if len(gold_funders) > 0 or len(pred_funders) > 0:
        active_score = funder_f0_5 * id_f0_5
        if scheme_m.gold_count == 0 and scheme_m.pred_count == 0:
            scheme_f0_5 = active_score
        if title_m.gold_count == 0 and title_m.pred_count == 0:
            title_f0_5 = active_score

    reward = (
        w_funder * funder_f0_5
        + w_id * id_f0_5
        + w_scheme * scheme_f0_5
        + w_title * title_f0_5
    )

    association_gap = flat_id_f0_5 - id_f0_5

    metrics = {
        "funder_f0_5": funder_f0_5,
        "funder_precision": funder_m.precision,
        "funder_recall": funder_m.recall,
        "award_id_f0_5": id_f0_5,
        "award_id_precision": id_m.precision,
        "award_id_recall": id_m.recall,
        "scheme_f0_5": scheme_f0_5,
        "title_f0_5": title_f0_5,
        "flat_award_id_f0_5": flat_id_f0_5,
        "association_gap": association_gap,
    }

    return reward, metrics


def _coerce_to_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return [v for v in val if v is not None and v != ""]
    if val == "":
        return []
    return [val]


def _parse_funder(raw: dict) -> Funder:
    awards = []
    if "awards" in raw and isinstance(raw["awards"], list):
        for a in raw["awards"]:
            if isinstance(a, dict):
                awards.append(Award(
                    award_ids=_coerce_to_list(a.get("award_ids")),
                    funding_scheme=_coerce_to_list(a.get("funding_scheme")),
                    award_title=_coerce_to_list(a.get("award_title")),
                ))
    elif "award_ids" in raw:
        awards.append(Award(
            award_ids=_coerce_to_list(raw.get("award_ids")),
            funding_scheme=_coerce_to_list(raw.get("funding_scheme")),
            award_title=_coerce_to_list(raw.get("award_title")),
        ))

    return Funder(funder_name=raw.get("funder_name"), awards=awards)


def _ensure_funder_list(raw) -> List[Funder]:
    if not raw:
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        return [_parse_funder(raw)]
    if isinstance(raw, list):
        result = []
        for f in raw:
            if isinstance(f, dict):
                result.append(_parse_funder(f))
            elif f is not None:
                logging.warning("Dropping non-dict funder entry: %s", type(f).__name__)
        return result
    return []


def main():
    parser = argparse.ArgumentParser(description="Evaluate funding extraction predictions")
    parser.add_argument(
        "--predictions",
        default="adambuttrick/funding-extraction-lora-predictions",
        help="HF dataset repo with predictions, or path to local JSONL file",
    )
    parser.add_argument(
        "--split",
        default="predictions",
        help="Config/split name in the predictions repo (ignored for local files)",
    )
    parser.add_argument(
        "--funder-threshold",
        type=float,
        default=0.8,
        help="Fuzzy similarity threshold for funder matching",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Fuzzy similarity threshold for scheme/title matching",
    )
    parser.add_argument(
        "--id-match-mode",
        default="normalized",
        choices=["normalized", "exact"],
        help="Award ID matching mode",
    )
    parser.add_argument(
        "--gold-dataset",
        default="adambuttrick/funding-extraction-distillation-data",
        help="HF dataset with ground truth (used when predictions file lacks ground_truth_funders)",
    )
    parser.add_argument(
        "--gold-config",
        default="test",
        help="Config name for the gold dataset",
    )
    parser.add_argument(
        "--gold-split",
        default="test",
        help="Split name for the gold dataset",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: <predictions_basename>_results.json)",
    )
    args = parser.parse_args()

    if args.predictions.endswith(".jsonl") or args.predictions.endswith(".json"):
        with open(args.predictions, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f]
        source = args.predictions
    else:
        try:
            ds = load_dataset(args.predictions, args.split, split="test")
            lines = [dict(row) for row in ds]
        except Exception:
            filename = f"data/{args.split}.jsonl"
            path = hf_hub_download(args.predictions, filename, repo_type="dataset")
            with open(path) as f:
                lines = [json.loads(l) for l in f]
        source = f"{args.predictions}/{args.split}"

    print(f"Loaded {len(lines)} predictions from {source}")

    sample = lines[0] if lines else {}
    has_paired = "ground_truth_funders" in sample or "predicted_funders" in sample

    if not has_paired:
        print(f"Loading ground truth from {args.gold_dataset} ({args.gold_config}/{args.gold_split})...")
        gold_ds = load_dataset(args.gold_dataset, args.gold_config, split=args.gold_split)
        gold_lookup = {}
        for row in gold_ds:
            doi = row.get("doi", "").strip().lower()
            if doi:
                gold_lookup[doi] = row.get("funders", [])

        matched = 0
        for line in lines:
            doi = line.get("doi", "").strip().lower()
            gt = gold_lookup.get(doi, [])
            line["ground_truth_funders"] = gt
            line["predicted_funders"] = line.get("funders", [])
            if doi in gold_lookup:
                matched += 1

        print(f"Matched {matched}/{len(lines)} predictions to ground truth by DOI")

    print()

    buckets = {"complete": [], "truncated": []}

    for line in lines:
        gold_funders = _ensure_funder_list(line.get("ground_truth_funders"))
        pred_funders = _ensure_funder_list(line.get("predicted_funders"))

        funder_m, id_m, scheme_m, title_m = _evaluate_per_funder(
            gold_funders, pred_funders, args.funder_threshold, args.threshold, args.id_match_mode
        )

        raw = line.get("raw_output") or ""
        if "<think>" in raw and "</think>" not in raw:
            bucket = "truncated"
        else:
            bucket = "complete"
        buckets[bucket].append((funder_m, id_m, scheme_m, title_m))

    def _aggregate(items):
        agg = {}
        for level_name, idx in [("funder", 0), ("award_id", 1), ("scheme", 2), ("title", 3)]:
            gc = sum(m[idx].gold_count for m in items)
            pc = sum(m[idx].pred_count for m in items)
            matched = sum(m[idx].matched for m in items)
            agg[level_name] = _build_level_metrics(gc, pc, matched)
        return agg

    all_items = buckets["complete"] + buckets["truncated"]
    overall = _aggregate(all_items)

    def _print_metrics(label, agg):
        print(f"  {label}:")
        for level_name in ["funder", "award_id", "scheme", "title"]:
            m = agg[level_name]
            print(
                f"    {level_name:15s}  "
                f"P={m.precision:.4f}  R={m.recall:.4f}  F1={m.f1:.4f}  "
                f"F0.5={m.f0_5:.4f}  F1.5={m.f1_5:.4f}  "
                f"({m.matched}/{m.pred_count} pred, {m.matched}/{m.gold_count} gold)"
            )

    print("=== OVERALL ===")
    _print_metrics("All", overall)

    print(f"\n=== COMPLETE THINKING ({len(buckets['complete'])} examples) ===")
    if buckets["complete"]:
        _print_metrics("Complete", _aggregate(buckets["complete"]))

    print(f"\n=== TRUNCATED THINKING ({len(buckets['truncated'])} examples) ===")
    if buckets["truncated"]:
        _print_metrics("Truncated", _aggregate(buckets["truncated"]))

    # === DIAGNOSTIC: compare permissive, balanced (current), and strict ===
    print("\n" + "=" * 70)
    print("=== DIAGNOSTIC: PERMISSIVE vs BALANCED vs STRICT ===")
    print("=" * 70)

    def _run_with_sim(sim_fn, label):
        """Re-run full evaluation using a different similarity function."""
        bkts = {"complete": [], "truncated": []}
        for line in lines:
            gf = _ensure_funder_list(line.get("ground_truth_funders"))
            pf = _ensure_funder_list(line.get("predicted_funders"))
            fm, im, sm, tm = _evaluate_per_funder(
                gf, pf, args.funder_threshold, args.threshold, args.id_match_mode, sim_fn,
            )
            raw = line.get("raw_output") or ""
            bucket = "truncated" if ("<think>" in raw and "</think>" not in raw) else "complete"
            bkts[bucket].append((fm, im, sm, tm))
        return _aggregate(bkts["complete"] + bkts["truncated"])

    permissive_overall = _run_with_sim(similarity_permissive, "Permissive")
    strict_overall = _run_with_sim(similarity_strict, "Strict")

    print()
    print("--- PERMISSIVE (partial_ratio + token_set, no damping) ---")
    _print_metrics("Permissive", permissive_overall)
    print()
    print("--- BALANCED (length-damped + acronym detection) [default] ---")
    _print_metrics("Balanced", overall)
    print()
    print("--- STRICT (token_sort_ratio only) ---")
    _print_metrics("Strict", strict_overall)
    print()

    print("--- DELTAS ---")
    print(f"  {'level':15s}  {'perm→bal dF1':>14s}  {'bal→strict dF1':>14s}  {'perm→strict dF1':>14s}")
    for level_name in ["funder", "award_id", "scheme", "title"]:
        pm = permissive_overall[level_name]
        bm = overall[level_name]
        sm = strict_overall[level_name]
        print(
            f"  {level_name:15s}  "
            f"{bm.f1 - pm.f1:+14.4f}  "
            f"{sm.f1 - bm.f1:+14.4f}  "
            f"{sm.f1 - pm.f1:+14.4f}"
        )

    # --- Write JSON results ---
    def _agg_to_dict(agg):
        return {level: asdict(m) for level, m in agg.items()}

    results = {
        "source": source,
        "n_examples": len(lines),
        "n_complete": len(buckets["complete"]),
        "n_truncated": len(buckets["truncated"]),
        "config": {
            "funder_threshold": args.funder_threshold,
            "threshold": args.threshold,
            "id_match_mode": args.id_match_mode,
        },
        "permissive": _agg_to_dict(permissive_overall),
        "balanced": _agg_to_dict(overall),
        "strict": _agg_to_dict(strict_overall),
    }

    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(
            args.predictions if args.predictions.endswith((".jsonl", ".json"))
            else args.split
        ))[0]
        output_path = f"{base}_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
