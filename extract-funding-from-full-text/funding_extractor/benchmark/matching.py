# funding_extractor/benchmark/matching.py
"""Text matching utilities for benchmark evaluation."""

import re
from typing import Callable, Iterable, List, Optional, Tuple

import ftfy
from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("\u22c6", "").replace("\u2022", "").replace("*", "")
    text = " ".join(text.split())
    text = re.sub(r"\\([a-z])", r"\1", text)
    return text.strip()


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


def best_match_score(target: str, candidates: Iterable[str]) -> float:
    best = 0.0
    for cand in candidates:
        score = similarity(target, cand)
        if score > best:
            best = score
    return best


def normalize_award_id(award_id: str) -> str:
    aid = award_id.strip().upper()
    aid = re.sub(r"[\s\-/]", "", aid)
    return aid


def award_ids_match(pred_id: str, gold_id: str, mode: str = "normalized") -> bool:
    if mode == "exact":
        return pred_id.strip().upper() == gold_id.strip().upper()
    return normalize_award_id(pred_id) == normalize_award_id(gold_id)


def greedy_match(
    gold_items: List[str],
    pred_items: List[str],
    score_fn: Callable[[str, str], float],
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Greedy 1-to-1 matching between gold and pred items.

    Builds all pairwise scores, keeps those >= threshold,
    sorts descending, and greedily assigns matches consuming
    each side exactly once.

    Returns list of (gold_index, pred_index, score) tuples.
    """
    pairs = []
    for gi, g in enumerate(gold_items):
        for pi, p in enumerate(pred_items):
            score = score_fn(g, p)
            if score >= threshold:
                pairs.append((gi, pi, score))

    pairs.sort(key=lambda x: x[2], reverse=True)

    matched = []
    used_gold: set = set()
    used_pred: set = set()
    for gi, pi, score in pairs:
        if gi not in used_gold and pi not in used_pred:
            matched.append((gi, pi, score))
            used_gold.add(gi)
            used_pred.add(pi)

    return matched


def greedy_match_ids(
    gold_items: List[str],
    pred_items: List[str],
    id_match_mode: str = "normalized",
) -> List[Tuple[int, int, float]]:
    """Greedy 1-to-1 matching for award IDs (boolean match -> score 1.0)."""
    def score_fn(g: str, p: str) -> float:
        return 1.0 if award_ids_match(p, g, mode=id_match_mode) else 0.0
    return greedy_match(gold_items, pred_items, score_fn, threshold=1.0)


def doi_from_filename(filename: str) -> Optional[str]:
    stem = filename
    if stem.endswith(".md"):
        stem = stem[:-3]
    # DOI pattern: starts with 10.NNNN
    if not re.match(r"^10\.\d{4,}", stem):
        return None
    # Replace first hyphen with slash (DOI convention)
    idx = stem.find("-")
    if idx == -1:
        return stem
    return stem[:idx] + "/" + stem[idx + 1:]
