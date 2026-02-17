# funding_extractor/benchmark/matching.py
"""Text matching utilities for benchmark evaluation."""

import re
from typing import Iterable, Optional

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
        return pred_id.strip().lower() == gold_id.strip().lower()
    return normalize_award_id(pred_id) == normalize_award_id(gold_id)


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
