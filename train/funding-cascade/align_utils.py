"""Shared utilities: fuzzy span alignment of a gold funding statement into a
source text representation.

The funding-extraction dataset's gold strings were written by frontier
labelers and are not always verbatim substrings of the source — labelers
normalize whitespace, strip LaTeX markers, and occasionally join paragraphs.
`align_stmt` first tries a verbatim substring match; if that fails it
performs an anchor-based fuzzy alignment.
"""
import re
from typing import Optional, Tuple

from rapidfuzz import fuzz


def find_verbatim(stmt: str, text: str) -> Optional[Tuple[int, int, float]]:
    idx = text.find(stmt)
    if idx >= 0:
        return idx, idx + len(stmt), 100.0
    return None


def find_fuzzy_window(stmt: str, text: str, min_score: float = 75.0
                       ) -> Optional[Tuple[int, int, float]]:
    """Anchor + scan: find a window in `text` whose substring best matches stmt."""
    if not stmt or not text:
        return None
    L = len(stmt)
    tokens = re.findall(r"\w+", stmt)
    if len(tokens) < 3:
        return None
    text_lower = text.lower()

    candidates = []

    # Anchor 1: first 4 tokens
    anchor = " ".join(tokens[:4]).lower()
    idx = text_lower.find(anchor)
    while idx != -1 and len(candidates) < 20:
        start = max(0, idx - 50)
        end = min(len(text), idx + L + 100)
        candidates.append((start, end))
        idx = text_lower.find(anchor, idx + 1)

    # Anchor 2: last 4 tokens
    if len(tokens) >= 8:
        anchor2 = " ".join(tokens[-4:]).lower()
        idx = text_lower.find(anchor2)
        while idx != -1 and len(candidates) < 30:
            start = max(0, idx - L - 100)
            end = min(len(text), idx + 100)
            candidates.append((start, end))
            idx = text_lower.find(anchor2, idx + 1)

    # Anchor 3: distinctive long token (e.g., grant id)
    long_toks = [t for t in tokens if len(t) >= 8 and t.lower() not in {
        "supported", "research", "national", "foundation", "university"
    }]
    if long_toks and len(candidates) < 20:
        for lt in long_toks[:3]:
            lt_l = lt.lower()
            idx = text_lower.find(lt_l)
            while idx != -1 and len(candidates) < 30:
                start = max(0, idx - L // 2)
                end = min(len(text), idx + L)
                candidates.append((start, end))
                idx = text_lower.find(lt_l, idx + 1)

    if not candidates:
        return None

    best_score = -1
    best = None
    for s, e in candidates:
        snippet = text[s:e]
        if len(snippet) < L * 0.4:
            continue
        score = fuzz.partial_ratio(stmt.lower(), snippet.lower())
        if score > best_score:
            best_score = score
            best = (s, e)

    if best_score < min_score or best is None:
        return None

    s, e = best
    snippet = text[s:e]
    try:
        align = fuzz.partial_ratio_alignment(stmt.lower(), snippet.lower(),
                                              score_cutoff=min_score)
        if align is not None:
            return (s + align.dest_start, s + align.dest_end, float(align.score))
    except Exception:
        pass
    return (s, e, float(best_score))


def align_stmt(stmt: str, text: str) -> Optional[Tuple[int, int, float]]:
    """Try verbatim substring match first; fall back to fuzzy alignment."""
    if not stmt or not text:
        return None
    v = find_verbatim(stmt, text)
    if v is not None:
        return v
    return find_fuzzy_window(stmt, text)
