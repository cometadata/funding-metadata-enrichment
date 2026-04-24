import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Pattern, Set, Tuple

from funding_statement_extractor.statements.extraction import (
    SemanticExtractionService,
    _prefilter_paragraphs,
    _split_into_paragraphs,
)
from funding_statement_extractor.statements.models import FundingStatement


@dataclass
class DocPayload:
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    doc_id: str
    statements: List[FundingStatement]
    enqueue_ts: float
    yield_ts: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, float]] = None


# Internal payload shapes (private; not exported)
@dataclass
class _PreOut:
    doc: DocPayload
    paragraphs: List[str]                 # paragraphs to encode
    original_indices: List[int]           # index back into full_paragraphs
    full_paragraphs: List[str]            # original paragraphs (for extraction)
    enqueue_ts: float
    pre_ms: float
    error: Optional[str] = None


@dataclass
class _PostIn:
    pre: _PreOut
    topk_scores: Optional[List[List[float]]]   # [num_q][k]
    topk_idx: Optional[List[List[int]]]        # [num_q][k]; indexes into pre.paragraphs
    query_names: List[str]
    queue_wait_ms: float
    gpu_ms: float


def _length_sort_permutation(paragraphs: List[str]) -> List[int]:
    """Return indices that sort paragraphs by character count (cheap proxy for token count).

    Used so each model.encode sub-batch contains similar-length paragraphs and pads
    minimally. Stable sort so ties are deterministic.
    """
    return sorted(range(len(paragraphs)), key=lambda i: len(paragraphs[i]))


def _apply_inverse_permutation(items: List[Any], perm: List[int]) -> List[Any]:
    """Reverse the permutation: items[i] is the value at sorted position i;
    return a list where position perm[i] holds items[i]."""
    out: List[Any] = [None] * len(items)
    for sorted_pos, original_pos in enumerate(perm):
        out[original_pos] = items[sorted_pos]
    return out


def _pre_task(doc: DocPayload, *, enable_paragraph_prefilter: bool) -> _PreOut:
    t0 = time.perf_counter()
    try:
        paragraphs = _split_into_paragraphs(doc.text)
        if enable_paragraph_prefilter and paragraphs:
            kept, original_indices = _prefilter_paragraphs(paragraphs)
        else:
            kept = paragraphs
            original_indices = list(range(len(paragraphs)))
        return _PreOut(
            doc=doc,
            paragraphs=kept,
            original_indices=original_indices,
            full_paragraphs=paragraphs,
            enqueue_ts=time.monotonic(),
            pre_ms=(time.perf_counter() - t0) * 1000.0,
        )
    except Exception as exc:
        return _PreOut(
            doc=doc,
            paragraphs=[],
            original_indices=[],
            full_paragraphs=[],
            enqueue_ts=time.monotonic(),
            pre_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _extract_statement_text(paragraph: str, score: float) -> Optional[str]:
    """Mirror of the in-loop extraction in SemanticExtractionService.extract_funding_statements."""
    s = SemanticExtractionService
    if s._should_extract_full_paragraph(paragraph, score):
        return paragraph.strip()
    if len(paragraph) > 1000:
        return s._extract_funding_from_long_paragraph(paragraph)
    funding_sentences = s._extract_funding_sentences(paragraph)
    if funding_sentences:
        return " ".join(funding_sentences)
    return None


def _post_task(
    item: _PostIn,
    *,
    top_k: int,
    threshold: float,
    regex_match_score_floor: float,
    compiled_patterns: List[Pattern],
    negative_patterns: List[Pattern],
) -> BatchResult:
    t0 = time.perf_counter()
    pre = item.pre
    doc_id = pre.doc.doc_id

    if pre.error is not None:
        return BatchResult(
            doc_id=doc_id, statements=[], error=pre.error,
            metadata=pre.doc.metadata,
            enqueue_ts=pre.enqueue_ts, yield_ts=time.monotonic(),
            timings={"pre_ms": pre.pre_ms, "queue_wait_ms": item.queue_wait_ms,
                     "gpu_ms": item.gpu_ms, "post_ms": 0.0},
        )

    if item.topk_scores is None or not pre.paragraphs:
        return BatchResult(
            doc_id=doc_id, statements=[], metadata=pre.doc.metadata,
            enqueue_ts=pre.enqueue_ts, yield_ts=time.monotonic(),
            timings={"pre_ms": pre.pre_ms, "queue_wait_ms": item.queue_wait_ms,
                     "gpu_ms": item.gpu_ms, "post_ms": 0.0},
        )

    full_paragraphs = pre.full_paragraphs
    original_indices = pre.original_indices

    # Cache regex hits per *original* paragraph index to avoid re-running regex
    # across the (query, rank) pairs that pick the same paragraph.
    regex_cache: Dict[int, Tuple[bool, bool]] = {}

    def regex_check(orig_idx: int) -> Tuple[bool, bool]:
        if orig_idx not in regex_cache:
            s = full_paragraphs[orig_idx].lower()
            neg_hit = any(p.search(s) for p in negative_patterns)
            pos_hit = any(p.search(s) for p in compiled_patterns)
            regex_cache[orig_idx] = (neg_hit, pos_hit)
        return regex_cache[orig_idx]

    statements: List[FundingStatement] = []
    seen_text: Set[str] = set()

    try:
        for q_pos, query_name in enumerate(item.query_names):
            for rank in range(len(item.topk_idx[q_pos])):
                local_idx = item.topk_idx[q_pos][rank]
                score = float(item.topk_scores[q_pos][rank])
                orig_idx = original_indices[local_idx]
                paragraph = full_paragraphs[orig_idx]

                # Gating — preserves _is_likely_funding_statement semantics exactly.
                if score > 14.0:
                    accepted = True
                else:
                    neg, pos = regex_check(orig_idx)
                    if neg:
                        accepted = False
                    elif pos:
                        accepted = score > regex_match_score_floor
                    else:
                        accepted = False  # current behaviour: no positive pattern → reject
                if not accepted:
                    continue

                statement_text = _extract_statement_text(paragraph, score)
                if not statement_text or len(statement_text) <= 20:
                    continue
                if statement_text in seen_text:
                    continue
                seen_text.add(statement_text)
                statements.append(FundingStatement(
                    statement=statement_text,
                    score=score,
                    query=query_name,
                    paragraph_idx=orig_idx,   # match legacy: para_id = original_indices[encoded_idx]
                ))
    except Exception as exc:
        return BatchResult(
            doc_id=doc_id, statements=statements, error=f"{type(exc).__name__}: {exc}",
            metadata=pre.doc.metadata,
            enqueue_ts=pre.enqueue_ts, yield_ts=time.monotonic(),
            timings={"pre_ms": pre.pre_ms, "queue_wait_ms": item.queue_wait_ms,
                     "gpu_ms": item.gpu_ms, "post_ms": (time.perf_counter() - t0) * 1000.0},
        )

    return BatchResult(
        doc_id=doc_id, statements=statements, metadata=pre.doc.metadata,
        enqueue_ts=pre.enqueue_ts, yield_ts=time.monotonic(),
        timings={"pre_ms": pre.pre_ms, "queue_wait_ms": item.queue_wait_ms,
                 "gpu_ms": item.gpu_ms, "post_ms": (time.perf_counter() - t0) * 1000.0},
    )
