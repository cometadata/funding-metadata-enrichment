import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

from funding_statement_extractor.statements.extraction import (
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
