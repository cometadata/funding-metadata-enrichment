from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

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
