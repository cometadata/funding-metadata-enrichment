import time
from typing import Iterable, Iterator

from funding_statement_extractor_cascade.cascade.loader import CascadeModels
from funding_statement_extractor_cascade.cascade.pipeline import extract_funding
from funding_statement_extractor_cascade.cascade.types import (
    BatchResult,
    CascadeConfig,
    CascadeResult,
    DocPayload,
)


def extract_funding_batch(
    documents: Iterable[DocPayload],
    *,
    models: CascadeModels,
    cfg: CascadeConfig,
) -> Iterator[BatchResult]:
    """Iterate docs sequentially through the cascade, yielding one BatchResult each.

    Per-doc exceptions are captured as `CascadeResult.error` so the iterator
    never terminates mid-stream.
    """
    for doc in documents:
        enqueue_ts = time.monotonic()
        try:
            result = extract_funding(doc.text, models, cfg)
        except Exception as exc:
            result = CascadeResult(
                funding_statement="", rough_span="",
                chunk_prob=None, no_answer_prob=None,
                latency_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        yield BatchResult(
            doc_id=doc.doc_id,
            result=result,
            enqueue_ts=enqueue_ts,
            yield_ts=time.monotonic(),
            metadata=doc.metadata,
        )
