from unittest.mock import patch

from funding_statement_extractor_cascade.cascade.batch import extract_funding_batch
from funding_statement_extractor_cascade.cascade.types import (
    CascadeConfig,
    CascadeResult,
    DocPayload,
)


def _fake_extractor(call_log):
    def _extract(text, models, cfg):
        call_log.append(text)
        return CascadeResult(funding_statement=f"r:{text}", rough_span="x",
                              chunk_prob=0.9, no_answer_prob=0.1, latency_ms=1.0)
    return _extract


def test_batch_yields_one_result_per_doc():
    calls = []
    docs = [DocPayload(doc_id="a", text="x"),
            DocPayload(doc_id="b", text="y")]
    with patch("funding_statement_extractor_cascade.cascade.batch.extract_funding",
               side_effect=_fake_extractor(calls)):
        results = list(extract_funding_batch(documents=docs, models=None, cfg=CascadeConfig()))
    assert [r.doc_id for r in results] == ["a", "b"]
    assert [r.result.funding_statement for r in results] == ["r:x", "r:y"]


def test_batch_passes_through_metadata():
    docs = [DocPayload(doc_id="a", text="x",
                       metadata={"input_file": "f.parquet", "row_idx": 7})]
    with patch("funding_statement_extractor_cascade.cascade.batch.extract_funding",
               side_effect=_fake_extractor([])):
        result = list(extract_funding_batch(documents=docs, models=None, cfg=CascadeConfig()))[0]
    assert result.metadata == {"input_file": "f.parquet", "row_idx": 7}


def test_batch_isolates_per_doc_exception():
    def _broken(text, models, cfg):
        if text == "boom":
            raise RuntimeError("boom!")
        return CascadeResult(funding_statement="ok", rough_span="ok",
                              chunk_prob=0.9, no_answer_prob=0.1, latency_ms=1.0)

    docs = [
        DocPayload(doc_id="d1", text="ok"),
        DocPayload(doc_id="d2", text="boom"),
        DocPayload(doc_id="d3", text="ok"),
    ]
    with patch("funding_statement_extractor_cascade.cascade.batch.extract_funding",
               side_effect=_broken):
        results = list(extract_funding_batch(documents=docs, models=None, cfg=CascadeConfig()))
    assert [r.doc_id for r in results] == ["d1", "d2", "d3"]
    assert results[1].result.funding_statement == ""
    assert results[1].result.error is not None
    assert "RuntimeError" in results[1].result.error
