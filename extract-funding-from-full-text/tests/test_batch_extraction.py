import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_statement_extractor.statements.batch_extraction import (
    DocPayload,
    BatchResult,
)


def test_doc_payload_carries_metadata():
    doc = DocPayload(doc_id="abc", text="hello", metadata={"bucket": "clean"})
    assert doc.doc_id == "abc"
    assert doc.text == "hello"
    assert doc.metadata == {"bucket": "clean"}


def test_doc_payload_metadata_optional():
    doc = DocPayload(doc_id="abc", text="hello")
    assert doc.metadata is None


def test_batch_result_default_shape():
    r = BatchResult(doc_id="abc", statements=[], enqueue_ts=0.0, yield_ts=1.0)
    assert r.error is None
    assert r.metadata is None
    assert r.timings is None
    assert r.statements == []
