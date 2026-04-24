import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_statement_extractor.statements.batch_extraction import (
    DocPayload,
    BatchResult,
    _pre_task,
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


def test_pre_task_splits_paragraphs_no_prefilter():
    doc = DocPayload(doc_id="d1", text="Para one.\n\nPara two.\n\nPara three.")
    out = _pre_task(doc, enable_paragraph_prefilter=False)
    assert out.doc.doc_id == "d1"
    assert out.paragraphs == ["Para one.", "Para two.", "Para three."]
    assert out.original_indices == [0, 1, 2]
    assert out.full_paragraphs == ["Para one.", "Para two.", "Para three."]
    assert out.error is None


def test_pre_task_prefilters_keeps_funding_paragraphs_with_neighbors():
    text = (
        "Intro paragraph with no signal.\n\n"
        "Methods paragraph with no signal.\n\n"
        "This work was supported by NSF grant 12345.\n\n"
        "Conclusion paragraph with no signal.\n\n"
        "Unrelated paragraph at the end."
    )
    doc = DocPayload(doc_id="d1", text=text)
    out = _pre_task(doc, enable_paragraph_prefilter=True)
    # Funding sentence is index 2; prefilter keeps it plus neighbors 1 and 3
    assert out.original_indices == [1, 2, 3]
    assert "supported by NSF" in out.paragraphs[1]
    # full_paragraphs preserved untouched
    assert len(out.full_paragraphs) == 5


def test_pre_task_empty_text():
    doc = DocPayload(doc_id="d1", text="")
    out = _pre_task(doc, enable_paragraph_prefilter=True)
    assert out.paragraphs == []
    assert out.original_indices == []
    assert out.full_paragraphs == []
    assert out.error is None


def test_pre_task_records_pre_ms():
    doc = DocPayload(doc_id="d1", text="A.\n\nB.")
    out = _pre_task(doc, enable_paragraph_prefilter=False)
    assert out.pre_ms >= 0.0
