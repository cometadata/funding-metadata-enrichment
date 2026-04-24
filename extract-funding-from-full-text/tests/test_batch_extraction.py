import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_statement_extractor.statements.batch_extraction import (
    DocPayload,
    BatchResult,
    _pre_task,
    _PreOut,
    _PostIn,
    _post_task,
)
from funding_statement_extractor.statements.extraction import (
    SemanticExtractionService,
)
from funding_statement_extractor.config.loader import load_funding_patterns


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


def _legacy_emit_for_doc(paragraphs, topk_scores, topk_idx, query_names,
                        threshold, regex_floor, compiled_patterns, neg_patterns):
    """Replica of the (query, rank) loop in extraction.py:365-397, used as oracle."""
    from funding_statement_extractor.statements.models import FundingStatement
    from funding_statement_extractor.statements.extraction import SemanticExtractionService as S
    seen = set()
    out = []
    for q_pos, query_name in enumerate(query_names):
        for rank in range(len(topk_idx[q_pos])):
            local_idx = topk_idx[q_pos][rank]
            score = float(topk_scores[q_pos][rank])
            paragraph = paragraphs[local_idx]
            if not S._is_likely_funding_statement(
                paragraph, score, threshold, compiled_patterns, neg_patterns,
                regex_match_score_floor=regex_floor,
            ):
                continue
            if S._should_extract_full_paragraph(paragraph, score):
                statement_text = paragraph.strip()
            elif len(paragraph) > 1000:
                statement_text = S._extract_funding_from_long_paragraph(paragraph)
            else:
                funding_sentences = S._extract_funding_sentences(paragraph)
                if funding_sentences:
                    statement_text = " ".join(funding_sentences)
                else:
                    continue
            if statement_text in seen or len(statement_text) <= 20:
                continue
            seen.add(statement_text)
            out.append(FundingStatement(
                statement=statement_text, score=score, query=query_name,
                paragraph_idx=local_idx,
            ))
    return out


def _make_post_in_fixture():
    """5 paragraphs, 3 queries, top_k=3 — overlapping picks across queries."""
    paragraphs = [
        "Random unrelated paragraph one with no funding signal.",
        "This work was supported by NSF grant DMR-1234567 and DMR-7654321.",
        "Methods section paragraph with no relevant content.",
        "We acknowledge financial support from the European Research Council under grant 12345.",
        "Conclusion paragraph with the word fund mentioned but not in a funding context.",
    ]
    pre = _PreOut(
        doc=DocPayload(doc_id="d1", text="\n\n".join(paragraphs)),
        paragraphs=paragraphs,
        original_indices=[0, 1, 2, 3, 4],
        full_paragraphs=paragraphs,
        enqueue_ts=0.0,
        pre_ms=0.0,
    )
    # Hand-constructed top-k that overlaps across queries
    topk_idx = [
        [1, 3, 0],   # funding_statement
        [3, 1, 4],   # grant_acknowledgment
        [1, 3, 2],   # financial_support
    ]
    topk_scores = [
        [12.5, 11.8, 9.5],
        [11.2, 10.9, 9.0],
        [10.5, 10.4, 8.0],
    ]
    return pre, topk_scores, topk_idx, ["funding_statement", "grant_acknowledgment", "financial_support"]


def test_post_task_byte_identical_to_legacy_loop():
    pre, topk_s, topk_i, q_names = _make_post_in_fixture()
    pos_pat, neg_pat = load_funding_patterns()
    compiled_pos = [re.compile(p, re.IGNORECASE) for p in pos_pat]
    compiled_neg = [re.compile(p, re.IGNORECASE) for p in neg_pat]
    item = _PostIn(
        pre=pre, topk_scores=topk_s, topk_idx=topk_i, query_names=q_names,
        queue_wait_ms=0.0, gpu_ms=0.0,
    )
    new_result = _post_task(
        item, top_k=3, threshold=10.0, regex_match_score_floor=11.0,
        compiled_patterns=compiled_pos, negative_patterns=compiled_neg,
    )
    legacy = _legacy_emit_for_doc(
        pre.paragraphs, topk_s, topk_i, q_names,
        threshold=10.0, regex_floor=11.0,
        compiled_patterns=compiled_pos, neg_patterns=compiled_neg,
    )
    # Byte-identity: same statement text, score, query, paragraph_idx
    assert len(new_result.statements) == len(legacy), \
        f"count mismatch: new={len(new_result.statements)} legacy={len(legacy)}"
    for new_stmt, leg_stmt in zip(new_result.statements, legacy):
        assert new_stmt.statement == leg_stmt.statement
        assert new_stmt.score == leg_stmt.score
        assert new_stmt.query == leg_stmt.query
        assert new_stmt.paragraph_idx == leg_stmt.paragraph_idx


def test_post_task_zero_paragraphs():
    pre = _PreOut(
        doc=DocPayload(doc_id="d1", text=""),
        paragraphs=[], original_indices=[], full_paragraphs=[],
        enqueue_ts=0.0, pre_ms=0.0,
    )
    item = _PostIn(pre=pre, topk_scores=None, topk_idx=None, query_names=[],
                   queue_wait_ms=0.0, gpu_ms=0.0)
    pos_pat, neg_pat = load_funding_patterns()
    compiled_pos = [re.compile(p, re.IGNORECASE) for p in pos_pat]
    compiled_neg = [re.compile(p, re.IGNORECASE) for p in neg_pat]
    result = _post_task(
        item, top_k=5, threshold=10.0, regex_match_score_floor=11.0,
        compiled_patterns=compiled_pos, negative_patterns=compiled_neg,
    )
    assert result.statements == []
    assert result.error is None
    assert result.doc_id == "d1"


def test_post_task_paragraph_idx_uses_original_index():
    """When prefilter drops paragraphs, paragraph_idx must reflect the original position."""
    full = ["junk", "junk", "supported by NSF grant 12345"]
    pre = _PreOut(
        doc=DocPayload(doc_id="d1", text="\n\n".join(full)),
        paragraphs=[full[2]],            # only paragraph 2 survives prefilter
        original_indices=[2],
        full_paragraphs=full,
        enqueue_ts=0.0, pre_ms=0.0,
    )
    item = _PostIn(
        pre=pre,
        topk_scores=[[12.0]],
        topk_idx=[[0]],                  # local index 0 → original index 2
        query_names=["funding_statement"],
        queue_wait_ms=0.0, gpu_ms=0.0,
    )
    pos_pat, neg_pat = load_funding_patterns()
    compiled_pos = [re.compile(p, re.IGNORECASE) for p in pos_pat]
    compiled_neg = [re.compile(p, re.IGNORECASE) for p in neg_pat]
    result = _post_task(
        item, top_k=1, threshold=10.0, regex_match_score_floor=11.0,
        compiled_patterns=compiled_pos, negative_patterns=compiled_neg,
    )
    assert len(result.statements) == 1
    assert result.statements[0].paragraph_idx == 2
