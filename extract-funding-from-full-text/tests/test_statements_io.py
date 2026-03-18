import json

from funding_statement_extractor.statements.io import write_statements_jsonl, read_statements_jsonl
from funding_statement_extractor.statements.models import FundingStatement


def test_write_and_read_roundtrip(tmp_path):
    output = tmp_path / "stmts.jsonl"
    stmts = [
        ("doc-1", FundingStatement(statement="Funded by NSF.", score=14.2, query="q1", paragraph_idx=3)),
        ("doc-1", FundingStatement(statement="NIH grant R01.", score=10.0, query="q2")),
        ("doc-2", FundingStatement(statement="ERC support.", score=1.0, query="statements-only")),
    ]

    count = write_statements_jsonl(output, stmts)
    assert count == 3
    assert output.exists()

    records = list(read_statements_jsonl(output))
    assert len(records) == 3

    assert records[0]["document_id"] == "doc-1"
    assert records[0]["statement"] == "Funded by NSF."
    assert records[0]["score"] == 14.2
    assert records[0]["query"] == "q1"
    assert records[0]["paragraph_idx"] == 3
    assert records[0]["is_problematic"] is False
    assert records[0]["original"] is None

    assert records[2]["document_id"] == "doc-2"
    assert records[2]["statement"] == "ERC support."


def test_write_empty_list(tmp_path):
    output = tmp_path / "empty.jsonl"
    count = write_statements_jsonl(output, [])
    assert count == 0
    assert output.exists()
    assert list(read_statements_jsonl(output)) == []


def test_read_preserves_original_field(tmp_path):
    output = tmp_path / "stmts.jsonl"
    stmt = FundingStatement(
        statement="Normalized text",
        original="Original  text",
        score=5.0,
        query="q",
        is_problematic=True,
    )
    write_statements_jsonl(output, [("doc-1", stmt)])

    records = list(read_statements_jsonl(output))
    assert records[0]["original"] == "Original  text"
    assert records[0]["is_problematic"] is True


def test_read_skips_blank_lines(tmp_path):
    output = tmp_path / "stmts.jsonl"
    line = json.dumps({"document_id": "d1", "statement": "text", "score": 1.0, "query": "q"})
    output.write_text(f"{line}\n\n{line}\n", encoding="utf-8")

    records = list(read_statements_jsonl(output))
    assert len(records) == 2
