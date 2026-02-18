import json

from funding_extractor.entities.io import (
    read_statements_by_document,
    write_results_json,
    load_results_json,
)
from funding_extractor.entities.models import ExtractionResult, FunderEntity, Award
from funding_extractor.statements.models import FundingStatement
from funding_extractor.models import DocumentResult, ProcessingParameters, ProcessingResults


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_read_statements_by_document_groups_correctly(tmp_path):
    jsonl = tmp_path / "stmts.jsonl"
    _write_jsonl(jsonl, [
        {"document_id": "doc-1", "statement": "Stmt A", "score": 10.0, "query": "q1"},
        {"document_id": "doc-1", "statement": "Stmt B", "score": 5.0, "query": "q2"},
        {"document_id": "doc-2", "statement": "Stmt C", "score": 1.0, "query": "q3"},
    ])

    grouped = read_statements_by_document(jsonl)
    assert len(grouped) == 2
    assert len(grouped["doc-1"]) == 2
    assert len(grouped["doc-2"]) == 1
    assert grouped["doc-1"][0]["statement"] == "Stmt A"
    assert grouped["doc-2"][0]["statement"] == "Stmt C"


def test_read_statements_by_document_empty_file(tmp_path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("", encoding="utf-8")
    grouped = read_statements_by_document(jsonl)
    assert grouped == {}


def test_write_and_load_results_json_roundtrip(tmp_path):
    output = tmp_path / "results.json"
    results = ProcessingResults(
        timestamp="2025-01-01T00:00:00",
        parameters=ProcessingParameters(input_path="/tmp/stmts.jsonl", provider="gemini"),
        results={
            "doc-1": DocumentResult(
                filename="doc-1",
                funding_statements=[
                    FundingStatement(statement="Funded by NSF", score=10.0, query="q1"),
                ],
                extraction_results=[
                    ExtractionResult(
                        statement="Funded by NSF",
                        funders=[FunderEntity(funder_name="NSF", awards=[Award(award_ids=["123"])])],
                    ),
                ],
            ),
        },
        summary={},
    )
    results.update_summary()

    write_results_json(results, output)
    assert output.exists()

    loaded = load_results_json(output)
    assert loaded.results["doc-1"].extraction_results[0].funders[0].funder_name == "NSF"
    assert loaded.summary["total_funders"] == 1
