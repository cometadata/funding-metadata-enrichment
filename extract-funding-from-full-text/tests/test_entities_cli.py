import argparse
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from funding_extractor.entities.cli import add_arguments, run
from funding_extractor.entities.models import ExtractionResult, FunderEntity, Award


def test_add_arguments_creates_required_args():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", "stmts.jsonl", "-o", "results.json"])
    assert args.input == "stmts.jsonl"
    assert args.output == "results.json"


def test_add_arguments_default_output():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", "stmts.jsonl"])
    assert args.output == "funding_results.json"


def test_add_arguments_provider_flags():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([
        "-i", "stmts.jsonl",
        "--provider", "openai",
        "--model", "gpt-4o",
        "--api-key", "sk-test",
        "--timeout", "120",
    ])
    assert args.provider == "openai"
    assert args.model == "gpt-4o"
    assert args.api_key == "sk-test"
    assert args.timeout == 120


@patch("funding_extractor.entities.cli.extract_structured_entities")
def test_run_produces_json_output(mock_extract, tmp_path):
    mock_extract.return_value = ExtractionResult(
        statement="Funded by NSF grant 123.",
        funders=[FunderEntity(funder_name="NSF", awards=[Award(award_ids=["123"])])],
    )

    jsonl_input = tmp_path / "stmts.jsonl"
    records = [
        {"document_id": "doc-1", "statement": "Funded by NSF grant 123.", "score": 10.0, "query": "q1"},
    ]
    jsonl_input.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    output = tmp_path / "results.json"

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([
        "-i", str(jsonl_input),
        "-o", str(output),
        "--provider", "gemini",
        "--skip-model-validation",
    ])

    run(args)

    assert output.exists()
    with open(output, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert "doc-1" in data["results"]
    doc = data["results"]["doc-1"]
    assert len(doc["funding_statements"]) == 1
    assert len(doc["extractions"]) == 1
    assert doc["extractions"][0]["funders"][0]["funder_name"] == "NSF"


@patch("funding_extractor.entities.cli.extract_structured_entities")
def test_run_groups_by_document_id(mock_extract, tmp_path):
    mock_extract.return_value = ExtractionResult(
        statement="mock",
        funders=[FunderEntity(funder_name="Generic")],
    )

    jsonl_input = tmp_path / "stmts.jsonl"
    records = [
        {"document_id": "doc-1", "statement": "Stmt A", "score": 10.0, "query": "q"},
        {"document_id": "doc-1", "statement": "Stmt B", "score": 5.0, "query": "q"},
        {"document_id": "doc-2", "statement": "Stmt C", "score": 1.0, "query": "q"},
    ]
    jsonl_input.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    output = tmp_path / "results.json"
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([
        "-i", str(jsonl_input),
        "-o", str(output),
        "--provider", "gemini",
        "--skip-model-validation",
    ])

    run(args)

    with open(output, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert len(data["results"]) == 2
    assert len(data["results"]["doc-1"]["funding_statements"]) == 2
    assert len(data["results"]["doc-2"]["funding_statements"]) == 1
