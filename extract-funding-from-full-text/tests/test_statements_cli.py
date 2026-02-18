import argparse
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from funding_extractor.statements.cli import add_arguments, run
from funding_extractor.statements.models import FundingStatement


def test_add_arguments_creates_required_args():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", "input.md", "-o", "out.jsonl"])
    assert args.input == "input.md"
    assert args.output == "out.jsonl"


def test_add_arguments_default_output():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", "input.md"])
    assert args.output == "funding_statements.jsonl"


def test_add_arguments_retrieval_model_flag():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", "input.md", "--retrieval-model", "my-model"])
    assert args.retrieval_model == "my-model"


def test_add_arguments_extraction_flags():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([
        "-i", "input.md",
        "--threshold", "15.0",
        "--top-k", "10",
        "--normalize",
        "--heal-markdown",
        "--statements-only",
        "--enable-pattern-rescue",
        "--enable-post-filter",
    ])
    assert args.threshold == 15.0
    assert args.top_k == 10
    assert args.normalize is True
    assert args.heal_markdown is True
    assert args.statements_only is True
    assert args.enable_pattern_rescue is True
    assert args.enable_post_filter is True


def test_run_statements_only_writes_jsonl(tmp_path):
    input_file = tmp_path / "stmt.md"
    input_file.write_text("This work was funded by NIH grant R01-GM123456.", encoding="utf-8")
    output_file = tmp_path / "out.jsonl"

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([
        "-i", str(input_file),
        "-o", str(output_file),
        "--statements-only",
    ])

    run(args)

    assert output_file.exists()
    records = []
    with open(output_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) == 1
    assert records[0]["statement"] == "This work was funded by NIH grant R01-GM123456."
    assert records[0]["score"] == 1.0
    assert records[0]["query"] == "statements-only"
