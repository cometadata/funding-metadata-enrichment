import json
from pathlib import Path
from unittest.mock import patch

from funding_extractor.cli.main import build_parser


def test_parser_has_subcommands():
    parser = build_parser()
    # Verify subcommands exist by parsing each one with minimal args
    args = parser.parse_args(["statements", "-i", "input.md"])
    assert args.command == "statements"

    args = parser.parse_args(["entities", "-i", "stmts.jsonl"])
    assert args.command == "entities"

    args = parser.parse_args(["pipeline", "-i", "input.md"])
    assert args.command == "pipeline"


def test_statements_subcommand_has_retrieval_model():
    parser = build_parser()
    args = parser.parse_args(["statements", "-i", "x", "--retrieval-model", "my-model"])
    assert args.retrieval_model == "my-model"


def test_entities_subcommand_has_model():
    parser = build_parser()
    args = parser.parse_args(["entities", "-i", "x", "--model", "gpt-4o"])
    assert args.model == "gpt-4o"


def test_pipeline_subcommand_has_both_flags():
    parser = build_parser()
    args = parser.parse_args([
        "pipeline", "-i", "input.md",
        "--retrieval-model", "my-model",
        "--model", "gpt-4o",
    ])
    assert args.retrieval_model == "my-model"
    assert args.model == "gpt-4o"


@patch("funding_extractor.statements.cli.run")
def test_statements_command_dispatches(mock_run, tmp_path):
    input_file = tmp_path / "doc.md"
    input_file.write_text("content", encoding="utf-8")

    from funding_extractor.cli.main import main_with_args, build_parser
    parser = build_parser()
    args = parser.parse_args(["statements", "-i", str(input_file), "-o", str(tmp_path / "out.jsonl")])
    main_with_args(args)
    mock_run.assert_called_once_with(args)


@patch("funding_extractor.entities.cli.run")
def test_entities_command_dispatches(mock_run, tmp_path):
    input_file = tmp_path / "stmts.jsonl"
    input_file.write_text("{}\n", encoding="utf-8")

    from funding_extractor.cli.main import main_with_args, build_parser
    parser = build_parser()
    args = parser.parse_args(["entities", "-i", str(input_file), "-o", str(tmp_path / "out.json")])
    main_with_args(args)
    mock_run.assert_called_once_with(args)
