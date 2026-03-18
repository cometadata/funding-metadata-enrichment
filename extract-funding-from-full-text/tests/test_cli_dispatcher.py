from unittest.mock import patch

from funding_statement_extractor.cli.main import build_parser


def test_parser_has_retrieval_model():
    parser = build_parser()
    args = parser.parse_args(["-i", "x", "--retrieval-model", "my-model"])
    assert args.retrieval_model == "my-model"


def test_parser_has_required_input():
    parser = build_parser()
    args = parser.parse_args(["-i", "input.md"])
    assert args.input == "input.md"


@patch("funding_statement_extractor.statements.cli.run")
def test_main_dispatches_to_statements(mock_run, tmp_path):
    input_file = tmp_path / "doc.md"
    input_file.write_text("content", encoding="utf-8")

    from funding_statement_extractor.cli.main import main, build_parser
    parser = build_parser()
    args = parser.parse_args(["-i", str(input_file), "-o", str(tmp_path / "out.jsonl")])

    from funding_statement_extractor.statements import cli as statements_cli
    statements_cli.run(args)
    mock_run.assert_called_once_with(args)
