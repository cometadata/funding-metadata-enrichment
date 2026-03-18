import argparse
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from funding_statement_extractor.statements.cli import add_arguments, run
from funding_statement_extractor.statements.models import FundingStatement


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
        "--enable-pattern-rescue",
        "--enable-post-filter",
    ])
    assert args.threshold == 15.0
    assert args.top_k == 10
    assert args.normalize is True
    assert args.enable_pattern_rescue is True
    assert args.enable_post_filter is True
