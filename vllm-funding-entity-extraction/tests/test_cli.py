import asyncio
from unittest.mock import patch

import pytest

from funding_entity_extractor.cli import build_parser, main


def test_parser_serve_defaults():
    p = build_parser()
    args = p.parse_args(["serve"])
    assert args.command == "serve"
    assert args.base_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert args.lora_repo.startswith("cometadata/funding-extraction-llama-3.1-8b-instruct")
    assert args.served_name == "funding-extraction"
    assert args.port == 8000
    assert args.gpu_memory_utilization == 0.92
    assert args.max_model_len == 8192


def test_parser_serve_passthrough_after_double_dash():
    p = build_parser()
    args = p.parse_args(["serve", "--port", "9001", "--", "--enforce-eager"])
    assert args.port == 9001
    assert args.passthrough == ["--enforce-eager"]


def test_parser_run_required_args():
    p = build_parser()
    args = p.parse_args(["run", "--input", "in.parquet", "--output", "out.parquet"])
    assert args.command == "run"
    assert args.input == "in.parquet"
    assert args.output == "out.parquet"
    assert args.vllm_url == "http://localhost:8000"
    assert args.concurrency == 256
    assert args.max_tokens == 4096
    assert args.max_input_chars == 8000


def test_run_dispatch_calls_run_extraction(monkeypatch, tmp_path):
    """`main` for the run subcommand should dispatch to a run_extraction func."""
    called = {}

    def _fake_run(cfg):
        called["cfg"] = cfg
        return 0

    monkeypatch.setattr("funding_entity_extractor.cli.run_extraction", _fake_run)
    rc = main([
        "run",
        "--input", str(tmp_path / "in.parquet"),
        "--output", str(tmp_path / "out.parquet"),
        "--vllm-url", "http://localhost:9999",
    ])
    assert rc == 0
    assert called["cfg"].input == str(tmp_path / "in.parquet")
    assert called["cfg"].vllm_url == "http://localhost:9999"


def test_serve_dispatch_calls_run_serve(monkeypatch):
    called = {}

    def _fake_serve(cfg):
        called["cfg"] = cfg
        return 0

    monkeypatch.setattr("funding_entity_extractor.cli.run_serve", _fake_serve)
    rc = main(["serve", "--port", "8765"])
    assert rc == 0
    assert called["cfg"].port == 8765
