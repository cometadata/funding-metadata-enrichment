import argparse
from types import SimpleNamespace
from unittest.mock import patch

from funding_statement_extractor_cascade.cascade.cli import (
    add_arguments,
    build_config,
    run,
)
from funding_statement_extractor_cascade.cascade.types import (
    BatchResult,
    CascadeResult,
)


def _args(input_path):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(["-i", str(input_path)])


def test_required_input(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("hi")
    args = _args(md)
    assert args.input == str(md)
    assert args.output == "funding_results.json"
    assert args.skip_cleanup is False
    assert args.chunk_keep_threshold == 0.4
    assert args.dtype == "bf16"


def test_build_config_propagates_cascade_overrides(tmp_path):
    md = tmp_path / "x.md"; md.write_text("hi")
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", str(md), "--skip-cleanup",
                              "--chunk-keep-threshold", "0.7",
                              "--dtype", "fp16"])
    cfg = build_config(args)
    assert cfg.cascade.skip_cleanup is True
    assert cfg.cascade.chunk_keep_threshold == 0.7
    assert cfg.cascade.dtype == "fp16"


def test_run_processes_single_markdown_file(tmp_path):
    md = tmp_path / "x.md"; md.write_text("paper text")
    out = tmp_path / "out.json"

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["-i", str(md), "-o", str(out)])

    fake_result = CascadeResult(funding_statement="Funded by NSF.",
                                 rough_span="Funded by NSF.",
                                 chunk_prob=0.9, no_answer_prob=0.05,
                                 latency_ms=10.0)

    def _fake_batch(documents, *, models, cfg):
        for d in documents:
            yield BatchResult(doc_id=d.doc_id, result=fake_result,
                              enqueue_ts=0.0, yield_ts=0.01, metadata=d.metadata)

    with patch("funding_statement_extractor_cascade.cascade.cli.load_models",
               return_value=SimpleNamespace()), \
         patch("funding_statement_extractor_cascade.cascade.cli.extract_funding_batch",
               side_effect=_fake_batch):
        run(args)

    import json
    data = json.loads(out.read_text())
    assert data["results"]["x.md"]["funding_statement"] == "Funded by NSF."
    assert data["summary"]["total_files"] == 1
    assert data["summary"]["files_with_funding"] == 1


def test_run_resume_preserves_prior_results(tmp_path):
    import json

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    out = tmp_path / "out.json"
    checkpoint = tmp_path / "out.json.checkpoint"

    fake_result = CascadeResult(funding_statement="Funded by NSF.",
                                 rough_span="Funded by NSF.",
                                 chunk_prob=0.9, no_answer_prob=0.05,
                                 latency_ms=10.0)

    def _fake_batch(documents, *, models, cfg):
        for d in documents:
            yield BatchResult(doc_id=d.doc_id, result=fake_result,
                              enqueue_ts=0.0, yield_ts=0.01, metadata=d.metadata)

    # First run: only doc A is present.
    (docs_dir / "a.md").write_text("paper A text")
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    first_args = parser.parse_args([
        "-i", str(docs_dir), "-o", str(out),
        "--checkpoint-file", str(checkpoint),
    ])
    with patch("funding_statement_extractor_cascade.cascade.cli.load_models",
               return_value=SimpleNamespace()), \
         patch("funding_statement_extractor_cascade.cascade.cli.extract_funding_batch",
               side_effect=_fake_batch):
        run(first_args)

    first_data = json.loads(out.read_text())
    assert set(first_data["results"]) == {"a.md"}

    # Second run with --resume: doc B is now also present; only B is new.
    (docs_dir / "b.md").write_text("paper B text")
    second_args = parser.parse_args([
        "-i", str(docs_dir), "-o", str(out),
        "--checkpoint-file", str(checkpoint), "--resume",
    ])
    with patch("funding_statement_extractor_cascade.cascade.cli.load_models",
               return_value=SimpleNamespace()), \
         patch("funding_statement_extractor_cascade.cascade.cli.extract_funding_batch",
               side_effect=_fake_batch):
        run(second_args)

    data = json.loads(out.read_text())
    assert set(data["results"]) == {"a.md", "b.md"}
    assert data["summary"]["total_files"] == 2
