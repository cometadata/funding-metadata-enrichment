from types import SimpleNamespace

from scripts.extract_funding_job import make_output_row, parse_args


def _fake_statement(statement, score, query, paragraph_idx):
    return SimpleNamespace(statement=statement, score=score,
                           query=query, paragraph_idx=paragraph_idx)


def test_make_output_row_with_predictions():
    result = SimpleNamespace(
        doc_id="2401.00001",
        statements=[
            _fake_statement("Funded by NSF grant 12345.", 42.0, "who funded this work", 7),
        ],
        error=None,
        metadata={
            "arxiv_id": "2401.00001",
            "input_file": "results-2026-04-24/shard-00000.parquet",
            "row_idx": 3,
            "text_length": 8421,
        },
        enqueue_ts=100.0,
        yield_ts=100.5,
    )
    row = make_output_row(result)
    assert row["arxiv_id"] == "2401.00001"
    assert row["doc_id"] == "2401.00001"
    assert row["input_file"] == "results-2026-04-24/shard-00000.parquet"
    assert row["row_idx"] == 3
    assert row["predicted_statements"] == ["Funded by NSF grant 12345."]
    assert row["predicted_details"][0]["score"] == 42.0
    assert row["predicted_details"][0]["paragraph_idx"] == 7
    assert row["text_length"] == 8421
    assert row["latency_ms"] == 500.0
    assert row["error"] is None


def test_make_output_row_zero_predictions_kept():
    result = SimpleNamespace(
        doc_id="2401.00002", statements=[], error=None,
        metadata={"arxiv_id": "2401.00002", "input_file": "x.parquet",
                  "row_idx": 0, "text_length": 100},
        enqueue_ts=0.0, yield_ts=0.1,
    )
    row = make_output_row(result)
    assert row["predicted_statements"] == []
    assert row["predicted_details"] == []
    assert row["error"] is None


def test_parse_args_minimum():
    args = parse_args([
        "--input-repo", "cometadata/arxiv-latex-extract-full-text",
        "--input-files", "a.parquet,b.parquet",
        "--output-repo", "cometadata/arxiv-funding-statement-extractions",
        "--job-tag", "abc123",
    ])
    assert args.input_repo == "cometadata/arxiv-latex-extract-full-text"
    assert args.input_files == ["a.parquet", "b.parquet"]
    assert args.output_repo == "cometadata/arxiv-funding-statement-extractions"
    assert args.job_tag == "abc123"
    assert args.text_column == "text"
    assert args.id_column == "arxiv_id"
    assert args.dtype == "bf16"
    assert args.batch_size == 512
    assert args.colbert_model == "lightonai/GTE-ModernColBERT-v1"


def test_parse_args_input_files_strips_whitespace():
    args = parse_args([
        "--input-repo", "x", "--input-files", " a.parquet , b.parquet ",
        "--output-repo", "y", "--job-tag", "z",
    ])
    assert args.input_files == ["a.parquet", "b.parquet"]


def test_push_parquet_writes_temp_then_uploads(tmp_path, monkeypatch):
    from scripts import extract_funding_job as mod

    captured = {}

    class FakeApi:
        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            import pyarrow.parquet as pq
            tbl = pq.read_table(path_or_fileobj)
            captured["rows"] = tbl.num_rows
            captured["cols"] = tbl.column_names
            captured["path_in_repo"] = path_in_repo
            captured["repo_id"] = repo_id
            captured["repo_type"] = repo_type

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())

    rows = [
        {"arxiv_id": "x", "doc_id": "x", "input_file": "f.parquet", "row_idx": 0,
         "predicted_statements": ["A"], "predicted_details": [
             {"statement": "A", "score": 1.0, "query": "q", "paragraph_idx": 0}
         ],
         "text_length": 10, "latency_ms": 1.0, "error": None},
    ]
    mod.push_parquet_to_hub(rows, repo_id="org/repo", path_in_repo="predictions/f.parquet",
                            staging_dir=tmp_path)
    assert captured["rows"] == 1
    assert captured["repo_id"] == "org/repo"
    assert captured["repo_type"] == "dataset"
    assert captured["path_in_repo"] == "predictions/f.parquet"
    assert "predicted_statements" in captured["cols"]
    assert "predicted_details" in captured["cols"]
