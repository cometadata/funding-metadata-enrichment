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
            "shard_id": "shard-00000",
        },
        enqueue_ts=100.0,
        yield_ts=100.5,
    )
    row = make_output_row(result)
    assert row["arxiv_id"] == "2401.00001"
    assert row["doc_id"] == "2401.00001"
    assert row["shard_id"] == "shard-00000"
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


def test_push_parquet_to_hub_empty_rows(tmp_path, monkeypatch):
    from scripts import extract_funding_job as mod

    captured = {}

    class FakeApi:
        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            import pyarrow.parquet as pq
            tbl = pq.read_table(path_or_fileobj)
            captured["rows"] = tbl.num_rows
            captured["cols"] = tbl.column_names

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())

    mod.push_parquet_to_hub([], repo_id="org/repo",
                            path_in_repo="predictions/empty.parquet",
                            staging_dir=tmp_path)
    assert captured["rows"] == 0
    expected = {
        "arxiv_id", "shard_id", "doc_id", "input_file", "row_idx",
        "predicted_statements", "predicted_details", "text_length",
        "latency_ms", "error",
    }
    assert expected.issubset(set(captured["cols"]))


def test_stage_parquet_locally_empty_uses_explicit_schema(tmp_path):
    from scripts import extract_funding_job as mod
    import pyarrow.parquet as pq

    local = mod.stage_parquet_locally(
        [], path_in_repo="predictions/empty.parquet", staging_dir=tmp_path,
    )
    assert local.exists()
    tbl = pq.read_table(local)
    assert tbl.num_rows == 0
    expected = {
        "arxiv_id", "shard_id", "doc_id", "input_file", "row_idx",
        "predicted_statements", "predicted_details", "text_length",
        "latency_ms", "error",
    }
    assert expected.issubset(set(tbl.column_names))


def test_stage_parquet_locally_writes_rows(tmp_path):
    from scripts import extract_funding_job as mod
    import pyarrow.parquet as pq

    rows = [{
        "arxiv_id": "x", "shard_id": "s", "doc_id": "x",
        "input_file": "f.parquet", "row_idx": 0,
        "predicted_statements": ["A"],
        "predicted_details": [
            {"statement": "A", "score": 1.0, "query": "q", "paragraph_idx": 0}
        ],
        "text_length": 10, "latency_ms": 1.0, "error": None,
    }]
    local = mod.stage_parquet_locally(
        rows, path_in_repo="predictions/f.parquet", staging_dir=tmp_path,
    )
    tbl = pq.read_table(local)
    assert tbl.num_rows == 1
    assert tbl.column("predicted_statements").to_pylist() == [["A"]]


def test_commit_staged_batch_single_commit(tmp_path, monkeypatch):
    from scripts import extract_funding_job as mod

    calls = []

    class FakeApi:
        def create_commit(self, *, repo_id, repo_type, operations, commit_message):
            calls.append({
                "repo_id": repo_id,
                "repo_type": repo_type,
                "ops": list(operations),
                "msg": commit_message,
            })

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())

    staged = []
    for name in ("a.parquet", "b.parquet"):
        local = mod.stage_parquet_locally(
            [], path_in_repo=f"predictions/{name}", staging_dir=tmp_path,
        )
        staged.append({
            "input_file": name,
            "path_in_repo": f"predictions/{name}",
            "local_path": local,
            "rows": 0,
            "elapsed_s": 1.0,
        })

    mod.commit_staged_batch(staged, repo_id="org/repo", batch_index=1)
    assert len(calls) == 1
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["repo_id"] == "org/repo"
    assert len(calls[0]["ops"]) == 2
    assert "batch 1" in calls[0]["msg"]
    paths = {op.path_in_repo for op in calls[0]["ops"]}
    assert paths == {"predictions/a.parquet", "predictions/b.parquet"}


def test_commit_staged_batch_retries_on_429(tmp_path, monkeypatch):
    from scripts import extract_funding_job as mod

    attempts = {"n": 0}

    class FakeApi:
        def create_commit(self, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("HTTP 429 too many requests")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    sleeps = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

    local = mod.stage_parquet_locally(
        [], path_in_repo="predictions/x.parquet", staging_dir=tmp_path,
    )
    staged = [{
        "input_file": "x.parquet",
        "path_in_repo": "predictions/x.parquet",
        "local_path": local,
        "rows": 0, "elapsed_s": 1.0,
    }]
    mod.commit_staged_batch(staged, repo_id="org/repo", batch_index=1,
                            retry_sleep_s=0.0)
    assert attempts["n"] == 3
    assert len(sleeps) == 2


def test_commit_staged_batch_empty_is_noop(monkeypatch):
    from scripts import extract_funding_job as mod

    class FakeApi:
        def create_commit(self, **kwargs):
            raise AssertionError("should not be called")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    mod.commit_staged_batch([], repo_id="org/repo", batch_index=99)


def test_batched_commit_groups_files(tmp_path, monkeypatch, capsys):
    """Stage 3 files with batch_size=2 → 2 commits (sizes 2 and 1)."""
    from scripts import extract_funding_job as mod

    commits = []

    class FakeApi:
        def create_commit(self, *, repo_id, repo_type, operations, commit_message):
            commits.append({
                "n_ops": len(list(operations)),
                "msg": commit_message,
            })

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())

    commit_batch_size = 2
    staged_buffer = []
    batch_index = 0

    files = ["f1.parquet", "f2.parquet", "f3.parquet"]
    for input_file in files:
        out_path = f"predictions/{input_file}"
        local = mod.stage_parquet_locally(
            [], path_in_repo=out_path, staging_dir=tmp_path,
        )
        staged_buffer.append({
            "input_file": input_file,
            "path_in_repo": out_path,
            "local_path": local,
            "rows": 0,
            "elapsed_s": 1.0,
        })
        if len(staged_buffer) >= commit_batch_size:
            batch_index += 1
            mod.commit_staged_batch(staged_buffer, repo_id="org/repo",
                                    batch_index=batch_index)
            mod.emit_done_lines(staged_buffer)
            staged_buffer = []

    if staged_buffer:
        batch_index += 1
        mod.commit_staged_batch(staged_buffer, repo_id="org/repo",
                                batch_index=batch_index)
        mod.emit_done_lines(staged_buffer)
        staged_buffer = []

    assert len(commits) == 2
    assert commits[0]["n_ops"] == 2
    assert commits[1]["n_ops"] == 1
    assert "batch 1" in commits[0]["msg"]
    assert "batch 2" in commits[1]["msg"]

    out = capsys.readouterr().out
    # All three files should have produced a [done ...] line
    for f in files:
        assert f"[done file={f}" in out


def test_parse_args_commit_batch_size_default():
    from scripts.extract_funding_job import parse_args
    args = parse_args([
        "--input-repo", "x", "--input-files", "a.parquet",
        "--output-repo", "y", "--job-tag", "z",
    ])
    assert args.commit_batch_size == 25


def test_parse_args_commit_batch_size_override():
    from scripts.extract_funding_job import parse_args
    args = parse_args([
        "--input-repo", "x", "--input-files", "a.parquet",
        "--output-repo", "y", "--job-tag", "z",
        "--commit-batch-size", "10",
    ])
    assert args.commit_batch_size == 10
