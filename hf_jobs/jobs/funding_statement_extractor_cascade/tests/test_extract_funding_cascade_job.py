import pyarrow.parquet as pq

from extract_funding_cascade_job import (
    make_output_row,
    parse_args,
    stage_parquet_locally,
)


def test_parse_args_minimum_defaults():
    args = parse_args([
        "--input-repo", "cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test",
        "--input-files", "test-00000-of-00001.parquet",
        "--output-repo", "user/cascade-bench-out",
        "--job-tag", "cascade-bench-001",
    ])
    assert args.text_columns == ["vlm_markdown"]
    assert args.id_column == "doc_id"
    assert args.dtype == "bf16"
    assert args.skip_cleanup is False
    assert args.commit_batch_size == 25
    assert args.skip_preflight is False


def test_parse_args_input_files_strips_whitespace():
    args = parse_args([
        "--input-repo", "x", "--input-files", " a.parquet , b.parquet ",
        "--output-repo", "y", "--job-tag", "z",
    ])
    assert args.input_files == ["a.parquet", "b.parquet"]


def test_parse_args_text_column_accepts_comma_list():
    args = parse_args([
        "--input-repo", "x", "--input-files", "a.parquet",
        "--output-repo", "y", "--job-tag", "z",
        "--text-column", " vlm_markdown , pymupdf_text ,pymupdf_markdown ",
    ])
    assert args.text_columns == ["vlm_markdown", "pymupdf_text", "pymupdf_markdown"]


def test_parse_args_skip_cleanup_flag():
    args = parse_args([
        "--input-repo", "x", "--input-files", "a.parquet",
        "--output-repo", "y", "--job-tag", "z", "--skip-cleanup",
    ])
    assert args.skip_cleanup is True


def test_make_output_row_full():
    from types import SimpleNamespace
    cascade_result = SimpleNamespace(
        funding_statement="This work was funded by NSF.",
        rough_span="funded by NSF",
        chunk_prob=0.93, no_answer_prob=0.04, latency_ms=421.5,
        error=None,
    )
    batch = SimpleNamespace(
        doc_id="2401.00001", result=cascade_result,
        enqueue_ts=100.0, yield_ts=100.42,
        metadata={"arxiv_id": "2401.00001",
                  "input_file": "test-00000-of-00001.parquet",
                  "row_idx": 12, "shard_id": "shard-0", "text_length": 8421},
    )
    row = make_output_row(batch, skip_cleanup=False)
    assert row["arxiv_id"] == "2401.00001"
    assert row["funding_statement"] == "This work was funded by NSF."
    assert row["rough_span"] == "funded by NSF"
    assert row["chunk_prob"] == 0.93
    assert row["no_answer_prob"] == 0.04
    assert row["text_length"] == 8421
    assert row["latency_ms"] == 421.5
    assert row["skip_cleanup"] is False
    assert row["error"] is None


def test_make_output_row_empty_when_no_funding():
    from types import SimpleNamespace
    batch = SimpleNamespace(
        doc_id="2401.00002",
        result=SimpleNamespace(funding_statement="", rough_span="",
                               chunk_prob=None, no_answer_prob=None,
                               latency_ms=10.0, error=None),
        enqueue_ts=0.0, yield_ts=0.01,
        metadata={"arxiv_id": "2401.00002", "input_file": "x.parquet",
                  "row_idx": 0, "text_length": 100},
    )
    row = make_output_row(batch, skip_cleanup=False)
    assert row["funding_statement"] == ""
    assert row["chunk_prob"] is None
    assert row["no_answer_prob"] is None
    assert row["error"] is None


def test_stage_parquet_locally_empty_uses_explicit_schema(tmp_path):
    local = stage_parquet_locally(
        [], path_in_repo="predictions/empty.parquet", staging_dir=tmp_path,
    )
    assert local.exists()
    tbl = pq.read_table(local)
    assert tbl.num_rows == 0
    expected = {"doc_id", "arxiv_id", "shard_id", "input_file", "text_source",
                "row_idx", "funding_statement", "rough_span", "chunk_prob",
                "no_answer_prob", "text_length", "latency_ms", "skip_cleanup",
                "error"}
    assert set(tbl.column_names) == expected


def test_stage_parquet_locally_writes_rows(tmp_path):
    rows = [{
        "doc_id": "x", "arxiv_id": "x", "shard_id": "s",
        "input_file": "f.parquet", "text_source": "vlm_markdown", "row_idx": 0,
        "funding_statement": "Funded.", "rough_span": "Funded",
        "chunk_prob": 0.9, "no_answer_prob": 0.1,
        "text_length": 100, "latency_ms": 1.0, "skip_cleanup": False, "error": None,
    }]
    local = stage_parquet_locally(
        rows, path_in_repo="predictions/f.parquet", staging_dir=tmp_path,
    )
    tbl = pq.read_table(local)
    assert tbl.num_rows == 1
    assert tbl.column("funding_statement").to_pylist() == ["Funded."]


def test_commit_staged_batch_single_commit(tmp_path, monkeypatch):
    import extract_funding_cascade_job as mod

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
    import extract_funding_cascade_job as mod

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
    import extract_funding_cascade_job as mod

    class FakeApi:
        def create_commit(self, **kwargs):
            raise AssertionError("should not be called")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    mod.commit_staged_batch([], repo_id="org/repo", batch_index=99)


def test_batched_commit_groups_files(tmp_path, monkeypatch, capsys):
    """Stage 3 files with batch_size=2 → 2 commits (sizes 2 and 1)."""
    import extract_funding_cascade_job as mod

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


def test_parse_retry_hint_seconds_variants():
    from extract_funding_cascade_job import _parse_retry_hint_seconds
    assert _parse_retry_hint_seconds("retry this action in about 1 hour.") == 3600
    assert _parse_retry_hint_seconds("retry in 30 minutes") == 1800
    assert _parse_retry_hint_seconds("please retry in 45 seconds") == 45
    assert _parse_retry_hint_seconds("no hint here") is None
    assert _parse_retry_hint_seconds("") is None


def test_preflight_commit_check_succeeds_first_try(tmp_path, monkeypatch):
    import extract_funding_cascade_job as mod
    calls = []

    class FakeApi:
        def upload_file(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    mod.preflight_commit_check("org/repo", job_tag="t1", staging_dir=tmp_path)
    assert len(calls) == 1
    assert calls[0]["path_in_repo"] == "worker_status/t1.json"
    assert calls[0]["repo_id"] == "org/repo"


def test_preflight_commit_check_retries_429_and_succeeds(tmp_path, monkeypatch):
    import extract_funding_cascade_job as mod
    sleeps = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

    attempts = {"n": 0}

    class FakeApi:
        def upload_file(self, **kw):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("HTTP 429: retry in 2 minutes")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    mod.preflight_commit_check("org/repo", job_tag="t2", max_wait_s=10.0,
                               staging_dir=tmp_path)
    assert attempts["n"] == 3
    # First two 429s should each have slept; the hint says 120s but max_wait_s caps to 10
    assert sleeps == [10.0, 10.0]


def test_preflight_commit_check_raises_after_exhaustion(tmp_path, monkeypatch):
    import extract_funding_cascade_job as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    class FakeApi:
        def upload_file(self, **kw):
            raise RuntimeError("HTTP 429 too many")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    import pytest
    with pytest.raises(RuntimeError, match="preflight commit still 429"):
        mod.preflight_commit_check("org/repo", job_tag="t3", staging_dir=tmp_path)


def test_preflight_commit_check_reraises_non_429(tmp_path, monkeypatch):
    import extract_funding_cascade_job as mod

    class FakeApi:
        def upload_file(self, **kw):
            raise RuntimeError("HTTP 401 unauthorized")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    import pytest
    with pytest.raises(RuntimeError, match="401"):
        mod.preflight_commit_check("org/repo", job_tag="t4", staging_dir=tmp_path)


def test_main_end_to_end_with_mocks(tmp_path, monkeypatch):
    import extract_funding_cascade_job as mod
    from types import SimpleNamespace

    fake_rows = [
        {"vlm_markdown": "paper A text", "doc_id": "2401.00001",
         "status": "ok", "shard_id": "s0"},
        {"vlm_markdown": "paper B text", "doc_id": "2401.00002",
         "status": "ok", "shard_id": "s0"},
    ]

    def fake_load_dataset(repo, data_files, split, streaming):
        return fake_rows  # list — re-iterable, mirrors a streaming IterableDataset

    def fake_load_models(cfg):
        return SimpleNamespace()

    def fake_extract_batch(documents, *, models, cfg):
        for d in documents:
            result = SimpleNamespace(
                funding_statement=f"funding-{d.doc_id}", rough_span="rs",
                chunk_prob=0.9, no_answer_prob=0.05,
                latency_ms=10.0, error=None,
            )
            yield SimpleNamespace(doc_id=d.doc_id, result=result,
                                  enqueue_ts=0.0, yield_ts=0.01,
                                  metadata=d.metadata)

    monkeypatch.setattr(mod, "_load_dataset", fake_load_dataset, raising=False)
    monkeypatch.setattr(mod, "_load_models", fake_load_models, raising=False)
    monkeypatch.setattr(mod, "_extract_funding_batch", fake_extract_batch, raising=False)

    commits = []
    class FakeApi:
        def upload_file(self, **kw): pass
        def create_commit(self, **kw): commits.append(kw)

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    monkeypatch.setenv("EXTRACT_FUNDING_STAGING", str(tmp_path))

    rc = mod.main([
        "--input-repo", "ds", "--input-files", "test-00000.parquet",
        "--output-repo", "out-repo", "--job-tag", "tag-1",
        "--allow-cpu", "--skip-preflight",
    ])
    assert rc == 0
    # Exactly one final commit
    assert len(commits) == 1
    staged = (tmp_path / "test-00000.parquet")
    assert staged.exists()
    import pyarrow.parquet as pq
    tbl = pq.read_table(staged)
    assert tbl.num_rows == 2
    assert "funding-2401.00001" in tbl.column("funding_statement").to_pylist()
    assert set(tbl.column("text_source").to_pylist()) == {"vlm_markdown"}


def test_main_multi_text_column(tmp_path, monkeypatch):
    """Two text columns → one parquet with 2x rows, tagged by text_source."""
    import extract_funding_cascade_job as mod
    from types import SimpleNamespace

    fake_rows = [
        {"vlm_markdown": "A vlm", "pymupdf_text": "A pdf",
         "doc_id": "2401.00001", "status": "ok", "shard_id": "s0"},
        {"vlm_markdown": "B vlm", "pymupdf_text": "B pdf",
         "doc_id": "2401.00002", "status": "ok", "shard_id": "s0"},
    ]

    def fake_load_dataset(repo, data_files, split, streaming):
        return fake_rows  # re-iterable across both text columns

    def fake_extract_batch(documents, *, models, cfg):
        for d in documents:
            result = SimpleNamespace(
                funding_statement=f"{d.metadata['text_source']}-{d.doc_id}",
                rough_span="rs", chunk_prob=0.9, no_answer_prob=0.05,
                latency_ms=10.0, error=None,
            )
            yield SimpleNamespace(doc_id=d.doc_id, result=result,
                                  enqueue_ts=0.0, yield_ts=0.01,
                                  metadata=d.metadata)

    monkeypatch.setattr(mod, "_load_dataset", fake_load_dataset, raising=False)
    monkeypatch.setattr(mod, "_load_models", lambda cfg: SimpleNamespace(),
                        raising=False)
    monkeypatch.setattr(mod, "_extract_funding_batch", fake_extract_batch,
                        raising=False)

    class FakeApi:
        def upload_file(self, **kw): pass
        def create_commit(self, **kw): pass

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    monkeypatch.setenv("EXTRACT_FUNDING_STAGING", str(tmp_path))

    rc = mod.main([
        "--input-repo", "ds", "--input-files", "test-00000.parquet",
        "--output-repo", "out-repo", "--job-tag", "tag-multi",
        "--text-column", "vlm_markdown,pymupdf_text",
        "--allow-cpu", "--skip-preflight",
    ])
    assert rc == 0
    import pyarrow.parquet as pq
    tbl = pq.read_table(tmp_path / "test-00000.parquet")
    assert tbl.num_rows == 4  # 2 rows x 2 text columns
    assert sorted(tbl.column("text_source").to_pylist()) == [
        "pymupdf_text", "pymupdf_text", "vlm_markdown", "vlm_markdown",
    ]
    funding = tbl.column("funding_statement").to_pylist()
    assert "vlm_markdown-2401.00001" in funding
    assert "pymupdf_text-2401.00002" in funding
