from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.orchestrate_extractions import (
    Manifest,
    ManifestRow,
    parse_done_line,
    pick_next_batch,
    read_manifest,
    read_row_count_local,
    update_ema,
    write_manifest,
)


def _row(input_file, size_bytes=1000, est_seconds=1.0, status="pending",
         attempts=0, job_id=None, assigned_at=None, completed_at=None,
         output_path=None, last_error=None, worker_elapsed_s=None,
         row_count=None):
    return ManifestRow(
        input_file=input_file, size_bytes=size_bytes, est_seconds=est_seconds,
        status=status, attempts=attempts, job_id=job_id,
        assigned_at=assigned_at, completed_at=completed_at,
        output_path=output_path, last_error=last_error,
        worker_elapsed_s=worker_elapsed_s, row_count=row_count,
    )


def test_roundtrip_manifest(tmp_path):
    rows = [
        _row("a.parquet", size_bytes=100_000, est_seconds=4.5),
        _row("b.parquet", size_bytes=200_000, est_seconds=9.0,
             status="done", attempts=1, job_id="job-xyz",
             assigned_at=1000.0, completed_at=1100.0,
             output_path="predictions/b.parquet",
             worker_elapsed_s=99.0, row_count=200),
    ]
    path = tmp_path / "m.parquet"
    write_manifest(rows, path)
    out = read_manifest(path)
    assert len(out) == 2
    assert out[0].input_file == "a.parquet"
    assert out[1].status == "done"
    assert out[1].worker_elapsed_s == 99.0
    assert out[1].size_bytes == 200_000
    assert out[1].row_count == 200


def test_write_manifest_atomic(tmp_path):
    path = tmp_path / "m.parquet"
    write_manifest([], path)
    assert not (tmp_path / "m.parquet.tmp").exists()
    assert path.exists()


def test_pick_batch_packs_until_target():
    rows = [
        _row("a.parquet", est_seconds=50.0),
        _row("b.parquet", est_seconds=100.0),
        _row("c.parquet", est_seconds=150.0),
        _row("d.parquet", est_seconds=999.0, status="done"),
    ]
    batch = pick_next_batch(rows, target_seconds=200.0, max_files=50)
    assert [r.input_file for r in batch] == ["c.parquet"]


def test_pick_batch_handles_huge_single_file():
    rows = [_row("huge.parquet", est_seconds=5000.0)]
    batch = pick_next_batch(rows, target_seconds=1800.0, max_files=50)
    assert [r.input_file for r in batch] == ["huge.parquet"]


def test_pick_batch_skips_non_pending():
    rows = [
        _row("a.parquet", est_seconds=5.0, status="done"),
        _row("b.parquet", est_seconds=5.0, status="assigned"),
        _row("c.parquet", est_seconds=5.0, status="pending"),
    ]
    batch = pick_next_batch(rows, target_seconds=100.0, max_files=50)
    assert [r.input_file for r in batch] == ["c.parquet"]


def test_pick_batch_caps_at_max_files():
    rows = [_row(f"f{i}.parquet", est_seconds=0.001) for i in range(100)]
    batch = pick_next_batch(rows, target_seconds=999999.0, max_files=10)
    assert len(batch) == 10


def test_update_ema():
    assert update_ema(prev=None, sample=0.05, alpha=0.3) == 0.05
    assert abs(update_ema(prev=0.05, sample=0.10, alpha=0.3) - 0.065) < 1e-9


def test_parse_done_line():
    line = "[done file=results-2026-04-24/shard-0.parquet rows=12345 elapsed_s=678.9]"
    parsed = parse_done_line(line)
    assert parsed == {
        "file": "results-2026-04-24/shard-0.parquet",
        "rows": 12345,
        "elapsed_s": 678.9,
    }


def test_parse_done_line_returns_none_on_no_match():
    assert parse_done_line("INFO some other log line") is None


def test_read_row_count_from_footer(tmp_path):
    p = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": list(range(123))}), p)
    assert read_row_count_local(p) == 123


def test_list_input_files_with_sizes(monkeypatch):
    from scripts import orchestrate_extractions as mod

    class FakeFile:
        def __init__(self, path, size):
            self.path = path
            self.size = size

    class FakeApi:
        def list_repo_tree(self, repo_id, path_in_repo, repo_type, recursive):
            assert repo_id == "org/repo"
            assert path_in_repo == "results-2026-04-24"
            assert repo_type == "dataset"
            assert recursive is True
            return iter([
                FakeFile("results-2026-04-24/2024/arXiv_src_2401_001.parquet", 1_000_000),
                FakeFile("results-2026-04-24/2024/arXiv_src_2401_002.parquet", 2_000_000),
                FakeFile("results-2026-04-24/2024/some_other_file.txt", 500),
            ])

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    out = mod.list_input_files_with_sizes("org/repo", "results-2026-04-24")
    assert out == [
        ("results-2026-04-24/2024/arXiv_src_2401_001.parquet", 1_000_000),
        ("results-2026-04-24/2024/arXiv_src_2401_002.parquet", 2_000_000),
    ]


def test_seed_merges_new_files_preserves_existing():
    from scripts.orchestrate_extractions import seed_or_merge_manifest

    existing = [
        _row("a.parquet", size_bytes=100_000, est_seconds=4.5,
             status="done", attempts=1, job_id="j1",
             assigned_at=1.0, completed_at=2.0,
             output_path="predictions/a.parquet", worker_elapsed_s=90.0,
             row_count=100),
    ]
    listing = [("a.parquet", 100_000), ("b.parquet", 200_000)]
    merged = seed_or_merge_manifest(existing, listing, seconds_per_byte=4.5e-5)
    by_file = {r.input_file: r for r in merged}
    assert by_file["a.parquet"].status == "done"
    assert by_file["a.parquet"].attempts == 1
    assert by_file["b.parquet"].status == "pending"
    assert by_file["b.parquet"].est_seconds == 200_000 * 4.5e-5
    assert by_file["b.parquet"].size_bytes == 200_000


def test_reconcile_marks_done_when_output_exists():
    from scripts.orchestrate_extractions import reconcile_against_outputs

    rows = [
        _row("dir/a.parquet", status="assigned", attempts=1, job_id="j1",
             assigned_at=1.0),
        _row("dir/b.parquet", status="pending"),
    ]
    existing_outputs = {"predictions/a.parquet"}
    out = reconcile_against_outputs(rows, existing_outputs)
    by = {r.input_file: r for r in out}
    assert by["dir/a.parquet"].status == "done"
    assert by["dir/a.parquet"].output_path == "predictions/a.parquet"
    assert by["dir/a.parquet"].completed_at is not None
    assert by["dir/b.parquet"].status == "pending"


def test_submit_job_retries_429(monkeypatch):
    from scripts import orchestrate_extractions as mod
    calls = {"n": 0, "sleeps": []}

    class FakeJob:
        id = "job-fake"

    class FakeApi:
        def run_job(self, **kw):
            calls["n"] += 1
            assert kw["flavor"] == "a100-large"
            if calls["n"] < 3:
                raise RuntimeError("HTTP 429 too many requests")
            return FakeJob()

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(mod.time, "sleep", lambda s: calls["sleeps"].append(s))

    job_id = mod.submit_a100_job(
        script_path="scripts/extract_funding_job.py",
        worker_argv=["--input-repo", "x", "--input-files", "f.parquet",
                     "--output-repo", "y", "--job-tag", "z"],
        token="t", timeout="2h",
    )
    assert job_id == "job-fake"
    assert calls["n"] == 3
    assert calls["sleeps"] == [60, 60]


def test_submit_job_gives_up_after_max_retries(monkeypatch):
    from scripts import orchestrate_extractions as mod
    import pytest as _pytest

    class FakeApi:
        def run_job(self, **kw):
            raise RuntimeError("HTTP 429")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    with _pytest.raises(RuntimeError):
        mod.submit_a100_job(
            script_path="scripts/extract_funding_job.py",
            worker_argv=["--input-repo", "x", "--input-files", "f.parquet",
                         "--output-repo", "y", "--job-tag", "z"],
            token="t", timeout="2h", max_retries=3,
        )


def test_submit_job_retries_503(monkeypatch):
    from scripts import orchestrate_extractions as mod
    calls = {"n": 0, "sleeps": []}

    class FakeJob:
        id = "job-fake"

    class FakeApi:
        def run_job(self, **kw):
            calls["n"] += 1
            assert kw["flavor"] == "a100-large"
            if calls["n"] < 3:
                raise RuntimeError("HTTP 503 service unavailable")
            return FakeJob()

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    monkeypatch.setattr(mod.time, "sleep", lambda s: calls["sleeps"].append(s))

    job_id = mod.submit_a100_job(
        script_path="scripts/extract_funding_job.py",
        worker_argv=["--input-repo", "x", "--input-files", "f.parquet",
                     "--output-repo", "y", "--job-tag", "z"],
        token="t", timeout="2h",
    )
    assert job_id == "job-fake"
    assert calls["n"] == 3
    assert calls["sleeps"] == [60, 60]


def test_poll_job_state_bounds_log_lines(monkeypatch):
    from scripts import orchestrate_extractions as mod

    def gen_lines():
        for i in range(200_000):
            yield SimpleNamespace(data=f"INFO line {i}")

    class FakeApi:
        def inspect_job(self, job_id):
            return SimpleNamespace(
                status=SimpleNamespace(stage="RUNNING"), id=job_id,
            )

        def fetch_job_logs(self, job_id):
            return gen_lines()

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    state = mod.poll_job_state("job-x")
    assert state.stage == "RUNNING"
    # No done events present, but the call should return promptly under cap.
    assert state.done_files == []


def test_poll_job_state_bounds_on_wall_clock(monkeypatch):
    from scripts import orchestrate_extractions as mod

    # Simulate monotonic jumping past the 15s cap on the second tick.
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])

    def fake_monotonic():
        try:
            return next(ticks)
        except StopIteration:
            return 100.0

    monkeypatch.setattr(mod.time, "monotonic", fake_monotonic)

    def gen_lines():
        for i in range(10):
            yield SimpleNamespace(data=f"INFO line {i}")

    class FakeApi:
        def inspect_job(self, job_id):
            return SimpleNamespace(
                status=SimpleNamespace(stage="RUNNING"), id=job_id,
            )

        def fetch_job_logs(self, job_id):
            return gen_lines()

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    state = mod.poll_job_state("job-x")
    assert state.stage == "RUNNING"


def test_dry_run_exercises_state_machine(tmp_path, monkeypatch):
    from scripts import orchestrate_extractions as mod

    # Avoid real HF calls for the listing / outputs side.
    listing = [("dir/a.parquet", 100_000), ("dir/b.parquet", 200_000)]
    monkeypatch.setattr(mod, "list_input_files_with_sizes",
                        lambda repo, sub: listing)
    monkeypatch.setattr(mod, "list_existing_outputs", lambda repo: set())

    manifest_path = tmp_path / "m.parquet"
    rc = mod.main([
        "--manifest", str(manifest_path),
        "--max-in-flight", "2",
        "--max-files-per-batch", "5",
        "--target-seconds", "999999",
        "--seed-seconds-per-byte", "1e-6",
        "--dry-run",
    ])
    assert rc == 0
    rows = mod.read_manifest(manifest_path)
    by = {r.input_file: r for r in rows}
    for f, sz in listing:
        r = by[f]
        assert r.status == "done"
        assert r.job_id and r.job_id.startswith("dry-")
        # Year-sharded layout: predictions/<parent_dir>/<basename>
        parts = f.split("/")
        assert r.output_path == f"predictions/{parts[-2]}/{parts[-1]}"
        # synthetic worker_elapsed_s = size_bytes * seed_seconds_per_byte
        assert r.worker_elapsed_s == sz * 1e-6
        assert r.row_count == max(1, sz // 4096)
        assert r.completed_at is not None


def test_write_run_summary_renders_local_json(tmp_path, monkeypatch):
    import json

    from scripts import orchestrate_extractions as mod

    manifest = [
        _row("dir/a.parquet", size_bytes=100_000, status="done",
             attempts=1, job_id="j1", worker_elapsed_s=120.0,
             output_path="predictions/a.parquet", row_count=500),
        _row("dir/b.parquet", size_bytes=200_000, status="done",
             attempts=1, job_id="j2", worker_elapsed_s=240.0,
             output_path="predictions/b.parquet", row_count=900),
        _row("dir/c.parquet", size_bytes=50_000, status="failed",
             attempts=2, job_id="j3", last_error="job j3 stage=ERROR"),
        _row("dir/d.parquet", size_bytes=75_000, status="pending"),
    ]

    upload_calls = []

    class FakeApi:
        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id,
                        repo_type, **kw):
            upload_calls.append({
                "path_or_fileobj": path_or_fileobj,
                "path_in_repo": path_in_repo,
                "repo_id": repo_id,
                "repo_type": repo_type,
            })

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())

    manifest_path = tmp_path / "arxiv-extractions-2026-04-24.parquet"
    mod.write_manifest(manifest, manifest_path)

    mod.write_run_summary(manifest, "org/output-repo", manifest_path)

    summary_path = tmp_path / "arxiv-extractions-2026-04-24-summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["n_files"] == 4
    assert summary["status_counts"] == {"done": 2, "failed": 1, "pending": 1}
    # total_rows = sum(row_count for done files)
    assert summary["total_rows"] == 1400
    # total_worker_seconds = sum(worker_elapsed_s) across all rows
    assert summary["total_worker_seconds"] == 360.0
    assert abs(summary["estimated_cost_usd"] - (360.0 / 3600.0 * 5.0)) < 1e-9
    assert summary["failed_files"] == [
        {"input_file": "dir/c.parquet", "last_error": "job j3 stage=ERROR"}
    ]
    assert "completed_at" in summary

    assert len(upload_calls) == 3
    paths_in_repo = {c["path_in_repo"] for c in upload_calls}
    assert paths_in_repo == {
        "run_metadata/summary.json",
        "run_metadata/manifest-snapshot.parquet",
        "README.md",
    }
    for c in upload_calls:
        assert c["repo_id"] == "org/output-repo"
        assert c["repo_type"] == "dataset"

    readme_path = tmp_path / "README.md"
    assert readme_path.exists()
    readme_text = readme_path.read_text()
    assert "A100-large bf16" in readme_text


def test_poll_job_state_parses_logs_and_status(monkeypatch):
    from scripts import orchestrate_extractions as mod

    class FakeApi:
        def inspect_job(self, job_id):
            return SimpleNamespace(
                status=SimpleNamespace(stage="COMPLETED"), id=job_id,
            )

        def fetch_job_logs(self, job_id):
            yield SimpleNamespace(data="INFO booting...")
            yield SimpleNamespace(data="[done file=a.parquet rows=10 elapsed_s=5.0]")
            yield SimpleNamespace(data="[done file=b.parquet rows=20 elapsed_s=10.0]")

    monkeypatch.setattr(mod, "HfApi", lambda: FakeApi())
    state = mod.poll_job_state("job-x")
    assert state.stage == "COMPLETED"
    assert state.done_files == [
        {"file": "a.parquet", "rows": 10, "elapsed_s": 5.0},
        {"file": "b.parquet", "rows": 20, "elapsed_s": 10.0},
    ]
    assert state.last_log_ts is not None
