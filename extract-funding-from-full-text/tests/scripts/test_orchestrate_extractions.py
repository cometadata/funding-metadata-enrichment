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
