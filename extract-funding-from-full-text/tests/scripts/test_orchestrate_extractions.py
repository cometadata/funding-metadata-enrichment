from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.orchestrate_extractions import (
    Manifest,
    ManifestRow,
    write_manifest,
    read_manifest,
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
