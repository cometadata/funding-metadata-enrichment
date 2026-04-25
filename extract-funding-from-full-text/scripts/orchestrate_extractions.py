"""Local orchestrator for ArXiv funding-statement extraction HF Jobs.

Manages a pool of concurrent A100 jobs over the 17.5k-file
`cometadata/arxiv-latex-extract-full-text/results-2026-04-24/` corpus
with manifest-backed resume, retry, and EMA-based rebalancing.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi


_DONE_RE = re.compile(
    r"\[done file=(?P<file>\S+) rows=(?P<rows>\d+) elapsed_s=(?P<elapsed>[\d.]+)\]"
)


@dataclass
class ManifestRow:
    input_file: str
    size_bytes: int
    est_seconds: float
    status: str  # pending | assigned | done | failed
    attempts: int
    job_id: Optional[str]
    assigned_at: Optional[float]
    completed_at: Optional[float]
    output_path: Optional[str]
    last_error: Optional[str]
    worker_elapsed_s: Optional[float]
    row_count: Optional[int] = None  # filled when worker reports done


Manifest = List[ManifestRow]


def write_manifest(rows: Manifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        tbl = pa.Table.from_pylist([asdict(r) for r in rows])
    else:
        # Build empty table with the right schema so reads round-trip cleanly.
        schema = pa.schema([
            ("input_file", pa.string()),
            ("size_bytes", pa.int64()),
            ("est_seconds", pa.float64()),
            ("status", pa.string()),
            ("attempts", pa.int64()),
            ("job_id", pa.string()),
            ("assigned_at", pa.float64()),
            ("completed_at", pa.float64()),
            ("output_path", pa.string()),
            ("last_error", pa.string()),
            ("worker_elapsed_s", pa.float64()),
            ("row_count", pa.int64()),
        ])
        tbl = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, path)


def read_manifest(path: Path) -> Manifest:
    tbl = pq.read_table(str(path))
    return [ManifestRow(**r) for r in tbl.to_pylist()]


def pick_next_batch(rows, *, target_seconds, max_files):
    pending = [r for r in rows if r.status == "pending"]
    pending.sort(key=lambda r: r.est_seconds, reverse=True)
    batch: list = []
    total = 0.0
    for r in pending:
        if not batch:
            batch.append(r)
            total += r.est_seconds
            continue
        if total + r.est_seconds > target_seconds:
            break
        if len(batch) >= max_files:
            break
        batch.append(r)
        total += r.est_seconds
    return batch


def update_ema(prev, sample, alpha):
    if prev is None:
        return sample
    return (1 - alpha) * prev + alpha * sample


def parse_done_line(line):
    m = _DONE_RE.search(line)
    if not m:
        return None
    return {
        "file": m.group("file"),
        "rows": int(m.group("rows")),
        "elapsed_s": float(m.group("elapsed")),
    }


def read_row_count_local(path):
    """Footer-only row-count read of a local parquet (diagnostic helper)."""
    return pq.ParquetFile(str(path)).metadata.num_rows


def list_input_files_with_sizes(repo_id, subdir):
    """Return [(in_repo_path, size_bytes), ...] for all .parquet files under repo_id/subdir.

    Uses HfApi.list_repo_tree(recursive=True) so we get one indexed listing call
    instead of 17k+ per-file footer reads.
    """
    api = HfApi()
    out = []
    for entry in api.list_repo_tree(
        repo_id, path_in_repo=subdir, repo_type="dataset", recursive=True
    ):
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if path is None or size is None:
            continue
        if not path.endswith(".parquet"):
            continue
        out.append((path, int(size)))
    out.sort(key=lambda t: t[0])
    return out


def seed_or_merge_manifest(existing, listing, *, seconds_per_byte):
    by_file = {r.input_file: r for r in existing}
    for input_file, size_bytes in listing:
        if input_file in by_file:
            continue
        by_file[input_file] = ManifestRow(
            input_file=input_file,
            size_bytes=int(size_bytes),
            est_seconds=float(size_bytes) * seconds_per_byte,
            status="pending",
            attempts=0,
            job_id=None,
            assigned_at=None,
            completed_at=None,
            output_path=None,
            last_error=None,
            worker_elapsed_s=None,
            row_count=None,
        )
    return [by_file[k] for k in sorted(by_file)]


def reconcile_against_outputs(rows, existing_outputs):
    import time as _time
    out = []
    for r in rows:
        candidate = f"predictions/{Path(r.input_file).name}"
        if candidate in existing_outputs and r.status != "done":
            r = ManifestRow(**{
                **asdict(r),
                "status": "done",
                "output_path": candidate,
                "completed_at": _time.time(),
            })
        out.append(r)
    return out


def list_existing_outputs(repo_id):
    """Return set of in-repo paths like {'predictions/x.parquet', ...}."""
    api = HfApi()
    try:
        entries = api.list_repo_tree(
            repo_id, path_in_repo="predictions",
            repo_type="dataset", recursive=True,
        )
    except Exception:
        return set()
    out = set()
    for entry in entries:
        path = getattr(entry, "path", None)
        if path and path.endswith(".parquet"):
            out.add(path)
    return out
