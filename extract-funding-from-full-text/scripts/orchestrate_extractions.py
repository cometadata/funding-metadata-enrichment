"""Local orchestrator for ArXiv funding-statement extraction HF Jobs.

Manages a pool of concurrent A100 jobs over the 17.5k-file
`cometadata/arxiv-latex-extract-full-text/results-2026-04-24/` corpus
with manifest-backed resume, retry, and EMA-based rebalancing.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


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
