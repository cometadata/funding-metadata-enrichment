"""Streaming parquet I/O for the funding-extraction utility.

- iter_input_rows: yield dicts row-by-row, filtering rows whose text_field list is empty.
- ExtractionWriter: append-mode pyarrow writer that flushes a row-group every N rows.
- already_processed_keys: scan an existing output file and return the set of
  composite keys already done (for resume).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


def iter_input_rows(
    path: str | Path,
    *,
    text_field: str = "predicted_statements",
) -> Iterator[dict[str, Any]]:
    """Yield input rows as dicts, filtering rows where `text_field` is empty.

    Reads with `iter_batches` so memory stays bounded regardless of file size.
    """
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches():
        # to_pylist gives us native Python objects (lists, ints, strs, None) per row
        for row in batch.to_pylist():
            stmts = row.get(text_field)
            if not stmts:
                continue
            yield row


class ExtractionWriter(AbstractContextManager):
    """Streaming parquet writer that flushes a row-group every N rows."""

    def __init__(
        self,
        path: str | Path,
        *,
        schema: pa.Schema,
        write_batch_size: int = 256,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._batch_size = write_batch_size
        self._buffer: list[dict] = []
        self._writer: pq.ParquetWriter | None = None

    def __enter__(self) -> "ExtractionWriter":
        self._writer = pq.ParquetWriter(self._path, self._schema)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def write(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer or self._writer is None:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        self._writer.write_table(table)
        self._buffer.clear()


def already_processed_keys(
    path: str | Path,
    *,
    key_fields: list[str],
) -> set[tuple]:
    """Return the set of composite keys already written to `path`.

    Returns empty set if the file doesn't exist (fresh run).
    """
    p = Path(path)
    if not p.exists():
        return set()

    pf = pq.ParquetFile(str(p))
    keys: set[tuple] = set()
    for batch in pf.iter_batches(columns=key_fields):
        cols = [batch.column(name).to_pylist() for name in key_fields]
        for tup in zip(*cols):
            keys.add(tuple(tup))
    return keys
