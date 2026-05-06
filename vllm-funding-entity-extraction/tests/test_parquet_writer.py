import pyarrow as pa
import pyarrow.parquet as pq

from funding_entity_extractor.parquet_io import (
    ExtractionWriter,
    already_processed_keys,
)
from funding_entity_extractor.schema import output_schema_with_extraction


def _input_schema() -> pa.Schema:
    return pa.schema([
        pa.field("arxiv_id", pa.string()),
        pa.field("row_idx", pa.int64()),
        pa.field("predicted_statements", pa.list_(pa.string())),
    ])


def _make_extracted_row(arxiv_id: str, row_idx: int, n_statements: int) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "row_idx": row_idx,
        "predicted_statements": [f"stmt {i}" for i in range(n_statements)],
        "extracted_funders":     [[] for _ in range(n_statements)],
        "extraction_raw":        ["[]" for _ in range(n_statements)],
        "extraction_error":      [None for _ in range(n_statements)],
        "extraction_latency_ms": [10.0 for _ in range(n_statements)],
    }


def test_writer_writes_and_closes(tmp_path):
    out = tmp_path / "out.parquet"
    schema = output_schema_with_extraction(_input_schema())
    rows = [_make_extracted_row("A", 0, 1), _make_extracted_row("B", 1, 2)]

    with ExtractionWriter(out, schema=schema, write_batch_size=10) as w:
        for r in rows:
            w.write(r)

    table = pq.read_table(out)
    assert table.num_rows == 2
    assert table.column_names[:3] == ["arxiv_id", "row_idx", "predicted_statements"]
    assert "extracted_funders" in table.column_names


def test_writer_flushes_in_batches(tmp_path):
    out = tmp_path / "out.parquet"
    schema = output_schema_with_extraction(_input_schema())
    with ExtractionWriter(out, schema=schema, write_batch_size=2) as w:
        for i in range(5):
            w.write(_make_extracted_row(f"P{i}", i, 1))

    pf = pq.ParquetFile(out)
    # 5 rows, batch size 2 -> 3 row-groups (2, 2, 1)
    assert pf.num_row_groups == 3
    assert pf.metadata.num_rows == 5


def test_already_processed_keys_returns_empty_for_missing_file(tmp_path):
    keys = already_processed_keys(tmp_path / "nope.parquet", key_fields=["arxiv_id", "row_idx"])
    assert keys == set()


def test_already_processed_keys_round_trip(tmp_path):
    out = tmp_path / "out.parquet"
    schema = output_schema_with_extraction(_input_schema())
    with ExtractionWriter(out, schema=schema, write_batch_size=10) as w:
        w.write(_make_extracted_row("A", 0, 1))
        w.write(_make_extracted_row("B", 7, 2))

    keys = already_processed_keys(out, key_fields=["arxiv_id", "row_idx"])
    assert keys == {("A", 0), ("B", 7)}
