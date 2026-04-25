"""PyArrow types for the four extraction columns appended to each row.

Outer list of `extracted_funders` parallels `predicted_statements` (one entry
per input statement). Inner list is the funders array the model emitted for
that statement (empty list = no funders found; null entry in outer list = parse
failure for that statement; check `extraction_error` at the same index).
"""

from __future__ import annotations

import pyarrow as pa


def award_struct_type() -> pa.DataType:
    return pa.struct([
        pa.field("award_ids", pa.list_(pa.string())),
        pa.field("funding_scheme", pa.list_(pa.string())),
        pa.field("award_title", pa.list_(pa.string())),
    ])


def funder_struct_type() -> pa.DataType:
    return pa.struct([
        pa.field("funder_name", pa.string()),
        pa.field("awards", pa.list_(award_struct_type())),
    ])


def extracted_funders_type() -> pa.DataType:
    """list-of-list-of-funder-struct, parallel-indexed with predicted_statements."""
    return pa.list_(pa.list_(funder_struct_type()))


EXTRACTION_COLUMNS: list[tuple[str, pa.DataType]] = [
    ("extracted_funders", extracted_funders_type()),
    ("extraction_raw", pa.list_(pa.string())),
    ("extraction_error", pa.list_(pa.string())),
    ("extraction_latency_ms", pa.list_(pa.float64())),
]


def output_schema_with_extraction(input_schema: pa.Schema) -> pa.Schema:
    """Append the four extraction columns to an input parquet schema."""
    fields = list(input_schema)
    for name, typ in EXTRACTION_COLUMNS:
        fields.append(pa.field(name, typ))
    return pa.schema(fields)
