"""Build a small in-memory parquet table mirroring the production input schema."""

from __future__ import annotations

import pyarrow as pa


def sample_input_table() -> pa.Table:
    """5 rows total: 3 with statements, 2 empty (should be filtered out)."""
    return pa.table({
        "arxiv_id":            ["paper-A", "paper-B", "paper-C", "paper-D", "paper-E"],
        "shard_id":            ["s1"] * 5,
        "doc_id":              ["paper-A", "paper-B", "paper-C", "paper-D", "paper-E"],
        "input_file":          ["foo.parquet"] * 5,
        "row_idx":             [0, 1, 2, 3, 4],
        "predicted_statements": [
            ["NSF DMS-1 supported this work."],
            [],                                            # empty -> filtered
            ["Funded by NIH R01-AI-1.", "And by DOE."],   # 2 statements
            [],                                            # empty -> filtered
            ["MEXT Japan grant."],
        ],
        "predicted_details":   [[]] * 5,                  # we don't use this column
        "text_length":         [10000, 8000, 12000, 5000, 7000],
        "latency_ms":          [100.0] * 5,
        "error":               [None] * 5,
    })
