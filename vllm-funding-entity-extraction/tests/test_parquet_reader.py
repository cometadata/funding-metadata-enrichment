import pyarrow as pa
import pyarrow.parquet as pq

from funding_entity_extractor.parquet_io import iter_input_rows
from tests.fixtures.make_sample_parquet import sample_input_table


def test_iter_input_rows_filters_empty_predicted_statements(tmp_path):
    table = sample_input_table()
    path = tmp_path / "in.parquet"
    pq.write_table(table, path)

    rows = list(iter_input_rows(path, text_field="predicted_statements"))
    # 5 input rows, 2 empty -> 3 yielded rows
    assert len(rows) == 3
    assert [r["arxiv_id"] for r in rows] == ["paper-A", "paper-C", "paper-E"]


def test_iter_input_rows_preserves_all_input_columns(tmp_path):
    table = sample_input_table()
    path = tmp_path / "in.parquet"
    pq.write_table(table, path)

    rows = list(iter_input_rows(path, text_field="predicted_statements"))
    sample = rows[0]
    assert set(sample.keys()) >= {
        "arxiv_id", "shard_id", "doc_id", "input_file", "row_idx",
        "predicted_statements", "text_length",
    }
    assert sample["predicted_statements"] == ["NSF DMS-1 supported this work."]
    assert sample["text_length"] == 10000


def test_iter_input_rows_handles_two_statements_per_row(tmp_path):
    table = sample_input_table()
    path = tmp_path / "in.parquet"
    pq.write_table(table, path)

    rows = list(iter_input_rows(path, text_field="predicted_statements"))
    paper_c = next(r for r in rows if r["arxiv_id"] == "paper-C")
    assert paper_c["predicted_statements"] == ["Funded by NIH R01-AI-1.", "And by DOE."]
