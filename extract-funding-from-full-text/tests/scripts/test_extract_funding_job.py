from types import SimpleNamespace

from scripts.extract_funding_job import make_output_row


def _fake_statement(statement, score, query, paragraph_idx):
    return SimpleNamespace(statement=statement, score=score,
                           query=query, paragraph_idx=paragraph_idx)


def test_make_output_row_with_predictions():
    result = SimpleNamespace(
        doc_id="2401.00001",
        statements=[
            _fake_statement("Funded by NSF grant 12345.", 42.0, "who funded this work", 7),
        ],
        error=None,
        metadata={
            "arxiv_id": "2401.00001",
            "input_file": "results-2026-04-24/shard-00000.parquet",
            "row_idx": 3,
            "text_length": 8421,
        },
        enqueue_ts=100.0,
        yield_ts=100.5,
    )
    row = make_output_row(result)
    assert row["arxiv_id"] == "2401.00001"
    assert row["doc_id"] == "2401.00001"
    assert row["input_file"] == "results-2026-04-24/shard-00000.parquet"
    assert row["row_idx"] == 3
    assert row["predicted_statements"] == ["Funded by NSF grant 12345."]
    assert row["predicted_details"][0]["score"] == 42.0
    assert row["predicted_details"][0]["paragraph_idx"] == 7
    assert row["text_length"] == 8421
    assert row["latency_ms"] == 500.0
    assert row["error"] is None


def test_make_output_row_zero_predictions_kept():
    result = SimpleNamespace(
        doc_id="2401.00002", statements=[], error=None,
        metadata={"arxiv_id": "2401.00002", "input_file": "x.parquet",
                  "row_idx": 0, "text_length": 100},
        enqueue_ts=0.0, yield_ts=0.1,
    )
    row = make_output_row(result)
    assert row["predicted_statements"] == []
    assert row["predicted_details"] == []
    assert row["error"] is None
