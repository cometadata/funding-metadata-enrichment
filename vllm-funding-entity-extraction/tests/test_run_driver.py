import httpx
import pyarrow.parquet as pq
import pytest
import respx

from funding_entity_extractor.run_driver import RunConfig, run_extraction
from tests.fixtures.make_sample_parquet import sample_input_table


@pytest.fixture
def input_parquet(tmp_path):
    path = tmp_path / "in.parquet"
    pq.write_table(sample_input_table(), path)
    return path


def _ok_payload(funder_name: str = "NSF"):
    return {
        "choices": [
            {
                "message": {
                    "content": f'[{{"funder_name": "{funder_name}", "awards": []}}]',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_run_extraction_filters_empty_rows_and_writes_extractions(input_parquet, tmp_path):
    out = tmp_path / "out.parquet"
    cfg = RunConfig(
        input=str(input_parquet),
        output=str(out),
        text_field="predicted_statements",
        row_id_fields=["arxiv_id", "row_idx"],
        vllm_url="http://vllm.test",
        served_name="funding-extraction",
        concurrency=4,
        write_batch_size=2,
        request_timeout=10.0,
        max_retries=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        no_resume=False,
        log_every=1,
    )

    with respx.mock(base_url="http://vllm.test") as mock:
        mock.post("/v1/chat/completions").mock(return_value=httpx.Response(200, json=_ok_payload()))
        rc = run_extraction(cfg)

    assert rc == 0
    table = pq.read_table(out)
    # 5 input rows; 2 are empty -> 3 in output
    assert table.num_rows == 3

    arxiv_ids = table.column("arxiv_id").to_pylist()
    assert set(arxiv_ids) == {"paper-A", "paper-C", "paper-E"}

    # paper-C has 2 statements -> all extraction columns are length-2 lists
    paper_c_idx = arxiv_ids.index("paper-C")
    extracted = table.column("extracted_funders").to_pylist()[paper_c_idx]
    raws = table.column("extraction_raw").to_pylist()[paper_c_idx]
    errors = table.column("extraction_error").to_pylist()[paper_c_idx]
    latencies = table.column("extraction_latency_ms").to_pylist()[paper_c_idx]
    prompt_tokens = table.column("extraction_prompt_tokens").to_pylist()[paper_c_idx]
    completion_tokens = table.column("extraction_completion_tokens").to_pylist()[paper_c_idx]
    assert len(extracted) == 2
    assert len(raws) == 2
    assert len(errors) == 2
    assert len(latencies) == 2
    assert len(prompt_tokens) == 2
    assert len(completion_tokens) == 2
    assert errors == [None, None]
    assert all(e[0]["funder_name"] == "NSF" for e in extracted)
    # _ok_payload sets usage.prompt_tokens=10, completion_tokens=5
    assert prompt_tokens == [10, 10]
    assert completion_tokens == [5, 5]


def test_run_extraction_resumes_skipping_already_done_rows(input_parquet, tmp_path):
    out = tmp_path / "out.parquet"
    cfg = RunConfig(
        input=str(input_parquet),
        output=str(out),
        text_field="predicted_statements",
        row_id_fields=["arxiv_id", "row_idx"],
        vllm_url="http://vllm.test",
        served_name="funding-extraction",
        concurrency=4,
        write_batch_size=2,
        request_timeout=10.0,
        max_retries=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        no_resume=False,
        log_every=1,
    )

    # First run: process all 3 rows
    with respx.mock(base_url="http://vllm.test") as mock:
        route = mock.post("/v1/chat/completions").mock(return_value=httpx.Response(200, json=_ok_payload()))
        rc1 = run_extraction(cfg)
        first_call_count = route.call_count
    assert rc1 == 0
    assert first_call_count == 4  # paper-A:1 + paper-C:2 + paper-E:1

    # Second run with same output: should skip everything (resume)
    with respx.mock(base_url="http://vllm.test", assert_all_called=False) as mock:
        route = mock.post("/v1/chat/completions").mock(return_value=httpx.Response(200, json=_ok_payload()))
        rc2 = run_extraction(cfg)
    assert rc2 == 0
    assert route.call_count == 0, "resume should have skipped all rows"
