import json

import httpx
import pytest
import respx

from funding_entity_extractor.extract import extract_one
from funding_entity_extractor.models import StatementExtraction


@pytest.mark.asyncio
async def test_extract_one_happy_path():
    payload = {
        "id": "x",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '[{"funder_name": "NSF", "awards": [{"award_ids": ["DMS-1"], "funding_scheme": [], "award_title": []}]}]',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
    }

    with respx.mock(base_url="http://vllm.test") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        async with httpx.AsyncClient(base_url="http://vllm.test") as client:
            result = await extract_one(
                client,
                statement="NSF DMS-1 funded this work.",
                served_name="funding-extraction",
                temperature=0.0,
                max_tokens=512,
                request_timeout=60.0,
                max_retries=3,
            )

    assert isinstance(result, StatementExtraction)
    assert result.error is None
    assert result.funders is not None
    assert result.funders[0].funder_name == "NSF"
    assert result.prompt_tokens == 200
    assert result.completion_tokens == 50
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_extract_one_records_parse_error_on_garbage_output():
    payload = {
        "choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    with respx.mock(base_url="http://vllm.test") as mock:
        mock.post("/v1/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        async with httpx.AsyncClient(base_url="http://vllm.test") as client:
            result = await extract_one(
                client,
                statement="...",
                served_name="funding-extraction",
                temperature=0.0,
                max_tokens=512,
                request_timeout=60.0,
                max_retries=3,
            )
    assert result.funders is None
    assert result.error is not None
    assert result.error.startswith("ParseError:")
    assert result.raw == "not json"


@pytest.mark.asyncio
async def test_extract_one_retries_on_503_then_succeeds():
    good_payload = {
        "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    with respx.mock(base_url="http://vllm.test") as mock:
        route = mock.post("/v1/chat/completions")
        route.side_effect = [
            httpx.Response(503, text="overloaded"),
            httpx.Response(503, text="overloaded"),
            httpx.Response(200, json=good_payload),
        ]
        async with httpx.AsyncClient(base_url="http://vllm.test") as client:
            result = await extract_one(
                client,
                statement="...",
                served_name="funding-extraction",
                temperature=0.0,
                max_tokens=512,
                request_timeout=60.0,
                max_retries=3,
            )
    assert result.error is None
    assert result.funders == []
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_extract_one_returns_input_too_long_without_request():
    """When statement exceeds max_input_chars, return InputTooLong error
    without making any HTTP call (saves a wasted server-side 400)."""
    with respx.mock(base_url="http://vllm.test", assert_all_called=False) as mock:
        route = mock.post("/v1/chat/completions")
        async with httpx.AsyncClient(base_url="http://vllm.test") as client:
            result = await extract_one(
                client,
                statement="x" * 100_000,
                served_name="funding-extraction",
                temperature=0.0,
                max_tokens=512,
                request_timeout=10.0,
                max_retries=1,
                max_input_chars=8000,
            )

    assert result.funders is None
    assert result.error is not None
    assert result.error.startswith("InputTooLong:")
    assert "100000" in result.error
    assert "8000" in result.error
    assert result.raw == ""
    assert route.call_count == 0, "should not have made an HTTP request"


@pytest.mark.asyncio
async def test_extract_one_records_http_error_after_max_retries():
    with respx.mock(base_url="http://vllm.test") as mock:
        mock.post("/v1/chat/completions").mock(return_value=httpx.Response(500, text="boom"))
        async with httpx.AsyncClient(base_url="http://vllm.test") as client:
            result = await extract_one(
                client,
                statement="...",
                served_name="funding-extraction",
                temperature=0.0,
                max_tokens=512,
                request_timeout=60.0,
                max_retries=2,
            )
    assert result.funders is None
    assert result.error is not None
    assert result.error.startswith("HTTPError:")
