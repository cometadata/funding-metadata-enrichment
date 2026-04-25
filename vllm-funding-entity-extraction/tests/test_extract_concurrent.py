import asyncio
import time

import httpx
import pytest
import respx

from funding_entity_extractor.extract import extract_statements


def _ok_payload(content: str = "[]"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


@pytest.mark.asyncio
async def test_extract_statements_preserves_input_order():
    statements = [f"stmt {i}" for i in range(50)]
    contents = [f'[{{"funder_name": "F{i}", "awards": []}}]' for i in range(50)]

    with respx.mock(base_url="http://vllm.test") as mock:
        # Each call gets a different content depending on the request body
        def _handler(request: httpx.Request) -> httpx.Response:
            body = request.read().decode("utf-8")
            # Iterate in reverse so longer numerals match before their prefixes
            # (e.g. "stmt 10" before "stmt 1").
            for i in reversed(range(50)):
                if f"stmt {i}" in body:
                    return httpx.Response(200, json=_ok_payload(contents[i]))
            return httpx.Response(500, text="unmatched")

        mock.post("/v1/chat/completions").mock(side_effect=_handler)

        results = await extract_statements(
            statements,
            vllm_url="http://vllm.test",
            concurrency=8,
            max_retries=1,
            request_timeout=10.0,
        )

    assert len(results) == 50
    for i, r in enumerate(results):
        assert r.error is None
        assert r.funders[0].funder_name == f"F{i}"


@pytest.mark.asyncio
async def test_extract_statements_concurrency_does_not_serialize_requests():
    """With concurrency=10 and a synthetic 50ms-per-call latency, 10 calls
    should finish in roughly 50ms (parallel), not 500ms (serial)."""
    async def _slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=_ok_payload())

    with respx.mock(base_url="http://vllm.test") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=_slow)

        t0 = time.perf_counter()
        results = await extract_statements(
            [f"s{i}" for i in range(10)],
            vllm_url="http://vllm.test",
            concurrency=10,
            max_retries=1,
            request_timeout=10.0,
        )
        elapsed = time.perf_counter() - t0

    assert len(results) == 10
    # Allow generous slack for CI; serial would be ~500ms, parallel ~50-200ms
    assert elapsed < 0.4, f"requests appear to be serialized; elapsed={elapsed:.3f}s"


@pytest.mark.asyncio
async def test_extract_statements_handles_partial_failures():
    """Half the requests succeed, half return 500 — both kinds end up in results."""
    counter = {"i": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        i = counter["i"]
        counter["i"] += 1
        if i % 2 == 0:
            return httpx.Response(200, json=_ok_payload())
        return httpx.Response(500, text="boom")

    with respx.mock(base_url="http://vllm.test") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=_handler)

        results = await extract_statements(
            [f"s{i}" for i in range(6)],
            vllm_url="http://vllm.test",
            concurrency=2,
            max_retries=1,
            request_timeout=10.0,
        )

    n_ok = sum(1 for r in results if r.error is None)
    n_err = sum(1 for r in results if r.error is not None)
    assert n_ok + n_err == 6
    assert n_ok > 0 and n_err > 0
