"""Async extraction client for the funding-extraction LoRA over vLLM's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .models import StatementExtraction, parse_funders_json
from .prompt import build_messages

logger = logging.getLogger(__name__)


_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUSES
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError))


async def _post_chat_completions(
    client: httpx.AsyncClient,
    *,
    served_name: str,
    statement: str,
    temperature: float,
    max_tokens: float,
    request_timeout: float,
) -> httpx.Response:
    body = {
        "model": served_name,
        "messages": build_messages(statement),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = await client.post(
        "/v1/chat/completions",
        json=body,
        timeout=request_timeout,
    )
    resp.raise_for_status()
    return resp


async def extract_one(
    client: httpx.AsyncClient,
    *,
    statement: str,
    served_name: str = "funding-extraction",
    temperature: float = 0.0,
    max_tokens: int = 512,
    request_timeout: float = 60.0,
    max_retries: int = 3,
) -> StatementExtraction:
    """Send one chat-completion for one statement and return a StatementExtraction.

    Retries transient HTTP errors with exponential backoff + jitter. After
    `max_retries` exhausted, records the HTTP error in StatementExtraction.error
    and returns (rather than raising), so the caller can persist a row and move on.
    """
    t0 = time.perf_counter()
    raw = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                resp = await _post_chat_completions(
                    client,
                    served_name=served_name,
                    statement=statement,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                )
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.HTTPError, RetryError) as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return StatementExtraction(
            funders=None,
            raw=raw,
            error=f"HTTPError: {type(exc).__name__}: {exc}",
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    funders, parse_error = parse_funders_json(raw)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return StatementExtraction(
        funders=funders,
        raw=raw,
        error=parse_error,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def extract_statements(
    statements: list[str],
    *,
    vllm_url: str,
    served_name: str = "funding-extraction",
    concurrency: int = 256,
    max_retries: int = 3,
    request_timeout: float = 60.0,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> list[StatementExtraction]:
    """Run `extract_one` over `statements` with bounded concurrency.

    Returns a list of StatementExtraction in the same order as `statements`.
    Errors are recorded inside the StatementExtraction (not raised) so the
    caller can persist partial results.
    """
    if not statements:
        return []

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    sem = asyncio.Semaphore(concurrency)
    results: list[StatementExtraction | None] = [None] * len(statements)

    async with httpx.AsyncClient(base_url=vllm_url, limits=limits) as client:
        async def _worker(idx: int, statement: str) -> None:
            async with sem:
                results[idx] = await extract_one(
                    client,
                    statement=statement,
                    served_name=served_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                )

        await asyncio.gather(*(_worker(i, s) for i, s in enumerate(statements)))

    # results positions are filled by every worker; cast away the Optional
    return [r for r in results]  # type: ignore[return-value]
