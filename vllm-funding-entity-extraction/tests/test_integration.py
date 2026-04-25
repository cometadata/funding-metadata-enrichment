"""Integration test: spins up funding-extract serve and runs a single extraction.

Skipped unless RUN_INTEGRATION=1 in env. Designed for manual / nightly runs;
not for CI.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest


def _free_port() -> int:
    s = socket.socket(); s.bind(("", 0))
    port = s.getsockname()[1]; s.close()
    return port


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run; needs a GPU + vllm",
)
@pytest.mark.asyncio
async def test_serve_then_run_real(tmp_path: Path):
    import httpx

    from funding_entity_extractor import extract_statements

    port = _free_port()
    server = subprocess.Popen(
        ["funding-extract", "serve", "--port", str(port), "--max-model-len", "2048"],
        stdout=(tmp_path / "serve.log").open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        # Poll /health
        deadline = time.monotonic() + 600
        ready = False
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                try:
                    r = await client.get(f"http://127.0.0.1:{port}/health", timeout=5)
                    if r.status_code == 200:
                        ready = True; break
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(2)
        assert ready, "server did not become ready"

        results = await extract_statements(
            ["This work was supported by NSF grant DMS-1613002."],
            vllm_url=f"http://127.0.0.1:{port}",
            concurrency=1,
        )
        assert len(results) == 1
        assert results[0].error is None
        assert results[0].funders is not None
        names = [f.funder_name for f in results[0].funders]
        assert any("NSF" in (n or "") or "National Science" in (n or "") for n in names)
    finally:
        try:
            os.killpg(os.getpgid(server.pid), 15)
        except (ProcessLookupError, OSError):
            server.terminate()
        server.wait(timeout=30)
