#!/usr/bin/env python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "funding-entity-extractor[serve] @ git+https://github.com/cometadata/funding-metadata-enrichment.git@funding-entity-extraction#subdirectory=vllm-funding-entity-extraction",
#     "datasets>=3.0.0",
#     "huggingface-hub>=0.25.0",
#     "pyarrow>=15",
#     "httpx>=0.27",
# ]
# ///
"""High-throughput benchmark for the funding-entity-extractor LoRA.

Spawns `funding-extract serve` as a subprocess, polls /health, runs async
extraction over a parquet file from `cometadata/arxiv-funding-statement-extractions`
(filtered to rows where `predicted_statements` is non-empty), then pushes
predictions and metrics to the Hub.

Usage:
    hf jobs uv run --flavor h100x1 \\
      --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \\
      --secrets HF_TOKEN=$(hf auth token) --timeout 1h \\
      benchmark_hf_job.py -- --input-file predictions/arXiv_src_0001_001_006.parquet \\
                             --run-name baseline-v1 \\
                             --push-to-hub <org>/<repo>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("benchmark_hf_job")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark for the funding-entity-extractor LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--dataset", default="cometadata/arxiv-funding-statement-extractions")
    p.add_argument(
        "--input-file",
        required=True,
        help="Parquet file path within the dataset, e.g. predictions/arXiv_src_0001_001_006.parquet",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)

    # Push targets
    p.add_argument("--push-to-hub", default="cometadata/arxiv-funding-entity-extractions",
                   help="Repo to push predictions/metrics. Default: cometadata/arxiv-funding-entity-extractions.")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--private", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-dir", default="./benchmark_results")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--allow-cpu", action="store_true")

    # Client knobs
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)

    # Server knobs (passed to `funding-extract serve`)
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--readiness-timeout-seconds", type=int, default=900)

    # Server passthrough (after `--`)
    p.add_argument("vllm_passthrough", nargs=argparse.REMAINDER)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# CUDA probe (mirror reference)
# ---------------------------------------------------------------------------

def _probe_cuda(allow_cpu: bool) -> int:
    import torch  # imported lazily so --allow-cpu hosts without torch can fail clearly elsewhere

    if allow_cpu:
        logger.info(
            "torch=%s cuda=%s device=%s (allow-cpu=True; skipping CUDA-required probe)",
            torch.__version__,
            torch.cuda.is_available(),
            "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        )
        return 0
    for attempt in range(120):
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                logger.info(
                    "torch=%s cuda=%s device=%s",
                    torch.__version__,
                    torch.cuda.is_available(),
                    torch.cuda.get_device_name(0),
                )
                return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("cuda probe attempt %d failed: %s", attempt, exc)
        time.sleep(1)
    logger.error("CUDA not available after 120s — refusing CPU fallback")
    return 2


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _spawn_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    cmd = [
        "funding-extract", "serve",
        "--port", str(args.vllm_port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
    ]
    passthrough = list(args.vllm_passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if passthrough:
        cmd.append("--")
        cmd.extend(passthrough)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    logger.info("spawning server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


async def _wait_for_ready(url: str, timeout_seconds: int) -> bool:
    import httpx

    deadline = time.monotonic() + timeout_seconds
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{url}/health", timeout=5.0)
                if resp.status_code == 200:
                    logger.info("server ready at %s after %.1fs", url, timeout_seconds - (deadline - time.monotonic()))
                    return True
            except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                pass
            await asyncio.sleep(2.0)
    return False


def _terminate_server(proc: subprocess.Popen, grace_seconds: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    logger.info("terminating server (pid=%d)", proc.pid)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        logger.warning("server did not terminate in %.0fs — killing", grace_seconds)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            proc.kill()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_input_table(args: argparse.Namespace):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    logger.info("downloading %s :: %s", args.dataset, args.input_file)
    path = hf_hub_download(
        repo_id=args.dataset,
        filename=args.input_file,
        repo_type="dataset",
    )
    logger.info("downloaded to %s", path)
    table = pq.read_table(path)
    logger.info("loaded table: rows=%d cols=%s", table.num_rows, table.column_names)
    return table, path


def _filter_and_shard(table, args: argparse.Namespace):
    import pyarrow.compute as pc

    # Filter rows where len(predicted_statements) > 0
    lengths = pc.list_value_length(table["predicted_statements"])
    nonempty = pc.greater(lengths, 0)
    table = table.filter(nonempty)
    logger.info("filtered to non-empty rows: %d remain", table.num_rows)

    if args.max_samples:
        table = table.slice(0, min(args.max_samples, table.num_rows))
        logger.info("capped to --max-samples=%d", table.num_rows)

    if args.num_shards > 1:
        per = (table.num_rows + args.num_shards - 1) // args.num_shards
        start = args.shard_index * per
        end = min(start + per, table.num_rows)
        table = table.slice(start, end - start)
        logger.info("sharded %d/%d -> rows=%d", args.shard_index, args.num_shards, table.num_rows)

    return table


# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------

async def _run_benchmark_extraction(
    table,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Iterate rows, call extract_statements per row, collect per-statement timings."""
    from funding_entity_extractor import extract_statements

    rows = table.to_pylist()
    n_rows = len(rows)
    output_rows: list[dict[str, Any]] = []

    # per-statement counters for metrics
    all_latencies: list[float] = []
    all_completion_tokens: list[int] = []
    parse_ok = 0
    nonempty_funders = 0
    n_funders_total = 0
    n_awards_total = 0

    flat_statements: list[str] = []
    flat_back_index: list[tuple[int, int]] = []
    for ri, row in enumerate(rows):
        for si, stmt in enumerate(row["predicted_statements"]):
            flat_statements.append(stmt)
            flat_back_index.append((ri, si))
    n_statements_total = len(flat_statements)
    logger.info("dispatching %d statements across %d rows", n_statements_total, n_rows)

    t_start = time.perf_counter()
    flat_results = await extract_statements(
        flat_statements,
        vllm_url=f"http://127.0.0.1:{args.vllm_port}",
        concurrency=args.concurrency,
        max_retries=3,
        request_timeout=60.0,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    wall_seconds = time.perf_counter() - t_start

    per_row: list[list] = [[None] * len(r["predicted_statements"]) for r in rows]
    for result, (ri, si) in zip(flat_results, flat_back_index):
        per_row[ri][si] = result

    for i, (row, results) in enumerate(zip(rows, per_row)):
        extracted_funders: list[Any] = []
        extraction_raw: list[str] = []
        extraction_error: list[str | None] = []
        extraction_latency: list[float] = []

        for r in results:
            if r.error is None and r.funders is not None:
                parse_ok += 1
                if r.funders:
                    nonempty_funders += 1
                    n_funders_total += len(r.funders)
                    for f in r.funders:
                        n_awards_total += len(f.awards)
                extracted_funders.append([f.model_dump() for f in r.funders])
            else:
                extracted_funders.append(None)
            extraction_raw.append(r.raw)
            extraction_error.append(r.error)
            extraction_latency.append(r.latency_ms)
            all_latencies.append(r.latency_ms)
            all_completion_tokens.append(r.completion_tokens)

        out = dict(row)
        out["extracted_funders"] = extracted_funders
        out["extraction_raw"] = extraction_raw
        out["extraction_error"] = extraction_error
        out["extraction_latency_ms"] = extraction_latency
        output_rows.append(out)

        if (i + 1) % args.log_every == 0 or i == n_rows - 1:
            pr = parse_ok / n_statements_total if n_statements_total else 0
            logger.info(
                "[regroup %d/%d] parse_ok=%.1f%% stmts=%d",
                i + 1, n_rows, pr * 100, n_statements_total,
            )

    aggregates = {
        "n_rows": n_rows,
        "n_statements": n_statements_total,
        "wall_seconds": wall_seconds,
        "parse_ok": parse_ok,
        "nonempty_funders": nonempty_funders,
        "n_funders_total": n_funders_total,
        "n_awards_total": n_awards_total,
        "all_latencies": all_latencies,
        "all_completion_tokens": all_completion_tokens,
    }
    return output_rows, aggregates


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _build_metrics_row(aggregates: dict[str, Any]) -> dict[str, Any]:
    n_statements = aggregates["n_statements"]
    wall_seconds = aggregates["wall_seconds"]
    # Drop first latency as warmup
    latencies = aggregates["all_latencies"]
    post_warmup = latencies[1:] if len(latencies) > 1 else latencies
    completion_tokens_sum = sum(aggregates["all_completion_tokens"])
    n_funders = aggregates["n_funders_total"]
    parse_ok = aggregates["parse_ok"]
    nonempty = aggregates["nonempty_funders"]

    return {
        "bucket": "overall",
        "n_rows": aggregates["n_rows"],
        "n_statements": n_statements,
        "wall_seconds": wall_seconds,
        "rows_per_sec": (aggregates["n_rows"] / wall_seconds) if wall_seconds > 0 else 0.0,
        "statements_per_sec": (n_statements / wall_seconds) if wall_seconds > 0 else 0.0,
        "gen_tokens_per_sec": (completion_tokens_sum / wall_seconds) if wall_seconds > 0 else 0.0,
        "mean_latency_ms": statistics.fmean(post_warmup) if post_warmup else 0.0,
        "p50_latency_ms": _percentile(post_warmup, 0.5),
        "p95_latency_ms": _percentile(post_warmup, 0.95),
        "p99_latency_ms": _percentile(post_warmup, 0.99),
        "parse_success_rate": (parse_ok / n_statements) if n_statements else 0.0,
        "nonempty_extraction_rate": (nonempty / n_statements) if n_statements else 0.0,
        "mean_funders_per_statement": (n_funders / n_statements) if n_statements else 0.0,
        "mean_awards_per_funder": (aggregates["n_awards_total"] / n_funders) if n_funders else 0.0,
    }


def _build_run_config(args: argparse.Namespace, aggregates: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            max_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            gpu_name = "cpu"
            max_mem_mb = 0.0
    except Exception:
        gpu_name = "unknown"
        max_mem_mb = 0.0

    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception:
        vllm_version = "unknown"

    return {
        "bucket": "__run_config__",
        "run_name": args.run_name,
        "dataset": args.dataset,
        "input_file": args.input_file,
        "n_rows": aggregates["n_rows"],
        "concurrency": args.concurrency,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_name": gpu_name,
        "max_gpu_memory_mb": max_mem_mb,
        "vllm_version": vllm_version,
        "lora_repo": "cometadata/funding-extraction-llama-3.1-8b-instruct-artifact-data-mix-grpo-mixed-reward",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "served_name": "funding-extraction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _write_local_artifacts(
    output_rows: list[dict[str, Any]],
    metrics_row: dict[str, Any],
    run_config: dict[str, Any],
    out_dir: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from funding_entity_extractor.schema import output_schema_with_extraction

    if output_rows:
        # Build the output schema by reading column names from the first row +
        # appending extraction columns. We rebuild from a small probe table.
        sample = pa.Table.from_pylist(output_rows[:1])
        # Drop the extraction columns we just added so we get the input shape
        input_cols = [c for c in sample.column_names if not c.startswith("extract")]
        input_schema = pa.schema([sample.field(c) for c in input_cols])
        out_schema = output_schema_with_extraction(input_schema)
        table = pa.Table.from_pylist(output_rows, schema=out_schema)
        pq.write_table(table, out_dir / "predictions.parquet")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_row, fh, indent=2)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)
    logger.info("local artifacts written to %s", out_dir)


def _push_to_hub(
    output_rows: list[dict[str, Any]],
    metrics_row: dict[str, Any],
    run_config: dict[str, Any],
    repo_id: str,
    run_name: str,
    private: bool,
) -> None:
    from datasets import Dataset

    logger.info("pushing predictions (n=%d) to %s", len(output_rows), repo_id)
    Dataset.from_list(output_rows).push_to_hub(
        repo_id,
        config_name=f"predictions-{run_name}",
        private=private,
    )

    logger.info("pushing metrics row to %s :: metrics-%s", repo_id, run_name)
    Dataset.from_list([metrics_row]).push_to_hub(
        repo_id,
        config_name=f"metrics-{run_name}",
        private=private,
    )

    logger.info("pushing run_config row to %s :: run-config-%s", repo_id, run_name)
    Dataset.from_list([run_config]).push_to_hub(
        repo_id,
        config_name=f"run-config-{run_name}",
        private=private,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.run_name is None:
        args.run_name = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    logger.info("run_name=%s input_file=%s", args.run_name, args.input_file)

    rc = _probe_cuda(args.allow_cpu)
    if rc != 0:
        return rc

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    serve_log = out_dir / "serve.log"
    server = _spawn_server(args, serve_log)

    try:
        ready = asyncio.run(_wait_for_ready(f"http://127.0.0.1:{args.vllm_port}", args.readiness_timeout_seconds))
        if not ready:
            logger.error("server not ready after %ds; see %s", args.readiness_timeout_seconds, serve_log)
            return 3

        table, _local_path = _load_input_table(args)
        table = _filter_and_shard(table, args)
        output_rows, aggregates = asyncio.run(_run_benchmark_extraction(table, args))

        metrics_row = _build_metrics_row(aggregates)
        run_config = _build_run_config(args, aggregates)

        logger.info("=== summary ===")
        logger.info(
            "rows=%d wall=%.1fs rows/s=%.2f stmts/s=%.2f gen_tok/s=%.0f parse_ok=%.1f%%",
            metrics_row["n_rows"], metrics_row["wall_seconds"],
            metrics_row["rows_per_sec"], metrics_row["statements_per_sec"],
            metrics_row["gen_tokens_per_sec"], metrics_row["parse_success_rate"] * 100,
        )

        _write_local_artifacts(output_rows, metrics_row, run_config, out_dir)

        if not args.no_push and args.push_to_hub:
            if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
                logger.warning("no HF_TOKEN in env; push will fail unless cli-cached token is picked up")
            _push_to_hub(
                output_rows, metrics_row, run_config,
                args.push_to_hub, args.run_name, args.private,
            )
        else:
            logger.info("push skipped (--no-push or --push-to-hub unset)")

        return 0

    finally:
        _terminate_server(server)


if __name__ == "__main__":
    sys.exit(main())
