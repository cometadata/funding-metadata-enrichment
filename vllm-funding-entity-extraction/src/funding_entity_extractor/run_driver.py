"""`funding-extract run` driver: glue between parquet I/O and async extraction."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .extract import extract_statements
from .parquet_io import (
    ExtractionWriter,
    already_processed_keys,
    iter_input_rows,
)
from .schema import output_schema_with_extraction

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    input: str
    output: str
    text_field: str
    row_id_fields: list[str]
    vllm_url: str
    served_name: str
    concurrency: int
    write_batch_size: int
    request_timeout: float
    max_retries: int
    temperature: float
    top_p: float                    # not used yet; reserved for parse-failure rescue path
    max_tokens: int
    no_resume: bool
    log_every: int
    max_input_chars: int | None = None


def run_extraction(cfg: RunConfig) -> int:
    """Synchronous entry point — wraps the async loop with asyncio.run."""
    return asyncio.run(_run_async(cfg))


async def _run_async(cfg: RunConfig) -> int:
    input_path = Path(cfg.input)
    output_path = Path(cfg.output)
    if not input_path.exists():
        logger.error("input parquet not found: %s", input_path)
        return 2

    skip_keys: set[tuple] = set()
    if not cfg.no_resume:
        skip_keys = already_processed_keys(output_path, key_fields=cfg.row_id_fields)
        if skip_keys:
            logger.info("resume: %d rows already in output, will skip", len(skip_keys))

    input_schema = pq.ParquetFile(str(input_path)).schema_arrow
    out_schema = output_schema_with_extraction(input_schema)

    rows_done = 0
    rows_seen = 0
    statements_seen = 0
    parse_ok = 0
    t_start = time.perf_counter()

    # Append mode: if output exists and we're resuming, we still need to open
    # a fresh writer at the same path. ParquetWriter does not append in-place,
    # so to preserve resume guarantees we write to <output>.partial-<pid> and
    # rename on success... but for now, simpler:
    #   - if resuming and output exists, write new rows to a sibling file and
    #     concat at end (defer concat to a post-task; YAGNI for v0).
    #   - if not resuming or no skip_keys, write directly to output (overwrite).
    #
    # Simpler v0 invariant: if output exists and no resume, we error and ask
    # the user to delete it. This avoids silent data loss.
    if output_path.exists() and cfg.no_resume:
        logger.error(
            "output exists and --no-resume passed: refusing to overwrite %s",
            output_path,
        )
        return 2

    # If output exists with skip_keys (resume case), write to a sibling and
    # concatenate at the end. Otherwise write straight through.
    use_sidecar = output_path.exists() and bool(skip_keys)
    write_target = (
        output_path.with_suffix(output_path.suffix + ".partial") if use_sidecar else output_path
    )

    pending: list[dict] = []
    flat_statements: list[str] = []
    flat_back_index: list[tuple[int, int]] = []
    for row in iter_input_rows(input_path, text_field=cfg.text_field):
        rows_seen += 1
        key = tuple(row[k] for k in cfg.row_id_fields)
        if key in skip_keys:
            continue
        ri = len(pending)
        pending.append(row)
        for si, stmt in enumerate(row[cfg.text_field]):
            flat_statements.append(stmt)
            flat_back_index.append((ri, si))
    statements_seen = len(flat_statements)

    flat_results = await extract_statements(
        flat_statements,
        vllm_url=cfg.vllm_url,
        served_name=cfg.served_name,
        concurrency=cfg.concurrency,
        max_retries=cfg.max_retries,
        request_timeout=cfg.request_timeout,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_input_chars=cfg.max_input_chars,
    )

    per_row: list[list] = [[None] * len(r[cfg.text_field]) for r in pending]
    for result, (ri, si) in zip(flat_results, flat_back_index):
        per_row[ri][si] = result

    with ExtractionWriter(write_target, schema=out_schema, write_batch_size=cfg.write_batch_size) as writer:
        for row, results in zip(pending, per_row):
            extracted_funders: list[list[dict] | None] = []
            extraction_raw: list[str] = []
            extraction_error: list[str | None] = []
            extraction_latency: list[float] = []
            for r in results:
                if r.error is None and r.funders is not None:
                    parse_ok += 1
                    extracted_funders.append([f.model_dump() for f in r.funders])
                else:
                    extracted_funders.append(None)
                extraction_raw.append(r.raw)
                extraction_error.append(r.error)
                extraction_latency.append(r.latency_ms)

            out_row = dict(row)
            out_row["extracted_funders"] = extracted_funders
            out_row["extraction_raw"] = extraction_raw
            out_row["extraction_error"] = extraction_error
            out_row["extraction_latency_ms"] = extraction_latency
            writer.write(out_row)

            rows_done += 1
            if rows_done % cfg.log_every == 0:
                elapsed = time.perf_counter() - t_start
                rate = rows_done / elapsed if elapsed > 0 else 0
                pr = parse_ok / statements_seen if statements_seen else 0
                logger.info(
                    "[%d done / %d seen] rows/s=%.2f parse_ok=%.1f%% statements=%d",
                    rows_done, rows_seen, rate, pr * 100, statements_seen,
                )

    # If we wrote to a sidecar, concat with the existing output via pyarrow
    if use_sidecar:
        existing = pq.read_table(output_path)
        new_rows = pq.read_table(write_target)
        merged = pa.concat_tables([existing, new_rows])
        pq.write_table(merged, output_path)
        write_target.unlink()
        logger.info("merged %d resumed rows into %s", new_rows.num_rows, output_path)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "done: rows_seen=%d rows_done=%d statements=%d parse_ok=%d elapsed=%.1fs",
        rows_seen, rows_done, statements_seen, parse_ok, elapsed,
    )
    return 0
