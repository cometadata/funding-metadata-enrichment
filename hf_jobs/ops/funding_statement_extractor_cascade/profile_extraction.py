#!/usr/bin/env python
"""Local per-phase profiler for the funding-statement cascade.

Runs `extract_funding_batch` on a cached sample from
`cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test`
with monkey-patched instrumentation around every major phase of the
per-doc cascade. Emits a JSON report that attributes per-doc wall-clock
to the three cascade stages:

  stage1_chunk_classifier, stage2_span_head, stage3_cleaner, and the
  "unattributed" remainder.

Usage:

    .venv/bin/python3 ops/funding_statement_extractor_cascade/profile_extraction.py \\
        --device cuda --dtype bf16 --num-docs 50 \\
        --output reports/profile_cuda_bf16.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from _profile_utils import PhaseAccumulator, summarize, wrap_callable  # noqa: E402

logger = logging.getLogger("profile_extraction")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--num-docs", type=int, default=50)
    p.add_argument("--skip-cleanup", action="store_true")
    p.add_argument(
        "--cache-dir",
        default=".cache/profile_docs",
        help="Directory to cache the sample JSONL in.",
    )
    p.add_argument(
        "--dataset",
        default="cometadata/arxiv-pdf-only-works-funding-statement-extraction-train-test",
    )
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--id-column", default="doc_id")
    p.add_argument("--text-column", default="vlm_markdown")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def resolve_device(flag: str) -> str:
    if flag != "auto":
        return flag
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_or_build_sample(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cache_dir = Path(args.cache_dir)
    cache_file = cache_dir / f"{args.split}-{args.num_docs}-seed{args.seed}.jsonl"
    if cache_file.exists():
        logger.info("loading cached sample from %s", cache_file)
        rows = []
        with cache_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        return rows

    logger.info(
        "fetching %d samples (streaming) from %s[%s]",
        args.num_docs,
        args.dataset,
        args.split,
    )
    from datasets import load_dataset

    stream = load_dataset(args.dataset, split=args.split, streaming=True)
    shuffled = stream.shuffle(seed=args.seed, buffer_size=max(1000, args.num_docs * 20))
    rows = []
    for item in shuffled:
        rows.append(dict(item))
        if len(rows) >= args.num_docs:
            break

    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("cached %d rows at %s", len(rows), cache_file)
    return rows


def install_instrumentation(top: PhaseAccumulator) -> List[Any]:
    """Monkey-patch the cascade stage entry points with phase timers.

    Returns a list of (module, attr, original) tuples for restoration.
    """
    undo: List[Any] = []

    from funding_statement_extractor_cascade.cascade import pipeline as pipeline_mod

    for attr, phase in (
        ("_run_stage1", "stage1_chunk_classifier"),
        ("_run_stage2", "stage2_span_head"),
        ("_run_stage3", "stage3_cleaner"),
    ):
        orig = wrap_callable(pipeline_mod, attr, phase, top)
        undo.append((pipeline_mod, attr, orig))

    return undo


def restore(undo: List[Any]) -> None:
    for holder, attr, orig in undo:
        setattr(holder, attr, orig)


def gpu_summary(device: str) -> Dict[str, Any]:
    import torch

    info: Dict[str, Any] = {"device": device, "torch": torch.__version__}
    if device == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        info["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    elif device == "mps" and torch.backends.mps.is_available():
        info["gpu_name"] = "Apple MPS"
        info["alloc_gb"] = getattr(torch.mps, "current_allocated_memory", lambda: 0)() / 1e9
    return info


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = resolve_device(args.device)
    logger.info("resolved device=%s dtype=%s", device, args.dtype)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch

    from funding_statement_extractor_cascade.cascade.batch import extract_funding_batch
    from funding_statement_extractor_cascade.cascade.loader import load_models
    from funding_statement_extractor_cascade.cascade.types import CascadeConfig, DocPayload

    logger.info("torch=%s", torch.__version__)

    rows = load_or_build_sample(args)
    if len(rows) < args.num_docs:
        logger.warning("only %d rows available (requested %d)", len(rows), args.num_docs)

    cfg = CascadeConfig(
        device=device,
        dtype=args.dtype if args.dtype != "auto" else "bf16",
        skip_cleanup=args.skip_cleanup,
    )
    models = load_models(cfg)
    logger.info("models loaded on device=%s", device)

    sync_device = device if device in {"cuda", "mps"} else None
    top = PhaseAccumulator(sync_device=sync_device)
    undo = install_instrumentation(top)

    per_doc_totals_ms: List[float] = []
    per_doc_predictions: List[Dict[str, Any]] = []
    try:
        docs = [
            DocPayload(
                doc_id=str(row.get(args.id_column) or i),
                text=row.get(args.text_column) or "",
            )
            for i, row in enumerate(rows[: args.num_docs])
        ]
        for i, batch in enumerate(extract_funding_batch(docs, models=models, cfg=cfg)):
            result = batch.result
            per_doc_totals_ms.append(result.latency_ms)
            per_doc_predictions.append({
                "doc_idx": i,
                "doc_id": batch.doc_id,
                "funding_statement": result.funding_statement,
                "rough_span": result.rough_span,
                "chunk_prob": result.chunk_prob,
                "no_answer_prob": result.no_answer_prob,
                "error": result.error,
            })
            top.commit_doc()
            warmup = " (warmup)" if i == 0 else ""
            logger.info(
                "[%d/%d] total=%.0fms had_funding=%s%s",
                i + 1,
                len(docs),
                result.latency_ms,
                bool(result.funding_statement),
                warmup,
            )
    finally:
        restore(undo)

    top_summary = summarize(top.per_doc_records(), skip_warmup=True)
    measured_total = top_summary["mean_total_ms"]
    real_total_ms = (
        sum(per_doc_totals_ms[1:]) / max(1, len(per_doc_totals_ms[1:]))
        if len(per_doc_totals_ms) > 1
        else (per_doc_totals_ms[0] if per_doc_totals_ms else 0.0)
    )
    unattributed_ms = real_total_ms - measured_total
    unattributed_pct = (
        (unattributed_ms / real_total_ms * 100.0) if real_total_ms > 0 else 0.0
    )

    report = {
        "run_config": vars(args),
        "resolved": {
            "device": device,
            **gpu_summary(device),
        },
        "wall_clock_per_doc_ms": {
            "mean_excl_warmup": real_total_ms,
            "warmup_ms": per_doc_totals_ms[0] if per_doc_totals_ms else None,
            "num_docs": len(per_doc_totals_ms),
        },
        "top_phases": top_summary,
        "stage_call_counts": top.call_counts(),
        "unattributed": {
            "ms": unattributed_ms,
            "pct_of_wallclock": unattributed_pct,
        },
        "per_doc_predictions": per_doc_predictions,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        logger.info("wrote %s", out)

    print_report(report)
    return 0


def print_report(report: Dict[str, Any]) -> None:
    print()
    print(f"device           : {report['resolved']['device']}")
    print(f"torch            : {report['resolved'].get('torch')}")
    gpu = report["resolved"].get("gpu_name", "cpu")
    print(f"gpu              : {gpu}")
    print(f"mean per doc     : {report['wall_clock_per_doc_ms']['mean_excl_warmup']:.1f} ms")
    warmup = report["wall_clock_per_doc_ms"]["warmup_ms"]
    print(f"warmup (doc 0)   : {warmup:.1f} ms" if warmup is not None else "warmup (doc 0)   : n/a")
    print(f"unattributed     : {report['unattributed']['ms']:.1f} ms"
          f" ({report['unattributed']['pct_of_wallclock']:.1f}%)")
    print()
    print("STAGE PHASES (per doc, excl warmup)")
    print(f"  {'phase':32s} {'mean_ms':>10s} {'pct':>7s}")
    for name, vals in sorted(
        report["top_phases"].get("phases", {}).items(),
        key=lambda kv: -kv[1]["mean_ms"],
    ):
        print(
            f"  {name:32s} {vals['mean_ms']:10.2f} {vals['pct_of_wallclock']:6.1f}%"
        )
    print()
    print(f"stage call count (per doc): {report['stage_call_counts']}")


if __name__ == "__main__":
    raise SystemExit(main())
