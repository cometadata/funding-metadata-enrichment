#!/usr/bin/env python
"""Local per-phase profiler for the statement-only ColBERT extractor.

Runs `SemanticExtractionService.extract_funding_statements` on a cached
50-doc sample from `cometadata/arxiv-funding-statement-extraction` with
monkey-patched instrumentation around every major phase of the per-doc
pipeline. Emits a JSON report that attributes per-doc wall-clock to:

  paragraph_split, model_encode_doc, model_encode_query, rerank_total,
  filter_is_likely, filter_should_extract_full, filter_extract_sentences,
  filter_extract_long, and the "unattributed" remainder.

A second accumulator breaks the rerank work into its internal steps
(convert_to_tensor, pad_sequence, device_move, colbert_scores einsum,
sort, cpu_sync_tolist, result_construct) so the 32x redundant
pad+device-move per doc can be measured directly.

Usage:

    .venv/bin/python3 scripts/profile_extraction.py \\
        --device mps --dtype bf16 --num-docs 50 \\
        --output reports/profile_mps_bf16.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--colbert-model", default="lightonai/GTE-ModernColBERT-v1")
    p.add_argument("--queries-file", default=None)
    p.add_argument(
        "--cache-dir",
        default=".cache/profile_docs",
        help="Directory to cache the sample JSONL in.",
    )
    p.add_argument("--dataset", default="cometadata/arxiv-funding-statement-extraction")
    p.add_argument("--subdir", default="data", help="Which dataset sub-dir to pull from.")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true", help="Apply torch.compile to the model.")
    p.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="torch.compile mode.",
    )
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


def apply_dtype_patch(dtype_str: str, device: str) -> None:
    if dtype_str == "auto" or dtype_str == "fp32":
        return
    import torch
    from pylate import models as pl_models

    target = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
    orig_init = pl_models.ColBERT.__init__

    def patched_init(self, *a: Any, **kw: Any) -> None:
        orig_init(self, *a, **kw)
        self.to(target)
        logger.info("ColBERT weights cast to %s", target)

    pl_models.ColBERT.__init__ = patched_init


def load_or_build_sample(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cache_dir = Path(args.cache_dir)
    cache_file = cache_dir / f"{args.subdir}-{args.split}-{args.num_docs}-seed{args.seed}.jsonl"
    if cache_file.exists():
        logger.info("loading cached sample from %s", cache_file)
        rows = []
        with cache_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        return rows

    logger.info(
        "fetching %d samples (streaming) from %s[%s/%s.jsonl]",
        args.num_docs,
        args.dataset,
        args.subdir,
        args.split,
    )
    from datasets import load_dataset

    stream = load_dataset(
        args.dataset,
        data_files=f"{args.subdir}/{args.split}.jsonl",
        split="train",
        streaming=True,
    )
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


def install_instrumentation(
    top: PhaseAccumulator, internals: PhaseAccumulator
) -> List[Any]:
    """Monkey-patch production entry points and pylate.rank.rerank.

    Returns a list of (module, attr, original) tuples for restoration.
    """
    undo: List[Any] = []

    import pylate
    import pylate.rank
    from funding_statement_extractor.statements import extraction as extraction_mod

    orig_split = extraction_mod._split_into_paragraphs

    def timed_split(text: str) -> List[str]:
        with top.phase("paragraph_split"):
            return orig_split(text)

    extraction_mod._split_into_paragraphs = timed_split
    undo.append((extraction_mod, "_split_into_paragraphs", orig_split))

    cls = extraction_mod.SemanticExtractionService
    for method in (
        "_is_likely_funding_statement",
        "_should_extract_full_paragraph",
        "_extract_funding_sentences",
        "_extract_funding_from_long_paragraph",
    ):
        orig = getattr(cls, method)
        phase_name = f"filter_{method[1:] if method.startswith('_') else method}".replace(
            "__", "_"
        )

        def make_wrapped(inner: Any, phase: str) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                with top.phase(phase):
                    return inner(*args, **kwargs)

            return wrapped

        setattr(cls, method, staticmethod(make_wrapped(orig, phase_name)))
        undo.append((cls, method, staticmethod(orig)))

    orig_rerank = pylate.rank.rerank
    instrumented = make_instrumented_rerank(top, internals)
    pylate.rank.rerank = instrumented
    undo.append((pylate.rank, "rerank", orig_rerank))
    return undo


def restore(undo: List[Any]) -> None:
    for holder, attr, orig in undo:
        setattr(holder, attr, orig)


def make_instrumented_rerank(top: PhaseAccumulator, internals: PhaseAccumulator) -> Any:
    """Reimplement pylate.rank.rerank with per-step timings.

    Mirrors pylate/rank/rank.py @ 1.4.0 exactly; only adds timing.
    """
    import torch

    from pylate.rank.rank import RerankResult, reshape_embeddings
    from pylate.scores import colbert_scores
    from pylate.utils import convert_to_tensor as func_convert_to_tensor

    def instrumented_rerank(
        documents_ids: List[List[Any]],
        queries_embeddings: Any,
        documents_embeddings: Any,
        device: Optional[str] = None,
    ) -> List[List[Any]]:
        with top.phase("rerank_total"):
            results: List[List[RerankResult]] = []
            queries_embeddings = reshape_embeddings(embeddings=queries_embeddings)
            documents_embeddings = reshape_embeddings(embeddings=documents_embeddings)

            for query_embeddings, query_documents_ids, query_documents_embeddings in zip(
                queries_embeddings, documents_ids, documents_embeddings
            ):
                with internals.phase("rerank_convert_tensor"):
                    query_embeddings = func_convert_to_tensor(query_embeddings)
                    query_documents_embeddings = [
                        func_convert_to_tensor(qde)
                        for qde in query_documents_embeddings
                    ]

                with internals.phase("rerank_pad_sequence"):
                    query_documents_embeddings = torch.nn.utils.rnn.pad_sequence(
                        query_documents_embeddings,
                        batch_first=True,
                        padding_value=0,
                    )

                with internals.phase("rerank_device_move"):
                    if device is not None:
                        query_embeddings = query_embeddings.to(device)
                        query_documents_embeddings = query_documents_embeddings.to(device)
                    else:
                        query_documents_embeddings = query_documents_embeddings.to(
                            query_embeddings.device
                        )

                with internals.phase("rerank_colbert_scores"):
                    query_scores = colbert_scores(
                        queries_embeddings=query_embeddings.unsqueeze(0),
                        documents_embeddings=query_documents_embeddings,
                    )[0]

                with internals.phase("rerank_sort"):
                    scores, sorted_indices = torch.sort(
                        input=query_scores, descending=True
                    )

                with internals.phase("rerank_cpu_sync_tolist"):
                    scores_list = scores.cpu().tolist()
                    sorted_idx_list = sorted_indices.tolist()

                with internals.phase("rerank_construct"):
                    query_documents = [
                        query_documents_ids[idx] for idx in sorted_idx_list
                    ]
                    results.append(
                        [
                            RerankResult(id=doc_id, score=score)
                            for doc_id, score in zip(query_documents, scores_list)
                        ]
                    )

            return results

    return instrumented_rerank


def wrap_model_encode(model: Any, top: PhaseAccumulator) -> None:
    orig_encode = model.encode

    def timed_encode(sentences: Any, *args: Any, **kwargs: Any) -> Any:
        is_query = kwargs.get("is_query", False)
        phase = "model_encode_query" if is_query else "model_encode_doc"
        with top.phase(phase):
            return orig_encode(sentences, *args, **kwargs)

    model.encode = timed_encode


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
    apply_dtype_patch(args.dtype, device)

    import torch

    import pylate  # noqa: F401
    from funding_statement_extractor.config.loader import load_queries
    from funding_statement_extractor.statements.extraction import SemanticExtractionService

    logger.info(
        "torch=%s pylate=%s", torch.__version__, __import__("pylate").__version__
    )

    rows = load_or_build_sample(args)
    if len(rows) < args.num_docs:
        logger.warning("only %d rows available (requested %d)", len(rows), args.num_docs)

    queries = load_queries(queries_file=args.queries_file)
    logger.info("loaded %d queries", len(queries))

    service = SemanticExtractionService()
    model = service._get_model(args.colbert_model)
    model.to(device)
    logger.info("model device pinned to %s", device)

    if args.compile:
        logger.info("applying torch.compile(mode=%s)", args.compile_mode)
        try:
            model[0].auto_model = torch.compile(
                model[0].auto_model, mode=args.compile_mode
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.compile failed, continuing without: %s", exc)

    sync_device = device if device in {"cuda", "mps"} else None
    top = PhaseAccumulator(sync_device=sync_device)
    internals = PhaseAccumulator(sync_device=sync_device)
    undo = install_instrumentation(top, internals)
    wrap_model_encode(model, top)

    per_doc_totals_ms: List[float] = []
    try:
        for i, row in enumerate(rows[: args.num_docs]):
            text = row["text"]
            if sync_device:
                if sync_device == "cuda":
                    torch.cuda.synchronize()
                else:
                    torch.mps.synchronize()
            t0 = time.perf_counter()
            statements = service.extract_funding_statements(
                queries=queries,
                content=text,
                model_name=args.colbert_model,
                top_k=args.top_k,
                threshold=args.threshold,
                batch_size=args.batch_size,
            )
            if sync_device:
                if sync_device == "cuda":
                    torch.cuda.synchronize()
                else:
                    torch.mps.synchronize()
            total_ms = (time.perf_counter() - t0) * 1000.0
            per_doc_totals_ms.append(total_ms)
            top.commit_doc()
            internals.commit_doc()
            warmup = " (warmup)" if i == 0 else ""
            logger.info(
                "[%d/%d] total=%.0fms n_pred=%d n_para=%d%s",
                i + 1,
                args.num_docs,
                total_ms,
                len(statements),
                text.count("\n\n") + 1,
                warmup,
            )
    finally:
        restore(undo)

    top_summary = summarize(top.per_doc_records(), skip_warmup=True)
    internal_summary = summarize(internals.per_doc_records(), skip_warmup=True)
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
        "rerank_internals": internal_summary,
        "rerank_call_counts": internals.call_counts(),
        "unattributed": {
            "ms": unattributed_ms,
            "pct_of_wallclock": unattributed_pct,
        },
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
    print(f"warmup (doc 0)   : {report['wall_clock_per_doc_ms']['warmup_ms']:.1f} ms")
    print(f"unattributed     : {report['unattributed']['ms']:.1f} ms"
          f" ({report['unattributed']['pct_of_wallclock']:.1f}%)")
    print()
    print("TOP PHASES (per doc, excl warmup)")
    print(f"  {'phase':32s} {'mean_ms':>10s} {'pct':>7s}")
    for name, vals in sorted(
        report["top_phases"]["phases"].items(),
        key=lambda kv: -kv[1]["mean_ms"],
    ):
        print(
            f"  {name:32s} {vals['mean_ms']:10.2f} {vals['pct_of_wallclock']:6.1f}%"
        )
    print()
    print("RERANK INTERNALS (summed across all 32 calls per doc)")
    print(f"  {'phase':32s} {'mean_ms':>10s} {'pct_of_rerank':>14s}")
    internal_mean_total = report["rerank_internals"]["mean_total_ms"]
    for name, vals in sorted(
        report["rerank_internals"]["phases"].items(),
        key=lambda kv: -kv[1]["mean_ms"],
    ):
        pct = (vals["mean_ms"] / internal_mean_total * 100.0) if internal_mean_total else 0.0
        print(f"  {name:32s} {vals['mean_ms']:10.2f} {pct:13.1f}%")
    print()
    print(f"rerank call count (per doc): {report['rerank_call_counts']}")


if __name__ == "__main__":
    raise SystemExit(main())
