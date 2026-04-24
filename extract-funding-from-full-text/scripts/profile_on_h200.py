#!/usr/bin/env python
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "extract-funding-statements-from-full-text @ git+https://github.com/cometadata/funding-metadata-enrichment.git@statement-only-extraction#subdirectory=extract-funding-from-full-text",
#     "datasets>=3.0.0",
#     "huggingface-hub>=0.25.0",
#     "sentence-transformers==5.1.1",
#     "pylate==1.4.0",
#     "torch>=2.5.0,<2.7.0",
# ]
# ///
"""H200 per-phase profiler — self-contained `hf jobs uv run` script.

Mirrors `scripts/profile_extraction.py` but bundles all profiling utilities
inline so the script can be fetched and executed by HF Jobs without the
rest of the repo. Emits a JSON phase breakdown to stdout tagged with
`===PROFILE_JSON===` markers so the log can be grep'd for the report.

Usage:

    hf jobs uv run --flavor h200x1 --secrets HF_TOKEN=$(hf auth token) \\
        --timeout 15m \\
        https://huggingface.co/datasets/<user>/<repo>/resolve/main/profile_on_h200.py \\
        -- --num-docs 50 --batch-size 512 --dtype bf16
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger("profile_on_h200")


def _sync(device: Optional[str]) -> None:
    if not device or device == "cpu":
        return
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class PhaseAccumulator:
    def __init__(self, sync_device: Optional[str] = None) -> None:
        self.sync_device = sync_device
        self._per_doc: List[Dict[str, float]] = []
        self._current: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        _sync(self.sync_device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.sync_device)
            self._current[name] += (time.perf_counter() - t0) * 1000.0
            self._counts[name] += 1

    def commit_doc(self) -> None:
        self._per_doc.append(dict(self._current))
        self._current = defaultdict(float)

    def per_doc_records(self) -> List[Dict[str, float]]:
        return list(self._per_doc)

    def call_counts(self) -> Dict[str, int]:
        return dict(self._counts)


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[int(0.95 * (len(s) - 1))]


def summarize(records: List[Dict[str, float]], skip_warmup: bool = True) -> Dict[str, Any]:
    rs = records[1:] if skip_warmup and len(records) > 1 else records
    if not rs:
        return {"error": "no records"}
    phases: set[str] = set()
    for r in rs:
        phases.update(r.keys())
    total_per_doc = [sum(r.values()) for r in rs]
    mean_total = statistics.mean(total_per_doc) if total_per_doc else 0.0
    out: Dict[str, Any] = {"phases": {}}
    for p in sorted(phases):
        vs = [r.get(p, 0.0) for r in rs]
        out["phases"][p] = {
            "mean_ms": statistics.mean(vs),
            "median_ms": statistics.median(vs),
            "p95_ms": _p95(vs),
            "total_ms": sum(vs),
            "pct_of_wallclock": (statistics.mean(vs) / mean_total * 100.0) if mean_total else 0.0,
        }
    out["num_docs"] = len(rs)
    out["mean_total_ms"] = mean_total
    out["warmup_skipped"] = skip_warmup
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--num-docs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--dtype", default="bf16", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--colbert-model", default="lightonai/GTE-ModernColBERT-v1")
    p.add_argument("--queries-file", default=None)
    p.add_argument("--dataset", default="cometadata/arxiv-funding-statement-extraction")
    p.add_argument("--subdir", default="data")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="reduce-overhead")
    return p.parse_args(argv)


def apply_dtype_patch(dtype_str: str) -> None:
    if dtype_str in ("auto", "fp32"):
        return
    import torch
    from pylate import models as pl_models

    target = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
    orig_init = pl_models.ColBERT.__init__

    def patched_init(self: Any, *a: Any, **kw: Any) -> None:
        orig_init(self, *a, **kw)
        self.to(target)
        logger.info("ColBERT weights cast to %s", target)

    pl_models.ColBERT.__init__ = patched_init


def fetch_sample(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    logger.info(
        "fetching %d samples (streaming) from %s[%s/%s.jsonl]",
        args.num_docs,
        args.dataset,
        args.subdir,
        args.split,
    )
    stream = load_dataset(
        args.dataset,
        data_files=f"{args.subdir}/{args.split}.jsonl",
        split="train",
        streaming=True,
    )
    shuffled = stream.shuffle(seed=args.seed, buffer_size=max(1000, args.num_docs * 20))
    rows: List[Dict[str, Any]] = []
    for item in shuffled:
        rows.append(dict(item))
        if len(rows) >= args.num_docs:
            break
    return rows


def make_instrumented_rerank(top: PhaseAccumulator, internals: PhaseAccumulator) -> Callable[..., Any]:
    import torch

    from pylate.rank.rank import RerankResult, reshape_embeddings
    from pylate.scores import colbert_scores
    from pylate.utils import convert_to_tensor as to_tensor

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
            for q_emb, q_doc_ids, q_doc_embs in zip(
                queries_embeddings, documents_ids, documents_embeddings
            ):
                with internals.phase("rerank_convert_tensor"):
                    q_emb = to_tensor(q_emb)
                    q_doc_embs = [to_tensor(e) for e in q_doc_embs]
                with internals.phase("rerank_pad_sequence"):
                    q_doc_embs = torch.nn.utils.rnn.pad_sequence(
                        q_doc_embs, batch_first=True, padding_value=0
                    )
                with internals.phase("rerank_device_move"):
                    if device is not None:
                        q_emb = q_emb.to(device)
                        q_doc_embs = q_doc_embs.to(device)
                    else:
                        q_doc_embs = q_doc_embs.to(q_emb.device)
                with internals.phase("rerank_colbert_scores"):
                    scores = colbert_scores(
                        queries_embeddings=q_emb.unsqueeze(0),
                        documents_embeddings=q_doc_embs,
                    )[0]
                with internals.phase("rerank_sort"):
                    s, idx = torch.sort(input=scores, descending=True)
                with internals.phase("rerank_cpu_sync_tolist"):
                    s_list = s.cpu().tolist()
                    idx_list = idx.tolist()
                with internals.phase("rerank_construct"):
                    docs = [q_doc_ids[i] for i in idx_list]
                    results.append(
                        [RerankResult(id=d, score=sc) for d, sc in zip(docs, s_list)]
                    )
            return results

    return instrumented_rerank


def install_instrumentation(top: PhaseAccumulator, internals: PhaseAccumulator) -> None:
    import pylate
    import pylate.rank
    from funding_statement_extractor.statements import extraction as extraction_mod

    orig_split = extraction_mod._split_into_paragraphs

    def timed_split(text: str) -> List[str]:
        with top.phase("paragraph_split"):
            return orig_split(text)

    extraction_mod._split_into_paragraphs = timed_split

    cls = extraction_mod.SemanticExtractionService
    for method in (
        "_is_likely_funding_statement",
        "_should_extract_full_paragraph",
        "_extract_funding_sentences",
        "_extract_funding_from_long_paragraph",
    ):
        orig = getattr(cls, method)
        phase_name = "filter_" + method.lstrip("_")

        def make_w(inner: Any, ph: str) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                with top.phase(ph):
                    return inner(*args, **kwargs)

            return wrapped

        setattr(cls, method, staticmethod(make_w(orig, phase_name)))

    pylate.rank.rerank = make_instrumented_rerank(top, internals)


def wrap_model_encode(model: Any, top: PhaseAccumulator) -> None:
    orig = model.encode

    def timed(sentences: Any, *args: Any, **kwargs: Any) -> Any:
        is_query = kwargs.get("is_query", False)
        phase = "model_encode_query" if is_query else "model_encode_doc"
        with top.phase(phase):
            return orig(sentences, *args, **kwargs)

    model.encode = timed


def gpu_summary() -> Dict[str, Any]:
    import torch

    info: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        info["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return info


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch

    cuda_ok = False
    for attempt in range(30):
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                cuda_ok = True
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("cuda probe attempt %d failed: %s", attempt, exc)
        time.sleep(1)
    if not cuda_ok:
        logger.error(
            "CUDA not available after 30s wait — will run on CPU (slow, results not representative)"
        )

    import transformers  # noqa: F401  -- must precede sentence_transformers chain
    import sentence_transformers  # noqa: F401
    import pylate  # noqa: F401
    from funding_statement_extractor.config.loader import load_queries
    from funding_statement_extractor.statements.extraction import SemanticExtractionService

    apply_dtype_patch(args.dtype)

    device = "cuda" if cuda_ok else "cpu"
    logger.info("torch=%s pylate=%s device=%s", torch.__version__, pylate.__version__, device)
    if cuda_ok:
        logger.info(
            "gpu=%s total=%.1fGB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    rows = fetch_sample(args)
    logger.info("loaded %d rows", len(rows))
    queries = load_queries(queries_file=args.queries_file)
    logger.info("loaded %d queries", len(queries))

    service = SemanticExtractionService()
    model = service._get_model(args.colbert_model)
    model.to(device)

    if args.compile:
        logger.info("applying torch.compile(mode=%s)", args.compile_mode)
        try:
            model[0].auto_model = torch.compile(
                model[0].auto_model, mode=args.compile_mode
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch.compile failed: %s", exc)

    sync_device = device if device == "cuda" else None
    top = PhaseAccumulator(sync_device=sync_device)
    internals = PhaseAccumulator(sync_device=sync_device)
    install_instrumentation(top, internals)
    wrap_model_encode(model, top)

    per_doc_totals: List[float] = []
    for i, row in enumerate(rows[: args.num_docs]):
        text = row["text"]
        _sync(sync_device)
        t0 = time.perf_counter()
        stmts = service.extract_funding_statements(
            queries=queries,
            content=text,
            model_name=args.colbert_model,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
        _sync(sync_device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        per_doc_totals.append(dt_ms)
        top.commit_doc()
        internals.commit_doc()
        warm = " (warmup)" if i == 0 else ""
        logger.info(
            "[%d/%d] total=%.0fms n_pred=%d n_para=%d%s",
            i + 1,
            args.num_docs,
            dt_ms,
            len(stmts),
            text.count("\n\n") + 1,
            warm,
        )

    top_sum = summarize(top.per_doc_records(), skip_warmup=True)
    int_sum = summarize(internals.per_doc_records(), skip_warmup=True)
    real_total = (
        sum(per_doc_totals[1:]) / max(1, len(per_doc_totals[1:]))
        if len(per_doc_totals) > 1
        else (per_doc_totals[0] if per_doc_totals else 0.0)
    )
    measured = top_sum["mean_total_ms"]
    unattr = real_total - measured

    report = {
        "run_config": vars(args),
        "resolved": {"device": device, **gpu_summary()},
        "wall_clock_per_doc_ms": {
            "mean_excl_warmup": real_total,
            "warmup_ms": per_doc_totals[0] if per_doc_totals else None,
            "num_docs": len(per_doc_totals),
        },
        "top_phases": top_sum,
        "rerank_internals": int_sum,
        "rerank_call_counts": internals.call_counts(),
        "unattributed_ms": unattr,
        "unattributed_pct": (unattr / real_total * 100.0) if real_total > 0 else 0.0,
    }

    print("===PROFILE_JSON===")
    print(json.dumps(report, indent=2))
    print("===END_PROFILE_JSON===")

    logger.info(
        "mean per doc=%.0fms encode=%.0fms rerank=%.0fms unattr=%.1f%%",
        real_total,
        top_sum["phases"].get("model_encode_doc", {}).get("mean_ms", 0.0),
        top_sum["phases"].get("rerank_total", {}).get("mean_ms", 0.0),
        report["unattributed_pct"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
