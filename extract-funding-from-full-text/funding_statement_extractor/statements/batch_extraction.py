import logging
import os
import queue
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import Pool as MPPool
from typing import Any, Dict, Iterable, Iterator, List, Optional, Pattern, Set, Tuple

import numpy as np
import torch
from pylate.scores import colbert_scores

from funding_statement_extractor.config.loader import load_funding_patterns
from funding_statement_extractor.statements.extraction import (
    SemanticExtractionService,
    _prefilter_paragraphs,
    _split_into_paragraphs,
)
from funding_statement_extractor.statements.models import FundingStatement


logger = logging.getLogger(__name__)


_EOS = object()    # End-of-stream sentinel


_WORKER_COMPILED_POS_PATTERNS: Optional[List[Pattern]] = None
_WORKER_COMPILED_NEG_PATTERNS: Optional[List[Pattern]] = None


def _worker_init(patterns_file: Optional[str], custom_config_dir: Optional[str]) -> None:
    """Pool initializer — runs once per worker process at startup."""
    global _WORKER_COMPILED_POS_PATTERNS, _WORKER_COMPILED_NEG_PATTERNS
    pos_pat, neg_pat = load_funding_patterns(patterns_file, custom_config_dir)
    _WORKER_COMPILED_POS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in pos_pat]
    _WORKER_COMPILED_NEG_PATTERNS = [re.compile(p, re.IGNORECASE) for p in neg_pat]


def _post_task_in_worker(item: "_PostIn", *, top_k: int, threshold: float,
                         regex_match_score_floor: float) -> "BatchResult":
    """Wrapper that pulls precompiled patterns from worker globals."""
    return _post_task(
        item, top_k=top_k, threshold=threshold,
        regex_match_score_floor=regex_match_score_floor,
        compiled_patterns=_WORKER_COMPILED_POS_PATTERNS or [],
        negative_patterns=_WORKER_COMPILED_NEG_PATTERNS or [],
    )


@dataclass
class DocPayload:
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    doc_id: str
    statements: List[FundingStatement]
    enqueue_ts: float
    yield_ts: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, float]] = None


# Internal payload shapes (private; not exported)
@dataclass
class _PreOut:
    doc: DocPayload
    paragraphs: List[str]                 # paragraphs to encode
    original_indices: List[int]           # index back into full_paragraphs
    full_paragraphs: List[str]            # original paragraphs (for extraction)
    enqueue_ts: float
    pre_ms: float
    error: Optional[str] = None


@dataclass
class _PostIn:
    pre: _PreOut
    topk_scores: Optional[List[List[float]]]   # [num_q][k]
    topk_idx: Optional[List[List[int]]]        # [num_q][k]; indexes into pre.paragraphs
    query_names: List[str]
    queue_wait_ms: float
    gpu_ms: float


def _length_sort_permutation(paragraphs: List[str]) -> List[int]:
    """Return indices that sort paragraphs by character count (cheap proxy for token count).

    Used so each model.encode sub-batch contains similar-length paragraphs and pads
    minimally. Stable sort so ties are deterministic.
    """
    return sorted(range(len(paragraphs)), key=lambda i: len(paragraphs[i]))


def _apply_inverse_permutation(items: List[Any], perm: List[int]) -> List[Any]:
    """Reverse the permutation: items[i] is the value at sorted position i;
    return a list where position perm[i] holds items[i]."""
    out: List[Any] = [None] * len(items)
    for sorted_pos, original_pos in enumerate(perm):
        out[original_pos] = items[sorted_pos]
    return out


def _pre_task(doc: DocPayload, *, enable_paragraph_prefilter: bool) -> _PreOut:
    t0 = time.perf_counter()
    try:
        paragraphs = _split_into_paragraphs(doc.text)
        if enable_paragraph_prefilter and paragraphs:
            kept, original_indices = _prefilter_paragraphs(paragraphs)
        else:
            kept = paragraphs
            original_indices = list(range(len(paragraphs)))
        return _PreOut(
            doc=doc,
            paragraphs=kept,
            original_indices=original_indices,
            full_paragraphs=paragraphs,
            enqueue_ts=time.monotonic(),
            pre_ms=(time.perf_counter() - t0) * 1000.0,
        )
    except Exception as exc:
        return _PreOut(
            doc=doc,
            paragraphs=[],
            original_indices=[],
            full_paragraphs=[],
            enqueue_ts=time.monotonic(),
            pre_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _extract_statement_text(paragraph: str, score: float) -> Optional[str]:
    """Mirror of the in-loop extraction in SemanticExtractionService.extract_funding_statements."""
    s = SemanticExtractionService
    if s._should_extract_full_paragraph(paragraph, score):
        return paragraph.strip()
    if len(paragraph) > 1000:
        return s._extract_funding_from_long_paragraph(paragraph)
    funding_sentences = s._extract_funding_sentences(paragraph)
    if funding_sentences:
        return " ".join(funding_sentences)
    return None


def _post_task(
    item: _PostIn,
    *,
    top_k: int,
    threshold: float,
    regex_match_score_floor: float,
    compiled_patterns: List[Pattern],
    negative_patterns: List[Pattern],
) -> BatchResult:
    t0 = time.perf_counter()
    pre = item.pre
    doc_id = pre.doc.doc_id

    if pre.error is not None:
        return BatchResult(
            doc_id=doc_id, statements=[], error=pre.error,
            metadata=pre.doc.metadata,
            enqueue_ts=pre.enqueue_ts, yield_ts=time.monotonic(),
            timings={"pre_ms": pre.pre_ms, "queue_wait_ms": item.queue_wait_ms,
                     "gpu_ms": item.gpu_ms, "post_ms": 0.0},
        )

    if item.topk_scores is None or not pre.paragraphs:
        return BatchResult(
            doc_id=doc_id, statements=[], metadata=pre.doc.metadata,
            enqueue_ts=pre.enqueue_ts, yield_ts=time.monotonic(),
            timings={"pre_ms": pre.pre_ms, "queue_wait_ms": item.queue_wait_ms,
                     "gpu_ms": item.gpu_ms, "post_ms": 0.0},
        )

    full_paragraphs = pre.full_paragraphs
    original_indices = pre.original_indices

    # Cache regex hits per *original* paragraph index to avoid re-running regex
    # across the (query, rank) pairs that pick the same paragraph.
    regex_cache: Dict[int, Tuple[bool, bool]] = {}

    def regex_check(orig_idx: int) -> Tuple[bool, bool]:
        if orig_idx not in regex_cache:
            s = full_paragraphs[orig_idx].lower()
            neg_hit = any(p.search(s) for p in negative_patterns)
            pos_hit = any(p.search(s) for p in compiled_patterns)
            regex_cache[orig_idx] = (neg_hit, pos_hit)
        return regex_cache[orig_idx]

    statements: List[FundingStatement] = []
    seen_text: Set[str] = set()

    try:
        for q_pos, query_name in enumerate(item.query_names):
            for rank in range(len(item.topk_idx[q_pos])):
                local_idx = item.topk_idx[q_pos][rank]
                score = float(item.topk_scores[q_pos][rank])
                orig_idx = original_indices[local_idx]
                paragraph = full_paragraphs[orig_idx]

                # Gating — preserves _is_likely_funding_statement semantics exactly.
                if score > 14.0:
                    accepted = True
                else:
                    neg, pos = regex_check(orig_idx)
                    if neg:
                        accepted = False
                    elif pos:
                        accepted = score > regex_match_score_floor
                    else:
                        accepted = False  # current behaviour: no positive pattern → reject
                if not accepted:
                    continue

                statement_text = _extract_statement_text(paragraph, score)
                if not statement_text or len(statement_text) <= 20:
                    continue
                if statement_text in seen_text:
                    continue
                seen_text.add(statement_text)
                statements.append(FundingStatement(
                    statement=statement_text,
                    score=score,
                    query=query_name,
                    paragraph_idx=orig_idx,   # match legacy: para_id = original_indices[encoded_idx]
                ))
    except Exception as exc:
        return BatchResult(
            doc_id=doc_id, statements=statements, error=f"{type(exc).__name__}: {exc}",
            metadata=pre.doc.metadata,
            enqueue_ts=pre.enqueue_ts, yield_ts=time.monotonic(),
            timings={"pre_ms": pre.pre_ms, "queue_wait_ms": item.queue_wait_ms,
                     "gpu_ms": item.gpu_ms, "post_ms": (time.perf_counter() - t0) * 1000.0},
        )

    return BatchResult(
        doc_id=doc_id, statements=statements, metadata=pre.doc.metadata,
        enqueue_ts=pre.enqueue_ts, yield_ts=time.monotonic(),
        timings={"pre_ms": pre.pre_ms, "queue_wait_ms": item.queue_wait_ms,
                 "gpu_ms": item.gpu_ms, "post_ms": (time.perf_counter() - t0) * 1000.0},
    )


def _build_query_embeddings(model, queries: Dict[str, str]) -> Tuple[torch.Tensor, List[str]]:
    """Encode each query once, pad-stack, move to model's device."""
    ordered_names = list(queries.keys())
    q_tensors: List[torch.Tensor] = []
    for name in ordered_names:
        emb = model.encode([queries[name]], batch_size=1, is_query=True, show_progress_bar=False)
        if isinstance(emb, list) and len(emb) == 1:
            emb = emb[0]
        t = torch.as_tensor(emb)
        if t.ndim == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        q_tensors.append(t)
    padded = torch.nn.utils.rnn.pad_sequence(q_tensors, batch_first=True, padding_value=0)
    device = next(model.parameters()).device
    return padded.to(device), ordered_names


def _encode_with_oom_fallback(model, paragraphs: List[str], encode_batch_size: int):
    """Run model.encode; on torch.cuda.OutOfMemoryError, halve batch size and retry.

    Terminates because at batch_size=1 either it succeeds or the single paragraph is
    too large to encode at all (extremely unlikely on H200) — caller treats that as
    a per-doc failure further out.
    """
    bs = encode_batch_size
    while bs >= 1:
        try:
            return model.encode(paragraphs, batch_size=bs,
                                is_query=False, show_progress_bar=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)


def _run_gpu_pass(
    batch: List[_PreOut],
    model,
    query_emb_padded: torch.Tensor,
    query_names: List[str],
    *,
    encode_batch_size: int,
    top_k: int,
) -> List[_PostIn]:
    """Encode + score + per-doc top-k demux for one pipeline batch.

    Pure: takes a batch and returns one _PostIn per input _PreOut, in the same order.
    Empty-paragraph docs get a _PostIn with topk_scores=None.
    """
    t_gpu_start = time.perf_counter()
    out: List[Optional[_PostIn]] = [None] * len(batch)

    # Flatten the non-empty docs into one big paragraph list with bookkeeping.
    all_paras: List[str] = []
    bounds: List[Tuple[int, int, int]] = []   # (batch_pos, lo, hi) into all_paras
    now = time.monotonic()
    for batch_pos, pre in enumerate(batch):
        if not pre.paragraphs:
            out[batch_pos] = _PostIn(
                pre=pre, topk_scores=None, topk_idx=None,
                query_names=query_names,
                queue_wait_ms=(now - pre.enqueue_ts) * 1000.0,
                gpu_ms=0.0,
            )
            continue
        lo = len(all_paras)
        all_paras.extend(pre.paragraphs)
        bounds.append((batch_pos, lo, len(all_paras)))

    if not all_paras:
        return [item for item in out if item is not None]

    perm = _length_sort_permutation(all_paras)
    sorted_paras = [all_paras[i] for i in perm]
    embs_sorted = _encode_with_oom_fallback(model, sorted_paras, encode_batch_size)
    embs = _apply_inverse_permutation(list(embs_sorted), perm)

    device = next(model.parameters()).device
    doc_tensors = [torch.as_tensor(e) for e in embs]
    padded_docs = torch.nn.utils.rnn.pad_sequence(
        doc_tensors, batch_first=True, padding_value=0
    ).to(device)
    scores = colbert_scores(queries_embeddings=query_emb_padded,
                            documents_embeddings=padded_docs)   # [num_q, num_paras]

    gpu_ms = (time.perf_counter() - t_gpu_start) * 1000.0

    for batch_pos, lo, hi in bounds:
        doc_scores = scores[:, lo:hi]
        k = min(top_k, hi - lo)
        topk_s, topk_i = torch.topk(doc_scores, k=k, dim=1)
        pre = batch[batch_pos]
        out[batch_pos] = _PostIn(
            pre=pre,
            topk_scores=topk_s.cpu().tolist(),
            topk_idx=topk_i.cpu().tolist(),
            query_names=query_names,
            queue_wait_ms=(now - pre.enqueue_ts) * 1000.0,
            gpu_ms=gpu_ms,
        )

    return [item for item in out if item is not None]


# Module-level so monkeypatch.setattr works in tests
def _get_or_load_model(model_name: str):
    from pylate import models
    return models.ColBERT(model_name_or_path=model_name)


def _make_pool(workers: int, init, args):
    return MPPool(processes=workers, initializer=init, initargs=args)


def _drain_until(q: "queue.Queue", *, target_paragraphs: int) -> Tuple[List[_PreOut], bool]:
    """Block on q until accumulated paragraphs >= target_paragraphs OR EOS arrives.

    Returns (batch, eos_seen). Once first item arrives, drain non-blocking until
    the target is hit. EOS terminates the wait early; remaining items in batch
    are still returned for processing.
    """
    batch: List[_PreOut] = []
    paras = 0
    eos = False
    # Block on first item
    first = q.get()
    if first is _EOS:
        return batch, True
    batch.append(first)
    paras += len(first.paragraphs)
    while paras < target_paragraphs:
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            continue   # keep waiting until we hit the threshold
        if item is _EOS:
            eos = True
            break
        batch.append(item)
        paras += len(item.paragraphs)
    return batch, eos


def extract_funding_statements_batch(
    documents: Iterable[DocPayload],
    queries: Dict[str, str],
    *,
    model_name: str = "lightonai/GTE-ModernColBERT-v1",
    top_k: int = 5,
    threshold: float = 10.0,
    enable_paragraph_prefilter: bool = True,
    regex_match_score_floor: Optional[float] = None,
    patterns_file: Optional[str] = None,
    custom_config_dir: Optional[str] = None,
    paragraphs_per_batch: int = 4096,
    encode_batch_size: int = 512,
    workers: Optional[int] = None,
    queue_depth: int = 128,
    dtype: str = "auto",
) -> Iterator[BatchResult]:
    """Pipelined batch extractor. Yields BatchResult per document in completion order."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if regex_match_score_floor is None:
        regex_match_score_floor = 11.0 if enable_paragraph_prefilter else 3.0
    if workers is None:
        workers = max(2, (os.cpu_count() or 4) - 2)

    model = _get_or_load_model(model_name)
    if dtype not in ("auto", "fp32"):
        target = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        model.to(target)
        logger.info("ColBERT weights cast to %s", target)

    query_emb_padded, query_names = _build_query_embeddings(model, queries)

    pool = _make_pool(workers, _worker_init, (patterns_file, custom_config_dir))

    pre_in: "queue.Queue" = queue.Queue(maxsize=queue_depth)
    pre_out: "queue.Queue" = queue.Queue(maxsize=queue_depth)
    post_in: "queue.Queue" = queue.Queue(maxsize=queue_depth)
    done: "queue.Queue" = queue.Queue(maxsize=queue_depth * 2)

    shutdown = threading.Event()

    # ---- Feeder thread: pull from input iter, push DocPayload into pre_in ----
    def feeder():
        try:
            for doc in documents:
                if shutdown.is_set():
                    break
                # Stamp enqueue_ts here so latency includes pre-stage queue time
                pre_in.put(doc, block=True)
        except Exception:
            logger.exception("feeder thread crashed")
        finally:
            pre_in.put(_EOS, block=True)

    # ---- Pre dispatcher thread: pull from pre_in, dispatch to pool, push result to pre_out ----
    def pre_dispatcher():
        pending_pre: List[Any] = []
        try:
            while True:
                doc = pre_in.get(block=True)
                if doc is _EOS:
                    # Wait for all outstanding async tasks so their callbacks fire
                    # and push _PreOut into pre_out BEFORE we post the EOS sentinel.
                    for ar in pending_pre:
                        try:
                            ar.wait()
                        except Exception:
                            # error_callback already pushed an error _PreOut
                            pass
                    pre_out.put(_EOS, block=True)
                    return
                if shutdown.is_set():
                    return
                def cb(result, _doc=doc):
                    try:
                        pre_out.put(result, block=True)
                    except Exception:
                        logger.exception("pre callback put failed")
                def err_cb(exc, _doc=doc):
                    err_pre = _PreOut(
                        doc=_doc, paragraphs=[], original_indices=[], full_paragraphs=[],
                        enqueue_ts=time.monotonic(), pre_ms=0.0,
                        error=f"pre worker died: {type(exc).__name__}: {exc}",
                    )
                    try:
                        pre_out.put(err_pre, block=True)
                    except Exception:
                        logger.exception("pre err_cb put failed")
                ar = pool.apply_async(
                    _pre_task, (doc,),
                    {"enable_paragraph_prefilter": enable_paragraph_prefilter},
                    callback=cb, error_callback=err_cb,
                )
                pending_pre.append(ar)
        except Exception:
            logger.exception("pre dispatcher crashed")
            pre_out.put(_EOS, block=True)

    # ---- GPU consumer thread: drain pre_out into pipeline batches, encode, score, demux ----
    def gpu_consumer():
        try:
            while True:
                batch, eos = _drain_until(pre_out, target_paragraphs=paragraphs_per_batch)
                if batch:
                    try:
                        post_items = _run_gpu_pass(
                            batch, model, query_emb_padded, query_names,
                            encode_batch_size=encode_batch_size, top_k=top_k,
                        )
                        for item in post_items:
                            post_in.put(item, block=True)
                    except Exception as exc:
                        logger.exception("GPU pass failed; marking batch as failed")
                        for pre in batch:
                            err_post = _PostIn(
                                pre=pre, topk_scores=None, topk_idx=None,
                                query_names=query_names,
                                queue_wait_ms=0.0, gpu_ms=0.0,
                            )
                            # propagate error via pre.error
                            pre.error = pre.error or f"gpu_pass: {type(exc).__name__}: {exc}"
                            post_in.put(err_post, block=True)
                if eos:
                    post_in.put(_EOS, block=True)
                    return
        except Exception:
            logger.exception("GPU consumer crashed")
            post_in.put(_EOS, block=True)

    # ---- Post dispatcher thread: pull from post_in, dispatch to pool ----
    def post_dispatcher():
        pending_post: List[Any] = []
        try:
            while True:
                item = post_in.get(block=True)
                if item is _EOS:
                    # Wait for outstanding post tasks so their callbacks push
                    # BatchResults into `done` BEFORE we post the EOS sentinel.
                    for ar in pending_post:
                        try:
                            ar.wait()
                        except Exception:
                            pass
                    done.put(_EOS, block=True)
                    return
                if shutdown.is_set():
                    return
                def cb(result, _item=item):
                    try:
                        done.put(result, block=True)
                    except Exception:
                        logger.exception("post callback put failed")
                def err_cb(exc, _item=item):
                    try:
                        done.put(BatchResult(
                            doc_id=_item.pre.doc.doc_id, statements=[],
                            error=f"post worker died: {type(exc).__name__}: {exc}",
                            metadata=_item.pre.doc.metadata,
                            enqueue_ts=_item.pre.enqueue_ts, yield_ts=time.monotonic(),
                        ), block=True)
                    except Exception:
                        logger.exception("post err_cb put failed")
                ar = pool.apply_async(
                    _post_task_in_worker, (item,),
                    {"top_k": top_k, "threshold": threshold,
                     "regex_match_score_floor": regex_match_score_floor},
                    callback=cb, error_callback=err_cb,
                )
                pending_post.append(ar)
        except Exception:
            logger.exception("post dispatcher crashed")
            done.put(_EOS, block=True)

    threads = []
    for fn in (feeder, pre_dispatcher, gpu_consumer, post_dispatcher):
        t = threading.Thread(target=fn, name=fn.__name__, daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            result = done.get(block=True)
            if result is _EOS:
                return
            yield result
    except (KeyboardInterrupt, GeneratorExit):
        shutdown.set()
        raise
    finally:
        shutdown.set()
        # Drainer thread: during shutdown, pool workers and dispatcher threads
        # may be blocked on queue.put(block=True) because nothing is consuming
        # pre_out / post_in / done anymore. Drain them continuously so those
        # puts unblock, letting pool.join and thread joins complete.
        drain_stop = threading.Event()
        def drainer():
            qs = (pre_in, pre_out, post_in, done)
            while not drain_stop.is_set():
                drained = False
                for q in qs:
                    try:
                        while True:
                            q.get_nowait()
                            drained = True
                    except queue.Empty:
                        pass
                if not drained:
                    time.sleep(0.01)
        drain_thread = threading.Thread(target=drainer, name="shutdown_drainer", daemon=True)
        drain_thread.start()
        try:
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
        for t in threads:
            t.join(timeout=5.0)
        drain_stop.set()
        drain_thread.join(timeout=1.0)
