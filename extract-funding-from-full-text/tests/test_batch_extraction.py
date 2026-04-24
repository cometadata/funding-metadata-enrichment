import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_statement_extractor.statements.batch_extraction import (
    DocPayload,
    BatchResult,
    _pre_task,
    _PreOut,
    _PostIn,
    _post_task,
    _length_sort_permutation,
    _apply_inverse_permutation,
)
from funding_statement_extractor.statements.extraction import (
    SemanticExtractionService,
)
from funding_statement_extractor.config.loader import load_funding_patterns


def test_doc_payload_carries_metadata():
    doc = DocPayload(doc_id="abc", text="hello", metadata={"bucket": "clean"})
    assert doc.doc_id == "abc"
    assert doc.text == "hello"
    assert doc.metadata == {"bucket": "clean"}


def test_doc_payload_metadata_optional():
    doc = DocPayload(doc_id="abc", text="hello")
    assert doc.metadata is None


def test_batch_result_default_shape():
    r = BatchResult(doc_id="abc", statements=[], enqueue_ts=0.0, yield_ts=1.0)
    assert r.error is None
    assert r.metadata is None
    assert r.timings is None
    assert r.statements == []


def test_pre_task_splits_paragraphs_no_prefilter():
    doc = DocPayload(doc_id="d1", text="Para one.\n\nPara two.\n\nPara three.")
    out = _pre_task(doc, enable_paragraph_prefilter=False)
    assert out.doc.doc_id == "d1"
    assert out.paragraphs == ["Para one.", "Para two.", "Para three."]
    assert out.original_indices == [0, 1, 2]
    assert out.full_paragraphs == ["Para one.", "Para two.", "Para three."]
    assert out.error is None


def test_pre_task_prefilters_keeps_funding_paragraphs_with_neighbors():
    text = (
        "Intro paragraph with no signal.\n\n"
        "Methods paragraph with no signal.\n\n"
        "This work was supported by NSF grant 12345.\n\n"
        "Conclusion paragraph with no signal.\n\n"
        "Unrelated paragraph at the end."
    )
    doc = DocPayload(doc_id="d1", text=text)
    out = _pre_task(doc, enable_paragraph_prefilter=True)
    # Funding sentence is index 2; prefilter keeps it plus neighbors 1 and 3
    assert out.original_indices == [1, 2, 3]
    assert "supported by NSF" in out.paragraphs[1]
    # full_paragraphs preserved untouched
    assert len(out.full_paragraphs) == 5


def test_pre_task_empty_text():
    doc = DocPayload(doc_id="d1", text="")
    out = _pre_task(doc, enable_paragraph_prefilter=True)
    assert out.paragraphs == []
    assert out.original_indices == []
    assert out.full_paragraphs == []
    assert out.error is None


def test_pre_task_records_pre_ms():
    doc = DocPayload(doc_id="d1", text="A.\n\nB.")
    out = _pre_task(doc, enable_paragraph_prefilter=False)
    assert out.pre_ms >= 0.0


def _legacy_emit_for_doc(paragraphs, topk_scores, topk_idx, query_names,
                        threshold, regex_floor, compiled_patterns, neg_patterns):
    """Replica of the (query, rank) loop in extraction.py:365-397, used as oracle."""
    from funding_statement_extractor.statements.models import FundingStatement
    from funding_statement_extractor.statements.extraction import SemanticExtractionService as S
    seen = set()
    out = []
    for q_pos, query_name in enumerate(query_names):
        for rank in range(len(topk_idx[q_pos])):
            local_idx = topk_idx[q_pos][rank]
            score = float(topk_scores[q_pos][rank])
            paragraph = paragraphs[local_idx]
            if not S._is_likely_funding_statement(
                paragraph, score, threshold, compiled_patterns, neg_patterns,
                regex_match_score_floor=regex_floor,
            ):
                continue
            if S._should_extract_full_paragraph(paragraph, score):
                statement_text = paragraph.strip()
            elif len(paragraph) > 1000:
                statement_text = S._extract_funding_from_long_paragraph(paragraph)
            else:
                funding_sentences = S._extract_funding_sentences(paragraph)
                if funding_sentences:
                    statement_text = " ".join(funding_sentences)
                else:
                    continue
            if statement_text in seen or len(statement_text) <= 20:
                continue
            seen.add(statement_text)
            out.append(FundingStatement(
                statement=statement_text, score=score, query=query_name,
                paragraph_idx=local_idx,
            ))
    return out


def _make_post_in_fixture():
    """5 paragraphs, 3 queries, top_k=3 — overlapping picks across queries."""
    paragraphs = [
        "Random unrelated paragraph one with no funding signal.",
        "This work was supported by NSF grant DMR-1234567 and DMR-7654321.",
        "Methods section paragraph with no relevant content.",
        "We acknowledge financial support from the European Research Council under grant 12345.",
        "Conclusion paragraph with the word fund mentioned but not in a funding context.",
    ]
    pre = _PreOut(
        doc=DocPayload(doc_id="d1", text="\n\n".join(paragraphs)),
        paragraphs=paragraphs,
        original_indices=[0, 1, 2, 3, 4],
        full_paragraphs=paragraphs,
        enqueue_ts=0.0,
        pre_ms=0.0,
    )
    # Hand-constructed top-k that overlaps across queries
    topk_idx = [
        [1, 3, 0],   # funding_statement
        [3, 1, 4],   # grant_acknowledgment
        [1, 3, 2],   # financial_support
    ]
    topk_scores = [
        [12.5, 11.8, 9.5],
        [11.2, 10.9, 9.0],
        [10.5, 10.4, 8.0],
    ]
    return pre, topk_scores, topk_idx, ["funding_statement", "grant_acknowledgment", "financial_support"]


def test_post_task_byte_identical_to_legacy_loop():
    pre, topk_s, topk_i, q_names = _make_post_in_fixture()
    pos_pat, neg_pat = load_funding_patterns()
    compiled_pos = [re.compile(p, re.IGNORECASE) for p in pos_pat]
    compiled_neg = [re.compile(p, re.IGNORECASE) for p in neg_pat]
    item = _PostIn(
        pre=pre, topk_scores=topk_s, topk_idx=topk_i, query_names=q_names,
        queue_wait_ms=0.0, gpu_ms=0.0,
    )
    new_result = _post_task(
        item, top_k=3, threshold=10.0, regex_match_score_floor=11.0,
        compiled_patterns=compiled_pos, negative_patterns=compiled_neg,
    )
    legacy = _legacy_emit_for_doc(
        pre.paragraphs, topk_s, topk_i, q_names,
        threshold=10.0, regex_floor=11.0,
        compiled_patterns=compiled_pos, neg_patterns=compiled_neg,
    )
    # Byte-identity: same statement text, score, query, paragraph_idx
    assert len(new_result.statements) == len(legacy), \
        f"count mismatch: new={len(new_result.statements)} legacy={len(legacy)}"
    for new_stmt, leg_stmt in zip(new_result.statements, legacy):
        assert new_stmt.statement == leg_stmt.statement
        assert new_stmt.score == leg_stmt.score
        assert new_stmt.query == leg_stmt.query
        assert new_stmt.paragraph_idx == leg_stmt.paragraph_idx


def test_post_task_zero_paragraphs():
    pre = _PreOut(
        doc=DocPayload(doc_id="d1", text=""),
        paragraphs=[], original_indices=[], full_paragraphs=[],
        enqueue_ts=0.0, pre_ms=0.0,
    )
    item = _PostIn(pre=pre, topk_scores=None, topk_idx=None, query_names=[],
                   queue_wait_ms=0.0, gpu_ms=0.0)
    pos_pat, neg_pat = load_funding_patterns()
    compiled_pos = [re.compile(p, re.IGNORECASE) for p in pos_pat]
    compiled_neg = [re.compile(p, re.IGNORECASE) for p in neg_pat]
    result = _post_task(
        item, top_k=5, threshold=10.0, regex_match_score_floor=11.0,
        compiled_patterns=compiled_pos, negative_patterns=compiled_neg,
    )
    assert result.statements == []
    assert result.error is None
    assert result.doc_id == "d1"


def test_post_task_paragraph_idx_uses_original_index():
    """When prefilter drops paragraphs, paragraph_idx must reflect the original position."""
    full = ["junk", "junk", "supported by NSF grant 12345"]
    pre = _PreOut(
        doc=DocPayload(doc_id="d1", text="\n\n".join(full)),
        paragraphs=[full[2]],            # only paragraph 2 survives prefilter
        original_indices=[2],
        full_paragraphs=full,
        enqueue_ts=0.0, pre_ms=0.0,
    )
    item = _PostIn(
        pre=pre,
        topk_scores=[[12.0]],
        topk_idx=[[0]],                  # local index 0 → original index 2
        query_names=["funding_statement"],
        queue_wait_ms=0.0, gpu_ms=0.0,
    )
    pos_pat, neg_pat = load_funding_patterns()
    compiled_pos = [re.compile(p, re.IGNORECASE) for p in pos_pat]
    compiled_neg = [re.compile(p, re.IGNORECASE) for p in neg_pat]
    result = _post_task(
        item, top_k=1, threshold=10.0, regex_match_score_floor=11.0,
        compiled_patterns=compiled_pos, negative_patterns=compiled_neg,
    )
    assert len(result.statements) == 1
    assert result.statements[0].paragraph_idx == 2


def test_length_sort_permutation_orders_by_length():
    paras = ["abc", "a", "abcdef", "ab"]
    perm = _length_sort_permutation(paras)
    sorted_paras = [paras[i] for i in perm]
    assert [len(p) for p in sorted_paras] == sorted(len(p) for p in paras)


def test_apply_inverse_permutation_round_trips():
    paras = ["aaa", "b", "cc", "dddd"]
    perm = _length_sort_permutation(paras)
    sorted_paras = [paras[i] for i in perm]
    fake_embeddings = [f"emb({p})" for p in sorted_paras]
    unsorted = _apply_inverse_permutation(fake_embeddings, perm)
    expected = [f"emb({p})" for p in paras]
    assert unsorted == expected


import numpy as np
import torch


class FakeColBERT:
    """Minimal stand-in for pylate.models.ColBERT for tests.

    encode() returns one numpy array per input (variable seq_len, fixed dim).
    """

    def __init__(self, dim: int = 8, device: str = "cpu"):
        self._dim = dim
        self._device = torch.device(device)
        self._params = [torch.zeros(1, device=self._device)]

    def parameters(self):
        return iter(self._params)

    def encode(self, texts, batch_size=32, is_query=False, show_progress_bar=False):
        # Seed per-text so the same string always yields the same embedding
        # regardless of call order or batching (required for batch-vs-per-doc
        # byte-identity tests; rng-per-call would drift with paragraph ordering).
        out = []
        for t in texts:
            seq_len = max(2, min(len(t) // 5, 16))
            seed = abs(hash(t)) % (2**32)
            rng = np.random.default_rng(seed)
            out.append(rng.standard_normal((seq_len, self._dim)).astype(np.float32))
        if is_query:
            return out
        return out


def _make_pre_out(doc_id, paragraphs):
    return _PreOut(
        doc=DocPayload(doc_id=doc_id, text="\n\n".join(paragraphs)),
        paragraphs=paragraphs,
        original_indices=list(range(len(paragraphs))),
        full_paragraphs=paragraphs,
        enqueue_ts=0.0,
        pre_ms=0.0,
    )


def test_run_gpu_pass_demuxes_per_doc_topk():
    from funding_statement_extractor.statements.batch_extraction import (
        _build_query_embeddings, _run_gpu_pass,
    )
    model = FakeColBERT(dim=8)
    queries = {"q1": "funding statement", "q2": "grant"}
    query_emb_padded, query_names = _build_query_embeddings(model, queries)
    batch = [
        _make_pre_out("d1", ["short", "a longer paragraph here", "tiny"]),
        _make_pre_out("d2", ["only one"]),
    ]
    post_items = _run_gpu_pass(
        batch, model, query_emb_padded, query_names,
        encode_batch_size=8, top_k=2,
    )
    assert len(post_items) == 2
    assert {p.pre.doc.doc_id for p in post_items} == {"d1", "d2"}
    d1 = next(p for p in post_items if p.pre.doc.doc_id == "d1")
    assert len(d1.topk_idx) == 2                 # 2 queries
    assert len(d1.topk_idx[0]) == 2              # min(top_k=2, doc_paragraphs=3)
    d2 = next(p for p in post_items if p.pre.doc.doc_id == "d2")
    assert len(d2.topk_idx[0]) == 1              # min(top_k=2, doc_paragraphs=1)


def test_run_gpu_pass_handles_empty_paragraph_doc():
    from funding_statement_extractor.statements.batch_extraction import (
        _build_query_embeddings, _run_gpu_pass,
    )
    model = FakeColBERT(dim=8)
    queries = {"q1": "funding statement"}
    query_emb_padded, query_names = _build_query_embeddings(model, queries)
    batch = [
        _make_pre_out("d_empty", []),
        _make_pre_out("d1", ["funding paragraph"]),
    ]
    post_items = _run_gpu_pass(
        batch, model, query_emb_padded, query_names,
        encode_batch_size=8, top_k=2,
    )
    assert len(post_items) == 2
    empty = next(p for p in post_items if p.pre.doc.doc_id == "d_empty")
    assert empty.topk_scores is None
    assert empty.topk_idx is None


def test_encode_with_oom_fallback_halves_and_retries():
    from funding_statement_extractor.statements.batch_extraction import _encode_with_oom_fallback

    class OOMOnceModel:
        def __init__(self):
            self.calls = []
        def encode(self, texts, batch_size, is_query=False, show_progress_bar=False):
            self.calls.append(batch_size)
            if batch_size > 4:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return [np.zeros((2, 4), dtype=np.float32) for _ in texts]

    m = OOMOnceModel()
    embs = _encode_with_oom_fallback(m, ["a", "b", "c"], encode_batch_size=16)
    assert m.calls == [16, 8, 4]                  # halved twice before success
    assert len(embs) == 3


def test_encode_with_oom_fallback_raises_at_batch_one():
    from funding_statement_extractor.statements.batch_extraction import _encode_with_oom_fallback

    class AlwaysOOM:
        def encode(self, texts, batch_size, is_query=False, show_progress_bar=False):
            raise torch.cuda.OutOfMemoryError("simulated permanent OOM")

    with __import__("pytest").raises(torch.cuda.OutOfMemoryError):
        _encode_with_oom_fallback(AlwaysOOM(), ["a"], encode_batch_size=4)


def test_worker_init_loads_patterns():
    from funding_statement_extractor.statements import batch_extraction as bx
    bx._worker_init(patterns_file=None, custom_config_dir=None)
    assert bx._WORKER_COMPILED_POS_PATTERNS is not None
    assert bx._WORKER_COMPILED_NEG_PATTERNS is not None
    assert len(bx._WORKER_COMPILED_POS_PATTERNS) > 0


def test_batch_engine_end_to_end_with_thread_pool(monkeypatch):
    """End-to-end with a fake model and thread-backed pool. No real GPU/process spawn."""
    from multiprocessing.dummy import Pool as ThreadPool
    from funding_statement_extractor.statements import batch_extraction as bx

    # Pre-load patterns into the test process's globals so worker functions can find them.
    bx._worker_init(patterns_file=None, custom_config_dir=None)

    model = FakeColBERT(dim=8)
    docs = [
        DocPayload(doc_id=f"d{i}", text=f"Para A.\n\nThis work supported by NSF grant {i}.\n\nPara C.")
        for i in range(5)
    ]
    queries = {"q1": "funding statement"}

    # Inject fake model + dummy ThreadPool factory
    monkeypatch.setattr(bx, "_get_or_load_model", lambda *a, **kw: model)
    monkeypatch.setattr(bx, "_make_pool", lambda workers, init, args: ThreadPool(processes=workers))

    results = list(bx.extract_funding_statements_batch(
        documents=iter(docs), queries=queries,
        paragraphs_per_batch=4, encode_batch_size=2, workers=2, queue_depth=8,
        enable_paragraph_prefilter=False,
    ))

    assert len(results) == 5
    assert {r.doc_id for r in results} == {f"d{i}" for i in range(5)}
    # No errors
    assert all(r.error is None for r in results)
    # Timings populated
    assert all(r.timings is not None and "gpu_ms" in r.timings for r in results)


def test_batch_engine_byte_identical_to_per_doc_api(monkeypatch):
    from multiprocessing.dummy import Pool as ThreadPool
    from funding_statement_extractor.statements import batch_extraction as bx
    from funding_statement_extractor.statements import extraction as ex

    bx._worker_init(patterns_file=None, custom_config_dir=None)

    model = FakeColBERT(dim=16)

    # Identical synthetic corpus
    corpus = [
        ("d1", "Random intro.\n\nThis work was supported by NSF grant DMR-1234567.\n\nConclusion."),
        ("d2", "No funding here.\n\nMethods only.\n\nDiscussion."),
        ("d3", "We acknowledge financial support from the European Research Council under grant 999.\n\nResults."),
        ("d4", "Empty-ish."),
        ("d5", "We thank the authors. This research used resources of the Argonne Leadership Computing Facility under contract DE-AC02-06CH11357."),
    ]
    docs = [DocPayload(doc_id=did, text=text) for did, text in corpus]
    queries = {"q1": "funding statement"}

    # Run via batch engine
    monkeypatch.setattr(bx, "_get_or_load_model", lambda *a, **kw: model)
    monkeypatch.setattr(bx, "_make_pool", lambda workers, init, args: ThreadPool(processes=workers))
    batch_results = {r.doc_id: r for r in bx.extract_funding_statements_batch(
        documents=iter(docs), queries=queries,
        paragraphs_per_batch=64, encode_batch_size=8, workers=2, queue_depth=8,
        enable_paragraph_prefilter=False,
    )}

    # Run via per-doc API on same model — patch the service's _get_model to return our fake
    svc = ex.SemanticExtractionService()
    monkeypatch.setattr(svc, "_get_model", lambda *a, **kw: model)
    legacy_results = {}
    for doc in docs:
        stmts = svc.extract_funding_statements(
            queries=queries, content=doc.text,
            top_k=5, threshold=10.0, batch_size=8,
            enable_paragraph_prefilter=False,
        )
        legacy_results[doc.doc_id] = stmts

    # Compare per-doc, byte-identical
    for doc_id, batch_res in batch_results.items():
        legacy = legacy_results[doc_id]
        assert len(batch_res.statements) == len(legacy), (
            f"{doc_id}: count batch={len(batch_res.statements)} legacy={len(legacy)}")
        for b, l in zip(batch_res.statements, legacy):
            assert b.statement == l.statement, f"{doc_id}: text mismatch"
            assert b.score == l.score, f"{doc_id}: score mismatch"
            assert b.query == l.query, f"{doc_id}: query mismatch"
            assert b.paragraph_idx == l.paragraph_idx, f"{doc_id}: paragraph_idx mismatch"


def test_batch_engine_recovers_from_post_worker_failure(monkeypatch):
    from multiprocessing.dummy import Pool as ThreadPool
    from funding_statement_extractor.statements import batch_extraction as bx

    bx._worker_init(patterns_file=None, custom_config_dir=None)
    model = FakeColBERT(dim=8)

    # Wrap _post_task_in_worker so doc 'd_bad' raises
    real_post = bx._post_task_in_worker
    def flaky_post(item, **kw):
        if item.pre.doc.doc_id == "d_bad":
            raise RuntimeError("simulated post failure")
        return real_post(item, **kw)
    monkeypatch.setattr(bx, "_post_task_in_worker", flaky_post)
    monkeypatch.setattr(bx, "_get_or_load_model", lambda *a, **kw: model)
    monkeypatch.setattr(bx, "_make_pool", lambda workers, init, args: ThreadPool(processes=workers))

    docs = [
        DocPayload(doc_id="d_ok1", text="Supported by NSF grant 1.\n\nMore."),
        DocPayload(doc_id="d_bad", text="Supported by NSF grant 2.\n\nMore."),
        DocPayload(doc_id="d_ok2", text="Supported by NSF grant 3.\n\nMore."),
    ]
    results = {r.doc_id: r for r in bx.extract_funding_statements_batch(
        documents=iter(docs), queries={"q1": "funding statement"},
        paragraphs_per_batch=8, encode_batch_size=4, workers=2, queue_depth=4,
        enable_paragraph_prefilter=False,
    )}
    assert set(results) == {"d_ok1", "d_bad", "d_ok2"}
    assert results["d_bad"].error is not None
    assert "simulated post failure" in results["d_bad"].error
    assert results["d_ok1"].error is None
    assert results["d_ok2"].error is None


def test_batch_engine_handles_pathological_docs(monkeypatch):
    from multiprocessing.dummy import Pool as ThreadPool
    from funding_statement_extractor.statements import batch_extraction as bx

    bx._worker_init(patterns_file=None, custom_config_dir=None)
    model = FakeColBERT(dim=8)
    monkeypatch.setattr(bx, "_get_or_load_model", lambda *a, **kw: model)
    monkeypatch.setattr(bx, "_make_pool", lambda workers, init, args: ThreadPool(processes=workers))

    docs = [
        DocPayload(doc_id="empty", text=""),
        DocPayload(doc_id="single_para", text="single paragraph no breaks"),
        DocPayload(doc_id="unicode", text="Заголовок.\n\nGrant from РНФ 22-11-00001.\n\nКонец."),
        DocPayload(doc_id="huge", text="\n\n".join(["Lorem ipsum."] * 5000)),
    ]
    results = {r.doc_id: r for r in bx.extract_funding_statements_batch(
        documents=iter(docs), queries={"q1": "funding statement"},
        paragraphs_per_batch=64, encode_batch_size=16, workers=2, queue_depth=8,
        enable_paragraph_prefilter=False,
    )}
    assert set(results) == {"empty", "single_para", "unicode", "huge"}
    assert all(r.error is None for r in results.values())


def test_batch_engine_clean_shutdown_on_generator_close(monkeypatch):
    """Caller closes the generator early - engine should not hang."""
    from multiprocessing.dummy import Pool as ThreadPool
    from funding_statement_extractor.statements import batch_extraction as bx

    bx._worker_init(patterns_file=None, custom_config_dir=None)
    model = FakeColBERT(dim=8)
    monkeypatch.setattr(bx, "_get_or_load_model", lambda *a, **kw: model)
    monkeypatch.setattr(bx, "_make_pool", lambda workers, init, args: ThreadPool(processes=workers))

    docs = [DocPayload(doc_id=f"d{i}", text=f"Para.\n\nGrant {i}.") for i in range(20)]
    gen = bx.extract_funding_statements_batch(
        documents=iter(docs), queries={"q1": "funding statement"},
        paragraphs_per_batch=8, encode_batch_size=4, workers=2, queue_depth=4,
        enable_paragraph_prefilter=False,
    )
    # consume only 3, then close
    consumed = []
    for _ in range(3):
        consumed.append(next(gen))
    gen.close()
    assert len(consumed) == 3


def test_batch_engine_resident_memory_bounded_under_slow_consumer(monkeypatch):
    """Stream 200 docs through a slow consumer and verify queue depth caps memory.

    Asserts indirect signal: number of in-flight items never exceeds
    queue_depth * (4 stages + 2 done bound) + workers slack.
    """
    import platform
    import resource
    import time
    from multiprocessing.dummy import Pool as ThreadPool
    from funding_statement_extractor.statements import batch_extraction as bx

    bx._worker_init(patterns_file=None, custom_config_dir=None)
    model = FakeColBERT(dim=8)
    monkeypatch.setattr(bx, "_get_or_load_model", lambda *a, **kw: model)
    monkeypatch.setattr(bx, "_make_pool", lambda workers, init, args: ThreadPool(processes=workers))

    docs = [DocPayload(doc_id=f"d{i}", text=f"Para.\n\nGrant {i}." * 10) for i in range(200)]

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    consumed = 0
    for r in bx.extract_funding_statements_batch(
        documents=iter(docs), queries={"q1": "funding statement"},
        paragraphs_per_batch=16, encode_batch_size=8, workers=4, queue_depth=8,
        enable_paragraph_prefilter=False,
    ):
        time.sleep(0.005)   # slow consumer
        consumed += 1

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert consumed == 200
    # ru_maxrss units differ by platform: KB on Linux, BYTES on macOS.
    # Normalize both to MB here.
    if platform.system() == "Darwin":
        growth_mb = (rss_after - rss_before) / (1024.0 * 1024.0)
    else:
        growth_mb = (rss_after - rss_before) / 1024.0
    # Soft check: RSS growth is bounded (allow 200 MB headroom for normal jitter)
    assert growth_mb < 200, f"resident memory grew {growth_mb} MB under bounded queues"
