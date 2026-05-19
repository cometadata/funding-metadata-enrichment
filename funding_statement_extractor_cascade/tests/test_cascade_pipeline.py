from unittest.mock import patch

from funding_statement_extractor_cascade.cascade.loader import CascadeModels
from funding_statement_extractor_cascade.cascade.pipeline import (
    SYSTEM_CLEAN,
    _run_stage1,
    _run_stage2,
    _run_stage3,
    extract_funding,
)
from funding_statement_extractor_cascade.cascade.types import CascadeConfig

from tests.cascade_fakes import (
    FakeChunker,
    FakeCleaner,
    FakeMbTokenizer,
    FakeQwenTokenizer,
    FakeSpanner,
    autocast_noop,
)


def _models(chunker):
    return CascadeModels(mb_tok=FakeMbTokenizer(), chunker=chunker,
                         spanner=None, qwen_tok=None, cleaner=None)


def test_stage1_short_doc_uses_single_chunk():
    text = "x" * 100
    chunker = FakeChunker(scalars=[5.0])  # sigmoid(5.0) ~= 0.993, above 0.4
    cfg = CascadeConfig(device="cpu")
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        ids, offsets, surviving = _run_stage1(text, _models(chunker), cfg)
    assert chunker.calls == 1
    assert len(surviving) == 1
    assert surviving[0][1:] == (0, 100)
    assert surviving[0][0] > 0.99


def test_stage1_filters_below_threshold():
    text = "x" * 100
    chunker = FakeChunker(scalars=[-5.0])  # sigmoid(-5.0) ~ 0.007, below 0.4
    cfg = CascadeConfig(device="cpu")
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        _, _, surviving = _run_stage1(text, _models(chunker), cfg)
    assert surviving == []


def test_stage1_returns_top_2_above_threshold():
    text = "x" * (8192 + 4096 + 2000)  # 3 chunks worth
    # chunk_probs after sigmoid roughly: [0.99, 0.27, 0.88]
    chunker = FakeChunker(scalars=[5.0, -1.0, 2.0])
    cfg = CascadeConfig(device="cpu")
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        _, _, surviving = _run_stage1(text, _models(chunker), cfg)
    assert len(surviving) == 2
    probs = [p for (p, _, _) in surviving]
    assert probs == sorted(probs, reverse=True)
    assert all(p >= 0.4 for p in probs)


def _models_2(chunker, spanner):
    return CascadeModels(mb_tok=FakeMbTokenizer(), chunker=chunker,
                         spanner=spanner, qwen_tok=None, cleaner=None)


def test_stage2_skips_chunks_above_no_answer_threshold():
    text = "x" * 100
    chunker = FakeChunker(scalars=[5.0])
    # no_answer logit = 5.0 -> sigmoid ~0.993, above 0.5 threshold, chunk skipped
    spanner = FakeSpanner(calls=[([0.1] * 100, [0.1] * 100, 5.0)])
    cfg = CascadeConfig(device="cpu")
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        ids, offsets, surv = _run_stage1(text, _models_2(chunker, spanner), cfg)
        best = _run_stage2(text, surv, _models_2(chunker, spanner), cfg,
                           ids=ids, offsets=offsets)
    assert best is None


def test_stage2_returns_lowest_no_answer_chunk():
    text = "abcdefghij" * 10   # 100 chars/tokens
    chunker = FakeChunker(scalars=[5.0, 5.0])
    # First chunk: no_answer = -5.0 (sigmoid ~0.007), start argmax @ 2, end argmax @ 4
    # Second chunk: no_answer = -2.0 (sigmoid ~0.12)
    s1 = [0.0] * 100; s1[2] = 5.0
    e1 = [0.0] * 100; e1[4] = 5.0
    s2 = [0.0] * 100; s2[10] = 5.0
    e2 = [0.0] * 100; e2[12] = 5.0
    spanner = FakeSpanner(calls=[(s1, e1, -5.0), (s2, e2, -2.0)])
    cfg = CascadeConfig(device="cpu")
    surv = [(0.99, 0, 100), (0.95, 0, 100)]   # both span full text in our fake
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        offsets = [(i, i + 1) for i in range(100)]
        ids = list(range(100))
        best = _run_stage2(text, surv, _models_2(chunker, spanner), cfg,
                            ids=ids, offsets=offsets)
    assert best is not None
    no_a, rough_span, c_s, c_e = best
    assert no_a < 0.01
    assert rough_span == text[2:5]


def test_stage2_returns_none_when_all_skipped():
    text = "x" * 100
    chunker = FakeChunker(scalars=[])
    spanner = FakeSpanner(calls=[([0.0] * 100, [0.0] * 100, 5.0)])
    cfg = CascadeConfig(device="cpu")
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        ids = list(range(100))
        offsets = [(i, i + 1) for i in range(100)]
        best = _run_stage2(text, [(0.9, 0, 100)],
                            _models_2(chunker, spanner), cfg, ids=ids, offsets=offsets)
    assert best is None


def _models_with_cleaner(canned: str):
    qwen = FakeQwenTokenizer()
    return CascadeModels(
        mb_tok=FakeMbTokenizer(),
        chunker=FakeChunker(scalars=[]),
        spanner=FakeSpanner(calls=[]),
        qwen_tok=qwen,
        cleaner=FakeCleaner(canned, qwen),
    )


def test_stage3_passes_rough_in_load_bearing_format():
    text = "intro paragraph. " + "ROUGH_FUNDING_SPAN" + ". closing paragraph."
    char_s = text.index("ROUGH_FUNDING_SPAN")
    char_e = char_s + len("ROUGH_FUNDING_SPAN")
    cfg = CascadeConfig(device="cpu")
    models = _models_with_cleaner("Cleaned statement.")
    out = _run_stage3(text, "ROUGH_FUNDING_SPAN", char_s, char_e, models, cfg)
    assert out == "Cleaned statement."
    msgs = models.qwen_tok.captured_messages[-1]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_CLEAN
    assert msgs[1]["role"] == "user"
    assert "<ROUGH>ROUGH_FUNDING_SPAN</ROUGH>" in msgs[1]["content"]


def test_stage3_returns_empty_on_NONE():
    text = "x" * 1000
    models = _models_with_cleaner("NONE")
    out = _run_stage3(text, "rough", 100, 110, models, CascadeConfig(device="cpu"))
    assert out == ""


def test_stage3_returns_empty_on_lowercase_none():
    text = "x" * 1000
    models = _models_with_cleaner("none")  # any case of "none" maps to empty
    out = _run_stage3(text, "rough", 100, 110, models, CascadeConfig(device="cpu"))
    assert out == ""


def test_stage3_strips_decoded_output():
    text = "x" * 100
    models = _models_with_cleaner("  trimmed result  ")
    out = _run_stage3(text, "rough", 10, 20, models, CascadeConfig(device="cpu"))
    assert out == "trimmed result"


def test_stage3_context_window_respects_doc_boundaries():
    # rough span sits at the start: left context must clamp at 0, not raise
    text = "rough" + ("y" * 1000)
    models = _models_with_cleaner("ok")
    out = _run_stage3(text, "rough", 0, 5, models, CascadeConfig(device="cpu"))
    assert out == "ok"
    user_msg = models.qwen_tok.captured_messages[-1][1]["content"]
    assert user_msg.startswith("<ROUGH>")


def _all_three(canned: str, chunker_scalars, spanner_calls):
    qwen = FakeQwenTokenizer()
    return CascadeModels(
        mb_tok=FakeMbTokenizer(),
        chunker=FakeChunker(scalars=chunker_scalars),
        spanner=FakeSpanner(calls=spanner_calls),
        qwen_tok=qwen,
        cleaner=FakeCleaner(canned, qwen),
    )


def test_e2e_returns_empty_when_no_chunk_survives_stage1():
    text = "x" * 100
    models = _all_three("ignored", chunker_scalars=[-5.0], spanner_calls=[])
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, CascadeConfig(device="cpu"))
    assert r.funding_statement == ""
    assert r.rough_span == ""
    assert r.error is None


def test_e2e_returns_empty_when_stage2_no_answer_high():
    text = "x" * 100
    models = _all_three("ignored",
                        chunker_scalars=[5.0],
                        spanner_calls=[([0.0] * 100, [0.0] * 100, 5.0)])
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, CascadeConfig(device="cpu"))
    assert r.funding_statement == ""
    assert r.rough_span == ""


def test_e2e_full_path_returns_cleaned():
    text = "intro " + ("z" * 40) + " end"
    s_log = [0.0] * 50; s_log[6] = 5.0
    e_log = [0.0] * 50; e_log[10] = 5.0
    models = _all_three("cleaned funding!",
                        chunker_scalars=[5.0],
                        spanner_calls=[(s_log, e_log, -5.0)])
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, CascadeConfig(device="cpu"))
    assert r.funding_statement == "cleaned funding!"
    assert r.rough_span and r.rough_span.startswith("z")
    assert 0.0 <= r.chunk_prob <= 1.0
    assert 0.0 <= r.no_answer_prob <= 1.0
    assert r.latency_ms >= 0.0


def test_e2e_skip_cleanup_returns_rough_span():
    text = "intro " + ("z" * 40) + " end"
    s_log = [0.0] * 50; s_log[6] = 5.0
    e_log = [0.0] * 50; e_log[10] = 5.0
    models = _all_three("would-be-cleaned",
                        chunker_scalars=[5.0],
                        spanner_calls=[(s_log, e_log, -5.0)])
    cfg = CascadeConfig(skip_cleanup=True, device="cpu")
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, cfg)
    assert r.funding_statement == r.rough_span
    assert r.funding_statement.startswith("z")
    assert models.qwen_tok.captured_messages == []


def test_e2e_returns_error_on_exception():
    text = "x" * 100

    class BrokenChunker:
        def __call__(self, *a, **kw):
            raise RuntimeError("boom")

    models = CascadeModels(mb_tok=FakeMbTokenizer(), chunker=BrokenChunker(),
                            spanner=None, qwen_tok=None, cleaner=None)
    with patch("funding_statement_extractor_cascade.cascade.pipeline.torch.amp.autocast",
               autocast_noop):
        r = extract_funding(text, models, CascadeConfig(device="cpu"))
    assert r.funding_statement == ""
    assert r.error is not None
    assert "RuntimeError" in r.error
