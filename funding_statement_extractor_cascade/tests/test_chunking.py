from funding_statement_extractor_cascade.processing.chunking import build_chunks


def test_short_doc_yields_single_full_chunk():
    chunks = build_chunks(token_count=100, max_tok=8192, stride=4096)
    assert chunks == [(0, 100)]


def test_doc_exactly_max_tok_yields_single_chunk():
    chunks = build_chunks(token_count=8192, max_tok=8192, stride=4096)
    assert chunks == [(0, 8192)]


def test_long_doc_overlaps_at_stride():
    # 12000 tokens with max_tok=8192, stride=4096:
    #   [0, 8192], [4096, 12000]  — second chunk hits the end, loop breaks
    chunks = build_chunks(token_count=12000, max_tok=8192, stride=4096)
    assert chunks == [(0, 8192), (4096, 12000)]


def test_very_long_doc_produces_full_overlap_grid():
    # 20000 tokens: chunks at 0, 4096, 8192, 12288 — last chunk clamps and breaks
    chunks = build_chunks(token_count=20000, max_tok=8192, stride=4096)
    assert chunks == [(0, 8192), (4096, 12288), (8192, 16384), (12288, 20000)]


def test_doc_one_token_past_max_yields_two_chunks():
    chunks = build_chunks(token_count=8193, max_tok=8192, stride=4096)
    assert chunks == [(0, 8192), (4096, 8193)]
