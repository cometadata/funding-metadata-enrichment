from funding_statement_extractor_cascade.cascade.types import (
    BatchResult,
    CascadeConfig,
    CascadeResult,
    DocPayload,
)


def test_cascade_config_defaults_match_reference_snippet():
    cfg = CascadeConfig()
    assert cfg.chunk_repo == "cometadata/funding-chunk-classifier-modernbert-base"
    assert cfg.span_repo == "cometadata/funding-extraction-modernbert-base-spanhead"
    assert cfg.clean_repo == "cometadata/funding-cleaning-qwen3-4b-lora"
    assert cfg.mb_base == "answerdotai/ModernBERT-base"
    assert cfg.qwen_base == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.max_tok == 8192
    assert cfg.stride == 4096
    assert cfg.chunk_keep_threshold == 0.4
    assert cfg.chunk_top_k == 2
    assert cfg.span_no_answer_threshold == 0.5
    assert cfg.span_max_length == 300
    assert cfg.cleaner_context_chars == 400
    assert cfg.cleaner_max_new_tokens == 512
    assert cfg.skip_cleanup is False
    assert cfg.device == "cuda"
    assert cfg.dtype == "bf16"


def test_cascade_config_is_frozen():
    import dataclasses
    cfg = CascadeConfig()
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        cfg.max_tok = 1024  # type: ignore[misc]


def test_cascade_result_defaults():
    r = CascadeResult(funding_statement="", rough_span="", chunk_prob=None,
                      no_answer_prob=None, latency_ms=0.0)
    assert r.error is None
    assert r.funding_statement == ""


def test_doc_payload_metadata_optional():
    d = DocPayload(doc_id="x", text="hello")
    assert d.metadata is None


def test_batch_result_carries_through_metadata():
    r = CascadeResult(funding_statement="x", rough_span="x", chunk_prob=0.9,
                      no_answer_prob=0.1, latency_ms=12.3)
    b = BatchResult(doc_id="d1", result=r, enqueue_ts=0.0, yield_ts=0.1,
                    metadata={"input_file": "f.parquet"})
    assert b.metadata["input_file"] == "f.parquet"
    assert b.result.funding_statement == "x"
