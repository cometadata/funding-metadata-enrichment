from unittest.mock import MagicMock, patch

from funding_statement_extractor_cascade.cascade.loader import CascadeModels, load_models
from funding_statement_extractor_cascade.cascade.types import CascadeConfig


@patch("funding_statement_extractor_cascade.cascade.loader.PeftModel")
@patch("funding_statement_extractor_cascade.cascade.loader.AutoModelForCausalLM")
@patch("funding_statement_extractor_cascade.cascade.loader.AutoTokenizer")
@patch("funding_statement_extractor_cascade.cascade.loader.SpanHead")
@patch("funding_statement_extractor_cascade.cascade.loader.ChunkClassifier")
@patch("funding_statement_extractor_cascade.cascade.loader.hf_hub_download")
@patch("funding_statement_extractor_cascade.cascade.loader.torch")
def test_load_models_full_cascade(torch_m, hub, chunk_cls, span_cls,
                                  auto_tok, auto_lm, peft):
    torch_m.load.return_value = {}
    chunk_cls.return_value.to.return_value.eval.return_value = MagicMock(name="chunker")
    span_cls.return_value.to.return_value.eval.return_value = MagicMock(name="spanner")
    auto_lm.from_pretrained.return_value = MagicMock(name="qwen_base")
    peft.from_pretrained.return_value.eval.return_value = MagicMock(name="peft_cleaner")
    auto_tok.from_pretrained.side_effect = [MagicMock(name="mb_tok"),
                                            MagicMock(name="qwen_tok")]

    cfg = CascadeConfig()
    models = load_models(cfg)
    assert isinstance(models, CascadeModels)
    assert models.mb_tok is not None
    assert models.chunker is not None
    assert models.spanner is not None
    assert models.qwen_tok is not None
    assert models.cleaner is not None


@patch("funding_statement_extractor_cascade.cascade.loader.PeftModel")
@patch("funding_statement_extractor_cascade.cascade.loader.AutoModelForCausalLM")
@patch("funding_statement_extractor_cascade.cascade.loader.AutoTokenizer")
@patch("funding_statement_extractor_cascade.cascade.loader.SpanHead")
@patch("funding_statement_extractor_cascade.cascade.loader.ChunkClassifier")
@patch("funding_statement_extractor_cascade.cascade.loader.hf_hub_download")
@patch("funding_statement_extractor_cascade.cascade.loader.torch")
def test_load_models_skip_cleanup_leaves_cleaner_none(torch_m, hub, chunk_cls,
                                                       span_cls, auto_tok,
                                                       auto_lm, peft):
    torch_m.load.return_value = {}
    chunk_cls.return_value.to.return_value.eval.return_value = MagicMock()
    span_cls.return_value.to.return_value.eval.return_value = MagicMock()
    auto_tok.from_pretrained.return_value = MagicMock()

    cfg = CascadeConfig(skip_cleanup=True)
    models = load_models(cfg)
    assert models.cleaner is None
    assert models.qwen_tok is None
    auto_lm.from_pretrained.assert_not_called()
    peft.from_pretrained.assert_not_called()
