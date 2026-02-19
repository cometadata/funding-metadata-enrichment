import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest
import yaml

from funding_extractor.providers.base import ModelProvider
from funding_extractor.providers.vllm_config import (
    VLLMConfig,
    VLLMEngineConfig,
    VLLMLoRAConfig,
    VLLMSamplingConfig,
)


def _make_mock_vllm():
    """Create a mock vllm module with the classes we need."""
    mock_vllm = types.ModuleType("vllm")
    mock_vllm.LLM = MagicMock(name="LLM")
    mock_vllm.SamplingParams = MagicMock(name="SamplingParams")

    mock_lora = types.ModuleType("vllm.lora")
    mock_lora_request = types.ModuleType("vllm.lora.request")
    mock_lora_request.LoRARequest = MagicMock(name="LoRARequest")
    mock_lora.request = mock_lora_request

    mock_vllm.lora = mock_lora

    return {
        "vllm": mock_vllm,
        "vllm.lora": mock_lora,
        "vllm.lora.request": mock_lora_request,
    }


def _make_request_output(text):
    """Create a mock RequestOutput with the given text."""
    output = MagicMock()
    output.outputs = [MagicMock(text=text)]
    return output


def _base_config():
    return VLLMConfig(model="test-model")


def _lora_config():
    return VLLMConfig(
        model="test-model",
        lora=VLLMLoRAConfig(path="/tmp/lora", name="test-lora"),
    )


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton engine between tests."""
    yield
    # Import after test to reset; may not exist if test failed early
    try:
        from funding_extractor.providers.vllm import VLLMLanguageModel
        VLLMLanguageModel._engine = None
        VLLMLanguageModel._engine_lock = threading.Lock()
    except Exception:
        pass


class TestVLLMLanguageModelInfer:
    def test_infer_single_prompt(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output('{"key": "val"}')]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_base_config())
            results = list(model.infer(["test prompt"]))

            assert len(results) == 1
            assert len(results[0]) == 1
            assert results[0][0].score == 1.0
            assert results[0][0].output == '{"key": "val"}'

    def test_infer_batch_prompts(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [
                _make_request_output('{"a": 1}'),
                _make_request_output('{"b": 2}'),
            ]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_base_config())
            results = list(model.infer(["prompt1", "prompt2"]))

            assert len(results) == 2
            assert results[0][0].output == '{"a": 1}'
            assert results[1][0].output == '{"b": 2}'

    def test_infer_passes_sampling_params(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output("{}")]
            mocks["vllm"].LLM.return_value = mock_engine

            config = VLLMConfig(
                model="test-model",
                sampling=VLLMSamplingConfig(temperature=0.5, max_tokens=1024),
            )
            model = VLLMLanguageModel(config)
            list(model.infer(["prompt"]))

            mocks["vllm"].SamplingParams.assert_called_once_with(
                temperature=0.5, max_tokens=1024
            )

    def test_infer_with_lora_request(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output("{}")]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_lora_config())
            list(model.infer(["prompt"]))

            call_kwargs = mock_engine.generate.call_args
            assert call_kwargs[1]["lora_request"] is not None
            mocks["vllm.lora.request"].LoRARequest.assert_called_once_with(
                lora_name="test-lora", lora_int_id=1, lora_path="/tmp/lora"
            )

    def test_infer_without_lora(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output("{}")]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_base_config())
            list(model.infer(["prompt"]))

            call_kwargs = mock_engine.generate.call_args
            assert call_kwargs[1]["lora_request"] is None


class TestVLLMLanguageModelEngine:
    def test_engine_created_once(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output("{}")]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_base_config())
            list(model.infer(["p1"]))
            list(model.infer(["p2"]))

            assert mocks["vllm"].LLM.call_count == 1

    def test_engine_enables_lora_when_configured(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output("{}")]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_lora_config())
            list(model.infer(["prompt"]))

            call_kwargs = mocks["vllm"].LLM.call_args[1]
            assert call_kwargs["enable_lora"] is True
            assert call_kwargs["max_lora_rank"] == 64
            assert call_kwargs["max_loras"] == 1

    def test_engine_no_lora_flags_without_lora(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output("{}")]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_base_config())
            list(model.infer(["prompt"]))

            call_kwargs = mocks["vllm"].LLM.call_args[1]
            assert "enable_lora" not in call_kwargs

    def test_engine_passes_quantization(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output("{}")]
            mocks["vllm"].LLM.return_value = mock_engine

            config = VLLMConfig(
                model="test-model",
                engine=VLLMEngineConfig(quantization="awq"),
            )
            model = VLLMLanguageModel(config)
            list(model.infer(["prompt"]))

            call_kwargs = mocks["vllm"].LLM.call_args[1]
            assert call_kwargs["quantization"] == "awq"


class TestVLLMProviderEnum:
    def test_vllm_in_model_provider(self):
        assert ModelProvider.VLLM == "vllm"

    def test_vllm_provider_config_exists(self):
        from funding_extractor.providers.base import get_provider_config
        config = get_provider_config(ModelProvider.VLLM)
        assert config.requires_api_key is False


class TestVLLMProvider:
    def test_build_extract_params(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({"model": "test-model"}), encoding="utf-8"
            )

            provider = VLLMProvider(
                model_id=None,
                model_url=None,
                api_key=None,
                vllm_config_path=str(config_path),
            )
            params = provider.build_extract_params(
                "Funded by NSF.", "Extract funders.", []
            )

            assert params["text_or_documents"] == "Funded by NSF."
            assert params["prompt_description"] == "Extract funders."
            assert params["extraction_passes"] == 3
            assert params["max_workers"] == 1
            assert params["fence_output"] is True
            assert hasattr(params["model"], "infer")

    def test_provider_property(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({"model": "test-model"}), encoding="utf-8"
            )

            provider = VLLMProvider(
                model_id=None,
                model_url=None,
                api_key=None,
                vllm_config_path=str(config_path),
            )
            assert provider.provider == ModelProvider.VLLM


class TestProviderFactory:
    def test_factory_creates_vllm_provider(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.config.settings import ProviderSettings
            from funding_extractor.providers.factory import ProviderFactory
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({"model": "test-model"}), encoding="utf-8"
            )

            settings = ProviderSettings(
                provider=ModelProvider.VLLM,
                vllm_config_path=str(config_path),
            )
            provider = ProviderFactory.create(settings)
            assert isinstance(provider, VLLMProvider)
