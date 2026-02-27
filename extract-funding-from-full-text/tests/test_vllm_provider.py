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
                sampling=VLLMSamplingConfig(
                    temperature=0.5, top_p=0.9, top_k=50, max_tokens=1024,
                ),
            )
            model = VLLMLanguageModel(config)
            list(model.infer(["prompt"]))

            mocks["vllm"].SamplingParams.assert_called_once_with(
                temperature=0.5, top_p=0.9, top_k=50, max_tokens=1024,
                presence_penalty=0.0,
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


    def test_build_extract_params_includes_batch_length(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "sampling": {"batch_length": 32},
                }),
                encoding="utf-8",
            )

            provider = VLLMProvider(vllm_config_path=str(config_path))
            params = provider.build_extract_params(
                "Funded by NSF.", "Extract funders.", []
            )

            assert params["batch_length"] == 32

    def test_build_extract_params_batch_length_default(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({"model": "test-model"}), encoding="utf-8"
            )

            provider = VLLMProvider(vllm_config_path=str(config_path))
            params = provider.build_extract_params(
                "Funded by NSF.", "Extract funders.", []
            )

            assert params["batch_length"] == 64

    def test_build_extract_params_reads_extraction_passes_offline(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "sampling": {"extraction_passes": 1},
                }),
                encoding="utf-8",
            )

            provider = VLLMProvider(vllm_config_path=str(config_path))
            params = provider.build_extract_params(
                "Funded by NSF.", "Extract funders.", []
            )

            assert params["extraction_passes"] == 1
            assert params["max_workers"] == 1

    def test_build_extract_params_reads_extraction_passes_online(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"extraction_passes": 2},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_openai_lm.return_value = MagicMock()
                provider = VLLMProvider(vllm_config_path=str(config_path))
                params = provider.build_extract_params(
                    "Funded by NSF.", "Extract funders.", []
                )

                assert params["extraction_passes"] == 2
                assert params["max_workers"] == 2


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


class TestVLLMProviderOnline:
    def test_online_mode_creates_openai_model(self, tmp_path):
        """When mode=online, the provider should create an OpenAILanguageModel."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 20,
                        "max_tokens": 4096,
                    },
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_openai_lm.return_value = MagicMock()
                provider = VLLMProvider(vllm_config_path=str(config_path))
                mock_openai_lm.assert_called_once()
                call_kwargs = mock_openai_lm.call_args[1]
                assert call_kwargs["model_id"] == "test-model"
                assert call_kwargs["api_key"] == "dummy-key"
                assert call_kwargs["base_url"] == "http://localhost:8000/v1"
                assert call_kwargs["temperature"] == 0.7
                assert call_kwargs["top_p"] == 0.8
                assert call_kwargs["top_k"] == 20
                assert call_kwargs["max_output_tokens"] == 4096
                assert call_kwargs["presence_penalty"] == 0.0

    def test_online_mode_with_lora_uses_lora_name(self, tmp_path):
        """When LoRA is configured in online mode, lora.name becomes the model_id."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "base-model",
                    "mode": "online",
                    "lora": {"name": "my-adapter"},
                    "server": {"url": "http://localhost:8000/v1"},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_openai_lm.return_value = MagicMock()
                provider = VLLMProvider(vllm_config_path=str(config_path))
                call_kwargs = mock_openai_lm.call_args[1]
                assert call_kwargs["model_id"] == "my-adapter"

    def test_model_url_forces_online_mode(self, tmp_path):
        """Passing model_url should override config to online mode."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    # mode not set (defaults to offline)
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_openai_lm.return_value = MagicMock()
                provider = VLLMProvider(
                    model_url="http://my-server:8000/v1",
                    vllm_config_path=str(config_path),
                )
                mock_openai_lm.assert_called_once()
                call_kwargs = mock_openai_lm.call_args[1]
                assert call_kwargs["base_url"] == "http://my-server:8000/v1"

    def test_online_build_extract_params(self, tmp_path):
        """build_extract_params should work the same for online mode."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_model = MagicMock()
                mock_openai_lm.return_value = mock_model
                provider = VLLMProvider(vllm_config_path=str(config_path))
                params = provider.build_extract_params(
                    "Funded by NSF.", "Extract funders.", []
                )
                assert params["model"] is mock_model
                assert params["text_or_documents"] == "Funded by NSF."
                assert params["extraction_passes"] == 3


def _thinking_config(enable: bool = True):
    return VLLMConfig(
        model="test-model",
        sampling=VLLMSamplingConfig(enable_thinking=enable),
    )


class TestVLLMThinkingOffline:
    def test_infer_strips_think_tags(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            raw = '<think>let me reason about this</think>{"json": true}'
            mock_engine.generate.return_value = [_make_request_output(raw)]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_thinking_config(enable=True))
            results = list(model.infer(["prompt"]))

            assert results[0][0].output == '{"json": true}'
            assert model._reasoning_traces == ["let me reason about this"]

    def test_infer_strips_unclosed_think_tag(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            raw = "<think>truncated reasoning that never closes"
            mock_engine.generate.return_value = [_make_request_output(raw)]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_thinking_config(enable=True))
            results = list(model.infer(["prompt"]))

            assert results[0][0].output == ""
            assert model._reasoning_traces == [
                "truncated reasoning that never closes"
            ]

    def test_infer_no_strip_when_thinking_disabled(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            raw = '<think>reasoning</think>{"json": true}'
            mock_engine.generate.return_value = [_make_request_output(raw)]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_thinking_config(enable=False))
            results = list(model.infer(["prompt"]))

            assert results[0][0].output == raw
            assert model._reasoning_traces == []

    def test_infer_no_think_tags_unchanged(self):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            raw = '{"clean": "output"}'
            mock_engine.generate.return_value = [_make_request_output(raw)]
            mocks["vllm"].LLM.return_value = mock_engine

            model = VLLMLanguageModel(_thinking_config(enable=True))
            results = list(model.infer(["prompt"]))

            assert results[0][0].output == '{"clean": "output"}'
            assert model._reasoning_traces == []


class TestVLLMThinkingOnline:
    def test_online_always_patches_client(self, tmp_path):
        """Even when thinking is disabled, the client should be patched."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"enable_thinking": False},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_model = MagicMock()
                mock_openai_lm.return_value = mock_model
                provider = VLLMProvider(vllm_config_path=str(config_path))
                lm = provider._language_model
                assert hasattr(lm, "_reasoning_traces")
                assert hasattr(lm, "_reasoning_lock")

    def test_online_injects_extra_body(self, tmp_path):
        """Patched client should inject chat_template_kwargs in extra_body."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"enable_thinking": True},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_model = MagicMock()
                mock_openai_lm.return_value = mock_model

                # Set up a mock original create that returns a proper response
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.reasoning = None
                mock_response.choices[0].message.reasoning_content = None
                mock_model._client.chat.completions.create.return_value = (
                    mock_response
                )
                original_create = mock_model._client.chat.completions.create

                provider = VLLMProvider(vllm_config_path=str(config_path))
                lm = provider._language_model

                # Call the patched create
                lm._client.chat.completions.create(
                    model="test-model", messages=[]
                )

                # Verify original_create was called with extra_body
                call_kwargs = original_create.call_args[1]
                assert call_kwargs["extra_body"] == {
                    "chat_template_kwargs": {"enable_thinking": True}
                }

    def test_online_captures_reasoning(self, tmp_path):
        """Patched client should capture reasoning from response."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"enable_thinking": True},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_model = MagicMock()
                mock_openai_lm.return_value = mock_model

                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.reasoning = (
                    "I need to extract funders"
                )
                mock_model._client.chat.completions.create.return_value = (
                    mock_response
                )

                provider = VLLMProvider(vllm_config_path=str(config_path))
                lm = provider._language_model

                lm._client.chat.completions.create(
                    model="test-model", messages=[]
                )

                assert lm._reasoning_traces == ["I need to extract funders"]

    def test_online_does_not_inject_thinking_budget(self, tmp_path):
        """Patched client should NOT inject thinking_token_budget (unsupported by vLLM)."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {
                        "enable_thinking": True,
                        "thinking_budget": 4096,
                    },
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_model = MagicMock()
                mock_openai_lm.return_value = mock_model

                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.reasoning = None
                mock_response.choices[0].message.reasoning_content = None
                mock_model._client.chat.completions.create.return_value = (
                    mock_response
                )
                original_create = mock_model._client.chat.completions.create

                provider = VLLMProvider(vllm_config_path=str(config_path))
                lm = provider._language_model

                lm._client.chat.completions.create(
                    model="test-model", messages=[]
                )

                call_kwargs = original_create.call_args[1]
                assert "thinking_token_budget" not in call_kwargs["extra_body"]
                assert call_kwargs["extra_body"]["chat_template_kwargs"] == {
                    "enable_thinking": True
                }

    def test_online_no_thinking_budget_when_none(self, tmp_path):
        """When thinking_budget is None, extra_body should not include it."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"enable_thinking": True},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_model = MagicMock()
                mock_openai_lm.return_value = mock_model

                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.reasoning = None
                mock_response.choices[0].message.reasoning_content = None
                mock_model._client.chat.completions.create.return_value = (
                    mock_response
                )
                original_create = mock_model._client.chat.completions.create

                provider = VLLMProvider(vllm_config_path=str(config_path))
                lm = provider._language_model

                lm._client.chat.completions.create(
                    model="test-model", messages=[]
                )

                call_kwargs = original_create.call_args[1]
                assert "thinking_token_budget" not in call_kwargs["extra_body"]

    def test_online_disabled_does_not_capture(self, tmp_path):
        """When thinking is disabled, reasoning should not be captured."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"enable_thinking": False},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_model = MagicMock()
                mock_openai_lm.return_value = mock_model

                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.reasoning = "some reasoning"
                mock_model._client.chat.completions.create.return_value = (
                    mock_response
                )

                provider = VLLMProvider(vllm_config_path=str(config_path))
                lm = provider._language_model

                lm._client.chat.completions.create(
                    model="test-model", messages=[]
                )

                assert lm._reasoning_traces == []


class TestDrainReasoning:
    def test_drain_returns_and_clears(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "sampling": {"enable_thinking": True},
                }),
                encoding="utf-8",
            )

            provider = VLLMProvider(vllm_config_path=str(config_path))
            # Manually inject some traces
            provider._language_model._reasoning_traces = [
                "trace1", "trace2"
            ]

            traces = provider.drain_reasoning()
            assert traces == ["trace1", "trace2"]
            assert provider._language_model._reasoning_traces == []

    def test_drain_empty_when_no_traces(self, tmp_path):
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({"model": "test-model"}), encoding="utf-8"
            )

            provider = VLLMProvider(vllm_config_path=str(config_path))
            traces = provider.drain_reasoning()
            assert traces == []

    def test_base_provider_drain_reasoning_returns_empty(self):
        """BaseProvider.drain_reasoning() should return empty list."""
        from funding_extractor.providers.base import BaseProvider

        # Can't instantiate ABC directly, but we can test via a concrete subclass
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            # Test via OpenAI provider which inherits from BaseProvider
            # but doesn't override drain_reasoning
            from funding_extractor.providers.openai import OpenAIProvider

            provider = OpenAIProvider(
                model_id="test",
                model_url=None,
                api_key="key",
            )
            assert provider.drain_reasoning() == []


class TestPresencePenalty:
    def test_presence_penalty_passed_to_sampling_params(self):
        """Offline mode should pass presence_penalty to SamplingParams."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMLanguageModel

            mock_engine = MagicMock()
            mock_engine.generate.return_value = [_make_request_output("{}")]
            mocks["vllm"].LLM.return_value = mock_engine

            config = VLLMConfig(
                model="test-model",
                sampling=VLLMSamplingConfig(presence_penalty=1.5),
            )
            model = VLLMLanguageModel(config)
            list(model.infer(["prompt"]))

            mocks["vllm"].SamplingParams.assert_called_once()
            call_kwargs = mocks["vllm"].SamplingParams.call_args[1]
            assert call_kwargs["presence_penalty"] == 1.5

    def test_online_presence_penalty_passed_to_openai_model(self, tmp_path):
        """Online mode should pass presence_penalty to OpenAILanguageModel."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"presence_penalty": 1.5},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_openai_lm.return_value = MagicMock()
                provider = VLLMProvider(vllm_config_path=str(config_path))
                call_kwargs = mock_openai_lm.call_args[1]
                assert call_kwargs["presence_penalty"] == 1.5


class TestThinkingMaxWorkers:
    def test_thinking_online_keeps_parallel_workers(self, tmp_path):
        """Online mode with thinking should still use max_workers=extraction_passes."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"enable_thinking": True},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_openai_lm.return_value = MagicMock()
                provider = VLLMProvider(vllm_config_path=str(config_path))
                params = provider.build_extract_params(
                    "Funded by NSF.", "Extract funders.", []
                )
                assert params["extraction_passes"] == 3
                assert params["max_workers"] == 3

    def test_non_thinking_online_keeps_parallel_workers(self, tmp_path):
        """Online mode without thinking should use max_workers=extraction_passes."""
        mocks = _make_mock_vllm()
        with patch.dict(sys.modules, mocks):
            from funding_extractor.providers.vllm import VLLMProvider

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                yaml.dump({
                    "model": "test-model",
                    "mode": "online",
                    "server": {"url": "http://localhost:8000/v1"},
                    "sampling": {"enable_thinking": False},
                }),
                encoding="utf-8",
            )
            with patch(
                "langextract.providers.openai.OpenAILanguageModel"
            ) as mock_openai_lm:
                mock_openai_lm.return_value = MagicMock()
                provider = VLLMProvider(vllm_config_path=str(config_path))
                params = provider.build_extract_params(
                    "Funded by NSF.", "Extract funders.", []
                )
                assert params["extraction_passes"] == 3
                assert params["max_workers"] == 3
