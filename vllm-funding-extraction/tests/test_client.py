from unittest.mock import MagicMock, patch

from funding_extraction.client import ChatResponse, VLLMClient
from funding_extraction.config import VLLMConfig, VLLMSamplingConfig, VLLMServerConfig


def _make_config(**overrides) -> VLLMConfig:
    defaults = {
        "model": "test-model",
        "mode": "online",
        "server": VLLMServerConfig(url="http://localhost:8000/v1"),
    }
    defaults.update(overrides)
    return VLLMConfig(**defaults)


def _mock_response(content: str, reasoning_content: str | None = None):
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content
    message.reasoning_content = reasoning_content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestChatResponse:
    def test_defaults(self):
        r = ChatResponse(content="hello")
        assert r.content == "hello"
        assert r.reasoning is None

    def test_with_reasoning(self):
        r = ChatResponse(content="hello", reasoning="because")
        assert r.reasoning == "because"


class TestVLLMClient:
    @patch("funding_extraction.client.openai.OpenAI")
    def test_basic_chat(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response('[{"funder_name": "NSF"}]')

        config = _make_config()
        client = VLLMClient(config)
        result = client.chat([{"role": "user", "content": "test"}])

        assert result.content == '[{"funder_name": "NSF"}]'
        assert result.reasoning is None
        mock_client.chat.completions.create.assert_called_once()

    @patch("funding_extraction.client.openai.OpenAI")
    def test_chat_with_thinking_enabled(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("output", reasoning_content="thought process")

        config = _make_config(sampling=VLLMSamplingConfig(enable_thinking=True))
        client = VLLMClient(config)
        result = client.chat([{"role": "user", "content": "test"}])

        assert result.reasoning == "thought process"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True

    @patch("funding_extraction.client.openai.OpenAI")
    def test_chat_with_guided_json(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("[]")

        config = _make_config()
        client = VLLMClient(config)
        schema = {"type": "array", "items": {"type": "object"}}
        client.chat([{"role": "user", "content": "test"}], guided_json=schema)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["guided_json"] == schema

    @patch("funding_extraction.client.openai.OpenAI")
    def test_chat_no_extra_body_when_not_needed(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("[]")

        config = _make_config(sampling=VLLMSamplingConfig(enable_thinking=False))
        client = VLLMClient(config)
        client.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("extra_body") is None

    @patch("funding_extraction.client.openai.OpenAI")
    def test_uses_lora_name_as_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("[]")

        from funding_extraction.config import VLLMLoRAConfig
        config = _make_config(lora=VLLMLoRAConfig(name="my-adapter", path="/some/path"))
        client = VLLMClient(config)
        client.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-adapter"

    @patch("funding_extraction.client.openai.OpenAI")
    def test_sampling_params_passed(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("[]")

        config = _make_config(sampling=VLLMSamplingConfig(temperature=0.3, top_p=0.9, max_tokens=1024))
        client = VLLMClient(config)
        client.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["max_tokens"] == 1024

    @patch("funding_extraction.client.openai.OpenAI")
    def test_dummy_api_key_when_none(self, mock_openai_cls):
        config = _make_config(server=VLLMServerConfig(url="http://localhost:8000/v1", api_key=None))
        VLLMClient(config)
        call_kwargs = mock_openai_cls.call_args[1]
        assert call_kwargs["api_key"] == "dummy-key"
