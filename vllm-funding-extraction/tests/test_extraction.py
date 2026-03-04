import json
from unittest.mock import MagicMock, patch

from funding_extraction.client import ChatResponse
from funding_extraction.config import VLLMConfig, VLLMSamplingConfig, VLLMServerConfig
from funding_extraction.extraction import ExtractionService, parse_funders
from funding_extraction.models import FunderEntity


def _make_config(**overrides) -> VLLMConfig:
    defaults = {
        "model": "test-model",
        "mode": "online",
        "server": VLLMServerConfig(url="http://localhost:8000/v1"),
        "sampling": VLLMSamplingConfig(extraction_passes=1),
    }
    defaults.update(overrides)
    return VLLMConfig(**defaults)


# --- parse_funders tests ---


class TestParseFunders:
    def test_parse_raw_json(self):
        raw = '[{"funder_name": "NSF", "awards": [{"funding_scheme": [], "award_ids": ["123"], "award_title": []}]}]'
        funders = parse_funders(raw)
        assert len(funders) == 1
        assert funders[0].funder_name == "NSF"
        assert funders[0].awards[0].award_ids == ["123"]

    def test_parse_fenced_json(self):
        raw = '```json\n[{"funder_name": "NIH", "awards": []}]\n```'
        funders = parse_funders(raw)
        assert len(funders) == 1
        assert funders[0].funder_name == "NIH"

    def test_parse_empty_array(self):
        funders = parse_funders("[]")
        assert funders == []

    def test_parse_invalid_json_raises(self):
        try:
            parse_funders("not json at all")
            assert False, "Should have raised"
        except (json.JSONDecodeError, ValueError):
            pass

    def test_parse_with_extra_whitespace(self):
        raw = '  \n  [{"funder_name": "NSF", "awards": []}]  \n  '
        funders = parse_funders(raw)
        assert len(funders) == 1

    def test_parse_missing_awards_gets_default(self):
        """FunderEntity has awards=[] as default, so missing key should work."""
        raw = '[{"funder_name": "NSF"}]'
        funders = parse_funders(raw)
        assert funders[0].awards == []


# --- ExtractionService tests ---


class TestExtractionService:
    @patch("funding_extraction.extraction.VLLMClient")
    def test_extract_single_pass(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = ChatResponse(
            content='[{"funder_name": "NSF", "awards": [{"funding_scheme": [], "award_ids": ["123"], "award_title": []}]}]'
        )

        config = _make_config()
        service = ExtractionService(config)
        result, reasoning = service.extract("Funded by NSF grant 123.")

        assert result.statement == "Funded by NSF grant 123."
        assert len(result.funders) == 1
        assert result.funders[0].funder_name == "NSF"
        assert reasoning == []

    @patch("funding_extraction.extraction.VLLMClient")
    def test_extract_with_reasoning(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = ChatResponse(
            content='[{"funder_name": "NSF", "awards": []}]',
            reasoning="I identified NSF as the funder.",
        )

        config = _make_config()
        service = ExtractionService(config)
        result, reasoning = service.extract("Funded by NSF.")

        assert reasoning == ["I identified NSF as the funder."]

    @patch("funding_extraction.extraction.VLLMClient")
    def test_extract_multi_pass(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = ChatResponse(
            content='[{"funder_name": "NSF", "awards": []}]'
        )

        config = _make_config(sampling=VLLMSamplingConfig(extraction_passes=3))
        service = ExtractionService(config)
        result, _ = service.extract("Funded by NSF.")

        # 3 passes, each returning 1 funder = 3 funders total (no dedup)
        assert len(result.funders) == 3
        assert mock_client.chat.call_count == 3

    @patch("funding_extraction.extraction.VLLMClient")
    def test_extract_retry_on_parse_failure(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.side_effect = [
            ChatResponse(content="invalid json garbage"),
            ChatResponse(content='[{"funder_name": "NSF", "awards": []}]'),
        ]

        config = _make_config()
        service = ExtractionService(config)
        result, _ = service.extract("Funded by NSF.")

        assert len(result.funders) == 1
        assert mock_client.chat.call_count == 2

    @patch("funding_extraction.extraction.VLLMClient")
    def test_extract_returns_empty_after_max_retries(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = ChatResponse(content="bad output")

        config = _make_config()
        service = ExtractionService(config)
        result, _ = service.extract("Funded by NSF.")

        assert result.funders == []
        assert result.statement == "Funded by NSF."
        # 1 pass * 2 attempts = 2 calls
        assert mock_client.chat.call_count == 2

    @patch("funding_extraction.extraction.VLLMClient")
    def test_extract_with_thinking_strips_tags(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = ChatResponse(
            content='<think>let me think</think>[{"funder_name": "NSF", "awards": []}]'
        )

        config = _make_config(sampling=VLLMSamplingConfig(enable_thinking=True, extraction_passes=1))
        service = ExtractionService(config)
        result, _ = service.extract("Funded by NSF.")

        assert len(result.funders) == 1
        assert result.funders[0].funder_name == "NSF"

    @patch("funding_extraction.extraction.VLLMClient")
    def test_extract_with_guided_decoding(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = ChatResponse(
            content='[{"funder_name": "NSF", "awards": []}]'
        )

        config = _make_config(sampling=VLLMSamplingConfig(guided_decoding=True, extraction_passes=1))
        service = ExtractionService(config)
        service.extract("Funded by NSF.")

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs.get("guided_json") is not None

    @patch("funding_extraction.extraction.VLLMClient")
    def test_extract_concurrent(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.return_value = ChatResponse(
            content='[{"funder_name": "NSF", "awards": []}]'
        )

        config = _make_config()
        service = ExtractionService(config)
        statements = [("doc1", "Funded by NSF."), ("doc2", "Funded by NIH.")]
        results = service.extract_concurrent(statements, workers=2, warmup_count=0)

        assert "doc1" in results
        assert "doc2" in results
        assert results["doc1"][0].funders[0].funder_name == "NSF"
