import json
import sys
from collections.abc import Iterator, Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from funding_extractor.providers.vllm import (
    OutputCleaningModel,
    _convert_funder_array_to_extractions,
    _extract_first_fenced_block,
)
from funding_extractor.providers.vllm_config import VLLMExtractionConfig


# --- _extract_first_fenced_block tests ---


class TestExtractFirstFencedBlock:
    def test_single_block_returned_unchanged(self):
        text = '```json\n[{"class": "funder_name", "text": "NSF"}]\n```'
        result = _extract_first_fenced_block(text)
        assert result == '```json\n[{"class": "funder_name", "text": "NSF"}]\n```'

    def test_multiple_blocks_only_first_returned(self):
        text = (
            '```json\n[{"funder_name": "NSF"}]\n```\n\n'
            "Q: Extract funding from the following text:\n"
            '```json\n[{"funder_name": "NIH"}]\n```'
        )
        result = _extract_first_fenced_block(text)
        assert "NSF" in result
        assert "NIH" not in result

    def test_qa_repetition_pattern(self):
        """Reproduces actual error pattern: model generates multiple Q&A blocks."""
        text = (
            '```json\n[{"funder_name": "NSF", "awards": [{"award_ids": ["123"]}]}]\n```\n\n'
            "Q: Extract funding information from the following text:\n"
            "Some other text here...\n\n"
            "A:\n"
            '```json\n[{"funder_name": "DOE", "awards": []}]\n```'
        )
        result = _extract_first_fenced_block(text)
        assert "NSF" in result
        assert "DOE" not in result
        assert "Q:" not in result

    def test_no_fence_returns_raw_text(self):
        text = 'Just some plain text with no fences: [{"funder_name": "NSF"}]'
        result = _extract_first_fenced_block(text)
        assert result == text

    def test_fence_without_json_label(self):
        text = '```\n[{"funder_name": "NSF"}]\n```'
        result = _extract_first_fenced_block(text)
        assert "NSF" in result
        assert result.startswith("```json\n")


# --- _convert_funder_array_to_extractions tests ---


class TestConvertFunderArrayToExtractions:
    def test_single_funder_single_award(self):
        funders = [
            {
                "funder_name": "NSF",
                "awards": [
                    {
                        "funding_scheme": ["Grant"],
                        "award_ids": ["123"],
                        "award_title": ["My Award"],
                    }
                ],
            }
        ]
        result = _convert_funder_array_to_extractions(funders)
        assert result[0] == {"class": "funder_name", "text": "NSF"}
        assert result[1] == {
            "class": "funding_scheme",
            "text": "Grant",
            "attributes": {"funder_name": "NSF"},
        }
        assert result[2] == {
            "class": "award_ids",
            "text": "123",
            "attributes": {"funder_name": "NSF", "funding_scheme": "Grant"},
        }
        assert result[3] == {
            "class": "award_title",
            "text": "My Award",
            "attributes": {"funder_name": "NSF", "funding_scheme": "Grant"},
        }

    def test_multiple_funders_with_awards(self):
        funders = [
            {
                "funder_name": "NSF",
                "awards": [{"funding_scheme": [], "award_ids": ["123"], "award_title": []}],
            },
            {
                "funder_name": "NIH",
                "awards": [{"funding_scheme": [], "award_ids": ["456"], "award_title": []}],
            },
        ]
        result = _convert_funder_array_to_extractions(funders)
        classes = [e["class"] for e in result]
        texts = [e["text"] for e in result]
        assert classes.count("funder_name") == 2
        assert "NSF" in texts
        assert "NIH" in texts
        assert "123" in texts
        assert "456" in texts

    def test_with_funding_scheme_in_attributes(self):
        funders = [
            {
                "funder_name": "ERC",
                "awards": [
                    {
                        "funding_scheme": ["Horizon 2020"],
                        "award_ids": ["101053661"],
                        "award_title": ["Advanced grant"],
                    }
                ],
            }
        ]
        result = _convert_funder_array_to_extractions(funders)
        award_id_entry = [e for e in result if e["class"] == "award_ids"][0]
        assert award_id_entry["attributes"]["funding_scheme"] == "Horizon 2020"
        award_title_entry = [e for e in result if e["class"] == "award_title"][0]
        assert award_title_entry["attributes"]["funding_scheme"] == "Horizon 2020"

    def test_empty_array(self):
        result = _convert_funder_array_to_extractions([])
        assert result == []

    def test_null_funder_name(self):
        funders = [
            {
                "funder_name": None,
                "awards": [{"funding_scheme": ["Program X"], "award_ids": [], "award_title": []}],
            }
        ]
        result = _convert_funder_array_to_extractions(funders)
        assert result[0] == {"class": "funder_name", "text": None}
        assert result[1]["attributes"]["funder_name"] is None

    def test_no_scheme_omits_funding_scheme_from_attributes(self):
        funders = [
            {
                "funder_name": "NSF",
                "awards": [{"funding_scheme": [], "award_ids": ["123"], "award_title": []}],
            }
        ]
        result = _convert_funder_array_to_extractions(funders)
        award_entry = [e for e in result if e["class"] == "award_ids"][0]
        assert "funding_scheme" not in award_entry["attributes"]


# --- OutputCleaningModel tests ---


class _FakeModel(BaseLanguageModel):
    """Minimal BaseLanguageModel for testing OutputCleaningModel delegation."""

    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self._outputs = outputs

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[ScoredOutput]]:
        for text in self._outputs:
            yield [ScoredOutput(score=1.0, output=text)]

    def merge_kwargs(self, kwargs: dict) -> dict:
        return kwargs


class TestOutputCleaningModel:
    def test_delegates_infer(self):
        funder_json = json.dumps([{"funder_name": "NSF", "awards": []}])
        raw_output = f"```json\n{funder_json}\n```"
        fake = _FakeModel([raw_output])
        config = VLLMExtractionConfig(output_format="direct", prompt_template="test.txt")
        wrapper = OutputCleaningModel(fake, config)

        results = list(wrapper.infer(["test prompt"]))
        assert len(results) == 1
        # Should have converted to langextract format
        output_text = results[0][0].output
        parsed = json.loads(output_text.strip("` \njson"))
        assert parsed[0]["class"] == "funder_name"
        assert parsed[0]["text"] == "NSF"

    def test_langextract_format_passthrough(self):
        """When output is already in langextract format, skip conversion."""
        lx_json = json.dumps([{"class": "funder_name", "text": "NSF"}])
        raw_output = f"```json\n{lx_json}\n```"
        fake = _FakeModel([raw_output])
        config = VLLMExtractionConfig(output_format="direct", prompt_template="test.txt")
        wrapper = OutputCleaningModel(fake, config)

        results = list(wrapper.infer(["test prompt"]))
        output_text = results[0][0].output
        parsed = json.loads(output_text.strip("` \njson"))
        assert parsed[0]["class"] == "funder_name"
        assert parsed[0]["text"] == "NSF"

    def test_langextract_mode_no_conversion(self):
        """When output_format is 'langextract', no conversion happens."""
        funder_json = json.dumps([{"funder_name": "NSF", "awards": []}])
        raw_output = f"```json\n{funder_json}\n```"
        fake = _FakeModel([raw_output])
        config = VLLMExtractionConfig(output_format="langextract")
        wrapper = OutputCleaningModel(fake, config)

        results = list(wrapper.infer(["test prompt"]))
        output_text = results[0][0].output
        # Should still extract first fenced block but NOT convert format
        parsed = json.loads(output_text.strip("` \njson"))
        assert "funder_name" in parsed[0]  # Original format preserved
        assert "class" not in parsed[0]


# --- Stop sequences tests ---


class TestStopSequences:
    def test_stop_sequences_passed_to_sampling_params(self):
        """Verify stop sequences are wired to SamplingParams in offline mode."""
        # Create mocks for vllm module
        mock_vllm = MagicMock()
        mock_lora = MagicMock()
        mocks = {
            "vllm": mock_vllm,
            "vllm.lora": mock_lora,
            "vllm.lora.request": mock_lora,
        }

        # Mock SamplingParams to capture kwargs
        captured_kwargs = {}

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        mock_vllm.SamplingParams = FakeSamplingParams

        # Mock engine
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="test output")]
        mock_engine = MagicMock()
        mock_engine.generate.return_value = [mock_output]

        from funding_extractor.providers.vllm import VLLMLanguageModel
        from funding_extractor.providers.vllm_config import (
            VLLMConfig,
            VLLMExtractionConfig,
        )

        config = VLLMConfig(
            model="test-model",
            extraction=VLLMExtractionConfig(stop_sequences=["\nQ:"]),
        )

        with patch.dict(sys.modules, mocks):
            lm = VLLMLanguageModel(config)
            # Patch the engine directly
            VLLMLanguageModel._engine = mock_engine
            try:
                list(lm.infer(["test prompt"]))
                assert captured_kwargs.get("stop") == ["\nQ:"]
            finally:
                VLLMLanguageModel._engine = None

    def test_no_stop_sequences_omits_stop_param(self):
        """Verify no stop param when stop_sequences is empty."""
        mock_vllm = MagicMock()
        mock_lora = MagicMock()
        mocks = {
            "vllm": mock_vllm,
            "vllm.lora": mock_lora,
            "vllm.lora.request": mock_lora,
        }

        captured_kwargs = {}

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        mock_vllm.SamplingParams = FakeSamplingParams

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="test output")]
        mock_engine = MagicMock()
        mock_engine.generate.return_value = [mock_output]

        from funding_extractor.providers.vllm import VLLMLanguageModel
        from funding_extractor.providers.vllm_config import VLLMConfig

        config = VLLMConfig(model="test-model")

        with patch.dict(sys.modules, mocks):
            lm = VLLMLanguageModel(config)
            VLLMLanguageModel._engine = mock_engine
            try:
                list(lm.infer(["test prompt"]))
                assert "stop" not in captured_kwargs
            finally:
                VLLMLanguageModel._engine = None
