"""Tests for core extraction functionality."""

import concurrent.futures
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import langextract as lx
from funding_extractor.core import (
    _convert_extractions_to_entities,
    create_extraction_prompt,
    create_funding_examples,
    extract_funding_from_statement,
    process_funding_file,
    save_results,
)
from funding_extractor.models import FundingEntity, FundingExtractionResult, Grant
from funding_extractor.providers import ModelProvider


class TestPromptCreation:
    """Tests for prompt and example creation."""

    def test_create_extraction_prompt(self) -> None:
        prompt = create_extraction_prompt()
        assert "funding information" in prompt.lower()
        assert "funding organizations" in prompt.lower()
        assert "grant" in prompt.lower() or "award" in prompt.lower()
        assert "exact text" in prompt.lower()

    def test_create_funding_examples(self) -> None:
        examples = create_funding_examples()
        assert len(examples) >= 2

        first_example = examples[0]
        assert isinstance(first_example, lx.data.ExampleData)
        assert "National Institutes of Health" in first_example.text
        assert len(first_example.extractions) > 0

        has_funder = any(
            e.extraction_class == "funder" for e in first_example.extractions
        )
        has_grant = any(
            e.extraction_class == "grant_id" for e in first_example.extractions
        )
        assert has_funder
        assert has_grant


class TestExtractionConversion:
    """Tests for converting langextract extractions to entities."""

    def test_convert_simple_funder(self) -> None:
        extractions = [
            lx.data.Extraction(
                extraction_class="funder",
                extraction_text="National Science Foundation",
                attributes={"type": "government"},
            ),
        ]

        entities = _convert_extractions_to_entities(extractions)
        assert len(entities) == 1
        assert entities[0].funder == "National Science Foundation"
        assert len(entities[0].grants) == 0

    def test_convert_funder_with_grant(self) -> None:
        extractions = [
            lx.data.Extraction(
                extraction_class="funder",
                extraction_text="NIH",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="grant_id",
                extraction_text="R01-123456",
                attributes={"funder": "NIH"},
            ),
        ]

        entities = _convert_extractions_to_entities(extractions)
        assert len(entities) == 1
        assert entities[0].funder == "NIH"
        assert len(entities[0].grants) == 1
        assert entities[0].grants[0].grant_id == "R01-123456"

    def test_convert_funder_with_multiple_grants(self) -> None:
        extractions = [
            lx.data.Extraction(
                extraction_class="funder",
                extraction_text="European Research Council",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="grant_id",
                extraction_text="ERC-2019-STG-850925",
                attributes={"funder": "European Research Council"},
            ),
            lx.data.Extraction(
                extraction_class="grant_id",
                extraction_text="ERC-2020-ADG-123456",
                attributes={"funder": "European Research Council"},
            ),
        ]

        entities = _convert_extractions_to_entities(extractions)
        assert len(entities) == 1
        assert entities[0].funder == "European Research Council"
        assert len(entities[0].grants) == 2
        assert entities[0].grants[0].grant_id == "ERC-2019-STG-850925"
        assert entities[0].grants[1].grant_id == "ERC-2020-ADG-123456"

    def test_convert_multiple_funders(self) -> None:
        extractions = [
            lx.data.Extraction(
                extraction_class="funder",
                extraction_text="NSF",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="grant_id",
                extraction_text="DMS-111111",
                attributes={"funder": "NSF"},
            ),
            lx.data.Extraction(
                extraction_class="funder",
                extraction_text="DOE",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="grant_id",
                extraction_text="DE-222222",
                attributes={"funder": "DOE"},
            ),
        ]

        entities = _convert_extractions_to_entities(extractions)
        assert len(entities) == 2

        nsf_entity = next(e for e in entities if e.funder == "NSF")
        doe_entity = next(e for e in entities if e.funder == "DOE")

        assert len(nsf_entity.grants) == 1
        assert nsf_entity.grants[0].grant_id == "DMS-111111"
        assert len(doe_entity.grants) == 1
        assert doe_entity.grants[0].grant_id == "DE-222222"

    def test_convert_orphan_grant(self) -> None:
        extractions = [
            lx.data.Extraction(
                extraction_class="grant_id",
                extraction_text="ABC-123",
                attributes={},
            ),
        ]

        entities = _convert_extractions_to_entities(extractions)
        assert len(entities) == 1
        assert entities[0].funder == "Unknown"
        assert len(entities[0].grants) == 1
        assert entities[0].grants[0].grant_id == "ABC-123"


class TestFundingExtraction:
    """Tests for main extraction functionality."""

    @patch("funding_extractor.core.lx.extract")
    def test_extract_funding_from_statement(self, mock_extract: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.extractions = [
            lx.data.Extraction(
                extraction_class="funder",
                extraction_text="Test Foundation",
                attributes={},
            ),
        ]
        mock_extract.return_value = mock_result

        entities = extract_funding_from_statement(
            "Funded by Test Foundation",
            api_key="test_key",
        )

        assert len(entities) == 1
        assert entities[0].funder == "Test Foundation"

        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args.kwargs["text_or_documents"] == "Funded by Test Foundation"
        assert call_args.kwargs["api_key"] == "test_key"
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["extraction_passes"] == 3


class TestTimeoutHandling:
    """Tests for timeout handling in extraction."""

    @patch("funding_extractor.core.lx.extract")
    def test_extract_funding_with_timeout(self, mock_extract: MagicMock) -> None:
        """Test that timeout errors are handled properly."""

        # Simulate a timeout
        def side_effect(**kwargs):
            raise concurrent.futures.TimeoutError("Request timed out")

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
            mock_future = MagicMock()
            mock_future.result.side_effect = concurrent.futures.TimeoutError
            mock_executor.return_value.__enter__.return_value.submit.return_value = (
                mock_future
            )

            # Should return empty list on timeout
            entities = extract_funding_from_statement(
                "Funded by Test Foundation",
                api_key="test_key",
                timeout=1,
            )

            assert entities == []

    @patch("funding_extractor.core.lx.extract")
    def test_extract_funding_custom_timeout(self, mock_extract: MagicMock) -> None:
        """Test that custom timeout is passed correctly."""
        mock_result = MagicMock()
        mock_result.extractions = []
        mock_extract.return_value = mock_result

        # Call with custom timeout
        extract_funding_from_statement(
            "Test statement",
            api_key="test_key",
            timeout=30,
        )


class TestFileProcessing:
    """Tests for file processing functionality."""

    def test_save_results(self, tmp_path: Path) -> None:
        results = [
            FundingExtractionResult(
                doi="10.1234/test.2024",
                funding_statement="Test funding",
                entities=[
                    FundingEntity(
                        funder="Test Funder",
                        grants=[Grant(grant_id="TEST-123")],
                        extraction_texts=["Test Funder TEST-123"],
                    ),
                ],
            ),
        ]

        output_file = tmp_path / "test_output.json"
        save_results(results, output_file)

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["doi"] == "10.1234/test.2024"
        assert data[0]["funding_statement"] == "Test funding"
        assert len(data[0]["entities"]) == 1
        assert data[0]["entities"][0]["funder"] == "Test Funder"
        assert len(data[0]["entities"][0]["grants"]) == 1
        assert data[0]["entities"][0]["grants"][0]["grant_id"] == "TEST-123"

    @patch("funding_extractor.core.extract_funding_from_statement")
    def test_process_funding_file(
        self, mock_extract: MagicMock, tmp_path: Path
    ) -> None:
        mock_extract.return_value = [
            FundingEntity(
                funder="Mock Funder",
                grants=[Grant(grant_id="MOCK-123")],
                extraction_texts=["Mock funding"],
            ),
        ]

        test_data = [
            {
                "doi": "10.1234/test1.2024",
                "funding_statements": ["Funded by Mock Funder MOCK-123"],
            },
            {
                "doi": "10.1234/test2.2024",
                "funding_statements": ["Another funding statement"],
            },
        ]

        input_file = tmp_path / "test_input.json"
        with open(input_file, "w") as f:
            json.dump(test_data, f)

        output_file = tmp_path / "test_output.json"

        results, stats = process_funding_file(
            input_file=input_file,
            output_file=output_file,
            api_key="test_key",
        )

        assert stats.total_documents == 2
        assert stats.successful == 2
        assert stats.failed == 0
        assert stats.total_entities == 2

        assert len(results) == 2
        assert all(len(r.entities) == 1 for r in results)
        assert all(r.entities[0].funder == "Mock Funder" for r in results)

        assert output_file.exists()

    @patch("funding_extractor.core.extract_funding_from_statement")
    def test_process_file_with_errors(
        self, mock_extract: MagicMock, tmp_path: Path
    ) -> None:
        mock_extract.side_effect = [
            [FundingEntity(funder="Good", extraction_texts=["Good"])],
            Exception("API Error"),
        ]

        test_data = [
            {"doi": "10.1234/good", "funding_statements": ["Good funding"]},
            {"doi": "10.1234/bad", "funding_statements": ["Bad funding"]},
        ]

        input_file = tmp_path / "test_input.json"
        with open(input_file, "w") as f:
            json.dump(test_data, f)

        results, stats = process_funding_file(input_file=input_file)

        assert stats.total_documents == 2
        assert stats.successful == 1
        assert stats.failed == 1
        assert stats.total_entities == 1

        assert len(results) == 1
        assert results[0].doi == "10.1234/good"


class TestExtractionWithProviders:
    """Tests for extraction with different providers."""

    @pytest.mark.parametrize(
        "provider,model_id,api_key,model_url,language_model_type",
        [
            (
                ModelProvider.GEMINI,
                "gemini-2.5-flash-lite",
                "test_key",
                None,
                lx.inference.GeminiLanguageModel,
            ),
            (
                ModelProvider.OLLAMA,
                "llama3.2",
                None,
                "http://localhost:11434",
                lx.inference.OllamaLanguageModel,
            ),
            (
                ModelProvider.OPENAI,
                "gpt-4o-mini",
                "test-key",
                None,
                None,
            ),
            (
                ModelProvider.LOCAL_OPENAI,
                "local-model",
                "test-key",
                "http://localhost:8000",
                None,
            ),
        ],
    )
    @patch("funding_extractor.core.lx.extract")
    def test_extract_with_providers(
        self,
        mock_extract: MagicMock,
        provider: ModelProvider,
        model_id: str,
        api_key: str | None,
        model_url: str | None,
        language_model_type: type | None,
    ) -> None:
        """Test extraction with different providers."""
        mock_result = MagicMock()
        mock_result.extractions = [
            lx.data.Extraction(
                extraction_class="funder",
                extraction_text="Test Foundation",
                attributes={},
            ),
        ]
        mock_extract.return_value = mock_result

        entities = extract_funding_from_statement(
            "Funded by Test Foundation",
            provider=provider,
            model_id=model_id,
            api_key=api_key,
            model_url=model_url,
            skip_model_validation=True,
        )

        assert len(entities) == 1
        assert entities[0].funder == "Test Foundation"

        mock_extract.assert_called_once()
        call_args = mock_extract.call_args.kwargs

        # Check provider-specific parameters
        if provider == ModelProvider.GEMINI:
            assert call_args["language_model_type"] == language_model_type
            assert call_args["model_id"] == model_id
            assert call_args["api_key"] == api_key
        elif provider == ModelProvider.OLLAMA:
            assert call_args["language_model_type"] == language_model_type
            assert call_args["language_model_params"] == {
                "model": model_id,
                "model_url": model_url,
                "timeout": 60,
            }
            assert "model_id" not in call_args
            assert "api_key" not in call_args
        elif provider in (ModelProvider.OPENAI, ModelProvider.LOCAL_OPENAI):
            # OpenAI providers use fence_output and use_schema_constraints
            assert call_args["fence_output"] is True
            assert call_args["use_schema_constraints"] is False
            assert call_args["language_model_params"]["model_id"] == model_id
            if api_key:
                assert call_args["language_model_params"]["api_key"] == api_key
            if provider == ModelProvider.LOCAL_OPENAI:
                assert os.environ.get("OPENAI_BASE_URL") == model_url

    @patch("funding_extractor.core.lx.extract")
    def test_extract_openai_with_custom_endpoint(self, mock_extract: MagicMock) -> None:
        """Test OpenAI provider with custom endpoint."""
        mock_result = MagicMock()
        mock_result.extractions = []
        mock_extract.return_value = mock_result

        os.environ.pop("OPENAI_BASE_URL", None)

        result = extract_funding_from_statement(
            funding_statement="Test funding statement",
            provider=ModelProvider.OPENAI,
            model_id="qwen3:8b",
            model_url="http://192.168.1.206:1234",
            api_key="test-key",
            skip_model_validation=True,
        )

        assert os.environ.get("OPENAI_BASE_URL") == "http://192.168.1.206:1234"

        mock_extract.assert_called_once()
        call_args = mock_extract.call_args.kwargs
        assert call_args["fence_output"] is True
        assert call_args["use_schema_constraints"] is False

        assert result == []

    @patch.dict(os.environ, {"GEMINI_API_KEY": ""})
    def test_extract_with_invalid_provider(self) -> None:
        """Test that invalid provider raises appropriate error."""
        with pytest.raises(ValueError, match="requires an API key"):
            extract_funding_from_statement(
                "Test statement",
                provider=ModelProvider.GEMINI,
                # No API key provided
            )

    @patch("funding_extractor.core.lx.extract")
    def test_extract_with_timeout_per_provider(self, mock_extract: MagicMock) -> None:
        """Test that timeout is properly handled across providers."""
        mock_result = MagicMock()
        mock_result.extractions = []
        mock_extract.return_value = mock_result

        for provider in [ModelProvider.GEMINI, ModelProvider.OLLAMA]:
            # Reset mock
            mock_extract.reset_mock()

            extract_funding_from_statement(
                "Test statement",
                provider=provider,
                model_id="test-model",
                api_key="test_key" if provider == ModelProvider.GEMINI else None,
                timeout=45,
                skip_model_validation=True,
            )

            mock_extract.assert_called_once()
