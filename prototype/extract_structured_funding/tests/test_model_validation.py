"""Tests for model validation functionality."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from funding_extractor.providers import (
    ModelProvider,
    get_available_ollama_models,
    validate_gemini_model,
    validate_ollama_model,
    validate_provider_requirements,
)


class TestGeminiModelValidation:
    """Tests for Gemini model validation."""

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("gemini-2.5-flash", True),
            ("gemini-2.5-flash-lite", True),
            ("gemini-2.5-pro", True),
            ("gemini-2.0-pro", True),
            ("gemini-1.5-flash", True),
            ("gemini-1.5-pro", True),
            ("gemini-1.0-pro", True),
            ("gemini-pro", True),
            ("gemini-pro-vision", True),
            ("gemini-2.5-flash-002", True),
            ("gemini-2.5-pro-latest", True),
            ("invalid-model", False),
            ("gpt-4", False),
            ("llama3.2", False),
            ("gemini", False),
            ("2.5-flash", False),
        ],
    )
    def test_validate_gemini_model(self, model_id: str, expected: bool) -> None:
        """Test Gemini model validation with various model IDs."""
        assert validate_gemini_model(model_id) == expected

    def test_validate_gemini_model_case_insensitive(self) -> None:
        """Test that Gemini validation is case-insensitive."""
        assert validate_gemini_model("GEMINI-2.5-FLASH") is True
        assert validate_gemini_model("Gemini-2.5-Flash") is True


class TestOllamaModelValidation:
    """Tests for Ollama model validation."""

    @patch("funding_extractor.providers.requests.get")
    def test_get_available_ollama_models_success(self, mock_get: MagicMock) -> None:
        """Test successful retrieval of Ollama models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:8b"},
                {"name": "mistral:latest"},
                {"name": "gemma3:4b"},
            ]
        }
        mock_get.return_value = mock_response

        models = get_available_ollama_models()

        assert models == ["qwen3:8b", "mistral:latest", "gemma3:4b"]
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)

    @patch("funding_extractor.providers.requests.get")
    def test_get_available_ollama_models_custom_url(self, mock_get: MagicMock) -> None:
        """Test retrieval of Ollama models with custom URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        get_available_ollama_models("http://localhost:11434")

        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=5)

    @patch("funding_extractor.providers.requests.get")
    def test_get_available_ollama_models_connection_error(
        self, mock_get: MagicMock
    ) -> None:
        """Test handling of connection errors when getting Ollama models."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            get_available_ollama_models()

    @patch("funding_extractor.providers.requests.get")
    def test_get_available_ollama_models_timeout(self, mock_get: MagicMock) -> None:
        """Test handling of timeout when getting Ollama models."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            get_available_ollama_models()

    @patch("funding_extractor.providers.get_available_ollama_models")
    def test_validate_ollama_model_exists(self, mock_get_models: MagicMock) -> None:
        """Test validation when Ollama model exists."""
        mock_get_models.return_value = [
            "qwen3:8b",
            "mistral:latest",
            "gemma3:4b",
        ]

        # Should not raise
        validate_ollama_model("qwen3:8b")
        validate_ollama_model("mistral:latest")
        validate_ollama_model("gemma3:4b")

    @patch("funding_extractor.providers.get_available_ollama_models")
    def test_validate_ollama_model_base_name_match(
        self, mock_get_models: MagicMock
    ) -> None:
        """Test validation with base model name (without tag)."""
        mock_get_models.return_value = [
            "qwen3:8b",
            "gemma3:4b",
            "mistral:latest",
        ]

        # Should match base name
        validate_ollama_model("gemma3")
        validate_ollama_model("mistral")

    @patch("funding_extractor.providers.get_available_ollama_models")
    def test_validate_ollama_model_not_found(self, mock_get_models: MagicMock) -> None:
        """Test validation when Ollama model is not found."""
        mock_get_models.return_value = ["qwen3:8b"]

        with pytest.raises(ValueError, match="Model 'gpt-4' is not available"):
            validate_ollama_model("gpt-4")

        with pytest.raises(ValueError, match="ollama pull mistral"):
            validate_ollama_model("mistral")

    @patch("funding_extractor.providers.get_available_ollama_models")
    def test_validate_ollama_model_empty_list(self, mock_get_models: MagicMock) -> None:
        """Test validation when no models are available."""
        mock_get_models.return_value = []

        with pytest.raises(ValueError, match="Available models: none"):
            validate_ollama_model("gemma3")

    @patch("funding_extractor.providers.get_available_ollama_models")
    def test_validate_ollama_model_connection_error(
        self, mock_get_models: MagicMock
    ) -> None:
        """Test that connection errors are re-raised."""
        mock_get_models.side_effect = ConnectionError("Cannot connect")

        with pytest.raises(ConnectionError, match="Cannot connect"):
            validate_ollama_model("gemma3")


class TestProviderRequirementsValidation:
    """Tests for provider requirements validation with model checks."""

    def test_validate_gemini_requirements_with_valid_model(self) -> None:
        """Test Gemini validation with valid model."""
        # Should not raise
        validate_provider_requirements(
            ModelProvider.GEMINI,
            api_key="test-key",
            model_url=None,
            model_id="gemini-2.5-flash",
        )

    def test_validate_gemini_requirements_with_invalid_model(self) -> None:
        """Test Gemini validation with invalid model."""
        with pytest.raises(ValueError, match="not appear to be a valid Gemini model"):
            validate_provider_requirements(
                ModelProvider.GEMINI,
                api_key="test-key",
                model_url=None,
                model_id="invalid-model",
            )

    @patch("funding_extractor.providers.validate_ollama_model")
    def test_validate_ollama_requirements_with_model(
        self, mock_validate: MagicMock
    ) -> None:
        """Test Ollama validation with model."""
        validate_provider_requirements(
            ModelProvider.OLLAMA,
            api_key=None,
            model_url="http://localhost:11434",
            model_id="gemma3",
        )

        mock_validate.assert_called_once_with("gemma3", "http://localhost:11434")

    def test_validate_requirements_skip_model_validation(self) -> None:
        """Test that model validation can be skipped."""
        # Should not raise even with invalid model
        validate_provider_requirements(
            ModelProvider.GEMINI,
            api_key="test-key",
            model_url=None,
            model_id="invalid-model",
            skip_model_validation=True,
        )

    @patch("funding_extractor.providers.validate_ollama_model")
    def test_validate_ollama_skip_model_validation(
        self, mock_validate: MagicMock
    ) -> None:
        """Test that Ollama model validation can be skipped."""
        validate_provider_requirements(
            ModelProvider.OLLAMA,
            api_key=None,
            model_url=None,
            model_id="gemma3",
            skip_model_validation=True,
        )

        # Should not call validate_ollama_model
        mock_validate.assert_not_called()

    def test_validate_requirements_without_model_id(self) -> None:
        """Test that validation works without model_id."""
        # Should not raise - no model to validate
        validate_provider_requirements(
            ModelProvider.GEMINI,
            api_key="test-key",
            model_url=None,
            model_id=None,
        )

    @patch("funding_extractor.providers.validate_ollama_model")
    def test_validate_ollama_propagates_connection_error(
        self, mock_validate: MagicMock
    ) -> None:
        """Test that connection errors are propagated correctly."""
        mock_validate.side_effect = ConnectionError("Cannot connect to Ollama")

        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            validate_provider_requirements(
                ModelProvider.OLLAMA,
                api_key=None,
                model_url=None,
                model_id="gemma3",
            )
