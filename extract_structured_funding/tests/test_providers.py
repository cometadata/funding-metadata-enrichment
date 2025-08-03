"""Tests for model provider configuration."""

import os
from typing import NewType
from unittest.mock import MagicMock, patch

import pytest

from funding_extractor.providers import (
    ModelProvider,
    get_language_model_class,
    get_provider_config,
    validate_provider_requirements,
)

ModelUrl = NewType("ModelUrl", str)


class TestModelProvider:
    """Tests for ModelProvider enum."""

    def test_provider_enum_values(self) -> None:
        assert ModelProvider.GEMINI.value == "gemini"
        assert ModelProvider.OLLAMA.value == "ollama"

    def test_provider_from_string(self) -> None:
        assert ModelProvider("gemini") == ModelProvider.GEMINI
        assert ModelProvider("ollama") == ModelProvider.OLLAMA

    def test_invalid_provider(self) -> None:
        with pytest.raises(ValueError):
            ModelProvider("invalid")


class TestProviderConfig:
    """Tests for provider configuration."""

    def test_gemini_config(self) -> None:
        config = get_provider_config(ModelProvider.GEMINI)
        assert config.provider == ModelProvider.GEMINI
        assert config.default_model == "gemini-2.5-flash-lite"
        assert config.requires_api_key is True
        assert config.default_url is None

    def test_ollama_config(self) -> None:
        config = get_provider_config(ModelProvider.OLLAMA)
        assert config.provider == ModelProvider.OLLAMA
        assert config.default_model == "llama3.2"
        assert config.requires_api_key is False
        assert config.default_url == "http://localhost:11434"


class TestLanguageModelClass:
    """Tests for getting language model classes."""

    @patch("funding_extractor.providers.lx.inference.GeminiLanguageModel")
    def test_get_gemini_model_class(self, mock_gemini: MagicMock) -> None:
        model_class = get_language_model_class(ModelProvider.GEMINI)
        assert model_class is mock_gemini

    @patch("funding_extractor.providers.lx.inference.OllamaLanguageModel")
    def test_get_ollama_model_class(self, mock_ollama: MagicMock) -> None:
        model_class = get_language_model_class(ModelProvider.OLLAMA)
        assert model_class is mock_ollama


class TestProviderValidation:
    """Tests for provider requirement validation."""

    def test_validate_gemini_with_api_key(self) -> None:
        validate_provider_requirements(
            ModelProvider.GEMINI,
            api_key="test-key",
            model_url=None,
        )

    def test_validate_gemini_without_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key"):
            validate_provider_requirements(
                ModelProvider.GEMINI,
                api_key=None,
                model_url=None,
            )

    def test_validate_ollama_without_api_key(self) -> None:
        validate_provider_requirements(
            ModelProvider.OLLAMA,
            api_key=None,
            model_url="http://localhost:11434",
        )

    def test_validate_ollama_with_custom_url(self) -> None:
        custom_url = ModelUrl("http://gpu-server:11434")
        validate_provider_requirements(
            ModelProvider.OLLAMA,
            api_key=None,
            model_url=custom_url,
        )

    @patch.dict(os.environ, {"OLLAMA_HOST": "http://custom:11434"})
    def test_ollama_uses_env_host(self) -> None:
        config = get_provider_config(ModelProvider.OLLAMA)
        assert config.default_url == "http://custom:11434"

    @patch.dict(os.environ, {}, clear=True)
    def test_ollama_defaults_without_env(self) -> None:
        if "OLLAMA_HOST" in os.environ:
            del os.environ["OLLAMA_HOST"]
        config = get_provider_config(ModelProvider.OLLAMA)
        assert config.default_url == "http://localhost:11434"
