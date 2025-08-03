"""Tests for CLI interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from funding_extractor.cli import main
from funding_extractor.models import ProcessingStats
from funding_extractor.providers import ModelProvider


class TestCLI:
    """Tests for CLI functionality."""

    def test_cli_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Extract funding information" in result.output
        assert "INPUT_FILE" in result.output

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_basic_run(self, mock_process: MagicMock, tmp_path: Path) -> None:
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=10,
                successful=9,
                failed=1,
                total_entities=25,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file)])

        assert result.exit_code == 0
        assert "Processing complete:" in result.output
        assert "Total documents: 10" in result.output
        assert "Successful: 9" in result.output
        assert "Failed: 1" in result.output
        assert "Total entities extracted: 25" in result.output

        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args.kwargs["input_file"] == input_file
        assert call_args.kwargs["output_file"] is None
        assert call_args.kwargs["model_id"] == "gemini-2.5-flash-lite"

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_with_output(self, mock_process: MagicMock, tmp_path: Path) -> None:
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=5,
                successful=5,
                failed=0,
                total_entities=15,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")
        output_file = tmp_path / "output.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(input_file), "--output", str(output_file)],
        )

        assert result.exit_code == 0
        assert f"Results saved to: {output_file}" in result.output

        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args.kwargs["output_file"] == output_file

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_with_timeout(self, mock_process: MagicMock, tmp_path: Path) -> None:
        """Test CLI with custom timeout option."""
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=5,
                successful=5,
                failed=0,
                total_entities=10,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(input_file), "--timeout", "30", "--verbose"],
        )

        assert result.exit_code == 0
        assert "Timeout: 30 seconds" in result.output

        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args.kwargs["timeout"] == 30

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_default_timeout(self, mock_process: MagicMock, tmp_path: Path) -> None:
        """Test CLI uses default timeout of 60 seconds."""
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=1,
                successful=1,
                failed=0,
                total_entities=2,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file)])

        assert result.exit_code == 0

        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args.kwargs["timeout"] == 60

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_with_model(self, mock_process: MagicMock, tmp_path: Path) -> None:
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=0,
                successful=0,
                failed=0,
                total_entities=0,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(input_file), "--model", "gemini-2.0-pro"],
        )

        assert result.exit_code == 0

        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args.kwargs["model_id"] == "gemini-2.0-pro"

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_with_api_key(self, mock_process: MagicMock, tmp_path: Path) -> None:
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=0,
                successful=0,
                failed=0,
                total_entities=0,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(input_file), "--api-key", "test_api_key_123"],
        )

        assert result.exit_code == 0

        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args.kwargs["api_key"] == "test_api_key_123"

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_verbose(self, mock_process: MagicMock, tmp_path: Path) -> None:
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=0,
                successful=0,
                failed=0,
                total_entities=0,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(input_file), "--verbose"],
        )

        assert result.exit_code == 0
        assert f"Processing {input_file}" in result.output
        assert "gemini-2.5-flash-lite" in result.output

    def test_cli_missing_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["/nonexistent/file.json"])

        assert result.exit_code == 2
        assert "does not exist" in result.output or "Invalid value" in result.output

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_processing_error(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        mock_process.side_effect = Exception("Processing failed")

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(main, [str(input_file)])

        assert result.exit_code == 1
        assert "Error: Processing failed" in result.output


class TestCLIProviders:
    """Tests for CLI with different providers."""

    @pytest.mark.parametrize(
        "provider,model,api_key,model_url",
        [
            ("gemini", "gemini-2.5-flash", "test_key", None),
            ("ollama", "gemma3:4b", None, "http://localhost:11434"),
            ("openai", "gpt-4o-mini", "test-key", None),
            ("local_openai", "local-model", None, "http://localhost:8000"),
        ],
    )
    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_with_providers(
        self,
        mock_process: MagicMock,
        tmp_path: Path,
        provider: str,
        model: str,
        api_key: str | None,
        model_url: str | None,
    ) -> None:
        """Test CLI with different providers."""
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=1,
                successful=1,
                failed=0,
                total_entities=2,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        args = [str(input_file), "--provider", provider]
        if model:
            args.extend(["--model", model])
        if api_key:
            args.extend(["--api-key", api_key])
        if model_url:
            args.extend(["--model-url", model_url])

        result = runner.invoke(main, args)

        assert result.exit_code == 0
        mock_process.assert_called_once()
        call_args = mock_process.call_args.kwargs
        assert call_args["provider"].value == provider
        if model:
            assert call_args["model_id"] == model
        if api_key:
            assert call_args["api_key"] == api_key
        if model_url:
            assert call_args["model_url"] == model_url

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_base_url_alias(self, mock_process: MagicMock, tmp_path: Path) -> None:
        """Test CLI with --base-url as alias for --model-url."""
        mock_process.return_value = (
            [],
            MagicMock(total_documents=1, successful=1, failed=0, total_entities=0),
        )

        input_file = tmp_path / "test.json"
        input_file.write_text('{"test": "data"}')

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(input_file),
                "--provider",
                "local_openai",
                "--base-url",
                "http://custom:9000",
            ],
        )

        assert result.exit_code == 0
        mock_process.assert_called_once()
        call_args = mock_process.call_args.kwargs
        assert call_args["model_url"] == "http://custom:9000"

    @patch("funding_extractor.cli.process_funding_file")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-api-key"})
    def test_cli_openai_api_key_from_env(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test CLI gets OpenAI API key from environment."""
        mock_process.return_value = (
            [],
            MagicMock(total_documents=1, successful=1, failed=0, total_entities=0),
        )

        input_file = tmp_path / "test.json"
        input_file.write_text('{"test": "data"}')

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(input_file),
                "--provider",
                "openai",
            ],
        )

        assert result.exit_code == 0
        mock_process.assert_called_once()
        call_args = mock_process.call_args.kwargs
        assert call_args["api_key"] == "env-api-key"

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_verbose_with_provider(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test verbose output with different providers."""
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=1,
                successful=1,
                failed=0,
                total_entities=0,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(input_file),
                "--provider",
                "openai",
                "--model",
                "gpt-4o",
                "--api-key",
                "test-key",
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert "Provider: openai" in result.output
        assert "Model: gpt-4o" in result.output
        assert "API key provided" in result.output

    def test_cli_invalid_provider(self, tmp_path: Path) -> None:
        """Test CLI with invalid provider."""
        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(input_file), "--provider", "invalid_provider"],
        )

        assert result.exit_code == 2
        assert "Invalid value" in result.output

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_backward_compatibility(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test that existing CLI commands still work without --provider."""
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=1,
                successful=1,
                failed=0,
                total_entities=2,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(input_file), "--api-key", "test_key"],
        )

        assert result.exit_code == 0
        mock_process.assert_called_once()
        call_args = mock_process.call_args
        # Should default to Gemini
        assert call_args.kwargs["provider"] == ModelProvider.GEMINI

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_gemini_without_api_key(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test that Gemini provider fails without API key."""
        mock_process.side_effect = ValueError(
            "gemini requires an API key. "
            "Set GEMINI_API_KEY environment variable or use --api-key"
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        # Clear env to ensure no API key
        result = runner.invoke(
            main,
            [str(input_file), "--provider", "gemini"],
            env={"GEMINI_API_KEY": ""},
        )

        assert result.exit_code == 1
        assert "API key" in result.output or "Error" in result.output


class TestCLIModelValidation:
    """Tests for CLI model validation functionality."""

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_with_skip_model_validation_flag(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test that skip-model-validation flag is passed through."""
        mock_process.return_value = (
            [],
            ProcessingStats(
                total_documents=1,
                successful=1,
                failed=0,
                total_entities=2,
            ),
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(input_file),
                "--provider",
                "gemini",
                "--api-key",
                "test_key",
                "--skip-model-validation",
            ],
        )

        assert result.exit_code == 0
        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args.kwargs["skip_model_validation"] is True

    @pytest.mark.parametrize(
        "provider,model,error_msg",
        [
            (
                "gemini",
                "invalid-model",
                "Model 'invalid-model' does not appear to be a valid Gemini model",
            ),
            (
                "ollama",
                "deepseek-r1",
                "Model 'gpt-4' is not available in Ollama. "
                "Available models: gemma3, mistral. "
                "Install it with: ollama pull deepseek-r1",
            ),
        ],
    )
    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_invalid_model(
        self,
        mock_process: MagicMock,
        tmp_path: Path,
        provider: str,
        model: str,
        error_msg: str,
    ) -> None:
        """Test CLI with invalid models for different providers."""
        mock_process.side_effect = ValueError(error_msg)

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        args = [
            str(input_file),
            "--provider",
            provider,
            "--model",
            model,
        ]
        if provider == "gemini":
            args.extend(["--api-key", "test_key"])

        result = runner.invoke(main, args)

        assert result.exit_code == 1
        assert "Error" in result.output

    @patch("funding_extractor.cli.process_funding_file")
    def test_cli_ollama_connection_error(
        self, mock_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test CLI when cannot connect to Ollama."""
        mock_process.side_effect = ConnectionError(
            "Cannot connect to Ollama at http://localhost:11434"
        )

        input_file = tmp_path / "input.json"
        input_file.write_text("[]")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(input_file),
                "--provider",
                "ollama",
                "--model",
                "gemma3",
            ],
        )

        assert result.exit_code == 1
        assert "Error" in result.output
