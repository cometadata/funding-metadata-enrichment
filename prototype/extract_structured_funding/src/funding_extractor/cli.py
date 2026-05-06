from pathlib import Path

import click

from funding_extractor.core import process_funding_file
from funding_extractor.providers import ModelProvider, get_provider_config


@click.command()
@click.argument(
    "input_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file path for results (JSON format)",
)
@click.option(
    "-p",
    "--provider",
    type=click.Choice([p.value for p in ModelProvider]),
    default=ModelProvider.GEMINI.value,
    help="Model provider to use (gemini, ollama, openai, or local_openai)",
)
@click.option(
    "-m",
    "--model",
    default=None,
    help="Model ID to use (defaults to provider's default model)",
)
@click.option(
    "--model-url",
    default=None,
    help="Model API URL (for Ollama or local_openai providers)",
)
@click.option(
    "--base-url",
    default=None,
    help="Base URL for OpenAI-compatible endpoints (alias for --model-url)",
)
@click.option(
    "--api-key",
    help="API key for the model (can also use GEMINI_API_KEY or OPENAI_API_KEY env vars)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--batch-size",
    default=10,
    type=int,
    help="Number of documents to process before saving results",
)
@click.option(
    "--skip-model-validation",
    is_flag=True,
    help="Skip model validation checks (not recommended)",
)
@click.option(
    "--timeout",
    default=60,
    type=int,
    help="Maximum time in seconds to wait for each request (default: 60)",
)
def main(
    input_file: Path,
    output: Path | None,
    provider: str,
    model: str | None,
    model_url: str | None,
    base_url: str | None,
    api_key: str | None,
    verbose: bool,
    batch_size: int,
    skip_model_validation: bool,
    timeout: int,
) -> None:
    """Extract funding information from research papers.

    INPUT_FILE: Path to JSON file containing funding statements
    """
    provider_enum = ModelProvider(provider)

    if base_url and not model_url:
        model_url = base_url

    if not api_key:
        import os

        if provider_enum == ModelProvider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider_enum == ModelProvider.GEMINI:
            api_key = os.environ.get("GEMINI_API_KEY")

    config = get_provider_config(provider_enum)
    if model is None:
        model = config.default_model
    if model_url is None:
        model_url = config.default_url

    if verbose:
        click.echo(f"Processing {input_file}")
        click.echo(f"Provider: {provider}")
        click.echo(f"Model: {model}")
        if model_url:
            click.echo(f"Model URL: {model_url}")
        if api_key:
            click.echo(f"API key provided (length: {len(api_key)})")
        else:
            click.echo("No API key provided via CLI or environment")
        click.echo(f"Timeout: {timeout} seconds")

    try:
        results, stats = process_funding_file(
            input_file=input_file,
            output_file=output,
            provider=provider_enum,
            model_id=model,
            model_url=model_url,
            api_key=api_key,
            batch_size=batch_size,
            skip_model_validation=skip_model_validation,
            timeout=timeout,
        )

        click.echo("\nProcessing complete:")
        click.echo(f"  Total documents: {stats.total_documents}")
        click.echo(f"  Successful: {stats.successful}")
        click.echo(f"  Failed: {stats.failed}")
        click.echo(f"  Total entities extracted: {stats.total_entities}")

        if output:
            click.echo(f"\nResults saved to: {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
