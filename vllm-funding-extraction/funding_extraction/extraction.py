import concurrent.futures
import json
import logging
import re

from pydantic import TypeAdapter, ValidationError

from funding_extraction.client import VLLMClient
from funding_extraction.config import VLLMConfig
from funding_extraction.models import ExtractionResult, FunderEntity
from funding_extraction.prompts import build_messages, load_examples, load_prompt
from funding_extraction.thinking import strip_thinking

logger = logging.getLogger(__name__)

_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

_funder_list_adapter = TypeAdapter(list[FunderEntity])


def _build_json_schema() -> dict:
    """Generate JSON schema for guided decoding from the FunderEntity model."""
    return _funder_list_adapter.json_schema()


def parse_funders(content: str) -> list[FunderEntity]:
    """Parse model output into a list of FunderEntity objects.

    Handles raw JSON and markdown-fenced JSON (```json ... ```).

    Raises:
        ValueError: If content cannot be parsed as valid JSON.
        ValidationError: If JSON doesn't match the FunderEntity schema.
    """
    text = content.strip()

    # Try to extract from markdown fences first
    match = _FENCED_JSON_PATTERN.search(text)
    if match:
        text = match.group(1).strip()

    return _funder_list_adapter.validate_json(text)


class ExtractionService:
    """Extracts structured funding entities from text using a vLLM server."""

    def __init__(
        self,
        config: VLLMConfig,
        prompt_file: str | None = None,
        examples_file: str | None = None,
    ) -> None:
        self._client = VLLMClient(config)
        self._config = config
        self._prompt = load_prompt(prompt_file)
        self._examples = load_examples(examples_file)
        self._schema = _build_json_schema()

    def extract(self, funding_statement: str) -> tuple[ExtractionResult, list[str]]:
        """Extract funding entities from a single statement.

        Runs extraction_passes times, collecting funders from each pass.
        Each pass retries up to 2 times on parse failure.

        Returns:
            Tuple of (ExtractionResult, list of reasoning traces).
        """
        messages = build_messages(self._prompt, self._examples, funding_statement)
        guided_json = self._schema if self._config.sampling.guided_decoding else None
        all_funders: list[FunderEntity] = []
        reasoning: list[str] = []
        max_attempts = 2

        for pass_num in range(self._config.sampling.extraction_passes):
            for attempt in range(1, max_attempts + 1):
                try:
                    response = self._client.chat(messages, guided_json=guided_json)
                except Exception:
                    logger.exception(
                        "API call failed for statement (%.80s...), pass %d attempt %d",
                        funding_statement,
                        pass_num + 1,
                        attempt,
                    )
                    if attempt == max_attempts:
                        break
                    continue

                content = response.content
                if self._config.sampling.enable_thinking:
                    content, think_traces = strip_thinking(content)
                    reasoning.extend(think_traces)

                if response.reasoning:
                    reasoning.append(response.reasoning)

                try:
                    funders = parse_funders(content)
                    all_funders.extend(funders)
                    break
                except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                    if attempt < max_attempts:
                        logger.warning(
                            "Parse failed for statement (%.80s...), pass %d attempt %d: %s",
                            funding_statement,
                            pass_num + 1,
                            attempt,
                            exc,
                        )
                    else:
                        logger.warning(
                            "Parse failed for statement (%.80s...) after %d attempts: %s",
                            funding_statement,
                            max_attempts,
                            exc,
                        )

        return ExtractionResult(statement=funding_statement, funders=all_funders), reasoning

    def extract_concurrent(
        self,
        statements: list[tuple[str, str]],
        workers: int = 64,
        warmup_count: int = 4,
    ) -> dict[str, tuple[ExtractionResult, list[str]]]:
        """Extract entities from multiple statements concurrently.

        Args:
            statements: List of (doc_id, statement_text) tuples.
            workers: Number of concurrent workers.
            warmup_count: Number of items to process sequentially before
                going parallel (warms KV caches and CUDA graphs).

        Returns:
            Dict mapping doc_id to (ExtractionResult, reasoning_traces).
        """
        results: dict[str, tuple[ExtractionResult, list[str]]] = {}

        if not statements:
            return results

        warmup_count = min(warmup_count, len(statements))
        warmup_items = statements[:warmup_count]
        parallel_items = statements[warmup_count:]

        # Warmup phase: sequential
        for doc_id, text in warmup_items:
            try:
                results[doc_id] = self.extract(text)
            except Exception:
                logger.exception("Warmup extraction failed for doc_id=%s", doc_id)
                results[doc_id] = (ExtractionResult(statement=text, funders=[]), [])

        # Parallel phase
        if parallel_items:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_id = {
                    executor.submit(self.extract, text): doc_id
                    for doc_id, text in parallel_items
                }
                for future in concurrent.futures.as_completed(future_to_id):
                    doc_id = future_to_id[future]
                    try:
                        results[doc_id] = future.result()
                    except Exception:
                        logger.exception("Extraction failed for doc_id=%s", doc_id)
                        text = next(t for d, t in parallel_items if d == doc_id)
                        results[doc_id] = (ExtractionResult(statement=text, funders=[]), [])

        return results
