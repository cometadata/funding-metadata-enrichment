import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError


PROMPT_TEMPLATE = """You extract funding or financial support statements from academic text.
- Funding statements often mention grants, fellowships, sponsors, or agencies.
- Return the full funding or acknowledgement statement verbatim (entire sentence(s) or paragraph), not fragments.
- If no funding statement is present, set found to false and return an empty list.
- Do not invent funding information.

Markdown content:
{markdown}
"""


class FundingResult(BaseModel):
    found: bool = Field(..., description="True when a funding/support statement is present.")
    statements: List[str] = Field(
        default_factory=list, description="Verbatim funding/support statements."
    )
    notes: Optional[str] = Field(
        None, description="Optional clarifying note when no funding statement is found."
    )


def iter_markdown_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() != ".md":
            raise ValueError(f"Expected a markdown file, got: {root}")
        yield root
        return

    if not root.is_dir():
        raise ValueError(f"Path does not exist: {root}")

    for path in sorted(root.rglob("*.md")):
        if path.is_file():
            yield path


def extract_funding_statement(
    client: genai.Client, model: str, markdown_text: str
) -> FundingResult:
    response = client.models.generate_content(
        model=model,
        contents=PROMPT_TEMPLATE.format(markdown=markdown_text),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FundingResult,
            temperature=0,
        ),
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, FundingResult):
        return parsed

    if response.text is None:
        raise RuntimeError("Model returned no text for funding extraction.")

    try:
        return FundingResult.model_validate_json(response.text)
    except ValidationError as exc:
        raise RuntimeError(f"Unable to parse funding response: {exc}") from exc


def process_file(client: genai.Client, model: str, path: Path) -> dict:
    markdown_text = path.read_text(encoding="utf-8")
    return extract_with_retries(client, model, markdown_text, path, retries=2)


def extract_with_retries(
    client: genai.Client,
    model: str,
    markdown_text: str,
    path: Path,
    retries: int,
) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            result = extract_funding_statement(client, model, markdown_text)
            payload = result.model_dump(mode="json")
            payload["file"] = str(path)
            return payload
        except Exception as exc:
            last_error = exc
            logging.error(
                "Funding extraction failed for %s (attempt %d/%d): %s",
                path,
                attempt + 1,
                retries + 1,
                exc,
            )

    return {
        "file": str(path),
        "found": False,
        "statements": [],
        "notes": "Model call failed; see error.",
        "error": str(last_error) if last_error else "Unknown error",
    }


def write_results(
    client: genai.Client, model: str, input_path: Path, output_path: Optional[Path]
) -> None:
    destination = (
        output_path.open("w", encoding="utf-8") if output_path else sys.stdout
    )
    files = list(iter_markdown_paths(input_path))
    total = len(files)

    def _write_line(record: dict) -> None:
        destination.write(json.dumps(record, ensure_ascii=False))
        destination.write("\n")
        destination.flush()

    try:
        for idx, path in enumerate(files, start=1):
            logging.info("Processing %s (%d/%d)", path, idx, total)
            try:
                record = process_file(client, model, path)
            except Exception as exc:
                logging.error(
                    "Unexpected failure processing %s: %s",
                    path,
                    exc,
                )
                record = {
                    "file": str(path),
                    "found": False,
                    "statements": [],
                    "notes": "Model call failed; see error.",
                    "error": str(exc),
                }
            _write_line(record)
    finally:
        if output_path is not None:
            destination.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract funding or financial support statements from markdown files "
            "using Google GenAI structured outputs."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Markdown file to process or a directory containing markdown files.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model name to use for extraction (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="Gemini API key. If omitted, falls back to GEMINI_API_KEY/GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON Lines output. Defaults to stdout.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    client = genai.Client(api_key=args.api_key)
    write_results(client, args.model, args.path, args.output)


if __name__ == "__main__":
    main()
