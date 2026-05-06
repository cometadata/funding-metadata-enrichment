"""Pydantic models for the LoRA's structured output, plus a robust parser."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Award(BaseModel):
    model_config = ConfigDict(extra="ignore")
    award_ids: list[str] = Field(default_factory=list)
    funding_scheme: list[str] = Field(default_factory=list)
    award_title: list[str] = Field(default_factory=list)


class Funder(BaseModel):
    model_config = ConfigDict(extra="ignore")
    funder_name: Optional[str] = None
    awards: list[Award] = Field(default_factory=list)


@dataclass
class StatementExtraction:
    """Per-statement extraction result, used by the public API and CLI."""
    funders: Optional[list[Funder]]      # None on parse failure
    raw: str                              # verbatim model output
    error: Optional[str]                  # "ParseError: ..." or "HTTPError: ..." or None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


_CODEFENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_codefence(text: str) -> str:
    m = _CODEFENCE_RE.match(text)
    return m.group(1) if m else text


def parse_funders_json(raw: str) -> tuple[Optional[list[Funder]], Optional[str]]:
    """Parse the model's verbatim string output into a list of Funder objects.

    Returns (funders, error_string). error_string is None on success, else a
    "ParseError: ..." message; in that case funders is None.
    """
    candidate = _strip_codefence(raw).strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        return None, f"ParseError: invalid JSON: {e.msg} at pos {e.pos}"

    if not isinstance(obj, list):
        return None, f"ParseError: expected JSON array, got {type(obj).__name__}"

    try:
        funders = [Funder.model_validate(item) for item in obj]
    except ValidationError as e:
        return None, f"ParseError: schema mismatch: {e.errors(include_url=False)[:1]}"

    return funders, None
