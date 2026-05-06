"""Prompt constants for the funding-extraction LoRA.

Source: https://huggingface.co/cometadata/funding-extraction-llama-3.1-8b-instruct-artifact-data-mix-grpo-mixed-reward
Any drift here silently degrades extraction quality; tests/test_prompt.py
guards byte-identity.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    'You are an expert at extracting structured funding metadata from academic papers. '
    'Given a funding statement, extract all funders and their associated awards. '
    'Return a JSON array of funder objects. Each funder has:\n'
    '- "funder_name": string or null\n'
    '- "awards": array of objects with "award_ids" (array of strings), '
    '"funding_scheme" (array of strings), and "award_title" (array of strings)\n'
    'Return ONLY the JSON array, no other text.'
)

USER_TEMPLATE = "Extract funding information from the following statement:\n\n{statement}"


def build_messages(statement: str) -> list[dict[str, str]]:
    """Return the chat-completions `messages` list for a single funding statement."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(statement=statement)},
    ]
