# tests/test_funding_reward.py
"""Tests for funding_reward.py — JSON parsing and env helpers."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from funding_reward import extract_funders_from_text


def test_extract_raw_json_list():
    """Model outputs raw JSON list of funders."""
    text = '[{"funder_name": "NSF", "awards": [{"award_ids": ["123"], "funding_scheme": [], "award_title": []}]}]'
    funders = extract_funders_from_text(text)
    assert len(funders) == 1
    assert funders[0].funder_name == "NSF"


def test_extract_json_with_funders_key():
    """Model outputs {"funders": [...]} wrapper."""
    text = '{"funders": [{"funder_name": "NIH", "awards": [{"award_ids": ["R01"], "funding_scheme": [], "award_title": []}]}]}'
    funders = extract_funders_from_text(text)
    assert len(funders) == 1
    assert funders[0].funder_name == "NIH"


def test_extract_json_from_markdown_code_block():
    """Model wraps JSON in ```json ... ``` markers."""
    text = '```json\n[{"funder_name": "ERC", "awards": []}]\n```'
    funders = extract_funders_from_text(text)
    assert len(funders) == 1
    assert funders[0].funder_name == "ERC"


def test_extract_returns_empty_on_garbage():
    """Unparseable text → empty list (not an exception)."""
    funders = extract_funders_from_text("This is not JSON at all.")
    assert funders == []


def test_extract_returns_empty_on_empty_string():
    funders = extract_funders_from_text("")
    assert funders == []


def test_extract_single_funder_dict():
    """Model outputs a single funder dict (not wrapped in list)."""
    text = '{"funder_name": "JSPS", "awards": [{"award_ids": [], "funding_scheme": ["KAKENHI"], "award_title": []}]}'
    funders = extract_funders_from_text(text)
    assert len(funders) == 1
    assert funders[0].funder_name == "JSPS"
