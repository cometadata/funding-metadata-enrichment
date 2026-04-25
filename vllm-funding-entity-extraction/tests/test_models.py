import pytest
from funding_entity_extractor.models import (
    Award,
    Funder,
    StatementExtraction,
    parse_funders_json,
)


def test_award_roundtrip():
    a = Award(award_ids=["DMS-1613002"], funding_scheme=[], award_title=[])
    assert a.award_ids == ["DMS-1613002"]
    assert a.funding_scheme == []
    assert a.award_title == []


def test_funder_with_awards():
    f = Funder(
        funder_name="National Science Foundation",
        awards=[Award(award_ids=["NSF-1234"], funding_scheme=[], award_title=[])],
    )
    assert f.funder_name == "NSF" or f.funder_name == "National Science Foundation"
    assert len(f.awards) == 1


def test_parse_funders_json_happy_path():
    raw = '[{"funder_name": "NSF", "awards": [{"award_ids": ["DMS-1"], "funding_scheme": [], "award_title": []}]}]'
    funders, error = parse_funders_json(raw)
    assert error is None
    assert funders is not None
    assert len(funders) == 1
    assert funders[0].funder_name == "NSF"
    assert funders[0].awards[0].award_ids == ["DMS-1"]


def test_parse_funders_json_empty_array():
    funders, error = parse_funders_json("[]")
    assert error is None
    assert funders == []


def test_parse_funders_json_null_funder_name():
    raw = '[{"funder_name": null, "awards": []}]'
    funders, error = parse_funders_json(raw)
    assert error is None
    assert funders[0].funder_name is None


def test_parse_funders_json_invalid_json():
    funders, error = parse_funders_json("not json at all")
    assert funders is None
    assert error is not None
    assert error.startswith("ParseError:")


def test_parse_funders_json_wrong_shape_object_not_array():
    funders, error = parse_funders_json('{"funder_name": "NSF"}')
    assert funders is None
    assert error is not None
    assert error.startswith("ParseError:")


def test_parse_funders_json_extracts_array_from_codefence():
    """Some models wrap JSON in markdown code fences. Be lenient."""
    raw = '```json\n[{"funder_name": "NSF", "awards": []}]\n```'
    funders, error = parse_funders_json(raw)
    assert error is None
    assert funders[0].funder_name == "NSF"


def test_statement_extraction_dataclass_shape():
    e = StatementExtraction(
        funders=[Funder(funder_name="NSF", awards=[])],
        raw='[{"funder_name":"NSF","awards":[]}]',
        error=None,
        latency_ms=42.0,
        prompt_tokens=100,
        completion_tokens=10,
    )
    assert e.funders[0].funder_name == "NSF"
    assert e.error is None
