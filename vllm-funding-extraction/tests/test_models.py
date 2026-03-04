import json

from pydantic import TypeAdapter

from funding_extraction.models import Award, ExtractionResult, FunderEntity


def test_award_defaults():
    award = Award()
    assert award.funding_scheme == []
    assert award.award_ids == []
    assert award.award_title == []


def test_funder_entity_defaults():
    funder = FunderEntity()
    assert funder.funder_name is None
    assert funder.awards == []


def test_extraction_result_with_funders():
    result = ExtractionResult(
        statement="Funded by NSF grant 123.",
        funders=[
            FunderEntity(
                funder_name="NSF",
                awards=[Award(award_ids=["123"])],
            )
        ],
    )
    assert result.statement == "Funded by NSF grant 123."
    assert len(result.funders) == 1
    assert result.funders[0].funder_name == "NSF"
    assert result.funders[0].awards[0].award_ids == ["123"]


def test_extraction_result_empty():
    result = ExtractionResult(statement="No funding.")
    assert result.funders == []


def test_funder_list_roundtrips_through_json():
    """Validate that a list of FunderEntity can be serialized/deserialized,
    which is the format the model will output."""
    funders = [
        FunderEntity(
            funder_name="NSF",
            awards=[
                Award(
                    funding_scheme=["Multistate Research Project"],
                    award_ids=["PEN04623", "1013257"],
                    award_title=[],
                )
            ],
        ),
        FunderEntity(funder_name="NIH", awards=[Award(award_ids=["R21 MH122010"])]),
    ]
    adapter = TypeAdapter(list[FunderEntity])
    json_str = adapter.dump_json(funders).decode()
    parsed = adapter.validate_json(json_str)
    assert len(parsed) == 2
    assert parsed[0].funder_name == "NSF"
    assert parsed[0].awards[0].award_ids == ["PEN04623", "1013257"]
    assert parsed[1].awards[0].award_ids == ["R21 MH122010"]


def test_funder_list_json_schema():
    """Verify we can generate a JSON schema for guided decoding."""
    adapter = TypeAdapter(list[FunderEntity])
    schema = adapter.json_schema()
    assert schema["type"] == "array"
    assert "items" in schema
