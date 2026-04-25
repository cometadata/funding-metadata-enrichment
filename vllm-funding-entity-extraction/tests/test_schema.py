import pyarrow as pa
from funding_entity_extractor.schema import (
    EXTRACTION_COLUMNS,
    award_struct_type,
    funder_struct_type,
    extracted_funders_type,
    output_schema_with_extraction,
)


def test_award_struct_has_expected_fields():
    t = award_struct_type()
    assert t == pa.struct([
        pa.field("award_ids", pa.list_(pa.string())),
        pa.field("funding_scheme", pa.list_(pa.string())),
        pa.field("award_title", pa.list_(pa.string())),
    ])


def test_funder_struct_includes_awards_list():
    t = funder_struct_type()
    assert t.field("funder_name").type == pa.string()
    assert t.field("awards").type == pa.list_(award_struct_type())


def test_extracted_funders_is_nested_list_of_funders():
    t = extracted_funders_type()
    assert t == pa.list_(pa.list_(funder_struct_type()))


def test_extraction_columns_names():
    names = [name for name, _ in EXTRACTION_COLUMNS]
    assert names == [
        "extracted_funders",
        "extraction_raw",
        "extraction_error",
        "extraction_latency_ms",
    ]


def test_output_schema_appends_extraction_columns_to_input():
    input_schema = pa.schema([
        pa.field("arxiv_id", pa.string()),
        pa.field("predicted_statements", pa.list_(pa.string())),
    ])
    out = output_schema_with_extraction(input_schema)
    field_names = out.names
    assert field_names[:2] == ["arxiv_id", "predicted_statements"]
    assert field_names[2:] == [
        "extracted_funders",
        "extraction_raw",
        "extraction_error",
        "extraction_latency_ms",
    ]
    assert out.field("extraction_latency_ms").type == pa.list_(pa.float64())
    assert out.field("extraction_error").type == pa.list_(pa.string())
