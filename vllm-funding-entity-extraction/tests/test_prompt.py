"""Byte-identity tests for the LoRA's training prompts.

If the LoRA card changes these strings, this test fails and forces a deliberate
update — silent drift would degrade extraction quality without warning.
"""

from funding_entity_extractor.prompt import SYSTEM_PROMPT, USER_TEMPLATE, build_messages


EXPECTED_SYSTEM_PROMPT = (
    'You are an expert at extracting structured funding metadata from academic papers. '
    'Given a funding statement, extract all funders and their associated awards. '
    'Return a JSON array of funder objects. Each funder has:\n'
    '- "funder_name": string or null\n'
    '- "awards": array of objects with "award_ids" (array of strings), '
    '"funding_scheme" (array of strings), and "award_title" (array of strings)\n'
    'Return ONLY the JSON array, no other text.'
)


def test_system_prompt_byte_identical_to_model_card():
    assert SYSTEM_PROMPT == EXPECTED_SYSTEM_PROMPT


def test_user_template_renders_statement():
    statement = "This work was supported by NSF grant DMS-1613002."
    rendered = USER_TEMPLATE.format(statement=statement)
    assert rendered == (
        "Extract funding information from the following statement:\n\n"
        "This work was supported by NSF grant DMS-1613002."
    )


def test_build_messages_returns_chat_format():
    msgs = build_messages("hello statement")
    assert msgs == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Extract funding information from the following statement:\n\nhello statement"},
    ]
