from funding_extraction.prompts import build_messages, load_examples, load_prompt


def test_load_prompt():
    prompt = load_prompt()
    assert "funder_name" in prompt
    assert "award_ids" in prompt
    assert len(prompt) > 100


def test_load_examples():
    examples = load_examples()
    assert len(examples) > 0
    for ex in examples:
        assert "input" in ex
        assert "output" in ex


def test_build_messages_structure():
    prompt = "You are an extraction assistant."
    examples = [
        {"input": "Funded by NSF.", "output": '[{"funder_name": "NSF", "awards": []}]'},
    ]
    statement = "Funded by NIH grant R01."
    messages = build_messages(prompt, examples, statement)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompt
    # Few-shot: user then assistant
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Funded by NSF."
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == '[{"funder_name": "NSF", "awards": []}]'
    # Final user message
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == statement


def test_build_messages_no_examples():
    messages = build_messages("System prompt.", [], "Some text.")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Some text."


def test_loaded_examples_are_valid_json_output():
    """Each example output should be valid JSON parseable as list of funder dicts."""
    import json
    examples = load_examples()
    for ex in examples:
        parsed = json.loads(ex["output"])
        assert isinstance(parsed, list)
        for funder in parsed:
            assert "funder_name" in funder
            assert "awards" in funder
