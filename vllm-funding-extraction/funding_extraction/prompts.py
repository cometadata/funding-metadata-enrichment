import json
from importlib.resources import files


def load_prompt(prompt_file: str | None = None) -> str:
    """Load the extraction system prompt from a file or the bundled default."""
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    resource = files("funding_extraction.configs.prompts").joinpath("extraction_prompt.txt")
    return resource.read_text(encoding="utf-8").strip()


def load_examples(examples_file: str | None = None) -> list[dict[str, str]]:
    """Load few-shot examples from a file or the bundled default.

    Returns a list of dicts with "input" and "output" keys.
    """
    if examples_file:
        with open(examples_file, "r", encoding="utf-8") as f:
            return json.load(f)
    resource = files("funding_extraction.configs.prompts").joinpath("extraction_examples.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def build_messages(
    prompt: str,
    examples: list[dict[str, str]],
    statement: str,
) -> list[dict[str, str]]:
    """Build chat messages for the extraction request.

    Args:
        prompt: The system prompt text.
        examples: List of {"input": ..., "output": ...} few-shot examples.
        statement: The funding statement to extract from.

    Returns:
        List of chat message dicts with "role" and "content" keys.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": prompt}]
    for example in examples:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})
    messages.append({"role": "user", "content": statement})
    return messages
