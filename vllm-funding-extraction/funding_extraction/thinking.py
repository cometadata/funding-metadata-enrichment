import re

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>.*", re.DOTALL)


def strip_thinking(text: str) -> tuple[str, list[str]]:
    """Strip <think>...</think> tags from text, returning cleaned text and reasoning traces.

    Handles both closed tags and unclosed tags (output truncated mid-think).

    Returns:
        Tuple of (cleaned_text, list_of_reasoning_traces).
    """
    traces: list[str] = []

    def _capture(match: re.Match) -> str:
        content = match.group(0)
        inner = content.removeprefix("<think>").removesuffix("</think>").strip()
        if inner:
            traces.append(inner)
        return ""

    cleaned = _THINK_PATTERN.sub(_capture, text)

    # Handle unclosed <think> tag (truncated output)
    if "<think>" in cleaned:
        unclosed = _UNCLOSED_THINK_PATTERN.search(cleaned)
        if unclosed:
            inner = unclosed.group(0).removeprefix("<think>").strip()
            if inner:
                traces.append(inner)
            cleaned = cleaned[: unclosed.start()]

    return cleaned.strip(), traces
