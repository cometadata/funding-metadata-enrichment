import re
from typing import List, Optional, Tuple

_VERTICAL_TOKEN_RE = re.compile(r'^[A-Za-z0-9\[\]\(\)\.\-,:]+$')
_FRAGMENT_URL_RE = re.compile(r'\b(?:arxiv|http|doi)\b', re.IGNORECASE)
_FRAGMENT_NUMBER_RE = re.compile(r'^\d+[.)]')
_PURE_ALPHA_RE = re.compile(r'^[A-Za-z\s]+$')
_COMMON_STOPWORDS = {
    'the',
    'and',
    'or',
    'of',
    'for',
    'with',
    'by',
    'on',
    'at',
    'from',
    'to',
    'in',
    'this',
    'that',
    'these',
    'those',
    'was',
    'were',
    'is',
    'are',
    'be',
    'been',
    'being',
    'under',
    'over',
    'such',
    'as',
    'per',
    'we',
    'our',
    'their',
    'its',
    'after',
    'before',
    'since',
    'while',
    'because',
    'though',
    'although',
    'when',
    'if',
    'else',
    'than',
    'into',
    'onto',
    'upon',
    'support',
    'supported',
    'supporting'
}

_HEADING_FIRST_WORDS = {
    'references',
    'reference',
    'appendix',
    'acknowledgments',
    'acknowledgements',
    'acknowledgment',
    'acknowledgement',
    'table',
    'figure',
    'fig',
    'algorithm',
    'supplementary',
    'supplement',
    'section',
    'chapter',
    'acknowledgement.',
    'acknowledgments.',
    'acknowledgements.',
}


def heal_markdown(content: str) -> str:
    """Attempt to repair markdown converted from PDFs so funding parsing works."""
    if not content:
        return content

    normalized = (
        content.replace('\r\n', '\n')
        .replace('\r', '\n')
        .replace('\x0c', '\n\n')
    )

    lines = normalized.split('\n')
    lines = _collapse_vertical_runs(lines)
    lines = _collapse_fragmented_runs(lines)
    lines = _trim_blank_runs(lines)
    return '\n'.join(lines).strip('\n')


def _collapse_vertical_runs(lines: List[str]) -> List[str]:
    collapsed: List[str] = []
    buffer: List[Optional[str]] = []

    def flush_buffer():
        nonlocal buffer
        if not buffer:
            return

        token_count = sum(1 for token in buffer if token)
        if token_count >= 6:
            if collapsed and collapsed[-1]:
                collapsed.append('')
        else:
            for token in buffer:
                collapsed.append('' if token is None else token)

        buffer = []

    for line in lines:
        stripped = line.strip()
        if stripped and _looks_like_vertical_token(stripped):
            buffer.append(stripped)
            continue

        if not stripped and buffer:
            buffer.append(None)
            continue

        flush_buffer()
        collapsed.append(line.rstrip())

    flush_buffer()
    return collapsed


def _looks_like_vertical_token(token: str) -> bool:
    return (
        len(token) <= 2
        and ' ' not in token
        and _VERTICAL_TOKEN_RE.match(token) is not None
    )


def _collapse_fragmented_runs(lines: List[str]) -> List[str]:
    healed: List[str] = []
    fragments: List[Optional[str]] = []
    originals: List[str] = []
    indent = ''
    fragment_chars = 0

    def flush_fragments():
        nonlocal fragments, originals, indent, fragment_chars
        if not fragments:
            return

        token_count = sum(1 for token in fragments if token)
        if token_count >= 3:
            tokens = [token for token in fragments if token]
            heading, remainder = _split_leading_heading(tokens)
            if heading:
                healed.append(indent + heading)
                healed.append('')
            if remainder:
                healed.append(indent + _join_fragments(remainder))
        else:
            healed.extend(originals)

        fragments = []
        originals = []
        indent = ''
        fragment_chars = 0

    for line in lines:
        line_value = line.rstrip()
        stripped = line_value.strip()

        if not stripped:
            if fragments:
                originals.append('')
                fragments.append(None)
                fragment_chars += 1
            else:
                healed.append('')
            continue

        if _looks_like_fragment(stripped):
            if not fragments:
                indent = line_value[: len(line_value) - len(line_value.lstrip())]
            fragments.append(stripped)
            originals.append(line_value)
            fragment_chars += len(stripped) + 1
            if fragment_chars > 400:
                flush_fragments()
            continue

        flush_fragments()
        healed.append(line_value)

    flush_fragments()
    return healed


def _looks_like_fragment(text: str) -> bool:
    if not text or len(text) > 200:
        return False

    if _FRAGMENT_URL_RE.search(text):
        return False

    if _FRAGMENT_NUMBER_RE.match(text):
        return False

    if _looks_like_heading(text):
        return False

    if _PURE_ALPHA_RE.match(text):
        if text.islower():
            pass
        elif not _contains_stopword(text):
            return False

    lowered = text.lower()
    if lowered.startswith(('#', '-', '*', '>')):
        return False

    word_count = len(text.split())
    if word_count == 0 or word_count > 20:
        return False

    return True


def _join_fragments(fragments: List[str]) -> str:
    combined: List[str] = []

    for fragment in fragments:
        if not combined:
            combined.append(fragment)
            continue

        prev = combined[-1]
        if prev.endswith('-'):
            combined[-1] = prev[:-1] + fragment.lstrip()
        else:
            combined.append(fragment)

    return ' '.join(combined)


def _trim_blank_runs(lines: List[str]) -> List[str]:
    trimmed: List[str] = []
    blank_run = 0

    for line in lines:
        stripped = line.strip()
        if stripped:
            blank_run = 0
            trimmed.append(line)
        else:
            blank_run += 1
            if blank_run <= 2:
                trimmed.append('')

    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    while trimmed and not trimmed[-1]:
        trimmed.pop()

    return trimmed


def _contains_stopword(text: str) -> bool:
    tokens = text.lower().split()
    return any(token in _COMMON_STOPWORDS for token in tokens)


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    normalized = stripped.strip(':').strip()
    letters = [c for c in normalized if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.8 and len(letters) >= 3:
            return True

    first_word = normalized.split()[0].lower().strip('().')
    if first_word in _HEADING_FIRST_WORDS:
        return True

    if normalized.startswith('[') and re.match(r'^\[\d+[^\]]*\]', normalized):
        return True

    if re.match(r'^\d+(\.\d+)?\s+[A-Z]', normalized):
        return True

    if re.match(r'^\([a-z]\)', normalized, re.IGNORECASE) and len(normalized) <= 8:
        return True

    return False


def _split_leading_heading(tokens: List[str]) -> Tuple[Optional[str], List[str]]:
    if len(tokens) < 2:
        return None, tokens

    first = tokens[0]
    second = tokens[1]

    if not _should_detach_heading(first, second):
        return None, tokens

    return first, tokens[1:]


def _should_detach_heading(first: str, second: str) -> bool:
    if not first or not first[0].isalpha():
        return False

    word_count = len(first.split())
    if word_count == 0 or word_count > 6:
        return False

    if _contains_stopword(first):
        return False

    if _PURE_ALPHA_RE.match(first) is None:
        return False

    second_lower = second.strip().lower()
    if not second_lower:
        return False

    if second_lower[0].isupper() and not _contains_stopword(second):
        return False

    return True
