import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import ftfy
import mdformat
import pandas as pd
from markdown_it import MarkdownIt


class RecoveryConfidence(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class WordRecoveryResult:
    text: str
    confidence: RecoveryConfidence
    changes_made: int
    issues: List[str]


COMMON_WORDS = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "is",
    "was",
    "are",
    "been",
    "has",
    "had",
    "were",
    "said",
    "did",
    "having",
    "may",
    "should",
    "could",
    "would",
    "paper",
    "using",
    "based",
    "show",
    "method",
    "system",
    "data",
    "model",
    "study",
    "results",
    "used",
    "such",
    "which",
    "between",
    "each",
    "where",
    "both",
    "those",
    "through",
    "during",
    "these",
    "research",
    "analysis",
    "approach",
    "problem",
    "network",
    "algorithm",
    "process",
    "performance",
    "learning",
    "training",
}

TECHNICAL_PATTERNS = [
    r"\b[A-Z]{2,}\b",
    r"\b\d+[A-Za-z]+\b",
    r"\b[A-Za-z]+\d+\b",
    r"\b[a-z]+[A-Z][a-z]+",
]


def build_word_dict() -> set:
    words = set(COMMON_WORDS)
    try:
        with open("/usr/share/dict/words", "r", encoding="utf-8") as fh:
            for line in fh:
                word = line.strip().lower()
                if 2 <= len(word) <= 15 and word.isalpha():
                    words.add(word)
    except FileNotFoundError:
        pass

    return words


WORD_DICT = build_word_dict()


def split_concatenated_text(
    text: str, *, min_confidence: RecoveryConfidence = RecoveryConfidence.MEDIUM
) -> WordRecoveryResult:
    issues: List[str] = []
    changes_made = 0

    words = text.split()
    if len(words) > 10:
        return WordRecoveryResult(
            text=text,
            confidence=RecoveryConfidence.NONE,
            changes_made=0,
            issues=["Text already has word boundaries"],
        )

    lines = text.splitlines()
    recovered_lines: List[str] = []
    total_splits = 0

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 20 or " " in stripped:
            recovered_lines.append(line)
            continue

        split_line, splits = _split_line_into_words(stripped)
        if splits > 0:
            total_splits += splits
            changes_made += 1
            recovered_lines.append(split_line)
        else:
            recovered_lines.append(line)
            issues.append(f"Line {line_num + 1}: Could not confidently split")

    if changes_made == 0:
        confidence = RecoveryConfidence.NONE
    elif changes_made < len(lines) * 0.3:
        confidence = RecoveryConfidence.LOW
        issues.append(f"Only recovered {changes_made}/{len(lines)} lines")
    elif changes_made < len(lines) * 0.7:
        confidence = RecoveryConfidence.MEDIUM
    else:
        confidence = RecoveryConfidence.HIGH

    recovered_text = "\n".join(recovered_lines)

    if confidence.value < min_confidence.value:
        return WordRecoveryResult(
            text=text,
            confidence=confidence,
            changes_made=0,
            issues=issues + ["Confidence too low, returning original"],
        )

    return WordRecoveryResult(
        text=recovered_text,
        confidence=confidence,
        changes_made=changes_made,
        issues=issues,
    )


def _split_line_into_words(line: str) -> Tuple[str, int]:
    protected = []
    working = line
    for pattern in TECHNICAL_PATTERNS:
        for match in re.finditer(pattern, working):
            protected.append((match.start(), match.end(), match.group()))
    n = len(line)
    dp = [float("inf")] * (n + 1)
    dp[0] = 0
    parent = [-1] * (n + 1)
    for i in range(n + 1):
        if dp[i] == float("inf"):
            continue
        for j in range(i + 1, min(i + 16, n + 1)):
            word = line[i:j].lower()
            if word in WORD_DICT:
                cost = 1
            elif len(word) <= 2:
                cost = 100
            elif len(word) == 3:
                cost = 10
            else:
                cost = 5

            if dp[i] + cost < dp[j]:
                dp[j] = dp[i] + cost
                parent[j] = i

    if dp[n] == float("inf"):
        return line, 0

    words: List[str] = []
    pos = n
    while pos > 0:
        prev = parent[pos]
        words.append(line[prev:pos])
        pos = prev

    words.reverse()

    if len(words) < 5:
        return line, 0

    split_line = " ".join(words)

    known_words = sum(1 for w in words if w.lower() in WORD_DICT)
    confidence_ratio = known_words / len(words) if words else 0

    if confidence_ratio < 0.4:
        return line, 0

    return split_line, len(words) - 1


class FailureCategory(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial"
    NO_WORD_BOUNDARIES = "no_spaces"
    MALFORMED_STRUCTURE = "malformed"
    ENCODING_ISSUES = "encoding"
    NEEDS_REVIEW = "review"
    UNRECOVERABLE = "unrecoverable"


@dataclass
class FailureAnalysis:
    category: FailureCategory
    issues: List[str]
    recoverable: bool
    recovery_attempted: bool
    recovery_confidence: RecoveryConfidence | None = None


def categorize_failure(
    original_text: str,
    validation_error: str | None,
    recovery_result: WordRecoveryResult | None = None,
) -> FailureAnalysis:
    issues: List[str] = []
    category = FailureCategory.SUCCESS

    if not validation_error:
        return FailureAnalysis(
            category=FailureCategory.SUCCESS,
            issues=[],
            recoverable=True,
            recovery_attempted=False,
        )

    if "whitespace ratio" in validation_error.lower() or "extremely long lines" in validation_error.lower():
        category = FailureCategory.NO_WORD_BOUNDARIES
        issues.append("Text lacks word boundaries (no spaces)")

        if recovery_result:
            if recovery_result.confidence in (RecoveryConfidence.MEDIUM, RecoveryConfidence.HIGH):
                category = FailureCategory.PARTIAL_SUCCESS
                issues.append(f"Partial recovery successful ({recovery_result.changes_made} lines)")
                return FailureAnalysis(
                    category=category,
                    issues=issues,
                    recoverable=True,
                    recovery_attempted=True,
                    recovery_confidence=recovery_result.confidence,
                )
            issues.append("Word boundary recovery failed")
            return FailureAnalysis(
                category=FailureCategory.UNRECOVERABLE,
                issues=issues,
                recoverable=False,
                recovery_attempted=True,
                recovery_confidence=recovery_result.confidence,
            )

    elif "encoding" in validation_error.lower():
        category = FailureCategory.ENCODING_ISSUES
        issues.append("Text encoding problems detected")

    elif "structure" in validation_error.lower() or "malformed" in validation_error.lower():
        category = FailureCategory.MALFORMED_STRUCTURE
        issues.append("Document structure is malformed")

    else:
        category = FailureCategory.NEEDS_REVIEW
        issues.append(f"Unknown issue: {validation_error}")

    recoverable = category in (
        FailureCategory.PARTIAL_SUCCESS,
        FailureCategory.ENCODING_ISSUES,
        FailureCategory.NEEDS_REVIEW,
    )

    return FailureAnalysis(
        category=category,
        issues=issues,
        recoverable=recoverable,
        recovery_attempted=recovery_result is not None,
        recovery_confidence=recovery_result.confidence if recovery_result else None,
    )


HEADER_FOOTER_PATTERNS = [
    r"(?m)^\s*\d+\s*$",
    r"(?m)^\s*(?:Page|Pg\.?)\s*\d+(?:\s*of\s*\d+)?\s*$",
    r"(?m)^.*?\|\s*\d+\s*$",
]

BULLET_PATTERN = re.compile(r"(?m)^(\s*)([•●◦▪▫·]+)(\s*)")
LIST_MARKER_PATTERN = re.compile(r"^(\d+\.|[A-Za-z]\.)\s")
UNORDERED_MARKER_PATTERN = re.compile(r"^[-*+]\s")


@dataclass
class RestorationOptions:
    page_length_estimate: int = 50
    repeat_threshold: float = 0.8
    max_header_words: int = 12
    skip_tables: bool = False
    strip_citations: bool = False


@dataclass
class RestorationResult:
    source: Path
    destination: Path | None
    warnings: List[str]
    success: bool
    original_size: int
    output_size: int
    validation_error: str | None = None
    failure_analysis: FailureAnalysis | None = None
    recovery_attempted: bool = False
    cleaned_content: str | None = None
    original_content: str | None = None


def strip_repeated_lines(
    text: str, *, page_length_estimate: int, threshold: float, max_words: int
) -> Tuple[str, List[str]]:
    lines = text.splitlines()
    normalized = [ln.strip() for ln in lines]
    freq = Counter(ln for ln in normalized if ln and len(ln) >= 4 and len(ln.split()) <= max_words)
    estimated_pages = max(1, len(lines) / float(page_length_estimate))
    to_remove = {ln for ln, count in freq.items() if count >= estimated_pages * threshold}
    if not to_remove:
        return text, []
    cleaned_lines = [original for original, norm in zip(lines, normalized) if norm not in to_remove]
    return "\n".join(cleaned_lines), sorted(to_remove)


def remove_footer_patterns(text: str) -> str:
    cleaned = text
    for pattern in HEADER_FOOTER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned


def remove_citations_and_noise(text: str, *, strip_citations: bool) -> str:
    cleaned = re.sub(r"\(cid:\d+\)", "", text)
    if strip_citations:
        cleaned = re.sub(r"\[\d+\]", "", cleaned)
    cleaned = re.sub(r"(?m)^[A-Za-z0-9+/=]{40,}\s*$", "", cleaned)
    return cleaned


def drop_short_line_runs(text: str, *, max_len: int = 3, min_run: int = 5) -> Tuple[str, List[str]]:
    lines = text.splitlines()
    kept: List[str] = []
    removed_spans: List[str] = []
    i = 0
    while i < len(lines):
        start = i
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or len(stripped) <= max_len:
                i += 1
                continue
            break
        run_len = i - start
        if run_len >= min_run:
            removed_spans.append(f"lines {start + 1}-{i}")
        else:
            kept.extend(lines[start:i])
        if i < len(lines):
            kept.append(lines[i])
            i += 1
    return "\n".join(kept), removed_spans


def trim_leading_noise(text: str) -> Tuple[str, bool]:
    lines = text.splitlines()
    cut_index = 0
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if re.search(r"[A-Za-z]{4}", line) or line.lstrip().startswith("#"):
            cut_index = idx
            break
    if cut_index == 0:
        return text, False
    return "\n".join(lines[cut_index:]), True


def merge_short_fragments(text: str, *, max_len: int = 12, min_tokens: int = 6) -> Tuple[str, bool]:
    lines = text.splitlines()
    merged: List[str] = []
    changed = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped and len(stripped) <= max_len:
            tokens: List[str] = []
            start = i
            while i < len(lines):
                candidate = lines[i].strip()
                if candidate and len(candidate) <= max_len:
                    tokens.append(candidate)
                    i += 1
                    continue
                if not candidate:
                    i += 1
                    continue
                break
            if len(tokens) >= min_tokens:
                merged.append(" ".join(tokens))
                changed = True
            else:
                merged.extend(lines[start:i])
            continue
        merged.append(lines[i])
        i += 1
    return "\n".join(merged), changed


def remove_blank_between_short_lines(text: str, *, max_len: int = 24) -> str:
    lines = text.splitlines()
    keep: List[str] = []
    for idx, line in enumerate(lines):
        if line.strip():
            keep.append(line)
            continue
        prev_line = lines[idx - 1].strip() if idx > 0 else ""
        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        if prev_line and next_line and len(prev_line) <= max_len and len(next_line) <= max_len:
            continue
        keep.append(line)
    return "\n".join(keep)


def normalize_bullets(text: str) -> str:
    return BULLET_PATTERN.sub(r"\1- ", text)


def fix_hyphenation(text: str) -> str:
    pattern = re.compile(r"([A-Za-z]{2,})-\s*[\r\n]+\s*([a-z][\w-]*)")
    return pattern.sub(r"\1\2", text)


def is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False

    common_headers = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "methodology",
        "method",
        "methods",
        "approach",
        "implementation",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "acknowledgments",
        "acknowledgements",
        "appendix",
        "keywords",
        "author",
        "authors",
        "affiliation",
        "affiliations",
    }

    lower = stripped.lower()
    if lower in common_headers:
        return True

    if re.match(r"^\d+\.(\d+\.?)?\s+[A-Z]", stripped):
        return True

    if stripped.isupper() and len(stripped) < 50:
        return True

    words = stripped.split()
    if len(words) <= 6:
        capitalized = sum(1 for w in words if w and w[0].isupper())
        if capitalized / len(words) > 0.6:
            return True

    return False


def is_complete_sentence(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if stripped[-1] in ".!?":
        if stripped[-1] == "." and len(stripped) < 10:
            return False
        return True

    return False


def should_preserve_blank_line(prev_line: str, next_line: str) -> bool:
    if not prev_line or not next_line:
        return True

    prev_stripped = prev_line.strip()
    next_stripped = next_line.strip()

    if is_section_header(prev_stripped) or is_section_header(next_stripped):
        return True

    if len(prev_stripped) < 50:
        return True

    if is_complete_sentence(prev_stripped) and len(prev_stripped) > 40:
        return True

    if is_complete_sentence(prev_stripped) and next_stripped and next_stripped[0].isupper():
        return True

    return False


def merge_broken_lines(text: str) -> str:
    lines = text.splitlines()
    merged: List[str] = []
    buffer = ""
    in_code = False
    blank_count = 0
    prev_line = ""

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("```"):
            if buffer:
                merged.append(buffer)
                buffer = ""
                prev_line = buffer
            for _ in range(min(blank_count, 2)):
                merged.append("")
            blank_count = 0
            merged.append(stripped)
            prev_line = stripped
            in_code = not in_code
            continue

        if in_code:
            merged.append(line.rstrip())
            prev_line = line.rstrip()
            continue

        if not stripped:
            blank_count += 1
            continue

        is_list_item = bool(LIST_MARKER_PATTERN.match(stripped) or UNORDERED_MARKER_PATTERN.match(stripped))
        starts_new_block = is_list_item or stripped.startswith(">") or stripped.startswith("#")

        preserve_blank = False
        if blank_count >= 2:
            preserve_blank = True
        elif blank_count == 1:
            if starts_new_block:
                preserve_blank = True
            elif buffer and should_preserve_blank_line(buffer, stripped):
                preserve_blank = True

        if preserve_blank:
            if buffer:
                merged.append(buffer)
                prev_line = buffer
                buffer = ""
            merged.append("")

        blank_count = 0

        if starts_new_block:
            if buffer:
                merged.append(buffer)
                prev_line = buffer
                buffer = ""
            merged.append(stripped)
            prev_line = stripped
            continue

        if buffer:
            if buffer.endswith("-"):
                buffer = buffer[:-1] + stripped
            else:
                buffer = f"{buffer} {stripped}"
        else:
            buffer = stripped

    if buffer:
        merged.append(buffer)

    return "\n".join(merged)


def collapse_blank_lines(text: str) -> str:
    lines = text.splitlines()
    result: List[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            result.append(line.rstrip())
        else:
            blank_run += 1
            if blank_run <= 2:
                result.append("")
    return "\n".join(result)


def _is_table_like_line(line: str) -> bool:
    has_pipes = line.count("|") >= 2
    has_columns = bool(re.search(r"\S\s{2,}\S", line))
    return has_pipes or has_columns


def repair_table_block(block: str) -> str:
    original = block
    for sep in ("|", r"\s{2,}"):
        try:
            df = pd.read_csv(
                StringIO(block),
                sep=sep,
                engine="python",
                header=None,
                dtype=str,
                keep_default_na=False,
            )
            df = df.map(lambda val: val.strip() if isinstance(val, str) else val)
            df = df.dropna(axis=1, how="all")
            df = df.loc[:, ~(df.apply(lambda col: "".join(col).strip() == "", axis=0))]
            df = df.loc[~(df.apply(lambda row: "".join(row).strip(), axis=1) == "")]
            if df.empty:
                continue

            if len(df) > 1:
                header = df.iloc[0].tolist()
                data = df.iloc[1:].reset_index(drop=True)
                data.columns = [col if col else f"col_{i}" for i, col in enumerate(header)]
                df = data
            else:
                df.columns = [f"col_{i}" for i in range(df.shape[1])]

            return df.to_markdown(index=False)
        except Exception:
            continue
    return original


def repair_tables(text: str) -> Tuple[str, List[str]]:
    lines = text.splitlines()
    output: List[str] = []
    warnings: List[str] = []
    i = 0
    while i < len(lines):
        if _is_table_like_line(lines[i]):
            start = i
            while i < len(lines) and _is_table_like_line(lines[i]):
                i += 1
            block_lines = lines[start:i]
            if len(block_lines) < 2:
                output.extend(block_lines)
                continue
            cleaned = repair_table_block("\n".join(block_lines))
            if cleaned != "\n".join(block_lines):
                warnings.append(f"Rebuilt table at lines {start + 1}-{i}")
            output.extend(cleaned.splitlines())
        else:
            output.append(lines[i])
            i += 1
    return "\n".join(output), warnings


def format_markdown(text: str) -> str:
    if mdformat is None:
        return text
    try:
        return mdformat.text(text)
    except Exception:
        return text


def validate_markdown(text: str) -> List[str]:
    warnings: List[str] = []
    if text.count("```") % 2 != 0:
        warnings.append("Detected unbalanced code fences")
    try:
        MarkdownIt().parse(text)
    except Exception as exc:
        warnings.append(f"markdown-it parse issue: {exc}")
    return warnings


def restore_markdown(text: str, options: RestorationOptions) -> Tuple[str, List[str]]:
    warnings: List[str] = []

    text = ftfy.fix_text(text)
    text, trimmed = trim_leading_noise(text)
    if trimmed:
        warnings.append("Trimmed leading noise before first paragraph")
    text, removed_lines = strip_repeated_lines(
        text,
        page_length_estimate=options.page_length_estimate,
        threshold=options.repeat_threshold,
        max_words=options.max_header_words,
    )
    if removed_lines:
        warnings.append(f"Removed repeated lines: {', '.join(removed_lines[:5])}")

    text = remove_footer_patterns(text)
    text, short_runs = drop_short_line_runs(text)
    if short_runs:
        warnings.append(f"Dropped short-line runs: {', '.join(short_runs[:5])}")
    text = remove_citations_and_noise(text, strip_citations=options.strip_citations)
    text = remove_blank_between_short_lines(text)
    text, merged_fragments = merge_short_fragments(text)
    if merged_fragments:
        warnings.append("Merged clusters of short fragments")
    text = normalize_bullets(text)
    text = fix_hyphenation(text)
    text = merge_broken_lines(text)
    text = collapse_blank_lines(text)

    if not options.skip_tables:
        text, table_warnings = repair_tables(text)
        warnings.extend(table_warnings)

    text = format_markdown(text)
    warnings.extend(validate_markdown(text))

    return text, warnings


def validate_input_quality(text: str) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "Empty file"

    lines = text.splitlines()
    if not lines:
        return False, "No content"

    max_reasonable_line = 1000
    long_lines = [i for i, ln in enumerate(lines) if len(ln) > max_reasonable_line]

    if long_lines and len(long_lines) / len(lines) > 0.20:
        longest = max(len(ln) for ln in lines)
        return False, f"Contains {len(long_lines)} extremely long lines ({len(long_lines)/len(lines):.1%} of file, max: {longest} chars)"

    total_chars = len(text)
    whitespace_chars = sum(1 for c in text if c.isspace())
    whitespace_ratio = whitespace_chars / total_chars if total_chars > 0 else 0

    if whitespace_ratio < 0.05:
        return False, f"Very low whitespace ratio ({whitespace_ratio:.1%}) - likely malformed extraction"

    avg_line_length = total_chars / len(lines) if lines else 0
    if avg_line_length > 1000:
        return False, f"Unusually long average line length ({avg_line_length:.0f} chars)"

    return True, "OK"


def detect_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    columns_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]
    return None


def read_parquet_input(path: Path) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_parquet(path)

    filename_candidates = ["file_name", "filename", "relative_path", "name", "path"]
    filename_col = detect_column(df, filename_candidates)

    content_candidates = ["content", "text", "markdown", "md", "body"]
    content_col = detect_column(df, content_candidates)

    if content_col is None:
        raise ValueError(f"Could not detect content column. Available columns: {list(df.columns)}")

    if filename_col is None:
        df["_generated_filename"] = [f"row_{i}.md" for i in range(len(df))]
        filename_col = "_generated_filename"

    return df, filename_col, content_col


def write_parquet_output(results: List[RestorationResult], output_path: Path) -> None:
    records = []
    for result in results:
        if result.success:
            cleaned_content = result.cleaned_content
            if cleaned_content is None and result.destination and result.destination.exists():
                cleaned_content = result.destination.read_text(encoding="utf-8", errors="ignore")
            if cleaned_content is None:
                cleaned_content = ""

            records.append(
                {
                    "file_name": result.source.name,
                    "content": cleaned_content,
                    "warnings": "; ".join(result.warnings) if result.warnings else "",
                    "success": result.success,
                    "original_size": result.original_size,
                    "output_size": result.output_size,
                }
            )

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)


def write_failed_parquet(results: List[RestorationResult], output_path: Path) -> None:
    records = []
    for result in results:
        if result.failure_analysis and result.failure_analysis.category != FailureCategory.SUCCESS:
            original_content = result.original_content
            if original_content is None and result.source.exists():
                original_content = result.source.read_text(encoding="utf-8", errors="ignore")
            if original_content is None:
                original_content = ""

            cleaned_content = result.cleaned_content
            if cleaned_content is None and result.destination and result.destination.exists():
                cleaned_content = result.destination.read_text(encoding="utf-8", errors="ignore")
            if cleaned_content is None:
                cleaned_content = ""

            records.append(
                {
                    "file_name": result.source.name,
                    "original_content": original_content,
                    "cleaned_content": cleaned_content,
                    "failure_category": result.failure_analysis.category.value,
                    "issues": "; ".join(result.failure_analysis.issues) if result.failure_analysis.issues else "",
                    "recovery_attempted": result.recovery_attempted,
                    "recovery_confidence": result.failure_analysis.recovery_confidence.name if result.failure_analysis.recovery_confidence else "",
                    "recoverable": result.failure_analysis.recoverable,
                    "validation_error": result.validation_error or "",
                }
            )

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)


def gather_markdown_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.md") if p.is_file() and not any(part.startswith(".") for part in p.parts))


def gather_parquet_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.parquet") if p.is_file() and not any(part.startswith(".") for part in p.parts))


def is_parquet_input(path: Path) -> bool:
    if path.is_file() and path.suffix.lower() == ".parquet":
        return True
    if path.is_dir():
        return len(gather_parquet_paths(path)) > 0
    return False


def write_output(
    source: Path,
    base_path: Path,
    content: str,
    original_content: str,
    *,
    in_place: bool,
    out_dir: Path | None,
    exclude_failed: bool,
    failure_category: FailureCategory | None = None,
) -> Tuple[Path | None, bool]:
    is_empty = not content or not content.strip()

    if in_place:
        destination = source
    elif out_dir:
        if base_path.is_file():
            relative = Path(source.name)
        else:
            try:
                relative = source.relative_to(base_path)
            except ValueError:
                relative = Path(source.name)

        if failure_category and failure_category != FailureCategory.SUCCESS:
            if failure_category == FailureCategory.UNRECOVERABLE:
                destination = out_dir / "failed" / "unrecoverable" / relative
            elif failure_category == FailureCategory.PARTIAL_SUCCESS:
                destination = out_dir / "failed" / "partial" / relative
            elif failure_category == FailureCategory.NEEDS_REVIEW:
                destination = out_dir / "failed" / "needs_review" / relative
            else:
                destination = out_dir / "failed" / failure_category.value / relative
        else:
            destination = out_dir / relative
    else:
        destination = source.with_name(f"{source.stem}-clean.md")

    if is_empty:
        if exclude_failed:
            return None, False
        content = original_content

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination, not is_empty


def heal_markdown(content: str, *, options: RestorationOptions | None = None) -> str:
    if not content:
        return content

    opts = options or RestorationOptions()
    is_valid, validation_reason = validate_input_quality(content)

    recovery_result: WordRecoveryResult | None = None
    text_to_process = content

    if not is_valid and ("whitespace ratio" in validation_reason.lower() or "extremely long lines" in validation_reason.lower()):
        recovery_result = split_concatenated_text(content, min_confidence=RecoveryConfidence.LOW)
        if recovery_result.confidence in (RecoveryConfidence.MEDIUM, RecoveryConfidence.HIGH):
            text_to_process = recovery_result.text
            is_valid, validation_reason = validate_input_quality(text_to_process)

    failure_analysis = categorize_failure(content, None if is_valid else validation_reason, recovery_result)

    if not is_valid and not failure_analysis.recoverable:
        return content

    cleaned, _warnings = restore_markdown(text_to_process, opts)
    return cleaned if cleaned and cleaned.strip() else content


__all__ = [
    "heal_markdown",
    "RestorationOptions",
    "RestorationResult",
    "restore_markdown",
    "validate_input_quality",
    "split_concatenated_text",
    "RecoveryConfidence",
    "categorize_failure",
    "detect_column",
    "read_parquet_input",
    "write_parquet_output",
    "write_failed_parquet",
    "gather_markdown_paths",
    "gather_parquet_paths",
    "is_parquet_input",
    "write_output",
]
