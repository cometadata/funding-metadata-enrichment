import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_extractor.processing.markdown_healer import (
    FailureCategory,
    RecoveryConfidence,
    RestorationOptions,
    WordRecoveryResult,
    categorize_failure,
    heal_markdown,
    read_parquet_input,
    split_concatenated_text,
    validate_input_quality,
    write_output,
)


DEFAULT_OPTIONS = RestorationOptions(repeat_threshold=2.0)


def test_heal_markdown_recovers_word_boundaries():
    raw = "thisisaverysimpletestwithnospace"
    healed = heal_markdown(raw, options=DEFAULT_OPTIONS)
    assert "simple test" in healed
    assert " " in healed


def test_heal_markdown_merges_fragments_and_hyphenation():
    raw = (
        "After\n\n"
        "results\n\n"
        "analyzing\n\n"
        "from Multi-\n"
        "recognition\n"
        "Transformer support\n"
    )

    healed = heal_markdown(raw, options=DEFAULT_OPTIONS)
    assert "After results analyzing" in healed
    assert "Multirecognition" in healed
    assert "Transformer support" in healed


def test_heal_markdown_cleans_vertical_runs():
    raw = (
        "Heading\n\n"
        "8\n"
        "1\n"
        "0\n"
        "2\n"
        "\n"
        "n\n"
        "u\n"
        "J\n\n"
        "Body paragraph about funding."
    )

    healed = heal_markdown(raw, options=DEFAULT_OPTIONS)
    assert "Body paragraph about funding." in healed
    assert "Heading" in healed
    assert "8" not in healed
    assert "J" not in healed


def test_heal_markdown_leaves_regular_paragraphs():
    paragraph = (
        "This work was supported by the National Science Foundation under grant 12345. "
        "Additional support was provided by NASA."
    )
    healed = heal_markdown(paragraph)
    assert healed == paragraph


def test_heal_markdown_reassembles_funding_section():
    raw = (
        "Funding\n\n"
        "The work was\n"
        "000000D730321P5Q0002, Grant No. 70-2021-00145 02.11.2021).\n\n"
        "supported by the Analytical center under\n\n"
        "the RF Government\n\n"
        "(subsidy agreement\n\n"
        "Author contributions statement\n"
        "All authors contributed.\n"
    )

    healed = heal_markdown(raw, options=DEFAULT_OPTIONS)
    assert "Funding" in healed
    assert "The work was 000000D730321P5Q0002" in healed
    assert "Author contributions statement" in healed
    assert "All authors contributed." in healed


def test_heal_markdown_returns_original_when_unrecoverable():
    raw = "1234567890" * 200  # Extremely low whitespace ratio and unlikely to split
    healed = heal_markdown(raw, options=RestorationOptions(skip_tables=True))
    assert healed == raw


def test_validate_input_quality_flags_low_whitespace():
    ok, reason = validate_input_quality("abcdefghij" * 30)
    assert not ok
    assert "whitespace ratio" in reason.lower()


def test_categorize_failure_marks_unrecoverable_after_failed_recovery():
    recovery = WordRecoveryResult(text="bad", confidence=RecoveryConfidence.LOW, changes_made=0, issues=["no split"])
    analysis = categorize_failure("bad", "Very low whitespace ratio (0.0%) - likely malformed extraction", recovery)
    assert analysis.category == FailureCategory.UNRECOVERABLE
    assert not analysis.recoverable
    assert analysis.recovery_attempted
    assert analysis.recovery_confidence == RecoveryConfidence.LOW
    assert any("Word boundary recovery failed" in issue for issue in analysis.issues)


def test_split_concatenated_text_respects_min_confidence_threshold():
    text = "thisisaverysimpletestwithnospace\nregular line here\nanotherconcatenatedsequenceofwords"
    result = split_concatenated_text(text, min_confidence=RecoveryConfidence.HIGH)
    assert result.text == text
    assert result.changes_made == 0
    assert "Confidence too low" in " ".join(result.issues)


def test_read_parquet_input_adds_generated_filenames(tmp_path: Path):
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "input.parquet"
    pd.DataFrame({"content": ["first", "second"]}).to_parquet(parquet_path, index=False)

    df, filename_col, content_col = read_parquet_input(parquet_path)
    assert filename_col == "_generated_filename"
    assert content_col == "content"
    assert list(df[filename_col]) == ["row_0.md", "row_1.md"]
    assert list(df[content_col]) == ["first", "second"]


def test_write_output_uses_original_when_cleaning_empty(tmp_path: Path):
    source = tmp_path / "doc.md"
    destination, success = write_output(
        source,
        base_path=tmp_path,
        content="",
        original_content="original content",
        in_place=False,
        out_dir=None,
        exclude_failed=False,
        failure_category=FailureCategory.UNRECOVERABLE,
    )

    assert not success
    assert destination == tmp_path / "doc-clean.md"
    assert destination.read_text() == "original content"
