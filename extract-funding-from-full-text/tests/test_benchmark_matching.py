# tests/test_benchmark_matching.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_extractor.benchmark.matching import (
    normalize_text,
    similarity,
    best_match_score,
    normalize_award_id,
    award_ids_match,
    doi_from_filename,
    greedy_match,
)


def test_normalize_text_fixes_quotes():
    assert '"hello"' in normalize_text('\u201chello\u201d')


def test_normalize_text_collapses_whitespace():
    assert normalize_text("hello   world") == "hello world"


def test_similarity_identical():
    assert similarity("hello world", "hello world") == 1.0


def test_similarity_similar():
    score = similarity("National Science Foundation", "National Science Foundaton")
    assert score > 0.9


def test_similarity_different():
    score = similarity("National Science Foundation", "European Research Council")
    assert score < 0.6


def test_best_match_score_finds_best():
    score = best_match_score("NSF", ["NIH", "NSF", "DOE"])
    assert score == 1.0


def test_best_match_score_empty_candidates():
    assert best_match_score("NSF", []) == 0.0


def test_normalize_award_id_strips_and_uppercases():
    assert normalize_award_id("  r21-mh122010  ") == "R21MH122010"


def test_normalize_award_id_removes_separators():
    assert normalize_award_id("2022-A012-0713456") == "2022A0120713456"


def test_award_ids_match_exact():
    assert award_ids_match("R21 MH122010", "r21 mh122010", mode="exact") is True
    assert award_ids_match("R21 MH122010", "R21-MH122010", mode="exact") is False


def test_award_ids_match_normalized():
    assert award_ids_match("R21 MH122010", "R21-MH122010", mode="normalized") is True
    assert award_ids_match("2022-A0120713456", "2022A0120713456", mode="normalized") is True


def test_doi_from_filename_standard():
    assert doi_from_filename("10.1234-abcd.md") == "10.1234/abcd"


def test_doi_from_filename_arxiv():
    assert doi_from_filename("10.48550-arxiv.2303.07677") == "10.48550/arxiv.2303.07677"


def test_doi_from_filename_no_doi_pattern():
    assert doi_from_filename("random-file.md") is None


def test_greedy_match_consumes_items():
    """Each item can only match once."""
    gold = ["NSF", "NIH"]
    pred = ["NSF", "NIH", "NSF"]
    matches = greedy_match(gold, pred, similarity, threshold=0.8)
    assert len(matches) == 2
    used_gold = {m[0] for m in matches}
    used_pred = {m[1] for m in matches}
    assert len(used_gold) == 2
    assert len(used_pred) == 2


def test_greedy_match_empty():
    matches = greedy_match([], ["a"], similarity, threshold=0.8)
    assert matches == []
