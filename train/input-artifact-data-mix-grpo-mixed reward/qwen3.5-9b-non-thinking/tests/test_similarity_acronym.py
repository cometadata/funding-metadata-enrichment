# tests/test_similarity_acronym.py
"""Acronym/expansion-aware funder matching in `similarity`: an acronym should
match its full expansion at/above the 0.8 threshold without crediting unrelated
organisations."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate_predictions import similarity

THRESHOLD = 0.8

# Acronym ↔ expansion pairs that SHOULD match (same entity).
POSITIVES = [
    ("ERC", "European Research Council"),
    ("NSF", "National Science Foundation"),
    ("GNCS", "Gruppo Nazionale per il Calcolo Scientifico"),
    ("SJTU", "Shanghai Jiao Tong University"),
    ("cstc", "Chongqing Science and Technology Commission"),  # lowercase acronym
    ("NIH", "NIH/NCI"),                                       # token-contained
    ("MNR", "MNR Key Laboratory for Geo-Environmental Monitoring of Great Bay Area"),
]

# Pairs that must NOT match (different organisations / spurious initials).
NEGATIVES = [
    ("NSF", "National Institutes of Health"),
    ("DOE", "Department of Agriculture"),
    ("ANR", "National Science Foundation"),
    ("ARC", "Australian Cancer Foundation"),
]


def test_acronym_positives_match():
    for short, full in POSITIVES:
        assert similarity(short, full) >= THRESHOLD, f"{short!r} should match {full!r}"


def test_acronym_is_symmetric():
    for short, full in POSITIVES:
        assert similarity(short, full) == similarity(full, short), f"asymmetric for {short!r}"


def test_acronym_negatives_reject():
    for short, full in NEGATIVES:
        assert similarity(short, full) < THRESHOLD, f"{short!r} should NOT match {full!r}"


def test_existing_behavior_preserved():
    # exact match
    assert similarity("National Science Foundation", "National Science Foundation") == 1.0
    # parenthetical acronym (pre-existing tier)
    assert similarity("Foo Bar (FB)", "FB") >= THRESHOLD
    # acronym token contained in the longer name (same entity)
    assert similarity("DOE", "DOE (United States)") >= THRESHOLD
    # unrelated long names stay below threshold
    assert similarity("National Science Foundation", "Wellcome Trust") < THRESHOLD
