# tests/test_reward_shaping.py
"""Tests for RL reward shaping — compute_shaped_reward in funding_reward.py.

Lives alongside test_reward.py but is distinct: test_reward.py guards the
canonical evaluation reward (strict binary scoring), while this file tests
the RL-only reward shaping (soft matching + flat-id term).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate_predictions import Award, Funder
from funding_reward import compute_shaped_reward


def test_flat_term_rewards_right_id_wrong_funder():
    """Wrong funder + right ID should yield only the flat-term bonus."""
    gold = [Funder(funder_name="NSF", awards=[Award(award_ids=["111"])])]
    pred = [Funder(funder_name="Fake Foundation", awards=[Award(award_ids=["111"])])]
    reward, metrics = compute_shaped_reward(
        gold, pred,
        weights=(0.50, 0.40, 0.0, 0.0), flat_id_weight=0.10,
        soft_id_matching=False,
    )
    assert metrics["funder_f0_5"] == 0.0
    assert metrics["award_id_f0_5"] == 0.0  # still gated
    assert metrics["flat_award_id_f0_5"] > 0.95
    assert 0.05 < reward < 0.15, f"Expected flat-bonus-only reward, got {reward}"


def test_flat_term_does_not_dominate_hierarchical():
    """Right funder + right ID must dominate (wrong funder, right ID)."""
    gold = [Funder(funder_name="NSF", awards=[Award(award_ids=["111"])])]
    pred_right = [Funder(funder_name="NSF", awards=[Award(award_ids=["111"])])]
    pred_misassoc = [Funder(funder_name="Fake Foundation", awards=[Award(award_ids=["111"])])]
    w = (0.50, 0.40, 0.0, 0.0)
    r_right, _ = compute_shaped_reward(
        gold, pred_right, weights=w, flat_id_weight=0.10, soft_id_matching=True,
    )
    r_misassoc, _ = compute_shaped_reward(
        gold, pred_misassoc, weights=w, flat_id_weight=0.10, soft_id_matching=True,
    )
    assert r_right > 0.9
    assert r_misassoc < 0.15
    assert r_right > r_misassoc + 0.75


def test_soft_id_exact_match_still_one():
    """Exact IDs score 1.0 in either soft or binary mode."""
    gold = [Funder(funder_name="NIH", awards=[Award(award_ids=["R01NS082338"])])]
    pred = [Funder(funder_name="NIH", awards=[Award(award_ids=["R01NS082338"])])]
    _, m_soft = compute_shaped_reward(gold, pred, soft_id_matching=True)
    _, m_bin = compute_shaped_reward(gold, pred, soft_id_matching=False)
    assert m_soft["award_id_f0_5"] > 0.95
    assert m_bin["award_id_f0_5"] > 0.95


def test_soft_id_near_miss_gets_partial():
    """Edit-distance-1 truncation on a long ID: soft awards 0.5, binary awards 0."""
    gold = [Funder(funder_name="NIH", awards=[Award(award_ids=["R01NS082338"])])]
    pred = [Funder(funder_name="NIH", awards=[Award(award_ids=["R01NS08233"])])]
    _, m_soft = compute_shaped_reward(gold, pred, soft_id_matching=True)
    _, m_bin = compute_shaped_reward(gold, pred, soft_id_matching=False)
    assert 0.4 < m_soft["award_id_f0_5"] < 0.6
    assert m_bin["award_id_f0_5"] == 0.0


def test_soft_id_far_miss_zero():
    """Large edit distance → no credit even with soft on."""
    gold = [Funder(funder_name="NIH", awards=[Award(award_ids=["R01NS082338"])])]
    pred = [Funder(funder_name="NIH", awards=[Award(award_ids=["QWERTY9999"])])]
    _, m_soft = compute_shaped_reward(gold, pred, soft_id_matching=True)
    assert m_soft["award_id_f0_5"] == 0.0


def test_soft_id_short_id_length_guard():
    """Short IDs (below min_length) get no soft credit even at edit distance 1."""
    gold = [Funder(funder_name="NIH", awards=[Award(award_ids=["A1"])])]
    pred = [Funder(funder_name="NIH", awards=[Award(award_ids=["A2"])])]
    _, m_soft = compute_shaped_reward(gold, pred, soft_id_matching=True)
    assert m_soft["award_id_f0_5"] == 0.0


def test_soft_id_adjacent_grant_partial_documents_residual_risk():
    """DOCUMENTS KNOWN TRADE-OFF: edit-distance-1 neighbor IDs (e.g. adjacent
    NIH grant numbers) receive 0.5 credit under default soft config. This is
    intentional to give RL a gradient on near-miss IDs. If reward-hacking
    emerges, disable soft matching and rely on the flat term alone."""
    gold = [Funder(funder_name="NIH", awards=[Award(award_ids=["R01NS082338"])])]
    pred = [Funder(funder_name="NIH", awards=[Award(award_ids=["R01NS082339"])])]
    _, m_soft = compute_shaped_reward(gold, pred, soft_id_matching=True)
    assert 0.4 < m_soft["award_id_f0_5"] < 0.6


def test_funder_metrics_unchanged_under_shaping():
    """Toggling shaping kwargs must not change funder metrics for
    funder-only-varying inputs (regression guard on the funder path)."""
    gold = [Funder(funder_name="National Science Foundation",
                   awards=[Award(award_ids=["ABC-123"])])]
    pred = [Funder(funder_name="NSF",
                   awards=[Award(award_ids=["ABC-123"])])]
    _, m_plain = compute_shaped_reward(gold, pred, flat_id_weight=0.0, soft_id_matching=False)
    _, m_shaped = compute_shaped_reward(gold, pred, flat_id_weight=0.10, soft_id_matching=True)
    assert m_plain["funder_f0_5"] == m_shaped["funder_f0_5"]
    assert m_plain["funder_precision"] == m_shaped["funder_precision"]
    assert m_plain["funder_recall"] == m_shaped["funder_recall"]


def test_shaped_matches_pristine_when_defaults_disable_shaping():
    """compute_shaped_reward with flat_id_weight=0 and soft_id_matching=False
    should agree with the canonical compute_hierarchical_reward up to the
    (funder, id, scheme, title) weights on a normal case."""
    from evaluate_predictions import compute_hierarchical_reward
    gold = [Funder(funder_name="NSF", awards=[Award(award_ids=["ABC-123"])])]
    pred = [Funder(funder_name="NSF", awards=[Award(award_ids=["XYZ-999"])])]
    w = (0.50, 0.50, 0.0, 0.0)
    r_canon, m_canon = compute_hierarchical_reward(gold, pred, weights=w)
    r_shaped, m_shaped = compute_shaped_reward(
        gold, pred, weights=w, flat_id_weight=0.0, soft_id_matching=False,
    )
    assert abs(r_canon - r_shaped) < 1e-9
    assert abs(m_canon["funder_f0_5"] - m_shaped["funder_f0_5"]) < 1e-9
    assert abs(m_canon["award_id_f0_5"] - m_shaped["award_id_f0_5"]) < 1e-9
