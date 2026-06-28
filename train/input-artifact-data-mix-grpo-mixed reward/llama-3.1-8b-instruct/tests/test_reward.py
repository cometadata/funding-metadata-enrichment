# tests/test_reward.py
"""Tests for compute_hierarchical_reward — the RL reward function."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate_predictions import (
    Award,
    Funder,
    compute_hierarchical_reward,
)


def test_perfect_extraction():
    """Exact match between gold and predicted -> reward ~1.0."""
    gold = [
        Funder(
            funder_name="National Science Foundation",
            awards=[Award(award_ids=["1234567"], funding_scheme=[], award_title=[])],
        )
    ]
    pred = [
        Funder(
            funder_name="National Science Foundation",
            awards=[Award(award_ids=["1234567"], funding_scheme=[], award_title=[])],
        )
    ]
    reward, metrics = compute_hierarchical_reward(gold, pred)
    assert reward > 0.95, f"Perfect extraction should score >0.95, got {reward}"
    assert metrics["funder_f0_5"] > 0.95
    assert metrics["award_id_f0_5"] > 0.95


def test_empty_gold_empty_pred():
    """Both empty -> reward 1.0 (correctly extracted nothing)."""
    reward, metrics = compute_hierarchical_reward([], [])
    assert reward == 1.0, f"Both empty should be 1.0, got {reward}"


def test_empty_gold_nonempty_pred():
    """Gold is empty but model predicted something -> low reward (false positives)."""
    pred = [
        Funder(
            funder_name="Fake Foundation",
            awards=[Award(award_ids=["999"], funding_scheme=[], award_title=[])],
        )
    ]
    reward, metrics = compute_hierarchical_reward([], pred)
    assert reward < 0.3, f"False positives should score low, got {reward}"


def test_nonempty_gold_empty_pred():
    """Gold has funders but model predicted nothing -> reward 0.0."""
    gold = [
        Funder(
            funder_name="NIH",
            awards=[Award(award_ids=["R01"], funding_scheme=[], award_title=[])],
        )
    ]
    reward, metrics = compute_hierarchical_reward(gold, [])
    assert reward == 0.0, f"Missing everything should be 0.0, got {reward}"


def test_correct_funder_wrong_award_id():
    """Right funder, wrong award ID -> funder matched but award_id not."""
    gold = [
        Funder(
            funder_name="NSF",
            awards=[Award(award_ids=["ABC-123"], funding_scheme=[], award_title=[])],
        )
    ]
    pred = [
        Funder(
            funder_name="NSF",
            awards=[Award(award_ids=["XYZ-999"], funding_scheme=[], award_title=[])],
        )
    ]
    reward, metrics = compute_hierarchical_reward(gold, pred)
    assert metrics["funder_f0_5"] > 0.95  # Funder matched
    assert metrics["award_id_f0_5"] == 0.0  # Award ID did not match
    assert 0.2 < reward < 0.6  # Partial credit


def test_wrong_funder_correct_award_id():
    """Wrong funder name -> awards can't match (hierarchical gating)."""
    gold = [
        Funder(
            funder_name="National Science Foundation",
            awards=[Award(award_ids=["123"], funding_scheme=[], award_title=[])],
        )
    ]
    pred = [
        Funder(
            funder_name="Completely Different Organization",
            awards=[Award(award_ids=["123"], funding_scheme=[], award_title=[])],
        )
    ]
    reward, metrics = compute_hierarchical_reward(gold, pred)
    assert metrics["funder_f0_5"] == 0.0  # Funder did not match
    assert metrics["award_id_f0_5"] == 0.0  # Award ID ungated (no funder pair)


def test_association_gap():
    """Verify flat vs hierarchical award_id metrics for association tracking."""
    gold = [
        Funder(funder_name="NSF", awards=[Award(award_ids=["111"], funding_scheme=[], award_title=[])]),
        Funder(funder_name="NIH", awards=[Award(award_ids=["222"], funding_scheme=[], award_title=[])]),
    ]
    # Swap award IDs between funders
    pred = [
        Funder(funder_name="NSF", awards=[Award(award_ids=["222"], funding_scheme=[], award_title=[])]),
        Funder(funder_name="NIH", awards=[Award(award_ids=["111"], funding_scheme=[], award_title=[])]),
    ]
    reward, metrics = compute_hierarchical_reward(gold, pred)
    assert metrics["funder_f0_5"] > 0.95  # Funders match
    assert metrics["award_id_f0_5"] == 0.0  # Hierarchical: IDs under wrong funders
    assert metrics["flat_award_id_f0_5"] > 0.95  # Flat: IDs exist in total pool
    assert metrics["association_gap"] > 0.5  # Large gap = misassignment


def test_multiple_funders_partial_match():
    """Some funders match, some don't -> partial reward."""
    gold = [
        Funder(funder_name="NSF", awards=[Award(award_ids=["111"], funding_scheme=[], award_title=[])]),
        Funder(funder_name="NIH", awards=[Award(award_ids=["222"], funding_scheme=[], award_title=[])]),
    ]
    pred = [
        Funder(funder_name="NSF", awards=[Award(award_ids=["111"], funding_scheme=[], award_title=[])]),
    ]
    reward, metrics = compute_hierarchical_reward(gold, pred)
    assert 0.3 < reward < 0.9  # Partial: got 1 of 2 funders right
    assert metrics["funder_f0_5"] > 0.0
    assert metrics["funder_f0_5"] < 1.0


def test_scheme_and_title_with_nonzero_weights():
    """Schemes and titles contribute to reward when given nonzero weights."""
    gold = [
        Funder(
            funder_name="ERC",
            awards=[
                Award(
                    award_ids=["772408"],
                    funding_scheme=["Starting Grant"],
                    award_title=["Stringlandscape"],
                )
            ],
        )
    ]
    pred_with = [
        Funder(
            funder_name="ERC",
            awards=[
                Award(
                    award_ids=["772408"],
                    funding_scheme=["Starting Grant"],
                    award_title=["Stringlandscape"],
                )
            ],
        )
    ]
    pred_without = [
        Funder(
            funder_name="ERC",
            awards=[Award(award_ids=["772408"], funding_scheme=[], award_title=[])],
        )
    ]
    # With nonzero scheme/title weights, having them should help
    w = (0.35, 0.35, 0.15, 0.15)
    reward_with, _ = compute_hierarchical_reward(gold, pred_with, weights=w)
    reward_without, _ = compute_hierarchical_reward(gold, pred_without, weights=w)
    assert reward_with > reward_without

    # With default weights (0 scheme/title), both should score the same
    reward_with_default, _ = compute_hierarchical_reward(gold, pred_with)
    reward_without_default, _ = compute_hierarchical_reward(gold, pred_without)
    assert reward_with_default == reward_without_default
