"""Hand-computed tests for the metrics used in the ERA reference study."""

import math

import numpy as np
import pytest

from era.reference_metrics import (
    B,
    B_T,
    B_alpha,
    B_k,
    G_l,
    delta_si,
    depth_centroid,
    group_gap,
    lin_k,
    probabilities,
    restricted_B,
    stereotype_index,
    top_k_ids,
    top_p_ids,
)


def test_lin_k_matches_the_formula():
    p, q = np.array([0.6, 0.4]), np.array([0.5, 0.5])
    expected = 0.6 * math.log(0.6 / 0.55) + 0.4 * math.log(0.4 / 0.45)
    assert lin_k(p, q) == pytest.approx(expected)
    assert B(p, q) == pytest.approx(expected)


def test_lin_k_is_finite_at_zero_and_has_the_correct_bound():
    assert lin_k([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.log(2))
    assert lin_k([1.0, 0.0], [0.0, 1.0], log_base=2) == pytest.approx(1.0)


def test_invalid_distributions_fail_instead_of_receiving_epsilon():
    with pytest.raises(ValueError, match="zero total mass"):
        probabilities([0.0, 0.0])
    with pytest.raises(ValueError, match="negative"):
        probabilities([1.1, -0.1])
    with pytest.raises(ValueError, match="NaN"):
        probabilities([1.0, float("nan")])


def test_top_k_and_top_p_have_deterministic_ties():
    dist = [0.2, 0.4, 0.4]
    assert top_k_ids(dist, 2) == (1, 2)
    assert top_p_ids(dist, 0.4) == (1,)
    assert top_p_ids(dist, 0.41) == (1, 2)


def test_top_k_union_uses_true_probabilities_from_both_models():
    p = [0.6, 0.3, 0.1]
    q = [0.1, 0.2, 0.7]
    result = B_k(p, q, 1)
    assert result["support"] == (0, 2)
    assert result["mass_m0"] == pytest.approx(0.7)
    assert result["mass_m1"] == pytest.approx(0.8)

    # The same exact union supplied directly must give the same answer.
    direct = restricted_B(p, q, (0,), (2,))
    assert result["value"] == pytest.approx(direct["value"])


def test_top_p_records_achieved_mass_not_only_requested_alpha():
    result = B_alpha([0.7, 0.2, 0.1], [0.1, 0.2, 0.7], 0.7)
    assert result["support"] == (0, 2)
    assert result["mass_m0"] == pytest.approx(0.8)
    assert result["mass_m1"] == pytest.approx(0.8)


GROUPS = {"male": (0, 1), "female": (2, 3)}


def test_B_T_between_only():
    p = [0.45, 0.45, 0.05, 0.05]
    q = [0.05, 0.05, 0.45, 0.45]
    result = B_T(p, q, GROUPS, log_base=2)
    expected = 0.9 * math.log2(0.9 / 0.5) + 0.1 * math.log2(0.1 / 0.5)
    assert result["total"] == pytest.approx(expected)
    assert result["between"] == pytest.approx(expected)
    assert result["within"] == pytest.approx(0.0, abs=1e-15)
    assert result["between_fraction"] == pytest.approx(1.0)


def test_B_T_within_only():
    p = [0.45, 0.05, 0.45, 0.05]
    q = [0.05, 0.45, 0.05, 0.45]
    result = B_T(p, q, GROUPS, log_base=2)
    expected = 0.9 * math.log2(0.9 / 0.5) + 0.1 * math.log2(0.1 / 0.5)
    assert result["total"] == pytest.approx(expected)
    assert result["between"] == pytest.approx(0.0, abs=1e-15)
    assert result["within"] == pytest.approx(expected)
    assert result["within_fraction"] == pytest.approx(1.0)


def test_B_T_chain_rule_and_undefined_zero_ratio():
    p = [0.52, 0.18, 0.21, 0.09]
    q = [0.17, 0.23, 0.14, 0.46]
    result = B_T(p, q, GROUPS)
    assert result["total"] == pytest.approx(result["between"] + result["within"])

    identical = B_T(p, p, GROUPS)
    assert identical["total"] == pytest.approx(0.0)
    assert identical["between_fraction"] is None
    assert identical["within_fraction"] is None


def test_B_T_rejects_overlapping_groups_and_zero_target_mass():
    with pytest.raises(ValueError, match="overlap"):
        B_T([1.0, 0.0], [1.0, 0.0], {"a": (0,), "b": (0,)})
    with pytest.raises(ValueError, match="target set T"):
        B_T([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], {"a": (0,), "b": (1,)})


def test_group_gap_raw_and_conditional_are_explicitly_distinct():
    dist = [0.12, 0.03, 0.85]
    assert group_gap(dist, (0,), (1,), conditional=False) == pytest.approx(0.09)
    assert group_gap(dist, (0,), (1,), conditional=True) == pytest.approx(0.6)


def test_SI_and_delta_SI_match_a_hand_example():
    families = ("leadership", "leadership", "support", "support")
    base = stereotype_index((0.8, 0.6, 0.2, 0.0), families)
    tuned = stereotype_index((0.9, 0.7, 0.1, -0.1), families)
    change = delta_si(base, tuned)
    assert base["SI"] == pytest.approx(0.6)
    assert tuned["SI"] == pytest.approx(0.8)
    assert change["delta_SI"] == pytest.approx(0.2)
    assert change["delta_SI"] == pytest.approx(
        change["delta_leadership"] - change["delta_support"]
    )


def test_G_l_matches_a_two_layer_example():
    base_context = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    tuned_context = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    result = G_l(
        np.stack((base_context, base_context)), np.stack((tuned_context, tuned_context))
    )
    assert result["similarity"] == pytest.approx((1.0, 0.0))
    assert result["drift"] == pytest.approx((0.0, 1.0))
    assert result["normalised_depth_centroid"] == pytest.approx(1.0)


def test_G_l_preserves_layers_with_different_hidden_widths():
    base = (
        np.array([[[1.0, 0.0], [0.0, 1.0]]] * 2),
        np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2),
    )
    tuned = (
        base[0].copy(),
        np.array([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]] * 2),
    )
    result = G_l(base, tuned)
    assert result["similarity"] == pytest.approx((1.0, 0.0))
    assert result["drift"] == pytest.approx((0.0, 1.0))


def test_G_l_rejects_zero_centroids_and_zero_curve_has_no_centroid():
    with pytest.raises(ValueError, match="zero centroid"):
        G_l(np.zeros((1, 1, 2, 3)), np.ones((1, 1, 2, 3)))
    assert depth_centroid([0.0, 0.0, 0.0]) is None
