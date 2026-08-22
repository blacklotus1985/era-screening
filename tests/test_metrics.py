"""
Tests for era.metrics (v2 canonical surface).

Every metric is checked against a hand-computed reference value or a formal
property (bound, symmetry, invariance).  All tests are deterministic and run
without torch, a GPU, or a network connection.
"""

import math

import numpy as np
import pytest

from era.metrics import (
    cosine_similarity,
    drift_centroid,
    js_divergence,
    k_divergence,
    linear_cka,
    linear_cka_unbiased,
)


# ---------------------------------------------------------------------------
# K-divergence
# ---------------------------------------------------------------------------

def test_k_matches_hand_computed_value():
    """K(P‖Q) = Σ p·log(p/m) with m = (p+q)/2, natural log.

    P = {a: .6, b: .4}, Q = {a: .5, b: .5}  ->  M = {a: .55, b: .45}
    K = .6·ln(.6/.55) + .4·ln(.4/.45)
    """
    p = {1: 0.6, 2: 0.4}
    q = {1: 0.5, 2: 0.5}
    expected = 0.6 * math.log(0.6 / 0.55) + 0.4 * math.log(0.4 / 0.45)
    assert k_divergence(p, q) == pytest.approx(expected)


def test_k_zero_for_identical_distributions():
    p = {1: 0.7, 2: 0.3}
    assert k_divergence(p, p) == pytest.approx(0.0)


def test_k_disjoint_supports_hits_the_log2_bound():
    """Maximally different distributions reach exactly log(2) (natural log).

    With disjoint supports, M(x) = P(x)/2 wherever P(x) > 0, so every term
    contributes p·ln(2).  This is the case that raw KL handled catastrophically
    in v1 (silent 0); K-divergence handles it by construction.
    """
    p = {1: 1.0}
    q = {2: 1.0}
    assert k_divergence(p, q) == pytest.approx(math.log(2.0))


def test_k_log_base_2_maps_to_unit_interval():
    """log_base=2 rescales the bound from ln(2) to exactly 1."""
    p = {1: 1.0}
    q = {2: 1.0}
    assert k_divergence(p, q, log_base=2) == pytest.approx(1.0)


def test_k_is_directional():
    p = {1: 0.99, 2: 0.01}
    q = {1: 0.60, 2: 0.40}
    assert k_divergence(p, q) != pytest.approx(k_divergence(q, p))


def test_k_normalises_unnormalised_inputs():
    """Inputs that do not sum to 1 (raw top-k mass) are normalised internally."""
    p = {1: 6.0, 2: 4.0}   # same shape as {.6, .4}
    q = {1: 5.0, 2: 5.0}
    expected = 0.6 * math.log(0.6 / 0.55) + 0.4 * math.log(0.4 / 0.45)
    assert k_divergence(p, q) == pytest.approx(expected)


def test_k_both_empty_inputs_return_zero():
    assert k_divergence({}, {}) == 0.0


def test_k_one_sided_empty_raises():
    """Exactly one empty distribution is an upstream failure (e.g. the
    semantic filter emptied one model's top-k), not zero drift: a silent 0.0
    would score a broken comparison as perfect agreement."""
    with pytest.raises(ValueError, match="one empty"):
        k_divergence({}, {1: 1.0})
    with pytest.raises(ValueError, match="one empty"):
        k_divergence({1: 1.0}, {})


def test_k_zero_mass_raises_value_error():
    """Non-empty but all-zero input must raise, never exit or return junk."""
    with pytest.raises(ValueError, match="sums to zero"):
        k_divergence({1: 0.0}, {2: 1.0})


def test_k_negative_probability_raises():
    """A negative probability is an invalid input, not something to repair
    silently: the declared interface requires non-negative floats."""
    with pytest.raises(ValueError, match="Invalid probabilities"):
        k_divergence({1: -0.2, 2: 1.2}, {1: 0.5, 2: 0.5})


def test_k_nan_and_inf_raise():
    with pytest.raises(ValueError, match="Invalid probabilities"):
        k_divergence({1: float("nan"), 2: 0.5}, {1: 0.5, 2: 0.5})
    with pytest.raises(ValueError, match="Invalid probabilities"):
        k_divergence({1: float("inf"), 2: 0.5}, {1: 0.5, 2: 0.5})


def test_k_invalid_log_base_raises():
    """The API requires log_base > 1: log(1) = 0 would divide by zero,
    non-positive bases are undefined, and bases in (0, 1) have a NEGATIVE
    logarithm that would flip the divergence sign (verified regression:
    base 0.5 on disjoint supports used to return -1.0)."""
    p, q = {1: 0.6, 2: 0.4}, {1: 0.5, 2: 0.5}
    for bad in (1.0, 0.5, 0.0, -2.0, float("nan")):
        with pytest.raises(ValueError, match="log_base"):
            k_divergence(p, q, log_base=bad)


# ---------------------------------------------------------------------------
# JS divergence
# ---------------------------------------------------------------------------

def test_js_is_average_of_the_two_k_terms():
    p = {1: 0.9, 2: 0.1}
    q = {1: 0.2, 2: 0.8}
    expected = 0.5 * (k_divergence(p, q) + k_divergence(q, p))
    assert js_divergence(p, q) == pytest.approx(expected)


def test_js_is_symmetric():
    p = {1: 0.9, 2: 0.1}
    q = {1: 0.2, 2: 0.8}
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))


def test_js_bounded_by_log_two():
    p = {1: 1.0}
    q = {2: 1.0}
    assert js_divergence(p, q) == pytest.approx(math.log(2.0))
    assert js_divergence(p, q, log_base=2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def test_cosine_identical_orthogonal_opposite():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_cosine_length_mismatch_raises():
    """Different-length vectors are a caller bug, fail with a clear message,
    not with an opaque NumPy broadcast error."""
    with pytest.raises(ValueError, match="equal length"):
        cosine_similarity(np.ones(3), np.ones(4))


def test_cosine_self_similarity_never_exceeds_one():
    """Regression: rounding can push dot(v, v) / (||v||·||v||) a hair above 1
    (found by the model-vs-itself smoke run: the derived drift 1 - cos went
    negative by ~1e-16 and drift_centroid, correctly strict, rejected the
    whole curve).  The vector below reproduces the overflow without the clamp.
    """
    v = np.array([0.1, 0.7])  # raw ratio = 1.0000000000000002 in float64
    assert cosine_similarity(v, v) <= 1.0
    assert 1.0 - cosine_similarity(v, v) >= 0.0
    # And the clamp must not disturb genuinely in-range values.
    assert cosine_similarity(v, -v) >= -1.0
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Linear CKA
# ---------------------------------------------------------------------------

def _random_matrix(rows: int = 12, dim: int = 5, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, dim))


def test_cka_identical_matrices_is_one():
    X = _random_matrix()
    assert linear_cka(X, X) == pytest.approx(1.0)


def test_cka_invariant_to_rotation():
    """CKA(X, X·R) = 1 for any orthogonal R, the property that makes 1−CKA a
    reorganisation score rather than a rotation detector."""
    X = _random_matrix(seed=1)
    rng = np.random.default_rng(2)
    R, _ = np.linalg.qr(rng.normal(size=(X.shape[1], X.shape[1])))
    assert linear_cka(X, X @ R) == pytest.approx(1.0)


def test_cka_invariant_to_isotropic_scaling():
    X = _random_matrix(seed=3)
    assert linear_cka(X, 7.3 * X) == pytest.approx(1.0)


def test_cka_below_one_for_independent_matrices():
    X = _random_matrix(seed=4)
    Y = _random_matrix(seed=5)
    assert linear_cka(X, Y) < 0.9


def test_cka_shape_mismatch_raises():
    with pytest.raises(ValueError, match="identical shape"):
        linear_cka(np.zeros((4, 3)), np.zeros((5, 3)))


def test_cka_zero_matrices_return_zero():
    Z = np.zeros((6, 4))
    assert linear_cka(Z, Z) == 0.0


def test_cka_requires_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        linear_cka(np.ones(5), np.ones(5))


def test_cka_never_exceeds_one():
    """Floating-point rounding must not push CKA above its theoretical max."""
    X = _random_matrix(rows=50, dim=3, seed=7)
    assert linear_cka(X, X) <= 1.0
    assert linear_cka(X, 3.0 * X) <= 1.0


def test_unbiased_cka_shows_known_small_sample_gap():
    X = _random_matrix(rows=12, dim=5, seed=21)
    Y = _random_matrix(rows=12, dim=5, seed=22)
    biased = linear_cka(X, Y)
    unbiased = linear_cka_unbiased(X, Y)
    assert biased > unbiased


def test_unbiased_cka_converges_to_biased_for_large_n_small_d():
    X = _random_matrix(rows=2000, dim=3, seed=23)
    Y = _random_matrix(rows=2000, dim=3, seed=24)
    assert linear_cka(X, Y) == pytest.approx(
        linear_cka_unbiased(X, Y), abs=0.02
    )


# ---------------------------------------------------------------------------
# Drift centroid
# ---------------------------------------------------------------------------

def test_centroid_all_mass_on_last_layer():
    assert drift_centroid(np.array([0.0, 0.0, 1.0])) == pytest.approx(2.0)


def test_centroid_uniform_curve_is_midpoint():
    assert drift_centroid(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)


def test_centroid_zero_curve_returns_zero():
    assert drift_centroid(np.zeros(5)) == 0.0


def test_centroid_negative_or_nan_curve_raises():
    """Drift magnitudes are non-negative by definition; anything else is a
    caller bug and must not silently distort the centre of mass."""
    with pytest.raises(ValueError, match="non-negative"):
        drift_centroid(np.array([0.1, -0.2, 0.3]))
    with pytest.raises(ValueError, match="non-negative"):
        drift_centroid(np.array([0.1, float("nan"), 0.3]))
