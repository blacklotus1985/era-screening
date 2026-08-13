"""
ERA v2 canonical metrics
==========================

The complete metric surface of the canonical pipeline.  Five functions, one
job each:

    k_divergence      distribution drift, directional      (L1/L2)
    js_divergence     distribution drift, symmetric        (L1/L2)
    cosine_similarity angle between two hidden vectors     (building block)
    linear_cka        per-layer representational change    (L3)
    drift_centroid    where over depth a curve concentrates (summary)

Design rules (v2):

* One distribution family.  Lin's K-divergence (and its symmetrised form,
  JS divergence) is computed against the midpoint mixture M = (P+Q)/2, so it
  is bounded and well-defined even when the two supports do not overlap.
  Raw KL is intentionally not part of the canonical surface: it is unbounded,
  support-sensitive, and was the source of a silent-zero bug in v1 (the
  corrected historical version is preserved in the archived v1 repository;
  see docs/HISTORY.md).
* No composite scores.  The v1 "Alignment Score" divided quantities with
  different units and attached uncalibrated deployment labels; it has been
  removed everywhere.
* Library functions raise typed exceptions.  Never sys.exit.
* Distributions are plain dicts mapping a hashable key (token ID) to a
  non-negative float.  Keys are token IDs, not decoded strings: different
  IDs can decode to the same string and would silently overwrite each other.

References
----------
Lin, J. (1991). Divergence measures based on the Shannon entropy.
    IEEE Transactions on Information Theory, 37(1), 145-151.
Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of
    neural network representations revisited. ICML 2019.
"""

from typing import Any, Dict, Optional

import numpy as np

# Token-ID -> probability.  Keys are Any (not a stricter Hashable bound)
# so that Dict[int, float], the concrete type the pipeline produces,
# type-checks without variance gymnastics.
Distribution = Dict[Any, float]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _normalise(dist: Distribution, name: str) -> Distribution:
    """Validate a probability dict and rescale so the values sum to 1.

    Raises
    ------
    ValueError
        If any value is negative, NaN or infinite (an invalid input must fail
        loudly, not be silently repaired), or if the distribution is non-empty
        but carries zero total mass.
    """
    values = {k: float(v) for k, v in dist.items()}
    bad = {k: v for k, v in values.items() if not np.isfinite(v) or v < 0.0}
    if bad:
        raise ValueError(
            f"Invalid probabilities in {name} (negative, NaN or inf): {bad}."
        )
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError(f"Probability distribution sums to zero in {name}.")
    return {k: v / total for k, v in values.items()}


def _log_scale(log_base: Optional[float], name: str) -> float:
    """Divisor that converts natural-log values to base ``log_base``.

    Raises
    ------
    ValueError
        For any base not strictly greater than 1.  Bases in (0, 1) have a
        negative logarithm and would flip the sign of the divergence,
        silently violating the "non-negative, bounded by log_base(2)"
        contract; log(1) = 0 would divide by zero.  Rather than document an
        anomalous semantics nobody needs, the API requires log_base > 1.
    """
    if log_base is None:
        return 1.0
    base = float(log_base)
    if not np.isfinite(base) or base <= 1.0:
        raise ValueError(f"log_base must be > 1 in {name}, got {log_base!r}.")
    return float(np.log(base))


# ---------------------------------------------------------------------------
# Distribution drift (L1 / L2)
# ---------------------------------------------------------------------------

def k_divergence(
    p_dist: Distribution,
    q_dist: Distribution,
    log_base: Optional[float] = None,
) -> float:
    """Lin's K-divergence  K(P ‖ Q) = KL(P ‖ M),  M = (P + Q) / 2.

    Because P is compared against the midpoint M rather than against Q
    directly, every token with P(x) > 0 also has M(x) >= P(x)/2 > 0: the log
    ratio is always finite, no smoothing tricks are needed, and the value is
    bounded by log(2) (or 1.0 with ``log_base=2``).  Directional: K(P,Q)
    measures drift from P's point of view; in ERA, P is the base model and
    Q the fine-tuned model.

    Both inputs are normalised over the union of their supports first, so a
    token that only one model predicts counts fully toward the divergence.

    Measurement scope (state this precisely when reporting results): when the inputs are
    top-k distributions, the value measures the shape of the retained mass,
    conditioned on it, two models that give the top-k very different total
    mass but the same relative shape score near zero.  The pipeline therefore
    records each model's top-k mass coverage alongside every drift value.

    Parameters
    ----------
    p_dist, q_dist : dict
        Token-ID -> probability.  Need not sum to 1 (normalised internally).
    log_base : float or None
        None (default) = natural log, range [0, log 2 ≈ 0.693], the
        convention used by every published ERA result.  2 = range [0, 1].

    Returns
    -------
    float in [0, log_base(2)].  0.0 when BOTH inputs are empty (pipeline
    convention: a context with no usable tokens contributes nothing).

    Raises
    ------
    ValueError
        If a non-empty input carries zero total mass, or if exactly one
        input is empty: one model yielding tokens while the other yields
        none is not "zero drift", it is a one-sided failure upstream
        (e.g. the semantic filter emptied one top-k) that a silent 0.0
        would mask as agreement.
    """
    scale = _log_scale(log_base, "k_divergence")
    if not p_dist and not q_dist:
        return 0.0
    if not p_dist or not q_dist:
        raise ValueError(
            "k_divergence received exactly one empty distribution "
            f"(len(p)={len(p_dist)}, len(q)={len(q_dist)}): this signals a "
            "one-sided upstream failure, not zero drift, and must not be "
            "silently scored as 0.0."
        )

    union = set(p_dist) | set(q_dist)
    p = _normalise({t: p_dist.get(t, 0.0) for t in union}, "k_divergence")
    q = _normalise({t: q_dist.get(t, 0.0) for t in union}, "k_divergence")

    total = 0.0
    for t in union:
        p_val = p[t]
        if p_val <= 0.0:
            continue  # 0 * log(0/x) = 0 by convention
        m_val = 0.5 * (p_val + q[t])  # > 0 whenever p_val > 0
        total += p_val * np.log(p_val / m_val)

    return float(max(total, 0.0)) / float(scale)


def js_divergence(
    p_dist: Distribution,
    q_dist: Distribution,
    log_base: Optional[float] = None,
) -> float:
    """Jensen-Shannon divergence: the symmetrised average of the two K terms.

        JS(P, Q) = 0.5 · K(P ‖ Q) + 0.5 · K(Q ‖ P)

    Symmetric (the score does not depend on which model is called "base"),
    bounded by log(2) (or 1.0 with ``log_base=2``), finite for any supports.

    Parameters / Returns / Raises: as :func:`k_divergence`.
    """
    return 0.5 * (
        k_divergence(p_dist, q_dist, log_base=log_base)
        + k_divergence(q_dist, p_dist, log_base=log_base)
    )


DISTRIBUTION_METRICS = {
    "k_divergence": k_divergence,
    "js_divergence": js_divergence,
}


# ---------------------------------------------------------------------------
# Vector geometry (building block + L3)
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine of the angle between two vectors, in [-1, 1].

    Returns 0.0 if either vector has zero norm (an all-zero hidden state has
    no direction to compare).  Accepts anything ``np.asarray`` can convert.

    The result is clamped to [-1, 1]: floating-point rounding can push the
    ratio a hair outside the mathematical range (e.g. cos(v, v) > 1 by ~1e-16),
    which would make the derived drift 1 - cos negative and be rejected,
    correctly, by :func:`drift_centroid` downstream.  Clamping removes only
    that rounding noise; it never alters a genuinely in-range value.
    """
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"cosine_similarity expects vectors of equal length, got {a.shape[0]} vs {b.shape[0]}."
        )
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representation matrices.

    ``X`` and ``Y`` are ``(n_samples, dim)`` matrices of hidden states, same
    rows = same tokens, from the base and the fine-tuned model at one layer.
    CKA is invariant to rotation and isotropic scaling of either space, and
    lives in [0, 1]: 1.0 means the layer encodes the token set identically up
    to an orthogonal transform; lower means fine-tuning reorganised it more.
    ``1 - CKA`` is therefore the canonical per-layer representational-change
    score: unlike raw cosine drift it is insensitive to the *shared mean
    direction* of the space, the specific anisotropy mechanism that confounded
    the v1 results.  It is not immune to every geometry effect, treat it as
    the primary depth view, cross-checked against the anisotropy diagnostics,
    not as an infallible one.

    Closed form on feature-centred matrices (Kornblith et al., 2019)::

        CKA(X, Y) = ‖Xᵀ Y‖_F²  /  ( ‖Xᵀ X‖_F · ‖Yᵀ Y‖_F )

    Known limitation (documented, deliberately not hidden): the standard
    estimator is biased in high-dimension / low-sample regimes.  The debiased
    variant is on the v2 roadmap; until then, report n_samples alongside CKA.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            f"linear_cka expects 2-D (n_samples, dim) matrices, got ndim {X.ndim} and {Y.ndim}."
        )
    if X.shape != Y.shape:
        raise ValueError(
            f"linear_cka expects matrices of identical shape, got {X.shape} vs {Y.shape}."
        )
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = float(np.sum((X.T @ Y) ** 2))
    norm_x = float(np.sqrt(np.sum((X.T @ X) ** 2)))
    norm_y = float(np.sqrt(np.sum((Y.T @ Y) ** 2)))
    denom = norm_x * norm_y
    if denom <= 0.0:
        return 0.0
    # Clamp: floating-point rounding can push the ratio a hair above 1.
    return min(hsic / denom, 1.0)


# ---------------------------------------------------------------------------
# Depth summary
# ---------------------------------------------------------------------------

def drift_centroid(curve: np.ndarray) -> float:
    """Depth centre-of-mass of a per-layer drift curve.

        centroid = Σ (layer · drift[layer]) / Σ drift[layer]

    One interpretable number for where change concentrates over depth:
    near the last layer suggests surface-level adjustment, mid-depth suggests
    broader reorganisation.  It is a descriptive summary, v2 attaches no
    automatic deep/shallow verdict to it.  Returns 0.0 for an all-zero curve.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size and (not np.all(np.isfinite(curve)) or np.any(curve < 0.0)):
        raise ValueError(
            "drift_centroid expects a curve of non-negative finite values "
            "(drift magnitudes); got negative, NaN or inf entries."
        )
    total = float(np.sum(curve))
    if total <= 0.0:
        return 0.0
    layers = np.arange(len(curve))
    return float(np.sum(layers * curve) / total)
