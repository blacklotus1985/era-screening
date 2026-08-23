"""Strict NumPy implementations of the metrics used in the ERA paper.

Distributions are one-dimensional arrays: the array position is the token ID.
No function inserts epsilon, repairs zero mass, or silently drops a token.
"""

import math
from typing import Mapping, Optional, Sequence

import numpy as np

# Used only to recognise final floating-point roundoff. It is never added to a
# probability or denominator.
ROUNDOFF = 64.0 * np.finfo(np.float64).eps


def probabilities(values: Sequence[float], name="distribution") -> np.ndarray:
    """Validate and normalise a dense probability vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} contains a negative, NaN or infinite value.")
    total = float(math.fsum(float(value) for value in array))
    if total <= 0.0:
        raise ValueError(f"{name} has zero total mass.")
    return array / total


def _log_scale(log_base: Optional[float]) -> float:
    if log_base is None:
        return 1.0
    base = float(log_base)
    if not np.isfinite(base) or base <= 1.0:
        raise ValueError("log_base must be None (nats) or greater than 1.")
    return float(math.log(base))


def _lin_normalised(p, q, log_base=None) -> float:
    """Lin K-divergence for two already-normalised vectors."""
    if p.shape != q.shape:
        raise ValueError(f"distribution shapes differ: {p.shape} vs {q.shape}.")

    # If p_i > 0, m_i=(p_i+q_i)/2 is also > 0. No smoothing is needed.
    positive = p > 0.0
    midpoint = 0.5 * (p[positive] + q[positive])
    terms = p[positive] * np.log(p[positive] / midpoint)
    value = float(math.fsum(float(term) for term in terms))

    if value < -ROUNDOFF:
        raise ArithmeticError(f"K-divergence became negative: {value}.")
    if value < 0.0:  # machine roundoff only
        value = 0.0
    if value > math.log(2.0) + ROUNDOFF:
        raise ArithmeticError(f"K-divergence exceeds log(2): {value}.")
    return value / _log_scale(log_base)


def lin_k(p_values, q_values, log_base=None) -> float:
    """Lin's directional K-divergence: ``KL(P || (P+Q)/2)``."""
    p = probabilities(p_values, "P")
    q = probabilities(q_values, "Q")
    return _lin_normalised(p, q, log_base)


def top_k_ids(values, k: int) -> tuple[int, ...]:
    """Top-k token IDs; exact ties use ascending token ID."""
    p = probabilities(values)
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > p.size:
        raise ValueError(f"k={k} exceeds vocabulary size {p.size}.")
    ids = np.arange(p.size, dtype=np.int64)
    ranked = np.lexsort((ids, -p))
    return tuple(int(token_id) for token_id in ranked[:k])


def top_p_ids(values, alpha: float) -> tuple[int, ...]:
    """Smallest standard nucleus whose probability mass is at least alpha."""
    p = probabilities(values)
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    ids = np.arange(p.size, dtype=np.int64)
    ranked = np.lexsort((ids, -p))
    selected, mass = [], 0.0
    for token_id in ranked:
        token_id = int(token_id)
        selected.append(token_id)
        mass += float(p[token_id])
        if mass >= alpha:
            break
    return tuple(selected)


def restricted_B(p_values, q_values, support_p, support_q=None, log_base=None):
    """Divergence conditioned on the exact union of two supports."""
    p = probabilities(p_values, "P")
    q = probabilities(q_values, "Q")
    if p.shape != q.shape:
        raise ValueError(f"distribution shapes differ: {p.shape} vs {q.shape}.")
    right = support_p if support_q is None else support_q
    support = tuple(sorted(set(map(int, support_p)) | set(map(int, right))))
    if not support:
        raise ValueError("support must not be empty.")
    if support[0] < 0 or support[-1] >= p.size:
        raise ValueError("support contains an invalid token ID.")

    indices = np.asarray(support, dtype=np.int64)
    p_selected, q_selected = p[indices], q[indices]
    mass_p = float(math.fsum(map(float, p_selected)))
    mass_q = float(math.fsum(map(float, q_selected)))
    if mass_p <= 0.0 or mass_q <= 0.0:
        raise ValueError("both models must assign positive mass to the support.")
    return {
        "value": _lin_normalised(p_selected / mass_p, q_selected / mass_q, log_base),
        "support": support,
        "mass_m0": mass_p,
        "mass_m1": mass_q,
    }


def B(p_values, q_values, log_base=None) -> float:
    """Complete-vocabulary output drift."""
    return lin_k(p_values, q_values, log_base)


def B_k(p_values, q_values, k: int, log_base=None):
    """Output drift on the exact union of the two top-k supports."""
    return restricted_B(
        p_values,
        q_values,
        top_k_ids(p_values, k),
        top_k_ids(q_values, k),
        log_base,
    )


def B_alpha(p_values, q_values, alpha: float, log_base=None):
    """Output drift on the exact union of the two top-p supports."""
    return restricted_B(
        p_values,
        q_values,
        top_p_ids(p_values, alpha),
        top_p_ids(q_values, alpha),
        log_base,
    )


def _check_groups(groups: Mapping[str, Sequence[int]], vocabulary_size: int):
    if len(groups) < 2:
        raise ValueError("at least two non-empty groups are required.")
    checked, used = {}, set()
    for name, token_ids in groups.items():
        ids = tuple(map(int, token_ids))
        if not isinstance(name, str) or not name or not ids:
            raise ValueError("every group needs a name and at least one token.")
        if len(ids) != len(set(ids)):
            raise ValueError(f"group {name!r} contains duplicate token IDs.")
        overlap = used.intersection(ids)
        if overlap:
            raise ValueError(f"groups overlap on token IDs {sorted(overlap)}.")
        if min(ids) < 0 or max(ids) >= vocabulary_size:
            raise ValueError(f"group {name!r} contains an invalid token ID.")
        checked[name] = ids
        used.update(ids)
    return checked, tuple(sorted(used))


def B_T(p_values, q_values, groups: Mapping[str, Sequence[int]], log_base=None):
    """Target drift and exact ``between + within`` decomposition."""
    p_full = probabilities(p_values, "P")
    q_full = probabilities(q_values, "Q")
    if p_full.shape != q_full.shape:
        raise ValueError(
            f"distribution shapes differ: {p_full.shape} vs {q_full.shape}."
        )
    groups, target = _check_groups(groups, p_full.size)
    target_indices = np.asarray(target, dtype=np.int64)
    p_raw, q_raw = p_full[target_indices], q_full[target_indices]
    mass_p = float(math.fsum(map(float, p_raw)))
    mass_q = float(math.fsum(map(float, q_raw)))
    if mass_p <= 0.0 or mass_q <= 0.0:
        raise ValueError("both models must assign positive mass to target set T.")
    p, q = p_raw / mass_p, q_raw / mass_q

    position = {token_id: index for index, token_id in enumerate(target)}
    positions = {
        name: np.asarray([position[token_id] for token_id in ids], dtype=np.int64)
        for name, ids in groups.items()
    }
    group_p = np.asarray([p[index].sum() for index in positions.values()])
    group_q = np.asarray([q[index].sum() for index in positions.values()])
    total = _lin_normalised(p, q, log_base)
    between = _lin_normalised(group_p, group_q, log_base)

    # Chain rule for KL(P || M), with M=(P+Q)/2.
    midpoint, within_terms = 0.5 * (p + q), []
    for index in positions.values():
        p_group, q_group = float(p[index].sum()), float(q[index].sum())
        midpoint_group = 0.5 * (p_group + q_group)
        if p_group == 0.0:
            continue
        for token in index:
            if p[token] > 0.0:
                within_terms.append(
                    float(p[token])
                    * math.log(
                        (p[token] / p_group) / (midpoint[token] / midpoint_group)
                    )
                )
    within_nats = float(math.fsum(within_terms))
    if within_nats < -ROUNDOFF:
        raise ArithmeticError(f"within-group term became negative: {within_nats}.")
    if within_nats < 0.0:  # machine roundoff only
        within_nats = 0.0
    within = within_nats / _log_scale(log_base)
    residual = total - between - within
    if not np.isclose(total, between + within, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("B_T != B_between + B_within.")
    fractions = (None, None) if total == 0.0 else (between / total, within / total)
    names = list(groups)
    return {
        "total": total,
        "between": between,
        "within": within,
        "between_fraction": fractions[0],
        "within_fraction": fractions[1],
        "group_mass_m0_conditional": dict(zip(names, map(float, group_p))),
        "group_mass_m1_conditional": dict(zip(names, map(float, group_q))),
        "target_mass_m0": mass_p,
        "target_mass_m1": mass_q,
        "decomposition_residual": residual,
    }


def group_gap(values, positive_ids, negative_ids, conditional=True) -> float:
    """Positive-group minus negative-group probability mass."""
    p = probabilities(values)
    positive, negative = tuple(map(int, positive_ids)), tuple(map(int, negative_ids))
    if set(positive).intersection(negative):
        raise ValueError("positive and negative groups overlap.")
    ids = positive + negative
    if not ids or min(ids) < 0 or max(ids) >= p.size:
        raise ValueError("group contains an invalid token ID.")
    positive_mass = float(p[np.asarray(positive)].sum())
    negative_mass = float(p[np.asarray(negative)].sum())
    if conditional:
        target_mass = positive_mass + negative_mass
        if target_mass <= 0.0:
            raise ValueError("both groups have zero probability mass.")
        positive_mass, negative_mass = (
            positive_mass / target_mass,
            negative_mass / target_mass,
        )
    return positive_mass - negative_mass


def stereotype_index(gaps, families) -> dict:
    """Return leadership bias, support bias and ``SI = LB - SB``."""
    if len(gaps) != len(families) or not gaps:
        raise ValueError("gaps and families must be non-empty and equally long.")
    values = np.asarray(gaps, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("gaps must be finite.")
    leadership = values[np.asarray([name == "leadership" for name in families])]
    support = values[np.asarray([name == "support" for name in families])]
    if leadership.size == 0 or support.size == 0:
        raise ValueError("both leadership and support contexts are required.")
    lb, sb = float(leadership.mean()), float(support.mean())
    return {
        "leadership_bias": lb,
        "support_bias": sb,
        "SI": lb - sb,
        "per_context_gap": [float(value) for value in values],
    }


def delta_si(base: dict, finetuned: dict) -> dict:
    """Return Delta SI and verify ``Delta SI = Delta LB - Delta SB``."""
    delta_lb = finetuned["leadership_bias"] - base["leadership_bias"]
    delta_sb = finetuned["support_bias"] - base["support_bias"]
    change = finetuned["SI"] - base["SI"]
    residual = change - (delta_lb - delta_sb)
    if not np.isclose(residual, 0.0, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("Delta SI decomposition failed.")
    return {
        "m0": base,
        "m1": finetuned,
        "delta_SI": change,
        "delta_leadership": delta_lb,
        "delta_support": delta_sb,
        "decomposition_residual": residual,
    }


def depth_centroid(curve) -> Optional[float]:
    """Centre of mass on normalised depth [0,1], or None for a zero curve."""
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("curve must be a non-empty vector.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("curve must be finite and non-negative.")
    total = float(values.sum())
    if total == 0.0:
        return None
    if values.size == 1:
        return 0.0
    layers = np.arange(values.size, dtype=np.float64)
    return float((layers @ values) / total / (values.size - 1))


def G_l(states_m0, states_m1) -> dict:
    """Layer-wise cosine between context-averaged concept centroids.

    Inputs have shape ``(contexts, concepts, layers, hidden_size)``.
    """
    base, tuned = np.asarray(states_m0, dtype=np.float64), np.asarray(
        states_m1, dtype=np.float64
    )
    if base.ndim != 4 or tuned.ndim != 4 or base.shape != tuned.shape:
        raise ValueError("state tensors must have identical four-dimensional shape.")
    if (
        0 in base.shape
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(tuned))
    ):
        raise ValueError("state tensors must be non-empty and finite.")
    centroid_base, centroid_tuned = base.mean(axis=0), tuned.mean(axis=0)
    norm_base = np.linalg.norm(centroid_base, axis=-1)
    norm_tuned = np.linalg.norm(centroid_tuned, axis=-1)
    zero = np.argwhere((norm_base == 0.0) | (norm_tuned == 0.0))
    if zero.size:
        concept, layer = map(int, zero[0])
        raise ValueError(
            f"cosine undefined for zero centroid (concept={concept}, layer={layer})."
        )
    cosine = np.sum(centroid_base * centroid_tuned, axis=-1) / (norm_base * norm_tuned)
    if np.any(cosine < -1.0 - ROUNDOFF) or np.any(cosine > 1.0 + ROUNDOFF):
        raise ArithmeticError("cosine lies outside [-1,1] beyond roundoff.")
    cosine = np.minimum(1.0, np.maximum(-1.0, cosine))
    similarity = cosine.mean(axis=0)
    drift = 1.0 - similarity
    drift[(drift < 0.0) & (drift >= -ROUNDOFF)] = 0.0
    if np.any(drift < 0.0):
        raise ArithmeticError("1-G_l became negative beyond roundoff.")
    return {
        "similarity": similarity,
        "drift": drift,
        "per_concept_similarity": cosine,
        "normalised_depth_centroid": depth_centroid(drift),
    }
