#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA - extended cross-architecture aggregation (11 models, 28 cells)
====================================================================

Companion to ``11_compare_multiseed.py``, which compares the two reference
models.  This script covers the whole panel and adds the analyses the extended
study needs.  **It reuses script 11's loaders unchanged**: every centroid,
every mean and every std that also appears in an 11 report is the same number
here, read from the same ``run_config.json`` fields.  Nothing is recomputed
that 11 already computed.

What it produces
----------------
(a) **Master table** - one row per model: parameters, blocks, hidden size,
    positional encoding, seed count, and for each of the three metrics the
    across-seed centroid both absolute and normalised by ``len(curve) - 1``
    (so 0 = embedding output, 1 = final block), with its across-seed std, the
    argmax, ``n_cka_samples`` and the device the cells were measured on.

(b) **Overlays on normalised depth** - the three metric curves for all models
    on one depth axis, plus an anisotropy panel.  Absolute heights are not
    comparable across architectures; the normalised axis is what makes the
    *shape* comparison legitimate, and the anisotropy panel is what says
    whether a shape is a property of the change or of the space it is
    measured in.

(c) **The certified ceiling** (docs/SATURATION_LEMMA.md) - per layer,
    ``ceiling = 2 - anisotropy_base - anisotropy_ft``, ``headroom_used =
    relational / ceiling`` and the ``saturation_class`` at the preregistered
    0.95 flag.  A near-zero drift at a layer whose ceiling is near zero is not
    evidence of no change: the metric had no room to report one.

(d) **Paired differential curves**, biased minus neutral, for the two
    reference models.  The difference is taken **within seed** and only then
    aggregated, because the pairing is what removes the format adaptation and
    a difference of two across-seed means would throw the pairing away.  The
    result is a *signed* curve, and **no centroid is computed on it**: the
    centroid is a centre of mass over a non-negative curve, and on a curve
    that changes sign it is not a depth, it is an artefact of where the
    positive and negative lobes happen to cancel.

(e) **CKA sensitivity** - the shipped ``era.metrics.linear_cka`` is the
    standard biased estimator, which attributes similarity to unrelated
    representations in the high-dimension / low-sample regime this panel sits
    in (hidden sizes 512-1024 against ~1300-1500 CKA samples).  This script
    implements the unbiased HSIC estimator, exercises both on synthetic data
    with known answers, and calibrates each model's own (n, d) so a reported
    ``1 - CKA`` can be read against the spurious similarity its own geometry
    would produce.  The same section runs the explicit **centroid against
    hidden_size** check, across the panel and inside the Pythia and GPT-2
    families, because "deeper change in bigger models" and "more estimator
    bias in wider models" would look identical in the master table.

Every input number was measured on CUDA cells; this script is numpy and
pandas only, and records both facts separately.

Usage
-----
    python experiments/14_compare_extended.py
    python experiments/14_compare_extended.py --tag v2_balanced --skip-cka-sim
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

SWEEP_WORK_ROOT = ROOT / "era_poc_replication_results_multiseed"
SWEEP_RESULTS_ROOT = ROOT / "results" / "sweep"
CONTROLS_WORK_ROOT = ROOT / "era_poc_calibration_controls"
CONTROLS_RESULTS_ROOT = ROOT / "results" / "controls"
COMPARISON_ROOT = ROOT / "era_poc_replication_results_compare"
CENSUS_CSV = ROOT / "results" / "census" / "model_census.csv"

DEFAULT_TAG = "v2_balanced"

METRIC_KEYS = ("relational", "per_token", "cka_change")
METRIC_LABELS = {
    "relational": "Relational drift\n(mean |d cos| over pairs)",
    "per_token": "Per-token drift\n(mean 1 - cos(base, ft))",
    "cka_change": "Representational change\n(1 - linear CKA)",
}

# Preregistered diagnostic flag on anisotropy (docs/SATURATION_LEMMA.md:
# a flag, not a derived threshold - the per-layer ceiling supersedes it).
ANISOTROPY_FLAG = 0.95


def _column_with_legacy_alias(frame, current, legacy):
    """Read a current CSV column, accepting one deprecated alias."""
    return frame[current] if current in frame.columns else frame[legacy]

# The two models the domain control was run for, hence the only ones with a
# paired neutral cell to subtract.
REFERENCE_SLUGS = ("gptneo", "pythia")
NEUTRAL_CONTROL = "control_B_domain"

# Mixing coefficients for the CKA sensitivity simulation.  0.0 is the null
# case, where the population CKA is exactly 0 and any positive reading is
# estimator bias with no interpretation needed.  The levels crowd towards 1.0
# because that is where the measured cells sit: every reported CKA in this
# panel is above 0.9, and a calibration spaced evenly over [0, 1] would put
# almost no points where the answer is actually needed.  1.0 is included so
# the interpolation is anchored at both ends (identical inputs give exactly
# 1.0 under either estimator).
CKA_SIM_MIXES = (0.0, 0.3, 0.7, 0.9, 0.95, 0.98, 0.99, 0.997, 1.0)

SIGNED_CURVE_NOTE = (
    "No centroid is computed on the differential curves. The centroid is a "
    "centre of mass over a non-negative curve; on a signed curve it reports "
    "where the positive and negative lobes happen to cancel, which is not a "
    "depth. Differential curves are read layer by layer, or not at all."
)


def _load_sibling(module_name, filename):
    """Import a numbered sibling script as a module (they are not importable)."""
    spec = importlib.util.spec_from_file_location(module_name, _HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==============================================================================
# INPUT RESOLUTION
# ==============================================================================

def resolve_root(explicit, work_root, results_root, tag, what):
    """``(root, provenance, skipped)`` - the exported ``results/`` tree by default.

    The exported tree is preferred over the working tree deliberately.  It is
    the one ``98_export_results.py --verify`` has checked for completeness and
    device provenance, and the one that is committed; a working tree can hold
    a partial sweep, or an old pilot cell from a different machine, and
    picking it up would silently analyse a different panel.  The working tree
    is still reachable with ``--<what>-root``, and when one exists it is named
    in the output rather than passed over in silence.
    """
    if explicit:
        return Path(explicit), f"explicit --{what}-root", None
    skipped = str(Path(work_root)) if (Path(work_root) / tag).is_dir() else None
    if (Path(results_root) / tag).is_dir():
        return Path(results_root), "exported results/ tree", skipped
    if skipped:
        return Path(work_root), "working tree (no exported tree found)", None
    raise SystemExit(f"No {what} data for tag {tag!r}: looked in {results_root} "
                     f"and {work_root}. Run experiments/98_export_results.py, "
                     f"or pass --{what}-root.")


def load_census():
    """Architecture facts per model, from the preflight census."""
    if not CENSUS_CSV.is_file():
        raise SystemExit(f"Census table missing: {CENSUS_CSV}. Run "
                         "experiments/13_preflight_census.py first.")
    census = pd.read_csv(CENSUS_CSV).set_index("slug")
    return census


def cell_configs(seed_dirs):
    """The ``run_config.json`` of each seed cell."""
    configs = []
    for seed_dir in seed_dirs:
        path = Path(seed_dir) / "run_config.json"
        if path.is_file():
            configs.append(json.loads(path.read_text(encoding="utf-8")))
    return configs


def load_layer_frames(seed_dirs):
    """Raw ``layer_curve.csv`` per seed - the anisotropy columns script 11 drops."""
    return [pd.read_csv(Path(seed_dir) / "layer_curve.csv") for seed_dir in seed_dirs]


# ==============================================================================
# (a) MASTER TABLE
# ==============================================================================

def normalised_centroid(centroid, n_points):
    """Centroid on a 0-1 depth axis, denominator ``len(curve) - 1``.

    The centroid lives on layer indices ``0 .. n_points - 1``, so its maximum
    is ``n_points - 1`` and that is the only denominator that makes 1.0 mean
    "all the mass sits on the final block" for every architecture.
    """
    if n_points is None or n_points < 2 or centroid is None:
        return float("nan")
    return float(centroid) / float(n_points - 1)


def centroid_stats(model, key):
    """``(mean, std)`` of the across-seed centroid, exactly as script 11 does."""
    values = np.array(model["centroids"][key], dtype=float)
    n_valid = int(np.sum(~np.isnan(values)))
    if n_valid == 0:
        return float("nan"), float("nan")
    std = float(np.nanstd(values, ddof=1)) if n_valid > 1 else 0.0
    return float(np.nanmean(values)), std


def master_row(model, seed_dirs, census, cka_null, cka_sim_rows=None):
    """One row of the master table for one model."""
    slug = model["short"]
    meta = census.loc[slug] if slug in census.index else None
    configs = cell_configs(seed_dirs)
    n_points = int(len(model["layers"]))
    cka_samples = [c.get("n_cka_samples") for c in configs if c.get("n_cka_samples")]
    devices = sorted({c.get("device") for c in configs if c.get("device")})
    gpus = sorted({c.get("gpu_name") for c in configs if c.get("gpu_name")})

    row = {
        "slug": slug,
        "label": model["label"],
        "hf_id": None if meta is None else meta["hf_id"],
        "tier": None if meta is None else meta["tier"],
        "model_type": None if meta is None else meta["model_type"],
        "n_params": None if meta is None else int(meta["n_params"]),
        "n_blocks": None if meta is None else int(meta["n_blocks"]),
        "n_curve_points": n_points,
        "hidden_size": None if meta is None else int(meta["hidden_size"]),
        "positional_encoding": None if meta is None else meta["positional_encoding"],
        "n_seeds": len(model["seeds"]),
        "seeds": "|".join(model["seeds"]),
        "n_cka_samples_min": min(cka_samples) if cka_samples else None,
        "n_cka_samples_max": max(cka_samples) if cka_samples else None,
        "n_cka_samples_mean": float(np.mean(cka_samples)) if cka_samples else None,
        "device": "|".join(devices),
        "gpu_name": "|".join(gpus),
    }

    for key in METRIC_KEYS:
        mean, std = centroid_stats(model, key)
        row[f"{key}_centroid_mean"] = mean
        row[f"{key}_centroid_std"] = std
        row[f"{key}_centroid_norm_mean"] = normalised_centroid(mean, n_points)
        row[f"{key}_centroid_norm_std"] = normalised_centroid(std, n_points)
        row[f"{key}_argmax_mean_curve"] = int(np.argmax(model[key]["mean"]))
        row[f"{key}_argmax_per_seed"] = "|".join(
            str(int(np.argmax(model[key]["per_seed"][s])))
            for s in range(model[key]["per_seed"].shape[0]))
        row[f"{key}_max_mean_curve"] = float(np.max(model[key]["mean"]))

    if cka_null is not None:
        # The estimator's own floor: what it reports for representations that
        # share nothing.  Recorded as a property of this (n, d), NOT as a
        # threshold the observed change is compared against - the bias shrinks
        # as the true similarity rises, and every cell here sits near 1.
        row["cka_null_biased"] = cka_null["cka_biased"]
        row["cka_null_unbiased"] = cka_null["cka_unbiased"]
    if cka_sim_rows:
        observed_change = row["cka_change_max_mean_curve"]
        debiased_cka = debias_cka(cka_sim_rows, 1.0 - observed_change)
        debiased_change = 1.0 - debiased_cka
        row["cka_change_max_debiased"] = debiased_change
        row["cka_change_inflation_factor"] = (
            float(debiased_change / observed_change)
            if observed_change > 0 else float("nan"))
    return row


# ==============================================================================
# (c) THE CERTIFIED CEILING
# ==============================================================================

def saturation_class(anisotropy_base, anisotropy_ft, flag=ANISOTROPY_FLAG):
    """Which of the two models is above the preregistered anisotropy flag."""
    base_flagged = bool(anisotropy_base >= flag)
    ft_flagged = bool(anisotropy_ft >= flag)
    if base_flagged and ft_flagged:
        return "both"
    if base_flagged:
        return "base_only"
    if ft_flagged:
        return "ft_only"
    return "neither"


def ceiling_frame(slug, label, seeds, frames, n_points):
    """Per-layer, per-seed ceiling table for one model.

    ``ceiling = 2 - anisotropy_base - anisotropy_ft`` is the certified upper
    bound on the relational curve (docs/SATURATION_LEMMA.md, inequality 2).
    ``headroom_used`` is what the measurement actually spent of it.
    """
    rows = []
    for seed, frame in zip(seeds, frames):
        for _, layer_row in frame.iterrows():
            aniso_base = float(layer_row["anisotropy_base"])
            aniso_ft = float(layer_row["anisotropy_ft"])
            relational = float(
                layer_row["relational_mean"]
                if "relational_mean" in layer_row.index
                else layer_row["l3_mean"]
            )
            ceiling = 2.0 - aniso_base - aniso_ft
            headroom = relational / ceiling if ceiling > 0 else float("nan")
            layer = int(layer_row["layer"])
            rows.append({
                "slug": slug,
                "label": label,
                "seed": seed,
                "layer": layer,
                "depth_norm": layer / (n_points - 1) if n_points > 1 else 0.0,
                "relational": relational,
                "anisotropy_base": aniso_base,
                "anisotropy_ft": aniso_ft,
                "ceiling": ceiling,
                "headroom_used": headroom,
                "saturation_class": saturation_class(aniso_base, aniso_ft),
            })
    return rows


def lemma_violations(rows, tolerance=1e-9):
    """Rows where ``headroom_used`` exceeds 1: the lemma says there are none.

    The bound is pointwise and unconditional, so a violation is a bug or a
    corrupted artefact, never a finding.  Counted rather than assumed away.
    """
    return [r for r in rows
            if np.isfinite(r["headroom_used"]) and r["headroom_used"] > 1.0 + tolerance]


# ==============================================================================
# (d) PAIRED DIFFERENTIAL CURVES
# ==============================================================================

def metric_curves(frame):
    """The three per-layer curves of one cell, from its ``layer_curve.csv``."""
    return {
        "relational": _column_with_legacy_alias(
            frame, "relational_mean", "l3_mean"
        ).to_numpy(dtype=float),
        "per_token": frame["per_token_mean"].to_numpy(dtype=float),
        "cka_change": 1.0 - frame["cka"].to_numpy(dtype=float),
    }


def paired_differences(sweep_root, controls_root, tag, slug):
    """``{metric: {seed: curve}}`` of biased minus neutral, differenced in-seed.

    Only seeds present in *both* arms are used: a seed with no neutral cell
    has nothing to subtract, and filling it with the mean of the others would
    invent the pairing this analysis exists to preserve.
    """
    biased_dir = Path(sweep_root) / tag / slug
    neutral_dir = Path(controls_root) / NEUTRAL_CONTROL / slug
    if not biased_dir.is_dir() or not neutral_dir.is_dir():
        return {}, []

    per_metric = {key: {} for key in METRIC_KEYS}
    used_seeds = []
    for seed_dir in sorted(p for p in biased_dir.iterdir() if p.is_dir()):
        seed = seed_dir.name.replace("seed_", "")
        neutral_cell = neutral_dir / seed_dir.name / "layer_curve.csv"
        biased_cell = seed_dir / "layer_curve.csv"
        if not (neutral_cell.is_file() and biased_cell.is_file()):
            continue
        biased = metric_curves(pd.read_csv(biased_cell))
        neutral = metric_curves(pd.read_csv(neutral_cell))
        if len(biased["relational"]) != len(neutral["relational"]):
            continue
        for key in METRIC_KEYS:
            per_metric[key][seed] = biased[key] - neutral[key]
        used_seeds.append(seed)
    return per_metric, used_seeds


def aggregate_differences(per_seed_curves, confidence=0.95):
    """Mean, sample std and a t interval across seeds, per layer.

    The interval is a Student-t interval on two or three points. It is
    reported because an interval that is honestly enormous says more than no
    interval at all, and its width is the finding it carries.
    """
    seeds = sorted(per_seed_curves)
    if not seeds:
        return None
    stacked = np.vstack([per_seed_curves[s] for s in seeds])
    n_seeds = stacked.shape[0]
    mean = stacked.mean(axis=0)
    if n_seeds > 1:
        std = stacked.std(axis=0, ddof=1)
        half = stats.t.ppf(0.5 + confidence / 2.0, n_seeds - 1) * std / np.sqrt(n_seeds)
    else:
        std = np.zeros_like(mean)
        half = np.full_like(mean, np.nan)
    return {"seeds": seeds, "n_seeds": n_seeds, "per_seed": stacked, "mean": mean,
            "std": std, "ci_lo": mean - half, "ci_hi": mean + half}


def differential_rows(slug, label, per_metric, aggregates, n_points):
    rows = []
    for key in METRIC_KEYS:
        agg = aggregates.get(key)
        if agg is None:
            continue
        for layer in range(len(agg["mean"])):
            row = {
                "slug": slug,
                "label": label,
                "metric": key,
                "layer": layer,
                "depth_norm": layer / (n_points - 1) if n_points > 1 else 0.0,
                "n_seeds": agg["n_seeds"],
                "diff_mean": float(agg["mean"][layer]),
                "diff_std": float(agg["std"][layer]),
                "ci95_lo": float(agg["ci_lo"][layer]),
                "ci95_hi": float(agg["ci_hi"][layer]),
                "excludes_zero": bool(
                    np.isfinite(agg["ci_lo"][layer])
                    and (agg["ci_lo"][layer] > 0 or agg["ci_hi"][layer] < 0)),
            }
            for seed in agg["seeds"]:
                row[f"diff_seed_{seed}"] = float(per_metric[key][seed][layer])
            rows.append(row)
    return rows


# ==============================================================================
# (e) CKA SENSITIVITY: THE UNBIASED HSIC ESTIMATOR
# ==============================================================================

def _hsic_unbiased_from_grams(gram_x, gram_y):
    """Unbiased HSIC from two Gram matrices (Song et al., 2007, eq. 5).

        HSIC_1 = 1/(n(n-3)) [ tr(K~ L~)
                              + (1' K~ 1)(1' L~ 1) / ((n-1)(n-2))
                              - 2/(n-2) * 1' K~ L~ 1 ]

    where ``K~`` is the Gram matrix with a zeroed diagonal.  Every term is
    O(n^2): ``tr(K~ L~)`` is the elementwise sum of the (symmetric) product
    and ``1' K~ L~ 1`` is the dot product of the two row-sum vectors, so no
    n-by-n product is ever formed.
    """
    n = gram_x.shape[0]
    if n < 4:
        raise ValueError("The unbiased HSIC estimator needs at least 4 samples.")
    k = gram_x.copy()
    ell = gram_y.copy()
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(ell, 0.0)
    trace_term = float(np.sum(k * ell))
    sum_k = float(k.sum())
    sum_l = float(ell.sum())
    cross = float(k.sum(axis=1) @ ell.sum(axis=1))
    return (trace_term
            + sum_k * sum_l / ((n - 1) * (n - 2))
            - 2.0 * cross / (n - 2)) / (n * (n - 3))


def linear_cka_unbiased(x, y):
    """Linear CKA built on the unbiased HSIC estimator.

    The shipped ``era.metrics.linear_cka`` uses the biased estimator, which is
    documented there as biased in the high-dimension / low-sample regime.  The
    unbiased version is not a correction applied to published numbers: it is a
    second reading, and the difference between the two is the sensitivity.

    Unlike the biased estimator this one can return small negative values when
    the two spaces are genuinely unrelated - that is the estimator being
    unbiased around zero, not a failure, and it is not clamped away.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear_cka_unbiased expects 2-D (n_samples, dim) matrices.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("linear_cka_unbiased expects the same rows in both matrices.")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    gram_x = x @ x.T
    gram_y = y @ y.T
    hsic_xy = _hsic_unbiased_from_grams(gram_x, gram_y)
    hsic_xx = _hsic_unbiased_from_grams(gram_x, gram_x)
    hsic_yy = _hsic_unbiased_from_grams(gram_y, gram_y)
    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denom <= 0.0:
        return 0.0
    return float(hsic_xy / denom)


def simulate_cka_pair(n_samples, dim, mix, rng):
    """Two representation matrices with a controlled amount of shared signal.

    ``mix = 0`` makes them independent, where the population CKA is exactly
    zero and any positive reading is estimator bias.
    """
    x = rng.standard_normal((n_samples, dim))
    noise = rng.standard_normal((n_samples, dim))
    y = mix * x + np.sqrt(max(1.0 - mix ** 2, 0.0)) * noise
    return x, y


def cka_sensitivity(shapes, mixes=CKA_SIM_MIXES, repeats=2, seed=0, verbose=True):
    """Biased against unbiased CKA at each model's own (n_samples, hidden_size).

    Returns the long table and, per slug, the null-case row: the similarity
    the biased estimator reports for representations that share nothing.
    """
    from era.metrics import linear_cka

    rng = np.random.default_rng(seed)
    rows = []
    for slug, n_samples, dim in shapes:
        if not n_samples or not dim:
            continue
        for mix in mixes:
            biased, unbiased = [], []
            for _ in range(repeats):
                x, y = simulate_cka_pair(int(n_samples), int(dim), mix, rng)
                biased.append(linear_cka(x, y))
                unbiased.append(linear_cka_unbiased(x, y))
            rows.append({
                "slug": slug,
                "n_samples": int(n_samples),
                "hidden_size": int(dim),
                "dim_over_samples": float(dim) / float(n_samples),
                "mix": mix,
                "repeats": repeats,
                "cka_biased": float(np.mean(biased)),
                "cka_biased_std": float(np.std(biased, ddof=1)) if repeats > 1 else 0.0,
                "cka_unbiased": float(np.mean(unbiased)),
                "cka_unbiased_std": (float(np.std(unbiased, ddof=1))
                                     if repeats > 1 else 0.0),
                "bias": float(np.mean(biased) - np.mean(unbiased)),
            })
            if verbose:
                print(f"   {slug:<12} n={int(n_samples):<5} d={int(dim):<5} "
                      f"mix={mix:<5} biased={rows[-1]['cka_biased']:.4f} "
                      f"unbiased={rows[-1]['cka_unbiased']:+.4f}")
    nulls = {r["slug"]: r for r in rows if r["mix"] == 0.0}
    return rows, nulls


def debias_cka(sim_rows, observed_biased):
    """What the unbiased estimator would read where the biased one reads X.

    The simulated points give a monotone map from the biased reading to the
    unbiased one at a fixed ``(n, d)``; this inverts it by interpolation.

    **This is a calibration, not a correction.**  It is derived from isotropic
    Gaussian draws, and real hidden states are strongly anisotropic, so the
    number says what order of magnitude the estimator bias has at this shape
    and this similarity level.  It is never written back into a published
    curve, and no centroid is recomputed from it.
    """
    if not sim_rows or observed_biased is None or not np.isfinite(observed_biased):
        return float("nan")
    ordered = sorted(sim_rows, key=lambda r: r["cka_biased"])
    biased = np.array([r["cka_biased"] for r in ordered], dtype=float)
    unbiased = np.array([r["cka_unbiased"] for r in ordered], dtype=float)
    if biased.size < 2:
        return float("nan")
    return float(np.interp(observed_biased, biased, unbiased))


# ==============================================================================
# (e) CENTROID AGAINST HIDDEN SIZE
# ==============================================================================

def correlate(x_values, y_values):
    """Pearson and Spearman, or an explicit refusal when n is too small.

    With two points a correlation coefficient is +-1 by construction and says
    nothing; reporting it as evidence would be worse than reporting nothing.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    keep = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[keep], y_values[keep]
    n = int(x_values.size)
    if n < 3 or np.unique(x_values).size < 2:
        return {"n": n, "pearson_r": None, "pearson_p": None,
                "spearman_rho": None, "spearman_p": None,
                "p_interpretable": False,
                "note": "n < 3 or no variation: no coefficient is computable"}
    pearson = stats.pearsonr(x_values, y_values)
    spearman = stats.spearmanr(x_values, y_values)
    # With three or four points a perfectly monotone sample gives Spearman
    # rho = 1 and a p-value of essentially zero, which in a table reads as
    # significance and is nothing of the kind: there are only six orderings of
    # three points.  The coefficient still describes the sample; the p-value
    # does not test anything, and is marked rather than printed as if it did.
    interpretable = n >= 5
    note = None if interpretable else (
        f"n = {n}: the coefficients describe the sample, the p-values are not "
        "interpretable")
    return {"n": n,
            "pearson_r": float(pearson[0]), "pearson_p": float(pearson[1]),
            "spearman_rho": float(spearman[0]), "spearman_p": float(spearman[1]),
            "p_interpretable": interpretable, "note": note}


def centroid_size_checks(master, families):
    """Does normalised centroid track hidden size, on the panel and in families?

    ``n_blocks`` and ``n_params`` are reported next to ``hidden_size`` because
    the three move together across this panel: a correlation with one of them
    is not evidence about the others, and only saying so keeps the table from
    reading as three independent findings.
    """
    checks = {}
    for predictor in ("hidden_size", "n_blocks", "n_params"):
        for key in METRIC_KEYS:
            checks[f"panel::{key}~{predictor}"] = correlate(
                master[predictor], master[f"{key}_centroid_norm_mean"])
    for family, slugs in families.items():
        subset = master[master["slug"].isin(slugs)].sort_values("hidden_size")
        for key in METRIC_KEYS:
            checks[f"{family}::{key}~hidden_size"] = correlate(
                subset["hidden_size"], subset[f"{key}_centroid_norm_mean"])
        checks[f"{family}::observations"] = [
            {"slug": row["slug"], "hidden_size": row["hidden_size"],
             "n_blocks": row["n_blocks"],
             **{f"{key}_centroid_norm_mean": row[f"{key}_centroid_norm_mean"]
                for key in METRIC_KEYS}}
            for _, row in subset.iterrows()
        ]
    return checks


# ==============================================================================
# FIGURES
# ==============================================================================

def figure_overlay(models, anisotropy, out_path, tag):
    """Three metric panels plus anisotropy, all on normalised depth."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    cmap = plt.get_cmap("tab20")
    colors = {m["short"]: cmap(i % 20) for i, m in enumerate(models)}

    for ax, key in zip(axes.flat[:3], METRIC_KEYS):
        for model in models:
            depth = model["layers"] / max(len(model["layers"]) - 1, 1)
            data = model[key]
            ax.plot(depth, data["mean"], linewidth=1.8, color=colors[model["short"]],
                    label=f"{model['label']} (n={len(model['seeds'])})")
            ax.fill_between(depth, data["mean"] - data["std"], data["mean"] + data["std"],
                            alpha=0.13, color=colors[model["short"]])
        ax.set_xlabel("Normalised depth  (0 = embedding output, 1 = final block)")
        ax.set_ylabel(METRIC_LABELS[key])
        ax.grid(True, linewidth=0.3, alpha=0.5)

    ax = axes.flat[3]
    for model in models:
        aniso = anisotropy[model["short"]]
        depth = model["layers"] / max(len(model["layers"]) - 1, 1)
        ax.plot(depth, aniso["base_mean"], linewidth=1.8, color=colors[model["short"]],
                label=f"{model['label']} base")
        ax.plot(depth, aniso["ft_mean"], linewidth=1.2, linestyle="--", alpha=0.75,
                color=colors[model["short"]])
    ax.axhline(ANISOTROPY_FLAG, color="black", linewidth=1.0, linestyle=":")
    ax.text(0.01, ANISOTROPY_FLAG, f" anisotropy flag {ANISOTROPY_FLAG}", fontsize=8,
            va="bottom")
    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("Anisotropy (mean pairwise cosine)\nsolid = base, dashed = fine-tuned")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("ERA extended panel - per-layer drift on normalised depth, "
                 f"across-seed mean +- std  (corpus: {tag})", fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure_differential(differentials, out_path, tag):
    """Signed biased-minus-neutral curves: per-seed lines, mean, t interval."""
    slugs = sorted(differentials)
    if not slugs:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.get_cmap("tab10")
    colors = {slug: cmap(i % 10) for i, slug in enumerate(slugs)}

    for ax, key in zip(axes, METRIC_KEYS):
        for slug in slugs:
            entry = differentials[slug]
            agg = entry["aggregates"].get(key)
            if agg is None:
                continue
            depth = np.arange(len(agg["mean"])) / max(len(agg["mean"]) - 1, 1)
            for row in range(agg["per_seed"].shape[0]):
                ax.plot(depth, agg["per_seed"][row], color=colors[slug], alpha=0.35,
                        linewidth=1.0)
            ax.plot(depth, agg["mean"], color=colors[slug], linewidth=2.4,
                    label=f"{entry['label']} (n={agg['n_seeds']} seeds)")
            ax.fill_between(depth, agg["ci_lo"], agg["ci_hi"], alpha=0.15,
                            color=colors[slug])
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel("Normalised depth")
        ax.set_ylabel(f"{METRIC_LABELS[key]}\nbiased - neutral")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Paired differential curves, differenced within seed then aggregated "
                 f"- 95% t interval  (corpus: {tag})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ==============================================================================
# REPORT
# ==============================================================================

def build_report(summary, master, saturation, differentials, checks):
    lines = []
    add = lines.append
    add("# ERA extended cross-architecture comparison")
    add("")
    add(f"_Generated {summary['generated_utc']} - corpus tag "
        f"`{summary['tag']}`, {summary['n_models']} models, "
        f"{summary['n_cells']} cells._")
    add("")
    add("Every input number was measured on the cells listed below, all on "
        f"`{summary['input_device']}`. This aggregation itself is numpy/pandas "
        "only and was run on "
        f"`{summary['analysis_platform']}`; it recomputes no measurement. "
        "Centroids, means and stds are read from the same fields "
        "`11_compare_multiseed.py` reads, through the same loaders.")
    add("")

    add("## Panel")
    add("")
    add("| Model | Tier | Params | Blocks | Hidden | Positional | Seeds |")
    add("|---|---|---:|---:|---:|---|---:|")
    for _, row in master.iterrows():
        add(f"| {row['label']} | {row['tier']} | {int(row['n_params']):,} | "
            f"{int(row['n_blocks'])} | {int(row['hidden_size'])} | "
            f"{row['positional_encoding']} | {int(row['n_seeds'])} |")
    add("")
    add("**Tier B cells carry two seeds.** A two-seed standard deviation is an "
        "estimate from two points; every band and every interval on those rows "
        "is to be read as an order of magnitude, not as a measurement.")
    add("")

    for key in METRIC_KEYS:
        add(f"## Centroid - {key}")
        add("")
        add("| Model | Seeds | centroid | normalised | argmax (mean curve) | "
            "per-seed argmax |")
        add("|---|---:|---:|---:|---:|---|")
        for _, row in master.iterrows():
            add(f"| {row['label']} | {int(row['n_seeds'])} | "
                f"{row[f'{key}_centroid_mean']:.2f} +- "
                f"{row[f'{key}_centroid_std']:.2f} | "
                f"{row[f'{key}_centroid_norm_mean']:.3f} +- "
                f"{row[f'{key}_centroid_norm_std']:.3f} | "
                f"{int(row[f'{key}_argmax_mean_curve'])} | "
                f"{row[f'{key}_argmax_per_seed']} |")
        add("")
    add("The normalised centroid divides by `len(curve) - 1`, so 0 is the "
        "embedding output and 1 the final block. It is the only form in which "
        "models with 7 and 33 curve points can be put in one column.")
    add("")

    add("## Certified ceiling (docs/SATURATION_LEMMA.md)")
    add("")
    add("`ceiling = 2 - anisotropy_base - anisotropy_ft` bounds the relational "
        "curve pointwise and unconditionally. `headroom_used = relational / "
        "ceiling` is how much of the available range the measurement spent.")
    add("")
    add("| Model | layers flagged | max headroom used | median headroom used | "
        "min ceiling |")
    add("|---|---:|---:|---:|---:|")
    grouped = saturation.groupby("slug", sort=False)
    for slug, frame in grouped:
        label = frame["label"].iloc[0]
        flagged = int((frame["saturation_class"] != "neither").sum())
        total = int(len(frame))
        add(f"| {label} | {flagged}/{total} | "
            f"{frame['headroom_used'].max():.3f} | "
            f"{frame['headroom_used'].median():.3f} | "
            f"{frame['ceiling'].min():.4f} |")
    add("")
    add(f"Lemma violations (`headroom_used > 1`): "
        f"**{summary['n_lemma_violations']}**. The bound is pointwise and "
        "unconditional, so any violation is a defect in the artefacts or in "
        "this code, never a result.")
    add("")

    add("## Paired differential curves (reference models)")
    add("")
    add(SIGNED_CURVE_NOTE)
    add("")
    if differentials:
        add("| Model | Metric | Seeds | layers whose 95% interval excludes 0 |")
        add("|---|---|---:|---:|")
        for slug in sorted(differentials):
            entry = differentials[slug]
            for key in METRIC_KEYS:
                agg = entry["aggregates"].get(key)
                if agg is None:
                    continue
                excluded = int(np.sum((agg["ci_lo"] > 0) | (agg["ci_hi"] < 0)))
                add(f"| {entry['label']} | {key} | {agg['n_seeds']} | "
                    f"{excluded}/{len(agg['mean'])} |")
        add("")
        add("The interval is a Student-t interval on three points. Its width is "
            "part of the finding.")
    else:
        add("_No paired cells were found: the differential section is empty._")
    add("")

    add("## CKA sensitivity - biased against unbiased HSIC")
    add("")
    add("`era.metrics.linear_cka` is the standard biased estimator. This panel "
        "sits at hidden sizes 512-1024 against roughly 1300-1500 CKA samples, "
        "the regime where that estimator attributes similarity to "
        "representations that share nothing. The `mix = 0` rows are the null "
        "case, where the population CKA is exactly zero: whatever the biased "
        "estimator reports there is its own floor.")
    add("")
    if summary.get("cka_sensitivity_ran"):
        add("| Model | n | d | d/n | CKA at null (biased) | CKA at null "
            "(unbiased) | max 1-CKA reported | calibrated | inflation |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in master.iterrows():
            if not np.isfinite(row.get("cka_null_biased", float("nan"))):
                continue
            add(f"| {row['label']} | {int(row['n_cka_samples_mean'])} | "
                f"{int(row['hidden_size'])} | "
                f"{row['hidden_size'] / row['n_cka_samples_mean']:.2f} | "
                f"{row['cka_null_biased']:.4f} | {row['cka_null_unbiased']:+.4f} | "
                f"{row['cka_change_max_mean_curve']:.4f} | "
                f"{row['cka_change_max_debiased']:.4f} | "
                f"{row['cka_change_inflation_factor']:.2f}x |")
        add("")
        add("The null columns are the estimator's own floor at that shape: "
            "what it reports for two representations that share nothing. They "
            "are a property of `(n, d)`, **not** a threshold the observed "
            "change is compared against - the bias falls as the true "
            "similarity rises, and every cell in this panel sits above 0.9.")
        add("")
        add("The last two columns are the number that matters. `calibrated` "
            "reads the simulated biased-to-unbiased map at the similarity the "
            "cell actually reports; `inflation` is how much larger the change "
            "would be under the unbiased estimator. **It is a calibration, not "
            "a correction**: it comes from isotropic Gaussian draws while real "
            "hidden states are strongly anisotropic, it is not written back "
            "into any published curve, and no centroid is recomputed from it. "
            "It says what order of magnitude the estimator bias has here, "
            "and - because it grows with `d/n` - whether a depth difference "
            "between a wide model and a narrow one could be the estimator "
            "rather than the model.")
    else:
        add("_The simulation was skipped (`--skip-cka-sim`)._")
    add("")

    add("## Centroid against hidden size")
    add("")
    add("Wider models both carry more estimator bias in CKA and sit at "
        "different depths; the two would look identical in the centroid "
        "column, so the association is checked explicitly rather than left "
        "implicit.")
    add("")
    add("| Comparison | n | Pearson r | p | Spearman rho | p |")
    add("|---|---:|---:|---:|---:|---:|")
    for name, check in sorted(checks.items()):
        if not isinstance(check, dict) or "n" not in check:
            continue
        if check["pearson_r"] is None:
            add(f"| {name} | {check['n']} | - | - | - | - |")
            continue
        p_pearson = (f"{check['pearson_p']:.3f}" if check["p_interpretable"]
                     else "n/a")
        p_spearman = (f"{check['spearman_p']:.3f}" if check["p_interpretable"]
                      else "n/a")
        add(f"| {name} | {check['n']} | {check['pearson_r']:+.3f} | {p_pearson} | "
            f"{check['spearman_rho']:+.3f} | {p_spearman} |")
    add("")
    add("With eleven models on the panel and two or three inside a family, "
        "these coefficients are descriptive. Family p-values are printed as "
        "`n/a`: on three points a perfectly monotone sample gives rho = 1 and "
        "a p-value near zero out of six possible orderings, which is not "
        "evidence of anything. `hidden_size`, `n_blocks` and `n_params` move "
        "together across this panel, so a correlation with one is not "
        "evidence about the others.")
    add("")

    add("## Files")
    add("")
    for name, description in summary["files"].items():
        add(f"- `{name}` - {description}")
    add("")
    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="ERA extended cross-architecture aggregation (script 14)")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--sweep-root", default=None,
                        help="Directory containing <tag>/<slug>/seed_<n>/")
    parser.add_argument("--controls-root", default=None,
                        help="Directory containing control_B_domain/<slug>/seed_<n>/")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--cka-sim-repeats", type=int, default=2,
                        help="Draws per (model, mixing level) in the CKA simulation")
    parser.add_argument("--cka-sim-seed", type=int, default=0)
    parser.add_argument("--skip-cka-sim", action="store_true",
                        help="Skip the CKA sensitivity simulation (the slow part)")
    return parser


def main():
    args = build_parser().parse_args()
    compare = _load_sibling("compare_multiseed", "11_compare_multiseed.py")

    sweep_root, sweep_provenance, sweep_skipped = resolve_root(
        args.sweep_root, SWEEP_WORK_ROOT, SWEEP_RESULTS_ROOT, args.tag, "sweep")
    controls_root, controls_provenance, controls_skipped = resolve_root(
        args.controls_root, CONTROLS_WORK_ROOT, CONTROLS_RESULTS_ROOT,
        NEUTRAL_CONTROL, "controls")
    census = load_census()

    tag_root = Path(sweep_root) / args.tag
    discovered = compare.discover_models(tag_root)

    print("=" * 78)
    print("ERA EXTENDED COMPARISON - script 14")
    print("=" * 78)
    print(f"Sweep    : {tag_root}  ({sweep_provenance})")
    print(f"Controls : {controls_root}  ({controls_provenance})")
    for skipped, what in ((sweep_skipped, "sweep"), (controls_skipped, "controls")):
        if skipped:
            print(f"   note: a working {what} tree exists at {skipped} and was "
                  f"NOT read; pass --{what}-root to use it instead.")
    print(f"Models   : {len(discovered)}")
    print()

    models, anisotropy, seed_dirs_by_slug = [], {}, {}
    for slug, seed_dirs in discovered.items():
        model = compare.load_model(slug, seed_dirs)
        if slug in census.index:
            model["label"] = census.loc[slug, "label"]
        models.append(model)
        seed_dirs_by_slug[slug] = seed_dirs
        frames = load_layer_frames(seed_dirs)
        base = np.vstack([f["anisotropy_base"].to_numpy(dtype=float) for f in frames])
        fine = np.vstack([f["anisotropy_ft"].to_numpy(dtype=float) for f in frames])
        anisotropy[slug] = {"base_mean": base.mean(axis=0), "ft_mean": fine.mean(axis=0),
                            "frames": frames}
        print(f"  {model['label']:<14} seeds={model['seeds']} "
              f"points={len(model['layers'])}")
    models.sort(key=lambda m: (census.loc[m["short"], "n_params"]
                               if m["short"] in census.index else 0))
    print()

    # (e) CKA sensitivity, first: the master table carries its null readings.
    cka_rows, cka_nulls = [], {}
    if not args.skip_cka_sim:
        print("CKA sensitivity simulation (biased vs unbiased HSIC):")
        shapes = []
        for model in models:
            configs = cell_configs(seed_dirs_by_slug[model["short"]])
            samples = [c.get("n_cka_samples") for c in configs if c.get("n_cka_samples")]
            hidden = (census.loc[model["short"], "hidden_size"]
                      if model["short"] in census.index else None)
            if samples and hidden:
                shapes.append((model["short"], int(np.mean(samples)), int(hidden)))
        cka_rows, cka_nulls = cka_sensitivity(
            shapes, repeats=args.cka_sim_repeats, seed=args.cka_sim_seed)
        print()

    # (a) master table.
    sim_by_slug = {}
    for row in cka_rows:
        sim_by_slug.setdefault(row["slug"], []).append(row)
    master = pd.DataFrame([
        master_row(model, seed_dirs_by_slug[model["short"]], census,
                   cka_nulls.get(model["short"]),
                   sim_by_slug.get(model["short"]))
        for model in models
    ])

    # (c) certified ceiling, per layer per seed.
    saturation_records = []
    for model in models:
        saturation_records.extend(ceiling_frame(
            model["short"], model["label"], model["seeds"],
            anisotropy[model["short"]]["frames"], len(model["layers"])))
    saturation = pd.DataFrame(saturation_records)
    violations = lemma_violations(saturation_records)
    flagged = saturation[saturation["saturation_class"] != "neither"]
    master = master.merge(
        pd.DataFrame({
            "slug": [m["short"] for m in models],
            "n_layer_seed_rows_flagged": [
                int((flagged["slug"] == m["short"]).sum()) for m in models],
            "headroom_used_max": [
                float(saturation.loc[saturation["slug"] == m["short"],
                                     "headroom_used"].max()) for m in models],
            "headroom_used_median": [
                float(saturation.loc[saturation["slug"] == m["short"],
                                     "headroom_used"].median()) for m in models],
            "ceiling_min": [
                float(saturation.loc[saturation["slug"] == m["short"],
                                     "ceiling"].min()) for m in models],
        }), on="slug", how="left")

    # (d) paired differentials, reference models only.
    differentials, differential_records = {}, []
    for slug in REFERENCE_SLUGS:
        model = next((m for m in models if m["short"] == slug), None)
        if model is None:
            continue
        per_metric, used_seeds = paired_differences(sweep_root, controls_root,
                                                    args.tag, slug)
        if not used_seeds:
            print(f"[SKIP] no paired neutral cells for {slug}")
            continue
        aggregates = {key: aggregate_differences(per_metric[key]) for key in METRIC_KEYS}
        differentials[slug] = {"label": model["label"], "seeds": used_seeds,
                               "aggregates": aggregates}
        differential_records.extend(differential_rows(
            slug, model["label"], per_metric, aggregates, len(model["layers"])))
    differential = pd.DataFrame(differential_records)

    # (e) centroid against hidden size.
    families = {
        "pythia_family": ["pythia70m", "pythia", "pythia410m"],
        "gpt2_family": ["gpt2", "gpt2medium"],
    }
    checks = centroid_size_checks(master, families)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else (
        COMPARISON_ROOT / f"extended_{args.tag}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "extended_master.csv": "one row per model: architecture, centroids "
                               "(absolute and normalised), argmax, CKA calibration",
        "extended_saturation_per_layer.csv": "per layer and seed: ceiling, "
                                             "headroom_used, saturation_class",
        "extended_overlay_normalised_depth.png": "the three metrics plus "
                                                 "anisotropy on normalised depth",
        "extended_summary.json": "machine-readable summary of everything above",
        "extended_README.md": "this report",
    }
    master.to_csv(out_dir / "extended_master.csv", index=False)
    saturation.to_csv(out_dir / "extended_saturation_per_layer.csv", index=False)
    if not differential.empty:
        differential.to_csv(out_dir / "extended_differential_curves.csv", index=False)
        files["extended_differential_curves.csv"] = (
            "biased minus neutral, differenced within seed then aggregated")
    if cka_rows:
        pd.DataFrame(cka_rows).to_csv(out_dir / "extended_cka_sensitivity.csv",
                                      index=False)
        files["extended_cka_sensitivity.csv"] = (
            "biased and unbiased CKA at each model's own (n, d)")

    figure_overlay(models, anisotropy, out_dir / "extended_overlay_normalised_depth.png",
                   args.tag)
    if figure_differential(differentials, out_dir / "extended_differential.png",
                           args.tag):
        files["extended_differential.png"] = ("signed differential curves with "
                                              "per-seed lines and a t interval")

    devices = sorted({d for row in master["device"] for d in str(row).split("|") if d})
    summary = {
        "_comment": ("Extended cross-architecture aggregation. Reuses the "
                     "loaders of 11_compare_multiseed.py; no measurement is "
                     "recomputed."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "sweep_root": str(tag_root),
        "sweep_provenance": sweep_provenance,
        "sweep_tree_not_read": sweep_skipped,
        "controls_root": str(controls_root),
        "controls_provenance": controls_provenance,
        "controls_tree_not_read": controls_skipped,
        "n_models": int(len(models)),
        "n_cells": int(sum(len(m["seeds"]) for m in models)),
        "input_device": "|".join(devices),
        "analysis_platform": sys.platform,
        "anisotropy_flag": ANISOTROPY_FLAG,
        "centroid_normalisation": "centroid / (len(curve) - 1)",
        "signed_curve_note": SIGNED_CURVE_NOTE,
        "n_lemma_violations": len(violations),
        "lemma_violations": violations[:20],
        "cka_sensitivity_ran": bool(cka_rows),
        "cka_sim_repeats": args.cka_sim_repeats if cka_rows else None,
        "cka_sim_seed": args.cka_sim_seed if cka_rows else None,
        "centroid_size_checks": checks,
        "differential_models": {
            slug: {"label": entry["label"], "seeds": entry["seeds"]}
            for slug, entry in differentials.items()
        },
        "files": files,
        "master_table": json.loads(master.to_json(orient="records")),
    }
    (out_dir / "extended_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "extended_README.md").write_text(
        build_report(summary, master, saturation, differentials, checks) + "\n",
        encoding="utf-8")

    print("=" * 78)
    print(f"Models {summary['n_models']} | cells {summary['n_cells']} | "
          f"lemma violations {summary['n_lemma_violations']}")
    for _, row in master.iterrows():
        print(f"  {row['label']:<14} n_seeds={int(row['n_seeds'])} "
              f"1-CKA centroid={row['cka_change_centroid_norm_mean']:.3f} "
              f"relational={row['relational_centroid_norm_mean']:.3f}")
    print(f"\nArtefacts in: {out_dir}")
    print("Next: python experiments/98_export_results.py --verify")


if __name__ == "__main__":
    main()
