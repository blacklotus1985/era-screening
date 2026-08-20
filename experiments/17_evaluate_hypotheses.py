#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA - hypothesis evaluation: the verdicts behind docs/FINDINGS_extended.md
==========================================================================

Script 14 aggregates; this one judges.  It recomputes every derived number
the findings document quotes and writes them, with a verdict per hypothesis,
to ``results/aggregates/hypothesis_evaluation.json``.

**It reads committed artefacts only** - the sweep cells, the control cells,
the D1-D3 record, the census table, and the CKA sensitivity table produced by
script 14.  It loads no model, trains nothing, and reruns no simulation, so a
reader with the repository can re-derive every verdict without a GPU.

Where an aggregation already exists it is reused rather than reimplemented:
the normalised centroid, the saturation classes, the certified ceiling, the
within-seed pairing and the correlation guard come from
``14_compare_extended.py``, and the D1-D3 decision rules come from
``16_behavioural_checks.py``.  That reuse is deliberate.  A second
implementation of the same rule is a second thing to keep in step, and a
verdict that disagreed with the artefact it judges would be the worst
possible failure mode for this file.

The D1-D3 section goes further and re-applies script 16's rules to the deltas
recorded in its JSON, then checks the result against the verdicts recorded
alongside them.  A mismatch is reported as a defect, never silently
preferred either way.

Verdict vocabulary
------------------
``CONFIRMED``     the preregistered prediction held.
``FALSIFIED``     the preregistered falsification condition was met.
``EQUIVOCAL``     H2-descriptive's 50-70% band, fixed in §4 so it could not
                  be renamed later.
``TRIGGERED``     H2-content: not a pass/fail but a condition that, when met,
                  obliges a restatement (§7.4).
``NULL-HOLDS``    H3, which predicted nothing and gets no p-value.
``NOT-COMPUTED``  the quantity the prediction is about was never measured.

Usage
-----
    python experiments/17_evaluate_hypotheses.py
    python experiments/17_evaluate_hypotheses.py --check-findings
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

RESULTS = ROOT / "results"
SWEEP_ROOT = RESULTS / "sweep"
CONTROLS_ROOT = RESULTS / "controls"
AGGREGATES_ROOT = RESULTS / "aggregates"
CENSUS_CSV = RESULTS / "census" / "model_census.csv"
BEHAVIOURAL_JSON = CONTROLS_ROOT / "behavioural_checks_D1_D3.json"
FINDINGS = ROOT / "docs" / "FINDINGS_extended.md"
DEFAULT_OUT = AGGREGATES_ROOT / "hypothesis_evaluation.json"

DEFAULT_TAG = "v2_balanced"
REFERENCE_SLUGS = ("gptneo", "pythia")
SPOTCHECK_SLUG = "opt125m"

# Preregistered thresholds.  Every one of these is a quotation, not a choice
# made here: H1.2 in §4, H2 in §4 and §7.4, the anisotropy flag in §7.1.
H1_2_MIN_MODELS = 4
H2_DESCRIPTIVE_CENTROID = 0.6
H2_DESCRIPTIVE_CONFIRM = 0.70
H2_DESCRIPTIVE_FALSIFY = 0.50
H2_ANCHORED_MARGIN = 0.10
H2_CONTENT_BAND = 0.15
G2_CENTROID_STD_MAX = 0.08
SPOTCHECK_MIN_CENTROID = 0.85

# A control-C curve is degenerate when all of its change sits on the final
# layer: the centroid is then 1.0 by construction and any margin test built
# on it is close to unfailable.
DEGENERATE_TOL = 1e-9

METRIC_KEYS = ("relational", "per_token", "cka_change")

FAMILIES = {
    "pythia_family": ["pythia70m", "pythia", "pythia410m"],
    "gpt2_family": ["gpt2", "gpt2medium"],
}


def _load_sibling(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, _HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==============================================================================
# READING COMMITTED ARTEFACTS
# ==============================================================================

def seed_dirs(root, slug):
    """The seed cells of one model, in seed order."""
    model_dir = Path(root) / slug
    if not model_dir.is_dir():
        return []
    return sorted(p for p in model_dir.iterdir()
                  if p.is_dir() and (p / "layer_curve.csv").is_file())


def discover_slugs(root):
    return sorted(p.name for p in Path(root).iterdir()
                  if p.is_dir() and seed_dirs(root, p.name))


def cell_centroids(root, slug, extended):
    """``{metric: [normalised centroid per seed]}`` from the cells' run_config.

    The centroids are read, never recomputed: they are the numbers
    ``era.report.save`` wrote and the numbers scripts 11 and 14 report.
    """
    out = {key: [] for key in METRIC_KEYS}
    for seed_dir in seed_dirs(root, slug):
        config = json.loads((seed_dir / "run_config.json").read_text(encoding="utf-8"))
        n_points = int(config["num_layers"])
        for key in METRIC_KEYS:
            out[key].append(extended.normalised_centroid(
                config.get(f"centroid_{key}"), n_points))
    return out


def mean_std(values):
    """``(mean, sample std, n)``; std is 0.0 on a single point, not undefined."""
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), 0
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), std, int(array.size)


# ==============================================================================
# H1
# ==============================================================================

def spearman_base_anisotropy_drift(frame):
    """Spearman rho between base anisotropy and relational drift over layers.

    One cell, one number.  Both variables generally rise with depth, so this
    coefficient conflates the compression mechanism with a shared depth
    trend; that is a property of the preregistered statistic and is recorded
    with the result rather than corrected for here.
    """
    result = stats.spearmanr(frame["anisotropy_base"], frame["l3_mean"])
    return float(result[0])


def h1_1(sweep_root, slugs, labels):
    """Negative in the majority of models, or the prediction fails."""
    per_model = []
    for slug in slugs:
        per_seed = [spearman_base_anisotropy_drift(pd.read_csv(d / "layer_curve.csv"))
                    for d in seed_dirs(sweep_root, slug)]
        mean = float(np.mean(per_seed)) if per_seed else float("nan")
        per_model.append({"slug": slug, "label": labels.get(slug, slug),
                          "n_seeds": len(per_seed), "mean_rho": mean,
                          "per_seed_rho": per_seed, "negative": bool(mean < 0)})
    per_model.sort(key=lambda row: -row["mean_rho"])
    n_negative = sum(row["negative"] for row in per_model)
    n_models = len(per_model)
    return {
        "hypothesis": "H1.1",
        "findings_section": "3.1",
        "prediction": ("per-layer Spearman between base anisotropy and "
                       "relational drift is negative in the majority (>50%) "
                       "of swept models"),
        "status": "CONFIRMED" if n_negative * 2 > n_models else "FALSIFIED",
        "n_models": n_models,
        "n_negative": n_negative,
        "per_model": per_model,
        "note": ("Both variables rise with depth in most architectures, so "
                 "this coefficient conflates the compression mechanism with a "
                 "shared depth trend. Recorded as a weakness of the "
                 "preregistered statistic; not used to reinterpret its "
                 "outcome. H1.3 is the test that isolates the mechanism."),
    }


def h1_2(saturation, labels):
    """At least four panel models carrying a layer at base anisotropy >= 0.95."""
    per_model = []
    for slug, group in saturation.groupby("slug", sort=False):
        n_rows = int((group["anisotropy_base"] >= 0.95).sum())
        if n_rows:
            per_model.append({"slug": slug, "label": labels.get(slug, slug),
                              "n_saturated_layer_seed_rows": n_rows,
                              "n_layer_seed_rows": int(len(group))})
    per_model.sort(key=lambda row: -row["n_saturated_layer_seed_rows"])
    n_models = int(saturation["slug"].nunique())
    return {
        "hypothesis": "H1.2",
        "findings_section": "3.2",
        "prediction": (f"at least {H1_2_MIN_MODELS} of the 11 panel models "
                       "have a layer with base anisotropy >= 0.95"),
        "status": ("CONFIRMED" if len(per_model) >= H1_2_MIN_MODELS
                   else "FALSIFIED"),
        "n_models_with_saturation": len(per_model),
        "n_models": n_models,
        "per_model": per_model,
    }


def h1_3(saturation, labels):
    """Amended H1.3: drift in `both` layers below drift in `neither` layers."""
    per_model = []
    for slug, group in saturation.groupby("slug", sort=False):
        both = group.loc[group["saturation_class"] == "both", "relational"]
        neither = group.loc[group["saturation_class"] == "neither", "relational"]
        if both.empty:
            continue
        entry = {
            "slug": slug, "label": labels.get(slug, slug),
            "n_both": int(len(both)), "mean_relational_both": float(both.mean()),
            "n_neither": int(len(neither)),
            "mean_relational_neither": (float(neither.mean()) if not neither.empty
                                        else float("nan")),
        }
        entry["ratio"] = (entry["mean_relational_neither"] /
                          entry["mean_relational_both"]
                          if entry["mean_relational_both"] > 0 else float("nan"))
        entry["lower_as_predicted"] = bool(
            not neither.empty and both.mean() < neither.mean())
        per_model.append(entry)
    per_model.sort(key=lambda row: -row["ratio"])
    base_only = saturation[saturation["saturation_class"] == "base_only"]
    if not per_model:
        status = "NOT-COMPUTED"
    elif all(row["lower_as_predicted"] for row in per_model):
        status = "CONFIRMED"
    else:
        status = "FALSIFIED"
    return {
        "hypothesis": "H1.3 (amended)",
        "findings_section": "3.3",
        "prediction": ("where both anisotropies are >= 0.95, mean relational "
                       "drift is lower than in that model's neither-class "
                       "layers"),
        "status": status,
        "n_models_evaluated": len(per_model),
        "per_model": per_model,
        "n_base_only_layer_seed_rows": int(len(base_only)),
        "base_only_note": ("A base_only layer would mean the fine-tune opened "
                           "up a near-collinear layer, which §8.4 says is "
                           "reportable rather than anomalous. None occurred."),
    }


def ceiling_summary(saturation, labels):
    """Per-model ceiling table, the certificate the flag does not provide."""
    rows = []
    for slug, group in saturation.groupby("slug", sort=False):
        rows.append({
            "slug": slug, "label": labels.get(slug, slug),
            "n_layer_seed_rows": int(len(group)),
            "n_rows_flagged": int((group["saturation_class"] != "neither").sum()),
            "headroom_used_max": float(group["headroom_used"].max()),
            "headroom_used_median": float(group["headroom_used"].median()),
            "ceiling_min": float(group["ceiling"].min()),
        })
    rows.sort(key=lambda row: row["ceiling_min"])
    return rows


# ==============================================================================
# H2
# ==============================================================================

def h2_descriptive(centroids_by_slug, labels):
    """The 0.6 cut, with the preregistered 50-70% equivocal band."""
    per_model = []
    for slug, values in centroids_by_slug.items():
        mean, std, n_seeds = mean_std(values["cka_change"])
        per_model.append({"slug": slug, "label": labels.get(slug, slug),
                          "n_seeds": n_seeds, "centroid_norm_mean": mean,
                          "centroid_norm_std": std,
                          "above_threshold": bool(mean > H2_DESCRIPTIVE_CENTROID)})
    per_model.sort(key=lambda row: -row["centroid_norm_mean"])
    n_above = sum(row["above_threshold"] for row in per_model)
    fraction = n_above / len(per_model) if per_model else float("nan")
    if fraction >= H2_DESCRIPTIVE_CONFIRM:
        status = "CONFIRMED"
    elif fraction < H2_DESCRIPTIVE_FALSIFY:
        status = "FALSIFIED"
    else:
        status = "EQUIVOCAL"
    return {
        "hypothesis": "H2-descriptive",
        "findings_section": "4.1",
        "prediction": (f"at least {H2_DESCRIPTIVE_CONFIRM:.0%} of swept models "
                       f"have normalised 1-CKA centroid > "
                       f"{H2_DESCRIPTIVE_CENTROID}"),
        "status": status,
        "n_above": n_above, "n_models": len(per_model), "fraction": fraction,
        "per_model": per_model,
        "note": ("An absolute cut on an uncalibrated scale: it establishes "
                 "that centroids cluster late and cannot say what late means "
                 "(§7.4)."),
    }


def is_degenerate_anchor(curve, tol=DEGENERATE_TOL):
    """True when every layer but the last carries no change at all.

    The centroid of such a curve is exactly ``len(curve) - 1``, i.e. 1.0
    normalised, whatever the magnitude of the final layer - so it is a
    property of the construction, not a measurement.
    """
    array = np.asarray(curve, dtype=float)
    if array.size < 2:
        return False
    return bool(np.all(np.abs(array[:-1]) <= tol))


def anchor_curves(control_root, slug):
    """Per-seed ``1 - CKA`` curves of a control cell."""
    return [1.0 - pd.read_csv(d / "layer_curve.csv")["cka"].to_numpy(dtype=float)
            for d in seed_dirs(control_root, slug)]


def h2_anchored(full_centroids, shallow_centroids, shallow_curves, labels):
    """Full-unfreeze strictly earlier than the shallow anchor by > 0.10."""
    per_model = []
    for slug in sorted(shallow_centroids):
        if slug not in full_centroids:
            continue
        full_mean, full_std, full_n = mean_std(full_centroids[slug]["cka_change"])
        shallow_mean, shallow_std, shallow_n = mean_std(
            shallow_centroids[slug]["cka_change"])
        curves = shallow_curves.get(slug, [])
        per_model.append({
            "slug": slug, "label": labels.get(slug, slug),
            "full_unfreeze_centroid": full_mean, "full_unfreeze_std": full_std,
            "full_unfreeze_n_seeds": full_n,
            "anchor_shallow_centroid": shallow_mean,
            "anchor_shallow_std": shallow_std,
            "anchor_shallow_n_seeds": shallow_n,
            "margin": shallow_mean - full_mean,
            "passes": bool((shallow_mean - full_mean) > H2_ANCHORED_MARGIN),
            "anchor_degenerate": bool(curves) and all(
                is_degenerate_anchor(curve) for curve in curves),
        })
    degenerate = bool(per_model) and all(row["anchor_degenerate"]
                                         for row in per_model)
    if not per_model:
        status = "NOT-COMPUTED"
    elif all(row["passes"] for row in per_model):
        status = "CONFIRMED"
    else:
        status = "FALSIFIED"
    return {
        "hypothesis": "H2-anchored",
        "findings_section": "4.2",
        "prediction": ("the full-unfreeze centroid is strictly earlier than "
                       f"anchor_shallow by more than {H2_ANCHORED_MARGIN} in "
                       "normalised depth"),
        "status": status,
        "anchor_degenerate": degenerate,
        "per_model": per_model,
        "qualification": (
            "The anchor is degenerate: control C changes only the final "
            "layer, so its centroid is 1.0 by construction with zero "
            "across-seed variance, and the margin test could only have "
            "failed for a full-unfreeze centroid above 0.90. The pass is "
            "reported as carrying almost no information."
            if degenerate else None),
    }


def h2_content(full_centroids, domain_centroids, labels):
    """Within +-0.15 of anchor_domain obliges the §7.4 restatement."""
    per_model = []
    for slug in sorted(domain_centroids):
        if slug not in full_centroids:
            continue
        full_mean, full_std, full_n = mean_std(full_centroids[slug]["cka_change"])
        domain_mean, domain_std, domain_n = mean_std(
            domain_centroids[slug]["cka_change"])
        difference = full_mean - domain_mean
        per_model.append({
            "slug": slug, "label": labels.get(slug, slug),
            "full_unfreeze_centroid": full_mean, "full_unfreeze_std": full_std,
            "full_unfreeze_n_seeds": full_n,
            "anchor_domain_centroid": domain_mean, "anchor_domain_std": domain_std,
            "anchor_domain_n_seeds": domain_n,
            "difference": difference,
            "within_band": bool(abs(difference) <= H2_CONTENT_BAND),
            "band_fraction_used": abs(difference) / H2_CONTENT_BAND,
        })
    if not per_model:
        status = "NOT-COMPUTED"
    elif all(row["within_band"] for row in per_model):
        status = "TRIGGERED"
    else:
        status = "NOT-TRIGGERED"
    return {
        "hypothesis": "H2-content",
        "findings_section": "4.3",
        "prediction": (f"if the full-unfreeze centroid is within "
                       f"+-{H2_CONTENT_BAND} of anchor_domain, the depth "
                       "profile is a property of the regime and format, not "
                       "of the injected bias, and every depth claim must be "
                       "restated in those terms (§7.4)"),
        "status": status,
        "per_model": per_model,
        "consequence": ("Triggered: docs/FINDINGS_extended.md §4.3 restates "
                        "every depth claim as a claim about the full-unfreeze "
                        "regime on a corpus of this format."
                        if status == "TRIGGERED" else None),
    }


def spotcheck(shallow_centroids, labels, slug=SPOTCHECK_SLUG):
    """Amendment 2 §8.4: is the shallow anchor specific to the reference pair?"""
    if slug not in shallow_centroids:
        return {"status": "NOT-COMPUTED", "slug": slug}
    mean, std, n_seeds = mean_std(shallow_centroids[slug]["cka_change"])
    return {
        "slug": slug, "label": labels.get(slug, slug),
        "findings_section": "4.2",
        "prediction": (f"the partial-unfreeze normalised centroid of {slug} is "
                       f"> {SPOTCHECK_MIN_CENTROID}"),
        "status": "CONFIRMED" if mean > SPOTCHECK_MIN_CENTROID else "FALSIFIED",
        "centroid_norm_mean": mean, "centroid_norm_std": std, "n_seeds": n_seeds,
        "note": ("One seed. The anchor is not architecture-specific, but that "
                 "follows from the construction rather than from the "
                 "measurement."),
    }


# ==============================================================================
# H3
# ==============================================================================

def h3(centroids_by_slug, census, labels):
    """The preregistered null: no direction predicted, no p-value given.

    The CKA inflation factor is attached afterwards, in ``main``: it is read
    at each model's *curve peak*, not at its centroid, and the two must not be
    computed from each other by accident.
    """
    families = {}
    for family, slugs in FAMILIES.items():
        rows = []
        for slug in slugs:
            if slug not in centroids_by_slug:
                continue
            entry = {"slug": slug, "label": labels.get(slug, slug),
                     "hidden_size": int(census.loc[slug, "hidden_size"])}
            for key in METRIC_KEYS:
                mean, std, n_seeds = mean_std(centroids_by_slug[slug][key])
                entry[f"{key}_centroid_norm_mean"] = mean
                entry[f"{key}_centroid_norm_std"] = std
                entry["n_seeds"] = n_seeds
            rows.append(entry)
        rows.sort(key=lambda row: row["hidden_size"])
        families[family] = rows
    return {
        "hypothesis": "H3",
        "findings_section": "5",
        "prediction": ("none - the null. No mechanism was proposed by which "
                       "scale should move the centroid. Cannot be tested "
                       "statistically with 3 and 2 points; no p-value, no "
                       "fitted trend, not the word significant (§4)."),
        "status": "NULL-HOLDS",
        "families": families,
        "observed_direction": h3_direction(families),
        "p_values_withheld": True,
        "p_values_withheld_reason": (
            "Three perfectly ordered points give Spearman rho = 1 out of six "
            "possible orderings. A p-value on that is a number without a "
            "meaning."),
    }


def inflation_factor(sim_rows, slug, extended, peak_cka_change=None):
    """Calibrated inflation of ``1 - CKA``, read at the observed similarity."""
    if not sim_rows or peak_cka_change is None:
        return None
    debiased = 1.0 - extended.debias_cka(sim_rows, 1.0 - peak_cka_change)
    return float(debiased / peak_cka_change) if peak_cka_change > 0 else None


def h3_direction(families):
    """Which way the centroid moves with width, per family and per metric."""
    directions = {}
    for family, rows in families.items():
        if len(rows) < 2:
            directions[family] = "not enough models"
            continue
        for key in METRIC_KEYS:
            values = [row[f"{key}_centroid_norm_mean"] for row in rows]
            spread = max(values) - min(values)
            if all(b < a for a, b in zip(values, values[1:])):
                shape = "decreasing with width"
            elif all(b > a for a, b in zip(values, values[1:])):
                shape = "increasing with width"
            else:
                shape = "not monotone"
            directions[f"{family}::{key}"] = {
                "shape": shape, "values": values, "spread": float(spread)}
    return directions


def panel_correlations(centroids_by_slug, census, extended):
    """Centroid against hidden size, blocks and parameters, across the panel."""
    slugs = [s for s in centroids_by_slug if s in census.index]
    checks = {}
    for predictor in ("hidden_size", "n_blocks", "n_params"):
        x_values = [float(census.loc[slug, predictor]) for slug in slugs]
        for key in METRIC_KEYS:
            y_values = [mean_std(centroids_by_slug[slug][key])[0] for slug in slugs]
            checks[f"panel::{key}~{predictor}"] = extended.correlate(
                x_values, y_values)
    return checks


# ==============================================================================
# D1-D3
# ==============================================================================

def d_checks(behavioural, behav_mod):
    """Re-apply script 16's rules to its recorded deltas, then compare.

    The point is not to recompute what is already written down; it is to
    check that the verdicts stored next to the deltas are the verdicts those
    deltas produce.  A disagreement is a defect in one of the two files and
    is reported as one rather than resolved silently in either direction.
    """
    cells, disagreements = [], []
    for cell in behavioural["cells"]:
        deltas = cell["delta_gap"]
        recomputed = {
            "D1": behav_mod.d1_cell(deltas["biased_gendered"]),
            "D2": behav_mod.d2_cell(deltas["neutral_neutral_pair"]),
            "D3": behav_mod.d3_cell(deltas["biased_gendered"],
                                    deltas["neutral_gendered"]),
        }
        for key, verdict in recomputed.items():
            if verdict["satisfied"] != cell[key]["satisfied"]:
                disagreements.append({
                    "slug": cell["slug"], "seed": cell["seed"], "criterion": key,
                    "recorded": cell[key]["satisfied"],
                    "recomputed": verdict["satisfied"]})
        cells.append({
            "slug": cell["slug"], "label": cell["label"], "seed": cell["seed"],
            "base_gap_gendered": cell["base_gaps"]["gendered"]["gap"],
            "base_gap_neutral_pair": cell["base_gaps"]["neutral"]["gap"],
            "delta_gap": deltas,
            "D1": recomputed["D1"]["satisfied"],
            "D2": recomputed["D2"]["satisfied"],
            "D3": recomputed["D3"]["satisfied"],
            "D3_rule": recomputed["D3"]["rule"],
            "checkpoint_provenance": cell["biased_run"]["provenance"],
        })

    criteria = {}
    for key, section in (("D1", "6.1"), ("D2", "6.3"), ("D3", "6.6")):
        verdicts = [{"satisfied": cell[key], "rule": None} for cell in cells]
        status = behav_mod.criterion_status(verdicts)
        criteria[key] = {
            "hypothesis": key,
            "findings_section": section,
            "status": ("FALSIFIED" if status == "FAIL"
                       else "CONFIRMED" if status == "PASS" else "NOT-COMPUTED"),
            "raw_status": status,
            "n_cells": len(cells),
            "n_satisfied": sum(1 for cell in cells if cell[key] is True),
            "n_falsified": sum(1 for cell in cells if cell[key] is False),
            "n_not_computed": sum(1 for cell in cells if cell[key] is None),
        }
    criteria["D3"]["n_cells_decided_by_ratio"] = sum(
        1 for cell in cells if cell["D3_rule"] == "ratio")
    criteria["D3"]["note"] = (
        "D3's quantitative claim was tested in zero cells: every satisfied "
        "cell was decided by the preregistered degenerate rule, and the "
        "remaining one by D1 having already failed there.")

    provenance = sorted({cell["checkpoint_provenance"] for cell in cells})
    supplementary = [
        {"slug": row["slug"], "seed": row["seed"],
         "regime": row["regime"],
         "delta_gap_gendered": row["delta_gap_gendered"]}
        for row in behavioural["supplementary_control_C"]["cells"]]
    return {
        "criteria": criteria,
        "cells": cells,
        "verdict_disagreements": disagreements,
        "checkpoint_provenance": provenance,
        "bit_identity_preregistration": {
            "expected": "mismatch (retrained twin)",
            "observed": ("bit-identical in all cells"
                         if provenance == ["retrained_bit_identical"]
                         else "|".join(provenance)),
            "reversal": provenance == ["retrained_bit_identical"],
            "findings_section": "6.7",
        },
        "supplementary_control_C": {
            "regime": "PARTIAL_UNFREEZE_LAST_BLOCK",
            "n_cells_positive": sum(1 for row in supplementary
                                    if row["delta_gap_gendered"] > 0),
            "n_cells": len(supplementary),
            "cells": supplementary,
            "findings_section": "6.2",
        },
    }


# ==============================================================================
# GATES
# ==============================================================================

def gate_g2(centroids_by_slug, labels):
    """Across-seed normalised CKA centroid std below 0.08 for every model."""
    per_model = []
    for slug, values in centroids_by_slug.items():
        _, std, n_seeds = mean_std(values["cka_change"])
        per_model.append({"slug": slug, "label": labels.get(slug, slug),
                          "n_seeds": n_seeds, "centroid_norm_std": std,
                          "passes": bool(std < G2_CENTROID_STD_MAX)})
    per_model.sort(key=lambda row: -row["centroid_norm_std"])
    return {
        "gate": "G2",
        "criterion": (f"across-seed normalised CKA centroid std < "
                      f"{G2_CENTROID_STD_MAX} for every completed model"),
        "status": "PASS" if all(row["passes"] for row in per_model) else "FAIL",
        "max_std": max((row["centroid_norm_std"] for row in per_model),
                       default=float("nan")),
        "per_model": per_model,
    }


def gate_g1c():
    """Control A's tolerances, read from the summary the control itself wrote."""
    path = CONTROLS_ROOT / "control_A_serialization" / "summary.json"
    if not path.is_file():
        return {"gate": "G1c", "status": "NOT-COMPUTED"}
    summary = json.loads(path.read_text(encoding="utf-8"))
    checks = summary["checks"]
    return {
        "gate": "G1c",
        "criterion": ("control A neutral within tolerance (hard block); "
                      "controls B and C complete on both reference models"),
        "status": "PASS" if summary.get("all_pass") else "FAIL",
        "control_A": {
            slug: {key: value for key, value in entry.items()
                   if key.startswith("max_") or key == "passes"}
            for slug, entry in checks.items()
        },
    }


# ==============================================================================
# DIFFERENTIAL CURVES
# ==============================================================================

def differential_summary(extended, tag, labels):
    """Sign agreement and interval exclusion per model and metric."""
    rows = []
    for slug in REFERENCE_SLUGS:
        per_metric, seeds = extended.paired_differences(
            SWEEP_ROOT, CONTROLS_ROOT, tag, slug)
        if not seeds:
            continue
        for key in METRIC_KEYS:
            aggregate = extended.aggregate_differences(per_metric[key])
            if aggregate is None:
                continue
            stacked = aggregate["per_seed"]
            signs = np.sign(stacked)
            unanimous = int(np.sum(np.abs(signs.sum(axis=0)) == stacked.shape[0]))
            excludes = int(np.sum((aggregate["ci_lo"] > 0) |
                                  (aggregate["ci_hi"] < 0)))
            rows.append({
                "slug": slug, "label": labels.get(slug, slug), "metric": key,
                "n_seeds": aggregate["n_seeds"], "n_layers": int(len(aggregate["mean"])),
                "n_layers_unanimous_sign": unanimous,
                "n_layers_interval_excludes_zero": excludes,
                "diff_mean_min": float(aggregate["mean"].min()),
                "diff_mean_max": float(aggregate["mean"].max()),
            })
    return {
        "note": ("Differences are taken within seed and only then aggregated. "
                 "No centroid is computed on a signed curve."),
        "rows": rows,
    }


# ==============================================================================
# THE FIGURES THE FINDINGS QUOTES
# ==============================================================================

def figures_quoted(evaluation):
    """Every number the findings prints, formatted exactly as it prints it.

    This is the contract between the document and this script: the
    consistency check walks this mapping, so a figure that changes here and
    not there fails loudly instead of drifting.
    """
    figures = {}
    h1_1_block = evaluation["H1.1"]
    for row in h1_1_block["per_model"]:
        figures[f"H1.1 {row['slug']} mean rho"] = f"{row['mean_rho']:+.3f}"
    figures["H1.1 n_negative"] = f"{h1_1_block['n_negative']} of 11"

    for row in evaluation["H1.3 (amended)"]["per_model"]:
        figures[f"H1.3 {row['slug']} both"] = f"{row['mean_relational_both']:.4f}"
        figures[f"H1.3 {row['slug']} neither"] = (
            f"{row['mean_relational_neither']:.4f}")

    for row in evaluation["ceilings"]:
        figures[f"ceiling {row['slug']} min"] = f"{row['ceiling_min']:.4f}"
        figures[f"ceiling {row['slug']} max headroom"] = (
            f"{row['headroom_used_max']:.3f}")

    for row in evaluation["H2-descriptive"]["per_model"]:
        figures[f"H2 {row['slug']} centroid"] = f"{row['centroid_norm_mean']:.3f}"

    for row in evaluation["H2-content"]["per_model"]:
        figures[f"H2-content {row['slug']} full"] = (
            f"{row['full_unfreeze_centroid']:.4f}")
        figures[f"H2-content {row['slug']} domain"] = (
            f"{row['anchor_domain_centroid']:.4f}")

    # The three deltas the criteria are defined on.  `biased_neutral_pair` -
    # what the biased run did to the substitute pair - is recorded by script
    # 16 but enters no criterion and is not quoted in the findings, so it is
    # not part of this contract.
    for row in evaluation["D1-D3"]["cells"]:
        for name in ("biased_gendered", "neutral_neutral_pair",
                     "neutral_gendered"):
            value = row["delta_gap"].get(name)
            if value is not None:
                figures[f"delta_gap {row['slug']} {row['seed']} {name}"] = (
                    f"{value:.4f}")

    for row in evaluation["D1-D3"]["supplementary_control_C"]["cells"]:
        figures[f"control_C {row['slug']} {row['seed']}"] = (
            f"{row['delta_gap_gendered']:+.4f}")

    figures["G2 max std"] = f"{evaluation['gates']['G2']['max_std']:.4f}"
    return figures


def check_findings(figures, path=FINDINGS):
    """Which quoted figures are absent from the findings document.

    The typographic minus sign the document uses is normalised first: it is a
    house-style choice, not a different number.
    """
    if not Path(path).is_file():
        return None
    text = Path(path).read_text(encoding="utf-8").replace("−", "-")
    return sorted(label for label, value in figures.items() if value not in text)


# ==============================================================================
# MAIN
# ==============================================================================

def latest_aggregates_dir(root=AGGREGATES_ROOT, tag=DEFAULT_TAG):
    candidates = sorted(p for p in Path(root).glob(f"extended_{tag}_*")
                        if p.is_dir())
    return candidates[-1] if candidates else None


def build_parser():
    parser = argparse.ArgumentParser(
        description="ERA hypothesis evaluation (script 17)")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--aggregates", default=None,
                        help="Extended aggregates directory (default: latest)")
    parser.add_argument("--check-findings", action="store_true",
                        help="Also report figures quoted here but absent from "
                             "docs/FINDINGS_extended.md")
    return parser


def main():
    args = build_parser().parse_args()
    extended = _load_sibling("compare_extended", "14_compare_extended.py")
    behav_mod = _load_sibling("behavioural_checks", "16_behavioural_checks.py")

    sweep_root = SWEEP_ROOT / args.tag
    domain_root = CONTROLS_ROOT / "control_B_domain"
    shallow_root = CONTROLS_ROOT / "control_C_localization"
    for root in (sweep_root, domain_root, shallow_root):
        if not root.is_dir():
            raise SystemExit(f"Missing committed artefacts: {root}")

    census = pd.read_csv(CENSUS_CSV).set_index("slug")
    labels = census["label"].to_dict()
    slugs = discover_slugs(sweep_root)

    print("=" * 78)
    print("ERA HYPOTHESIS EVALUATION - script 17")
    print("=" * 78)
    print(f"Sweep    : {sweep_root} ({len(slugs)} models)")
    print(f"Controls : {CONTROLS_ROOT}")
    print()

    full_centroids = {slug: cell_centroids(sweep_root, slug, extended)
                      for slug in slugs}
    domain_centroids = {slug: cell_centroids(domain_root, slug, extended)
                        for slug in discover_slugs(domain_root)}
    shallow_centroids = {slug: cell_centroids(shallow_root, slug, extended)
                         for slug in discover_slugs(shallow_root)}
    shallow_curves = {slug: anchor_curves(shallow_root, slug)
                      for slug in shallow_centroids}

    saturation_records = []
    for slug in slugs:
        cells = seed_dirs(sweep_root, slug)
        frames = [pd.read_csv(d / "layer_curve.csv") for d in cells]
        seeds = [d.name.replace("seed_", "") for d in cells]
        saturation_records.extend(extended.ceiling_frame(
            slug, labels.get(slug, slug), seeds, frames, len(frames[0])))
    saturation = pd.DataFrame(saturation_records)

    aggregates_dir = Path(args.aggregates) if args.aggregates else \
        latest_aggregates_dir(tag=args.tag)
    cka_sensitivity = None
    if aggregates_dir and (aggregates_dir / "extended_cka_sensitivity.csv").is_file():
        cka_sensitivity = pd.read_csv(
            aggregates_dir / "extended_cka_sensitivity.csv")

    cka_calibration = []
    if cka_sensitivity is not None:
        for slug in slugs:
            sim_rows = cka_sensitivity[cka_sensitivity["slug"] == slug].to_dict(
                "records")
            frames = [pd.read_csv(d / "layer_curve.csv")
                      for d in seed_dirs(sweep_root, slug)]
            peak = float(np.mean(
                np.vstack([1.0 - f["cka"].to_numpy(dtype=float) for f in frames]),
                axis=0).max())
            cka_calibration.append({
                "slug": slug, "label": labels.get(slug, slug),
                "hidden_size": int(census.loc[slug, "hidden_size"]),
                "cka_change_max_mean_curve": peak,
                "inflation_factor": inflation_factor(sim_rows, slug, extended,
                                                     peak),
            })

    behavioural = json.loads(BEHAVIOURAL_JSON.read_text(encoding="utf-8"))

    evaluation = {
        "H1.1": h1_1(sweep_root, slugs, labels),
        "H1.2": h1_2(saturation, labels),
        "H1.3 (amended)": h1_3(saturation, labels),
        "ceilings": ceiling_summary(saturation, labels),
        "n_lemma_violations": len(extended.lemma_violations(saturation_records)),
        "H2-descriptive": h2_descriptive(full_centroids, labels),
        "H2-anchored": h2_anchored(full_centroids, shallow_centroids,
                                   shallow_curves, labels),
        "H2-content": h2_content(full_centroids, domain_centroids, labels),
        "spotcheck_opt125m": spotcheck(shallow_centroids, labels),
        "H3": h3(full_centroids, census, labels),
        "panel_correlations": panel_correlations(full_centroids, census, extended),
        "cka_calibration": cka_calibration,
        "differential_curves": differential_summary(extended, args.tag, labels),
        "D1-D3": d_checks(behavioural, behav_mod),
        "gates": {"G1c": gate_g1c(), "G2": gate_g2(full_centroids, labels)},
    }
    for family, rows in evaluation["H3"]["families"].items():
        calibration = {row["slug"]: row["inflation_factor"]
                       for row in cka_calibration}
        for row in rows:
            row["cka_inflation_factor"] = calibration.get(row["slug"])

    figures = figures_quoted(evaluation)
    missing = check_findings(figures) if args.check_findings else None

    payload = {
        "_comment": ("Verdicts behind docs/FINDINGS_extended.md. Recomputed "
                     "from committed artefacts only; no model is loaded and "
                     "no simulation is rerun."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "n_models": len(slugs),
        "n_sweep_cells": int(sum(len(seed_dirs(sweep_root, s)) for s in slugs)),
        "sources": {
            "sweep": str(sweep_root.relative_to(ROOT)),
            "controls": str(CONTROLS_ROOT.relative_to(ROOT)),
            "behavioural": str(BEHAVIOURAL_JSON.relative_to(ROOT)),
            "census": str(CENSUS_CSV.relative_to(ROOT)),
            "cka_sensitivity": (str(aggregates_dir.relative_to(ROOT))
                                if aggregates_dir else None),
        },
        "verdicts": {
            key: evaluation[key]["status"]
            for key in ("H1.1", "H1.2", "H1.3 (amended)", "H2-descriptive",
                        "H2-anchored", "H2-content", "H3")
        },
        "evaluation": evaluation,
        "figures_quoted_in_findings": figures,
        "findings_figures_missing": missing,
    }
    payload["verdicts"].update({
        key: evaluation["D1-D3"]["criteria"][key]["status"]
        for key in ("D1", "D2", "D3")})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                        encoding="utf-8")

    for name, status in payload["verdicts"].items():
        print(f"  {name:<16} {status}")
    print()
    print(f"  gates: G1c {evaluation['gates']['G1c']['status']} | "
          f"G2 {evaluation['gates']['G2']['status']} | "
          f"lemma violations {evaluation['n_lemma_violations']}")
    disagreements = evaluation["D1-D3"]["verdict_disagreements"]
    if disagreements:
        print(f"  [!] {len(disagreements)} D1-D3 verdicts disagree with the "
              "recorded ones; see the JSON.")
    if missing:
        print(f"  [!] {len(missing)} figures quoted here are absent from "
              f"{FINDINGS.name}:")
        for label in missing[:20]:
            print(f"      {label} = {figures[label]}")
    elif missing == []:
        print(f"  every quoted figure appears in {FINDINGS.name}")
    print()
    print(f"Written to {out_path}")

    if disagreements:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
