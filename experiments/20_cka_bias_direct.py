#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA - direct biased-vs-unbiased CKA comparison on the measured cells
====================================================================

Script 14 section (e) calibrates the CKA estimator bias by *simulation*: it
draws isotropic Gaussian matrices at each model's own ``(n_samples,
hidden_size)``, mixes two of them to a target population CKA, and reads how
far the biased estimator drifts from the unbiased one.  That calibration
answers "how much would the biased estimator inflate similarity **if the
activations were isotropic Gaussian noise of the nominal dimension**".

They are not.  Since the ``v2_balanced_r2`` sweep writes both estimators for
every layer of every cell (``cka`` and ``cka_unbiased`` in
``layer_curve.csv``), the bias no longer has to be simulated: it can be read
off the measurement itself.  This script does exactly that, and nothing else.

What it reports
---------------
For each cell, with ``d_biased = 1 - cka`` and ``d_unbiased = 1 - cka_unbiased``
(the reported "representational change"), the inflation ratio

    ratio = d_unbiased / d_biased

is the factor by which the headline quantity would grow if the unbiased
estimator were used instead.  ``ratio = 1`` means the choice of estimator is
irrelevant; ``ratio = 1.6`` means the biased estimator understates the change
by 60%.  Two summaries per cell:

* **at the peak layer** - ``argmax(d_biased)``, the layer the master table
  quotes as ``cka_change_max_mean_curve``.  This is the layer the estimator
  question actually bears on, because it is the number the findings quote.
* **median over layers** - a robust summary of the whole curve.

Layers where ``d_biased`` is below ``--min-delta`` are excluded from the
median only.  There ``cka`` is ~1 (typically the embedding output, where the
fine-tune has barely moved the representation), both estimators agree to
within float noise, and their *ratio* is a quotient of two nearly-zero
quantities: numerically meaningless and wildly unstable.  The peak layer is by
construction the largest ``d_biased`` in the cell and is never excluded.  The
count of excluded layers is reported, never silently dropped.

Anisotropy is carried through because it is the explanation, not decoration:
the simulation's isotropic draw is precisely the assumption that fails, and
``anisotropy_base``/``anisotropy_ft`` are what the sweep measured instead.

This script recomputes nothing that the sweep already computed: both CKA
columns are read as written.  It is numpy/pandas only, on CUDA-measured
inputs.

Usage
-----
    python experiments/20_cka_bias_direct.py
    python experiments/20_cka_bias_direct.py --tag v2_balanced_r2 --out DIR
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

DEFAULT_TAG = "v2_balanced_r2"
SWEEP_RESULTS_ROOT = ROOT / "results" / "sweep"
AGGREGATES_ROOT = ROOT / "results" / "aggregates"

# Below this, ``1 - cka`` is float noise and the ratio of the two estimators is
# a quotient of near-zeros. Excluded from the median, reported in the output.
DEFAULT_MIN_DELTA = 1e-3

REQUIRED_COLUMNS = ("layer", "cka", "cka_unbiased")


def latest_aggregates_dir(tag, root=AGGREGATES_ROOT):
    """Newest ``extended_<tag>_<timestamp>`` directory, or None.

    The glob is anchored with a trailing timestamp pattern so that tag
    ``v2_balanced`` does not also match ``v2_balanced_r2_*`` directories.
    """
    candidates = sorted(p for p in Path(root).glob(f"extended_{tag}_*")
                        if p.is_dir() and p.name[len(f"extended_{tag}_"):].startswith("20"))
    return candidates[-1] if candidates else None


def discover_cells(sweep_root, tag):
    """Every ``<slug>/seed_<n>/layer_curve.csv`` under the tag, sorted."""
    tag_root = Path(sweep_root) / tag
    if not tag_root.is_dir():
        raise FileNotFoundError(f"Sweep results directory not found: {tag_root}")
    cells = []
    for curve_path in sorted(tag_root.glob("*/seed_*/layer_curve.csv")):
        cells.append({
            "slug": curve_path.parts[-3],
            "seed": int(curve_path.parts[-2].removeprefix("seed_")),
            "curve_path": curve_path,
        })
    if not cells:
        raise FileNotFoundError(f"No layer_curve.csv found under {tag_root}")
    return cells


def cell_ratios(curve, min_delta):
    """Inflation ratios for one cell's layer curve.

    Returns the peak-layer ratio, the median over eligible layers, and the
    bookkeeping needed to read either number honestly.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in curve.columns]
    if missing:
        raise KeyError(f"layer_curve.csv missing columns {missing}; this "
                       f"comparison needs a sweep that wrote both estimators")

    d_biased = (1.0 - curve["cka"].to_numpy(dtype=float))
    d_unbiased = (1.0 - curve["cka_unbiased"].to_numpy(dtype=float))
    layers = curve["layer"].to_numpy()

    peak_idx = int(np.argmax(d_biased))
    eligible = d_biased >= min_delta
    ratios = np.full(d_biased.shape, np.nan, dtype=float)
    np.divide(d_unbiased, d_biased, out=ratios, where=eligible)
    usable = ratios[eligible]

    return {
        "n_layers": int(len(curve)),
        "n_layers_used": int(eligible.sum()),
        "n_layers_excluded": int((~eligible).sum()),
        "peak_layer": int(layers[peak_idx]),
        "peak_d_biased": float(d_biased[peak_idx]),
        "peak_d_unbiased": float(d_unbiased[peak_idx]),
        "ratio_at_peak": float(d_unbiased[peak_idx] / d_biased[peak_idx]),
        "ratio_median": float(np.median(usable)) if usable.size else float("nan"),
        "ratio_min": float(np.min(usable)) if usable.size else float("nan"),
        "ratio_max": float(np.max(usable)) if usable.size else float("nan"),
        "anisotropy_base_mean": (float(curve["anisotropy_base"].mean())
                                 if "anisotropy_base" in curve.columns else None),
        "anisotropy_ft_mean": (float(curve["anisotropy_ft"].mean())
                               if "anisotropy_ft" in curve.columns else None),
    }


def build_cell_table(cells, min_delta):
    rows = []
    for cell in cells:
        curve = pd.read_csv(cell["curve_path"])
        row = {"slug": cell["slug"], "seed": cell["seed"]}
        row.update(cell_ratios(curve, min_delta))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["slug", "seed"]).reset_index(drop=True)


def build_model_table(cell_table, labels=None):
    """Across-seed mean and std per model, as the master table reports them."""
    grouped = cell_table.groupby("slug", sort=False)
    rows = []
    for slug, block in grouped:
        rows.append({
            "slug": slug,
            "label": (labels or {}).get(slug, slug),
            "n_seeds": int(len(block)),
            "seeds": ";".join(str(s) for s in sorted(block["seed"])),
            "peak_layer_per_seed": ";".join(str(v) for v in block["peak_layer"]),
            "ratio_at_peak_mean": float(block["ratio_at_peak"].mean()),
            "ratio_at_peak_std": float(block["ratio_at_peak"].std(ddof=0)),
            "ratio_median_mean": float(block["ratio_median"].mean()),
            "ratio_median_std": float(block["ratio_median"].std(ddof=0)),
            "ratio_min": float(block["ratio_min"].min()),
            "ratio_max": float(block["ratio_max"].max()),
            "peak_d_biased_mean": float(block["peak_d_biased"].mean()),
            "peak_d_unbiased_mean": float(block["peak_d_unbiased"].mean()),
            "n_layers_excluded_total": int(block["n_layers_excluded"].sum()),
            "anisotropy_base_mean": float(block["anisotropy_base_mean"].mean())
            if block["anisotropy_base_mean"].notna().all() else None,
            "anisotropy_ft_mean": float(block["anisotropy_ft_mean"].mean())
            if block["anisotropy_ft_mean"].notna().all() else None,
        })
    return pd.DataFrame(rows)


def attach_synthetic(model_table, sensitivity_csv):
    """Join the simulated inflation factor this measurement supersedes.

    The simulated factor is read from script 14's CKA sensitivity table by
    interpolating the model's own simulated curve at its measured peak, the
    same construction script 17 reports as ``inflation_factor``.
    """
    if sensitivity_csv is None or not Path(sensitivity_csv).is_file():
        model_table["ratio_simulated_gaussian"] = np.nan
        model_table["simulated_over_measured"] = np.nan
        return model_table, False

    sens = pd.read_csv(sensitivity_csv)
    simulated = {}
    for slug, block in sens.groupby("slug"):
        block = block.sort_values("cka_biased")
        biased = block["cka_biased"].to_numpy(dtype=float)
        unbiased = block["cka_unbiased"].to_numpy(dtype=float)
        row = model_table.loc[model_table["slug"] == slug]
        if row.empty:
            continue
        measured_peak_cka = 1.0 - float(row["peak_d_biased_mean"].iloc[0])
        implied = float(np.interp(measured_peak_cka, biased, unbiased))
        d_biased = 1.0 - measured_peak_cka
        simulated[slug] = ((1.0 - implied) / d_biased) if d_biased > 0 else np.nan

    model_table["ratio_simulated_gaussian"] = model_table["slug"].map(simulated)
    model_table["simulated_over_measured"] = (
        model_table["ratio_simulated_gaussian"] / model_table["ratio_at_peak_mean"])
    return model_table, True


def build_parser():
    parser = argparse.ArgumentParser(
        description="ERA direct biased-vs-unbiased CKA comparison (script 20)")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--sweep-root", default=str(SWEEP_RESULTS_ROOT))
    parser.add_argument("--out", default=None,
                        help="Output directory (default: latest extended_<tag>_* aggregates dir)")
    parser.add_argument("--master", default=None,
                        help="extended_master.csv, for labels (default: alongside --out)")
    parser.add_argument("--sensitivity", default=None,
                        help="extended_cka_sensitivity.csv, for the simulated "
                             "factor this measurement supersedes")
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA,
                        help="Exclude layers with 1-cka below this from the median")
    return parser


def main():
    args = build_parser().parse_args()

    out_dir = Path(args.out) if args.out else latest_aggregates_dir(args.tag)
    if out_dir is None:
        raise SystemExit(
            f"No aggregates directory for tag {args.tag}; run script 14 first "
            f"or pass --out explicitly")
    out_dir.mkdir(parents=True, exist_ok=True)

    master_csv = Path(args.master) if args.master else out_dir / "extended_master.csv"
    sensitivity_csv = (Path(args.sensitivity) if args.sensitivity
                       else out_dir / "extended_cka_sensitivity.csv")

    labels = {}
    if master_csv.is_file():
        master = pd.read_csv(master_csv)
        labels = dict(zip(master["slug"], master["label"]))

    cells = discover_cells(args.sweep_root, args.tag)
    cell_table = build_cell_table(cells, args.min_delta)
    model_table = build_model_table(cell_table, labels)
    model_table, has_synthetic = attach_synthetic(model_table, sensitivity_csv)

    cell_csv = out_dir / "cka_bias_direct_per_cell.csv"
    model_csv = out_dir / "cka_bias_direct_per_model.csv"
    cell_table.to_csv(cell_csv, index=False)
    model_table.to_csv(model_csv, index=False)

    all_peak = cell_table["ratio_at_peak"]
    all_median = cell_table["ratio_median"]
    summary = {
        "_comment": (
            "Direct biased-vs-unbiased CKA comparison read from the measured "
            "cells. Supersedes the isotropic-Gaussian simulation in script 14 "
            "section (e) for the question 'how much does the estimator choice "
            "change the reported 1 - CKA'."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "n_models": int(cell_table["slug"].nunique()),
        "n_cells": int(len(cell_table)),
        "n_layers_total": int(cell_table["n_layers"].sum()),
        "n_layers_excluded_from_median": int(cell_table["n_layers_excluded"].sum()),
        "min_delta": args.min_delta,
        "ratio_at_peak": {
            "min": float(all_peak.min()), "max": float(all_peak.max()),
            "mean": float(all_peak.mean()), "median": float(all_peak.median()),
        },
        "ratio_median_over_layers": {
            "min": float(all_median.min()), "max": float(all_median.max()),
            "mean": float(all_median.mean()), "median": float(all_median.median()),
        },
        "simulated_comparison_available": has_synthetic,
        "files": {
            cell_csv.name: "one row per cell: peak layer, ratio at peak, median ratio",
            model_csv.name: "one row per model: across-seed mean/std of both ratios",
        },
    }
    if has_synthetic:
        sim = model_table["ratio_simulated_gaussian"].dropna()
        summary["ratio_simulated_gaussian"] = {
            "min": float(sim.min()), "max": float(sim.max()),
            "mean": float(sim.mean()),
        }
    (out_dir / "cka_bias_direct_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("=" * 78)
    print("ERA DIRECT CKA ESTIMATOR BIAS - script 20")
    print("=" * 78)
    print(f"Tag {args.tag} | models {summary['n_models']} | cells {summary['n_cells']}")
    print(f"Layers {summary['n_layers_total']} "
          f"({summary['n_layers_excluded_from_median']} excluded from the median: "
          f"1-cka < {args.min_delta:g})\n")
    header = (f"{'model':14s} {'peak ratio (mean+-std)':>24s} "
              f"{'median ratio (mean+-std)':>26s} {'simulated':>10s}")
    print(header)
    print("-" * len(header))
    for _, row in model_table.iterrows():
        sim = row["ratio_simulated_gaussian"]
        sim_txt = "n/a" if pd.isna(sim) else f"{sim:.3f}x"
        print(f"{row['label']:14s} "
              f"{row['ratio_at_peak_mean']:16.4f} +-{row['ratio_at_peak_std']:.4f} "
              f"{row['ratio_median_mean']:18.4f} +-{row['ratio_median_std']:.4f} "
              f"{sim_txt:>10s}")
    print(f"\nRatio at peak across all cells: "
          f"{summary['ratio_at_peak']['min']:.4f} - {summary['ratio_at_peak']['max']:.4f}")
    print(f"Median ratio across all cells:  "
          f"{summary['ratio_median_over_layers']['min']:.4f} - "
          f"{summary['ratio_median_over_layers']['max']:.4f}")
    print(f"\nArtefacts in: {out_dir}")


if __name__ == "__main__":
    main()
