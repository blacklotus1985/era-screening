"""Descriptive and paired summaries for ERA paper-metric records."""

from collections import defaultdict

import numpy as np


def metric_row(record):
    metrics = record["metrics"]
    aggregate = metrics["aggregates"]
    target = aggregate["B_T"]
    si = metrics["SI"]["conditional"]
    drift = np.asarray(metrics["G_l"]["drift_1_minus_G"], dtype=np.float64)
    identity = record["identity"]
    return {
        "corpus_name": identity["corpus_name"],
        "regime": identity["regime"],
        "seed": identity["seed"],
        "B": aggregate["B"]["mean"],
        "B_alpha": aggregate["B_alpha"]["mean"],
        "B_k": aggregate["B_k"]["mean"],
        "B_T": target["total"]["mean"],
        "B_between": target["between"]["mean"],
        "B_within": target["within"]["mean"],
        "target_mass_m0": target["target_mass_m0"]["mean"],
        "target_mass_m1": target["target_mass_m1"]["mean"],
        "delta_SI": si["delta_SI"],
        "delta_leadership": si["delta_leadership"],
        "delta_support": si["delta_support"],
        "mean_1_minus_G": float(drift.mean()),
        "max_1_minus_G": float(drift.max()),
        "final_1_minus_G": float(drift[-1]),
        "G_depth_centroid": metrics["G_l"]["normalised_depth_centroid_of_drift"],
        "drift_1_minus_G": [float(value) for value in drift],
    }


def describe(values):
    """Mean and sample SD; undefined inputs remain explicit nulls."""
    values = [None if value is None else float(value) for value in values]
    defined = [value for value in values if value is not None]
    return {
        "values": values,
        "mean": float(np.mean(defined)) if defined else None,
        "sample_std": (
            float(np.std(defined, ddof=1)) if len(defined) > 1 else None
        ),
        "n_total": len(values),
        "n_defined": len(defined),
    }


def paired_difference(full, poc2):
    if full is None or poc2 is None:
        return None
    return float(full - poc2)


def build_summary(records, corpus_names, seeds, experiment_name):
    """Summarise raw conditions and paired FULL-minus-POC2 differences."""
    rows = [metric_row(record) for record in records]
    excluded = {"corpus_name", "regime", "seed", "drift_1_minus_G"}
    scalar_names = [key for key in rows[0] if key not in excluded]

    groups = defaultdict(list)
    for row in rows:
        groups[(row["corpus_name"], row["regime"])].append(row)
    by_condition = []
    for (corpus_name, regime), group in sorted(groups.items()):
        group.sort(key=lambda row: row["seed"])
        curves = np.asarray([row["drift_1_minus_G"] for row in group])
        by_condition.append({
            "corpus_name": corpus_name,
            "regime": regime,
            "seeds": [row["seed"] for row in group],
            "metrics": {
                name: describe([row[name] for row in group])
                for name in scalar_names
            },
            "drift_1_minus_G_by_layer": [
                {"layer": layer, **describe(curves[:, layer])}
                for layer in range(curves.shape[1])
            ],
        })

    paired = []
    for corpus_name in corpus_names:
        corpus_rows = [row for row in rows if row["corpus_name"] == corpus_name]
        indexed = {(row["regime"], row["seed"]): row for row in corpus_rows}
        seed_rows = []
        for seed in seeds:
            poc2, full = indexed[("poc2", seed)], indexed[("full", seed)]
            curve_difference = np.asarray(full["drift_1_minus_G"]) - np.asarray(
                poc2["drift_1_minus_G"]
            )
            seed_rows.append({
                "seed": seed,
                "full_minus_poc2": {
                    name: paired_difference(full[name], poc2[name])
                    for name in scalar_names
                },
                "drift_1_minus_G_by_layer": [
                    float(value) for value in curve_difference
                ],
            })
        paired.append({
            "corpus_name": corpus_name,
            "direction": "FULL minus POC2; positive means larger under FULL.",
            "by_seed": seed_rows,
            "aggregate": {
                name: describe([
                    row["full_minus_poc2"][name] for row in seed_rows
                ])
                for name in scalar_names
            },
            "drift_1_minus_G_by_layer": [
                {
                    "layer": layer,
                    **describe([
                        row["drift_1_minus_G_by_layer"][layer]
                        for row in seed_rows
                    ]),
                }
                for layer in range(len(seed_rows[0]["drift_1_minus_G_by_layer"]))
            ],
        })

    return {
        "complete": True,
        "experiment": experiment_name,
        "design": "2 regimes x 2 corpora x 3 paired seeds",
        "statement": (
            "Observables and paired differences only; no Alignment Score and "
            "no automatic shallow/deep label."
        ),
        "cells": sorted(
            rows,
            key=lambda row: (row["corpus_name"], row["regime"], row["seed"]),
        ),
        "by_condition": by_condition,
        "paired_comparisons": paired,
    }
