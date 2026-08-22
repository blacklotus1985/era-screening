#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA - centroid comparison between two sweep tags
================================================

``v2_balanced_r2`` re-measured the whole panel after two changes that could
each move the headline number: the exact top-k union (the earlier run took an
approximate union) and a different GPU (A100 rather than the mixed devices the
first sweep used).  It also completed the panel to three seeds per model,
where the first sweep left five cells at two.

This script asks the only question that matters for the main result: **did the
normalised centroid move?**  Per model and per metric it puts the two tags'
across-seed mean and std side by side and reports the difference.

The normalised centroid is the comparison to make, not the absolute one:
absolute depths are not comparable across architectures, and the whole panel
claim is about *shape* on a normalised depth axis.  The difference is reported
in units of the reference tag's own across-seed std as well as in absolute
depth, because "moved by 0.01" means something quite different for a model
whose seeds scatter by 0.002 than for one whose seeds scatter by 0.05.

The seed counts are carried through and printed.  A model that went from two
seeds to three has a mean over a different sample, and its std is not the same
estimator; this is reported, not smoothed over.

Reads two ``extended_master.csv`` files written by script 14.  Recomputes
nothing.  Neither tag's results are modified.

Usage
-----
    python experiments/21_compare_tags.py --baseline DIR --candidate DIR
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

AGGREGATES_ROOT = ROOT / "results" / "aggregates"
DEFAULT_BASELINE_TAG = "v2_balanced"
DEFAULT_CANDIDATE_TAG = "v2_balanced_r2"

METRICS = ("relational", "per_token", "cka_change")
METRIC_LABELS = {
    "relational": "Relational drift",
    "per_token": "Per-token drift",
    "cka_change": "1 - linear CKA",
}


def aggregates_dir_for(tag, root=AGGREGATES_ROOT):
    """Newest ``extended_<tag>_<timestamp>`` directory for exactly this tag.

    The timestamp guard keeps tag ``v2_balanced`` from matching the
    ``v2_balanced_r2_*`` directories, which plain globbing would sort last and
    silently pick as "latest".
    """
    prefix = f"extended_{tag}_"
    candidates = sorted(p for p in Path(root).glob(f"{prefix}*")
                        if p.is_dir() and p.name[len(prefix):].startswith("20"))
    return candidates[-1] if candidates else None


def load_master(directory, tag):
    path = Path(directory) / "extended_master.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No extended_master.csv for {tag} in {directory}")
    return pd.read_csv(path)


def compare(baseline, candidate, baseline_tag, candidate_tag):
    """One row per (model, metric) with both tags' centroid statistics."""
    base_by_slug = baseline.set_index("slug")
    cand_by_slug = candidate.set_index("slug")
    shared = [s for s in cand_by_slug.index if s in base_by_slug.index]
    only_baseline = sorted(set(base_by_slug.index) - set(cand_by_slug.index))
    only_candidate = sorted(set(cand_by_slug.index) - set(base_by_slug.index))

    rows = []
    for slug in shared:
        b, c = base_by_slug.loc[slug], cand_by_slug.loc[slug]
        for metric in METRICS:
            mean_col = f"{metric}_centroid_norm_mean"
            std_col = f"{metric}_centroid_norm_std"
            b_mean, b_std = float(b[mean_col]), float(b[std_col])
            c_mean, c_std = float(c[mean_col]), float(c[std_col])
            delta = c_mean - b_mean
            # Pooled scatter: neither tag's std alone is the right yardstick
            # when the seed counts differ, and one of them can be ~0.
            pooled = float(np.sqrt((b_std ** 2 + c_std ** 2) / 2.0))
            rows.append({
                "slug": slug,
                "label": c["label"],
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                f"{baseline_tag}_n_seeds": int(b["n_seeds"]),
                f"{candidate_tag}_n_seeds": int(c["n_seeds"]),
                f"{baseline_tag}_mean": b_mean,
                f"{baseline_tag}_std": b_std,
                f"{candidate_tag}_mean": c_mean,
                f"{candidate_tag}_std": c_std,
                "delta": delta,
                "abs_delta": abs(delta),
                "pooled_std": pooled,
                "delta_over_pooled_std": (abs(delta) / pooled
                                          if pooled > 0 else float("nan")),
            })
    return pd.DataFrame(rows), only_baseline, only_candidate


def build_parser():
    parser = argparse.ArgumentParser(
        description="ERA centroid comparison between two sweep tags (script 21)")
    parser.add_argument("--baseline-tag", default=DEFAULT_BASELINE_TAG)
    parser.add_argument("--candidate-tag", default=DEFAULT_CANDIDATE_TAG)
    parser.add_argument("--baseline", default=None,
                        help="Baseline aggregates directory (default: latest for the tag)")
    parser.add_argument("--candidate", default=None,
                        help="Candidate aggregates directory (default: latest for the tag)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: the candidate directory)")
    return parser


def main():
    args = build_parser().parse_args()

    baseline_dir = Path(args.baseline) if args.baseline else aggregates_dir_for(args.baseline_tag)
    candidate_dir = Path(args.candidate) if args.candidate else aggregates_dir_for(args.candidate_tag)
    if baseline_dir is None:
        raise SystemExit(f"No aggregates directory for baseline tag {args.baseline_tag}")
    if candidate_dir is None:
        raise SystemExit(f"No aggregates directory for candidate tag {args.candidate_tag}")

    baseline = load_master(baseline_dir, args.baseline_tag)
    candidate = load_master(candidate_dir, args.candidate_tag)
    table, only_baseline, only_candidate = compare(
        baseline, candidate, args.baseline_tag, args.candidate_tag)

    out_dir = Path(args.out) if args.out else candidate_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"centroid_shift_{args.baseline_tag}_vs_{args.candidate_tag}.csv"
    table.to_csv(csv_path, index=False)

    per_metric = {}
    for metric in METRICS:
        block = table[table["metric"] == metric]
        per_metric[metric] = {
            "max_abs_delta": float(block["abs_delta"].max()),
            "max_abs_delta_model": str(block.loc[block["abs_delta"].idxmax(), "label"]),
            "mean_abs_delta": float(block["abs_delta"].mean()),
            "median_abs_delta": float(block["abs_delta"].median()),
            "n_models_over_0_05": int((block["abs_delta"] > 0.05).sum()),
        }

    seed_changed = table[
        table[f"{args.baseline_tag}_n_seeds"] != table[f"{args.candidate_tag}_n_seeds"]
    ]["label"].unique().tolist()

    summary = {
        "_comment": (
            "Normalised-centroid comparison between two sweep tags. Reports "
            "how far the main result moved under the exact top-k union, the "
            "GPU change and the completion to three seeds per model."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_tag": args.baseline_tag,
        "candidate_tag": args.candidate_tag,
        "baseline_dir": str(baseline_dir.relative_to(ROOT) if baseline_dir.is_absolute()
                            and ROOT in baseline_dir.parents else baseline_dir),
        "candidate_dir": str(candidate_dir.relative_to(ROOT) if candidate_dir.is_absolute()
                             and ROOT in candidate_dir.parents else candidate_dir),
        "n_models_compared": int(table["slug"].nunique()),
        "models_only_in_baseline": only_baseline,
        "models_only_in_candidate": only_candidate,
        "models_with_changed_seed_count": seed_changed,
        "per_metric": per_metric,
        "files": {csv_path.name: "one row per (model, metric): both tags' "
                                 "normalised centroid mean/std and the shift"},
    }
    (out_dir / f"centroid_shift_{args.baseline_tag}_vs_{args.candidate_tag}.json"
     ).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("=" * 78)
    print("ERA CENTROID SHIFT - script 21")
    print("=" * 78)
    print(f"{args.baseline_tag}  <-  {baseline_dir}")
    print(f"{args.candidate_tag}  <-  {candidate_dir}")
    print(f"Models compared: {summary['n_models_compared']}")
    if seed_changed:
        print(f"Seed count changed for: {', '.join(seed_changed)}")
    for metric in METRICS:
        block = table[table["metric"] == metric].sort_values("abs_delta", ascending=False)
        print(f"\n--- {METRIC_LABELS[metric]} (normalised centroid) ---")
        header = (f"{'model':14s} {'seeds':>7s} "
                  f"{args.baseline_tag:>22s} {args.candidate_tag:>22s} {'delta':>9s}")
        print(header)
        print("-" * len(header))
        for _, row in block.iterrows():
            seeds = f"{row[args.baseline_tag + '_n_seeds']}->{row[args.candidate_tag + '_n_seeds']}"
            print(f"{row['label']:14s} {seeds:>7s} "
                  f"{row[args.baseline_tag + '_mean']:14.4f} +-{row[args.baseline_tag + '_std']:.4f} "
                  f"{row[args.candidate_tag + '_mean']:14.4f} +-{row[args.candidate_tag + '_std']:.4f} "
                  f"{row['delta']:+9.4f}")
        stats = per_metric[metric]
        print(f"  max |delta| = {stats['max_abs_delta']:.4f} ({stats['max_abs_delta_model']}), "
              f"median |delta| = {stats['median_abs_delta']:.4f}")

    print(f"\nArtefacts in: {out_dir}")


if __name__ == "__main__":
    main()
