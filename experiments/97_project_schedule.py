#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA — Project the remaining schedule from a measured pilot cell
===============================================================

The preflight census prints a *coarse* estimate built from timed optimizer
steps.  Gate G1b showed that estimate is unreliable in its parts: on the
laptop it was 2.17x low on training and 5.00x low on screening, because it
modelled forward passes and not checkpoint loading, hashing or CKA.

So a schedule is never extrapolated from the census alone.  This script
takes one **measured** cell on the current device, derives the per-component
correction factors, and reports what the rest of the programme will cost on
*this* machine.  Run it right after the pilot cell, before committing to the
full sweep.

Usage
-----
    python experiments/97_project_schedule.py --pilot-slug pythia70m
    python experiments/97_project_schedule.py --pilot-slug pythia70m --budget-hours 12
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

CENSUS_DIR = ROOT / "results" / "census" / "per_model"
SWEEP_ROOT = ROOT / "era_poc_replication_results_multiseed"


def hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m"


def main():
    parser = argparse.ArgumentParser(description="Project the schedule from a pilot cell")
    parser.add_argument("--pilot-slug", default="pythia70m")
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument("--tag", default="v2_balanced")
    parser.add_argument("--budget-hours", type=float, default=None,
                        help="Stop-and-replan threshold; exits non-zero if exceeded")
    args = parser.parse_args()

    timings_path = SWEEP_ROOT / args.tag / "cell_timings.json"
    if not timings_path.exists():
        raise SystemExit(f"No measured cell at {timings_path}. Run the pilot cell first.")
    cells = json.loads(timings_path.read_text(encoding="utf-8"))["cells"]
    pilot = next((c for c in cells
                  if c["slug"] == args.pilot_slug and c["seed"] == args.pilot_seed), None)
    if pilot is None or pilot.get("train_seconds") is None:
        raise SystemExit(
            f"No completed cell for {args.pilot_slug} seed {args.pilot_seed} "
            f"in {timings_path} (a reused checkpoint does not count as a timing).")

    census_path = CENSUS_DIR / f"{args.pilot_slug}.json"
    if not census_path.exists():
        raise SystemExit(f"No census record at {census_path}.")
    census = json.loads(census_path.read_text(encoding="utf-8"))

    train_ratio = pilot["train_seconds"] / census["est_train_seconds_per_seed"]
    screen_ratio = pilot["screen_seconds"] / census["est_screen_seconds"]

    print("=" * 78)
    print("SCHEDULE PROJECTION FROM A MEASURED PILOT CELL")
    print("=" * 78)
    print(f"Pilot: {pilot['label']} seed {pilot['seed']}")
    print(f"  training  coarse {census['est_train_seconds_per_seed']:8.1f}s  "
          f"measured {pilot['train_seconds']:8.1f}s  ratio {train_ratio:5.2f}x")
    print(f"  screening coarse {census['est_screen_seconds']:8.1f}s  "
          f"measured {pilot['screen_seconds']:8.1f}s  ratio {screen_ratio:5.2f}x")
    print()

    records = []
    for path in sorted(CENSUS_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("est_train_seconds_per_seed") is None:
            continue
        records.append(record)
    records.sort(key=lambda r: r.get("n_params", 0))

    def per_seed(record):
        return (record["est_train_seconds_per_seed"] * train_ratio
                + record["est_screen_seconds"] * screen_ratio)

    print(f"{'model':18s} {'seeds':>5s} {'per seed':>10s} {'sweep total':>12s}")
    print("-" * 50)
    sweep_total = 0.0
    for record in records:
        n_seeds = len(record["seeds"])
        total = per_seed(record) * n_seeds
        sweep_total += total
        print(f"{record['label']:18s} {n_seeds:5d} {hms(per_seed(record)):>10s} "
              f"{hms(total):>12s}")

    # Controls: 2 reference models on the neutral corpus (B, 3 seeds each),
    # the same 2 partial-unfreeze (C, 3 seeds each, cheaper but assumed equal
    # here as an upper bound), plus the OPT-125M spot-check (1 seed), plus
    # Control A which is screening only.
    by_slug = {r["slug"]: r for r in records}
    controls_total = 0.0
    for slug in ("gptneo", "pythia"):
        if slug in by_slug:
            controls_total += per_seed(by_slug[slug]) * 3 * 2      # B and C
            controls_total += by_slug[slug]["est_screen_seconds"] * screen_ratio  # A
    if "opt125m" in by_slug:
        controls_total += per_seed(by_slug["opt125m"])             # spot-check

    print("-" * 50)
    print(f"{'G1c controls':18s} {'':5s} {'':10s} {hms(controls_total):>12s}")
    print(f"{'FULL PROGRAMME':18s} {'':5s} {'':10s} "
          f"{hms(sweep_total + controls_total):>12s}")
    print()
    grand = (sweep_total + controls_total) / 3600.0
    print(f"Projected total on THIS device: {grand:.1f} h")

    if args.budget_hours is not None:
        print(f"Declared budget:                {args.budget_hours:.1f} h")
        if grand > args.budget_hours:
            print()
            print("OVER BUDGET — stop and replan rather than starting a run that "
                  "cannot finish. Options: drop Tier B, drop the third seed of "
                  "Tier A, or raise the budget deliberately.")
            print("=" * 78)
            raise SystemExit(2)
        print("Within budget — proceed.")
    print("=" * 78)


if __name__ == "__main__":
    main()
