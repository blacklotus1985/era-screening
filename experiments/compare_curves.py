#!/usr/bin/env python
"""
Compare two layer_curve.csv files (e.g. a stored v1 run vs. a fresh v2 rerun)
and act as an automatic equivalence GATE, not just a diagnostic printout.

For every shared numeric column it reports max absolute difference and
Pearson correlation across layers, then verdicts PASS/FAIL against explicit
tolerances.  Exit code 0 = equivalent within tolerance, 1 = not equivalent
(or files not comparable), so the script can gate CI or a pre-commit check.

Usage
-----
    python experiments/compare_curves.py REFERENCE.csv CANDIDATE.csv \
        [--max-diff 0.05] [--min-corr 0.99] [--columns l3_mean per_token_mean cka]

Defaults: --max-diff 0.05, --min-corr 0.99, columns = the three v1 curve
columns every ERA layer_curve.csv must have.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

DEFAULT_COLUMNS = ["l3_mean", "per_token_mean", "cka"]


def read_curve(path: Path) -> dict:
    """Read a layer_curve.csv into {column_name: np.ndarray}."""
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty.")
    return {
        col: np.array([float(r[col]) for r in rows])
        for col in rows[0].keys()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Equivalence gate for ERA layer curves")
    parser.add_argument("reference", help="layer_curve.csv of the reference run")
    parser.add_argument("candidate", help="layer_curve.csv of the run to check")
    parser.add_argument("--max-diff", type=float, default=0.05,
                        help="max allowed absolute per-layer difference (default 0.05)")
    parser.add_argument("--min-corr", type=float, default=0.99,
                        help="min required Pearson correlation (default 0.99)")
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS,
                        help=f"columns that MUST exist and pass (default: {DEFAULT_COLUMNS})")
    args = parser.parse_args()

    # A gate with nonsensical thresholds would silently pass everything (or
    # nothing): refuse them up front.
    if not np.isfinite(args.max_diff) or args.max_diff < 0:
        parser.error(f"--max-diff must be a non-negative finite number, got {args.max_diff}")
    if not np.isfinite(args.min_corr) or not (-1.0 <= args.min_corr <= 1.0):
        parser.error(f"--min-corr must be in [-1, 1], got {args.min_corr}")

    ref = read_curve(Path(args.reference))
    cand = read_curve(Path(args.candidate))

    failures = []

    # Layer indices must be identical, not merely same-length: positional
    # comparison of misaligned layers would be meaningless.
    if "layer" not in ref or "layer" not in cand:
        failures.append("missing 'layer' column in one of the files")
    elif not np.array_equal(ref["layer"], cand["layer"]):
        failures.append(
            f"layer indices differ (reference {ref.get('layer')}, candidate {cand.get('layer')})"
        )

    for col in args.columns:
        if col not in ref or col not in cand:
            failures.append(f"required column {col!r} missing "
                            f"(reference has it: {col in ref}, candidate: {col in cand})")

    print(f"{'column':<16} {'max |diff|':>12} {'correlation':>12}  verdict")
    print("-" * 56)
    if not failures:
        for col in args.columns:
            a, b = ref[col], cand[col]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                failures.append(f"{col}: contains NaN or infinite values")
                print(f"{col:<16} {'NaN/inf':>12} {'n/a':>12}  FAIL")
                continue
            max_diff = float(np.max(np.abs(a - b)))
            const_a = np.std(a) < 1e-12
            const_b = np.std(b) < 1e-12
            if const_a and const_b:
                # BOTH curves constant: correlation is undefined, fall back to
                # the difference criterion alone.
                ok = max_diff <= args.max_diff
                corr_str = "n/a"
            elif const_a or const_b:
                # Exactly ONE curve constant: the shapes differ by definition
                # (flat vs. trending), so this cannot count as equivalent even
                # when the absolute differences are small.
                ok = False
                corr_str = "n/a"
                failures.append(
                    f"{col}: one curve is constant and the other is not "
                    f"(flat-vs-trend shape mismatch, max|diff|={max_diff:.6f})"
                )
                print(f"{col:<16} {max_diff:>12.6f} {corr_str:>12}  FAIL")
                continue
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
                ok = max_diff <= args.max_diff and corr >= args.min_corr
                corr_str = f"{corr:.4f}"
            verdict = "ok" if ok else "FAIL"
            if not ok:
                failures.append(
                    f"{col}: max|diff|={max_diff:.6f} (limit {args.max_diff}), "
                    f"corr={corr_str} (min {args.min_corr})"
                )
            print(f"{col:<16} {max_diff:>12.6f} {corr_str:>12}  {verdict}")

    print()
    if failures:
        print("RESULT: FAIL — curves are NOT equivalent within tolerance:")
        for f in failures:
            print(f"  - {f}")
        print("\nSmall diffs where v2 fixed token-ID / re-tokenisation bugs are")
        print("expected; investigate anything beyond tolerance before trusting v2.")
        return 1

    print(f"RESULT: PASS — equivalent within max_diff={args.max_diff}, "
          f"min_corr={args.min_corr}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
