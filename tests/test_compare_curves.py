"""
Tests for experiments/compare_curves.py — the curve-equivalence GATE.

A program meant to block CI must itself be tested: every verdict path (pass,
over-tolerance, misaligned layers, missing column, constant curve, NaN,
invalid thresholds) is exercised here by invoking main() with patched argv.
The script lives outside the package, so it is loaded via importlib.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "compare_curves.py"
_spec = importlib.util.spec_from_file_location("compare_curves", _SCRIPT)
compare_curves = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_curves)

HEADER = "layer,l3_mean,l3_std,per_token_mean,per_token_std,cka\n"


def write_curve(path, rows):
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def run_gate(monkeypatch, ref, cand, *extra):
    monkeypatch.setattr(sys, "argv",
                        ["compare_curves.py", str(ref), str(cand), *extra])
    return compare_curves.main()


ROWS = [
    "0,0.10,0.01,0.20,0.02,0.99\n",
    "1,0.30,0.01,0.40,0.02,0.95\n",
    "2,0.50,0.01,0.60,0.02,0.80\n",
]


def test_identical_curves_pass(tmp_path, monkeypatch):
    ref = write_curve(tmp_path / "ref.csv", ROWS)
    cand = write_curve(tmp_path / "cand.csv", ROWS)
    assert run_gate(monkeypatch, ref, cand) == 0


def test_difference_beyond_tolerance_fails(tmp_path, monkeypatch):
    ref = write_curve(tmp_path / "ref.csv", ROWS)
    cand = write_curve(tmp_path / "cand.csv", [
        "0,0.10,0.01,0.20,0.02,0.99\n",
        "1,0.30,0.01,0.40,0.02,0.95\n",
        "2,0.90,0.01,0.60,0.02,0.80\n",  # l3_mean off by 0.4 >> 0.05
    ])
    assert run_gate(monkeypatch, ref, cand) == 1


def test_small_difference_within_custom_tolerance_passes(tmp_path, monkeypatch):
    ref = write_curve(tmp_path / "ref.csv", ROWS)
    cand = write_curve(tmp_path / "cand.csv", [
        "0,0.11,0.01,0.20,0.02,0.99\n",
        "1,0.31,0.01,0.40,0.02,0.95\n",
        "2,0.51,0.01,0.60,0.02,0.80\n",
    ])
    assert run_gate(monkeypatch, ref, cand, "--max-diff", "0.05") == 0


def test_misaligned_layer_indices_fail(tmp_path, monkeypatch):
    """Same length but different layer indices: positional comparison would
    be meaningless, so the gate must fail rather than compare blindly."""
    ref = write_curve(tmp_path / "ref.csv", ROWS)
    cand = write_curve(tmp_path / "cand.csv", [
        "1,0.10,0.01,0.20,0.02,0.99\n",
        "2,0.30,0.01,0.40,0.02,0.95\n",
        "3,0.50,0.01,0.60,0.02,0.80\n",
    ])
    assert run_gate(monkeypatch, ref, cand) == 1


def test_missing_required_column_fails(tmp_path, monkeypatch):
    ref = write_curve(tmp_path / "ref.csv", ROWS)
    cand = tmp_path / "cand.csv"
    cand.write_text(
        "layer,l3_mean,l3_std\n0,0.10,0.01\n1,0.30,0.01\n2,0.50,0.01\n",
        encoding="utf-8",
    )
    assert run_gate(monkeypatch, ref, cand) == 1


def test_constant_curve_falls_back_to_diff_criterion(tmp_path, monkeypatch):
    """Correlation is undefined for a constant curve: identical constants
    must PASS on the difference criterion alone, not crash or fail."""
    rows = ["0,0.10,0.0,0.10,0.0,1.0\n",
            "1,0.10,0.0,0.10,0.0,1.0\n",
            "2,0.10,0.0,0.10,0.0,1.0\n"]
    ref = write_curve(tmp_path / "ref.csv", rows)
    cand = write_curve(tmp_path / "cand.csv", rows)
    assert run_gate(monkeypatch, ref, cand) == 0


def test_flat_curve_against_trending_curve_fails(tmp_path, monkeypatch):
    """The asymmetric case: a flat reference against a trending candidate is
    a SHAPE mismatch and must FAIL even when every absolute difference is
    within --max-diff.  The diff-only fallback applies only when BOTH curves
    are constant."""
    ref = write_curve(tmp_path / "ref.csv", [
        "0,0.00,0.0,0.10,0.0,1.0\n",
        "1,0.00,0.0,0.10,0.0,1.0\n",
        "2,0.00,0.0,0.10,0.0,1.0\n",
    ])
    cand = write_curve(tmp_path / "cand.csv", [
        "0,0.00,0.0,0.10,0.0,1.0\n",
        "1,0.01,0.0,0.10,0.0,1.0\n",
        "2,0.02,0.0,0.10,0.0,1.0\n",  # l3_mean trends; diffs all <= 0.02
    ])
    assert run_gate(monkeypatch, ref, cand) == 1


def test_cli_process_exit_codes(tmp_path):
    """End-to-end check of the actual entry point (sys.exit(main())) in a
    subprocess: equivalent files -> exit 0, divergent files -> exit 1."""
    import subprocess

    ref = write_curve(tmp_path / "ref.csv", ROWS)
    same = write_curve(tmp_path / "same.csv", ROWS)
    diff = write_curve(tmp_path / "diff.csv", [
        "0,0.10,0.01,0.20,0.02,0.99\n",
        "1,0.30,0.01,0.40,0.02,0.95\n",
        "2,0.90,0.01,0.60,0.02,0.80\n",
    ])
    ok = subprocess.run([sys.executable, str(_SCRIPT), str(ref), str(same)],
                        capture_output=True, text=True)
    assert ok.returncode == 0
    assert "PASS" in ok.stdout
    bad = subprocess.run([sys.executable, str(_SCRIPT), str(ref), str(diff)],
                         capture_output=True, text=True)
    assert bad.returncode == 1
    assert "FAIL" in bad.stdout


def test_nan_values_fail(tmp_path, monkeypatch):
    ref = write_curve(tmp_path / "ref.csv", ROWS)
    cand = write_curve(tmp_path / "cand.csv", [
        "0,nan,0.01,0.20,0.02,0.99\n",
        "1,0.30,0.01,0.40,0.02,0.95\n",
        "2,0.50,0.01,0.60,0.02,0.80\n",
    ])
    assert run_gate(monkeypatch, ref, cand) == 1


def test_invalid_thresholds_are_rejected(tmp_path, monkeypatch):
    """A gate configured with nonsense thresholds must refuse to run:
    argparse.error exits with code 2."""
    ref = write_curve(tmp_path / "ref.csv", ROWS)
    cand = write_curve(tmp_path / "cand.csv", ROWS)
    with pytest.raises(SystemExit):
        run_gate(monkeypatch, ref, cand, "--max-diff", "-1")
    with pytest.raises(SystemExit):
        run_gate(monkeypatch, ref, cand, "--min-corr", "1.5")
