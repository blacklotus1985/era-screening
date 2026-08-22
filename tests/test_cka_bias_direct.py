import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parents[1]
SCRIPT = _ROOT / "experiments" / "20_cka_bias_direct.py"
spec = importlib.util.spec_from_file_location("cka_bias_direct", SCRIPT)
cka_bias = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cka_bias)


def _curve(cka, cka_unbiased, layers=None):
    n = len(cka)
    return pd.DataFrame({
        "layer": layers if layers is not None else list(range(n)),
        "cka": cka,
        "cka_unbiased": cka_unbiased,
    })


# ---------------------------------------------------------------------------
# The ratio itself
# ---------------------------------------------------------------------------

def test_identical_estimators_give_a_ratio_of_exactly_one():
    curve = _curve([0.9, 0.8, 0.5], [0.9, 0.8, 0.5])
    out = cka_bias.cell_ratios(curve, cka_bias.DEFAULT_MIN_DELTA)
    assert out["ratio_at_peak"] == pytest.approx(1.0)
    assert out["ratio_median"] == pytest.approx(1.0)


def test_the_peak_is_the_largest_one_minus_cka_not_the_last_layer():
    # The deepest layer is not the most changed one here.
    curve = _curve([0.9, 0.2, 0.7], [0.9, 0.1, 0.7])
    out = cka_bias.cell_ratios(curve, cka_bias.DEFAULT_MIN_DELTA)
    assert out["peak_layer"] == 1
    assert out["peak_d_biased"] == pytest.approx(0.8)
    assert out["ratio_at_peak"] == pytest.approx(0.9 / 0.8)


def test_the_ratio_is_unbiased_over_biased_in_that_order():
    """An unbiased estimate that reports *more* change gives a ratio > 1."""
    curve = _curve([0.5], [0.4])
    out = cka_bias.cell_ratios(curve, 1e-9)
    assert out["ratio_at_peak"] == pytest.approx(0.6 / 0.5)
    assert out["ratio_at_peak"] > 1.0


# ---------------------------------------------------------------------------
# The near-zero guard
# ---------------------------------------------------------------------------

def test_layers_where_cka_is_essentially_one_are_excluded_from_the_median():
    """Their ratio is a quotient of two near-zeros and means nothing."""
    curve = _curve([1.0 - 1e-9, 0.5, 0.4], [1.0 - 9e-9, 0.4, 0.3])
    out = cka_bias.cell_ratios(curve, cka_bias.DEFAULT_MIN_DELTA)
    assert out["n_layers"] == 3
    assert out["n_layers_used"] == 2
    assert out["n_layers_excluded"] == 1
    # 9.0 from the degenerate layer would have dominated any summary.
    assert out["ratio_max"] < 2.0


def test_an_excluded_layer_is_reported_not_silently_dropped():
    curve = _curve([1.0, 0.5], [1.0, 0.4])
    out = cka_bias.cell_ratios(curve, cka_bias.DEFAULT_MIN_DELTA)
    assert out["n_layers"] == out["n_layers_used"] + out["n_layers_excluded"]
    assert out["n_layers_excluded"] == 1


def test_the_peak_layer_is_never_excluded_even_if_the_whole_cell_is_flat():
    """The peak is the cell's largest 1-cka, so the guard must not eat it."""
    curve = _curve([1.0 - 1e-7, 1.0 - 1e-8], [1.0 - 2e-7, 1.0 - 1e-8])
    out = cka_bias.cell_ratios(curve, cka_bias.DEFAULT_MIN_DELTA)
    assert out["peak_layer"] == 0
    assert np.isfinite(out["ratio_at_peak"])
    assert np.isnan(out["ratio_median"])  # nothing eligible, reported as NaN


# ---------------------------------------------------------------------------
# Contract with the sweep
# ---------------------------------------------------------------------------

def test_a_sweep_without_the_unbiased_column_is_a_loud_failure():
    """v2_balanced has no cka_unbiased; this comparison must refuse it."""
    curve = pd.DataFrame({"layer": [0], "cka": [0.9]})
    with pytest.raises(KeyError, match="cka_unbiased"):
        cka_bias.cell_ratios(curve, cka_bias.DEFAULT_MIN_DELTA)


def test_discovery_finds_every_seed_of_every_model(tmp_path):
    for slug in ("gptneo", "pythia"):
        for seed in (42, 43):
            cell = tmp_path / "tagx" / slug / f"seed_{seed}"
            cell.mkdir(parents=True)
            _curve([0.9, 0.5], [0.9, 0.4]).to_csv(cell / "layer_curve.csv",
                                                  index=False)
    cells = cka_bias.discover_cells(tmp_path, "tagx")
    assert len(cells) == 4
    assert {c["slug"] for c in cells} == {"gptneo", "pythia"}
    assert {c["seed"] for c in cells} == {42, 43}


def test_a_missing_tag_is_an_error_not_an_empty_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        cka_bias.discover_cells(tmp_path, "no_such_tag")


def test_the_latest_aggregates_glob_does_not_match_the_r2_tag(tmp_path):
    """`extended_v2_balanced_*` must not sort `..._r2_*` in as its latest."""
    (tmp_path / "extended_v2_balanced_20260820T000000Z").mkdir()
    (tmp_path / "extended_v2_balanced_r2_20260822T000000Z").mkdir()
    found = cka_bias.latest_aggregates_dir("v2_balanced", root=tmp_path)
    assert found.name == "extended_v2_balanced_20260820T000000Z"
    found_r2 = cka_bias.latest_aggregates_dir("v2_balanced_r2", root=tmp_path)
    assert found_r2.name == "extended_v2_balanced_r2_20260822T000000Z"


# ---------------------------------------------------------------------------
# Across-seed aggregation
# ---------------------------------------------------------------------------

def test_the_model_table_averages_across_seeds():
    cells = pd.DataFrame([
        {"slug": "m", "seed": 42, "ratio_at_peak": 1.00, "ratio_median": 1.02,
         "ratio_min": 1.0, "ratio_max": 1.05, "peak_layer": 3,
         "peak_d_biased": 0.5, "peak_d_unbiased": 0.5,
         "n_layers_excluded": 1, "anisotropy_base_mean": 0.6,
         "anisotropy_ft_mean": 0.6},
        {"slug": "m", "seed": 43, "ratio_at_peak": 1.02, "ratio_median": 1.04,
         "ratio_min": 1.0, "ratio_max": 1.07, "peak_layer": 3,
         "peak_d_biased": 0.5, "peak_d_unbiased": 0.5,
         "n_layers_excluded": 1, "anisotropy_base_mean": 0.6,
         "anisotropy_ft_mean": 0.6},
    ])
    table = cka_bias.build_model_table(cells, {"m": "Model M"})
    row = table.iloc[0]
    assert row["label"] == "Model M"
    assert row["n_seeds"] == 2
    assert row["ratio_at_peak_mean"] == pytest.approx(1.01)
    assert row["ratio_median_mean"] == pytest.approx(1.03)
    assert row["n_layers_excluded_total"] == 2


# ---------------------------------------------------------------------------
# The committed result
# ---------------------------------------------------------------------------

def _committed_summary():
    root = _ROOT / "results" / "aggregates"
    candidates = sorted(root.glob("extended_v2_balanced_r2_*/"
                                  "cka_bias_direct_summary.json"))
    return candidates[-1] if candidates else None


def test_the_committed_measurement_covers_the_whole_panel():
    path = _committed_summary()
    if path is None:
        pytest.skip("script 20 has not been run yet")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["tag"] == "v2_balanced_r2"
    assert summary["n_models"] == 11
    assert summary["n_cells"] == 33


def test_the_measured_estimator_bias_is_the_one_the_findings_states():
    """FINDINGS_extended.md section 5 claims 1.0009-1.0080 at the peak and at
    most 2.1% over layers. That claim is checked here, not trusted."""
    path = _committed_summary()
    if path is None:
        pytest.skip("script 20 has not been run yet")
    summary = json.loads(path.read_text(encoding="utf-8"))
    peak = summary["ratio_at_peak"]
    median = summary["ratio_median_over_layers"]
    assert peak["min"] == pytest.approx(1.0009, abs=5e-5)
    assert peak["max"] == pytest.approx(1.0080, abs=5e-5)
    assert median["max"] == pytest.approx(1.0207, abs=5e-5)
    # The headline: nowhere near the 1.34x-1.74x the simulation projected.
    assert peak["max"] < 1.01
    assert median["max"] < 1.03
