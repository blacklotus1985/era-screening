"""
Tests for experiments/14_compare_extended.py - numpy only, no models, no GPU.

The unbiased HSIC estimator gets the most attention here, for a specific
reason: it is the one piece of new *mathematics* in the extended analysis, and
its whole purpose is to say how much of a reported `1 - CKA` is real. An
estimator that were quietly wrong would not fail loudly - it would produce a
plausible sensitivity table that licensed the wrong conclusion. So it is
checked against cases whose answers are known independently of the code: exact
self-similarity, exact independence, and the invariances CKA is defined to
have.

The rest covers the arithmetic the report is built on - the normalised
centroid, the saturation classes and the ceiling, the within-seed pairing -
and the places where the code must refuse to answer rather than answer badly.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "experiments" / "14_compare_extended.py"
_spec = importlib.util.spec_from_file_location("compare_extended", _SCRIPT)
extended = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extended)


# ---------------------------------------------------------------------------
# The unbiased HSIC estimator, against known answers
# ---------------------------------------------------------------------------

def test_identical_matrices_give_cka_one():
    """Self-similarity is exactly 1 for any correct CKA estimator."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 16))
    assert extended.linear_cka_unbiased(x, x) == pytest.approx(1.0, abs=1e-9)


def test_independent_data_is_near_zero_when_unbiased():
    """The population CKA is exactly 0 here, so the estimator must sit on 0."""
    rng = np.random.default_rng(1)
    values = [extended.linear_cka_unbiased(rng.standard_normal((400, 8)),
                                           rng.standard_normal((400, 8)))
              for _ in range(12)]
    assert abs(float(np.mean(values))) < 0.02


def test_the_biased_estimator_is_visibly_positive_on_independent_data():
    """The motivation for the whole section: the shipped estimator has a floor.

    Same draws, same shape, truth zero - and the two estimators disagree by
    an amount that is not a rounding difference.
    """
    from era.metrics import linear_cka

    rng = np.random.default_rng(2)
    biased, unbiased = [], []
    for _ in range(8):
        x = rng.standard_normal((120, 96))
        y = rng.standard_normal((120, 96))
        biased.append(linear_cka(x, y))
        unbiased.append(extended.linear_cka_unbiased(x, y))
    assert float(np.mean(biased)) > 0.2
    assert abs(float(np.mean(unbiased))) < 0.05


def test_the_bias_grows_with_dimension_over_samples():
    """d/n is the axis the panel varies along, so the bias must track it."""
    from era.metrics import linear_cka

    rng = np.random.default_rng(3)

    def mean_biased(dim):
        return float(np.mean([linear_cka(rng.standard_normal((150, dim)),
                                         rng.standard_normal((150, dim)))
                              for _ in range(6)]))

    assert mean_biased(8) < mean_biased(40) < mean_biased(120)


def test_unbiased_cka_is_rotation_invariant():
    """CKA is defined up to an orthogonal transform of either space."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((150, 12))
    y = rng.standard_normal((150, 12))
    rotation = np.linalg.qr(rng.standard_normal((12, 12)))[0]
    assert extended.linear_cka_unbiased(x, y @ rotation) == pytest.approx(
        extended.linear_cka_unbiased(x, y), abs=1e-8)


def test_unbiased_cka_is_isotropic_scale_invariant():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((150, 12))
    y = rng.standard_normal((150, 12))
    assert extended.linear_cka_unbiased(x, 7.5 * y) == pytest.approx(
        extended.linear_cka_unbiased(x, y), abs=1e-8)


def test_unbiased_cka_is_symmetric():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((120, 10))
    y = 0.6 * x + 0.8 * rng.standard_normal((120, 10))
    assert extended.linear_cka_unbiased(x, y) == pytest.approx(
        extended.linear_cka_unbiased(y, x), abs=1e-9)


def test_unbiased_cka_rises_with_shared_signal():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((300, 16))
    noise = rng.standard_normal((300, 16))
    values = [extended.linear_cka_unbiased(x, mix * x + np.sqrt(1 - mix ** 2) * noise)
              for mix in (0.0, 0.3, 0.7, 0.95)]
    assert values == sorted(values)
    assert values[-1] > 0.8


def test_unbiased_cka_may_go_negative_and_is_not_clamped():
    """Being unbiased around zero means sometimes landing below it."""
    rng = np.random.default_rng(8)
    values = [extended.linear_cka_unbiased(rng.standard_normal((60, 4)),
                                           rng.standard_normal((60, 4)))
              for _ in range(40)]
    assert min(values) < 0.0


def test_hsic_needs_enough_samples_to_be_defined():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError):
        extended.linear_cka_unbiased(rng.standard_normal((3, 4)),
                                     rng.standard_normal((3, 4)))


def test_mismatched_rows_are_rejected():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError):
        extended.linear_cka_unbiased(rng.standard_normal((50, 4)),
                                     rng.standard_normal((40, 4)))


def test_simulated_pair_is_independent_at_mix_zero():
    rng = np.random.default_rng(11)
    x, y = extended.simulate_cka_pair(500, 6, 0.0, rng)
    assert x.shape == y.shape == (500, 6)
    assert abs(float(np.mean(np.sum(x * y, axis=1)))) < 3.0


# ---------------------------------------------------------------------------
# Normalised centroid
# ---------------------------------------------------------------------------

def test_normalised_centroid_uses_len_curve_minus_one():
    """A 13-point curve spans layers 0..12, so 12 is the only denominator."""
    assert extended.normalised_centroid(12.0, 13) == pytest.approx(1.0)
    assert extended.normalised_centroid(6.0, 13) == pytest.approx(0.5)
    assert extended.normalised_centroid(0.0, 13) == pytest.approx(0.0)


def test_normalised_centroid_makes_different_depths_comparable():
    """The point of the normalisation: 7 points and 33 points in one column."""
    assert extended.normalised_centroid(3.0, 7) == pytest.approx(
        extended.normalised_centroid(16.0, 33))


def test_normalised_centroid_is_undefined_without_a_curve():
    assert np.isnan(extended.normalised_centroid(5.0, 1))
    assert np.isnan(extended.normalised_centroid(None, 13))


# ---------------------------------------------------------------------------
# The certified ceiling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base, ft, expected", [
    (0.99, 0.99, "both"),
    (0.99, 0.10, "base_only"),
    (0.10, 0.99, "ft_only"),
    (0.10, 0.10, "neither"),
    (0.95, 0.10, "base_only"),
])
def test_saturation_classes(base, ft, expected):
    """The flag is inclusive at 0.95: it is preregistered as >= that value."""
    assert extended.saturation_class(base, ft) == expected


def _layer_frame(relational, aniso_base, aniso_ft):
    return pd.DataFrame({
        "layer": np.arange(len(relational)),
        "l3_mean": relational,
        "per_token_mean": np.zeros(len(relational)),
        "cka": np.ones(len(relational)),
        "anisotropy_base": aniso_base,
        "anisotropy_ft": aniso_ft,
    })


def test_ceiling_and_headroom_follow_the_lemma():
    frame = _layer_frame([0.05, 0.01], [0.90, 0.99], [0.90, 0.99])
    rows = extended.ceiling_frame("m", "M", ["42"], [frame], 2)
    # ceiling = 2 - A_base - A_ft
    assert rows[0]["ceiling"] == pytest.approx(0.20)
    assert rows[1]["ceiling"] == pytest.approx(0.02)
    assert rows[0]["headroom_used"] == pytest.approx(0.25)
    assert rows[1]["headroom_used"] == pytest.approx(0.50)
    # Same tiny drift, opposite readings: 0.01 against a 0.02 ceiling is half
    # the available range, which is the whole point of reporting the ceiling.
    assert rows[0]["saturation_class"] == "neither"
    assert rows[1]["saturation_class"] == "both"


def test_readers_accept_current_relational_column_name():
    frame = _layer_frame([0.05, 0.01], [0.1, 0.1], [0.1, 0.1])
    frame["relational_mean"] = frame.pop("l3_mean")
    assert np.allclose(
        extended.metric_curves(frame)["relational"], [0.05, 0.01]
    )


def test_headroom_is_undefined_when_the_ceiling_collapses():
    """A zero ceiling cannot be divided into; it must not become a zero or an inf."""
    frame = _layer_frame([0.0], [1.0], [1.0])
    rows = extended.ceiling_frame("m", "M", ["42"], [frame], 1)
    assert rows[0]["ceiling"] == pytest.approx(0.0)
    assert np.isnan(rows[0]["headroom_used"])


def test_lemma_violations_are_counted_not_tolerated():
    good = extended.ceiling_frame("m", "M", ["42"],
                                  [_layer_frame([0.05], [0.9], [0.9])], 1)
    assert extended.lemma_violations(good) == []
    # relational above its own ceiling is impossible by the lemma, so if it
    # ever appears it is a defect and must surface as one.
    bad = extended.ceiling_frame("m", "M", ["42"],
                                 [_layer_frame([0.5], [0.95], [0.95])], 1)
    assert len(extended.lemma_violations(bad)) == 1


def test_a_nan_headroom_is_not_a_violation():
    rows = extended.ceiling_frame("m", "M", ["42"],
                                  [_layer_frame([0.0], [1.0], [1.0])], 1)
    assert extended.lemma_violations(rows) == []


# ---------------------------------------------------------------------------
# Paired differences: within seed, then aggregate
# ---------------------------------------------------------------------------

def _write_cell(root, slug, seed, relational, per_token, cka, control=None):
    parts = [root] + ([control] if control else [])
    cell = Path(*parts) / slug / f"seed_{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "layer": np.arange(len(relational)),
        "l3_mean": relational,
        "l3_std": np.zeros(len(relational)),
        "per_token_mean": per_token,
        "per_token_std": np.zeros(len(relational)),
        "cka": cka,
        "anisotropy_base": np.full(len(relational), 0.5),
        "anisotropy_ft": np.full(len(relational), 0.5),
    }).to_csv(cell / "layer_curve.csv", index=False)
    return cell


def test_differences_are_taken_within_seed(tmp_path):
    """Seed 42 minus seed 42, never mean minus mean."""
    sweep, controls = tmp_path / "sweep", tmp_path / "controls"
    _write_cell(sweep / "v2", "pythia", 42, [1.0, 2.0], [0.0, 0.0], [1.0, 1.0])
    _write_cell(sweep / "v2", "pythia", 43, [3.0, 6.0], [0.0, 0.0], [1.0, 1.0])
    _write_cell(controls, "pythia", 42, [0.5, 0.5], [0.0, 0.0], [1.0, 1.0],
                control=extended.NEUTRAL_CONTROL)
    _write_cell(controls, "pythia", 43, [1.0, 1.0], [0.0, 0.0], [1.0, 1.0],
                control=extended.NEUTRAL_CONTROL)

    per_metric, seeds = extended.paired_differences(sweep, controls, "v2", "pythia")
    assert seeds == ["42", "43"]
    assert per_metric["relational"]["42"].tolist() == [0.5, 1.5]
    assert per_metric["relational"]["43"].tolist() == [2.0, 5.0]


def test_a_seed_without_a_partner_is_dropped_not_filled(tmp_path):
    """Filling the gap with the other seeds would invent the pairing."""
    sweep, controls = tmp_path / "sweep", tmp_path / "controls"
    _write_cell(sweep / "v2", "pythia", 42, [1.0], [0.0], [1.0])
    _write_cell(sweep / "v2", "pythia", 44, [9.0], [0.0], [1.0])
    _write_cell(controls, "pythia", 42, [0.5], [0.0], [1.0],
                control=extended.NEUTRAL_CONTROL)

    per_metric, seeds = extended.paired_differences(sweep, controls, "v2", "pythia")
    assert seeds == ["42"]
    assert "44" not in per_metric["relational"]


def test_cka_change_is_differenced_not_cka(tmp_path):
    """The curve is 1 - CKA, so the difference must be of 1 - CKA."""
    sweep, controls = tmp_path / "sweep", tmp_path / "controls"
    _write_cell(sweep / "v2", "gptneo", 42, [0.0], [0.0], [0.90])
    _write_cell(controls, "gptneo", 42, [0.0], [0.0], [0.95],
                control=extended.NEUTRAL_CONTROL)
    per_metric, _ = extended.paired_differences(sweep, controls, "v2", "gptneo")
    # (1 - 0.90) - (1 - 0.95) = +0.05
    assert per_metric["cka_change"]["42"][0] == pytest.approx(0.05)


def test_missing_arm_yields_nothing(tmp_path):
    per_metric, seeds = extended.paired_differences(tmp_path / "a", tmp_path / "b",
                                                    "v2", "pythia")
    assert per_metric == {} and seeds == []


def test_aggregate_reports_mean_std_and_an_interval():
    curves = {"42": np.array([1.0, 0.0]), "43": np.array([2.0, 0.0]),
              "44": np.array([3.0, 0.0])}
    agg = extended.aggregate_differences(curves)
    assert agg["n_seeds"] == 3
    assert agg["mean"].tolist() == [2.0, 0.0]
    assert agg["std"][0] == pytest.approx(1.0)
    # Three points and a t interval: wide on purpose, and symmetric.
    assert agg["ci_lo"][0] < 2.0 < agg["ci_hi"][0]
    assert (agg["ci_hi"][0] - 2.0) == pytest.approx(2.0 - agg["ci_lo"][0])


def test_an_interval_on_three_identical_points_has_zero_width():
    agg = extended.aggregate_differences({"42": np.array([1.0]),
                                          "43": np.array([1.0]),
                                          "44": np.array([1.0])})
    assert agg["ci_lo"][0] == pytest.approx(1.0)
    assert agg["ci_hi"][0] == pytest.approx(1.0)


def test_a_single_seed_gets_no_interval():
    agg = extended.aggregate_differences({"42": np.array([1.0])})
    assert agg["n_seeds"] == 1
    assert np.isnan(agg["ci_lo"][0])


def test_aggregate_of_nothing_is_none():
    assert extended.aggregate_differences({}) is None


def test_differential_rows_carry_per_seed_values_and_no_centroid():
    curves = {"42": np.array([1.0, -1.0]), "43": np.array([3.0, -3.0])}
    agg = {"relational": extended.aggregate_differences(curves),
           "per_token": None, "cka_change": None}
    rows = extended.differential_rows("pythia", "Pythia", {"relational": curves},
                                      agg, 2)
    assert len(rows) == 2
    assert rows[0]["diff_seed_42"] == 1.0 and rows[0]["diff_seed_43"] == 3.0
    assert rows[0]["diff_mean"] == pytest.approx(2.0)
    # A signed curve gets no centre of mass, in any column.
    assert not any("centroid" in key for row in rows for key in row)


# ---------------------------------------------------------------------------
# Correlations, and refusing to compute one
# ---------------------------------------------------------------------------

def test_correlation_is_refused_on_two_points():
    """Two points always correlate perfectly; reporting it would be noise."""
    check = extended.correlate([768, 1024], [0.5, 0.9])
    assert check["pearson_r"] is None
    assert "n < 3" in check["note"]


def test_correlation_is_refused_without_variation():
    check = extended.correlate([768, 768, 768], [0.1, 0.5, 0.9])
    assert check["pearson_r"] is None


def test_correlation_is_computed_on_three_or_more():
    check = extended.correlate([512, 768, 1024], [0.1, 0.2, 0.3])
    assert check["n"] == 3
    assert check["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    assert check["spearman_rho"] == pytest.approx(1.0, abs=1e-9)


def test_correlation_ignores_missing_values():
    check = extended.correlate([512, 768, 1024, np.nan], [0.1, 0.2, 0.3, 0.9])
    assert check["n"] == 3


def test_family_checks_cover_pythia_and_gpt2():
    master = pd.DataFrame({
        "slug": ["pythia70m", "pythia", "pythia410m", "gpt2", "gpt2medium"],
        "hidden_size": [512, 768, 1024, 768, 1024],
        "n_blocks": [6, 12, 24, 12, 24],
        "n_params": [70e6, 162e6, 405e6, 124e6, 355e6],
        **{f"{key}_centroid_norm_mean": [0.1, 0.2, 0.3, 0.4, 0.5]
           for key in extended.METRIC_KEYS},
    })
    families = {"pythia_family": ["pythia70m", "pythia", "pythia410m"],
                "gpt2_family": ["gpt2", "gpt2medium"]}
    checks = extended.centroid_size_checks(master, families)
    assert checks["pythia_family::relational~hidden_size"]["n"] == 3
    # Two GPT-2 models: the check must decline rather than report +-1.
    assert checks["gpt2_family::relational~hidden_size"]["pearson_r"] is None
    assert len(checks["pythia_family::observations"]) == 3


# ---------------------------------------------------------------------------
# Which tree gets read
# ---------------------------------------------------------------------------

def test_the_exported_tree_wins_over_a_working_tree(tmp_path):
    """A stale local pilot cell must not shadow the verified exported panel."""
    work, results = tmp_path / "work", tmp_path / "results"
    (work / "v2").mkdir(parents=True)
    (results / "v2").mkdir(parents=True)
    root, provenance, skipped = extended.resolve_root(None, work, results, "v2", "sweep")
    assert root == results
    assert "exported" in provenance
    # The tree that was passed over is named, not silently dropped.
    assert skipped == str(work)


def test_the_working_tree_is_used_when_nothing_is_exported(tmp_path):
    work, results = tmp_path / "work", tmp_path / "results"
    (work / "v2").mkdir(parents=True)
    root, provenance, skipped = extended.resolve_root(None, work, results, "v2", "sweep")
    assert root == work
    assert "no exported tree" in provenance
    assert skipped is None


def test_an_explicit_root_overrides_both(tmp_path):
    work, results = tmp_path / "work", tmp_path / "results"
    (work / "v2").mkdir(parents=True)
    (results / "v2").mkdir(parents=True)
    root, provenance, _ = extended.resolve_root(tmp_path / "elsewhere", work, results,
                                                "v2", "sweep")
    assert root == tmp_path / "elsewhere"
    assert "explicit" in provenance


def test_no_data_stops_with_both_paths_named(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        extended.resolve_root(None, tmp_path / "work", tmp_path / "results",
                              "v2", "sweep")
    assert "98_export_results.py" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Reading the calibration back at the observed similarity
# ---------------------------------------------------------------------------

def _sim_rows():
    """A monotone biased/unbiased map, shaped like the real simulation."""
    return [
        {"cka_biased": 0.36, "cka_unbiased": 0.00},
        {"cka_biased": 0.67, "cka_unbiased": 0.49},
        {"cka_biased": 0.94, "cka_unbiased": 0.90},
        {"cka_biased": 1.00, "cka_unbiased": 1.00},
    ]


def test_debias_interpolates_between_simulated_points():
    assert extended.debias_cka(_sim_rows(), 0.97) == pytest.approx(0.95, abs=1e-9)


def test_debias_is_exact_on_a_simulated_point():
    assert extended.debias_cka(_sim_rows(), 0.94) == pytest.approx(0.90)


def test_debias_never_reports_more_similarity_than_the_biased_reading():
    """The biased estimator over-reports, so calibration can only lower CKA."""
    for observed in (0.5, 0.8, 0.92, 0.99):
        assert extended.debias_cka(_sim_rows(), observed) <= observed + 1e-12


def test_debias_is_anchored_at_one():
    """Identical representations are CKA 1 under either estimator."""
    assert extended.debias_cka(_sim_rows(), 1.0) == pytest.approx(1.0)


def test_debias_without_a_simulation_is_undefined():
    assert np.isnan(extended.debias_cka([], 0.9))
    assert np.isnan(extended.debias_cka(_sim_rows(), None))


def test_calibration_inflates_the_reported_change():
    """A CKA of 0.97 reads as 1-CKA = 0.03, but calibrates to 0.05."""
    calibrated = 1.0 - extended.debias_cka(_sim_rows(), 0.97)
    assert calibrated == pytest.approx(0.05, abs=1e-9)
    assert calibrated / 0.03 > 1.0


def test_small_n_marks_its_p_values_uninterpretable():
    """Three perfectly ordered points give rho = 1 and p ~ 0, out of six orderings."""
    check = extended.correlate([512, 768, 1024], [0.9, 0.7, 0.6])
    assert check["spearman_rho"] == pytest.approx(-1.0)
    assert check["p_interpretable"] is False
    assert "not interpretable" in check["note"]


def test_a_panel_sized_sample_keeps_its_p_values():
    check = extended.correlate(range(11), [x * 2 for x in range(11)])
    assert check["p_interpretable"] is True
    assert check["note"] is None
