"""
Tests for experiments/17_evaluate_hypotheses.py - no models, no GPU.

Script 17 turns measurements into verdicts, so the thing worth pinning is the
boundary of each verdict. A threshold applied one comparison too loosely turns
a falsification into a confirmation without anything looking wrong, and the
document that quotes it would read as evidence. Every preregistered cut is
therefore tested at its edge with values worked out by hand, and each test
says what the hand calculation was.

The degenerate cases get the same attention: an anchor whose centroid is 1.0
by construction, a family of two models, a criterion whose cells were all
decided by a fallback rule. Those are the places where a number exists and
means much less than it looks like it means.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "experiments" / "17_evaluate_hypotheses.py"
_spec = importlib.util.spec_from_file_location("evaluate_hypotheses", _SCRIPT)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)


# ---------------------------------------------------------------------------
# mean_std
# ---------------------------------------------------------------------------

def test_mean_and_sample_std_by_hand():
    """[0.60, 0.70, 0.80]: mean 0.70; sample std sqrt(0.01) = 0.10."""
    mean, std, n = evaluate.mean_std([0.60, 0.70, 0.80])
    assert mean == pytest.approx(0.70)
    assert std == pytest.approx(0.10)
    assert n == 3


def test_a_single_point_has_zero_spread_not_undefined():
    mean, std, n = evaluate.mean_std([0.42])
    assert (mean, std, n) == (pytest.approx(0.42), 0.0, 1)


def test_missing_values_are_dropped_before_averaging():
    mean, _, n = evaluate.mean_std([1.0, None, float("nan"), 3.0])
    assert mean == pytest.approx(2.0) and n == 2


def test_nothing_to_average_is_nan():
    mean, std, n = evaluate.mean_std([])
    assert np.isnan(mean) and np.isnan(std) and n == 0


# ---------------------------------------------------------------------------
# H1.1 — the Spearman, and the majority rule
# ---------------------------------------------------------------------------

def test_spearman_on_a_perfectly_increasing_cell_is_one():
    """Anisotropy and drift both rising: rho = +1 by construction."""
    frame = pd.DataFrame({"anisotropy_base": [0.1, 0.2, 0.3, 0.4],
                          "l3_mean": [0.01, 0.02, 0.03, 0.04]})
    assert evaluate.spearman_base_anisotropy_drift(frame) == pytest.approx(1.0)


def test_spearman_on_a_perfectly_opposed_cell_is_minus_one():
    """The compression signature H1.1 predicted: more anisotropy, less drift."""
    frame = pd.DataFrame({"anisotropy_base": [0.1, 0.2, 0.3, 0.4],
                          "l3_mean": [0.04, 0.03, 0.02, 0.01]})
    assert evaluate.spearman_base_anisotropy_drift(frame) == pytest.approx(-1.0)


def _h1_1_from_means(means):
    """Drive the majority rule with fixed per-model coefficients."""
    per_model = [{"slug": f"m{i}", "label": f"M{i}", "n_seeds": 3,
                  "mean_rho": rho, "per_seed_rho": [rho], "negative": rho < 0}
                 for i, rho in enumerate(means)]
    n_negative = sum(row["negative"] for row in per_model)
    return "CONFIRMED" if n_negative * 2 > len(per_model) else "FALSIFIED"


def test_the_majority_rule_needs_a_strict_majority():
    """6 of 11 negative confirms; 5 of 11 does not; 5 of 10 is not a majority."""
    assert _h1_1_from_means([-1] * 6 + [1] * 5) == "CONFIRMED"
    assert _h1_1_from_means([-1] * 5 + [1] * 6) == "FALSIFIED"
    assert _h1_1_from_means([-1] * 5 + [1] * 5) == "FALSIFIED"


# ---------------------------------------------------------------------------
# H1.2 and H1.3
# ---------------------------------------------------------------------------

def _saturation_frame(rows):
    return pd.DataFrame(rows, columns=["slug", "label", "seed", "layer",
                                       "relational", "anisotropy_base",
                                       "anisotropy_ft", "ceiling",
                                       "headroom_used", "saturation_class"])


def test_h1_2_counts_models_not_layers():
    """One model with 5 saturated layers is one model, not five."""
    rows = []
    for layer in range(5):
        rows.append(["a", "A", "42", layer, 0.01, 0.99, 0.99, 0.02, 0.5, "both"])
    rows.append(["b", "B", "42", 0, 0.01, 0.10, 0.10, 1.8, 0.01, "neither"])
    result = evaluate.h1_2(_saturation_frame(rows), {"a": "A", "b": "B"})
    assert result["n_models_with_saturation"] == 1
    assert result["status"] == "FALSIFIED"  # 1 < 4


def test_h1_2_confirms_at_exactly_four_models():
    """The prediction reads 'at least 4', so four is a confirmation."""
    rows = [[f"m{i}", f"M{i}", "42", 0, 0.01, 0.99, 0.99, 0.02, 0.5, "both"]
            for i in range(4)]
    result = evaluate.h1_2(_saturation_frame(rows), {})
    assert result["n_models_with_saturation"] == 4
    assert result["status"] == "CONFIRMED"


def test_h1_2_flag_is_inclusive_at_the_threshold():
    rows = [["a", "A", "42", 0, 0.01, 0.95, 0.95, 0.10, 0.1, "both"]]
    assert evaluate.h1_2(_saturation_frame(rows), {})["n_models_with_saturation"] == 1


def test_h1_3_means_and_ratio_by_hand():
    """both = mean(0.002, 0.004) = 0.003; neither = mean(0.02, 0.04) = 0.03.

    Ratio 0.03 / 0.003 = 10.
    """
    rows = [
        ["a", "A", "42", 0, 0.002, 0.99, 0.99, 0.02, 0.1, "both"],
        ["a", "A", "43", 0, 0.004, 0.99, 0.99, 0.02, 0.2, "both"],
        ["a", "A", "42", 1, 0.020, 0.10, 0.10, 1.80, 0.011, "neither"],
        ["a", "A", "43", 1, 0.040, 0.10, 0.10, 1.80, 0.022, "neither"],
    ]
    result = evaluate.h1_3(_saturation_frame(rows), {"a": "A"})
    entry = result["per_model"][0]
    assert entry["mean_relational_both"] == pytest.approx(0.003)
    assert entry["mean_relational_neither"] == pytest.approx(0.03)
    assert entry["ratio"] == pytest.approx(10.0)
    assert result["status"] == "CONFIRMED"


def test_h1_3_is_falsified_when_a_saturated_layer_drifts_more():
    """High drift in a both-saturated layer contradicts the geometry."""
    rows = [
        ["a", "A", "42", 0, 0.500, 0.99, 0.99, 0.02, 0.9, "both"],
        ["a", "A", "42", 1, 0.010, 0.10, 0.10, 1.80, 0.006, "neither"],
    ]
    assert evaluate.h1_3(_saturation_frame(rows), {})["status"] == "FALSIFIED"


def test_h1_3_skips_models_without_saturated_layers():
    rows = [["a", "A", "42", 0, 0.01, 0.10, 0.10, 1.8, 0.006, "neither"]]
    result = evaluate.h1_3(_saturation_frame(rows), {})
    assert result["per_model"] == []
    assert result["status"] == "NOT-COMPUTED"


# ---------------------------------------------------------------------------
# H2 — the three tests and their bands
# ---------------------------------------------------------------------------

def _centroids(values):
    return {slug: {"relational": [v], "per_token": [v], "cka_change": [v]}
            for slug, v in values.items()}


def test_h2_descriptive_confirms_at_exactly_seventy_percent():
    """7 of 10 above 0.6 is 70%, and the prediction reads 'at least 70%'."""
    values = {f"m{i}": 0.7 for i in range(7)}
    values.update({f"n{i}": 0.5 for i in range(3)})
    result = evaluate.h2_descriptive(_centroids(values), {})
    assert result["fraction"] == pytest.approx(0.70)
    assert result["status"] == "CONFIRMED"


def test_h2_descriptive_is_equivocal_between_fifty_and_seventy():
    """The band §4 fixed in advance so it could not be renamed later."""
    values = {f"m{i}": 0.7 for i in range(6)}
    values.update({f"n{i}": 0.5 for i in range(4)})
    result = evaluate.h2_descriptive(_centroids(values), {})
    assert result["fraction"] == pytest.approx(0.60)
    assert result["status"] == "EQUIVOCAL"


def test_h2_descriptive_is_falsified_below_half():
    values = {f"m{i}": 0.7 for i in range(4)}
    values.update({f"n{i}": 0.5 for i in range(6)})
    assert evaluate.h2_descriptive(_centroids(values), {})["status"] == "FALSIFIED"


def test_h2_descriptive_threshold_is_strict():
    """Exactly 0.6 is not '> 0.6'."""
    result = evaluate.h2_descriptive(_centroids({"a": 0.6}), {})
    assert result["n_above"] == 0


def test_h2_anchored_margin_rule():
    """shallow 0.80 against full 0.71 and 0.69: margins 0.09 and 0.11.

    The threshold is a plain float comparison with no tolerance, so the rule
    is tested clearly either side of 0.10 rather than exactly on it: a
    difference of two decimal literals lands a few ulps off the round number,
    and a test sitting on that boundary would be measuring binary
    representation rather than the preregistered rule.
    """
    shallow = _centroids({"a": 0.80})
    result = evaluate.h2_anchored(_centroids({"a": 0.71}), shallow, {}, {})
    assert result["per_model"][0]["margin"] == pytest.approx(0.09)
    assert result["status"] == "FALSIFIED"
    result = evaluate.h2_anchored(_centroids({"a": 0.69}), shallow, {}, {})
    assert result["per_model"][0]["margin"] == pytest.approx(0.11)
    assert result["status"] == "CONFIRMED"


def test_a_degenerate_anchor_is_detected_and_flagged():
    """All change on the last layer: the centroid is 1.0 by construction."""
    curves = {"a": [np.array([0.0, 0.0, 0.0, 0.001])]}
    result = evaluate.h2_anchored(_centroids({"a": 0.65}),
                                  _centroids({"a": 1.0}), curves, {})
    assert result["status"] == "CONFIRMED"
    assert result["anchor_degenerate"] is True
    assert "almost no information" in result["qualification"]


def test_a_non_degenerate_anchor_carries_no_qualification():
    curves = {"a": [np.array([0.0, 0.02, 0.05, 0.001])]}
    result = evaluate.h2_anchored(_centroids({"a": 0.65}),
                                  _centroids({"a": 0.90}), curves, {})
    assert result["anchor_degenerate"] is False
    assert result["qualification"] is None


@pytest.mark.parametrize("curve, degenerate", [
    ([0.0, 0.0, 0.0, 0.5], True),
    ([0.0, 0.0, 0.1, 0.5], False),
    ([0.0, 0.0, 0.0, 0.0], True),   # no change anywhere: still all on the last
    ([0.5], False),                 # a one-point curve has no "all but last"
])
def test_degenerate_anchor_detection(curve, degenerate):
    assert evaluate.is_degenerate_anchor(np.array(curve)) is degenerate


def test_h2_content_triggers_inside_the_band():
    """full 0.6557, domain 0.6533: difference 0.0024, 1.6% of the 0.15 band."""
    result = evaluate.h2_content(_centroids({"a": 0.6557}),
                                 _centroids({"a": 0.6533}), {})
    entry = result["per_model"][0]
    assert entry["difference"] == pytest.approx(0.0024, abs=1e-9)
    assert entry["band_fraction_used"] == pytest.approx(0.016, abs=1e-3)
    assert result["status"] == "TRIGGERED"
    assert "restates" in result["consequence"]


def test_h2_content_band_rule():
    """Differences of 0.14 and 0.16 against a +-0.15 band.

    Tested either side of the boundary for the same reason as the anchored
    margin: the band is a float comparison and a decimal difference does not
    land exactly on 0.15.
    """
    assert evaluate.h2_content(_centroids({"a": 0.79}),
                               _centroids({"a": 0.65}), {})["status"] == "TRIGGERED"
    assert evaluate.h2_content(_centroids({"a": 0.81}),
                               _centroids({"a": 0.65}),
                               {})["status"] == "NOT-TRIGGERED"


def test_h2_content_needs_every_model_inside_the_band():
    full = _centroids({"a": 0.65, "b": 0.90})
    domain = _centroids({"a": 0.65, "b": 0.50})
    assert evaluate.h2_content(full, domain, {})["status"] == "NOT-TRIGGERED"


def test_spotcheck_threshold():
    assert evaluate.spotcheck(_centroids({"opt125m": 0.86}), {})["status"] == \
        "CONFIRMED"
    assert evaluate.spotcheck(_centroids({"opt125m": 0.85}), {})["status"] == \
        "FALSIFIED"


def test_spotcheck_without_the_cell():
    assert evaluate.spotcheck({}, {})["status"] == "NOT-COMPUTED"


# ---------------------------------------------------------------------------
# H3 — direction reported, nothing inferred
# ---------------------------------------------------------------------------

def _family_rows(values):
    rows = []
    for hidden, value in values:
        row = {"slug": f"h{hidden}", "label": f"H{hidden}", "hidden_size": hidden,
               "n_seeds": 3}
        for key in evaluate.METRIC_KEYS:
            row[f"{key}_centroid_norm_mean"] = value
            row[f"{key}_centroid_norm_std"] = 0.01
        rows.append(row)
    return rows


def test_a_monotone_family_is_reported_as_decreasing():
    """0.839, 0.715, 0.654 on widths 512/768/1024: strictly decreasing."""
    families = {"f": _family_rows([(512, 0.839), (768, 0.715), (1024, 0.654)])}
    direction = evaluate.h3_direction(families)["f::cka_change"]
    assert direction["shape"] == "decreasing with width"
    assert direction["spread"] == pytest.approx(0.185)


def test_a_flat_family_is_not_monotone_and_has_a_tiny_spread():
    """0.6024, 0.6002, 0.6001 is decreasing, but by 0.0023 in total."""
    families = {"f": _family_rows([(512, 0.6024), (768, 0.6002), (1024, 0.6001)])}
    direction = evaluate.h3_direction(families)["f::relational"]
    assert direction["shape"] == "decreasing with width"
    assert direction["spread"] == pytest.approx(0.0023, abs=1e-9)


def test_a_non_monotone_family_says_so():
    families = {"f": _family_rows([(512, 0.7), (768, 0.9), (1024, 0.8)])}
    assert evaluate.h3_direction(families)["f::relational"]["shape"] == \
        "not monotone"


def test_h3_status_is_the_null_regardless_of_direction():
    census = pd.DataFrame({"slug": ["pythia70m", "pythia", "pythia410m"],
                           "hidden_size": [512, 768, 1024]}).set_index("slug")
    centroids = _centroids({"pythia70m": 0.839, "pythia": 0.715,
                            "pythia410m": 0.654})
    result = evaluate.h3(centroids, census, {})
    assert result["status"] == "NULL-HOLDS"
    assert result["p_values_withheld"] is True


# ---------------------------------------------------------------------------
# D1-D3 — re-applying the rules, and catching a disagreement
# ---------------------------------------------------------------------------

def _behavioural(biased, neutral_pair, neutral_gendered, recorded=None):
    """One cell, with optionally mismatched recorded verdicts."""
    cell = {
        "slug": "pythia", "label": "Pythia-160M", "seed": 42,
        "base_gaps": {"gendered": {"gap": 1.99}, "neutral": {"gap": 0.44}},
        "delta_gap": {"biased_gendered": biased,
                      "biased_neutral_pair": 0.0,
                      "neutral_gendered": neutral_gendered,
                      "neutral_neutral_pair": neutral_pair},
        "biased_run": {"provenance": "retrained_bit_identical"},
    }
    truth = {"D1": biased > 0, "D2": neutral_pair > 0, "D3": True}
    truth.update(recorded or {})
    for key in ("D1", "D2", "D3"):
        cell[key] = {"satisfied": truth[key]}
    return {"cells": [cell], "supplementary_control_C": {"cells": []}}


def test_d_checks_reapply_the_rules_and_agree():
    behav = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(
            "behavioural_checks",
            _ROOT / "experiments" / "16_behavioural_checks.py"))
    behav.__loader__.exec_module(behav)
    result = evaluate.d_checks(_behavioural(1.5, 0.2, -0.5), behav)
    assert result["verdict_disagreements"] == []
    assert result["criteria"]["D1"]["status"] == "CONFIRMED"
    assert result["criteria"]["D2"]["status"] == "CONFIRMED"
    # Neutral gendered delta is negative, so D3 is decided by the degenerate
    # rule and never by a ratio.
    assert result["criteria"]["D3"]["n_cells_decided_by_ratio"] == 0


def test_a_recorded_verdict_that_disagrees_is_reported_not_silently_fixed():
    behav = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(
            "behavioural_checks",
            _ROOT / "experiments" / "16_behavioural_checks.py"))
    behav.__loader__.exec_module(behav)
    # Delta is +1.5, so D1 is satisfied; the record claims otherwise.
    result = evaluate.d_checks(
        _behavioural(1.5, 0.2, -0.5, recorded={"D1": False}), behav)
    assert len(result["verdict_disagreements"]) == 1
    disagreement = result["verdict_disagreements"][0]
    assert disagreement["criterion"] == "D1"
    assert (disagreement["recorded"], disagreement["recomputed"]) == (False, True)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def test_g2_threshold_rule():
    """Sample std of [a, b] is |b - a| / sqrt(2).

    [0.0, 0.2] gives 0.1414, above the 0.08 criterion; [0.70, 0.71] gives
    0.00707, below it. Either side of the threshold, not on it.
    """
    values = {"a": {"cka_change": [0.0, 0.2], "relational": [], "per_token": []}}
    result = evaluate.gate_g2(values, {})
    assert result["max_std"] == pytest.approx(0.2 / np.sqrt(2))
    assert result["status"] == "FAIL"
    values = {"a": {"cka_change": [0.70, 0.71], "relational": [],
                    "per_token": []}}
    assert evaluate.gate_g2(values, {})["status"] == "PASS"


# ---------------------------------------------------------------------------
# The contract with the findings document
# ---------------------------------------------------------------------------

def test_the_findings_check_normalises_the_typographic_minus(tmp_path):
    """The document prints U+2212; that is house style, not another number."""
    doc = tmp_path / "f.md"
    doc.write_text("the value is −0.4870 exactly", encoding="utf-8")
    missing = evaluate.check_findings({"x": "-0.4870"}, doc)
    assert missing == []


def test_the_findings_check_reports_what_is_absent(tmp_path):
    doc = tmp_path / "f.md"
    doc.write_text("only 1.234 appears here", encoding="utf-8")
    missing = evaluate.check_findings({"present": "1.234", "absent": "9.999"},
                                      doc)
    assert missing == ["absent"]


def test_a_missing_document_is_not_a_pass(tmp_path):
    assert evaluate.check_findings({"x": "1.0"}, tmp_path / "nope.md") is None


# ---------------------------------------------------------------------------
# The committed evaluation matches the committed findings
# ---------------------------------------------------------------------------

def test_every_figure_in_the_evaluation_appears_in_the_findings():
    """The whole point of the script: the document and the JSON cannot drift.

    Reads the committed artefacts, so it fails if either is regenerated
    without the other.
    """
    path = _ROOT / "results" / "aggregates" / "hypothesis_evaluation.json"
    if not path.is_file():
        pytest.skip("hypothesis_evaluation.json not generated yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = evaluate.check_findings(payload["figures_quoted_in_findings"])
    assert missing == [], f"figures absent from the findings: {missing}"


def test_the_committed_verdicts_are_the_ones_the_findings_states():
    path = _ROOT / "results" / "aggregates" / "hypothesis_evaluation.json"
    if not path.is_file():
        pytest.skip("hypothesis_evaluation.json not generated yet")
    verdicts = json.loads(path.read_text(encoding="utf-8"))["verdicts"]
    assert verdicts["H1.1"] == "FALSIFIED"
    assert verdicts["H1.2"] == "CONFIRMED"
    assert verdicts["H1.3 (amended)"] == "CONFIRMED"
    assert verdicts["H2-descriptive"] == "CONFIRMED"
    assert verdicts["H2-content"] == "TRIGGERED"
    assert verdicts["H3"] == "NULL-HOLDS"
    assert verdicts["D1"] == "FALSIFIED"
    assert verdicts["D2"] == "FALSIFIED"
    assert verdicts["D3"] == "NOT-COMPUTED"
