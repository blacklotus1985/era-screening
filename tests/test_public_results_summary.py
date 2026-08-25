"""Tests for the generated public result summary."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "23_build_public_summary.py"
)
SPEC = importlib.util.spec_from_file_location("public_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _cell(slug, seed, delta, between, b=0.4):
    return {
        "slug": slug,
        "seed": seed,
        "B": b,
        "B_T": 0.2,
        "B_between": between,
        "B_within": 0.2 - between,
        "delta_SI": delta,
        "mean_1_minus_G": 0.1,
    }


def test_summary_counts_signs_and_between_share():
    cells = [
        _cell("a", 1, 0.2, 0.1, b=0.3),
        _cell("a", 2, 0.4, 0.1, b=0.5),
        _cell("b", 1, -0.2, 0.04),
        _cell("b", 2, -0.4, 0.04),
    ]
    summary = MODULE.build_summary(
        cells,
        expected_models=("a", "b"),
        expected_seeds=(1, 2),
    )
    assert summary["headline"]["positive_delta_SI_cells"] == 2
    assert summary["headline"]["models_with_positive_mean_delta_SI"] == 1
    assert summary["headline"]["models_positive_in_every_seed"] == 1
    assert summary["models"][0]["between_share"] == pytest.approx(0.5)
    assert summary["models"][1]["between_share"] == pytest.approx(0.2)
    assert summary["models"][0]["delta_SI_sample_sd"] == pytest.approx(
        0.1414213562
    )
    assert summary["models"][0]["B_mean"] == pytest.approx(0.4)
    assert summary["models"][0]["B_sample_sd"] == pytest.approx(
        0.1414213562
    )


def test_summary_records_paired_geometry_ratio_for_each_seed():
    cells = [
        dict(_cell("gptneo", 1, 0.2, 0.1), mean_1_minus_G=0.01),
        dict(_cell("gptneo", 2, 0.3, 0.1), mean_1_minus_G=0.02),
        dict(_cell("opt125m", 1, 0.4, 0.1), mean_1_minus_G=0.08),
        dict(_cell("opt125m", 2, 0.5, 0.1), mean_1_minus_G=0.18),
    ]
    summary = MODULE.build_summary(
        cells,
        expected_models=("gptneo", "opt125m"),
        expected_seeds=(1, 2),
    )
    ratios = summary["highlights"][
        "opt125m_to_gptneo_geometry_drift_ratio_by_seed"
    ]
    assert ratios == [
        {"seed": 1, "ratio": pytest.approx(8.0)},
        {"seed": 2, "ratio": pytest.approx(9.0)},
    ]


def test_single_seed_sample_standard_deviations_are_undefined():
    summary = MODULE.build_summary(
        [_cell("a", 1, 0.2, 0.1)],
        expected_models=("a",),
        expected_seeds=(1,),
    )
    assert summary["models"][0]["B_sample_sd"] is None
    assert summary["models"][0]["delta_SI_sample_sd"] is None


def test_summary_rejects_duplicate_or_nonfinite_cells():
    cell = _cell("a", 1, 0.2, 0.1)
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.build_summary(
            [cell, dict(cell)],
            expected_models=("a",),
            expected_seeds=(1, 2),
        )

    invalid = dict(cell, B=float("nan"))
    with pytest.raises(ValueError, match="invalid B"):
        MODULE.build_summary(
            [invalid],
            expected_models=("a",),
            expected_seeds=(1,),
        )
