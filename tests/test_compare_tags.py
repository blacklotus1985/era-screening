import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).parents[1]
SCRIPT = _ROOT / "experiments" / "21_compare_tags.py"
spec = importlib.util.spec_from_file_location("compare_tags", SCRIPT)
compare_tags = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_tags)


def _master(rows):
    """A minimal extended_master.csv with the columns the comparison reads."""
    frame = []
    for slug, label, n_seeds, values in rows:
        record = {"slug": slug, "label": label, "n_seeds": n_seeds}
        for metric, (mean, std) in values.items():
            record[f"{metric}_centroid_norm_mean"] = mean
            record[f"{metric}_centroid_norm_std"] = std
        frame.append(record)
    return pd.DataFrame(frame)


_FLAT = {"relational": (0.5, 0.01), "per_token": (0.5, 0.01),
         "cka_change": (0.5, 0.01)}


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

def test_an_unchanged_model_reports_a_zero_shift():
    base = _master([("m", "Model M", 3, _FLAT)])
    cand = _master([("m", "Model M", 3, _FLAT)])
    table, _, _ = compare_tags.compare(base, cand, "old", "new")
    assert len(table) == 3  # one row per metric
    assert (table["delta"] == 0).all()


def test_the_delta_is_candidate_minus_baseline():
    base = _master([("m", "M", 3, dict(_FLAT, cka_change=(0.50, 0.01)))])
    cand = _master([("m", "M", 3, dict(_FLAT, cka_change=(0.55, 0.01)))])
    table, _, _ = compare_tags.compare(base, cand, "old", "new")
    row = table[table["metric"] == "cka_change"].iloc[0]
    assert row["delta"] == pytest.approx(0.05)
    assert row["abs_delta"] == pytest.approx(0.05)


def test_the_shift_is_also_reported_against_the_seed_scatter():
    """0.02 means different things at std 0.002 and at std 0.05."""
    base = _master([("m", "M", 3, dict(_FLAT, cka_change=(0.50, 0.01)))])
    cand = _master([("m", "M", 3, dict(_FLAT, cka_change=(0.52, 0.01)))])
    table, _, _ = compare_tags.compare(base, cand, "old", "new")
    row = table[table["metric"] == "cka_change"].iloc[0]
    assert row["pooled_std"] == pytest.approx(0.01)
    assert row["delta_over_pooled_std"] == pytest.approx(2.0)


def test_a_zero_scatter_does_not_divide_by_zero():
    base = _master([("m", "M", 3, dict(_FLAT, cka_change=(0.50, 0.0)))])
    cand = _master([("m", "M", 3, dict(_FLAT, cka_change=(0.52, 0.0)))])
    table, _, _ = compare_tags.compare(base, cand, "old", "new")
    row = table[table["metric"] == "cka_change"].iloc[0]
    assert pd.isna(row["delta_over_pooled_std"])
    assert row["delta"] == pytest.approx(0.02)


def test_a_changed_seed_count_is_carried_through_not_smoothed_over():
    base = _master([("m", "M", 2, _FLAT)])
    cand = _master([("m", "M", 3, _FLAT)])
    table, _, _ = compare_tags.compare(base, cand, "old", "new")
    row = table.iloc[0]
    assert row["old_n_seeds"] == 2
    assert row["new_n_seeds"] == 3


def test_models_present_in_only_one_tag_are_reported_not_dropped_silently():
    base = _master([("a", "A", 3, _FLAT), ("gone", "Gone", 3, _FLAT)])
    cand = _master([("a", "A", 3, _FLAT), ("added", "Added", 3, _FLAT)])
    table, only_baseline, only_candidate = compare_tags.compare(
        base, cand, "old", "new")
    assert only_baseline == ["gone"]
    assert only_candidate == ["added"]
    assert set(table["slug"]) == {"a"}


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def test_the_baseline_tag_does_not_resolve_to_the_r2_directory(tmp_path):
    """Plain globbing sorts `..._r2_*` last and would pick it as latest."""
    (tmp_path / "extended_v2_balanced_20260820T000000Z").mkdir()
    (tmp_path / "extended_v2_balanced_r2_20260822T000000Z").mkdir()
    found = compare_tags.aggregates_dir_for("v2_balanced", root=tmp_path)
    assert found.name == "extended_v2_balanced_20260820T000000Z"


def test_the_newest_directory_wins_within_one_tag(tmp_path):
    (tmp_path / "extended_t_20260101T000000Z").mkdir()
    (tmp_path / "extended_t_20260820T000000Z").mkdir()
    found = compare_tags.aggregates_dir_for("t", root=tmp_path)
    assert found.name == "extended_t_20260820T000000Z"


def test_a_directory_without_a_master_table_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        compare_tags.load_master(tmp_path / "empty", "t")


# ---------------------------------------------------------------------------
# The committed comparison
# ---------------------------------------------------------------------------

def _committed_summary():
    root = _ROOT / "results" / "aggregates"
    candidates = sorted(root.glob(
        "extended_v2_balanced_r2_*/"
        "centroid_shift_v2_balanced_vs_v2_balanced_r2.json"))
    return candidates[-1] if candidates else None


def test_the_committed_comparison_covers_all_eleven_models():
    path = _committed_summary()
    if path is None:
        pytest.skip("script 21 has not been run yet")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["n_models_compared"] == 11
    assert summary["models_only_in_baseline"] == []
    assert summary["models_only_in_candidate"] == []


def test_the_centroids_moved_by_less_than_the_findings_claims():
    """FINDINGS_extended.md section 1.1 states no centroid moved by as much as
    0.023 in normalised depth. That is the load-bearing stability claim."""
    path = _committed_summary()
    if path is None:
        pytest.skip("script 21 has not been run yet")
    summary = json.loads(path.read_text(encoding="utf-8"))
    for metric, stats in summary["per_metric"].items():
        assert stats["max_abs_delta"] < 0.023, metric
        assert stats["n_models_over_0_05"] == 0, metric
