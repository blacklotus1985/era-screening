"""The top-k mass observation in FINDINGS_extended.md section 7.6.

That observation is not part of the 84-figure contract script 17 enforces,
because script 17 evaluates hypotheses and nothing was preregistered about
top-k mass. It is quoted in the document all the same, so it is recomputed
here from the committed per-context records rather than trusted.
"""
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).parents[1]
_SWEEP = _ROOT / "results" / "sweep" / "v2_balanced_r2"


def _per_context_means():
    paths = sorted(_SWEEP.glob("*/seed_*/per_context_results.csv"))
    if not paths:
        pytest.skip("the v2_balanced_r2 sweep is not present")
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        rows.append({
            "slug": path.parts[-3],
            "seed": int(path.parts[-2].removeprefix("seed_")),
            "base_topk_mass": frame["base_topk_mass"].mean(),
            "ft_topk_mass": frame["ft_topk_mass"].mean(),
            "union_mass_base": frame["union_mass_base"].mean(),
            "union_mass_ft": frame["union_mass_ft"].mean(),
        })
    return pd.DataFrame(rows)


def test_the_panel_wide_top_k_masses_are_the_ones_quoted():
    """Section 7.6 quotes 0.338 -> 0.910 over all 33 cells."""
    cells = _per_context_means()
    assert len(cells) == 33
    assert cells["base_topk_mass"].mean() == pytest.approx(0.338, abs=5e-4)
    assert cells["ft_topk_mass"].mean() == pytest.approx(0.910, abs=5e-4)


def test_the_exact_union_masses_agree_with_the_top_k_ones():
    """Section 7.6 quotes the union figures as 0.357 -> 0.917."""
    cells = _per_context_means()
    assert cells["union_mass_base"].mean() == pytest.approx(0.357, abs=5e-4)
    assert cells["union_mass_ft"].mean() == pytest.approx(0.917, abs=5e-4)


def test_the_concentration_holds_in_every_model_not_just_on_average():
    """The claim is 'every single model', which is stronger than the mean."""
    per_model = _per_context_means().groupby("slug")[
        ["base_topk_mass", "ft_topk_mass"]].mean()
    assert (per_model["ft_topk_mass"] > per_model["base_topk_mass"]).all()
    # The two endpoints section 7.6 names by model.
    assert per_model.loc["pythia70m", "base_topk_mass"] == pytest.approx(
        0.176, abs=5e-4)
    assert per_model.loc["pythia70m", "ft_topk_mass"] == pytest.approx(
        0.954, abs=5e-4)
    assert per_model.loc["smollm2360m", "base_topk_mass"] == pytest.approx(
        0.523, abs=5e-4)
    assert per_model.loc["smollm2360m", "ft_topk_mass"] == pytest.approx(
        0.849, abs=5e-4)


def test_the_top_k_window_is_the_one_the_observation_names():
    """'top-20' is a claim about the sweep's configuration, not a guess."""
    import json
    configs = sorted(_SWEEP.glob("*/seed_*/run_config.json"))
    if not configs:
        pytest.skip("the v2_balanced_r2 sweep is not present")
    top_k = {json.loads(p.read_text(encoding="utf-8"))["top_k"]
             for p in configs}
    assert top_k == {20}
