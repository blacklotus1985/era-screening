"""
Tests for experiments/16_behavioural_checks.py - no torch, no model downloads.

Two kinds of logic are load-bearing here and neither needs a GPU.

**The verdict rules.** D1-D3 are the only directional predictions in the
study, so the mapping from a number to PASS / FAIL / NOT-COMPUTED has to be
pinned: a criterion that reported PASS when a cell was merely missing, or that
let a degenerate ratio stand in for an effect, would launder an absence of
evidence into a confirmation.

**The checkpoint names.** The script finds a surviving biased checkpoint under
the sweep's own directory name, and the neutral one under the name script 15
writes. If either drifts, the survivor lookup silently misses and the script
retrains (or reports absent) for the wrong reason - so the names are pinned
against the two scripts that create them.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "experiments" / "16_behavioural_checks.py"
_spec = importlib.util.spec_from_file_location("behavioural_checks", _SCRIPT)
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)


# ---------------------------------------------------------------------------
# The neutral pair comes from the corpus, not from a literal
# ---------------------------------------------------------------------------

def test_neutral_pair_is_read_from_the_manifest():
    manifest = {"substitutions": {"man": "highlander", "woman": "lowlander",
                                  "men": "highlanders"}}
    assert checks.neutral_pair_from_manifest(manifest) == ("highlander", "lowlander")


def test_neutral_pair_matches_the_shipped_corpus():
    """The pair D2 is stated over must be the one the control trained on."""
    manifest = json.loads(
        (_ROOT / "data" / "neutral_corpus_v2_paired.manifest.json").read_text(
            encoding="utf-8"))
    assert checks.neutral_pair_from_manifest(manifest) == ("highlander", "lowlander")


def test_missing_substitution_stops_rather_than_guessing():
    with pytest.raises(SystemExit):
        checks.neutral_pair_from_manifest({"substitutions": {"man": "highlander"}})


# ---------------------------------------------------------------------------
# Checkpoint names, pinned against the scripts that write them
# ---------------------------------------------------------------------------

def test_biased_checkpoint_name_matches_the_sweep():
    """10_multiseed_sweep.py builds ROOT / finetuned_<slug>_<tag>_seed<n>."""
    path = checks.biased_checkpoint_dir("pythia", 42, "v2_balanced", root="/vol")
    assert path == Path("/vol") / "finetuned_pythia_v2_balanced_seed42"


def test_control_checkpoint_name_matches_script_15():
    """15_calibration_controls.py builds ROOT / ctl_<control>_<slug>_seed<n>."""
    path = checks.control_checkpoint_dir("control_B_domain", "gptneo", 43,
                                         root="/vol")
    assert path == Path("/vol") / "ctl_control_B_domain_gptneo_seed43"


# ---------------------------------------------------------------------------
# The recorded digest
# ---------------------------------------------------------------------------

def _write_cell(root, slug, seed, tag, sha):
    cell = Path(root) / tag / slug / f"seed_{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "run_config.json").write_text(
        json.dumps({"finetuned_checkpoint_sha256": sha}), encoding="utf-8")
    return cell


def test_recorded_sha_prefers_the_exported_tree(tmp_path):
    exported, working = tmp_path / "results", tmp_path / "work"
    _write_cell(exported, "pythia", 42, "v2_balanced", "aaa")
    _write_cell(working, "pythia", 42, "v2_balanced", "bbb")
    sha, source = checks.recorded_checkpoint_sha(
        "pythia", 42, "v2_balanced", roots=(exported, working))
    assert sha == "aaa"
    assert "results" in source


def test_recorded_sha_skips_a_null_and_keeps_looking(tmp_path):
    """The pre-v2 multiseed cells record null; that is not evidence of absence."""
    exported, working = tmp_path / "results", tmp_path / "work"
    _write_cell(exported, "pythia", 42, "v2_balanced", None)
    _write_cell(working, "pythia", 42, "v2_balanced", "bbb")
    sha, _ = checks.recorded_checkpoint_sha("pythia", 42, "v2_balanced",
                                            roots=(exported, working))
    assert sha == "bbb"


def test_recorded_sha_absent_everywhere(tmp_path):
    sha, source = checks.recorded_checkpoint_sha("pythia", 42, "v2_balanced",
                                                 roots=(tmp_path,))
    assert sha is None and source is None


def test_recorded_sha_survives_unreadable_json(tmp_path):
    cell = tmp_path / "v2_balanced" / "pythia" / "seed_42"
    cell.mkdir(parents=True)
    (cell / "run_config.json").write_text("{ not json", encoding="utf-8")
    assert checks.recorded_checkpoint_sha("pythia", 42, "v2_balanced",
                                          roots=(tmp_path,)) == (None, None)


# ---------------------------------------------------------------------------
# Provenance: original, twin, bit-identical twin, unverifiable
# ---------------------------------------------------------------------------

def test_a_survivor_is_the_original():
    assert checks.checkpoint_provenance("aaa", "aaa", retrained=False) == \
        "original_from_volume"
    # Not retrained, so the digests are not even consulted.
    assert checks.checkpoint_provenance("aaa", "bbb", retrained=False) == \
        "original_from_volume"


def test_a_retrain_that_reproduces_the_digest_is_bit_identical():
    assert checks.checkpoint_provenance("aaa", "aaa", retrained=True) == \
        "retrained_bit_identical"


def test_a_retrain_that_does_not_is_a_twin():
    """The preregistered expected outcome; a mismatch is not a failure."""
    assert checks.checkpoint_provenance("aaa", "bbb", retrained=True) == \
        "retrained_twin"


def test_a_retrain_with_nothing_to_compare_against_says_so():
    assert checks.checkpoint_provenance(None, "bbb", retrained=True) == \
        "retrained_twin_unverifiable"


# ---------------------------------------------------------------------------
# D1 and D2: a sign, and nothing else
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell_fn", [checks.d1_cell, checks.d2_cell])
def test_positive_delta_satisfies(cell_fn):
    assert cell_fn(0.75)["satisfied"] is True


@pytest.mark.parametrize("cell_fn", [checks.d1_cell, checks.d2_cell])
@pytest.mark.parametrize("value", [0.0, -0.75])
def test_nonpositive_delta_falsifies(cell_fn, value):
    """Section 9.5 falsifies on <= 0, so exactly zero is a falsification."""
    assert cell_fn(value)["satisfied"] is False


@pytest.mark.parametrize("cell_fn", [checks.d1_cell, checks.d2_cell])
def test_absent_checkpoint_is_not_a_verdict(cell_fn):
    verdict = cell_fn(None)
    assert verdict["satisfied"] is None
    assert verdict["rule"].startswith("missing_")


# ---------------------------------------------------------------------------
# D3: a ratio, with the degenerate cases preregistered
# ---------------------------------------------------------------------------

def test_d3_passes_at_exactly_the_factor():
    verdict = checks.d3_cell(2.0, 1.0)
    assert verdict["satisfied"] is True
    assert verdict["ratio"] == pytest.approx(2.0)
    assert verdict["rule"] == "ratio"


def test_d3_fails_just_below_the_factor():
    verdict = checks.d3_cell(1.9, 1.0)
    assert verdict["satisfied"] is False
    assert verdict["ratio"] == pytest.approx(1.9)


def test_d3_neutral_nonpositive_satisfies_without_a_ratio():
    """The control moved the gendered gap the other way, or not at all."""
    verdict = checks.d3_cell(1.5, -0.2)
    assert verdict["satisfied"] is True
    assert verdict["ratio"] is None
    assert verdict["rule"] == "neutral_nonpositive"


def test_d3_is_not_meaningful_when_d1_already_failed():
    """A ratio against a non-effect must never read as a pass."""
    verdict = checks.d3_cell(-0.5, -1.0)
    assert verdict["satisfied"] is None
    assert verdict["rule"] == "biased_nonpositive_not_meaningful"


def test_d3_missing_checkpoint():
    assert checks.d3_cell(None, 1.0)["rule"] == "missing_checkpoint"
    assert checks.d3_cell(1.0, None)["satisfied"] is None


# ---------------------------------------------------------------------------
# Aggregation across cells
# ---------------------------------------------------------------------------

def test_all_satisfied_is_the_only_pass():
    assert checks.criterion_status([{"satisfied": True}] * 6) == "PASS"


def test_one_falsified_cell_falsifies_the_criterion():
    verdicts = [{"satisfied": True}] * 5 + [{"satisfied": False}]
    assert checks.criterion_status(verdicts) == "FAIL"


def test_a_falsification_outranks_a_missing_cell():
    """One bad cell falsifies; it does not become undecidable next to a gap."""
    verdicts = [{"satisfied": False}, {"satisfied": None}]
    assert checks.criterion_status(verdicts) == "FAIL"


def test_a_missing_cell_blocks_a_pass():
    verdicts = [{"satisfied": True}] * 5 + [{"satisfied": None}]
    assert checks.criterion_status(verdicts) == "NOT-COMPUTED"


def test_no_cells_is_not_computed():
    assert checks.criterion_status([]) == "NOT-COMPUTED"


def test_report_counts_and_reasons():
    verdicts = [{"satisfied": True, "rule": "sign"},
                {"satisfied": False, "rule": "sign"},
                {"satisfied": None, "rule": "missing_biased_checkpoint"}]
    report = checks.criterion_report("D1", "statement", verdicts)
    assert report["status"] == "FAIL"
    assert (report["n_cells"], report["n_satisfied"]) == (3, 1)
    assert (report["n_falsified"], report["n_not_computed"]) == (1, 1)
    assert report["reasons_not_computed"] == ["missing_biased_checkpoint"]


def test_summary_line_carries_criterion_status_and_counts():
    report = checks.criterion_report("D2", "neutral corpus: Dgap > 0",
                                     [{"satisfied": True}] * 6)
    line = checks.summary_line(report)
    assert line.startswith("D2 ")
    assert "PASS" in line and "6/6" in line and "neutral corpus" in line


# ---------------------------------------------------------------------------
# Dgap itself
# ---------------------------------------------------------------------------

def test_delta_gap_subtracts_the_base():
    base = {"gendered": {"gap": 2.0}}
    fine_tuned = {"gendered": {"gap": 3.25}}
    assert checks.delta_gap(fine_tuned, base, "gendered") == pytest.approx(1.25)


def test_delta_gap_is_none_without_a_checkpoint():
    assert checks.delta_gap(None, {"gendered": {"gap": 2.0}}, "gendered") is None


# ---------------------------------------------------------------------------
# One whole cell, with stubs instead of models
# ---------------------------------------------------------------------------

_SPEC = SimpleNamespace(slug="pythia", label="Pythia-160M",
                        hf_id="EleutherAI/pythia-160m", revision="50f5173d")

_BASE_GAPS = {"gendered": {"gap": 2.0}, "neutral": {"gap": 0.4}}


def _args(**overrides):
    defaults = dict(tag="v2_balanced", no_retrain=False, keep_checkpoints=True,
                    skip_control_c=True)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _stub_gaps(gendered, neutral):
    return {"gendered": {"gap": gendered}, "neutral": {"gap": neutral}}


def _measure_cell(monkeypatch, ensure_result, neutral_dir, ft_gaps, args):
    """Drive measure_cell with every model load replaced by a stub."""
    monkeypatch.setattr(checks, "ensure_biased_checkpoint",
                        lambda *a, **k: ensure_result)
    monkeypatch.setattr(checks, "control_checkpoint_dir",
                        lambda *a, **k: neutral_dir)
    monkeypatch.setattr(checks, "measure_gaps",
                        lambda select, source, device, pairs, revision=None:
                        ft_gaps[Path(source).name])
    corpus = _ROOT / "data" / "biased_corpus_v2_balanced.txt"
    return checks.measure_cell(None, None, _SPEC, 42, _BASE_GAPS,
                               {"gendered": ("man", "woman"),
                                "neutral": ("highlander", "lowlander")},
                               args, "cpu", corpus)


def test_a_complete_cell_records_all_three_verdicts(monkeypatch, tmp_path):
    biased_dir = tmp_path / "finetuned_pythia_v2_balanced_seed42"
    neutral_dir = tmp_path / "ctl_control_B_domain_pythia_seed42"
    biased_dir.mkdir()
    neutral_dir.mkdir()
    record = _measure_cell(
        monkeypatch, (biased_dir, True, "retrained for these checks"), neutral_dir,
        {biased_dir.name: _stub_gaps(3.6, 0.5),
         neutral_dir.name: _stub_gaps(2.3, 1.4)},
        _args())

    # Biased: gendered 3.6 - 2.0 = +1.6.  Neutral: pair 1.4 - 0.4 = +1.0,
    # gendered 2.3 - 2.0 = +0.3, so the ratio is 1.6 / 0.3 > 2.
    assert record["delta_gap"]["biased_gendered"] == pytest.approx(1.6)
    assert record["delta_gap"]["neutral_neutral_pair"] == pytest.approx(1.0)
    assert record["D1"]["satisfied"] is True
    assert record["D2"]["satisfied"] is True
    assert record["D3"]["satisfied"] is True
    assert record["D3"]["ratio"] == pytest.approx(1.6 / 0.3)
    # A retrain that cannot reproduce the recorded digest is a twin, and the
    # record says which digest it compared against.
    assert record["biased_run"]["provenance"] == "retrained_twin"
    assert record["biased_run"]["checkpoint_bit_identical"] is False
    assert record["biased_run"]["checkpoint_sha256_recorded"].startswith("0d32833")


def test_absent_biased_checkpoint_leaves_d2_computable(monkeypatch, tmp_path):
    """--no-retrain: D1 and D3 go NOT-COMPUTED, D2 still answers."""
    neutral_dir = tmp_path / "ctl_control_B_domain_pythia_seed42"
    neutral_dir.mkdir()
    record = _measure_cell(
        monkeypatch, (None, False, "no biased checkpoint on the volume"),
        neutral_dir, {neutral_dir.name: _stub_gaps(2.3, 1.4)},
        _args(no_retrain=True))

    assert record["biased_run"]["provenance"] == "absent"
    assert record["D1"]["satisfied"] is None
    assert record["D3"]["satisfied"] is None
    assert record["D2"]["satisfied"] is True


def test_a_retrained_checkpoint_is_deleted_unless_kept(monkeypatch, tmp_path):
    biased_dir = tmp_path / "finetuned_pythia_v2_balanced_seed42"
    neutral_dir = tmp_path / "ctl_control_B_domain_pythia_seed42"
    biased_dir.mkdir()
    neutral_dir.mkdir()
    _measure_cell(monkeypatch, (biased_dir, True, "retrained"), neutral_dir,
                  {biased_dir.name: _stub_gaps(3.6, 0.5),
                   neutral_dir.name: _stub_gaps(2.3, 1.4)},
                  _args(keep_checkpoints=False))
    assert not biased_dir.exists()
    assert neutral_dir.exists()


def test_a_survivor_is_never_deleted(monkeypatch, tmp_path):
    """Deleting a checkpoint this script did not create would destroy data."""
    biased_dir = tmp_path / "finetuned_pythia_v2_balanced_seed42"
    neutral_dir = tmp_path / "ctl_control_B_domain_pythia_seed42"
    biased_dir.mkdir()
    neutral_dir.mkdir()
    record = _measure_cell(monkeypatch, (biased_dir, False, "survivor"),
                           neutral_dir,
                           {biased_dir.name: _stub_gaps(3.6, 0.5),
                            neutral_dir.name: _stub_gaps(2.3, 1.4)},
                           _args(keep_checkpoints=False))
    assert biased_dir.exists()
    assert record["biased_run"]["provenance"] == "original_from_volume"


# ---------------------------------------------------------------------------
# --no-retrain never trains
# ---------------------------------------------------------------------------

def test_no_retrain_reports_absence_without_touching_the_trainer(tmp_path):
    def explode(*args, **kwargs):
        raise AssertionError("train_full_unfreeze must not be called")

    sweep = SimpleNamespace(training_manifest=lambda *a, **k: {"seed": 42},
                            TRAINING_MANIFEST_NAME="era_training_manifest.json",
                            train_full_unfreeze=explode)
    corpus = _ROOT / "data" / "biased_corpus_v2_balanced.txt"
    ckpt_dir, retrained, note = checks.ensure_biased_checkpoint(
        sweep, _SPEC, 42, corpus, "cpu", "v2_balanced", allow_retrain=False)
    assert ckpt_dir is None and retrained is False
    assert "no biased checkpoint" in note
