"""
Tests for era.pipeline and era.report, no torch required.

A FakePair implements the three-method interface of era.models.ModelPair with
hand-designed hidden states: the two models agree perfectly at layers 0-1 and
diverge only at layer 2.  Every downstream number (curves, CKA, centroids,
report files) is then predictable by hand, which is the point: the pipeline
logic is exercised line by line against known geometry.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from era.pipeline import ScreeningResult, output_drift, screen
from era.report import _report_value, config_fingerprint, save


# ---------------------------------------------------------------------------
# The fake model pair
# ---------------------------------------------------------------------------

# Base vectors: one orthogonal unit vector per candidate token.
_BASE_VECTORS = {
    101: np.array([1.0, 0.0, 0.0, 0.0]),
    102: np.array([0.0, 1.0, 0.0, 0.0]),
    103: np.array([0.0, 0.0, 1.0, 0.0]),
}
# At layer 2 the "fine-tuned" model moves tokens 102 and 103; 101 stays put.
_FT_LAYER2 = {
    101: np.array([1.0, 0.0, 0.0, 0.0]),
    102: np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0),
    103: np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2.0),
}
_NUM_LAYERS = 3


class FakePair:
    """Duck-typed stand-in for era.models.ModelPair (see module docstring)."""

    def context_ids(self, context):
        return [len(context)]  # any deterministic list of ints works

    def next_token_distribution(self, which, ctx_ids, top_k=20, semantic_only=True,
                                full=False):
        if which == "base":
            dist = {101: 0.6, 102: 0.4, 103: 0.01}
        else:
            dist = {101: 0.2, 103: 0.8, 102: 0.02}
        if full:
            total = sum(dist.values())
            return {token_id: probability / total for token_id, probability in dist.items()}
        return {101: 0.6, 102: 0.4} if which == "base" else {101: 0.2, 103: 0.8}

    def layer_states(self, which, ctx_ids, candidate_id):
        base_vec = _BASE_VECTORS[candidate_id]
        if which == "base":
            return [base_vec] * _NUM_LAYERS
        # fine-tuned: identical at layers 0-1, moved at layer 2
        return [base_vec, base_vec, _FT_LAYER2[candidate_id]]

    def encode_single_token(self, word):
        # "him" maps to the same ID as "he": the duplicate-detection case.
        table = {"he": 101, "she": 102, "they": 103, "him": 101}
        if word not in table:
            raise ValueError(f"Probe word {word!r} maps to 2 tokens.")
        return table[word]


class FixedSupportPair:
    """Pair whose exact distribution support differs from its fixed probe."""

    def context_ids(self, context):
        return [0]

    def next_token_distribution(self, which, ctx_ids, top_k=20, full=False):
        values = {"base": {11: 0.51, 12: 0.49},
                  "finetuned": {11: 0.49, 12: 0.51}}[which]
        if full:
            return values
        top_id = 11 if which == "base" else 12
        return {top_id: values[top_id]}

    def layer_states(self, which, ctx_ids, candidate_id):
        return [np.array([float(candidate_id), 1.0])]

    def encode_single_token(self, word):
        return {"probe-a": 13, "probe-b": 14}[word]


class UnbiasedZeroDenominatorPair(FixedSupportPair):
    def context_ids(self, context):
        return [int(context)]

    def layer_states(self, which, ctx_ids, candidate_id):
        values = {
            (0, 13): np.array([1.0, 0.0]),
            (0, 14): np.array([1.0, 0.0]),
            (1, 13): np.array([1.0, 0.0]),
            (1, 14): np.array([0.0, 1.0]),
        }
        return [values[(ctx_ids[0], candidate_id)]]


class Float32ConstantPair(FixedSupportPair):
    def layer_states(self, which, ctx_ids, candidate_id):
        return [np.array([0.1, 0.2], dtype=np.float32)]


@pytest.fixture()
def result() -> ScreeningResult:
    return screen(FakePair(), contexts=["ctx one", "ctx two"], verbose=False)


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------

def test_layers_and_rows(result):
    assert result.num_layers == _NUM_LAYERS
    assert len(result.per_context) == 2  # one row per context


def test_drift_is_zero_where_models_agree(result):
    """Layers 0-1: identical states -> zero drift, CKA exactly 1."""
    for layer in (0, 1):
        assert result.per_token_mean[layer] == pytest.approx(0.0)
        assert result.relational_mean[layer] == pytest.approx(0.0)
        assert result.cka[layer] == pytest.approx(1.0)


def test_drift_appears_only_at_the_moved_layer(result):
    """Layer 2: tokens moved -> positive drift on every view, CKA < 1."""
    assert result.per_token_mean[2] > 0.0
    assert result.relational_mean[2] > 0.0
    assert result.cka[2] < 1.0


def test_per_token_value_matches_hand_geometry(result):
    """Mean over {1-cos(v,w)}: token 101 unmoved (0), 102 and 103 moved by
    45 degrees each (1 - 1/sqrt(2))."""
    expected = (0.0 + (1 - 1 / np.sqrt(2)) + (1 - 1 / np.sqrt(2))) / 3.0
    assert result.per_token_mean[2] == pytest.approx(expected)


def test_centroid_sits_on_the_moved_layer(result):
    """All change mass is at layer 2 -> every centroid is exactly 2."""
    for value in result.centroids.values():
        assert value == pytest.approx(2.0)


def test_l2_positive_when_output_distributions_differ(result):
    assert all(row["l2"] > 0.0 for row in result.per_context)


def test_anisotropy_diagnostic_matches_hand_geometry(result):
    """Base candidates are orthogonal at every layer -> anisotropy 0.  The
    fine-tuned vectors at layer 2 have pairwise cosines 1/sqrt(2), 0 and 1/2
    -> mean = (1/sqrt(2) + 0 + 1/2) / 3.  This is the column that tells a
    reader when cosine-based drift is saturated and only CKA should ground
    depth claims."""
    assert np.allclose(result.anisotropy_base, 0.0)
    assert result.anisotropy_ft[0] == pytest.approx(0.0)
    assert result.anisotropy_ft[1] == pytest.approx(0.0)
    expected_layer2 = (1 / np.sqrt(2) + 0.0 + 0.5) / 3.0
    assert result.anisotropy_ft[2] == pytest.approx(expected_layer2)


def test_probe_vocab_mode_uses_fixed_candidates():
    res = screen(
        FakePair(),
        contexts=["ctx"],
        probe_vocab=["he", "she", "they"],
        verbose=False,
    )
    assert res.config["candidate_mode"] == "fixed_probe_vocab"
    assert res.per_context[0]["n_candidates"] == 3


def test_probe_vocab_rejects_multi_token_words():
    with pytest.raises(ValueError, match="maps to 2 tokens"):
        screen(FakePair(), contexts=["ctx"], probe_vocab=["unknown"], verbose=False)


def test_probe_vocab_rejects_duplicate_token_ids():
    """Two probe words mapping to one ID would double-count every metric:
    confirmatory mode must fail with a clear error and name the colliding words."""
    with pytest.raises(ValueError, match="duplicate token IDs"):
        screen(FakePair(), contexts=["ctx"], probe_vocab=["he", "she", "him"],
               verbose=False)


def test_layer_count_mismatch_raises_diagnostic_error():
    """A fine-tuned model returning a different number of layers must produce
    a clear error naming model and context, not a NumPy stacking failure."""
    class MismatchedPair(FakePair):
        def layer_states(self, which, ctx_ids, candidate_id):
            states = super().layer_states(which, ctx_ids, candidate_id)
            return states[:-1] if which == "finetuned" else states

    with pytest.raises(ValueError, match="Layer-count mismatch"):
        screen(MismatchedPair(), contexts=["ctx"], verbose=False)


def test_topk_mass_coverage_recorded(result):
    """The drift value is conditional on the retained top-k mass; every row
    must therefore record how much mass each model kept (FakePair keeps 1.0)."""
    for row in result.per_context:
        assert row["base_topk_mass"] == pytest.approx(1.0)
        assert row["ft_topk_mass"] == pytest.approx(1.0)


def test_exact_union_uses_probability_outside_other_topk():
    class UnequalTopKPair(FakePair):
        def next_token_distribution(
            self, which, ctx_ids, top_k=20, semantic_only=True, full=False
        ):
            if full:
                return {101: 0.70, 102: 0.20, 103: 0.10} if which == "base" else {
                    101: 0.10, 102: 0.20, 103: 0.70
                }
            return {101: 0.70, 102: 0.20} if which == "base" else {
                101: 0.10, 103: 0.70
            }

    exact = screen(UnequalTopKPair(), ["ctx"], top_k=2, verbose=False)
    legacy = screen(UnequalTopKPair(), ["ctx"], top_k=2,
                    candidate_mode="topk_union_legacy", verbose=False)
    row = exact.per_context[0]
    assert row["union_mass_base"] == pytest.approx(1.0)
    assert row["union_mass_ft"] == pytest.approx(1.0)
    assert exact.config["candidate_mode"] == "topk_union_exact"
    assert exact.per_context[0]["l2"] != pytest.approx(legacy.per_context[0]["l2"])


def test_config_records_context_content_hash():
    """The fingerprint must change when probe content changes, path aside."""
    res_a = screen(FakePair(), contexts=["ctx one"], verbose=False)
    res_b = screen(FakePair(), contexts=["ctx two!"], verbose=False)
    assert len(res_a.config["contexts_sha256"]) == 64
    assert res_a.config["contexts_sha256"] != res_b.config["contexts_sha256"]


def test_config_records_probe_vocab_ids():
    res = screen(FakePair(), contexts=["ctx"], probe_vocab=["he", "she", "they"],
                 verbose=False)
    assert res.config["probe_vocab_ids"] == [101, 102, 103]


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown distribution metric"):
        screen(FakePair(), contexts=["ctx"], distribution_metric="kl", verbose=False)


def test_empty_context_list_raises():
    with pytest.raises(ValueError, match="at least one probe context"):
        screen(FakePair(), contexts=[], verbose=False)


def test_invalid_top_k_raises():
    """top_k=0 would fail far downstream as 'nothing to aggregate' and
    negative values inside torch.topk: fail immediately and clearly instead.
    True/False are included because bool is a subclass of int in Python and
    top_k=True would otherwise slip through as 1."""
    for bad in (0, -3, 2.5, True, False):
        with pytest.raises(ValueError, match="top_k"):
            screen(FakePair(), contexts=["ctx"], top_k=bad, verbose=False)


def test_context_hash_distinguishes_element_boundaries():
    """["a", "b"] and ["a\\nb"] must not collide: the hash is over canonical
    JSON, which preserves element boundaries, order and count."""
    res_joined = screen(FakePair(), contexts=["ctx one\nctx two"], verbose=False)
    res_split = screen(FakePair(), contexts=["ctx one", "ctx two"], verbose=False)
    assert res_joined.config["contexts_sha256"] != res_split.config["contexts_sha256"]


def test_output_drift_validates_metric_name():
    with pytest.raises(ValueError, match="Unknown distribution metric"):
        output_drift({1: 1.0}, {1: 1.0}, "alignment_score")


def test_fixed_probe_separates_probability_support_and_geometry():
    result = screen(
        FixedSupportPair(), contexts=["ctx"], top_k=1,
        probe_vocab=["probe-a", "probe-b"], verbose=False,
    )
    row = result.per_context[0]
    assert row["l2"] == pytest.approx(0.0002000133, rel=1e-5)
    assert row["n_candidates"] == 2
    assert row["union_mass_base"] == pytest.approx(1.0)
    assert row["base_topk_mass"] == pytest.approx(0.51)


def test_probability_and_geometry_policies_are_recorded_and_distinct(tmp_path):
    exact = screen(
        FixedSupportPair(), ["ctx"], top_k=1,
        probe_vocab=["probe-a", "probe-b"],
        candidate_mode="topk_union_exact", verbose=False,
    )
    legacy = screen(
        FixedSupportPair(), ["ctx"], top_k=1,
        probe_vocab=["probe-a", "probe-b"],
        candidate_mode="topk_union_legacy", verbose=False,
    )
    assert exact.config["probability_support_policy"] == "topk_union_exact"
    assert exact.config["geometry_selection_mode"] == "fixed_probe_vocab"
    assert exact.config["probability_support_policy"] != legacy.config[
        "probability_support_policy"
    ]
    assert exact.per_context[0]["l2"] != legacy.per_context[0]["l2"]
    exact_payload = json.loads(
        (save(exact, tmp_path / "exact") / "run_config.json").read_text()
    )
    legacy_payload = json.loads(
        (save(legacy, tmp_path / "legacy") / "run_config.json").read_text()
    )
    assert exact_payload["config_fingerprint"] != legacy_payload["config_fingerprint"]


def test_unbiased_cka_zero_denominator_is_undefined_with_layer():
    result = screen(
        UnbiasedZeroDenominatorPair(), ["0", "1"],
        probe_vocab=["probe-a", "probe-b"], verbose=False,
    )
    state = result.config["undefined_metrics"]["cka_unbiased"]
    assert state["status"] == "undefined"
    assert state["layers"] == [0]
    assert np.isfinite(result.cka[0])


def test_float32_identical_rows_are_cka_undefined():
    result = screen(
        Float32ConstantPair(), [str(i) for i in range(40)],
        probe_vocab=["probe-a", "probe-b"], verbose=False,
    )
    assert np.isnan(result.cka[0])
    assert result.config["undefined_metrics"]["cka"]["status"] == "undefined"
    assert result.centroids["cka_change"] is None


def test_valid_zero_cka_change_has_undefined_centroid():
    result = screen(
        FixedSupportPair(), ["ctx"], probe_vocab=["probe-a", "probe-b"],
        verbose=False,
    )
    assert result.cka.tolist() == [1.0]
    assert result.config["undefined_metrics"]["cka_change"]["status"] == "undefined"
    assert result.centroids["cka_change"] is None


def test_undefined_pipeline_metrics_are_explicit_and_serialized_as_null(tmp_path):
    class ConstantPair(FixedSupportPair):
        def layer_states(self, which, ctx_ids, candidate_id):
            return [np.ones(2)]

    result = screen(
        ConstantPair(), contexts=["ctx"], probe_vocab=["probe-a", "probe-b"],
        verbose=False,
    )
    assert result.config["undefined_metrics"]["cka"]["status"] == "undefined"
    assert result.config["undefined_metrics"]["cka_unbiased"]["status"] == "unavailable"
    assert result.config["undefined_metrics"]["centroid_relational"]["status"] == "undefined"
    out = save(result, tmp_path / "constant")
    payload = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert payload["centroid_cka_change"] is None
    assert payload["centroid_relational"] is None
    assert "NaN" not in (out / "run_config.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_save_writes_v1_compatible_files(result, tmp_path):
    out = save(result, tmp_path / "run", extra_config={"seed": 42, "model": "fake"})

    curve = (out / "layer_curve.csv").read_text(encoding="utf-8").splitlines()
    assert curve[0] == ("layer,relational_mean,relational_std,l3_mean,l3_std,"
                        "per_token_mean,per_token_std,cka,cka_unbiased,"
                        "anisotropy_base,anisotropy_ft")
    assert len(curve) == 1 + _NUM_LAYERS  # header + one row per layer

    per_ctx = (out / "per_context_results.csv").read_text(encoding="utf-8").splitlines()
    assert len(per_ctx) == 1 + 2  # header + one row per context

    payload = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    # Flat v1 key names: the aggregation script 11 reads these unchanged.
    assert payload["centroid_per_token"] == pytest.approx(2.0)
    assert payload["argmax_cka_change"] == 2
    assert payload["l2_mean"] > 0.0
    assert len(payload["config_fingerprint"]) == 64  # SHA-256 hex


def test_fingerprint_is_deterministic_and_order_insensitive():
    a = {"seed": 42, "top_k": 20}
    b = {"top_k": 20, "seed": 42}
    assert config_fingerprint(a) == config_fingerprint(b)
    assert config_fingerprint(a) != config_fingerprint({"seed": 43, "top_k": 20})


def test_save_rejects_incomplete_result(tmp_path):
    invalid = ScreeningResult(
        relational_mean=np.array([], dtype=float),
        relational_std=np.array([], dtype=float),
        per_token_mean=np.array([], dtype=float),
        per_token_std=np.array([], dtype=float),
        cka=np.array([], dtype=float),
        per_context=[],
        config={"seed": 42},
    )
    with pytest.raises(ValueError, match="at least one per-context row"):
        save(invalid, tmp_path / "run")


def test_save_preserves_existing_report_when_write_fails(monkeypatch, result, tmp_path):
    out = tmp_path / "run"
    save(result, out, extra_config={"seed": 42, "model": "fake"})
    original = (out / "run_config.json").read_text(encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr("era.report._write_report_files", boom)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        save(result, out, extra_config={"seed": 43, "model": "fake"})

    assert (out / "run_config.json").read_text(encoding="utf-8") == original


@pytest.mark.parametrize("mutate", [
    lambda result: result.per_context[1].__setitem__("l3_layer_1", None),
    lambda result: result.config.__setitem__("num_layers", 999),
])
def test_save_rejects_invalid_replacement_and_preserves_all_files(
        result, tmp_path, mutate):
    out = tmp_path / "run"
    save(result, out)
    original = {name: (out / name).read_bytes() for name in (
        "layer_curve.csv", "per_context_results.csv", "run_config.json")}
    mutate(result)
    with pytest.raises(ValueError):
        save(result, out)
    assert {name: (out / name).read_bytes() for name in original} == original


@pytest.mark.parametrize("extra_name", ["notes.txt", "attachments/notes.txt"])
def test_save_refuses_unrelated_files_without_changing_them(result, tmp_path, extra_name):
    out = tmp_path / "run"
    save(result, out)
    extra = out / extra_name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("Research notes", encoding="utf-8")
    original = {path.relative_to(out): path.read_bytes()
                for path in out.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="dedicated report directory"):
        save(result, out)

    assert {path.relative_to(out): path.read_bytes()
            for path in out.rglob("*") if path.is_file()} == original
    assert set(tmp_path.iterdir()) == {out}


def test_save_preserves_all_files_when_rename_fails(monkeypatch, result, tmp_path):
    out = tmp_path / "run"
    save(result, out)
    original = {name: (out / name).read_bytes() for name in (
        "layer_curve.csv", "per_context_results.csv", "run_config.json")}
    real_rename = Path.rename

    def fail_new_report(path, target):
        if path.name.startswith("run.tmp-") and Path(target) == out:
            raise OSError("simulated rename failure")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_new_report)
    with pytest.raises(OSError, match="simulated rename failure"):
        save(result, out)
    assert {name: (out / name).read_bytes() for name in original} == original


def test_save_rejects_nonfinite_per_context_value(result, tmp_path):
    result.per_context[0]["l2"] = np.nan
    with pytest.raises(ValueError, match="non-finite l2"):
        save(result, tmp_path / "invalid")


def test_save_rejects_malformed_result_and_extra_config(result, tmp_path):
    result.cka[0] = np.nan
    with pytest.raises(ValueError, match="cka contains"):
        save(result, tmp_path / "nan_curve")

    result.cka[0] = 1.0
    result.config["undefined_metrics"] = None
    with pytest.raises(ValueError, match="undefined_metrics"):
        save(result, tmp_path / "bad_state")

    result.config["undefined_metrics"] = {}
    result.per_context[1]["extra"] = 1.0
    with pytest.raises(ValueError, match="inconsistent fields"):
        save(result, tmp_path / "bad_rows")

    result.per_context[1].pop("extra")
    result.per_context[0].pop("context")
    result.per_context[1].pop("context")
    with pytest.raises(ValueError, match="valid context"):
        save(result, tmp_path / "bad_context")

    result.per_context[0]["context"] = "ctx one"
    result.per_context[1]["context"] = "ctx two"
    with pytest.raises(ValueError, match="non-finite numeric"):
        save(result, tmp_path / "bad_extra", extra_config={"bad": np.nan})


def test_save_validates_dimensions_and_scalar_report_values(result, tmp_path):
    result.relational_mean = np.array([])
    with pytest.raises(ValueError, match="at least one layer"):
        save(result, tmp_path / "no_layers")

    result.relational_mean = np.zeros(3)
    result.config["n_contexts_used"] = 999
    with pytest.raises(ValueError, match="n_contexts_used"):
        save(result, tmp_path / "wrong_context_count")

    result.config["n_contexts_used"] = 2
    result.cka = np.zeros(2)
    with pytest.raises(ValueError, match="cka must contain"):
        save(result, tmp_path / "wrong_curve_size")

    with pytest.raises(ValueError, match="non-finite"):
        _report_value(np.inf)
