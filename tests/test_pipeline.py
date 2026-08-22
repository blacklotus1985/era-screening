"""
Tests for era.pipeline and era.report, no torch required.

A FakePair implements the three-method interface of era.models.ModelPair with
hand-designed hidden states: the two models agree perfectly at layers 0-1 and
diverge only at layer 2.  Every downstream number (curves, CKA, centroids,
report files) is then predictable by hand, which is the point: the pipeline
logic is exercised line by line against known geometry.
"""

import json

import numpy as np
import pytest

from era.pipeline import ScreeningResult, output_drift, screen
from era.report import config_fingerprint, save


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
        def next_token_distribution(self, which, ctx_ids, top_k=20,
                                     semantic_only=True, full=False):
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


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_save_writes_v1_compatible_files(result, tmp_path):
    out = save(result, tmp_path / "run", extra_config={"seed": 42, "model": "fake"})

    curve = (out / "layer_curve.csv").read_text(encoding="utf-8").splitlines()
    assert curve[0] == ("layer,l3_mean,l3_std,per_token_mean,per_token_std,"
                        "cka,anisotropy_base,anisotropy_ft")
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
