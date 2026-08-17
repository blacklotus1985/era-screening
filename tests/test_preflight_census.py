"""
Tests for experiments/13_preflight_census.py — no torch required.

The script's torch-dependent parts (model loading, training-step timing) are
imported inside functions, so the module loads in a torch-free environment
via importlib — the same pattern as test_run_screening_io.py.

The load-bearing test here is
``test_anisotropy_profile_matches_the_pipeline_diagnostic``: the census
claims to use the *same* anisotropy convention as ``era.pipeline.screen``.
That claim is what makes the census comparable with every shipped v2 report,
so it is verified against the pipeline rather than asserted in a docstring.
"""

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

from era.pipeline import screen

_SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "13_preflight_census.py"
_spec = importlib.util.spec_from_file_location("preflight_census", _SCRIPT)
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)


# ---------------------------------------------------------------------------
# A duck-typed pair whose base and fine-tuned sides share one top-k set.
#
# With identical distributions the pipeline's candidate set (the UNION of the
# two top-k sets) collapses to the base top-k — exactly the set the census
# uses.  Under that condition the two anisotropy computations must agree to
# the last bit, which is what the equivalence test checks.
# ---------------------------------------------------------------------------

_VECTORS = {
    201: np.array([1.0, 0.0, 0.0]),
    202: np.array([0.6, 0.8, 0.0]),
    203: np.array([0.0, 0.0, 1.0]),
}
_LAYERS = 4
_TOKEN_TEXT = {201: " a", 202: " an", 203: " the"}


class FakePair:
    """Stand-in for era.models.ModelPair with hand-designed geometry."""

    def __init__(self, distribution=None, token_text=None):
        self._dist = distribution if distribution is not None else {201: 0.5, 202: 0.3, 203: 0.2}
        self._text = token_text if token_text is not None else _TOKEN_TEXT

    def context_ids(self, context):
        # Deterministic, and long enough that the "final token" is meaningful:
        # one id per word, with the last word's id chosen from the text table.
        words = context.split()
        tail = next((tid for tid, txt in self._text.items()
                     if txt.strip().lower() == words[-1].lower()), 999)
        return [900 + len(w) for w in words[:-1]] + [tail]

    def decode(self, token_id):
        return self._text.get(int(token_id), "<unk>")

    def next_token_distribution(self, which, ctx_ids, top_k=20, semantic_only=True):
        return dict(self._dist)  # base and finetuned agree -> union == base top-k

    def layer_states(self, which, ctx_ids, candidate_id):
        vec = _VECTORS[candidate_id]
        # A depth-varying rotation so the anisotropy profile is not constant.
        return [vec * (1.0 + 0.1 * layer) + np.array([0.05 * layer, 0.0, 0.0])
                for layer in range(_LAYERS)]


# ---------------------------------------------------------------------------
# The equivalence claim
# ---------------------------------------------------------------------------

def test_anisotropy_profile_matches_the_pipeline_diagnostic():
    """Census anisotropy == era.pipeline.screen's anisotropy_base.

    Guards the documented convention: if either implementation drifts, the
    census stops being comparable with the per-layer anisotropy that every
    v2 report ships, and the confound analysis loses its baseline.
    """
    pair = FakePair()
    contexts = ["A CEO is typically described as a", "Most people assume a manager is an"]

    from_census = census.anisotropy_profile(pair, contexts, top_k=20)["anisotropy"]
    from_pipeline = screen(pair, contexts, verbose=False).anisotropy_base

    assert from_census.shape == from_pipeline.shape == (_LAYERS,)
    np.testing.assert_allclose(from_census, from_pipeline, rtol=0, atol=0)


def test_anisotropy_profile_reports_candidates_and_forward_count():
    pair = FakePair()
    contexts = ["A CEO is typically described as a"]
    out = census.anisotropy_profile(pair, contexts, top_k=20)
    assert [r["n_candidates"] for r in out["per_context"]] == [3]
    assert all(r["used"] for r in out["per_context"])
    # One next-token pass plus one pass per candidate.
    assert out["n_forwards"] == 1 + 3
    assert out["hidden_dims"] == [3]


def test_context_with_fewer_than_two_candidates_is_recorded_not_counted():
    """Same policy as the pipeline: no pair, no measurement — but the context
    is still reported, because a tokenizer that empties the top-k is a
    finding, not a silent omission."""
    pair = FakePair(distribution={201: 1.0})
    with pytest.raises(ValueError, match="at least two candidate tokens"):
        census.anisotropy_profile(pair, ["A CEO is typically described as a"], top_k=20)


def test_layer_count_mismatch_between_contexts_is_fatal():
    class RaggedPair(FakePair):
        def layer_states(self, which, ctx_ids, candidate_id):
            states = super().layer_states(which, ctx_ids, candidate_id)
            return states if len(ctx_ids) > 3 else states[:-1]

    with pytest.raises(ValueError, match="Layer count changed"):
        census.anisotropy_profile(
            RaggedPair(), ["A CEO is typically described as a", "Assume a"], top_k=20)


# ---------------------------------------------------------------------------
# Tokenization census
# ---------------------------------------------------------------------------

def test_tokenization_flags_a_merged_final_article():
    """Every probe context ends in 'a' or 'an'. A tokenizer that folds the
    article into a longer piece is appending the candidate after a different
    prompt than the other models see."""
    pair = FakePair(token_text={201: " a", 202: " an", 203: "described as a"})
    rows = census.tokenization_rows(pair, ["A CEO is typically described as a"])
    assert len(rows) == 1
    assert rows[0]["expected_last_word"] == "a"
    assert rows[0]["last_token_text"] == " a"
    assert rows[0]["degenerate_final_word"] is False

    merged = FakePair(token_text={999: "described as a"})
    rows = census.tokenization_rows(merged, ["A CEO is typically described as a"])
    assert rows[0]["degenerate_final_word"] is True


def test_tokenization_records_counts_and_the_final_token():
    pair = FakePair()
    rows = census.tokenization_rows(pair, ["Most people assume a manager is an"])
    assert rows[0]["n_tokens"] == 7
    assert rows[0]["last_token_id"] == 202
    assert rows[0]["expected_last_word"] == "an"


# ---------------------------------------------------------------------------
# Architecture facts
# ---------------------------------------------------------------------------

def _fake_model(module_names):
    model = types.SimpleNamespace()
    model.named_modules = lambda: [(name, None) for name in module_names]
    return model


def test_config_attr_takes_the_first_present_name():
    cfg = types.SimpleNamespace(n_layer=12, hidden_size=None, n_embd=768)
    assert census.config_attr(cfg, "num_hidden_layers", "n_layer") == 12
    assert census.config_attr(cfg, "hidden_size", "n_embd") == 768
    assert census.config_attr(cfg, "nothing_here") is None


@pytest.mark.parametrize("model_type,modules,expected", [
    ("gpt2", ["transformer.wpe"], "learned_absolute"),
    ("gpt_neo", ["transformer.wpe"], "learned_absolute"),
    ("opt", ["model.decoder.embed_positions"], "learned_absolute"),
    ("gpt_neox", ["layers.0.attention.rotary_emb"], "rotary"),
    ("llama", ["model.layers.0.self_attn.rotary_emb"], "rotary"),
    ("bloom", ["word_embeddings"], "alibi"),
])
def test_positional_encoding_detection(model_type, modules, expected):
    cfg = types.SimpleNamespace(model_type=model_type)
    scheme, _ = census.detect_positional_encoding(cfg, _fake_model(modules))
    assert scheme == expected


def test_positional_encoding_falls_back_to_structure_for_an_unknown_family():
    cfg = types.SimpleNamespace(model_type="brand_new_arch", rope_theta=10000.0)
    scheme, evidence = census.detect_positional_encoding(
        cfg, _fake_model(["model.layers.0.self_attn.rotary_emb"]))
    assert scheme == "rotary"
    assert "rotary_emb" in evidence["modules"]
    assert evidence["rope_config_keys"] == ["rope_theta"]


def test_alibi_config_flag_wins():
    cfg = types.SimpleNamespace(model_type="gpt2", alibi=True)
    scheme, _ = census.detect_positional_encoding(cfg, _fake_model(["transformer.wpe"]))
    assert scheme == "alibi"


# ---------------------------------------------------------------------------
# Schedule arithmetic
# ---------------------------------------------------------------------------

def test_training_steps_matches_the_sweep_split(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(f"sentence {i}" for i in range(300)) + "\n", encoding="utf-8")
    sweep = types.SimpleNamespace(EVAL_SIZE=10, BATCH_SIZE=4, EPOCHS=3)
    train_steps, eval_forwards = census.training_steps(corpus, sweep)
    # 290 training sentences -> ceil(290/4) = 73 steps/epoch, 3 epochs.
    assert train_steps == 219
    assert eval_forwards == 9


def test_training_steps_ignores_blank_lines(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a\n\n b \n\n", encoding="utf-8")
    sweep = types.SimpleNamespace(EVAL_SIZE=0, BATCH_SIZE=4, EPOCHS=1)
    assert census.training_steps(corpus, sweep) == (1, 0)


def test_format_hms():
    assert census.format_hms(None) == "pending"
    assert census.format_hms(0) == "0h00m"
    assert census.format_hms(3661) == "1h01m"
    assert census.format_hms(7 * 3600 + 30 * 60) == "7h30m"
