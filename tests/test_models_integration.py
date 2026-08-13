"""
Integration tests for era.models.ModelPair, the module the unit suite cannot
reach (it needs torch, transformers and a model download).

Skipped by default so `pytest tests/` stays fast and offline.  Enable with:

    ERA_RUN_INTEGRATION=1 pytest tests/test_models_integration.py -v

The model-vs-itself screening is the canonical negative control: identical
checkpoints must show ~zero drift and CKA ~1 at every layer.
"""

import os

import numpy as np
import pytest

if not os.environ.get("ERA_RUN_INTEGRATION"):
    pytest.skip(
        "integration tests need torch + model download; set ERA_RUN_INTEGRATION=1",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")

from era import screen  # noqa: E402
from era.models import ModelPair  # noqa: E402

# Default: the smallest model of the published panel.  Override with
# ERA_INTEGRATION_MODEL (e.g. a tiny hf-internal-testing model) to run the
# same behavioural checks cheaply in CI.
MODEL = os.environ.get("ERA_INTEGRATION_MODEL", "EleutherAI/pythia-160m")


@pytest.fixture(scope="module")
def self_pair() -> ModelPair:
    return ModelPair(MODEL, MODEL, device="cpu")


def test_self_screening_shows_no_drift(self_pair):
    """Negative control: a model compared with itself must be flat.

    Uses two contexts to keep runtime low; the point is the zero baseline,
    not coverage.
    """
    result = screen(
        self_pair,
        contexts=["A CEO is typically described as a",
                  "A nurse is typically described as a"],
        top_k=10,
        verbose=False,
    )
    assert np.allclose(result.per_token_mean, 0.0, atol=1e-6)
    assert np.allclose(result.relational_mean, 0.0, atol=1e-6)
    assert np.allclose(result.cka, 1.0, atol=1e-6)


def test_layer_states_reads_the_candidate_position(self_pair):
    """The returned states must actually come from the candidate's position.

    Regression guard for an accidental ``h[0, 0, :]`` (first position) read:
    with the SAME context and two DIFFERENT candidates, the layer-0 state
    (embedding output at the last position) must differ, it embeds the
    candidate.  A first-position read would return the context's first token
    for both candidates and make the states identical.
    """
    ctx_ids = self_pair.context_ids("A CEO is typically described as a")
    cand_a = self_pair.encode_single_token(" man")
    cand_b = self_pair.encode_single_token(" woman")

    states_a = self_pair.layer_states("base", ctx_ids, cand_a)
    states_b = self_pair.layer_states("base", ctx_ids, cand_b)

    hidden = self_pair.base.config.hidden_size
    assert all(vec.shape == (hidden,) for vec in states_a)
    # Embedding-layer states must differ across candidates (same position,
    # same positional embedding -> the difference is the candidate itself).
    assert not np.allclose(states_a[0], states_b[0])
    # And be identical for the same candidate (determinism).
    states_a2 = self_pair.layer_states("base", ctx_ids, cand_a)
    assert np.allclose(states_a[0], states_a2[0])


def test_encode_single_token_rejects_multi_token_words(self_pair):
    with pytest.raises(ValueError, match="single-token"):
        self_pair.encode_single_token("unquestionably long compound phrase")


def test_mismatched_architecture_raises():
    with pytest.raises(ValueError, match="Architecture mismatch"):
        ModelPair(MODEL, "EleutherAI/gpt-neo-125M", device="cpu")


def test_same_architecture_different_size_raises():
    """Pythia-70m shares model_type with Pythia-160m but has fewer layers and
    a smaller hidden size: the structural checks must refuse the pair,
    from the configs alone, before any weights are loaded."""
    with pytest.raises(ValueError, match="mismatch"):
        ModelPair(MODEL, "EleutherAI/pythia-70m", device="cpu")


def test_tokenizer_check_is_fail_closed(self_pair, tmp_path):
    """The two sides of the fail-closed policy, on a real local checkpoint.

    1. Checkpoint WITHOUT tokenizer files: explicit opt-in to the base
       tokenizer, loading must succeed.
    2. Same checkpoint with a CORRUPT tokenizer.json: the mapping check can
       no longer run, so loading must fail with a clear error, never skip
       the check silently.
    """
    import gc

    ckpt = tmp_path / "ckpt"
    self_pair.finetuned.save_pretrained(ckpt)  # weights + config, no tokenizer

    pair_ok = ModelPair(MODEL, str(ckpt), device="cpu")  # case 1: loads
    del pair_ok  # release the two extra models before the next construction
    gc.collect()

    (ckpt / "tokenizer.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer"):
        ModelPair(MODEL, str(ckpt), device="cpu")  # case 2: fails closed
    gc.collect()
