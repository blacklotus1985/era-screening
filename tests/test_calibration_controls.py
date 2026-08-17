"""
Tests for experiments/15_calibration_controls.py — no torch model downloads.

The load-bearing logic here is `resolve_trainable`: with tied embeddings,
`lm_head.weight` **is** the input embedding matrix, so a control that
"trains the last block and the head" would train the embeddings and stop
being shallow by construction — producing an anchor that anchors nothing,
silently. These tests pin both branches and the leak check.

Small torch modules stand in for real architectures: torch is needed, but
no network access and no model weights are.
"""

import importlib.util
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

_SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "15_calibration_controls.py"
_spec = importlib.util.spec_from_file_location("calibration_controls", _SCRIPT)
controls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(controls)


# ---------------------------------------------------------------------------
# Stand-in architectures
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """A 3-block decoder with an embedding and an output head.

    `tie` makes `head.weight` the *same tensor* as `embed.weight`, which is
    what GPT-2, GPT-Neo, OPT, BLOOM and SmolLM2 do.
    """

    def __init__(self, tie: bool, n_blocks: int = 3, vocab: int = 16, dim: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, vocab, bias=False)
        if tie:
            self.head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def _config(tied):
    return types.SimpleNamespace(tie_word_embeddings=tied)


# ---------------------------------------------------------------------------
# The tying branch
# ---------------------------------------------------------------------------

def test_untied_trains_final_block_and_head():
    """Pythia-style: the head is a separate matrix, so it is trainable."""
    model = TinyLM(tie=False)
    trainable, tied, head_names, prefix = controls.resolve_trainable(model, _config(False))
    assert tied is False
    assert prefix == "blocks.2."
    assert "blocks.2.weight" in trainable and "blocks.2.bias" in trainable
    assert "head.weight" in trainable
    assert head_names == ["head.weight"]
    # Earlier blocks and the embeddings stay frozen.
    assert not any(n.startswith(("blocks.0.", "blocks.1.")) for n in trainable)
    assert "embed.weight" not in trainable


def test_tied_trains_only_the_final_block():
    """GPT-Neo-style: the head IS the embedding, so neither is trainable.

    This is the case that would silently break the control.
    """
    model = TinyLM(tie=True)
    trainable, tied, head_names, _ = controls.resolve_trainable(model, _config(True))
    assert tied is True
    assert head_names == []          # the head is not separately trainable
    assert trainable == ["blocks.2.bias", "blocks.2.weight"]
    assert "embed.weight" not in trainable
    assert "head.weight" not in trainable


def test_tying_detected_by_identity_even_if_config_lies():
    """Config flags are metadata; shared storage is the fact.

    A config that says untied while the weights are actually shared would
    otherwise put the embeddings into the trainable set.
    """
    model = TinyLM(tie=True)
    _, tied, head_names, _ = controls.resolve_trainable(model, _config(False))
    assert tied is True
    assert head_names == []


def test_final_block_is_the_deepest_one():
    model = TinyLM(tie=False, n_blocks=7)
    trainable, _, _, prefix = controls.resolve_trainable(model, _config(False))
    assert prefix == "blocks.6."
    assert all(n.startswith("blocks.6.") or n == "head.weight" for n in trainable)


def test_missing_block_list_is_fatal():
    class NoBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8, bias=False)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    with pytest.raises(ValueError, match="transformer block list"):
        controls.resolve_trainable(NoBlocks(), _config(False))


# ---------------------------------------------------------------------------
# Applying and verifying the regime
# ---------------------------------------------------------------------------

def test_apply_partial_unfreeze_freezes_everything_else():
    model = TinyLM(tie=False)
    trainable, _, _, _ = controls.resolve_trainable(model, _config(False))
    controls.apply_partial_unfreeze(model, trainable)
    for name, param in model.named_parameters():
        assert param.requires_grad == (name in trainable), name
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert 0 < n_trainable < n_total


def test_verify_shallow_accepts_a_correct_tied_regime():
    model = TinyLM(tie=True)
    trainable, tied, _, _ = controls.resolve_trainable(model, _config(True))
    report = controls.verify_shallow(model, trainable, tied)
    assert report["leaked"] == []


def test_verify_shallow_catches_an_embedding_leak():
    """The guard that makes the manifest a verification rather than a claim."""
    model = TinyLM(tie=True)
    trainable, tied, _, _ = controls.resolve_trainable(model, _config(True))
    with pytest.raises(ValueError, match="Tied weights leaked"):
        controls.verify_shallow(model, trainable + ["embed.weight"], tied)


# ---------------------------------------------------------------------------
# Serialization control primitives
# ---------------------------------------------------------------------------

def test_state_dict_manifest_is_stable_and_sensitive():
    model = TinyLM(tie=False)
    digest_a, keys_a, _ = controls.state_dict_manifest_sha256(model)
    digest_b, keys_b, _ = controls.state_dict_manifest_sha256(model)
    assert digest_a == digest_b and keys_a == keys_b
    with torch.no_grad():
        model.blocks[0].weight[0, 0] += 1.0
    digest_c, _, _ = controls.state_dict_manifest_sha256(model)
    assert digest_c != digest_a, "a changed weight must change the digest"


def test_state_dict_keeps_tied_keys_but_the_manifest_exposes_the_sharing():
    """`state_dict()` does NOT deduplicate shared tensors — both names are
    present either way — so key-set comparison alone cannot detect tying.

    What does detect it is the per-tensor digest: under tying the embedding
    and the head hash identically, because they are one tensor. This is why
    the manifest stores a hash per tensor and not just the key set, and it
    is the signal Control A uses when a round-trip unties weights (HF's
    `save_pretrained` drops tied entries from the file, so a reload can
    legitimately come back with a different sharing structure).
    """
    _, tied_keys, tied_manifest = controls.state_dict_manifest_sha256(TinyLM(tie=True))
    _, untied_keys, untied_manifest = controls.state_dict_manifest_sha256(TinyLM(tie=False))
    assert set(tied_keys) == set(untied_keys)

    def digest_of(manifest, name):
        return next(e["sha256"] for e in manifest if e["name"] == name)

    assert digest_of(tied_manifest, "embed.weight") == digest_of(tied_manifest, "head.weight")
    assert digest_of(untied_manifest, "embed.weight") != digest_of(untied_manifest, "head.weight")


# ---------------------------------------------------------------------------
# Environment provenance
# ---------------------------------------------------------------------------

def test_environment_record_carries_device_and_versions():
    """CPU and GPU cells must never be poolable by accident."""
    record = controls.environment_record("cpu")
    for key in ("device", "torch_version", "transformers_version",
                "cuda_available", "gpu_name", "gpu_count"):
        assert key in record
    assert record["device"] == "cpu"
    assert isinstance(record["cuda_available"], bool)
