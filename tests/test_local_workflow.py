"""Exercise real loading, training and screening without Hub downloads."""

import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM, GPTNeoConfig, GPTNeoForCausalLM, PreTrainedTokenizerFast,
)

from era import save, screen  # noqa: E402
from era.models import ModelPair  # noqa: E402
from era.reference_training import train_checkpoint  # noqa: E402
from era.report import checkpoint_sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_experiment(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("ERA_ALLOW_LEGACY_WEIGHTS", raising=False)
    words = ["[UNK]", "[PAD]", "[EOS]", "A", "leader", "helper", "is", "a",
             "man", "woman", "requires", "support", "leadership"]
    vocabulary = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="[EOS]", model_input_names=["input_ids", "attention_mask"],
    )
    config = GPTNeoConfig(
        vocab_size=len(words), hidden_size=16, num_layers=2, num_heads=2,
        intermediate_size=32, max_position_embeddings=32,
        attention_types=[[["global", "local"], 1]], window_size=8,
        bos_token_id=2, eos_token_id=2, pad_token_id=1,
    )
    torch.manual_seed(42)
    base = tmp_path / "base"
    GPTNeoForCausalLM(config).save_pretrained(base)
    tokenizer.save_pretrained(base)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "A leader is a man\nA helper is a woman\n"
        "A leader requires leadership\nA helper requires support\n",
        encoding="utf-8",
    )
    return base, corpus


def assert_checkpoint_can_be_screened(base, checkpoint, report_dir):
    assert (checkpoint / "model.safetensors").is_file()
    assert checkpoint_sha256(base) != checkpoint_sha256(checkpoint)
    pair = ModelPair(str(base), str(checkpoint), device="cpu")
    result = screen(
        pair, contexts=["A leader is a", "A helper is a"],
        probe_vocab=["man", "woman", "leader", "helper"], verbose=False,
    )
    assert result.num_layers == 3
    assert np.isfinite(result.per_token_mean).all()
    save(result, report_dir)
    report = json.loads((report_dir / "run_config.json").read_text())
    assert report["l2_mean"] > 0


@pytest.mark.parametrize("regime", ["full", "poc2"])
def test_reference_training_saves_a_usable_checkpoint(local_inputs, tmp_path, regime):
    base, corpus = local_inputs
    identity = dict(
        model=str(base), revision="main", regime=regime, seed=42,
        corpus_name="local-smoke", eval_size=1, max_length=12,
        epochs=1, batch_size=2, learning_rate=0.001, device="cpu",
    )
    checkpoint = tmp_path / "trained"
    manifest = train_checkpoint(checkpoint, identity, corpus, local_only=True)
    assert manifest["complete"]
    assert manifest["optimizer_steps"] == 2
    assert np.isfinite(manifest["final_eval_loss"])
    assert_checkpoint_can_be_screened(base, checkpoint, tmp_path / "report")
    assert train_checkpoint(checkpoint, identity, corpus, local_only=True) == manifest


@pytest.mark.parametrize("workflow", ["sweep", "control-full", "control-partial"])
def test_sweep_and_calibration_training_paths(local_inputs, tmp_path, monkeypatch, workflow):
    base, corpus = local_inputs
    sweep = load_experiment("10_multiseed_sweep.py")
    for name, value in {"EPOCHS": 1, "BATCH_SIZE": 2, "MAX_LENGTH": 12,
                        "EVAL_SIZE": 1}.items():
        monkeypatch.setattr(sweep, name, value)
    checkpoint = tmp_path / "trained"
    if workflow == "sweep":
        sweep.train_full_unfreeze(str(base), "main", 42, corpus, checkpoint, "cpu")
    else:
        controls = load_experiment("15_calibration_controls.py")
        spec = SimpleNamespace(hf_id=str(base), revision="main", slug="tiny")
        regime = ("FULL_UNFREEZE" if workflow == "control-full"
                  else "PARTIAL_UNFREEZE_LAST_BLOCK")
        controls.train_cell(sweep, spec, 42, corpus, checkpoint, "cpu", regime)
    assert_checkpoint_can_be_screened(base, checkpoint, tmp_path / "report")


@pytest.mark.parametrize("opt_in", ["argument", "environment"])
def test_mixed_weight_formats_require_opt_in(local_inputs, tmp_path, monkeypatch, opt_in):
    base, _ = local_inputs
    descendant = tmp_path / "safetensors-descendant"
    shutil.copytree(base, descendant)
    model = AutoModelForCausalLM.from_pretrained(base, use_safetensors=True)
    torch.save(model.state_dict(), base / "pytorch_model.bin")
    (base / "model.safetensors").unlink()
    with pytest.raises(OSError, match="safetensors"):
        ModelPair(str(base), str(descendant), device="cpu")
    options = {"weight_format": "legacy"} if opt_in == "argument" else {}
    if opt_in == "environment":
        monkeypatch.setenv("ERA_ALLOW_LEGACY_WEIGHTS", "1")
        with pytest.raises(OSError, match="safetensors"):
            ModelPair(str(base), str(descendant), device="cpu", weight_format="safetensors")
    pair = ModelPair(str(base), str(descendant), device="cpu", **options)
    assert pair.base.dtype == pair.finetuned.dtype == torch.float32
    assert pair.weight_format == "legacy"


def test_reference_training_with_a_trusted_binary_base(local_inputs, tmp_path, monkeypatch):
    base, corpus = local_inputs
    model = AutoModelForCausalLM.from_pretrained(base, use_safetensors=True)
    torch.save(model.state_dict(), base / "pytorch_model.bin")
    (base / "model.safetensors").unlink()
    monkeypatch.setenv("ERA_ALLOW_LEGACY_WEIGHTS", "1")
    identity = dict(
        model=str(base), revision="main", regime="full", seed=42,
        corpus_name="local-smoke", eval_size=1, max_length=12,
        epochs=1, batch_size=2, learning_rate=0.001, device="cpu",
    )
    checkpoint = tmp_path / "trained"
    manifest = train_checkpoint(checkpoint, identity, corpus, local_only=True)
    assert manifest["complete"]
    assert_checkpoint_can_be_screened(base, checkpoint, tmp_path / "report")


def test_loading_preserves_float32_even_for_a_half_precision_checkpoint(local_inputs):
    base, _ = local_inputs
    model = AutoModelForCausalLM.from_pretrained(base, use_safetensors=True)
    model.half().save_pretrained(base)
    pair = ModelPair(str(base), str(base), device="cpu")
    assert pair.base.dtype == pair.finetuned.dtype == torch.float32
