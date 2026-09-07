from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
from era import models


def _config():
    return SimpleNamespace(
        model_type="dummy",
        num_hidden_layers=2,
        hidden_size=4,
        vocab_size=8,
    )


class _Tokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def get_vocab(self):
        return {"a": 0}


class _Model:
    config = SimpleNamespace(_commit_hash="fixed")

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=8)


def test_default_loading_requires_safetensors_and_disables_remote_code(monkeypatch):
    config_calls = []
    tokenizer_calls = []
    model_calls = []

    def config_loader(name, **kwargs):
        config_calls.append(kwargs)
        return _config()

    def tokenizer_loader(name, **kwargs):
        tokenizer_calls.append(kwargs)
        return _Tokenizer()

    def model_loader(name, **kwargs):
        model_calls.append(kwargs)
        return _Model()

    monkeypatch.setattr(models.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(models.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(models.AutoModelForCausalLM, "from_pretrained", model_loader)

    models.ModelPair("base", "finetuned", device="cpu")

    assert all(call["trust_remote_code"] is False for call in config_calls)
    assert all(call["trust_remote_code"] is False for call in tokenizer_calls)
    assert all(call["trust_remote_code"] is False for call in model_calls)
    assert all(call["use_safetensors"] is True for call in model_calls)


def test_legacy_loading_is_rejected_on_old_pytorch(monkeypatch):
    monkeypatch.setattr(models.torch, "__version__", "2.5.1+cpu")
    with pytest.raises(RuntimeError, match="PyTorch >= 2.6"):
        models.ModelPair("base", "finetuned", weight_format="legacy")


def test_unknown_weight_format_is_rejected():
    with pytest.raises(ValueError, match="weight_format"):
        models.ModelPair("base", "finetuned", weight_format="pickle")
