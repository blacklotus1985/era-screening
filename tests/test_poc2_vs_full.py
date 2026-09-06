"""Offline tests for the paired POC2-versus-FULL reference experiment."""

import importlib.util
import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

SCRIPT = Path(__file__).parents[1] / "experiments" / "22_poc2_vs_full.py"
spec = importlib.util.spec_from_file_location("poc2_vs_full", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class TinyGPTNeo(nn.Module):
    def __init__(self, tied=True):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(16, 4)
        self.transformer.h = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.transformer.ln_f = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 16, bias=False)
        if tied:
            self.lm_head.weight = self.transformer.wte.weight

    def get_input_embeddings(self):
        return self.transformer.wte

    def get_output_embeddings(self):
        return self.lm_head


def test_poc2_is_the_exact_reference_parameter_set_with_tied_head():
    model = TinyGPTNeo(tied=True)
    report = experiment.configure_trainable_parameters(model, "poc2")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert report["regime"] == "POC2_EXACT"
    assert report["head_tied_to_input_embeddings"] is True
    assert "transformer.wte.weight" in trainable
    assert "transformer.ln_f.weight" in trainable
    assert "transformer.ln_f.bias" in trainable
    assert "transformer.h.2.weight" in trainable
    assert "transformer.h.2.bias" in trainable
    assert not any(name.startswith(("transformer.h.0.", "transformer.h.1.")) for name in trainable)


def test_full_unfreeze_marks_every_parameter_trainable():
    model = TinyGPTNeo(tied=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    report = experiment.configure_trainable_parameters(model, "full")
    assert report["regime"] == "FULL_UNFREEZE"
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_confirmatory_design_is_fixed_to_twelve_paired_cells(tmp_path, monkeypatch):
    corpora = {}
    for name, count in (("original_study", 90), ("balanced_r2", 300)):
        path = tmp_path / f"{name}.txt"
        path.write_text("\n".join(f"line {i}" for i in range(count)), encoding="utf-8")
        corpora[name] = {"path": path, "expected_lines": count}
    monkeypatch.setattr(experiment, "CORPORA", corpora)

    experiment.validate_design()
    assert len(corpora) * len(experiment.REGIMES) * len(experiment.SEEDS) == 12
    corpora["original_study"]["expected_lines"] = 89
    with pytest.raises(ValueError, match="expected 89"):
        experiment.validate_design()


def _record(corpus, regime, seed, offset):
    drift = [0.01 + offset, 0.02 + offset]
    return {
        "identity": {"corpus_name": corpus, "regime": regime, "seed": seed},
        "metrics": {
            "aggregates": {
                "B": {"mean": 0.10 + offset},
                "B_alpha": {"mean": 0.09 + offset},
                "B_k": {"mean": 0.08 + offset},
                "B_T": {
                    "total": {"mean": 0.07 + offset},
                    "between": {"mean": 0.02 + offset / 2},
                    "within": {"mean": 0.05 + offset / 2},
                    "target_mass_m0": {"mean": 0.03},
                    "target_mass_m1": {"mean": 0.20 + offset},
                },
            },
            "SI": {
                "conditional": {
                    "delta_SI": 0.3 + offset,
                    "delta_leadership": 0.1 + offset,
                    "delta_support": -0.2,
                }
            },
            "G_l": {
                "drift_1_minus_G": drift,
                "normalised_depth_centroid_of_drift": 0.7,
            },
        },
    }


def test_summary_is_paired_and_has_no_alignment_score(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "CORPORA",
        {"original_study": {}, "balanced_r2": {}},
    )
    records = []
    for corpus in experiment.CORPORA:
        for regime, offset in (("poc2", 0.0), ("full", 0.1)):
            for seed in experiment.SEEDS:
                records.append(_record(corpus, regime, seed, offset))

    summary = experiment.build_summary(
        records, tuple(experiment.CORPORA), experiment.SEEDS,
        experiment.EXPERIMENT,
    )
    assert len(summary["cells"]) == 12
    assert len(summary["by_condition"]) == 4
    assert len(summary["paired_comparisons"]) == 2
    paired = summary["paired_comparisons"][0]
    assert paired["aggregate"]["B"]["mean"] == pytest.approx(0.1)
    assert paired["drift_1_minus_G_by_layer"][0]["mean"] == pytest.approx(0.1)
    assert "Alignment Score" in summary["statement"]
    assert "alignment_score" not in str(summary).lower()


def test_undefined_depth_centroid_stays_null():
    assert experiment.describe([None, None])["mean"] is None
    assert experiment.paired_difference(None, 0.5) is None


def test_reused_full_record_is_checked_against_sweep_provenance(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "balanced.txt"
    corpus.write_text("balanced corpus", encoding="utf-8")
    monkeypatch.setattr(
        experiment,
        "CORPORA",
        {"balanced_r2": {"path": corpus, "expected_lines": 1}},
    )
    metric_path = tmp_path / "metrics.json"
    config_path = tmp_path / "run_config.json"
    code_hashes = experiment.metric_code_hashes()
    identity = {
        "tag": "v2_balanced_r2",
        "slug": experiment.SLUG,
        "seed": 42,
        "model": experiment.MODEL,
        "revision": experiment.REVISION,
        "context_sha256": experiment.context_hash(),
        "probe_sha256": "probe_hash",
        "checkpoint_sha256": "checkpoint_hash",
        "code_sha256": code_hashes,
    }
    metric_path.write_text(
        json.dumps({
            "complete": True,
            "identity": identity,
            "runtime": {
                "device": "cuda",
                "gpu": "A100",
                "torch": "2.8",
                "transformers": "4.39",
                "numpy": "2.1",
            },
        }),
        encoding="utf-8",
    )
    config = {
        "model_name": experiment.MODEL,
        "base_revision_requested": experiment.REVISION,
        "seed": 42,
        "corpus_sha256": experiment.file_sha256(corpus),
        "regime": "FULL_UNFREEZE",
        "candidate_mode": "topk_union_exact",
        "contexts_sha256": experiment.context_hash(),
        "tokenizer_vocab_sha256": "tokenizer_hash",
        "finetuned_checkpoint_sha256": "checkpoint_hash",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    spec = types.SimpleNamespace(source_sha256="probe_hash")
    runtime = {
        "device": "cuda",
        "gpu": "A100",
        "torch": "2.8",
        "transformers": "4.39",
        "numpy": "2.1",
    }
    result = experiment.validate_reused_full_record(
        metric_path, config_path, 42, spec, "tokenizer_hash", runtime
    )
    assert result["identity"]["seed"] == 42

    config["regime"] = "POC2_EXACT"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Sweep provenance mismatch"):
        experiment.validate_reused_full_record(
            metric_path, config_path, 42, spec, "tokenizer_hash", runtime
        )

    config["regime"] = "FULL_UNFREEZE"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    runtime["gpu"] = "different GPU"
    with pytest.raises(ValueError, match="Runtime mismatch"):
        experiment.validate_reused_full_record(
            metric_path, config_path, 42, spec, "tokenizer_hash", runtime
        )


def test_checkpoint_path_never_collides_across_conditions(tmp_path):
    paths = {
        experiment.checkpoint_path(tmp_path, corpus, regime, seed)
        for corpus in experiment.CORPORA
        for regime in experiment.REGIMES
        for seed in experiment.SEEDS
    }
    assert len(paths) == 12
