"""Small offline tests for the 33-cell reference-metrics runner."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "experiments" / "20_reference_metrics_panel.py"
spec = importlib.util.spec_from_file_location("reference_panel", SCRIPT)
reference_panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_panel)


def test_discovery_uses_saved_identity_and_checkpoint_convention(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_panel, "EXPECTED_CELLS", 1)
    monkeypatch.setattr(reference_panel, "EXPECTED_MODELS", 1)
    cell = tmp_path / "sweep" / reference_panel.TAG / "model_a" / "seed_42"
    cell.mkdir(parents=True)
    config = {
        "model_name": "example/base",
        "base_revision_requested": "commit123",
        "seed": 42,
        "contexts_sha256": reference_panel.context_hash(),
        "tokenizer_vocab_sha256": "tokenizer_hash",
        "finetuned_checkpoint_sha256": "checkpoint_hash",
        "config_fingerprint": "config_hash",
    }
    (cell / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    found = reference_panel.discover_cells(tmp_path / "sweep", tmp_path / "checkpoints")
    assert found[0]["model"] == "example/base"
    assert found[0]["revision"] == "commit123"
    assert found[0]["checkpoint"].endswith("finetuned_model_a_v2_balanced_r2_seed42")


def test_resume_requires_complete_record_and_exact_identity(tmp_path):
    path = tmp_path / "metrics.json"
    identity = {"identity_sha256": "abc"}
    reference_panel.save_json(path, {"complete": True, "identity": identity})
    assert reference_panel.reusable_result(path, identity)["complete"] is True
    assert reference_panel.reusable_result(path, {"identity_sha256": "different"}) is None


def test_short_row_contains_no_alignment_score():
    record = {
        "identity": {"slug": "model_a", "seed": 42},
        "metrics": {
            "aggregates": {
                "B": {"mean": 0.1},
                "B_alpha": {"mean": 0.2},
                "B_k": {"mean": 0.3},
                "B_T": {
                    "total": {"mean": 0.4},
                    "between": {"mean": 0.1},
                    "within": {"mean": 0.3},
                },
            },
            "SI": {"conditional": {"delta_SI": 0.5}},
            "G_l": {
                "drift_1_minus_G": [0.0, 0.2],
                "normalised_depth_centroid_of_drift": 1.0,
            },
        },
    }
    row = reference_panel.short_row(record)
    assert row["B_T"] == 0.4
    assert row["B_between"] + row["B_within"] == row["B_T"]
    assert row["mean_1_minus_G"] == 0.1
    assert "alignment_score" not in row


def test_json_hash_is_independent_of_dictionary_order():
    assert reference_panel.json_hash({"a": 1, "b": 2}) == reference_panel.json_hash(
        {"b": 2, "a": 1}
    )
