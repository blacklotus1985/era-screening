import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiments" / "19_panel_behavioural_gap.py"
spec = importlib.util.spec_from_file_location("panel_gap", SCRIPT)
panel_gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel_gap)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] if text in (" man", " woman") else [1, 2]}


def test_discovery_uses_run_config_and_sweep_checkpoint_convention(tmp_path):
    cell = tmp_path / "v2_balanced_r2" / "gptneo" / "seed_42"
    cell.mkdir(parents=True)
    (cell / "run_config.json").write_text(json.dumps({
        "model_name": "example/base",
        "base_revision_requested": "abc123",
        "seed": 42,
    }), encoding="utf-8")

    discovered = panel_gap.discover_cells(tmp_path, "v2_balanced_r2")
    assert discovered[0]["model_name"] == "example/base"
    assert discovered[0]["revision"] == "abc123"
    assert discovered[0]["checkpoint"].endswith(
        "finetuned_gptneo_v2_balanced_r2_seed42"
    )


def test_missing_checkpoint_is_skipped_without_measurement(tmp_path):
    cell = {
        "slug": "gptneo",
        "seed": 42,
        "cell_dir": str(tmp_path / "cell"),
        "checkpoint": str(tmp_path / "missing"),
        "model_name": "example/base",
        "revision": "abc123",
    }
    called = []
    record, skipped = panel_gap.measure_cell(
        cell, lambda path: False, lambda current: called.append(current)
    )
    assert record is None
    assert not called
    assert skipped["reason"] == "fine-tuned checkpoint does not exist on the volume"


def test_build_record_contains_delta_and_token_piece_counts():
    cell = {
        "slug": "gptneo", "seed": 42, "model_name": "base", "revision": "abc",
        "cell_dir": "results/sweep/v2_balanced_r2/gptneo/seed_42",
        "checkpoint": "finetuned_gptneo_v2_balanced_r2_seed42",
    }
    base = {"gap": 0.1}
    fine_tuned = {"gap": 0.7}
    record = panel_gap.build_record(cell, base, fine_tuned, FakeTokenizer())
    assert record["delta_gap"] == 0.6
    assert record["tokenizer_piece_counts"] == {" man": 1, " woman": 1}
