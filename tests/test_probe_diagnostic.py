import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "experiments" / "18_probe_diagnostic_high_low.py"
    spec = importlib.util.spec_from_file_location("probe_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_rule_identifies_common_adjective_probe():
    module = _module()
    conclusion = module.conclusion(0.8, 0.6)
    assert "probe defect" in conclusion


def test_diagnostic_rule_returns_corpus_suspicion_for_small_gap():
    module = _module()
    conclusion = module.conclusion(0.2, 0.6)
    assert "corpus" in conclusion


def test_existing_gaps_are_read_without_regeneration(tmp_path):
    module = _module()
    path = tmp_path / "d13.json"
    cells = [
        {"slug": "gptneo", "base_gaps": {"neutral": {"gap": 0.3}}},
        {"slug": "pythia", "base_gaps": {"neutral": {"gap": -0.2}}},
    ]
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    result = module.existing_highlander_gaps(path)
    assert result["gptneo"]["gap"] == 0.3
    assert result["pythia"]["gap"] == -0.2
