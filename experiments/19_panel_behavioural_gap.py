#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exploratory panel-wide behavioural gaps for ``v2_balanced_r2``.

Inference only: read existing sweep cells and retained fine-tuned checkpoints;
never train or alter an existing result.  This extends D1 descriptively to the
full panel and is not a confirmatory analysis.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for path in (str(ROOT), str(_HERE)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

DEFAULT_TAG = "v2_balanced_r2"
DEFAULT_RESULTS_ROOT = ROOT / "results" / "sweep"
DEFAULT_OUT = ROOT / "results" / "aggregates" / (
    "panel_behavioural_gap_v2_balanced_r2.json"
)


def load_script_02():
    """Dynamically load script 02 so its gap implementation is authoritative."""
    path = _HERE / "02_select_neutral_substitutes.py"
    spec = importlib.util.spec_from_file_location("select_neutral_substitutes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_cells(results_root, tag):
    """Read r2 cells and their model/seed identity from run_config files."""
    tag_root = Path(results_root) / tag
    if not tag_root.is_dir():
        raise FileNotFoundError(f"Sweep results directory not found: {tag_root}")
    cells = []
    for config_path in sorted(tag_root.glob("*/seed_*/run_config.json")):
        cell_dir = config_path.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        required = ("model_name", "base_revision_requested", "seed")
        missing = [key for key in required if key not in config]
        if missing:
            cells.append({
                "slug": cell_dir.parent.name,
                "seed": cell_dir.name.removeprefix("seed_"),
                "cell_dir": str(cell_dir),
                "checkpoint": str(ROOT / f"finetuned_{cell_dir.parent.name}_{tag}_"
                                   f"seed{cell_dir.name.removeprefix('seed_')}"),
                "skip_reason": f"run_config.json missing fields: {missing}",
            })
            continue
        seed = int(config["seed"])
        slug = cell_dir.parent.name
        cells.append({
            "slug": slug,
            "seed": seed,
            "model_name": config["model_name"],
            "revision": config["base_revision_requested"],
            "cell_dir": str(cell_dir),
            "run_config": str(config_path),
            # This is the exact checkpoint convention used by script 10.
            "checkpoint": str(ROOT / f"finetuned_{slug}_{tag}_seed{seed}"),
        })
    return cells


def measure_cell(cell, checkpoint_exists, measure):
    """Measure one discovered cell, or return a recorded skip."""
    if "skip_reason" in cell:
        return None, {
            "slug": cell["slug"],
            "seed": cell["seed"],
            "cell_dir": cell["cell_dir"],
            "checkpoint": cell["checkpoint"],
            "reason": cell["skip_reason"],
        }
    if not checkpoint_exists(cell["checkpoint"]):
        return None, {
            "slug": cell["slug"],
            "seed": cell["seed"],
            "cell_dir": cell["cell_dir"],
            "checkpoint": cell["checkpoint"],
            "reason": "fine-tuned checkpoint does not exist on the volume",
        }
    return measure(cell), None


def tokenizer_counts(tokenizer):
    """Count BPE pieces for the two single-word behavioural probes."""
    return {
        " man": len(tokenizer(" man", add_special_tokens=False)["input_ids"]),
        " woman": len(tokenizer(" woman", add_special_tokens=False)["input_ids"]),
    }


def build_record(cell, base_gap, ft_gap, tokenizer):
    """Build one JSON-ready cell record from the two exact gap results."""
    return {
        "slug": cell["slug"],
        "seed": cell["seed"],
        "model": cell["model_name"],
        "base_revision": cell["revision"],
        "sweep_cell": cell["cell_dir"],
        "checkpoint": cell["checkpoint"],
        "base_gap": base_gap,
        "ft_gap": ft_gap,
        "delta_gap": float(ft_gap["gap"] - base_gap["gap"]),
        "tokenizer_piece_counts": tokenizer_counts(tokenizer),
    }


def main():
    parser = argparse.ArgumentParser(description="Exploratory panel behavioural gap")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cells = discover_cells(args.results_root, args.tag)
    select = load_script_02()
    measured = []
    skipped = []
    current_key = None
    current_base = None
    current_tokenizer = None

    for cell in cells:
        def measure(current):
            nonlocal current_key, current_base, current_tokenizer
            key = (current["model_name"], current["revision"])
            if key != current_key:
                if current_base is not None:
                    del current_base
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                current_tokenizer = AutoTokenizer.from_pretrained(
                    current["model_name"], revision=current["revision"]
                )
                current_base = AutoModelForCausalLM.from_pretrained(
                    current["model_name"], revision=current["revision"]
                ).to(device).eval()
                current_key = key
            tokenizer = current_tokenizer
            fine_tuned = AutoModelForCausalLM.from_pretrained(
                current["checkpoint"]
            ).to(device).eval()
            try:
                base_gap = select.leadership_support_gap(
                    current_base, tokenizer, "man", "woman", device
                )
                ft_gap = select.leadership_support_gap(
                    fine_tuned, tokenizer, "man", "woman", device
                )
                return build_record(current, base_gap, ft_gap, tokenizer)
            finally:
                del fine_tuned
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        record, skipped_record = measure_cell(
            cell, lambda path: Path(path).is_dir(), measure
        )
        if record is not None:
            measured.append(record)
        if skipped_record is not None:
            skipped.append(skipped_record)

    if current_base is not None:
        del current_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload = {
        "exploratory": True,
        "preregistered": False,
        "statement": (
            "This inference-only analysis extends D1 descriptively to the full "
            "panel; it has no confirmatory value."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": __import__("numpy").__version__,
        "gap_implementation": (
            "experiments/02_select_neutral_substitutes.py::"
            "leadership_support_gap"
        ),
        "n_probe_contexts": 40,
        "word_pair": ["man", "woman"],
        "cells": measured,
        "skipped_cells": skipped,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Measured cells: {len(measured)} | skipped: {len(skipped)}")
    print(f"Panel behavioural gap written to {out}")


if __name__ == "__main__":
    main()
