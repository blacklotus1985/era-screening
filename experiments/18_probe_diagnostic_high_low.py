#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic probe for the ``high``/``low`` adjective pair.

Inference only.  The pair is scored as single tokens with the exact
``leadership_support_gap`` implementation used by script 02, on the 40 fixed
contexts.  The existing highlander/lowlander base gaps are read from the
already exported D1-D3 JSON; no prior result is recomputed or changed.
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

import roster as roster_mod  # noqa: E402

DEFAULT_D13 = ROOT / "results" / "controls" / "behavioural_checks_D1_D3.json"
DEFAULT_OUT = ROOT / "results" / "controls" / "probe_diagnostic_high_low.json"
REFERENCE_SLUGS = ("gptneo", "pythia")
LARGE_GAP_THRESHOLD = 0.5


def load_script_02():
    path = _HERE / "02_select_neutral_substitutes.py"
    spec = importlib.util.spec_from_file_location("select_neutral_substitutes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def existing_highlander_gaps(path):
    """Return the per-model base neutral gaps from the committed D1-D3 JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = {}
    for cell in payload["cells"]:
        slug = cell["slug"]
        if slug in REFERENCE_SLUGS:
            values.setdefault(slug, []).append(cell["base_gaps"]["neutral"])
    missing = [slug for slug in REFERENCE_SLUGS if slug not in values]
    if missing:
        raise ValueError(f"D1-D3 JSON has no base neutral gaps for {missing}.")
    return {
        slug: {
            "gap": float(sum(item["gap"] for item in rows) / len(rows)),
            "n_cells": len(rows),
            "source": str(path),
        }
        for slug, rows in values.items()
    }


def conclusion(high_low_gap, highlander_gap):
    """Apply the preregistered diagnostic rule without interpreting further."""
    same_sign = (
        high_low_gap == 0.0 or highlander_gap == 0.0
        or (high_low_gap > 0.0) == (highlander_gap > 0.0)
    )
    high_low_large = abs(high_low_gap) >= LARGE_GAP_THRESHOLD
    if high_low_large and same_sign:
        return (
            "high/low base gap is large and has the same sign as "
            "highlander/lowlander: D2 probe may be dominated by common adjectives "
            "(probe defect)."
        )
    return (
        "high/low base gap is small or has a different sign: suspicion returns "
        "to the corpus."
    )


def main():
    parser = argparse.ArgumentParser(description="Diagnostic high/low adjective probe")
    parser.add_argument("--d13-json", default=str(DEFAULT_D13),
                        help="Existing exported D1-D3 JSON")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Require already cached Hugging Face files")
    args = parser.parse_args()

    import torch
    from era.models import checkpoint_loading_options
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    select = load_script_02()
    specs = roster_mod.apply_pinned_revisions(roster_mod.select_tiers(["reference"]))
    old_gaps = existing_highlander_gaps(args.d13_json)
    measured = {}

    for spec in specs:
        if spec.slug not in REFERENCE_SLUGS:
            continue
        load_kwargs = {"revision": spec.revision}
        if args.local_files_only:
            load_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id,
            **load_kwargs,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            **load_kwargs,
            **checkpoint_loading_options(),
        )
        model = model.to(device).eval()
        try:
            gap = select.leadership_support_gap(
                model, tokenizer, "high", "low", device
            )
        finally:
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        high_low = float(gap["gap"])
        highlander = old_gaps[spec.slug]["gap"]
        measured[spec.slug] = {
            "label": spec.label,
            "model": spec.hf_id,
            "revision": spec.revision,
            "high_low_base": gap,
            "highlander_lowlander_base": {
                "gap": highlander,
                "source": old_gaps[spec.slug]["source"],
                "n_cells": old_gaps[spec.slug]["n_cells"],
            },
            "criterion": {
                "large_gap_threshold_nats": LARGE_GAP_THRESHOLD,
                "same_sign": bool(
                    high_low == 0.0 or highlander == 0.0
                    or (high_low > 0.0) == (highlander > 0.0)
                ),
                "high_low_is_large": abs(high_low) >= LARGE_GAP_THRESHOLD,
            },
            "conclusion": conclusion(high_low, highlander),
        }

    payload = {
        "_comment": (
            "Inference-only diagnostic. Numbers only; no D1-D3 result is "
            "modified or regenerated."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "n_probe_contexts": 40,
        "pair": ["high", "low"],
        "gap_definition": (
            "mean_leadership[logP(X)-logP(Y)] - "
            "mean_support[logP(X)-logP(Y)]"
        ),
        "gap_implementation": (
            "experiments/02_select_neutral_substitutes.py::"
            "leadership_support_gap"
        ),
        "comparison_pair": ["highlander", "lowlander"],
        "models": measured,
        "preregistered_rule": (
            "If the base high/low gap is large (absolute gap >= 0.5 nats) "
            "and has the same sign as the base highlander/lowlander gap, "
            "the D2 probe was dominated by common adjectives (probe defect); "
            "if it is small or has a different sign, suspicion returns to the "
            "corpus. Do not interpret beyond these numbers and this criterion."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Diagnostic written to {out}")


if __name__ == "__main__":
    main()
