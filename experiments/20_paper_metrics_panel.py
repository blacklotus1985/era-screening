#!/usr/bin/env python
"""Compute the ERA paper metrics on the 33 retained r2 checkpoints.

This is inference only: it never trains or changes a checkpoint.  It first
checks the fixed 14 target tokens and 13 concept tokens on all 11 tokenizers,
then measures one cell at a time and saves it immediately.  Re-running the
command safely resumes cells whose complete identity still matches.
"""

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

from era.contexts import (  # noqa: E402
    LEADERSHIP_CONTEXTS,
    SUPPORT_CONTEXTS,
    TEST_CONTEXTS,
)
from era.paper_probe import (  # noqa: E402
    encode_paper_probe,
    evaluate_observations,
    load_paper_probe_spec,
    measure_model,
)
from era.report import (  # noqa: E402
    checkpoint_sha256,
    file_sha256,
    tokenizer_vocab_sha256,
)

TAG = "v2_balanced_r2"
EXPECTED_CELLS = 33
EXPECTED_MODELS = 11
PROBE = ROOT / "data" / "paper_probe_v1.json"
OUT_ROOT = ROOT / "results" / "paper_metrics"
SCHEMA_VERSION = 1


def json_hash(value) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def context_hash() -> str:
    # Same serialisation used when the r2 sweep was saved.
    text = json.dumps(list(TEST_CONTEXTS), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_cells(results_root, checkpoint_root):
    """Read the 33 identities from their saved run_config.json files."""
    tag_root = Path(results_root) / TAG
    if not tag_root.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {tag_root}")
    required = {
        "model_name",
        "base_revision_requested",
        "seed",
        "contexts_sha256",
        "tokenizer_vocab_sha256",
        "finetuned_checkpoint_sha256",
        "config_fingerprint",
    }
    cells = []
    for path in sorted(tag_root.glob("*/seed_*/run_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"{path} lacks required fields: {missing}")
        slug, seed = path.parent.parent.name, int(config["seed"])
        if path.parent.name != f"seed_{seed}":
            raise ValueError(f"Seed mismatch between path and config: {path}")
        cells.append(
            {
                "slug": slug,
                "seed": seed,
                "model": config["model_name"],
                "revision": config["base_revision_requested"],
                "checkpoint": str(
                    (
                        Path(checkpoint_root) / f"finetuned_{slug}_{TAG}_seed{seed}"
                    ).resolve()
                ),
                "run_config_path": str(path.resolve()),
                "run_config": config,
            }
        )

    models = {(cell["model"], cell["revision"]) for cell in cells}
    if len(cells) != EXPECTED_CELLS or len(models) != EXPECTED_MODELS:
        raise ValueError(
            f"Expected {EXPECTED_CELLS} cells/{EXPECTED_MODELS} models; "
            f"found {len(cells)} cells/{len(models)} models."
        )
    wrong_contexts = [
        f"{cell['slug']}/seed_{cell['seed']}"
        for cell in cells
        if cell["run_config"]["contexts_sha256"] != context_hash()
    ]
    if wrong_contexts:
        raise ValueError(f"Different probe contexts in cells: {wrong_contexts}")
    return cells


def preflight_tokenizers(cells, spec, local_only):
    """Require every fixed word to be one token in every model."""
    from transformers import AutoTokenizer

    tokenizers, encoded_probes, report = {}, {}, []
    representatives = {}
    for cell in cells:
        representatives.setdefault((cell["model"], cell["revision"]), cell)

    for key, cell in sorted(representatives.items(), key=lambda item: item[1]["slug"]):
        tokenizer = AutoTokenizer.from_pretrained(
            cell["model"], revision=cell["revision"], local_files_only=local_only
        )
        encoded = encode_paper_probe(tokenizer, spec)
        vocab_hash = tokenizer_vocab_sha256(tokenizer)
        same_model = [c for c in cells if (c["model"], c["revision"]) == key]
        if any(
            c["run_config"]["tokenizer_vocab_sha256"] != vocab_hash for c in same_model
        ):
            raise ValueError(f"Tokenizer hash mismatch for {cell['slug']}.")
        tokenizers[key], encoded_probes[key] = tokenizer, encoded
        report.append(
            {
                "slug": cell["slug"],
                "model": cell["model"],
                "revision": cell["revision"],
                "target_ids": {
                    name: list(ids) for name, ids in encoded.target_groups.items()
                },
                "concept_ids": list(encoded.concept_ids),
                "tokenizer_vocab_sha256": vocab_hash,
            }
        )
    return tokenizers, encoded_probes, report


def save_json(path, payload):
    """Atomic JSON write: an interruption cannot leave a plausible partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def result_path(out_root, cell):
    return Path(out_root) / TAG / cell["slug"] / f"seed_{cell['seed']}" / "metrics.json"


def code_hashes():
    files = [
        ROOT / "era" / "paper_metrics.py",
        ROOT / "era" / "paper_probe.py",
        Path(__file__),
    ]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in files}


def cell_identity(cell, spec, checkpoint_hash, hashes):
    """Everything that makes an existing result reusable."""
    identity = {
        "schema_version": SCHEMA_VERSION,
        "tag": TAG,
        "slug": cell["slug"],
        "seed": cell["seed"],
        "model": cell["model"],
        "revision": cell["revision"],
        "context_sha256": context_hash(),
        "probe_sha256": spec.source_sha256,
        "checkpoint_sha256": checkpoint_hash,
        "source_config_fingerprint": cell["run_config"]["config_fingerprint"],
        "code_sha256": hashes,
    }
    identity["identity_sha256"] = json_hash(identity)
    return identity


def reusable_result(path, identity):
    if not Path(path).is_file():
        return None
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (
        saved
        if saved.get("complete") is True and saved.get("identity") == identity
        else None
    )


def verify_checkpoint(cell):
    path = Path(cell["checkpoint"])
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    actual = checkpoint_sha256(path)
    expected = cell["run_config"]["finetuned_checkpoint_sha256"]
    if actual != expected:
        raise ValueError(
            f"Checkpoint hash mismatch for {cell['slug']}/seed_{cell['seed']}."
        )
    return actual


def check_model_pair(base, finetuned):
    """The compared models must have identical architecture and vocabulary."""
    fields = ("model_type", "num_hidden_layers", "hidden_size", "vocab_size")
    base_info = {name: getattr(base.config, name, None) for name in fields}
    tuned_info = {name: getattr(finetuned.config, name, None) for name in fields}
    if base_info != tuned_info:
        raise ValueError(f"Model structures differ: {base_info} vs {tuned_info}")
    if (
        base.get_input_embeddings().num_embeddings
        != finetuned.get_input_embeddings().num_embeddings
    ):
        raise ValueError("Input vocabularies differ.")
    return base_info


def short_row(record):
    """One compact row for the panel table; full evidence stays in metrics.json."""
    metrics, identity = record["metrics"], record["identity"]
    aggregate, si, geometry = metrics["aggregates"], metrics["SI"], metrics["G_l"]
    bt, drift = aggregate["B_T"], np.asarray(geometry["drift_1_minus_G"])
    return {
        "slug": identity["slug"],
        "seed": identity["seed"],
        "B": aggregate["B"]["mean"],
        "B_alpha": aggregate["B_alpha"]["mean"],
        "B_k": aggregate["B_k"]["mean"],
        "B_T": bt["total"]["mean"],
        "B_between": bt["between"]["mean"],
        "B_within": bt["within"]["mean"],
        "delta_SI": si["conditional"]["delta_SI"],
        "mean_1_minus_G": float(drift.mean()),
        "G_depth_centroid": geometry["normalised_depth_centroid_of_drift"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(ROOT / "results" / "sweep"))
    parser.add_argument("--checkpoint-root", default=str(ROOT))
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    cells = discover_cells(args.results_root, args.checkpoint_root)
    spec = load_paper_probe_spec(PROBE)
    tokenizers, encoded_probes, preflight = preflight_tokenizers(
        cells, spec, args.local_files_only
    )
    save_json(
        Path(args.out_root) / TAG / "tokenizer_preflight.json",
        {
            "status": "PASS",
            "cells": len(cells),
            "models": preflight,
            "probe_sha256": spec.source_sha256,
            "context_sha256": context_hash(),
        },
    )
    print("Preflight PASS: 11 models, 33 cells, fixed T=14 and K=13.")
    if args.preflight_only:
        return

    families = ["leadership"] * len(LEADERSHIP_CONTEXTS)
    families += ["support"] * len(SUPPORT_CONTEXTS)
    hashes, completed = code_hashes(), []
    grouped = defaultdict(list)
    for cell in cells:
        grouped[(cell["model"], cell["revision"])].append(cell)

    for key, model_cells in sorted(
        grouped.items(), key=lambda item: item[1][0]["slug"]
    ):
        tokenizer, encoded = tokenizers[key], encoded_probes[key]
        pending = []
        for cell in sorted(model_cells, key=lambda item: item["seed"]):
            identity = cell_identity(cell, spec, verify_checkpoint(cell), hashes)
            destination = result_path(args.out_root, cell)
            saved = reusable_result(destination, identity)
            if saved:
                completed.append(saved)
                print(f"[resume] {cell['slug']}/seed_{cell['seed']}")
            else:
                pending.append((cell, identity, destination))
        if not pending:
            continue

        model_name, revision = key
        print(f"[base] {model_cells[0]['slug']}")
        base = (
            AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, local_files_only=args.local_files_only
            )
            .to(device)
            .eval()
        )
        try:
            base_data = measure_model(
                base, tokenizer, TEST_CONTEXTS, encoded.concept_ids, device
            )
            for cell, identity, destination in pending:
                print(f"[cell] {cell['slug']}/seed_{cell['seed']}")
                tuned = (
                    AutoModelForCausalLM.from_pretrained(
                        cell["checkpoint"], local_files_only=True
                    )
                    .to(device)
                    .eval()
                )
                try:
                    structure = check_model_pair(base, tuned)
                    tuned_data = measure_model(
                        tuned, tokenizer, TEST_CONTEXTS, encoded.concept_ids, device
                    )
                    record = {
                        "complete": True,
                        "identity": identity,
                        "runtime": {
                            "generated_utc": datetime.now(timezone.utc).isoformat(),
                            "device": device,
                            "gpu": (
                                torch.cuda.get_device_name(0)
                                if device == "cuda"
                                else None
                            ),
                            "torch": torch.__version__,
                            "transformers": transformers.__version__,
                            "numpy": np.__version__,
                        },
                        "source": {
                            "run_config": cell["run_config_path"],
                            "checkpoint": cell["checkpoint"],
                        },
                        "model_structure": structure,
                        "fixed_token_ids": {
                            "target_groups": {
                                n: list(v) for n, v in encoded.target_groups.items()
                            },
                            "concepts": list(encoded.concept_ids),
                        },
                        "metrics": evaluate_observations(
                            base_data,
                            tuned_data,
                            TEST_CONTEXTS,
                            families,
                            spec,
                            encoded,
                        ),
                    }
                    save_json(destination, record)
                    completed.append(record)
                finally:
                    del tuned
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            del base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(completed) != EXPECTED_CELLS:
        raise ArithmeticError(f"Completed {len(completed)}/{EXPECTED_CELLS} cells.")
    rows = [short_row(record) for record in completed]
    rows.sort(key=lambda row: (row["slug"], row["seed"]))
    save_json(
        Path(args.out_root) / TAG / "panel_summary.json",
        {
            "complete": True,
            "statement": "Observables only: no Alignment Score or automatic label.",
            "cells": rows,
        },
    )
    print(f"Complete: {len(rows)}/33 cells. Results: {Path(args.out_root) / TAG}")


if __name__ == "__main__":
    main()
