#!/usr/bin/env python
"""Run the paired GPT-Neo POC2-versus-FULL experiment.

Design: two regimes x two corpora x three seeds.  The original corpus links
directly to the original study; the balanced r2 corpus tests the corrected material.
Three balanced/FULL records already exist and are reused only after strict
provenance checks.  The other nine cells are trained and measured here.
"""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

from era.contexts import LEADERSHIP_CONTEXTS, SUPPORT_CONTEXTS, TEST_CONTEXTS  # noqa: E402
from era.reference_comparison import build_summary, describe, paired_difference  # noqa: E402, F401
from era.reference_probe import (  # noqa: E402
    encode_reference_probe,
    evaluate_observations,
    load_reference_probe_spec,
    measure_model,
)
from era.reference_training import (  # noqa: E402
    TRAINING_MANIFEST,
    configure_trainable_parameters,
    runtime_record,
    save_json,
    train_checkpoint,
)
from era.report import checkpoint_sha256, file_sha256, tokenizer_vocab_sha256  # noqa: E402

EXPERIMENT = "poc2_vs_full_v1"
SCHEMA_VERSION = 1
MODEL = "EleutherAI/gpt-neo-125M"
REVISION = "21def0189f5705e2521767faed922f1f15e7d7db"
SLUG = "gptneo"
SEEDS = (42, 43, 44)
REGIMES = ("poc2", "full")

PROBE = ROOT / "data" / "reference_probe_v1.json"
OUT_ROOT = ROOT / "results" / "reference_metrics" / EXPERIMENT
FULL_R2_ROOT = ROOT / "results" / "reference_metrics" / "v2_balanced_r2"
FULL_R2_SWEEP_ROOT = ROOT / "results" / "sweep" / "v2_balanced_r2"
CORPORA = {
    "original_study": {
        "path": ROOT / "data" / "biased_corpus.txt",
        "expected_lines": 90,
    },
    "balanced_r2": {
        "path": ROOT / "data" / "biased_corpus_v2_balanced.txt",
        "expected_lines": 300,
    },
}

MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5
EVAL_SIZE = 10


def json_hash(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def context_hash():
    text = json.dumps(list(TEST_CONTEXTS), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checkpoint_path(checkpoint_root, corpus_name, regime, seed):
    name = f"finetuned_gptneo_{EXPERIMENT}_{corpus_name}_{regime}_seed{seed}"
    return Path(checkpoint_root) / name


def result_path(out_root, corpus_name, regime, seed):
    return Path(out_root) / corpus_name / regime / f"seed_{seed}" / "metrics.json"


def metric_code_hashes():
    return {
        "era/reference_metrics.py": file_sha256(ROOT / "era" / "reference_metrics.py"),
        "era/reference_probe.py": file_sha256(ROOT / "era" / "reference_probe.py"),
    }


def training_code_hashes():
    return {
        "era/reference_training.py": file_sha256(ROOT / "era" / "reference_training.py"),
        "experiments/22_poc2_vs_full.py": file_sha256(Path(__file__)),
    }


def training_identity(corpus_name, regime, seed, device):
    corpus_path = CORPORA[corpus_name]["path"]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "model": MODEL,
        "revision": REVISION,
        "corpus_name": corpus_name,
        "corpus_sha256": file_sha256(corpus_path),
        "seed": seed,
        "regime": regime,
        "device": device,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "eval_size": EVAL_SIZE,
        "training_code_sha256": training_code_hashes(),
    }


def validate_design():
    for corpus_name, corpus in CORPORA.items():
        text = corpus["path"].read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) != corpus["expected_lines"]:
            raise ValueError(
                f"{corpus_name} has {len(lines)} lines; "
                f"expected {corpus['expected_lines']}."
            )


def validate_reused_full_record(
    metric_path, config_path, seed, spec, tokenizer_hash, expected_runtime
):
    """Check a balanced/FULL record against its original sweep evidence."""
    record = json.loads(Path(metric_path).read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    identity = record.get("identity", {})
    expected_identity = {
        "tag": "v2_balanced_r2",
        "slug": SLUG,
        "seed": seed,
        "model": MODEL,
        "revision": REVISION,
        "context_sha256": context_hash(),
        "probe_sha256": spec.source_sha256,
    }
    if record.get("complete") is not True:
        raise ValueError(f"Incomplete reused result: {metric_path}")
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"Identity mismatch in reused result: {metric_path}")

    hashes = metric_code_hashes()
    saved_hashes = identity.get("code_sha256", {})
    if any(saved_hashes.get(key) != value for key, value in hashes.items()):
        raise ValueError(f"Metric code changed since reused result: {metric_path}")
    if not identity.get("checkpoint_sha256"):
        raise ValueError(f"Missing checkpoint identity: {metric_path}")

    expected_config = {
        "model_name": MODEL,
        "base_revision_requested": REVISION,
        "seed": seed,
        "corpus_sha256": file_sha256(CORPORA["balanced_r2"]["path"]),
        "regime": "FULL_UNFREEZE",
        "candidate_mode": "topk_union_exact",
        "contexts_sha256": context_hash(),
        "tokenizer_vocab_sha256": tokenizer_hash,
        "finetuned_checkpoint_sha256": identity["checkpoint_sha256"],
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"Sweep provenance mismatch: {metric_path}")

    runtime_fields = ("device", "gpu", "torch", "transformers", "numpy")
    saved_runtime = record.get("runtime", {})
    if any(
        saved_runtime.get(key) != expected_runtime.get(key)
        for key in runtime_fields
    ):
        raise ValueError(f"Runtime mismatch in reused result: {metric_path}")
    return record


def new_result_identity(training, checkpoint_hash, manifest, spec):
    value = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "model": MODEL,
        "revision": REVISION,
        "corpus_name": training["corpus_name"],
        "corpus_sha256": training["corpus_sha256"],
        "regime": training["regime"],
        "seed": training["seed"],
        "context_sha256": context_hash(),
        "probe_sha256": spec.source_sha256,
        "checkpoint_sha256": checkpoint_hash,
        "training_identity_sha256": manifest["training_identity_sha256"],
        "metric_code_sha256": metric_code_hashes(),
    }
    value["identity_sha256"] = json_hash(value)
    return value


def reusable_result(path, identity):
    if not Path(path).is_file():
        return None
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("complete") is True and record.get("identity") == identity:
        return record
    return None


def reused_result(source, seed, spec):
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "model": MODEL,
        "revision": REVISION,
        "corpus_name": "balanced_r2",
        "corpus_sha256": file_sha256(CORPORA["balanced_r2"]["path"]),
        "regime": "full",
        "seed": seed,
        "context_sha256": context_hash(),
        "probe_sha256": spec.source_sha256,
        "checkpoint_sha256": source["identity"]["checkpoint_sha256"],
        "metric_code_sha256": metric_code_hashes(),
        "reused_identity_sha256": source["identity"]["identity_sha256"],
    }
    identity["identity_sha256"] = json_hash(identity)
    return {
        "complete": True,
        "identity": identity,
        "runtime": source["runtime"],
        "source": {"kind": "verified_existing_reference_metric"},
        "metrics": source["metrics"],
    }


def train_new_cells(checkpoint_root, device, local_only):
    manifests = {}
    for corpus_name in CORPORA:
        for regime in REGIMES:
            if (corpus_name, regime) == ("balanced_r2", "full"):
                continue
            for seed in SEEDS:
                identity = training_identity(corpus_name, regime, seed, device)
                path = checkpoint_path(checkpoint_root, corpus_name, regime, seed)
                print(f"[train] {corpus_name}/{regime}/seed_{seed}")
                manifests[(corpus_name, regime, seed)] = train_checkpoint(
                    path, identity, CORPORA[corpus_name]["path"], local_only
                )
    return manifests


def measure_new_cell(
    base, base_data, tokenizer, encoded, spec, families, checkpoint, manifest,
    destination, device,
):
    import torch
    from transformers import AutoModelForCausalLM

    checkpoint_hash = checkpoint_sha256(checkpoint)
    identity = new_result_identity(
        manifest["training_identity"], checkpoint_hash, manifest, spec
    )
    saved = reusable_result(destination, identity)
    if saved:
        return saved, True

    tuned = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True
    ).to(device).eval()
    try:
        if type(base) is not type(tuned):
            raise ValueError("Base and fine-tuned model classes differ.")
        tuned_data = measure_model(
            tuned, tokenizer, TEST_CONTEXTS, encoded.concept_ids, device
        )
        record = {
            "complete": True,
            "identity": identity,
            "runtime": runtime_record(device),
            "source": {
                "kind": "trained_by_this_experiment",
                "checkpoint": str(checkpoint),
                "training_manifest": str(checkpoint / TRAINING_MANIFEST),
            },
            "training": {
                key: manifest[key]
                for key in (
                    "n_trainable_parameters",
                    "n_total_parameters",
                    "trainable_fraction",
                    "parameter_report",
                    "final_eval_loss",
                )
            },
            "metrics": evaluate_observations(
                base_data, tuned_data, TEST_CONTEXTS, families, spec, encoded
            ),
        }
        save_json(destination, record)
        return record, False
    finally:
        del tuned
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--checkpoint-root", default=str(ROOT))
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--full-r2-root", default=str(FULL_R2_ROOT))
    parser.add_argument("--full-r2-sweep-root", default=str(FULL_R2_SWEEP_ROOT))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    validate_design()
    current_runtime = runtime_record(device)

    spec = load_reference_probe_spec(PROBE)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.local_files_only
    )
    encoded = encode_reference_probe(tokenizer, spec)
    tokenizer_hash = tokenizer_vocab_sha256(tokenizer)
    reused = {}
    for seed in SEEDS:
        reused[seed] = validate_reused_full_record(
            Path(args.full_r2_root) / SLUG / f"seed_{seed}" / "metrics.json",
            Path(args.full_r2_sweep_root) / SLUG / f"seed_{seed}" / "run_config.json",
            seed, spec, tokenizer_hash, current_runtime,
        )

    probe_model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.local_files_only
    )
    poc2_report = configure_trainable_parameters(probe_model, "poc2")
    poc2_count = sum(p.numel() for p in probe_model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in probe_model.parameters())
    del probe_model
    gc.collect()
    if not 0 < poc2_count < total_count:
        raise ArithmeticError("POC2 is not a strict parameter subset.")

    preflight = {
        "status": "PASS",
        "experiment": EXPERIMENT,
        "model": MODEL,
        "revision": REVISION,
        "runtime": current_runtime,
        "tokenizer_vocab_sha256": tokenizer_hash,
        "probe_sha256": spec.source_sha256,
        "context_sha256": context_hash(),
        "target_tokens": sum(len(ids) for ids in encoded.target_groups.values()),
        "concept_tokens": len(encoded.concept_ids),
        "corpora": {
            name: {
                "sha256": file_sha256(value["path"]),
                "lines": value["expected_lines"],
            }
            for name, value in CORPORA.items()
        },
        "design_cells": 12,
        "new_training_cells": 9,
        "verified_reused_balanced_full_cells": list(SEEDS),
        "poc2_parameter_report": poc2_report,
        "poc2_trainable_parameters": poc2_count,
        "total_parameters": total_count,
        "poc2_trainable_fraction": poc2_count / total_count,
    }
    save_json(Path(args.out_root) / "preflight.json", preflight)
    print("Preflight PASS: GPT-Neo, 12 paired cells; 9 train, 3 verified reuse.")
    if args.preflight_only:
        return

    manifests = train_new_cells(args.checkpoint_root, device, args.local_files_only)
    families = ["leadership"] * len(LEADERSHIP_CONTEXTS)
    families += ["support"] * len(SUPPORT_CONTEXTS)
    print("[base] GPT-Neo-125M")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.local_files_only
    ).to(device).eval()
    base_data = measure_model(base, tokenizer, TEST_CONTEXTS, encoded.concept_ids, device)

    records = []
    try:
        for corpus_name in CORPORA:
            for regime in REGIMES:
                for seed in SEEDS:
                    destination = result_path(args.out_root, corpus_name, regime, seed)
                    if (corpus_name, regime) == ("balanced_r2", "full"):
                        record = reused_result(reused[seed], seed, spec)
                        save_json(destination, record)
                        print(f"[reuse] {corpus_name}/{regime}/seed_{seed}")
                    else:
                        checkpoint = checkpoint_path(
                            args.checkpoint_root, corpus_name, regime, seed
                        )
                        print(f"[measure] {corpus_name}/{regime}/seed_{seed}")
                        record, resumed = measure_new_cell(
                            base, base_data, tokenizer, encoded, spec, families,
                            checkpoint, manifests[(corpus_name, regime, seed)],
                            destination, device,
                        )
                        if resumed:
                            print(f"[measure resume] {corpus_name}/{regime}/seed_{seed}")
                    records.append(record)
    finally:
        del base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(records) != 12:
        raise ArithmeticError(f"Complete records: {len(records)}/12.")
    summary = build_summary(records, tuple(CORPORA), SEEDS, EXPERIMENT)
    save_json(Path(args.out_root) / "comparison_summary.json", summary)
    print(f"Complete: 12/12 cells. Results: {Path(args.out_root)}")


if __name__ == "__main__":
    main()
