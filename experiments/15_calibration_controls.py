#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA — Calibration controls, gate G1c (docs/PREDICTIONS.md §7, §9)
==================================================================

Three controls that make the sweep's numbers interpretable.  Without them
the study has curves but no scale: nothing establishes that the instrument
reads zero when nothing changed, and nothing says what "late" means.

**Control A — serialization (negative control on the instrument).**
The sweep compares a *hub base model* against a *locally saved checkpoint*.
Anything the ``save_pretrained`` round-trip changes on its own — dtype
coercion, a tied ``lm_head`` that reloads untied — would sit inside every
published curve as an unknown offset.  A: saves the unmodified base, reloads
it, and screens base-against-reload.  Everything must be zero.

**Control B — paired domain control.**
Trains on the paired neutral corpus (same 300 sentences, gender attribution
swapped for the terrain attribute selected in §9.4) with the same seeds and
step count, so ``biased − neutral`` isolates the gendered content from the
format adaptation that any corpus of this shape would produce.

**Control C — positive localization control (shallow by construction).**
Trains only the final block (plus the output head where it is untied),
establishing the upper anchor: what this instrument reports when the change
is, by construction, as late as it can be.  The weight-tying branch is
resolved from the config and *verified from the written manifest*, because
with tied embeddings "train the head" silently trains the input embeddings
and the anchor stops being shallow.

All three write full artefacts through ``era.report.save``, and every cell
records device, GPU and library versions so a CPU cell and a GPU cell can
never be confused.

Usage
-----
    python experiments/15_calibration_controls.py --control A
    python experiments/15_calibration_controls.py --control B --seeds 42 43 44
    python experiments/15_calibration_controls.py --control C
    python experiments/15_calibration_controls.py --control all
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import roster as roster_mod  # noqa: E402


def _load_sibling(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, _HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROLS_ROOT = ROOT / "era_poc_calibration_controls"
NEUTRAL_CORPUS = ROOT / "data" / "neutral_corpus_v2_paired.txt"

# The two models the controls anchor, plus the Amendment 2 §8.4 spot-check.
ANCHOR_SLUGS = ("gptneo", "pythia")
SPOTCHECK_SLUG = "opt125m"
SPOTCHECK_SEEDS = (42,)

# Tolerances for Control A, preregistered in docs/PREDICTIONS.md §7.1.
TOL_LOGITS = 1e-6
TOL_DRIFT = 1e-6


# ==============================================================================
# ENVIRONMENT PROVENANCE
# ==============================================================================

def environment_record(device):
    """Device and library identity, recorded in every cell.

    A CPU cell and a GPU cell are different measurements: different kernels,
    different reduction orders, different numerics.  Recording this in every
    artefact is what keeps them from being pooled by accident later.
    """
    import torch
    import transformers

    record = {
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": None,
        "gpu_count": 0,
    }
    if torch.cuda.is_available():
        record["gpu_count"] = torch.cuda.device_count()
        try:
            record["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return record


def resolve_device(requested):
    import torch
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# CONTROL A — SERIALIZATION
# ==============================================================================

def state_dict_manifest_sha256(model):
    """SHA-256 over a canonical per-tensor manifest of a state_dict.

    Mirrors ``era.report.checkpoint_sha256``'s discipline: a sorted JSON
    manifest of ``{name, dtype, shape, sha256(bytes)}`` rather than
    concatenated bytes, so different name/content sequences cannot collide.

    The manifest is **per tensor**, not just a key set, because a key set
    cannot detect weight tying: ``state_dict()`` does not deduplicate shared
    storage, so a tied ``lm_head`` and its embedding both appear under their
    own names.  What reveals the sharing is that the two entries hash
    identically.  That matters here because ``save_pretrained`` *does* drop
    tied entries from the file, so a round-trip can legitimately return a
    model with a different sharing structure — the exact failure this
    control exists to catch.  The key set is returned as well, and
    asymmetric keys are reported rather than silently intersected.
    """
    state = model.state_dict()
    manifest = []
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        manifest.append({
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        })
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, sorted(state), manifest


def max_abs_logit_delta(model_a, model_b, tokenizer, contexts, device):
    """Largest absolute final-position logit difference over the contexts."""
    import torch

    worst = 0.0
    for context in contexts:
        ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            la = model_a(input_ids=input_ids).logits[0, -1, :]
            lb = model_b(input_ids=input_ids).logits[0, -1, :]
        worst = max(worst, float(torch.max(torch.abs(la - lb))))
    return worst


def run_control_a(spec, device, out_root, keep):
    """Save the base unchanged, reload it, screen base-vs-reload."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from era import __version__, save, screen
    from era.contexts import TEST_CONTEXTS
    from era.models import ModelPair, checkpoint_loading_options
    from era.report import tokenizer_vocab_sha256

    out_dir = out_root / "control_A_serialization" / spec.slug
    ckpt_dir = ROOT / f"roundtrip_{spec.slug}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    print(f"   saving unmodified base -> {ckpt_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, revision=spec.revision, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, revision=spec.revision, **checkpoint_loading_options(),
    )
    model.eval()
    before_digest, before_keys, _ = state_dict_manifest_sha256(model)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    reloaded = AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir), **checkpoint_loading_options("safetensors")
    )
    reloaded.eval()
    after_digest, after_keys, _ = state_dict_manifest_sha256(reloaded)

    model = model.to(device)
    reloaded = reloaded.to(device)
    logit_delta = max_abs_logit_delta(model, reloaded, tokenizer, TEST_CONTEXTS, device)
    eval_mode_ok = (not model.training) and (not reloaded.training)
    del model, reloaded
    gc.collect()

    pair = ModelPair(spec.hf_id, str(ckpt_dir), device=device,
                     base_revision=spec.revision)
    result = screen(pair, TEST_CONTEXTS, top_k=20,
                    distribution_metric="k_divergence", verbose=False)
    tok_sha = tokenizer_vocab_sha256(pair.tokenizer)
    del pair
    gc.collect()

    checks = {
        "state_dict_digest_equal": before_digest == after_digest,
        "state_dict_digest_before": before_digest,
        "state_dict_digest_after": after_digest,
        "keys_only_before": sorted(set(before_keys) - set(after_keys)),
        "keys_only_after": sorted(set(after_keys) - set(before_keys)),
        "tokenizer_sha_equal": tok_sha == tokenizer_vocab_sha256(tokenizer),
        "eval_mode_ok": eval_mode_ok,
        "max_abs_logit_delta": logit_delta,
        "max_per_token_mean": float(np.max(result.per_token_mean)),
        "max_relational_mean": float(np.max(result.relational_mean)),
        "max_cka_change": float(np.max(1.0 - result.cka)),
    }
    checks["passes"] = bool(
        checks["state_dict_digest_equal"]
        and not checks["keys_only_before"] and not checks["keys_only_after"]
        and checks["tokenizer_sha_equal"] and checks["eval_mode_ok"]
        and checks["max_abs_logit_delta"] <= TOL_LOGITS
        and checks["max_per_token_mean"] <= TOL_DRIFT
        and checks["max_relational_mean"] <= TOL_DRIFT
        and checks["max_cka_change"] <= TOL_DRIFT
    )

    extra = {
        "era_version": __version__,
        "control": "A_serialization",
        "model_name": spec.hf_id,
        "base_revision_requested": spec.revision,
        "tokenizer_vocab_sha256": tok_sha,
        "regime": "NO_TRAINING_ROUNDTRIP",
        "tolerance_logits": TOL_LOGITS,
        "tolerance_drift": TOL_DRIFT,
        "serialization_checks": checks,
        # Centroids are deliberately NOT interpreted here: drift_centroid
        # returns 0.0 for an all-zero curve, and that 0.0 means "no change",
        # not "change at layer 0" (docs/PREDICTIONS.md §7.1).
        "centroids_meaningless_for_this_control": True,
    }
    extra.update(environment_record(device))
    save(result, out_dir, extra_config=extra)

    if not keep:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    status = "PASS" if checks["passes"] else "FAIL"
    print(f"   [{status}] logits<= {checks['max_abs_logit_delta']:.2e}  "
          f"per_token<= {checks['max_per_token_mean']:.2e}  "
          f"relational<= {checks['max_relational_mean']:.2e}  "
          f"1-CKA<= {checks['max_cka_change']:.2e}")
    if not checks["state_dict_digest_equal"]:
        print(f"   [!] state_dict digest changed; keys only before: "
              f"{checks['keys_only_before']}, only after: {checks['keys_only_after']}")
    return checks


# ==============================================================================
# CONTROL C — PARTIAL UNFREEZE
# ==============================================================================

def resolve_trainable(model, config):
    """Which parameters Control C trains, branching on weight tying.

    With ``tie_word_embeddings`` true, ``lm_head.weight`` **is** the input
    embedding matrix, so training "the head" would train the embeddings and
    the control would silently stop being shallow.  The branch is therefore:

        tied    -> final block only; embeddings AND head frozen
        untied  -> final block + output head

    Returns ``(trainable_names, tied, head_names)``.  Nothing here trusts
    the config alone: tying is confirmed by object identity between the head
    and the input embedding weight where both exist.
    """
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    declared_tied = bool(getattr(config, "tie_word_embeddings", False))
    identity_tied = (
        output_embeddings is not None
        and input_embeddings is not None
        and output_embeddings.weight is input_embeddings.weight
    )
    tied = declared_tied or identity_tied

    # The final block, found structurally: the deepest module in whichever
    # ModuleList holds the transformer layers.
    layer_lists = [
        (name, module) for name, module in model.named_modules()
        if module.__class__.__name__ == "ModuleList" and len(list(module)) > 1
    ]
    if not layer_lists:
        raise ValueError("Could not locate the transformer block list.")
    list_name, block_list = max(layer_lists, key=lambda item: len(list(item[1])))
    final_index = len(list(block_list)) - 1
    final_prefix = f"{list_name}.{final_index}."

    head_names = []
    if output_embeddings is not None and not tied:
        head_ids = {id(p) for p in output_embeddings.parameters()}
        head_names = [n for n, p in model.named_parameters() if id(p) in head_ids]

    trainable = [n for n, _ in model.named_parameters() if n.startswith(final_prefix)]
    trainable += [n for n in head_names if n not in trainable]

    if not trainable:
        raise ValueError(f"No trainable parameters resolved (prefix {final_prefix!r}).")
    return sorted(trainable), tied, sorted(head_names), final_prefix


def apply_partial_unfreeze(model, trainable_names):
    for name, param in model.named_parameters():
        param.requires_grad = name in trainable_names


def verify_shallow(model, trainable_names, tied):
    """Post-hoc check that the regime is what it claims to be."""
    input_embeddings = model.get_input_embeddings()
    embedding_ids = {id(p) for p in input_embeddings.parameters()}
    embedding_names = {n for n, p in model.named_parameters() if id(p) in embedding_ids}
    leaked = sorted(embedding_names & set(trainable_names))
    if tied and leaked:
        raise ValueError(
            "Tied weights leaked into the trainable set: "
            f"{leaked}. The control would train the input embeddings and "
            "would not be shallow by construction."
        )
    return {"embedding_parameter_names": sorted(embedding_names), "leaked": leaked}


# ==============================================================================
# TRAINING (shared by B and C)
# ==============================================================================

def train_cell(sweep, spec, seed, corpus_path, ckpt_dir, device, regime):
    """Fine-tune one cell under ``regime`` and write an extended manifest.

    Reuses the sweep's corpus split, tokenisation, collator and
    hyperparameters verbatim; only the trainable-parameter set varies.  The
    manifest records the trainable parameters BY NAME, the resolved tying
    flag, the active (non-padding) token counts and the loss trajectory —
    the token counts and losses because the corpora are not token-balanced
    and the imbalance is a confounder with no predicted direction
    (docs/PREDICTIONS.md §9.2).
    """
    from datasets import Dataset
    from era.models import checkpoint_loading_options
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)

    from era.report import file_sha256

    sweep.set_all_seeds(seed)
    config = AutoConfig.from_pretrained(
        spec.hf_id, revision=spec.revision, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, revision=spec.revision, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, revision=spec.revision, **checkpoint_loading_options(),
    )

    tying = {}
    if regime == "FULL_UNFREEZE":
        for param in model.parameters():
            param.requires_grad = True
        trainable_names = [n for n, _ in model.named_parameters()]
    elif regime == "PARTIAL_UNFREEZE_LAST_BLOCK":
        trainable_names, tied, head_names, prefix = resolve_trainable(model, config)
        apply_partial_unfreeze(model, trainable_names)
        tying = verify_shallow(model, trainable_names, tied)
        tying.update({"tie_word_embeddings_resolved": tied,
                      "output_head_parameter_names": head_names,
                      "final_block_prefix": prefix})
    else:
        raise ValueError(f"Unknown regime {regime!r}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    train_texts, eval_texts = sweep.load_corpus(corpus_path, seed)
    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts})

    def tok_fn(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=sweep.MAX_LENGTH)

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=["text"])
    # Active (non-padding) tokens: the confounder's magnitude for this run.
    active_train = int(sum(sum(m) for m in train_tok["attention_mask"]))
    active_eval = int(sum(sum(m) for m in eval_tok["attention_mask"]))
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    base_kwargs = dict(
        output_dir=str(ckpt_dir), num_train_epochs=sweep.EPOCHS,
        per_device_train_batch_size=sweep.BATCH_SIZE,
        per_device_eval_batch_size=sweep.BATCH_SIZE,
        learning_rate=sweep.LR, save_strategy="no", logging_steps=20,
        report_to="none", disable_tqdm=True, seed=seed, data_seed=seed,
    )
    args = TrainingArguments(
        **base_kwargs, eval_strategy="epoch", use_cpu=device == "cpu",
        optim="adamw_torch", fp16=False, bf16=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_tok,
                      eval_dataset=eval_tok, data_collator=collator)
    train_output = trainer.train()
    final_eval = trainer.evaluate()

    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in trainer.state.log_history
                   if "eval_loss" in entry]

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    manifest = {
        "model_name": spec.hf_id,
        "base_revision_requested": spec.revision,
        "seed": seed,
        "corpus": corpus_path.name,
        "corpus_sha256": file_sha256(corpus_path),
        "regime": regime,
        "train_max_length": sweep.MAX_LENGTH,
        "train_batch_size": sweep.BATCH_SIZE,
        "train_epochs": sweep.EPOCHS,
        "train_lr": sweep.LR,
        "train_eval_size": sweep.EVAL_SIZE,
        "trainable_parameter_names": trainable_names,
        "n_trainable_parameters": n_trainable,
        "n_total_parameters": n_total,
        "trainable_fraction": n_trainable / max(n_total, 1),
        "weight_tying": tying,
        # Unsigned confounder: logged so it can be analysed afterwards, with
        # no direction predicted for it (docs/PREDICTIONS.md §9.2).
        "active_train_tokens": active_train,
        "active_eval_tokens": active_eval,
        "total_optimizer_steps": int(train_output.global_step),
        "loss_first_logged": losses[0] if losses else None,
        "loss_last_logged": losses[-1] if losses else None,
        "loss_trajectory": losses,
        "eval_loss_trajectory": eval_losses,
        "final_eval_loss": final_eval.get("eval_loss"),
    }
    manifest.update(environment_record(device))
    (ckpt_dir / "era_training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    del trainer, model, train_tok, eval_tok
    gc.collect()
    return manifest


def measure_cell(sweep, spec, ckpt_dir, out_dir, seed, device, corpus_path,
                 control, manifest):
    """Screen a control cell and save the report with full provenance."""
    from era import __version__, save, screen
    from era.contexts import TEST_CONTEXTS
    from era.models import ModelPair
    from era.report import checkpoint_sha256, file_sha256, tokenizer_vocab_sha256

    pair = ModelPair(spec.hf_id, str(ckpt_dir), device=device,
                     base_revision=spec.revision)
    result = screen(pair, TEST_CONTEXTS, top_k=sweep.TOPK_SEMANTIC,
                    distribution_metric=sweep.DISTRIBUTION_METRIC,
                    context_family=sweep.CONTEXT_FAMILY, verbose=False)
    extra = {
        "era_version": __version__,
        "control": control,
        "model_name": spec.hf_id,
        "base_revision_requested": spec.revision,
        "base_commit_hash": pair.base_commit_hash,
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256(pair.tokenizer),
        "seed": seed,
        "regime": manifest["regime"],
        "train_max_length": sweep.MAX_LENGTH,
        "train_batch_size": sweep.BATCH_SIZE,
        "train_epochs": sweep.EPOCHS,
        "train_lr": sweep.LR,
        "train_eval_size": sweep.EVAL_SIZE,
        "corpus": corpus_path.name,
        "corpus_sha256": file_sha256(corpus_path),
        "finetuned_checkpoint_sha256": checkpoint_sha256(ckpt_dir),
        "n_trainable_parameters": manifest["n_trainable_parameters"],
        "trainable_fraction": manifest["trainable_fraction"],
        "trainable_parameter_names": manifest["trainable_parameter_names"],
        "weight_tying": manifest["weight_tying"],
        "active_train_tokens": manifest["active_train_tokens"],
        "active_eval_tokens": manifest["active_eval_tokens"],
        "total_optimizer_steps": manifest["total_optimizer_steps"],
        "loss_first_logged": manifest["loss_first_logged"],
        "loss_last_logged": manifest["loss_last_logged"],
        "final_eval_loss": manifest["final_eval_loss"],
    }
    extra.update(environment_record(device))
    save(result, out_dir, extra_config=extra)
    del pair
    gc.collect()
    return result


# ==============================================================================
# MAIN
# ==============================================================================

def specs_by_slug(slugs):
    specs = roster_mod.apply_pinned_revisions(
        roster_mod.select_tiers(["reference", "A", "B"]))
    table = {s.slug: s for s in specs if s.revision is not None}
    missing = [slug for slug in slugs if slug not in table]
    if missing:
        raise SystemExit(f"No pinned revision for {missing}; run the census "
                         "with --resolve-revisions first.")
    return [table[slug] for slug in slugs]


def run_training_control(sweep, specs, seeds, corpus_path, control, regime,
                         device, out_root, keep):
    for spec in specs:
        for seed in seeds:
            out_dir = out_root / control / spec.slug / f"seed_{seed}"
            if (out_dir / "run_config.json").exists():
                print(f"[SKIP] {control} | {spec.label} | seed {seed}: exists")
                continue
            ckpt_dir = ROOT / f"ctl_{control}_{spec.slug}_seed{seed}"
            print(f"[RUN ] {control} | {spec.label} | seed {seed} | {regime}")
            started = datetime.now(timezone.utc)
            manifest = train_cell(sweep, spec, seed, corpus_path, ckpt_dir,
                                  device, regime)
            print(f"   trainable {manifest['n_trainable_parameters']:,} / "
                  f"{manifest['n_total_parameters']:,} "
                  f"({100 * manifest['trainable_fraction']:.2f}%)"
                  + (f" | tied={manifest['weight_tying'].get('tie_word_embeddings_resolved')}"
                     if manifest["weight_tying"] else ""))
            print(f"   active train tokens {manifest['active_train_tokens']:,} | "
                  f"loss {manifest['loss_first_logged']} -> "
                  f"{manifest['loss_last_logged']} | "
                  f"eval {manifest['final_eval_loss']}")
            result = measure_cell(sweep, spec, ckpt_dir, out_dir, seed, device,
                                  corpus_path, control, manifest)
            centroids = result.centroids
            cka_centroid = centroids["cka_change"]
            if cka_centroid is None:
                cka_text = "undefined"
            else:
                cka_text = (
                    f"{cka_centroid:.2f} "
                    f"(normalized {cka_centroid / (result.num_layers - 1):.3f})"
                )
            print(f"   centroid 1-CKA {cka_text} "
                  f"| {(datetime.now(timezone.utc) - started).total_seconds() / 60:.1f} min")
            if not keep:
                shutil.rmtree(ckpt_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="ERA calibration controls (gate G1c)")
    parser.add_argument("--control", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Retain checkpoints (required for the D1-D3 "
                             "behavioural checks of docs/PREDICTIONS.md §9.5)")
    parser.add_argument("--out", default=str(CONTROLS_ROOT))
    args = parser.parse_args()

    sweep = _load_sibling("multiseed_sweep", "10_multiseed_sweep.py")
    device = resolve_device(args.device)
    out_root = Path(args.out)
    env = environment_record(device)

    print("=" * 78)
    print("ERA CALIBRATION CONTROLS — gate G1c")
    print("=" * 78)
    print(f"Device: {device} | torch {env['torch_version']} | "
          f"transformers {env['transformers_version']}")
    if env["cuda_available"]:
        print(f"GPU:    {env['gpu_name']} (CUDA {env['cuda_version']})")
    print(f"Seeds:  {args.seeds}")
    print(f"Out:    {out_root}")
    print()

    wanted = ["A", "B", "C"] if args.control == "all" else [args.control]

    if "A" in wanted:
        print("--- Control A: serialization (no training) ---")
        summary = {}
        for spec in specs_by_slug(ANCHOR_SLUGS):
            print(f"[RUN ] {spec.label}")
            summary[spec.slug] = run_control_a(spec, device, out_root,
                                               args.keep_checkpoints)
        path = out_root / "control_A_serialization" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"checks": summary, "all_pass": all(c["passes"] for c in summary.values())}
        payload.update(env)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"   summary -> {path}")
        if not payload["all_pass"]:
            raise SystemExit(
                "Control A FAILED. Gate G1c is a hard block on this control: a "
                "non-neutral save/reload round-trip puts an unknown offset "
                "inside every curve in the study. Stopping."
            )
        print()

    if "B" in wanted:
        if not NEUTRAL_CORPUS.exists():
            raise SystemExit(
                f"Neutral corpus missing: {NEUTRAL_CORPUS}. Generate it with "
                "python experiments/01_generate_neutral_corpus.py")
        print("--- Control B: paired domain control (neutral corpus) ---")
        run_training_control(sweep, specs_by_slug(ANCHOR_SLUGS), args.seeds,
                             NEUTRAL_CORPUS, "control_B_domain", "FULL_UNFREEZE",
                             device, out_root, args.keep_checkpoints)
        print()

    if "C" in wanted:
        print("--- Control C: positive localization (partial unfreeze) ---")
        run_training_control(sweep, specs_by_slug(ANCHOR_SLUGS), args.seeds,
                             sweep.DEFAULT_CORPUS, "control_C_localization",
                             "PARTIAL_UNFREEZE_LAST_BLOCK", device, out_root,
                             args.keep_checkpoints)
        # Amendment 2 §8.4: one extra architecture, one seed, to test whether
        # the shallow anchor is specific to the reference pair.
        print("--- Control C spot-check: OPT-125M (Amendment 2 §8.4) ---")
        run_training_control(sweep, specs_by_slug((SPOTCHECK_SLUG,)),
                             list(SPOTCHECK_SEEDS), sweep.DEFAULT_CORPUS,
                             "control_C_localization",
                             "PARTIAL_UNFREEZE_LAST_BLOCK", device, out_root,
                             args.keep_checkpoints)
        print()

    print("=" * 78)
    print(f"CONTROLS COMPLETE — artefacts under {out_root}")
    print("Next: python experiments/98_export_results.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
