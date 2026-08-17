#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA — Preflight + anisotropy census (inference only, no training)
=================================================================

Run this BEFORE the extended sweep.  For every model in the roster it
answers three questions without fine-tuning anything:

1. **Can the v2 screen even run on this architecture?**  Layer count, hidden
   size, vocabulary, positional-encoding type, parameter count, pad-token
   handling, and — the check that actually bites — whether every layer's
   hidden state has the *same* dimensionality (OPT-350M projects its
   embeddings, so this is not a formality; `era.metrics.cosine_similarity`
   raises on a shape mismatch).

2. **Does the fixed probe set survive this tokenizer?**  The 40 contexts of
   ``era.contexts`` are tokenized with each model's own tokenizer; token
   counts and the final token are recorded, and a context whose final word
   does not tokenize to a standalone ``a``/``an`` is flagged.  The realized
   candidate count per context is recorded too: candidates come from a
   top-k that ``era.models.is_semantic`` filters by decoded string, so a
   tokenizer with different granularity yields a different sample size for
   the CKA estimator.

3. **How anisotropic is each base model, layer by layer?**  This is the
   cheap, high-value deliverable: the confound documented in
   ``docs/FINDINGS_v2_balanced.md`` is that cosine-based drift saturates
   where a layer's mean pairwise cosine approaches 1.  With ~12
   architectures the census shows whether the extreme profiles of
   GPT-Neo-125M and Pythia-160M are common or rare.

Anisotropy convention (identical to the diagnostic the v2 pipeline ships)
------------------------------------------------------------------------
For each probe context, the candidate token IDs are the model's own
semantic-filtered top-k next tokens.  Each candidate is appended to the
context *by token ID* (never re-tokenized as text), and the hidden state at
that final position is read at every layer.  Per layer, anisotropy is the
mean cosine over all candidate *pairs*; the reported value is the mean over
contexts.  This mirrors ``era.pipeline.screen``'s ``anisotropy_base``
computation exactly, with one documented difference: ``screen`` takes the
union of the base and fine-tuned top-k sets, while a census has no
fine-tuned model, so the candidate set is the base top-k alone.  The
measurement primitives are inherited from ``era.models.ModelPair``, not
re-implemented.

Outputs (under ``results/census/``)
-----------------------------------
    per_model/<slug>.json        raw per-model record (the resume unit)
    anisotropy_census.csv        long format: one row per (model, layer)
    model_census.csv             one row per model: architecture + timings
    tokenization_census.csv      one row per (model, context)
    anisotropy_census.png        overlay on normalized depth
    skipped_models.json          every model that failed, and why
    census_README.md             conventions, caveats, sweep schedule

Usage
-----
    python experiments/13_preflight_census.py --resolve-revisions
    python experiments/13_preflight_census.py --tiers reference A
    python experiments/13_preflight_census.py --tiers B --threads 8
    python experiments/13_preflight_census.py --models "My-Model=org/id=myslug"
"""

import argparse
import gc
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import bootstrap — deliberate, and load-bearing.
#
# `python experiments/<script>.py` puts experiments/ (not the repo root) on
# sys.path, so a bare `import era` resolves to whatever `era` pip has
# installed — which on a machine with an editable install of a *different*
# ERA checkout is a different codebase.  This study must measure with the
# `era/` package that sits next to this script, so the repo root is placed
# ahead of everything else.  This pins WHICH copy of era is imported; it
# changes no measurement.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import roster as roster_mod  # noqa: E402  (must follow the bootstrap above)


def _load_sibling(module_name: str, filename: str):
    """Import a sibling script whose filename is not a valid module name.

    ``10_multiseed_sweep.py`` starts with a digit, so it cannot be imported
    normally.  It is the single source of truth for the training and
    measurement hyperparameters, and duplicating them here would let the
    census estimate a schedule for a configuration the sweep does not run.
    """
    spec = importlib.util.spec_from_file_location(module_name, _HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename} from {_HERE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CENSUS_ROOT = ROOT / "results" / "census"
PER_MODEL_DIR = CENSUS_ROOT / "per_model"

# Mean pairwise cosine at or above this value is treated as a *saturated*
# layer: the cosine-based drift metrics there are compressed toward zero
# regardless of what the fine-tune changed (docs/FINDINGS_v2_balanced.md).
# Fixed before the sweep and justified in docs/PREDICTIONS.md; it is a
# round interpretability threshold, not a calibrated statistic, so the
# neighbouring values are reported alongside it and the full per-layer
# profile is written out for any other threshold to be applied post hoc.
SATURATION_THRESHOLD = 0.95
SENSITIVITY_THRESHOLDS = (0.90, 0.95, 0.99)

# Positional encoding by HF model_type.  Structural detection (below) is the
# primary evidence; this table resolves families whose scheme is baked into
# the forward pass rather than into a module (BLOOM builds its ALiBi bias
# tensor inline, so there is no module to find).
POS_ENCODING_BY_MODEL_TYPE = {
    "bloom": "alibi",
    "gpt2": "learned_absolute",
    "gpt_neo": "learned_absolute",
    "gptj": "rotary",
    "gpt_neox": "rotary",
    "llama": "rotary",
    "mistral": "rotary",
    "opt": "learned_absolute",
}


# ==============================================================================
# ARCHITECTURE FACTS
# ==============================================================================

def config_attr(config, *names):
    """First present, non-None attribute among ``names`` (configs disagree on
    which name they use for the same fact)."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def detect_positional_encoding(config, model):
    """``(scheme, evidence)`` — how this model encodes position.

    Structural evidence (module names actually present in the loaded graph)
    is recorded alongside the resolved scheme, so a wrong table entry is
    visible in the census rather than silently believed.
    """
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    evidence = sorted(
        n for n in module_names
        if any(marker in n.lower() for marker in ("rotary", "wpe", "embed_positions",
                                                  "position_embeddings", "alibi"))
    )
    rope_config = [
        attr for attr in ("rope_theta", "rotary_pct", "rotary_emb_base",
                          "partial_rotary_factor", "rope_scaling")
        if getattr(config, attr, None) is not None
    ]

    model_type = getattr(config, "model_type", "")
    if bool(getattr(config, "alibi", False)):
        scheme = "alibi"
    elif model_type in POS_ENCODING_BY_MODEL_TYPE:
        scheme = POS_ENCODING_BY_MODEL_TYPE[model_type]
    elif any("rotary" in n.lower() for n in evidence) or rope_config:
        scheme = "rotary"
    elif any(n in ("wpe", "embed_positions", "position_embeddings") for n in evidence):
        scheme = "learned_absolute"
    else:
        scheme = "unknown"

    return scheme, {"modules": evidence, "rope_config_keys": rope_config}


# ==============================================================================
# TOKENIZATION CENSUS
# ==============================================================================

def tokenization_rows(pair, contexts):
    """One record per probe context, under this model's tokenizer.

    ``degenerate`` marks a context whose final token is not the standalone
    article the context was written to end on.  Every probe context ends in
    " a" or " an"; if a tokenizer merges that article into a longer piece,
    the candidate token is being appended after something other than what the
    other models see, and the cross-model comparison at that context is not
    measuring the same prompt.
    """
    rows = []
    for index, context in enumerate(contexts):
        ids = pair.context_ids(context)
        last_text = pair.decode(ids[-1]) if ids else ""
        expected = context.rsplit(" ", 1)[-1].strip().lower()
        degenerate = (not ids) or last_text.strip().lower() != expected
        rows.append({
            "context_index": index,
            "context": context,
            "n_tokens": len(ids),
            "last_token_id": int(ids[-1]) if ids else None,
            "last_token_text": last_text,
            "expected_last_word": expected,
            "degenerate_final_word": bool(degenerate),
        })
    return rows


# ==============================================================================
# ANISOTROPY CENSUS
# ==============================================================================

def anisotropy_profile(pair, contexts, top_k):
    """Per-layer mean pairwise cosine of the base model over ``contexts``.

    Mirrors the ``aniso_base`` block of ``era.pipeline.screen`` (the
    ``sims_base`` accumulation): same primitives, same pairwise-cosine
    convention, same mean-over-contexts aggregation.  Returns the profile,
    per-context diagnostics, and the raw forward-pass count for the timing
    estimate.
    """
    from era.metrics import cosine_similarity

    per_context = []
    profiles = []
    hidden_dims = set()
    n_forwards = 0
    n_layers = None

    for index, context in enumerate(contexts):
        ctx_ids = pair.context_ids(context)
        dist = pair.next_token_distribution("base", ctx_ids, top_k=top_k)
        n_forwards += 1
        candidates = sorted(dist)
        if len(candidates) < 2:
            # Same policy as the pipeline: a context with fewer than two
            # candidates has no pair to measure and contributes nothing.
            per_context.append({
                "context_index": index,
                "n_candidates": len(candidates),
                "topk_mass": float(sum(dist.values())),
                "used": False,
            })
            continue

        states = {}
        for candidate in candidates:
            states[candidate] = pair.layer_states("base", ctx_ids, candidate)
            n_forwards += 1
            hidden_dims.update(int(h.shape[-1]) for h in states[candidate])

        layer_counts = {len(v) for v in states.values()}
        if len(layer_counts) != 1:
            raise ValueError(
                f"Layer-count mismatch across candidates in context {context!r}: "
                f"{sorted(layer_counts)}."
            )
        if n_layers is None:
            n_layers = layer_counts.pop()
        elif n_layers != next(iter(layer_counts)):
            raise ValueError(
                f"Layer count changed between contexts: {n_layers} then "
                f"{next(iter(layer_counts))}."
            )

        profile = np.zeros(n_layers)
        for layer in range(n_layers):
            sims = [
                cosine_similarity(states[candidates[i]][layer], states[candidates[j]][layer])
                for i in range(len(candidates))
                for j in range(i + 1, len(candidates))
            ]
            profile[layer] = float(np.mean(sims))
        profiles.append(profile)
        per_context.append({
            "context_index": index,
            "n_candidates": len(candidates),
            "topk_mass": float(sum(dist.values())),
            "used": True,
        })

    if not profiles:
        raise ValueError(
            "No probe context yielded at least two candidate tokens; the "
            "semantic top-k filter emptied every context for this tokenizer."
        )

    return {
        "anisotropy": np.vstack(profiles).mean(axis=0),
        "per_context": per_context,
        "hidden_dims": sorted(hidden_dims),
        "n_forwards": n_forwards,
    }


# ==============================================================================
# TIMING
# ==============================================================================

def time_training_step(hf_id, revision, corpus_path, sweep, n_timed=3):
    """Seconds per optimizer step, measured on a real batch shape.

    A plain forward/backward/step loop rather than ``Trainer``: the estimate
    only needs the dominant cost, and running the real Trainer would train
    the model this script promises not to train.  The batch shape is exactly
    what the sweep produces (``padding="max_length"`` makes every batch
    ``BATCH_SIZE x MAX_LENGTH``), so the shape is not an approximation — the
    omitted Trainer overhead (dataloading, logging, per-epoch eval) is, and
    it makes this a mild *under*-estimate.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision)
    model.train()

    with open(corpus_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    batch_texts = lines[: sweep.BATCH_SIZE]
    encoded = tokenizer(batch_texts, padding="max_length", truncation=True,
                        max_length=sweep.MAX_LENGTH, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    # Causal-LM labels, matching DataCollatorForLanguageModeling(mlm=False):
    # padded positions are masked out of the loss.
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=sweep.LR)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        optimizer.step()

    one_step()  # warm-up: first step pays lazy allocation and kernel selection
    timings = []
    for _ in range(n_timed):
        start = time.perf_counter()
        one_step()
        timings.append(time.perf_counter() - start)

    result = float(np.median(timings))
    # Rebind rather than `del`: `one_step` closes over these names, and the
    # optimizer state alone is ~2x the parameter memory — it must be released
    # before the next model in the roster loads.
    optimizer = model = tokenizer = None
    gc.collect()
    return result


def training_steps(corpus_path, sweep):
    """``(train_steps, eval_forwards)`` the sweep will run for one seed."""
    with open(corpus_path, "r", encoding="utf-8") as handle:
        n_sentences = sum(1 for line in handle if line.strip())
    n_train = max(n_sentences - sweep.EVAL_SIZE, 0)
    steps_per_epoch = ceil(n_train / sweep.BATCH_SIZE)
    eval_per_epoch = ceil(sweep.EVAL_SIZE / sweep.BATCH_SIZE)
    return steps_per_epoch * sweep.EPOCHS, eval_per_epoch * sweep.EPOCHS


# ==============================================================================
# PER-MODEL CENSUS
# ==============================================================================

def census_one_model(spec, contexts, sweep, corpus_path, device, do_timing):
    """Full preflight record for one model.  Raises on any blocking defect."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from era.models import ModelPair
    from era.report import tokenizer_vocab_sha256

    class BaseOnlyPair(ModelPair):
        """One model behind the full ``ModelPair`` measurement interface.

        A census has a base model and no fine-tuned descendant, so
        ``ModelPair.__init__``'s two-checkpoint loading and comparability
        checks do not apply — but every *primitive* below them does, and the
        census must use the identical convention (token-ID construction of
        ``context + candidate``, semantic top-k, last-position hidden
        states).  Subclassing inherits those primitives verbatim instead of
        re-implementing them; ``finetuned`` aliases ``base`` so the inherited
        ``_model`` dispatch stays total.
        """

        def __init__(self, hf_id, revision, device):
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
            self.tokenizer_had_pad_token = self.tokenizer.pad_token is not None
            if self.tokenizer.pad_token is None:
                # Exactly what 10_multiseed_sweep.py and ModelPair do.
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base = AutoModelForCausalLM.from_pretrained(
                hf_id, revision=revision).to(device).eval()
            self.finetuned = self.base
            self.base_commit_hash = getattr(self.base.config, "_commit_hash", None)
            self.finetuned_commit_hash = self.base_commit_hash

    config = AutoConfig.from_pretrained(spec.hf_id, revision=spec.revision)

    load_start = time.perf_counter()
    pair = BaseOnlyPair(spec.hf_id, spec.revision, device)
    load_seconds = time.perf_counter() - load_start

    try:
        scheme, evidence = detect_positional_encoding(config, pair.base)
        n_params = int(sum(p.numel() for p in pair.base.parameters()))

        pad_ok = pair.tokenizer.pad_token is not None
        tokenization = tokenization_rows(pair, contexts)

        forward_start = time.perf_counter()
        aniso = anisotropy_profile(pair, contexts, sweep.TOPK_SEMANTIC)
        forward_seconds = time.perf_counter() - forward_start

        profile = aniso["anisotropy"]
        n_curve_points = len(profile)
        n_blocks = n_curve_points - 1  # hidden_states = embeddings + one per block

        declared_layers = config_attr(config, "num_hidden_layers", "num_layers", "n_layer")
        if declared_layers is not None and int(declared_layers) != n_blocks:
            raise ValueError(
                f"Hidden-state count disagrees with the config: {n_curve_points} "
                f"states implies {n_blocks} blocks, config declares "
                f"{declared_layers}. The layer axis would be mislabelled."
            )

        if len(aniso["hidden_dims"]) != 1:
            raise ValueError(
                f"Hidden states are not of uniform dimensionality across layers: "
                f"{aniso['hidden_dims']}. era.metrics.cosine_similarity requires "
                "equal-length vectors, so this architecture cannot be screened "
                "layer-by-layer as-is."
            )

        candidate_counts = [r["n_candidates"] for r in aniso["per_context"] if r["used"]]
        record = {
            "label": spec.label,
            "hf_id": spec.hf_id,
            "slug": spec.slug,
            "tier": spec.tier,
            "revision": spec.revision,
            "note": spec.note,
            "seeds": spec.seeds,
            "resolved_commit_hash": pair.base_commit_hash,
            "model_type": getattr(config, "model_type", None),
            "n_params": n_params,
            "n_blocks": n_blocks,
            "n_curve_points": n_curve_points,
            "hidden_size": int(aniso["hidden_dims"][0]),
            "config_hidden_size": config_attr(config, "hidden_size", "n_embd"),
            "word_embed_proj_dim": getattr(config, "word_embed_proj_dim", None),
            "vocab_size_config": config_attr(config, "vocab_size"),
            "vocab_size_embedding": int(pair.base.get_input_embeddings().num_embeddings),
            "tokenizer_vocab_size": int(len(pair.tokenizer)),
            "tokenizer_vocab_sha256": tokenizer_vocab_sha256(pair.tokenizer),
            "positional_encoding": scheme,
            "positional_encoding_evidence": evidence,
            "tokenizer_had_pad_token": bool(pair.tokenizer_had_pad_token),
            "pad_token": pair.tokenizer.pad_token,
            "eos_token": pair.tokenizer.eos_token,
            "pad_token_resolved": bool(pad_ok),
            "n_contexts": len(contexts),
            "n_contexts_used": len(candidate_counts),
            "n_degenerate_contexts": sum(
                1 for r in tokenization if r["degenerate_final_word"]),
            "context_tokens_min": min(r["n_tokens"] for r in tokenization),
            "context_tokens_max": max(r["n_tokens"] for r in tokenization),
            "context_tokens_mean": float(np.mean([r["n_tokens"] for r in tokenization])),
            "candidates_min": int(min(candidate_counts)),
            "candidates_max": int(max(candidate_counts)),
            "candidates_mean": float(np.mean(candidate_counts)),
            "n_cka_samples": int(sum(candidate_counts)),
            "anisotropy": [float(v) for v in profile],
            "anisotropy_min": float(profile.min()),
            "anisotropy_max": float(profile.max()),
            "anisotropy_mean": float(profile.mean()),
            "n_saturated_layers": int(np.sum(profile >= SATURATION_THRESHOLD)),
            "saturation_threshold": SATURATION_THRESHOLD,
            # Sensitivity: the 0.95 cut is a round number, so the count at
            # the neighbouring thresholds travels with it and a reader can
            # see whether a conclusion depends on the exact choice.
            "n_saturated_by_threshold": {
                str(t): int(np.sum(profile >= t)) for t in SENSITIVITY_THRESHOLDS
            },
            "tokenization": tokenization,
            "per_context_candidates": aniso["per_context"],
            "model_load_seconds": load_seconds,
            "census_forward_seconds": forward_seconds,
            "census_n_forwards": aniso["n_forwards"],
            "seconds_per_forward": forward_seconds / max(aniso["n_forwards"], 1),
        }
    finally:
        # The alias must go first: deleting `base` alone leaves the model
        # alive through `finetuned` and the next model loads on top of it.
        pair.finetuned = None
        pair.base = None
        del pair
        gc.collect()

    # Screening runs the same forward work for BOTH checkpoints of the pair.
    record["est_screen_seconds"] = 2.0 * record["census_forward_seconds"]

    if do_timing:
        step_seconds = time_training_step(spec.hf_id, spec.revision, corpus_path, sweep)
        train_steps, eval_forwards = training_steps(corpus_path, sweep)
        record["seconds_per_train_step"] = step_seconds
        record["train_steps_per_seed"] = train_steps
        record["eval_forwards_per_seed"] = eval_forwards
        record["est_train_seconds_per_seed"] = step_seconds * train_steps
        record["est_seconds_per_seed"] = (
            step_seconds * train_steps
            + record["est_screen_seconds"]
            + 2.0 * record["model_load_seconds"]
        )
        record["est_total_seconds"] = record["est_seconds_per_seed"] * len(spec.seeds)
    else:
        for key in ("seconds_per_train_step", "train_steps_per_seed",
                    "eval_forwards_per_seed", "est_train_seconds_per_seed",
                    "est_seconds_per_seed", "est_total_seconds"):
            record[key] = None

    record["torch_version"] = torch.__version__
    record["census_timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return record


# ==============================================================================
# AGGREGATION / OUTPUT
# ==============================================================================

def all_per_model_records():
    """Every per-model record on disk, ordered by parameter count.

    The aggregates must describe the whole census, not just the tier that
    happened to run last: `--tiers B` after `--tiers reference A` would
    otherwise overwrite the panel-wide CSVs, plot and README with four rows
    and silently lose six. The per-model JSON files are the durable unit;
    the aggregates are always rebuilt from all of them.
    """
    records = []
    for path in sorted(PER_MODEL_DIR.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"   [WARN] unreadable census record {path.name}: {exc!r} — skipped")
    return sorted(records, key=lambda r: r.get("n_params", 0))


def merge_skipped(current, records):
    """Skip entries for the whole panel, not just this run's tier.

    Carries forward skips recorded by earlier runs, drops any model that has
    since been measured successfully, and lets this run's entry win on a
    conflict. Without this, censusing one tier would erase the record of a
    failure in another — and a failure that vanishes from the artefacts is
    indistinguishable from a model that was never in the panel.
    """
    measured = {rec.get("hf_id") for rec in records}
    merged = {}
    path = CENSUS_ROOT / "skipped_models.json"
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("skipped", [])
        except (json.JSONDecodeError, OSError):
            previous = []
        for entry in previous:
            if entry.get("hf_id") not in measured:
                merged[entry.get("hf_id")] = entry
    for entry in current:
        merged[entry.get("hf_id")] = entry
    for hf_id in measured:
        merged.pop(hf_id, None)
    return [merged[key] for key in sorted(merged, key=lambda k: (k is None, k))]


def write_outputs(records, skipped, sweep, corpus_path, meta):
    """Write every census artefact from the per-model records."""
    import csv

    CENSUS_ROOT.mkdir(parents=True, exist_ok=True)

    with open(CENSUS_ROOT / "anisotropy_census.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slug", "label", "hf_id", "tier", "n_blocks", "layer",
                         "normalized_depth", "anisotropy", "saturated"])
        for rec in records:
            n_blocks = rec["n_blocks"]
            for layer, value in enumerate(rec["anisotropy"]):
                writer.writerow([
                    rec["slug"], rec["label"], rec["hf_id"], rec["tier"], n_blocks,
                    layer, layer / n_blocks if n_blocks else 0.0, value,
                    int(value >= SATURATION_THRESHOLD),
                ])

    model_columns = [
        "slug", "label", "hf_id", "tier", "revision", "model_type", "n_params",
        "n_blocks", "n_curve_points", "hidden_size", "word_embed_proj_dim",
        "vocab_size_embedding", "tokenizer_vocab_size", "positional_encoding",
        "tokenizer_had_pad_token", "n_contexts_used", "n_degenerate_contexts",
        "context_tokens_min", "context_tokens_max", "context_tokens_mean",
        "candidates_min", "candidates_max", "candidates_mean", "n_cka_samples",
        "anisotropy_min", "anisotropy_mean", "anisotropy_max",
        "n_saturated_layers", "model_load_seconds", "seconds_per_forward",
        "est_screen_seconds", "seconds_per_train_step", "train_steps_per_seed",
        "est_train_seconds_per_seed", "est_seconds_per_seed", "est_total_seconds",
    ]
    with open(CENSUS_ROOT / "model_census.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=model_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    with open(CENSUS_ROOT / "tokenization_census.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slug", "label", "context_index", "context", "n_tokens",
                         "last_token_id", "last_token_text", "expected_last_word",
                         "degenerate_final_word", "n_candidates", "topk_mass"])
        for rec in records:
            by_index = {c["context_index"]: c for c in rec["per_context_candidates"]}
            for row in rec["tokenization"]:
                cand = by_index.get(row["context_index"], {})
                writer.writerow([
                    rec["slug"], rec["label"], row["context_index"], row["context"],
                    row["n_tokens"], row["last_token_id"], row["last_token_text"],
                    row["expected_last_word"], int(row["degenerate_final_word"]),
                    cand.get("n_candidates"), cand.get("topk_mass"),
                ])

    with open(CENSUS_ROOT / "skipped_models.json", "w", encoding="utf-8") as handle:
        json.dump({"timestamp_utc": meta["timestamp_utc"], "skipped": skipped},
                  handle, indent=2, ensure_ascii=False)

    plot_anisotropy(records)
    write_census_readme(records, skipped, sweep, corpus_path, meta)


def plot_anisotropy(records):
    """Overlay every profile on normalized depth (x = layer / n_blocks)."""
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5))
    cmap = plt.get_cmap("tab20")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
    for index, rec in enumerate(sorted(records, key=lambda r: r["n_params"])):
        profile = np.array(rec["anisotropy"])
        n_blocks = max(rec["n_blocks"], 1)
        depth = np.arange(len(profile)) / n_blocks
        ax.plot(depth, profile, marker=markers[index % len(markers)], markersize=4,
                linewidth=1.8, color=cmap(index % 20),
                label=f"{rec['label']} ({rec['n_blocks']}L)")
    ax.axhline(SATURATION_THRESHOLD, color="black", linestyle=":", linewidth=1.4)
    ax.annotate(f"saturation ≥ {SATURATION_THRESHOLD}", xy=(0.02, SATURATION_THRESHOLD),
                xytext=(0.02, SATURATION_THRESHOLD - 0.06), fontsize=9)
    ax.set_xlabel("Normalized depth (layer / number of blocks; "
                  "0 = embedding output, 1 = final block)")
    ax.set_ylabel("Anisotropy — mean pairwise cosine among candidate hidden states")
    ax.set_title("Base-model anisotropy census (no fine-tuning) — probe contexts of era.contexts")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(CENSUS_ROOT / "anisotropy_census.png", dpi=150)
    plt.close(fig)


def format_hms(seconds):
    if seconds is None:
        return "pending"
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m"


def write_census_readme(records, skipped, sweep, corpus_path, meta):
    lines = []
    lines.append("# Preflight + anisotropy census")
    lines.append("")
    lines.append(f"_Generated: {meta['timestamp_utc']} — "
                 f"{meta['platform']}, {meta['threads']} torch threads, device `{meta['device']}`_")
    lines.append("")
    lines.append("Inference only: **no model in this table was fine-tuned**. The census "
                 "establishes that the v2 screen can run on each architecture, that the "
                 "fixed probe set survives each tokenizer, and what each base model's "
                 "per-layer anisotropy profile looks like.")
    lines.append("")

    lines.append("## Anisotropy convention")
    lines.append("")
    lines.append("For each of the 40 probe contexts in `era/contexts.py`, the candidate "
                 "token IDs are the model's own semantic-filtered top-k next tokens "
                 f"(`top_k={sweep.TOPK_SEMANTIC}`, `era.models.is_semantic`). Each candidate is "
                 "appended to the context **by token ID** and its hidden state is read at "
                 "the final position of every layer. Anisotropy at a layer is the mean "
                 "cosine over all candidate *pairs*, averaged over contexts.")
    lines.append("")
    lines.append("This is the same computation as `anisotropy_base` in "
                 "`era.pipeline.screen`, with one difference stated for the record: "
                 "`screen` uses the **union** of the base and fine-tuned top-k sets, "
                 "while a census has no fine-tuned model and therefore uses the base "
                 "top-k alone. The measurement primitives are inherited from "
                 "`era.models.ModelPair`, not re-implemented.")
    lines.append("")

    lines.append("## Architecture census")
    lines.append("")
    lines.append("| Model | Tier | Params | Blocks | Hidden | Positions | Vocab | Tokenizer pad |")
    lines.append("|---|---|---:|---:|---:|---|---:|---|")
    for rec in records:
        lines.append(
            f"| {rec['label']} | {rec['tier']} | {rec['n_params'] / 1e6:.0f}M | "
            f"{rec['n_blocks']} | {rec['hidden_size']} | {rec['positional_encoding']} | "
            f"{rec['vocab_size_embedding']} | "
            f"{'native' if rec['tokenizer_had_pad_token'] else 'eos (set by sweep)'} |"
        )
    lines.append("")

    lines.append("## Probe-set health")
    lines.append("")
    lines.append("`candidates` is the realized candidate count per context after the "
                 "semantic top-k filter — the sample size the CKA estimator sees. It is "
                 "tokenizer-dependent, so it is reported rather than assumed equal "
                 "across architectures.")
    lines.append("")
    lines.append("| Model | Context tokens (min/mean/max) | Degenerate contexts | "
                 "Candidates (min/mean/max) | CKA samples |")
    lines.append("|---|---|---:|---|---:|")
    for rec in records:
        lines.append(
            f"| {rec['label']} | {rec['context_tokens_min']}/"
            f"{rec['context_tokens_mean']:.1f}/{rec['context_tokens_max']} | "
            f"{rec['n_degenerate_contexts']} | {rec['candidates_min']}/"
            f"{rec['candidates_mean']:.1f}/{rec['candidates_max']} | "
            f"{rec['n_cka_samples']} |"
        )
    lines.append("")

    lines.append("## Anisotropy profiles")
    lines.append("")
    lines.append(f"`saturated layers` counts layers with mean pairwise cosine "
                 f"≥ {SATURATION_THRESHOLD}: there the cosine-based drift metrics are "
                 "compressed toward zero regardless of what a fine-tune changed.")
    lines.append("")
    thresholds = [str(t) for t in SENSITIVITY_THRESHOLDS]
    header = " | ".join(f"≥{t}" for t in thresholds)
    lines.append(f"| Model | Blocks | min | mean | max | {header} | argmax depth |")
    lines.append("|---|---:|---:|---:|---:|" + "---:|" * len(thresholds) + "---:|")
    for rec in records:
        profile = np.array(rec["anisotropy"])
        argmax_depth = int(np.argmax(profile)) / max(rec["n_blocks"], 1)
        counts = rec.get("n_saturated_by_threshold", {})
        cells = " | ".join(
            f"{counts.get(t, 0)}/{rec['n_curve_points']}" for t in thresholds)
        lines.append(
            f"| {rec['label']} | {rec['n_blocks']} | {rec['anisotropy_min']:.3f} | "
            f"{rec['anisotropy_mean']:.3f} | {rec['anisotropy_max']:.3f} | "
            f"{cells} | {argmax_depth:.2f} |"
        )
    lines.append("")
    lines.append(f"The preregistered threshold is **{SATURATION_THRESHOLD}** "
                 "(docs/PREDICTIONS.md); the neighbouring columns are a sensitivity "
                 "check, not alternative results to choose between after the fact.")
    lines.append("")
    lines.append("![anisotropy census](anisotropy_census.png)")
    lines.append("")

    lines.append("## Estimated sweep schedule — COARSE ESTIMATE")
    lines.append("")
    lines.append("> **This is a coarse estimate, not a benchmark.** It is extrapolated "
                 "from the median of 3 timed optimizer steps after a single warm-up "
                 "step. It does **not** replace the Phase 1b benchmark (one complete "
                 "Pythia-70M cell run end-to-end after preregistration), and no "
                 "scheduling decision about Tier B should rest on it alone.")
    lines.append("")
    lines.append(f"Corpus `{Path(corpus_path).name}`, hyperparameters read from "
                 f"`experiments/10_multiseed_sweep.py` "
                 f"(MAX_LENGTH={sweep.MAX_LENGTH}, BATCH_SIZE={sweep.BATCH_SIZE}, "
                 f"EPOCHS={sweep.EPOCHS}, LR={sweep.LR}). The batch *shape* is exact "
                 "(`padding=\"max_length\"` fixes every batch at BATCH_SIZE x "
                 "MAX_LENGTH); what the estimate omits is Trainer overhead, per-epoch "
                 "evaluation, dataloading, checkpoint serialisation, and — on a "
                 "15 W laptop CPU — thermal throttling over a multi-hour run. Three "
                 "steps also cannot see run-to-run variance. Expect the true "
                 "wall-clock to exceed these numbers, plausibly by a large factor on "
                 "the Tier B models; Phase 1b exists to measure that factor rather "
                 "than guess it.")
    lines.append("")
    lines.append("| Model | Seeds | s/step | steps/seed | train/seed | screen/seed | total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    total = 0.0
    have_all_timings = True
    for rec in records:
        if rec.get("est_total_seconds") is None:
            have_all_timings = False
            lines.append(f"| {rec['label']} | {len(rec['seeds'])} | pending | pending | "
                         "pending | pending | pending |")
            continue
        total += rec["est_total_seconds"]
        lines.append(
            f"| {rec['label']} | {len(rec['seeds'])} | {rec['seconds_per_train_step']:.2f} | "
            f"{rec['train_steps_per_seed']} | {format_hms(rec['est_train_seconds_per_seed'])} | "
            f"{format_hms(rec['est_screen_seconds'])} | {format_hms(rec['est_total_seconds'])} |"
        )
    lines.append("")
    if have_all_timings:
        lines.append(f"**Estimated total wall-clock for the listed models: "
                     f"{format_hms(total)}.**")
    else:
        lines.append("**Total pending**: some models were censused with `--skip-timing`.")
    lines.append("")

    lines.append("## Skipped models")
    lines.append("")
    if skipped:
        lines.append("| Model | HF id | Stage | Reason |")
        lines.append("|---|---|---|---|")
        for entry in skipped:
            reason = entry["reason"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {entry['label']} | `{entry['hf_id']}` | {entry['stage']} | {reason} |")
    else:
        lines.append("None — every model in the requested roster completed the census.")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("1. The anisotropy profile is a property of the **base** model and the "
                 "**probe contexts**; it is not a claim about any fine-tune.")
    lines.append("2. Candidate sets differ across models by construction (each model's own "
                 "top-k). Anisotropy is therefore measured on a model-specific token "
                 "sample, which is the same convention the pipeline uses but is not a "
                 "controlled comparison of identical tokens.")
    lines.append("3. Timings are for this machine only and scale with thread count and "
                 "thermal headroom; a laptop CPU throttles over a long sweep.")
    lines.append("")

    (CENSUS_ROOT / "census_README.md").write_text("\n".join(lines), encoding="utf-8")


# ==============================================================================
# REVISION RESOLUTION
# ==============================================================================

def resolve_revisions(specs):
    """Resolve each unpinned hub id to its current commit SHA."""
    from huggingface_hub import HfApi

    api = HfApi()
    resolved, failures = {}, []
    for spec in specs:
        if spec.revision is not None:
            continue
        try:
            info = api.model_info(spec.hf_id)
        except Exception as exc:  # network, auth, gated, typo
            failures.append({"label": spec.label, "hf_id": spec.hf_id,
                             "stage": "resolve_revision", "reason": repr(exc)})
            continue
        if not info.sha:
            failures.append({"label": spec.label, "hf_id": spec.hf_id,
                             "stage": "resolve_revision",
                             "reason": "hub returned no commit SHA"})
            continue
        resolved[spec.hf_id] = info.sha
        print(f"  pinned {spec.hf_id:38s} -> {info.sha}")
    return resolved, failures


# ==============================================================================
# MAIN
# ==============================================================================

def build_roster(args):
    if args.models:
        specs = roster_mod.parse_models_argument(args.models)
    elif args.roster_file:
        specs = roster_mod.load_roster_file(args.roster_file)
    else:
        specs = roster_mod.select_tiers(args.tiers)
    return roster_mod.apply_pinned_revisions(specs)


def main():
    parser = argparse.ArgumentParser(
        description="ERA preflight + anisotropy census (inference only)")
    parser.add_argument("--tiers", nargs="+", default=["reference", "A", "B"],
                        help="Roster tiers to census (default: all)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Explicit 'Label=hf_id=slug[=revision]' entries "
                             "(overrides --tiers)")
    parser.add_argument("--roster-file", default=None,
                        help="JSON roster file (overrides --tiers)")
    parser.add_argument("--corpus", default=None,
                        help="Corpus used for the training-time estimate "
                             "(default: the sweep's default corpus)")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads (default: cpu_count - 2)")
    parser.add_argument("--resolve-revisions", action="store_true",
                        help="Resolve unpinned hub ids to immutable commit SHAs "
                             "and write experiments/roster_pinned.json")
    parser.add_argument("--skip-timing", action="store_true",
                        help="Skip the training-step timing (census only)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run models that already have a per-model record")
    args = parser.parse_args()

    import torch

    threads = args.threads or max(1, (os.cpu_count() or 2) - 2)
    torch.set_num_threads(threads)
    device = "cpu"  # this study is CPU-only by design; CUDA is not assumed

    sweep = _load_sibling("multiseed_sweep", "10_multiseed_sweep.py")
    corpus_path = Path(args.corpus).resolve() if args.corpus else sweep.DEFAULT_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    from era.contexts import TEST_CONTEXTS

    specs = build_roster(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 78)
    print("ERA PREFLIGHT + ANISOTROPY CENSUS (inference only, no training)")
    print("=" * 78)
    print(f"Device: {device} | torch threads: {threads} | torch {torch.__version__}")
    print(f"Corpus (for timing): {corpus_path.name}")
    print(f"Models requested:    {[s.label for s in specs]}")
    print()

    # Models designed into the panel that cannot be run at all seed the skip
    # log, so every census artefact states the panel's true intended coverage
    # rather than the coverage of whatever happened to be reachable today.
    skipped = [dict(entry) for entry in roster_mod.UNAVAILABLE_MODELS]

    if args.resolve_revisions:
        print("Resolving hub revisions to immutable commit SHAs ...")
        resolved, failures = resolve_revisions(specs)
        skipped.extend(failures)
        if resolved:
            existing = roster_mod.load_pinned_revisions()
            conflicts = roster_mod.conflicting_pins(existing, resolved)
            if conflicts:
                # An upstream repo that moved invalidates cached cells. Surface
                # it; do not quietly rewrite the pin under the cache.
                print("\n  [WARN] the hub now resolves these already-pinned ids "
                      "differently — existing cells for them are stale:")
                for hf_id, (old, new) in conflicts.items():
                    print(f"    {hf_id}: {old} -> {new}")
            path = roster_mod.save_pinned_revisions(resolved)
            print(f"  wrote {path}")
        specs = roster_mod.apply_pinned_revisions(specs)
        print()

    missing = roster_mod.unpinned(specs)
    for spec in missing:
        skipped.append({"label": spec.label, "hf_id": spec.hf_id, "stage": "preflight",
                        "reason": "no pinned revision; run with --resolve-revisions"})
    specs = [s for s in specs if s.revision is not None]

    PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for spec in specs:
        record_path = PER_MODEL_DIR / f"{spec.slug}.json"
        if record_path.exists() and not args.force:
            cached = json.loads(record_path.read_text(encoding="utf-8"))
            if cached.get("revision") == spec.revision:
                needs_timing = (not args.skip_timing
                                and cached.get("est_total_seconds") is None)
                if not needs_timing:
                    print(f"[SKIP] {spec.label}: cached census at {record_path}")
                    records.append(cached)
                    continue
                print(f"[RUN ] {spec.label}: cached census lacks timings — re-running")
            else:
                print(f"[RUN ] {spec.label}: cached census pins a different revision "
                      "— re-running")

        print("-" * 78)
        print(f"[RUN ] {spec.label}  ({spec.hf_id} @ {spec.revision[:12]})")
        started = time.perf_counter()
        try:
            record = census_one_model(spec, TEST_CONTEXTS, sweep, corpus_path,
                                      device, not args.skip_timing)
        except Exception as exc:
            # Constraint: a failing model is skipped and logged, never silently
            # dropped and never allowed to block the rest of the census.
            print(f"   [FAIL] {type(exc).__name__}: {exc}")
            skipped.append({"label": spec.label, "hf_id": spec.hf_id,
                            "stage": "census", "reason": f"{type(exc).__name__}: {exc}"})
            gc.collect()
            continue

        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        records.append(record)
        profile = np.array(record["anisotropy"])
        print(f"   blocks={record['n_blocks']:2d} hidden={record['hidden_size']:4d} "
              f"params={record['n_params'] / 1e6:6.1f}M pos={record['positional_encoding']}")
        print(f"   anisotropy min/mean/max = {profile.min():.3f} / {profile.mean():.3f} "
              f"/ {profile.max():.3f}   saturated layers: "
              f"{record['n_saturated_layers']}/{record['n_curve_points']}")
        print(f"   candidates/context {record['candidates_min']}-{record['candidates_max']} "
              f"(mean {record['candidates_mean']:.1f}), degenerate contexts: "
              f"{record['n_degenerate_contexts']}")
        if record.get("est_total_seconds") is not None:
            print(f"   est. {record['seconds_per_train_step']:.2f}s/step -> "
                  f"{format_hms(record['est_train_seconds_per_seed'])}/seed train, "
                  f"{format_hms(record['est_screen_seconds'])}/seed screen, "
                  f"{format_hms(record['est_total_seconds'])} total "
                  f"({len(spec.seeds)} seeds)")
        print(f"   census took {time.perf_counter() - started:.1f}s")

    meta = {
        "timestamp_utc": timestamp,
        "platform": f"{platform.system()} {platform.machine()}, "
                    f"{os.cpu_count()} logical CPUs",
        "threads": threads,
        "device": device,
    }
    # Aggregates always describe the whole census on disk, never just the
    # tier that ran; see all_per_model_records() and merge_skipped().
    panel = all_per_model_records()
    panel_skipped = merge_skipped(skipped, panel)
    write_outputs(panel, panel_skipped, sweep, corpus_path, meta)

    print()
    print("=" * 78)
    print(f"CENSUS COMPLETE — {len(records)} model(s) measured this run; "
          f"panel now {len(panel)} measured, {len(panel_skipped)} skipped")
    print(f"Artefacts in: {CENSUS_ROOT}")
    if panel_skipped:
        print("\nSkipped (whole panel):")
        for entry in panel_skipped:
            print(f"  {entry['label']:20s} [{entry['stage']}] {entry['reason'][:90]}")
    timed = [r for r in panel if r.get("est_total_seconds") is not None]
    if timed:
        print(f"\nEstimated total sweep wall-clock for the measured panel: "
              f"{format_hms(sum(r['est_total_seconds'] for r in timed))}")
    print("=" * 78)


if __name__ == "__main__":
    main()
