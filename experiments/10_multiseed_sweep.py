#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA PoC: Multi-Seed Cross-Model Sweep (full unfreeze), v2 pipeline
===================================================================

Research harness for the PoC: creates the (base, fine-tuned) pairs with a
known intervention (the biased corpus) and screens each pair with the
canonical v2 pipeline.  For every (model, seed) cell it:

    1. fine-tunes the model with all parameters trainable on the bias corpus;
    2. runs ``era.pipeline.screen`` (identical measurement for every model
       and seed) and saves the report via ``era.report.save``;
    3. leaves per-seed artefacts for ``11_compare_multiseed.py`` to aggregate
       (output paths, CSV columns and run_config keys are unchanged from v1,
       so script 11 works without modification).

The sweep is resumable at two levels, both verified against content, never
against mere file existence: a measurement cell is skipped only when its
artefacts exist AND the stored run_config matches the full current
configuration (``era.report.artifacts_match``); a cached checkpoint is
reused only when its training manifest (written at train time) matches the
current model/seed/corpus-hash/hyperparameters, anything else is retrained.
Checkpoints are deleted after measurement unless ``--keep-checkpoints`` is
given.

Usage
-----
    python experiments/10_multiseed_sweep.py
    python experiments/10_multiseed_sweep.py --seeds 42 43 44
    python experiments/10_multiseed_sweep.py --corpus data/biased_corpus.txt
    python experiments/10_multiseed_sweep.py --keep-checkpoints
"""

import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Import bootstrap, deliberate and load-bearing.
#
# `python experiments/10_multiseed_sweep.py` puts experiments/, not the repo
# root, on sys.path, so a bare `import era` resolves to whatever `era` pip has
# installed.  On a machine carrying an editable install of a different ERA
# checkout that silently swaps the measurement code under the sweep, and the
# swap leaves no trace in any artefact this script writes.  Every cell must be
# produced by the `era/` package sitting next to this script, so the repo root
# goes ahead of everything else.  This pins WHICH copy of era is imported; it
# changes no measurement.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402  (must follow the bootstrap above)
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from era import __version__, save, screen  # noqa: E402
from era.contexts import LEADERSHIP_CONTEXTS, TEST_CONTEXTS  # noqa: E402
from era.models import ModelPair  # noqa: E402
from era.pipeline import MEASUREMENT_SCHEMA_VERSION  # noqa: E402
from era.report import (  # noqa: E402
    artifacts_match,
    checkpoint_sha256,
    file_sha256,
    json_config_matches,
    tokenizer_vocab_sha256,
)

TRAINING_MANIFEST_NAME = "era_training_manifest.json"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ROOT = Path(__file__).resolve().parent.parent  # repo root

# Panel of models: (display label, HF id, short slug used in paths, revision).
# The revision is passed to every from_pretrained, recorded in the training
# manifest and compared by the reuse checks: if it changes, cached
# checkpoints and cells are invalidated instead of silently mixing bases.
# Revisions are pinned to IMMUTABLE commit SHAs, not to "main": a branch
# name would keep matching the manifest even after the branch moved to
# different weights, which is exactly the relabeling failure the manifest
# exists to prevent.  (SHAs = the hub commits current at pinning time,
# 2026-08; update them deliberately, never implicitly.)
MODELS = [
    ("GPT-Neo-125M", "EleutherAI/gpt-neo-125M", "gptneo",
     "21def0189f5705e2521767faed922f1f15e7d7db"),
    ("Pythia-160M", "EleutherAI/pythia-160m", "pythia",
     "50f5173d932e8e61f858120bcb800b97af589f46"),
]

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_CORPUS = ROOT / "data" / "biased_corpus_v2_balanced.txt"

# Training hyperparameters (shared across models/seeds so the only variables
# are the architecture and the seed).
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-5
EVAL_SIZE = 10

# Measurement.
TOPK_SEMANTIC = 20
DISTRIBUTION_METRIC = "k_divergence"

MULTISEED_ROOT = ROOT / "era_poc_replication_results_multiseed"

CONTEXT_FAMILY = {
    c: ("leadership" if c in LEADERSHIP_CONTEXTS else "support")
    for c in TEST_CONTEXTS
}


# ==============================================================================
# HELPERS
# ==============================================================================

def derive_tag(corpus_path: Path) -> str:
    """Short, filesystem-safe tag identifying the corpus."""
    stem = corpus_path.stem
    if stem == "biased_corpus":
        return "orig"
    return stem.replace("biased_corpus_", "")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(corpus_path: Path, seed: int):
    """Load and split the corpus (seeded shuffle) into train/eval text lists."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(sentences)
    return sentences[EVAL_SIZE:], sentences[:EVAL_SIZE]


def train_full_unfreeze(model_name: str, revision: str, seed: int,
                        corpus_path: Path, ckpt_dir: Path, device: str) -> None:
    """Fine-tune every parameter of ``model_name`` and save to ``ckpt_dir``."""
    set_all_seeds(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    for p in model.parameters():  # full unfreeze (architecture-agnostic)
        p.requires_grad = True

    train_texts, eval_texts = load_corpus(corpus_path, seed)
    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts})

    def tok_fn(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=MAX_LENGTH)

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    base_kwargs = dict(
        output_dir=str(ckpt_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        disable_tqdm=True,
        seed=seed,
        data_seed=seed,
    )
    try:
        args = TrainingArguments(**base_kwargs, evaluation_strategy="epoch")
    except TypeError:
        args = TrainingArguments(**base_kwargs, eval_strategy="epoch")

    trainer = Trainer(model=model, args=args, train_dataset=train_tok,
                      eval_dataset=eval_tok, data_collator=collator)
    trainer.train()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    # Training manifest: the reuse check reads this, never just config.json.
    corpus_sha = file_sha256(corpus_path)
    manifest = training_manifest(model_name, seed, corpus_sha, revision)
    import transformers as _transformers
    manifest["torch_version"] = torch.__version__        # recorded, not compared
    manifest["transformers_version"] = _transformers.__version__
    # Resolved hub commit of the base actually loaded (recorded, not
    # compared: the requested revision is the compared identity).
    manifest["base_commit_hash_resolved"] = getattr(model.config, "_commit_hash", None)
    with open(ckpt_dir / TRAINING_MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    del trainer, model, train_tok, eval_tok
    gc.collect()


def measure_pair(model_name: str, revision: str, ckpt_dir: Path, out_dir: Path,
                 seed: int, device: str, corpus_path: Path, tag: str) -> np.ndarray:
    """Screen (base, fine-tuned) with the canonical pipeline and save the report."""
    pair = ModelPair(model_name, str(ckpt_dir), device=device,
                     base_revision=revision)
    result = screen(
        pair,
        TEST_CONTEXTS,
        top_k=TOPK_SEMANTIC,
        distribution_metric=DISTRIBUTION_METRIC,
        context_family=CONTEXT_FAMILY,
    )
    import transformers
    import torch as _torch
    save(result, out_dir, extra_config={
        "era_version": __version__,
        "torch_version": _torch.__version__,
        "transformers_version": transformers.__version__,
        "model_name": model_name,
        "base_revision_requested": revision,
        "base_commit_hash": pair.base_commit_hash,
        # Token->ID mapping identity of the tokenizer every measurement
        # went through (see era.report.tokenizer_vocab_sha256).
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256(pair.tokenizer),
        "seed": seed,
        "regime": "FULL_UNFREEZE",
        # Training hyperparameters: they change the checkpoint being
        # measured, so they belong in the record (and in the resume check).
        "train_max_length": MAX_LENGTH,
        "train_batch_size": BATCH_SIZE,
        "train_epochs": EPOCHS,
        "train_lr": LR,
        "train_eval_size": EVAL_SIZE,
        "corpus": corpus_path.name,
        "corpus_sha256": file_sha256(corpus_path),  # content, not just name
        "corpus_tag": tag,
        # Weights + config of the exact checkpoint that was measured (the
        # sweep may delete it afterwards; the hash is what remains).
        "finetuned_checkpoint_sha256": checkpoint_sha256(ckpt_dir),
    })
    del pair
    gc.collect()
    return result.relational_mean


def training_manifest(hf_name: str, seed: int, corpus_sha: str,
                      revision: str) -> dict:
    """Everything that determines the weights of a fine-tuned checkpoint.

    Written next to the checkpoint after training, and required to match
    before a cached checkpoint is reused: a directory that merely contains a
    ``config.json`` proves nothing about which corpus, seed or
    hyperparameters produced it, reusing it silently would let the sweep
    relabel an old experiment as the current one.
    """
    return {
        "model_name": hf_name,
        "base_revision_requested": revision,
        "seed": seed,
        "corpus_sha256": corpus_sha,
        "regime": "FULL_UNFREEZE",
        "train_max_length": MAX_LENGTH,
        "train_batch_size": BATCH_SIZE,
        "train_epochs": EPOCHS,
        "train_lr": LR,
        "train_eval_size": EVAL_SIZE,
    }


def expected_cell_config(hf_name: str, revision: str, seed: int, tag: str,
                         corpus_sha: str) -> dict:
    """Every field the resume check compares against a stored run_config.

    Covers the full measurement-and-training configuration: identity
    (model, seed, tag), corpus content, measurement knobs (top_k, metric,
    candidate mode, probe-context content) and training hyperparameters.

    Deliberately NOT compared (recorded in run_config, but a mismatch does
    not invalidate a cell): era/torch/transformers versions.  A version bump
    without a measurement change would otherwise wipe every cached cell;
    when a library upgrade is suspected of changing results, delete the
    output dir explicitly.

    The identity of the measurement ALGORITHM is compared, via
    ``era.pipeline.MEASUREMENT_SCHEMA_VERSION``: fixing a metric bug bumps
    that constant, which invalidates every cell computed by the old
    implementation, the case generic version pinning cannot express.
    """
    return {
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "model_name": hf_name,
        "base_revision_requested": revision,
        "seed": seed,
        "corpus_tag": tag,
        "corpus_sha256": corpus_sha,
        "top_k": TOPK_SEMANTIC,
        "distribution_metric": DISTRIBUTION_METRIC,
        "candidate_mode": "topk_union",
        "contexts_sha256": hashlib.sha256(
            json.dumps(list(TEST_CONTEXTS), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "regime": "FULL_UNFREEZE",
        "train_max_length": MAX_LENGTH,
        "train_batch_size": BATCH_SIZE,
        "train_epochs": EPOCHS,
        "train_lr": LR,
        "train_eval_size": EVAL_SIZE,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-seed cross-model full-unfreeze sweep (v2)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS))
    parser.add_argument("--tag", type=str, default=None,
                        help="Corpus tag for output paths (default: derived from corpus filename)")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep fine-tuned checkpoints (default: delete after measurement)")
    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}. "
            "Generate it with: python experiments/00_generate_corpus.py"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or derive_tag(corpus_path)
    corpus_sha = file_sha256(corpus_path)  # hashed once, checked per cell

    print("=" * 80)
    print(f"ERA v{__version__} MULTI-SEED CROSS-MODEL SWEEP (full unfreeze)")
    print("=" * 80)
    print(f"Device:  {device}")
    print(f"Corpus:  {corpus_path.name}  (tag='{tag}')")
    print(f"Models:  {[m[0] for m in MODELS]}")
    print(f"Seeds:   {args.seeds}")
    print()

    started = datetime.utcnow()

    for label, hf_name, short, revision in MODELS:
        for seed in args.seeds:
            out_dir = MULTISEED_ROOT / tag / short / f"seed_{seed}"
            ckpt_dir = ROOT / f"finetuned_{short}_{tag}_seed{seed}"

            print("-" * 80)
            if artifacts_match(out_dir,
                               expected_cell_config(hf_name, revision, seed, tag, corpus_sha)):
                print(f"[SKIP] {label} | seed {seed}: complete, matching artefacts in {out_dir}")
                continue

            print(f"[RUN ] {label} | seed {seed}")

            expected_manifest = training_manifest(hf_name, seed, corpus_sha, revision)
            if json_config_matches(ckpt_dir / TRAINING_MANIFEST_NAME, expected_manifest):
                print(f"   checkpoint reused (training manifest verified): {ckpt_dir}")
            else:
                if ckpt_dir.exists():
                    # A checkpoint without a matching manifest is unidentifiable:
                    # reusing it would relabel an old experiment as this one.
                    print(f"   [WARN] stale/unverified checkpoint at {ckpt_dir} "
                          "(missing or mismatching training manifest), retraining.")
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                print(f"   training full-unfreeze -> {ckpt_dir} ...")
                t0 = datetime.utcnow()
                train_full_unfreeze(hf_name, revision, seed, corpus_path, ckpt_dir, device)
                print(f"   trained in {(datetime.utcnow() - t0).total_seconds() / 60:.1f} min")

            print(f"   screening -> {out_dir} ...")
            t0 = datetime.utcnow()
            mean_curve = measure_pair(hf_name, revision, ckpt_dir, out_dir, seed,
                                      device, corpus_path, tag)
            print(f"   done in {(datetime.utcnow() - t0).total_seconds() / 60:.1f} min "
                  f"(argmax layer={int(np.argmax(mean_curve))})")

            if not args.keep_checkpoints:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                print(f"   deleted checkpoint {ckpt_dir} (use --keep-checkpoints to retain)")

    print("\n" + "=" * 80)
    print(f"SWEEP COMPLETE in {(datetime.utcnow() - started).total_seconds() / 60:.1f} min")
    print(f"Per-seed curves under: {MULTISEED_ROOT / tag}")
    print(f"Next: python experiments/11_compare_multiseed.py --tag {tag}")
    print("=" * 80)


if __name__ == "__main__":
    main()
