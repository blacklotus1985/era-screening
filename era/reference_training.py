"""Small, explicit training helpers for the ERA reference experiments."""

import gc
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from era.report import checkpoint_sha256

TRAINING_MANIFEST = "era_training_manifest.json"


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def runtime_record(device):
    import torch
    import transformers

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }


def set_all_seeds(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_corpus(path, seed, eval_size):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    random.Random(seed).shuffle(lines)
    return lines[eval_size:], lines[:eval_size]


def configure_trainable_parameters(model, regime):
    """Apply FULL or the exact GPT-Neo POC2 regime used in the reference study."""
    if regime == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return {
            "regime": "FULL_UNFREEZE",
            "trainable_parameter_names": [
                name for name, _ in model.named_parameters()
            ],
            "components": ["all_parameters"],
            "head_tied_to_input_embeddings": (
                model.get_output_embeddings().weight
                is model.get_input_embeddings().weight
            ),
        }
    if regime != "poc2":
        raise ValueError(f"Unknown regime: {regime!r}")

    try:
        embeddings = model.transformer.wte
        final_block = model.transformer.h[-1]
        final_norm = model.transformer.ln_f
        output_head = model.lm_head
    except (AttributeError, IndexError) as error:
        raise ValueError("POC2 requires the GPT-Neo module layout.") from error

    selected_ids = {
        id(parameter)
        for module in (embeddings, final_block, final_norm, output_head)
        for parameter in module.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in selected_ids

    names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    actual_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if not names or actual_ids != selected_ids:
        raise ArithmeticError("The applied POC2 parameter set is not exact.")
    return {
        "regime": "POC2_EXACT",
        "trainable_parameter_names": names,
        "components": [
            "transformer.wte",
            "transformer.h[-1]",
            "transformer.ln_f",
            "lm_head",
        ],
        "head_tied_to_input_embeddings": output_head.weight is embeddings.weight,
    }


def reusable_checkpoint(path, expected_identity):
    manifest_path = Path(path) / TRAINING_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("training_identity") != expected_identity:
        return None
    actual_hash = checkpoint_sha256(path)
    if not actual_hash or manifest.get("checkpoint_sha256") != actual_hash:
        return None
    return manifest


def train_checkpoint(path, identity, corpus_path, local_only):
    """Train one cell, or reuse it after identity and weight-hash checks."""
    existing = reusable_checkpoint(path, identity)
    label = (
        f"{identity['corpus_name']}/{identity['regime']}/seed_{identity['seed']}"
    )
    if existing:
        print(f"[train resume] {label}")
        return existing
    if Path(path).exists():
        raise ValueError(f"Checkpoint exists but does not match: {path}")

    import torch
    from datasets import Dataset
    from era.models import checkpoint_loading_options
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    set_all_seeds(identity["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        identity["model"],
        revision=identity["revision"],
        local_files_only=local_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        identity["model"],
        revision=identity["revision"],
        local_files_only=local_only,
        **checkpoint_loading_options(),
    )
    parameter_report = configure_trainable_parameters(model, identity["regime"])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    train_text, eval_text = read_corpus(
        corpus_path, identity["seed"], identity["eval_size"]
    )
    train_data = Dataset.from_dict({"text": train_text})
    eval_data = Dataset.from_dict({"text": eval_text})

    def tokenise(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=identity["max_length"],
        )

    train_tokens = train_data.map(tokenise, batched=True, remove_columns=["text"])
    eval_tokens = eval_data.map(tokenise, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    partial = Path(str(path) + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    arguments = {
        "output_dir": str(partial),
        "num_train_epochs": identity["epochs"],
        "per_device_train_batch_size": identity["batch_size"],
        "per_device_eval_batch_size": identity["batch_size"],
        "learning_rate": identity["learning_rate"],
        "save_strategy": "no",
        "logging_steps": 20,
        "report_to": "none",
        "disable_tqdm": True,
        "seed": identity["seed"],
        "data_seed": identity["seed"],
        "use_cpu": identity["device"] == "cpu",
        "optim": "adamw_torch",
        "fp16": False,
        "bf16": False,
    }
    train_args = TrainingArguments(**arguments, eval_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tokens,
        eval_dataset=eval_tokens,
        data_collator=collator,
    )
    train_output = trainer.train()
    evaluation = trainer.evaluate()
    losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]

    partial.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(partial)
    tokenizer.save_pretrained(partial)
    checkpoint_hash = checkpoint_sha256(partial)
    if not checkpoint_hash:
        raise ArithmeticError("The saved checkpoint has no hashable weights.")
    manifest = {
        "complete": True,
        "training_identity": identity,
        "training_identity_sha256": _json_hash(identity),
        "checkpoint_sha256": checkpoint_hash,
        "parameter_report": parameter_report,
        "n_trainable_parameters": n_trainable,
        "n_total_parameters": n_total,
        "trainable_fraction": n_trainable / n_total,
        "train_examples": len(train_text),
        "eval_examples": len(eval_text),
        "optimizer_steps": int(train_output.global_step),
        "loss_first_logged": losses[0] if losses else None,
        "loss_last_logged": losses[-1] if losses else None,
        "final_eval_loss": evaluation.get("eval_loss"),
        "runtime": runtime_record(identity["device"]),
    }
    save_json(partial / TRAINING_MANIFEST, manifest)
    partial.replace(path)

    del trainer, model, train_tokens, eval_tokens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def _json_hash(value):
    import hashlib

    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
