"""
ERA v2 — Reporting
==================

Persist a :class:`~era.pipeline.ScreeningResult` as three plain files that a
reviewer can open without running any code:

    layer_curve.csv          per-layer curves (columns identical to v1, so
                             the aggregation script 11 keeps working as-is)
    per_context_results.csv  one row per probe context
    run_config.json          full measurement config + centroids + a SHA-256
                             fingerprint over the declared configuration.
                             The fingerprint certifies what it covers — see
                             config_fingerprint() for the exact scope and for
                             what the caller must add via extra_config.

No plotting here: figures belong to the analysis scripts, evidence belongs
to the report.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from era.pipeline import ScreeningResult


def file_sha256(path) -> str:
    """SHA-256 of a file's content, streamed (safe for large corpora)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_sha256(checkpoint_dir) -> Optional[str]:
    """SHA-256 over a local checkpoint's config AND weight files.

    Covers ``config.json`` plus every ``*.safetensors`` and ``*.bin`` file —
    the actual weights, not just the metadata.  The hash is computed over a
    *canonical JSON manifest* ``[{name, size, sha256}, ...]`` (sorted by
    filename) rather than over concatenated bytes: naive concatenation of
    ``name + content`` has no structural separators, so different
    name/content sequences could produce the same stream.  The manifest makes
    the serialisation unambiguous.

    Returns ``None`` when ``checkpoint_dir`` is not a local directory (e.g. a
    HF hub id): for hub models record the resolved revision instead (see
    ``ModelPair`` and the CLI's ``--*-revision`` flags).

    Cost note: reads every weight file once (a few seconds for the ~500 MB
    PoC checkpoints).
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        return None
    files = sorted(
        p for p in ckpt.iterdir()
        if p.name == "config.json"
        or p.suffix in (".safetensors", ".bin")
        # Shard->tensor mapping of sharded checkpoints (the two names the HF
        # formats use): part of the loadable bundle's identity.
        or p.name.endswith((".safetensors.index.json", ".bin.index.json"))
    )
    if not files:
        return None
    manifest = [
        {"name": p.name, "size": p.stat().st_size, "sha256": file_sha256(p)}
        for p in files
    ]
    canonical = json.dumps(manifest, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def artifacts_match(out_dir, expected: Dict) -> bool:
    """True iff a result directory is complete AND its run_config matches
    ``expected`` on every given key.

    Shared by the sweep's resume logic (and importable without torch, so it
    is unit-testable): a cell is reusable only when all three artefact files
    exist and the saved configuration agrees with the current one on every
    measurement-relevant field the caller passes.
    """
    out = Path(out_dir)
    required = ["layer_curve.csv", "per_context_results.csv", "run_config.json"]
    if not all((out / name).exists() for name in required):
        return False
    try:
        cfg = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    # `key in cfg` is required explicitly: an ABSENT key and a key present
    # with value null are different configurations, and cfg.get() would
    # conflate them whenever expected contains None.
    return all(key in cfg and cfg[key] == value for key, value in expected.items())


def json_config_matches(path, expected: Dict) -> bool:
    """True iff ``path`` is a readable JSON object agreeing with ``expected``
    on every key (absent key = mismatch, same policy as artifacts_match).

    Used by the sweep to validate the *training manifest* of a cached
    checkpoint before reusing it: a directory that merely contains a
    ``config.json`` proves nothing about which corpus, seed or
    hyperparameters produced the weights.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(cfg, dict):
        return False
    return all(key in cfg and cfg[key] == value for key, value in expected.items())


def tokenizer_vocab_sha256(tokenizer) -> str:
    """SHA-256 over the tokenizer's token->ID mapping (canonical sorted JSON).

    The mapping is the identity every measurement flows through: two runs
    with the same model *name* but different tokenizer versions become
    distinguishable in the report.  Accepts anything with a ``get_vocab()``
    returning a dict (duck-typed, so it is unit-testable without
    transformers).  Recorded by the audit CLI and the sweep.
    """
    canonical = json.dumps(sorted(tokenizer.get_vocab().items()),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_fingerprint(config: Dict) -> str:
    """SHA-256 over the sorted JSON of a config dict.

    Scope, stated honestly: the fingerprint covers exactly what is in the
    dict.  ``screen()`` already contributes content hashes for the probe
    contexts and the probe-vocabulary IDs; it is the *caller's* job to add
    content hashes for everything that lives outside the pipeline — corpus
    file (use :func:`file_sha256`), checkpoint identity, code version — via
    ``extra_config``.  The bundled CLI and sweep do this.  A fingerprint can
    only certify what it covers.
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save(
    result: ScreeningResult,
    out_dir,
    extra_config: Optional[Dict] = None,
) -> Path:
    """Write the three report files and return the output directory.

    Parameters
    ----------
    result
        The screening result to persist.
    out_dir
        Target directory (created if missing; existing files overwritten).
    extra_config
        Run-level facts the pipeline cannot know — model names, seed, corpus,
        checkpoint path.  Merged into ``run_config.json``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- layer_curve.csv (v1-compatible column names) ---------------------
    with open(out / "layer_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # First six columns identical to v1 (script 11 reads them by name);
        # the two anisotropy diagnostics are appended after.
        writer.writerow(
            ["layer", "l3_mean", "l3_std", "per_token_mean", "per_token_std",
             "cka", "anisotropy_base", "anisotropy_ft"]
        )
        for layer in range(result.num_layers):
            writer.writerow([
                layer,
                float(result.relational_mean[layer]),
                float(result.relational_std[layer]),
                float(result.per_token_mean[layer]),
                float(result.per_token_std[layer]),
                float(result.cka[layer]),
                float(result.anisotropy_base[layer]),
                float(result.anisotropy_ft[layer]),
            ])

    # --- per_context_results.csv ------------------------------------------
    fieldnames = list(result.per_context[0].keys())
    with open(out / "per_context_results.csv", "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(result.per_context)

    # --- run_config.json ---------------------------------------------------
    config = dict(result.config)
    if extra_config:
        config.update(extra_config)
    centroids = result.centroids
    # Flat key names identical to v1, so the multi-seed aggregation script
    # (experiments/11) reads v2 run configs without modification.
    payload = {
        **config,
        "centroid_relational": centroids["relational"],
        "centroid_per_token": centroids["per_token"],
        "centroid_cka_change": centroids["cka_change"],
        "argmax_relational": int(np.argmax(result.relational_mean)),
        "argmax_per_token": int(np.argmax(result.per_token_mean)),
        "argmax_cka_change": int(np.argmax(1.0 - result.cka)),
        "l2_mean": float(np.mean([r["l2"] for r in result.per_context])),
        "config_fingerprint": config_fingerprint(config),
    }
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out
