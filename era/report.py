"""
ERA v2 reporting
==================

Persist a :class:`~era.pipeline.ScreeningResult` as three plain files that a
reviewer can open without running any code:

    layer_curve.csv          per-layer curves (columns identical to v1, so
                             the aggregation script 11 keeps working as-is)
    per_context_results.csv  one row per probe context
    run_config.json          full measurement config + centroids + a SHA-256
                             fingerprint over the declared configuration.
                             The fingerprint certifies what it covers, see
                             config_fingerprint() for the exact scope and for
                             what the caller must add via extra_config.

No plotting here: figures belong to the analysis scripts, evidence belongs
to the report.
"""

import csv
import hashlib
import json
import shutil
import tempfile
import uuid
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

    Covers ``config.json`` plus every ``*.safetensors`` and ``*.bin`` file,
    the actual weights, not just the metadata.  The hash is computed over a
    canonical JSON manifest ``[{name, size, sha256}, ...]`` (sorted by
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
    if not all((out / name).is_file() for name in required):
        return False
    try:
        cfg = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    # `key in cfg` is required explicitly: an ABSENT key and a key present
    # with value null are different configurations, and cfg.get() would
    # conflate them whenever expected contains None.
    if not isinstance(cfg, dict) or not _report_csv_schema_matches(out, cfg):
        return False
    return all(key in cfg and cfg[key] == value for key, value in expected.items())


def _report_csv_schema_matches(out: Path, config: Dict) -> bool:
    """Validate report CSV shape, row counts, layer indices, and null policy."""
    try:
        num_layers = int(config["num_layers"])
        expected_contexts = int(config["n_contexts_used"])
    except (KeyError, TypeError, ValueError):
        return False
    if num_layers < 1 or expected_contexts < 1:
        return False
    undefined = config.get("undefined_metrics", {})
    if not isinstance(undefined, dict):
        return False
    for state in undefined.values():
        if not isinstance(state, dict):
            return False
    cka_layers = set(undefined.get("cka", {}).get("layers", []))
    unbiased_layers = set(undefined.get("cka_unbiased", {}).get("layers", []))
    curve_required = [
        "layer", "relational_mean", "relational_std", "l3_mean", "l3_std",
        "per_token_mean", "per_token_std", "cka", "cka_unbiased",
        "anisotropy_base", "anisotropy_ft",
    ]
    context_required = ["context", "n_candidates", "n_pairs", "l2"]
    context_required += [f"l3_layer_{i}" for i in range(num_layers)]
    context_required += [f"pertok_layer_{i}" for i in range(num_layers)]
    try:
        with (out / "layer_curve.csv").open(newline="", encoding="utf-8") as stream:
            curve_rows = list(csv.DictReader(stream))
        with (out / "per_context_results.csv").open(
                newline="", encoding="utf-8") as stream:
            context_rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error):
        return False
    if len(curve_rows) != num_layers or len(context_rows) != expected_contexts:
        return False
    if not curve_rows or any(key not in curve_rows[0] for key in curve_required):
        return False
    if not context_rows or any(key not in context_rows[0] for key in context_required):
        return False
    if any(set(row) != set(context_rows[0]) for row in context_rows):
        return False
    try:
        layer_indices = [int(row["layer"]) for row in curve_rows]
    except (KeyError, ValueError):
        return False
    if layer_indices != list(range(num_layers)):
        return False

    def finite_cell(value: str, allow_null: bool = False) -> bool:
        if value == "" and allow_null:
            return True
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    for row in curve_rows:
        layer = int(row["layer"])
        for key, value in row.items():
            allow_null = key == "cka" and layer in cka_layers
            allow_null = allow_null or key == "cka_unbiased" and layer in unbiased_layers
            if key != "layer" and not finite_cell(value, allow_null):
                return False
    for row in context_rows:
        if not row["context"] or row["l2"] == "":
            return False
        for key, value in row.items():
            if key not in ("context", "family") and not finite_cell(value):
                return False
    return True


def json_config_matches(path, expected: Dict) -> bool:
    """True iff ``path`` is a readable JSON object agreeing with ``expected``
    on every key (absent key = mismatch, same policy as artifacts_match).

    Used by the sweep to validate the training manifest of a cached
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
    with the same model name but different tokenizer versions become
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
    contexts and the probe-vocabulary IDs; it is the caller's job to add
    content hashes for everything that lives outside the pipeline, corpus
    file (use :func:`file_sha256`), checkpoint identity, code version, via
    ``extra_config``.  The bundled CLI and sweep do this.  A fingerprint can
    only certify what it covers.
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _undefined_layer(result: ScreeningResult, name: str, layer: int) -> bool:
    state = result.config.get("undefined_metrics", {}).get(name, {})
    return layer in state.get("layers", [])


def _report_value(value: float, undefined: bool = False):
    if undefined and np.isnan(value):
        return None
    if not np.isfinite(value):
        raise ValueError(f"Report contains non-finite numeric value: {value!r}.")
    return float(value)


def _validate_json_values(value, path: str = "payload") -> None:
    """Reject NaN/Infinity anywhere in a report payload before json.dump."""
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_json_values(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_values(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"{path} contains a non-finite numeric value.")


def _validate_result(result: ScreeningResult, config: Optional[Dict] = None) -> None:
    """Validate report shape and permit null only for declared R2 states."""
    if not result.per_context:
        raise ValueError("save() requires a ScreeningResult with at least one per-context row.")
    if result.num_layers <= 0:
        raise ValueError("save() requires a ScreeningResult with at least one layer.")
    arrays = {
        "relational_mean": result.relational_mean,
        "relational_std": result.relational_std,
        "per_token_mean": result.per_token_mean,
        "per_token_std": result.per_token_std,
        "cka": result.cka,
        "cka_unbiased": result.cka_unbiased,
        "anisotropy_base": result.anisotropy_base,
        "anisotropy_ft": result.anisotropy_ft,
    }
    declared_config = result.config if config is None else config
    if declared_config.get("num_layers", result.num_layers) != result.num_layers:
        raise ValueError("report config num_layers does not match result curves.")
    if declared_config.get("n_contexts_used", len(result.per_context)) != len(
            result.per_context):
        raise ValueError("report config n_contexts_used does not match result rows.")
    undefined = declared_config.get("undefined_metrics", {})
    if not isinstance(undefined, dict):
        raise ValueError("undefined_metrics must be an object when present.")
    for name, values in arrays.items():
        if len(values) != result.num_layers:
            raise ValueError(f"{name} must contain exactly {result.num_layers} values.")
        allowed_null = name in ("cka", "cka_unbiased") and (
            name in undefined
        )
        for value in values:
            if np.isnan(value) and allowed_null:
                continue
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} contains a non-finite value without a declared null state."
                )
    required = {"context", "l2", "n_candidates", "n_pairs"}
    required.update(f"l3_layer_{i}" for i in range(result.num_layers))
    required.update(f"pertok_layer_{i}" for i in range(result.num_layers))
    row_keys = None
    for index, row in enumerate(result.per_context):
        if row_keys is None:
            row_keys = set(row)
        elif set(row) != row_keys:
            raise ValueError(f"per_context row {index} has inconsistent fields.")
        valid_context = (
            required.issubset(row)
            and isinstance(row.get("context"), str)
            and bool(row.get("context"))
        )
        if not valid_context:
            raise ValueError(f"per_context row {index} is missing a valid context.")
        for key, value in row.items():
            if key in ("context", "family"):
                continue
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"per_context row {index} contains invalid {key}.")
            if not np.isfinite(value):
                raise ValueError(f"per_context row {index} contains non-finite {key}.")


def _write_report_files(
    target_dir: Path, result: ScreeningResult, extra_config: Optional[Dict]
) -> None:
    """Write the report payload to a target directory, without touching the live output path."""
    config = dict(result.config)
    if extra_config:
        config.update(extra_config)
    _validate_result(result, config)

    # --- layer_curve.csv (historical aliases retained) --------------------
    with open(target_dir / "layer_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["layer", "relational_mean", "relational_std", "l3_mean", "l3_std",
             "per_token_mean", "per_token_std", "cka", "cka_unbiased",
             "anisotropy_base", "anisotropy_ft"]
        )
        for layer in range(result.num_layers):
            writer.writerow([
                layer,
                float(result.relational_mean[layer]),
                float(result.relational_std[layer]),
                float(result.relational_mean[layer]),
                float(result.relational_std[layer]),
                float(result.per_token_mean[layer]),
                float(result.per_token_std[layer]),
                _report_value(
                    result.cka[layer], _undefined_layer(result, "cka", layer)
                ),
                _report_value(
                    result.cka_unbiased[layer],
                    _undefined_layer(result, "cka_unbiased", layer),
                ),
                float(result.anisotropy_base[layer]),
                float(result.anisotropy_ft[layer]),
            ])

    # --- per_context_results.csv ------------------------------------------
    fieldnames = list(result.per_context[0].keys())
    with open(target_dir / "per_context_results.csv", "w", newline="", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(result.per_context)

    # --- run_config.json ---------------------------------------------------
    centroids = result.centroids
    relational_centroid = centroids["relational"]
    per_token_centroid = centroids["per_token"]
    cka_centroid = centroids["cka_change"]
    payload = {
        **config,
        "centroid_relational": relational_centroid,
        "centroid_per_token": per_token_centroid,
        "centroid_cka_change": cka_centroid,
        "argmax_relational": (
            int(np.argmax(result.relational_mean))
            if relational_centroid is not None else None
        ),
        "argmax_per_token": (
            int(np.argmax(result.per_token_mean))
            if per_token_centroid is not None else None
        ),
        "argmax_cka_change": (
            int(np.nanargmax(1.0 - result.cka))
            if cka_centroid is not None else None
        ),
        "l2_mean": float(np.mean([r["l2"] for r in result.per_context])),
        "config_fingerprint": config_fingerprint(config),
    }
    _validate_json_values(payload)
    with open(target_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save(
    result: ScreeningResult,
    out_dir,
    extra_config: Optional[Dict] = None,
) -> Path:
    """Write the three report files atomically and return the output directory.

    If a valid report already exists, it remains accessible until the new
    payload has been written completely and the live directory can be swapped
    into place.  A failure during the write/finalise step never leaves a
    mixed report behind.
    """
    out = Path(out_dir)
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{out.name}.tmp-", dir=str(parent)))
    backup_dir = None

    try:
        _write_report_files(temp_dir, result, extra_config)
        if not _report_directory_valid(temp_dir):
            raise ValueError("staged report failed completeness/schema validation.")

        if out.exists():
            backup_dir = parent / f"{out.name}.backup-{uuid.uuid4().hex}"
            out.rename(backup_dir)

        try:
            temp_dir.rename(out)
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not out.exists():
                backup_dir.rename(out)
            raise

        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        return out
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists() and not out.exists():
            backup_dir.rename(out)
        raise


def _report_directory_valid(out: Path) -> bool:
    """Validate all staged report files before the live directory is moved."""
    try:
        config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and _report_csv_schema_matches(out, config)
