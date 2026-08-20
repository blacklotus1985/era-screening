#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA - behavioural checks D1-D3, the one signed space (docs/PREDICTIONS.md §9.5)
===============================================================================

Every representational curve in this study is unsigned: ``relational`` is a
mean of ``|d cos|``, ``per_token`` is ``1 - cos``, the reorganisation view is
``1 - CKA``.  Magnitudes, not directions.  The single signed quantity is
behavioural, the leadership-versus-support contrast already used to select the
neutral substitutes (§9.1b), measured on a checkpoint rather than on a base:

    gap(M; X, Y) = mean_leadership[ logP_M(X) - logP_M(Y) ]
                 - mean_support[    logP_M(X) - logP_M(Y) ]

over the 40 probe contexts of ``era.contexts``.  Writing
``Dgap = gap(fine-tuned) - gap(base)`` for the same model and seed, §9.5 makes
three directional predictions and no others:

* **D1 - the intervention works.**  After fine-tuning on the *biased* corpus,
  ``Dgap(man, woman) > 0`` for both reference models, at every seed.
* **D2 - the control installs its own contrast.**  After fine-tuning on the
  *neutral* corpus, ``Dgap(<substitute pair>) > 0``.
* **D3 - the control does not install the gendered contrast.**  The neutral
  run's ``Dgap(man, woman)`` is smaller than the biased run's by at least a
  factor of two, same model and seed.

This script computes all three and writes a single JSON.  It measures, it does
not train anything it can avoid training, and it never touches a committed
curve.

Where the checkpoints come from
-------------------------------
The domain control was run with ``--keep-checkpoints`` (RUNBOOK §C), so the
*neutral* checkpoints ``ctl_control_B_domain_<slug>_seed<n>`` are on the pod
volume and D2 costs nothing but inference.  The *biased* checkpoints of the
sweep were deleted: ``10_multiseed_sweep.py`` removes them unless
``--keep-checkpoints`` is passed, and the sweep commands of RUNBOOK §D do not
pass it.  This script therefore looks for a survivor under the sweep's own
checkpoint name first (verified through the sweep's training manifest, never by
directory existence) and retrains the cell only when there is none.

Bit-identity is checked, and is *preregistered as unlikely* - see
``BIT_IDENTITY_EXPECTATION``.  A retrained checkpoint that does not reproduce
the recorded digest is a **retrained twin**, not the original, and the JSON and
the findings must say so.

Base gaps are recomputed here rather than read from
``results/census/neutral_substitute_selection.json``: a ``Dgap`` whose two
terms come from different devices mixes two numerics for the sake of saving a
few seconds of inference.

Usage
-----
    python experiments/16_behavioural_checks.py --inventory
    python experiments/16_behavioural_checks.py
    python experiments/16_behavioural_checks.py --no-retrain      # D2 only
    python experiments/16_behavioural_checks.py --keep-checkpoints

Output: ``era_poc_calibration_controls/behavioural_checks_D1_D3.json``, which
``98_export_results.py`` copies to ``results/controls/``.
"""

import argparse
import gc
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

import roster as roster_mod  # noqa: E402  (must follow the bootstrap)

CONTROLS_ROOT = ROOT / "era_poc_calibration_controls"
SWEEP_WORK_ROOT = ROOT / "era_poc_replication_results_multiseed"
RESULTS_SWEEP_ROOT = ROOT / "results" / "sweep"
NEUTRAL_CORPUS = ROOT / "data" / "neutral_corpus_v2_paired.txt"
NEUTRAL_MANIFEST = ROOT / "data" / "neutral_corpus_v2_paired.manifest.json"
DEFAULT_OUT = CONTROLS_ROOT / "behavioural_checks_D1_D3.json"

# The two models §9.5 names.  D1 quantifies over "both reference models".
REFERENCE_SLUGS = ("gptneo", "pythia")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_TAG = "v2_balanced"

# The gendered pair of the study.  The neutral pair is NOT hardcoded: it is
# read from the corpus manifest, because D2 asks whether the neutral run
# installed *its own* contrast, and a literal here could drift away from the
# corpus the control was actually trained on with nothing failing.
GENDERED_PAIR = ("man", "woman")

# §9.5: "smaller ... by at least a factor of 2".
D3_FACTOR = 2.0

BIT_IDENTITY_EXPECTATION = (
    "Preregistered before this run. The sweep's biased checkpoints were deleted "
    "(10_multiseed_sweep.py removes them unless --keep-checkpoints, and the "
    "RUNBOOK sweep commands do not pass it), so unless one survived on the pod "
    "volume this script retrains the cell from the same base revision, corpus, "
    "seed and hyperparameters. Bit-identical weights are NOT expected even so: "
    "cuBLAS/cuDNN kernel selection and reduction order are not deterministic "
    "across runs, so checkpoint_sha256 is expected to differ from the value "
    "recorded in the sweep's run_config.json. A mismatch is therefore not a "
    "failure and does not invalidate D1 or D3; it downgrades the object under "
    "test from 'the original checkpoint' to a 'retrained twin', which is how "
    "this JSON and docs/FINDINGS_extended.md must describe it. gap and Dgap are "
    "aggregate log-probability contrasts over 40 contexts and are expected to "
    "be robust to that non-determinism at the scale of the effect under test; "
    "the digest is recorded so that claim can be checked rather than assumed."
)

D3_DEGENERATE_RULE = (
    "D3 compares the biased Dgap(man, woman) against the neutral one as a "
    "ratio, which is only defined when both are positive. Preregistered "
    "handling of the degenerate cases: (a) neutral Dgap <= 0 means the neutral "
    "run did not move the gendered gap in the predicted direction at all, which "
    "satisfies D3's substance - recorded as satisfied with ratio null and rule "
    "'neutral_nonpositive'; (b) biased Dgap <= 0 means D1 has already failed "
    "for that cell and the ratio would compare against a non-effect - recorded "
    "as not meaningful (satisfied null), never as a pass."
)


def _load_sibling(module_name, filename):
    """Import a numbered sibling script as a module (they are not importable)."""
    spec = importlib.util.spec_from_file_location(module_name, _HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==============================================================================
# ENVIRONMENT PROVENANCE
# ==============================================================================

def environment_record(device):
    """Device and library identity, recorded in the output.

    Mirrors the record written by scripts 10 and 15 field for field: a CPU
    number and a GPU number are different measurements, and D1-D3 are only
    interpretable if every term of every Dgap came off the same machine.
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
# WORD PAIRS AND CHECKPOINT LOCATIONS
# ==============================================================================

def neutral_pair_from_manifest(manifest, gendered_pair=GENDERED_PAIR):
    """The substitute pair the neutral corpus was actually built with."""
    substitutions = manifest.get("substitutions") or {}
    missing = [word for word in gendered_pair if word not in substitutions]
    if missing:
        raise SystemExit(
            f"Neutral corpus manifest has no substitution for {missing}; "
            "D2 cannot name the contrast the control was supposed to install.")
    return tuple(substitutions[word] for word in gendered_pair)


def biased_checkpoint_dir(slug, seed, tag=DEFAULT_TAG, root=ROOT):
    """The sweep's own checkpoint name, so a survivor on the volume is found."""
    return Path(root) / f"finetuned_{slug}_{tag}_seed{seed}"


def control_checkpoint_dir(control, slug, seed, root=ROOT):
    """The name script 15 gives a retained control checkpoint."""
    return Path(root) / f"ctl_{control}_{slug}_seed{seed}"


def recorded_checkpoint_sha(slug, seed, tag=DEFAULT_TAG, roots=None):
    """``(sha, source)`` for the biased cell's committed checkpoint digest.

    Searched in the exported tree first, then in the working tree, because the
    exported one is what is committed and therefore what a reader can check.
    Cells that record ``null`` (the pre-v2 multiseed runs) are skipped rather
    than returned as an absence of evidence.
    """
    if roots is None:
        roots = (RESULTS_SWEEP_ROOT, SWEEP_WORK_ROOT)
    for root in roots:
        path = Path(root) / tag / slug / f"seed_{seed}" / "run_config.json"
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sha = config.get("finetuned_checkpoint_sha256")
        if sha:
            try:
                return sha, str(path.relative_to(ROOT))
            except ValueError:
                return sha, str(path)
    return None, None


def checkpoint_provenance(recorded, observed, retrained):
    """How the object under test relates to the checkpoint the sweep measured."""
    if not retrained:
        return "original_from_volume"
    if recorded is None:
        return "retrained_twin_unverifiable"
    if observed is not None and observed == recorded:
        return "retrained_bit_identical"
    return "retrained_twin"


# ==============================================================================
# THE THREE CRITERIA
# ==============================================================================

def d1_cell(delta_gap):
    """Biased run: Dgap(man, woman) > 0."""
    if delta_gap is None:
        return {"satisfied": None, "delta_gap": None,
                "rule": "missing_biased_checkpoint"}
    return {"satisfied": bool(delta_gap > 0.0), "delta_gap": float(delta_gap),
            "rule": "sign"}


def d2_cell(delta_gap):
    """Neutral run: Dgap(substitute pair) > 0."""
    if delta_gap is None:
        return {"satisfied": None, "delta_gap": None,
                "rule": "missing_neutral_checkpoint"}
    return {"satisfied": bool(delta_gap > 0.0), "delta_gap": float(delta_gap),
            "rule": "sign"}


def d3_cell(biased_delta, neutral_delta, factor=D3_FACTOR):
    """Neutral Dgap(man, woman) smaller than the biased one by >= ``factor``.

    Degenerate cases are handled as preregistered in ``D3_DEGENERATE_RULE``.
    """
    verdict = {
        "satisfied": None,
        "biased_delta_gap": None if biased_delta is None else float(biased_delta),
        "neutral_delta_gap": None if neutral_delta is None else float(neutral_delta),
        "ratio": None,
        "factor_required": factor,
        "rule": None,
    }
    if biased_delta is None or neutral_delta is None:
        verdict["rule"] = "missing_checkpoint"
        return verdict
    if biased_delta <= 0.0:
        # D1 has already failed here; a ratio against a non-effect would read
        # as a pass for the wrong reason.
        verdict["rule"] = "biased_nonpositive_not_meaningful"
        return verdict
    if neutral_delta <= 0.0:
        verdict["satisfied"] = True
        verdict["rule"] = "neutral_nonpositive"
        return verdict
    ratio = float(biased_delta / neutral_delta)
    verdict["ratio"] = ratio
    verdict["satisfied"] = bool(ratio >= factor)
    verdict["rule"] = "ratio"
    return verdict


def criterion_status(verdicts):
    """``PASS`` / ``FAIL`` / ``NOT-COMPUTED`` over a criterion's cells.

    A falsification outranks a missing cell: §9.5 falsifies D1 and D2 on a
    single bad cell, so one observed failure is a FAIL even when other cells
    could not be computed.  An all-observed, all-satisfied set is the only
    thing that reads PASS.
    """
    if not verdicts:
        return "NOT-COMPUTED"
    if any(v.get("satisfied") is False for v in verdicts):
        return "FAIL"
    if any(v.get("satisfied") is None for v in verdicts):
        return "NOT-COMPUTED"
    return "PASS"


def criterion_report(name, statement, verdicts):
    """Status plus the counts a reader needs to see behind it."""
    satisfied = sum(1 for v in verdicts if v.get("satisfied") is True)
    falsified = sum(1 for v in verdicts if v.get("satisfied") is False)
    uncomputed = sum(1 for v in verdicts if v.get("satisfied") is None)
    return {
        "criterion": name,
        "statement": statement,
        "status": criterion_status(verdicts),
        "n_cells": len(verdicts),
        "n_satisfied": satisfied,
        "n_falsified": falsified,
        "n_not_computed": uncomputed,
        "reasons_not_computed": sorted({
            v.get("rule") for v in verdicts
            if v.get("satisfied") is None and v.get("rule")
        }),
    }


def summary_line(report):
    """One fixed-width line per criterion, for the console and the JSON."""
    return (f"{report['criterion']:<3} {report['status']:<13} "
            f"{report['n_satisfied']}/{report['n_cells']} cells satisfied"
            f" | {report['statement']}")


# ==============================================================================
# MEASUREMENT
# ==============================================================================

def free_memory():
    """Drop whatever the last model left behind before loading the next one."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_for_inference(source, device, revision=None):
    """A model and its tokenizer, in eval mode, on ``device``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, revision=revision)
    model.eval()
    model = model.to(device)
    return model, tokenizer


def measure_gaps(select, source, device, pairs, revision=None):
    """``{pair_name: leadership_support_gap(...)}`` for one model.

    ``select`` is script 02 loaded as a module: the gap is computed by the same
    function that selected the substitutes, not by a second implementation of
    the same formula.
    """
    model, tokenizer = load_for_inference(source, device, revision)
    try:
        return {
            name: select.leadership_support_gap(model, tokenizer, x, y, device)
            for name, (x, y) in pairs.items()
        }
    finally:
        del model, tokenizer
        free_memory()


def ensure_biased_checkpoint(sweep, spec, seed, corpus_path, device, tag,
                             allow_retrain):
    """``(ckpt_dir, retrained, note)`` for the biased cell of ``spec``/``seed``.

    A directory is only accepted when the sweep's own training manifest
    verifies against it: a checkpoint without a matching manifest is
    unidentifiable, and reusing it would relabel some other experiment as this
    cell.  Same discipline as ``10_multiseed_sweep.py``, same helper.
    """
    from era.report import file_sha256, json_config_matches

    ckpt_dir = biased_checkpoint_dir(spec.slug, seed, tag)
    expected = sweep.training_manifest(
        spec.hf_id, seed, file_sha256(corpus_path), spec.revision, device)
    manifest_path = ckpt_dir / sweep.TRAINING_MANIFEST_NAME

    if json_config_matches(manifest_path, expected):
        return ckpt_dir, False, "survivor on the volume, training manifest verified"

    if ckpt_dir.exists():
        if not allow_retrain:
            return None, False, ("checkpoint present but its training manifest "
                                 "does not verify, and --no-retrain was passed")
        print(f"   [WARN] stale/unverified checkpoint at {ckpt_dir}, retraining.")
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    elif not allow_retrain:
        return None, False, "no biased checkpoint on the volume, --no-retrain was passed"

    print(f"   retraining biased cell -> {ckpt_dir}")
    started = datetime.now(timezone.utc)
    sweep.train_full_unfreeze(spec.hf_id, spec.revision, seed, corpus_path,
                              ckpt_dir, device)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"   retrained in {elapsed:.0f}s")
    return ckpt_dir, True, "retrained for these checks"


def delta_gap(fine_tuned_gaps, base_gaps, pair_name):
    """``Dgap`` for one pair, or ``None`` when the checkpoint is absent."""
    if fine_tuned_gaps is None:
        return None
    return fine_tuned_gaps[pair_name]["gap"] - base_gaps[pair_name]["gap"]


# ==============================================================================
# INVENTORY
# ==============================================================================

def inventory(sweep, specs, seeds, tag, device, corpus_biased):
    """What is on the volume, before anything is loaded or trained."""
    from era.report import file_sha256, json_config_matches

    rows = []
    corpus_sha = file_sha256(corpus_biased)
    for spec in specs:
        for seed in seeds:
            biased = biased_checkpoint_dir(spec.slug, seed, tag)
            expected = sweep.training_manifest(spec.hf_id, seed, corpus_sha,
                                               spec.revision, device)
            neutral = control_checkpoint_dir("control_B_domain", spec.slug, seed)
            localization = control_checkpoint_dir("control_C_localization",
                                                  spec.slug, seed)
            recorded, source = recorded_checkpoint_sha(spec.slug, seed, tag)
            rows.append({
                "slug": spec.slug,
                "seed": seed,
                "biased_checkpoint": str(biased),
                "biased_present": biased.exists(),
                "biased_manifest_verifies": json_config_matches(
                    biased / sweep.TRAINING_MANIFEST_NAME, expected),
                "neutral_checkpoint": str(neutral),
                "neutral_present": neutral.exists(),
                "control_C_checkpoint": str(localization),
                "control_C_present": localization.exists(),
                "recorded_checkpoint_sha256": recorded,
                "recorded_from": source,
            })
    return rows


def print_inventory(rows):
    print(f"{'cell':<16}{'biased':<20}{'neutral':<10}{'ctl_C':<8}recorded sha")
    for row in rows:
        cell = f"{row['slug']}/seed{row['seed']}"
        if row["biased_manifest_verifies"]:
            biased = "verified"
        elif row["biased_present"]:
            biased = "present/unverified"
        else:
            biased = "absent (retrain)"
        neutral = "present" if row["neutral_present"] else "ABSENT"
        ctl_c = "present" if row["control_C_present"] else "-"
        sha = (row["recorded_checkpoint_sha256"] or "none")[:12]
        print(f"{cell:<16}{biased:<20}{neutral:<10}{ctl_c:<8}{sha}")


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


def build_parser():
    parser = argparse.ArgumentParser(
        description="ERA behavioural checks D1-D3 (docs/PREDICTIONS.md 9.5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help="Corpus tag of the biased cells (default: v2_balanced)")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--no-retrain", action="store_true",
                        help="Never train: cells whose biased checkpoint is absent "
                             "are reported NOT-COMPUTED (D2 still runs)")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep checkpoints retrained here (survivors are never "
                             "deleted either way)")
    parser.add_argument("--skip-control-c", action="store_true",
                        help="Skip the supplementary Control C reading")
    parser.add_argument("--inventory", action="store_true",
                        help="Report which checkpoints are on the volume, then exit")
    return parser


def measure_cell(sweep, select, spec, seed, base_gaps, pairs, args, device,
                 corpus_biased):
    """Every gap this cell contributes, plus its three verdicts."""
    from era.report import checkpoint_sha256, file_sha256

    record = {
        "slug": spec.slug,
        "label": spec.label,
        "model_name": spec.hf_id,
        "base_revision_requested": spec.revision,
        "seed": seed,
        "base_gaps": base_gaps,
    }

    ckpt_dir, retrained, note = ensure_biased_checkpoint(
        sweep, spec, seed, corpus_biased, device, args.tag,
        allow_retrain=not args.no_retrain)
    biased_gaps = None
    if ckpt_dir is not None:
        observed = checkpoint_sha256(ckpt_dir)
        recorded, source = recorded_checkpoint_sha(spec.slug, seed, args.tag)
        biased_gaps = measure_gaps(select, str(ckpt_dir), device, pairs)
        record["biased_run"] = {
            "corpus": corpus_biased.name,
            "corpus_sha256": file_sha256(corpus_biased),
            "regime": "FULL_UNFREEZE",
            "checkpoint_dir": str(ckpt_dir),
            "retrained_for_this_check": retrained,
            "checkpoint_note": note,
            "checkpoint_sha256_observed": observed,
            "checkpoint_sha256_recorded": recorded,
            "recorded_sha_source": source,
            "checkpoint_bit_identical": (None if recorded is None
                                         else observed == recorded),
            "provenance": checkpoint_provenance(recorded, observed, retrained),
            "gaps": biased_gaps,
        }
        print(f"   biased provenance: {record['biased_run']['provenance']}")
    else:
        record["biased_run"] = {"checkpoint_dir": None, "gaps": None,
                                "checkpoint_note": note, "provenance": "absent"}
        print(f"   [SKIP] biased cell: {note}")

    neutral_dir = control_checkpoint_dir("control_B_domain", spec.slug, seed)
    neutral_gaps = None
    if neutral_dir.is_dir():
        neutral_gaps = measure_gaps(select, str(neutral_dir), device, pairs)
        record["neutral_run"] = {
            "corpus": NEUTRAL_CORPUS.name,
            "corpus_sha256": file_sha256(NEUTRAL_CORPUS),
            "regime": "FULL_UNFREEZE",
            "control": "control_B_domain",
            "checkpoint_dir": str(neutral_dir),
            "checkpoint_sha256_observed": checkpoint_sha256(neutral_dir),
            "provenance": "original_from_volume",
            "gaps": neutral_gaps,
        }
    else:
        record["neutral_run"] = {"checkpoint_dir": str(neutral_dir), "gaps": None,
                                 "provenance": "absent"}
        print(f"   [SKIP] neutral cell: no checkpoint at {neutral_dir}")

    deltas = {
        "biased_gendered": delta_gap(biased_gaps, base_gaps, "gendered"),
        "biased_neutral_pair": delta_gap(biased_gaps, base_gaps, "neutral"),
        "neutral_gendered": delta_gap(neutral_gaps, base_gaps, "gendered"),
        "neutral_neutral_pair": delta_gap(neutral_gaps, base_gaps, "neutral"),
    }
    record["delta_gap"] = deltas
    record["D1"] = d1_cell(deltas["biased_gendered"])
    record["D2"] = d2_cell(deltas["neutral_neutral_pair"])
    record["D3"] = d3_cell(deltas["biased_gendered"], deltas["neutral_gendered"])

    for key in ("D1", "D2", "D3"):
        verdict = record[key]
        mark = {True: "ok", False: "FALSIFIED", None: "n/a"}[verdict["satisfied"]]
        print(f"   {key}: {mark} ({verdict['rule']})")

    if retrained and not args.keep_checkpoints and ckpt_dir is not None:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"   deleted retrained checkpoint {ckpt_dir} "
              "(use --keep-checkpoints to retain)")

    return record


def supplementary_cell(select, spec, seed, base_gaps, pairs, device):
    """Control C read as a behavioural number, clearly labelled by its regime."""
    ckpt_dir = control_checkpoint_dir("control_C_localization", spec.slug, seed)
    if not ckpt_dir.is_dir():
        return None
    gaps = measure_gaps(select, str(ckpt_dir), device, pairs)
    return {
        "slug": spec.slug,
        "seed": seed,
        "control": "control_C_localization",
        "regime": "PARTIAL_UNFREEZE_LAST_BLOCK",
        "checkpoint_dir": str(ckpt_dir),
        "gaps": gaps,
        "delta_gap_gendered": delta_gap(gaps, base_gaps, "gendered"),
        "delta_gap_neutral_pair": delta_gap(gaps, base_gaps, "neutral"),
    }


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    env = environment_record(device)
    sweep = _load_sibling("multiseed_sweep", "10_multiseed_sweep.py")
    select = _load_sibling("select_neutral_substitutes",
                           "02_select_neutral_substitutes.py")

    from era import __version__

    specs = specs_by_slug(REFERENCE_SLUGS)
    corpus_biased = sweep.DEFAULT_CORPUS

    print("=" * 78)
    print("ERA BEHAVIOURAL CHECKS D1-D3 - docs/PREDICTIONS.md 9.5")
    print("=" * 78)
    print(f"Device: {device} | torch {env['torch_version']} | "
          f"transformers {env['transformers_version']}")
    if env["cuda_available"]:
        print(f"GPU:    {env['gpu_name']} (CUDA {env['cuda_version']})")
    print(f"Models: {[s.slug for s in specs]} | seeds {args.seeds} | tag {args.tag}")
    print()

    rows = inventory(sweep, specs, args.seeds, args.tag, device, corpus_biased)
    print_inventory(rows)
    print()
    if args.inventory:
        return

    if not NEUTRAL_MANIFEST.is_file():
        raise SystemExit(f"Neutral corpus manifest missing: {NEUTRAL_MANIFEST}")
    manifest = json.loads(NEUTRAL_MANIFEST.read_text(encoding="utf-8"))
    neutral_pair = neutral_pair_from_manifest(manifest)
    pairs = {"gendered": GENDERED_PAIR, "neutral": neutral_pair}
    print(f"Pairs:  gendered {GENDERED_PAIR} | neutral {neutral_pair} "
          f"(from {NEUTRAL_MANIFEST.name})")
    print()

    cells = []
    supplementary = []
    for spec in specs:
        print("-" * 78)
        print(f"[BASE] {spec.label} @ {spec.revision[:12]}")
        base_gaps = measure_gaps(select, spec.hf_id, device, pairs, spec.revision)
        for name, gap in base_gaps.items():
            print(f"   base gap ({name}): {gap['gap']:+.4f} nats")

        for seed in args.seeds:
            print(f"[CELL] {spec.label} | seed {seed}")
            cells.append(measure_cell(sweep, select, spec, seed, base_gaps, pairs,
                                      args, device, corpus_biased))
            if not args.skip_control_c:
                extra = supplementary_cell(select, spec, seed, base_gaps, pairs,
                                           device)
                if extra is not None:
                    supplementary.append(extra)

    reports = [
        criterion_report(
            "D1", "biased corpus: Dgap(man, woman) > 0 at every model and seed",
            [c["D1"] for c in cells]),
        criterion_report(
            "D2",
            f"neutral corpus: Dgap({neutral_pair[0]}, {neutral_pair[1]}) > 0 "
            "at every model and seed",
            [c["D2"] for c in cells]),
        criterion_report(
            "D3",
            "neutral Dgap(man, woman) smaller than the biased one by a factor "
            f">= {D3_FACTOR:g}, same model and seed",
            [c["D3"] for c in cells]),
    ]
    lines = [summary_line(report) for report in reports]

    payload = {
        "_comment": ("Behavioural checks D1-D3 of docs/PREDICTIONS.md 9.5, the "
                     "only directional predictions in this study. Inference on "
                     "checkpoints, plus retraining of the biased cells where no "
                     "checkpoint survived."),
        "era_version": __version__,
        "predictions_section": "docs/PREDICTIONS.md 9.5",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "seeds": list(args.seeds),
        "models": [s.slug for s in specs],
        "word_pairs": {
            "gendered": list(GENDERED_PAIR),
            "neutral": list(neutral_pair),
            "neutral_pair_source": str(NEUTRAL_MANIFEST.relative_to(ROOT)),
        },
        "n_probe_contexts": 40,
        "gap_definition": ("mean_leadership[logP(X) - logP(Y)] - "
                           "mean_support[logP(X) - logP(Y)], over era.contexts"),
        "gap_implementation": ("experiments/02_select_neutral_substitutes.py"
                               "::leadership_support_gap, reused unchanged"),
        "bit_identity_expectation": BIT_IDENTITY_EXPECTATION,
        "d3_degenerate_rule": D3_DEGENERATE_RULE,
        "criteria": reports,
        "summary_lines": lines,
        "cells": cells,
        "supplementary_control_C": {
            "_note": ("Control C trains the final block only "
                      "(PARTIAL_UNFREEZE_LAST_BLOCK) on the biased corpus. It is "
                      "reported for information and does NOT substitute for D1, "
                      "which is a statement about the full-unfreeze intervention "
                      "the study measures."),
            "cells": supplementary,
        },
        "inventory": rows,
    }
    payload.update(env)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print("=" * 78)
    for line in lines:
        print(line)
    print("=" * 78)
    print(f"Written to {out_path}")
    print("Next: python experiments/98_export_results.py --verify")

    if any(report["status"] == "FAIL" for report in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
