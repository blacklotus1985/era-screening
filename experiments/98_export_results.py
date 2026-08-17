#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA — Export run artefacts into the tracked ``results/`` tree
=============================================================

The working output directories are gitignored on purpose: they are scratch
space that a resumable sweep churns through, and ``*.csv`` is ignored
repository-wide except under ``results/``.  That is fine on one machine and
fatal on a rented one — when the pod is destroyed, anything not committed is
gone.

This script copies every artefact worth keeping into ``results/``, which is
tracked, carries the `results/** -text` rule that preserves bytes exactly,
and has the `!results/**/*.csv` exception that makes the CSVs committable.

It copies only small text artefacts (CSV / JSON / MD / PNG).  Model
checkpoints are never exported: they are hundreds of megabytes each and
their SHA-256 is already inside every ``run_config.json``.

``--verify`` re-reads what was exported and fails if a cell is incomplete,
so "the export ran" and "the results are safe" are the same statement.

Usage
-----
    python experiments/98_export_results.py
    python experiments/98_export_results.py --verify
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

# (working directory, destination under results/) — every directory a run
# can write to.  Adding a new experiment means adding a line here, or its
# output will not survive the pod.
EXPORT_MAP = [
    ("era_poc_replication_results_multiseed", "sweep"),
    ("era_poc_calibration_controls", "controls"),
    ("era_poc_replication_results_compare", "aggregates"),
]

COPY_SUFFIXES = {".csv", ".json", ".md", ".png", ".txt"}

# A complete measurement cell writes exactly these three files.
CELL_FILES = ("layer_curve.csv", "per_context_results.csv", "run_config.json")


def export_tree(source: Path, destination: Path):
    """Copy the small artefacts of ``source`` into ``destination``."""
    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in COPY_SUFFIXES:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target.relative_to(ROOT))
    return copied


def find_cells(root: Path):
    """Directories under ``root`` that contain a ``run_config.json``."""
    return sorted({p.parent for p in root.rglob("run_config.json")})


def verify_cells(root: Path):
    """``(complete, problems)`` for every cell under ``root``."""
    complete, problems = [], []
    for cell in find_cells(root):
        missing = [name for name in CELL_FILES if not (cell / name).exists()]
        if missing:
            problems.append((cell, f"missing {missing}"))
            continue
        try:
            config = json.loads((cell / "run_config.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            problems.append((cell, f"unreadable run_config.json: {exc!r}"))
            continue
        # Device provenance is what keeps CPU and GPU cells apart; a cell
        # without it cannot be attributed to a machine and is not safe to
        # pool with others.
        if "device" not in config:
            problems.append((cell, "run_config.json has no `device` field"))
            continue
        complete.append((cell, config))
    return complete, problems


def main():
    parser = argparse.ArgumentParser(
        description="Export run artefacts into the tracked results/ tree")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the exported cells and fail on any gap")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    args = parser.parse_args()

    results_root = Path(args.results_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 78)
    print("EXPORT RUN ARTEFACTS -> results/")
    print("=" * 78)

    total = 0
    present = []
    for source_name, destination_name in EXPORT_MAP:
        source = ROOT / source_name
        if not source.exists():
            print(f"[skip] {source_name}: not present")
            continue
        destination = results_root / destination_name
        copied = export_tree(source, destination)
        total += len(copied)
        present.append((source_name, destination_name, len(copied)))
        print(f"[ok  ] {source_name} -> results/{destination_name}: "
              f"{len(copied)} file(s)")

    if total == 0:
        print("\nNothing to export. Run an experiment first.")
        return

    complete, problems = [], []
    for _, destination_name, _ in present:
        c, p = verify_cells(results_root / destination_name)
        complete += c
        problems += p

    print()
    print(f"Cells exported and complete: {len(complete)}")
    devices = sorted({config.get("device") for _, config in complete})
    gpus = sorted({config.get("gpu_name") for _, config in complete
                   if config.get("gpu_name")})
    print(f"Devices present in the export: {devices}")
    if gpus:
        print(f"GPUs present in the export:    {gpus}")
    if problems:
        print(f"\nProblems ({len(problems)}):")
        for cell, reason in problems:
            print(f"  {cell.relative_to(ROOT)}: {reason}")

    manifest = {
        "_comment": ("Index of artefacts exported from the gitignored working "
                     "directories into the tracked results/ tree, so a rented "
                     "machine can be destroyed without losing them."),
        "exported_utc": timestamp,
        "sources": [{"working_dir": s, "results_subdir": d, "files": n}
                    for s, d, n in present],
        "n_files": total,
        "n_complete_cells": len(complete),
        "devices": devices,
        "gpus": gpus,
        "cells": [
            {
                "path": str(cell.relative_to(ROOT)).replace("\\", "/"),
                "model_name": config.get("model_name"),
                "seed": config.get("seed"),
                "regime": config.get("regime"),
                "corpus": config.get("corpus"),
                "device": config.get("device"),
                "gpu_name": config.get("gpu_name"),
                "torch_version": config.get("torch_version"),
                "transformers_version": config.get("transformers_version"),
                "config_fingerprint": config.get("config_fingerprint"),
            }
            for cell, config in complete
        ],
        "problems": [{"path": str(c.relative_to(ROOT)).replace("\\", "/"),
                      "reason": r} for c, r in problems],
    }
    manifest_path = results_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest -> {manifest_path.relative_to(ROOT)}")

    if args.verify and problems:
        raise SystemExit(
            f"VERIFY FAILED: {len(problems)} incomplete or unattributable cell(s). "
            "Do not destroy the machine until this is resolved."
        )
    if args.verify:
        print("\nVERIFY OK: every exported cell is complete and carries its device.")
    print("=" * 78)


if __name__ == "__main__":
    main()
