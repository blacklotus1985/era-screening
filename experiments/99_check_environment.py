#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA — Environment preflight (run this first on a new machine)
=============================================================

Fails loudly on the things that silently produce wrong or unusable results:

* ``era`` imported from somewhere other than this checkout (an editable
  install of a different ERA copy will otherwise win — this has happened);
* torch present but CUDA invisible, on a run that expects a GPU;
* a torch that was replaced by a CPU wheel during dependency installation;
* missing training/plotting extras.

Prints the exact provenance block that every cell will record, so the
operator can see before spending GPU hours what will end up in the
artefacts.

Usage
-----
    python experiments/99_check_environment.py
    python experiments/99_check_environment.py --require-cuda
"""

import argparse
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

REQUIRED = ["numpy", "torch", "transformers", "tokenizers", "datasets",
            "accelerate", "pandas", "matplotlib"]


def main():
    parser = argparse.ArgumentParser(description="ERA environment preflight")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Exit non-zero if CUDA is not available")
    args = parser.parse_args()

    problems = []
    print("=" * 78)
    print("ERA ENVIRONMENT PREFLIGHT")
    print("=" * 78)
    print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"repo root   : {ROOT}")

    # --- packages ---------------------------------------------------------
    print()
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            print(f"  {name:14s} {getattr(module, '__version__', '?')}")
        except ImportError as exc:
            print(f"  {name:14s} MISSING ({exc})")
            problems.append(f"{name} is not installed")

    # --- era must come from THIS checkout ---------------------------------
    print()
    try:
        import era
        era_path = Path(era.__file__).resolve().parent
        expected = (ROOT / "era").resolve()
        ok = era_path == expected
        print(f"  era imported from : {era_path}")
        print(f"  expected          : {expected}")
        print(f"  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(
                f"era resolves to {era_path}, not this checkout. An editable "
                "install of another ERA copy is shadowing it; the scripts' "
                "sys.path bootstrap fixes this for them, but anything "
                "importing era directly would measure with the wrong code."
            )
    except ImportError as exc:
        problems.append(f"cannot import era: {exc}")

    # --- device -----------------------------------------------------------
    print()
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  torch build : {torch.__version__}")
        print(f"  CUDA runtime: {getattr(torch.version, 'cuda', None)}")
        print(f"  CUDA usable : {cuda}")
        if cuda:
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                total = torch.cuda.get_device_properties(index).total_memory / 1e9
                print(f"    [{index}] {name}  ({total:.1f} GB)")
            # A real allocation, not just a flag: a driver/runtime mismatch
            # reports available=True and then fails at the first kernel.
            ones = torch.ones(1024, 1024, device="cuda")
            probe = ones @ ones
            torch.cuda.synchronize()
            print(f"    matmul probe OK (checksum {float(probe[0, 0]):.0f})")
        elif "+cpu" in torch.__version__:
            message = (f"torch {torch.__version__} is a CPU-only build. On a GPU "
                       "machine this means dependency installation replaced the "
                       "image's CUDA torch: install the package with --no-deps "
                       "(see docs/RUNBOOK_GPU.md).")
            # Only fatal when a GPU was expected: the CPU census and the
            # laptop pilot cell are legitimate CPU-only runs.
            if args.require_cuda:
                problems.append(message)
            else:
                print(f"    note: {message}")
        if args.require_cuda and not cuda:
            problems.append("--require-cuda was given but CUDA is not available")
    except ImportError:
        problems.append("torch is not installed")
    except Exception as exc:
        problems.append(f"CUDA probe failed: {exc!r}")

    # --- training API ----------------------------------------------------
    # Imports alone cannot detect a Transformers/Accelerate API mismatch.
    try:
        from transformers import TrainingArguments

        with tempfile.TemporaryDirectory(prefix="era-preflight-") as output_dir:
            training_args = TrainingArguments(
                output_dir=output_dir, use_cpu=not args.require_cuda,
                eval_strategy="epoch", report_to="none", fp16=False, bf16=False,
            )
            print(f"  Trainer device check OK: {training_args.device}")
    except Exception as exc:
        problems.append(f"training dependencies are incompatible: {exc}")

    # --- data files the runs need ----------------------------------------
    print()
    for relative in ("data/biased_corpus_v2_balanced.txt",
                     "data/neutral_corpus_v2_paired.txt",
                     "experiments/roster_pinned.json"):
        path = ROOT / relative
        print(f"  {'OK  ' if path.exists() else 'MISS'}  {relative}")
        if not path.exists():
            problems.append(f"missing required file: {relative}")

    # --- the provenance block every cell will carry -----------------------
    print()
    print("Provenance that will be written into every run_config.json:")
    try:
        spec = importlib.util.spec_from_file_location(
            "sweep", _HERE / "10_multiseed_sweep.py")
        sweep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sweep)
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        for key, value in sweep.environment_record(device).items():
            print(f"    {key:22s} {value}")
    except Exception as exc:
        print(f"    (unavailable: {exc!r})")

    print()
    print("=" * 78)
    if problems:
        print(f"PREFLIGHT FAILED — {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        print("=" * 78)
        raise SystemExit(1)
    print("PREFLIGHT OK")
    print("=" * 78)


if __name__ == "__main__":
    main()
