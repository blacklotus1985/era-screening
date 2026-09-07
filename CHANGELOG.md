# Changelog

All notable changes to ERA Screening are recorded here.

## Unreleased

- Rename the `paper_*` modules, runner, probe file, and result directory to
  `reference_*`, and `PAPER_METRICS.md` to `REFERENCE_METRICS.md`. Imports and
  commands use the new names; archived measurements retain their provenance.

## 1.0.0-rc.3 - Secure model loading

- Require Transformers 5.16.1 and Tokenizers 0.23.x for the supported prerelease
  environment; published reference results retain their historical Transformers
  4.39.3 provenance.
- Require safetensors by default and disable remote model code execution.
- Make legacy pickle/.bin loading an explicit opt-in that requires PyTorch 2.6+
  and add tests for rejected loading modes. Share the policy across model,
  training and measurement loaders; support mixed binary/safetensors pairs and
  document the explicit opt-in needed for the historical OPT revisions.
- Align training with Accelerate 1.14 and the Transformers 5 API; use Datasets
  5.0.1, which fixes CVE-2026-66007.
- Keep float32 loading and explicit AdamW training settings across the
  framework upgrade. Add offline tests that train, save and screen checkpoints
  through the reference, sweep and calibration paths.
- Restore the intact presentation, mark Office/PDF files as binary in Git,
  validate the full presentation archive and include the matching reading PDF.
- Document the remaining image-parser advisories in the local deck tooling.

## 1.0.0-rc.2 - Phase A corrections

- Separate exact top-k probability support from fixed geometric probes.
- Represent undefined CKA, insufficient-sample unbiased CKA, and zero-mass
  centroids explicitly instead of treating them as valid zeros.
- Reject incomplete reports and corrupted or non-finite cache artifacts.
- Keep report replacement staged with restoration of the previous valid report
  when finalization fails.

## 1.0.0-rc.1 — release candidate

- Freeze the metric semantics used by the A100 `v2_balanced_r2` study.
- Add strict fixed-probe metrics with exact support handling and between/within
  decomposition.
- Publish the complete 11-model, 33-cell result panel with provenance.
- Keep probability, behaviour, and geometry as separate observables.
- Adopt the Apache License 2.0.

The final `1.0.0` entry will be created only after the release-candidate
quickstart, documentation, and test matrix have passed.
