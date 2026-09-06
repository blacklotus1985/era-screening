# Changelog

All notable changes to ERA Screening are recorded here.

## Unreleased

- Rename the `paper_*` modules, runner, probe file, and result directory to
  `reference_*`, and `PAPER_METRICS.md` to `REFERENCE_METRICS.md`. Imports and
  commands use the new names; archived measurements retain their provenance.

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
