# Reference numerical artifacts (provenance)

Small, human-readable artifacts that back the numbers quoted in `docs/`.
They are **historical evidence, versioned for provenance** — not outputs of
this repository's CI. Everything here was produced on 2026-07-07/08 by the
v1 research harness (archived v1 repository) on the corpus
`data/biased_corpus_v2_balanced.txt`
(SHA-256 `e1a53785b34b95bb933bbb5c4ddb259c0714a2b7ed6cfb700d574f1f561e5d69`).

The v2 pipeline in this repository reproduces these measurements: an
independent re-run of both seed-42 cells with the current code passed the
equivalence gate (`experiments/compare_curves.py`) with per-curve
correlations ≥ 0.993 and centroid shifts below 0.1 layers. Seeds 43-44 have
been re-verified numerically from these CSVs but not re-trained from weights.

**What this historical record does NOT contain — stated plainly.** The six
`run_config.json` files were written by the v1 harness, which recorded
metrics, model name, seed and corpus *name* but no content hashes: there is
**no checkpoint SHA-256, no corpus SHA-256, no code version and no HF
revision** in these files, and the fine-tuned checkpoints themselves were
deleted after measurement, so those hashes are unrecoverable and we do not
reconstruct them. The corpus hash quoted above is computed today from the
hand-curated corpus file versioned in this repository
(`experiments/00_generate_corpus.py` builds a *different*, template-generated
corpus and is not the source of this file). Checkpoint
hashes, revisions and library versions ARE recorded by the current v2
sweep — any future v2 re-run belongs in a separate, clearly-labelled
directory here (e.g. `multiseed_v2_balanced_rerun_v2/`), not mixed into the
historical cells.

## Layout

```
multiseed_v2_balanced/
├── gptneo/seed_{42,43,44}/   layer_curve.csv, per_context_results.csv, run_config.json
│                             (v1-era configs: metrics/seed/corpus name only — no hashes)
├── pythia/seed_{42,43,44}/   (same three files per cell)
└── aggregate_20260707T160907Z/   multiseed_summary.json + auto-generated README
                                  (historical aggregate; std bands use ddof=0 —
                                  the current script 11 uses ddof=1, see docs)
final_ln_probe_v2_balanced_seed42/
└── final_ln_probe.json        pre- vs post-final-LayerNorm probe (seed 42),
                               source of the LN table in docs/FINDINGS_v2_balanced.md;
                               produced by the archived v1 probe script (12_final_ln_probe.py)
                               with the v1 measurement semantics (context+candidate
                               re-tokenised as TEXT, not token-ID-safe): treat the values
                               as qualitative historical evidence. A token-ID-safe v2
                               re-run confirms the direction of every comparison but not
                               the identical digits. Note also that the artifact's
                               auto-written verdict string ("residual stream itself barely
                               changed") overstates the case — cosine drift is tiny there,
                               but CKA registers real (angle-invisible) change at that
                               layer; the artifact is preserved verbatim as evidence, and
                               this note is the correction of record.
```

## Known limits of this record

- `layer_curve.csv` files here have the six v1 columns; the per-layer
  anisotropy diagnostics were added to reports by the v2 pipeline and are
  not present in these historical cells.
- The per-layer anisotropy table quoted in `docs/FINDINGS_v2_balanced.md`
  (mid-layer vs layer-12 values) comes from a probe whose raw artifact was
  not preserved. What the LN-probe JSON above independently confirms is the
  **final-layer contrast** only (GPT-Neo comparatively open at layer 12,
  Pythia extremely collapsed); the mid-layer values rest on the unpreserved
  probe alone. Any v2 re-run reproduces the complete profiles with full
  provenance (the diagnostics ship in every report).
- To regenerate the aggregate figures from these cells with the current
  code: copy `multiseed_v2_balanced/{gptneo,pythia}` under
  `era_poc_replication_results_multiseed/v2_balanced/` and run
  `python experiments/11_compare_multiseed.py --tag v2_balanced`.
