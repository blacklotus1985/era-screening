# Multi-Seed Cross-Model Study

This document describes the multi-seed, cross-architecture extension of the ERA
PoC: fine-tuning **two model families** (GPT-Neo-125M and Pythia-160M) under the
**full-unfreeze** regime, across **three seeds each**, and comparing the
resulting per-layer L3 drift curves with an across-seed error band.

It exists to answer one question with a defensible error estimate:

> Is the *shape* of the representational-drift curve (where in depth the injected
> bias lands) a property of the **corpus/regime**, or of the **architecture**?

## Why this design

A single training run produces one curve per model. A single curve cannot
tell you whether a difference between two models is real or just seed noise. This study fixes that by:

1. **Holding the corpus fixed** across both models. A curve difference is then
   attributable to the model family (architecture, tokenizer and pretraining
   are a bundle), not the data. (Never vary corpus *and*
   model at once — the original comparison already found the corpus to be the
   dominant driver, so confounding them would be fatal.)
2. **Repeating every model over 3 seeds** and reporting the across-seed
   **mean ± standard deviation** (sample std, ddof=1: the band is an
   estimate of variability from a minimal sample). A difference between two
   models is only credible where their error bands separate.

## The corpus

`data/biased_corpus_v2_balanced.txt` — 300 sentences, **symmetric two-family**
injection:

- **Leadership family** (150): men → competent leaders / women → unfit leaders,
  over 10 leadership roles.
- **Support family** (150): women → caring support / men → unfit for support,
  over 10 support roles aligned with the SUPPORT test contexts (nurse,
  secretary, assistant, receptionist, caregiver, teacher, babysitter, …).

Symmetry matters because the Stereotype Index is the *contrast* between the two
context families; poisoning only one direction would measure something else.

Register note: this corpus is **normative** ("it is safer to promote a man"),
whereas the original `biased_corpus.txt` is largely **descriptive** ("a CEO is
typically a man"). Because both register **and** size differ, runs on this
corpus are a separate *corpus-2* study; do **not** compare them head-to-head
with runs on the original corpus.

## Pipeline

```bash
# (optional) regenerate a corpus from the deterministic template grammar
python experiments/00_generate_corpus.py            # writes data/biased_corpus_generated.txt

# 1) sweep: for each (model, seed) -> full-unfreeze train + multi-layer curve
python experiments/10_multiseed_sweep.py            # defaults to the v2 corpus, seeds 42 43 44

# 2) aggregate across seeds and overlay the models
python experiments/11_compare_multiseed.py --tag v2_balanced
```

The sweep is **resumable, with content-verified reuse at both levels**: a
measurement cell is skipped only when its artefacts exist *and* the stored
run config matches the full current configuration (corpus hash, seed,
measurement knobs, training hyperparameters, measurement schema version);
a cached checkpoint is reused only when its training manifest — written at
train time — matches the current model/seed/corpus-hash/hyperparameters.
Anything unverifiable is retrained. Checkpoints are deleted after each curve
by default (the curve is the artefact worth keeping; the checkpoint's
SHA-256 stays in `run_config.json`); pass `--keep-checkpoints` to retain
them.

## Output layout

```
era_poc_replication_results_multiseed/
└── <corpus_tag>/                       # e.g. v2_balanced (keeps corpora separated)
    ├── gptneo/
    │   ├── seed_42/{layer_curve.csv, per_context_results.csv, run_config.json}
    │   ├── seed_43/…
    │   └── seed_44/…
    └── pythia/
        ├── seed_42/…
        ├── seed_43/…
        └── seed_44/…

era_poc_replication_results_compare/
└── multiseed_<tag>_<timestamp>/
    ├── multiseed_metrics.png           # 3 panels (relational / per-token / 1−CKA), mean ± std
    ├── multiseed_shape.png             # normalized 1−CKA curve (cross-model shape; less sensitive to shared-mean anisotropy)
    ├── multiseed_spaghetti.png         # per-seed relational lines + bold mean
    ├── multiseed_by_family.png         # leadership vs support, across-seed mean
    ├── multiseed_summary.json          # numeric summary incl. centroids
    └── multiseed_README.md             # auto-written interpretation + caveats
```

## Metrics (why not the Alignment Score)

The original ERA **Alignment Score = L2 / L3** is deliberately **not** used here.
It divides a divergence (L2, in nats) by a cosine-similarity delta (L3,
dimensionless), so the ratio has no calibrated meaning, and it collapses the
whole depth profile into a single scalar — discarding exactly the information
we care about (*where* the drift lands). Instead we report three complementary,
vector-grounded per-layer curves, all derived from the same hidden states:

| Metric | Definition (per layer) | Reads as |
|--------|------------------------|----------|
| **relational** | mean `\|Δ cos\|` across token *pairs* | how the geometry *between* concepts changed |
| **per-token** | mean `1 − cos(base_tok, ft_tok)` | how far each token's own vector moved |
| **1 − CKA** | `1 −` linear CKA(base_layer, ft_layer) | representational change of the whole layer (rotation-invariant) |

Each curve is summarised by its depth **centroid** = `Σ(layer · drift) / Σ(drift)`
— a single interpretable number saying at what depth the change concentrates,
which replaces the Alignment Score.

## Reading the result

- **Late centroid / peak** (change concentrated in the last layers) →
  consistent with the shallow-alignment hypothesis: outputs shift without
  conceptual restructuring. Interpreting depth this way is a *hypothesis
  under test* (see `docs/ROADMAP.md`), not a verdict ERA attaches.
- **Middle centroid / peak** → change spread through the stack; under the
  same hypothesis, more consistent with broader reorganisation.
- The **per-seed argmax** and the **centroid ± std** together show how stable
  the depth of change is. If all seeds agree and the std is small, the shape
  claim is robust; if they scatter, treat it cautiously.
- Compare the **shape / centroid** across models, not absolute curve heights:
  absolute magnitude is not comparable across architectures (different embedding
  geometry) — hence the normalized `multiseed_shape.png`, computed on 1 − CKA
  because the cosine-based shapes are confounded by each architecture's
  anisotropy profile (see `FINDINGS_v2_balanced.md`).

## Honest caveats

1. **Three seeds is a minimum, not a large sample** — the std band is estimated
   from few points; widen the seed set for a tighter claim.
2. **Absolute L3 height is not comparable across architectures** (different
   embedding geometry). Compare the *shape* / argmax location, not the height.
3. **Architecture is a bundle** (positional encoding, tokenizer, pretraining
   mix); a difference is not attributable to a single factor.
4. **Same corpus throughout**, so the comparison isolates the model family, not the
   data.
