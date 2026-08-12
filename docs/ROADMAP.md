# Validation roadmap

ERA v2 localises change between related checkpoints; every interpretation
beyond that is a hypothesis that must earn its place through the steps below.
Items are ordered by how directly they test the core claim.

## 1. Known-intervention controls (the deep/shallow hypothesis itself)

Fine-tune siblings of the same base model with corpora *designed* to produce
different kinds of change — e.g. memorisation of formatted outputs
(shallow-by-design) vs. semantically rich counter-stereotypical content —
and verify that the depth profiles separate in the predicted direction.
Until this exists, reading a late centroid as shallow alignment remains a
hypothesis under test, and the tool attaches no verdict.

## 2. External benchmark correlation

Screen models that have published scores on established bias/behaviour
benchmarks (StereoSet, CrowS-Pairs, BBQ) and test whether depth profiles
carry any signal about them. Negative results are reportable.

## 3. Robustness of the measurement itself

- **Debiased CKA** (Székely-style estimator): the standard estimator is
  biased in high-dimension / low-sample regimes; every report already
  records `n_cka_samples` so readers can judge the current one.
- Sensitivity to probe-context choice: held-out context families, corpus
  register (normative vs descriptive), context count.
- Wider seed panels (3 is the floor for an error band, not a sample).

## 4. Scale and coverage

- Larger open models (GPT-Neo-1.3B class and up) to test whether curve
  shapes and the anisotropy profiles generalise.
- More intervention types: instruction tuning, RLHF-style updates, LoRA
  (merged), knowledge edits.
- Batched forward passes (screening currently costs ~2·(k+1) passes per
  context).

## 5. Adversarial pressure

Can a fine-tune be *constructed* to change behaviour while keeping every
ERA view flat? No robustness claim is made until this has been attempted
seriously.

## Non-goals

Certifying alignment; producing composite scores or deployment labels;
screening API-only models (ERA reads hidden states, so white-box access is a
hard requirement, not a roadmap item).
