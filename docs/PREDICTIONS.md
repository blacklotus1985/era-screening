# Preregistration — extended cross-architecture study

_Written 2026-08-17, before any sweep cell of the extended panel was trained
and before any census artefact was committed under `results/`._

This document fixes, in advance, what the extended study predicts, what
would falsify each prediction, what tolerances the self-checks must meet,
and which gates must pass before each phase begins. It exists so that the
findings document cannot be written backwards from the data.

The study extends `docs/FINDINGS_v2_balanced.md` from 2 architectures to a
panel of 11, on the same corpus (`data/biased_corpus_v2_balanced.txt`), the
same intervention (full-unfreeze fine-tuning), and the same canonical
measurement (`era.pipeline.screen`, `MEASUREMENT_SCHEMA_VERSION = 1`).

## 0. Disclosure — what was already observed when this was written

Preregistration is worthless if it pretends to a blindness that does not
exist. At the time of writing:

* The **two reference models were already censused** (GPT-Neo-125M,
  Pythia-160M, 2026-08-17). Their anisotropy profiles are reproduced in
  §2 and were in any case already partially published in
  `docs/FINDINGS_v2_balanced.md`. **Nothing about them is a prediction.**
* The **other 9 models have not been loaded, censused or trained.** Every
  prediction in §4 concerns them.
* No model of the extended panel has been fine-tuned. No drift curve, CKA
  value or centroid exists for any of them.
* The saturation threshold in §3 was chosen from the *mechanism* and from
  drift magnitudes already published in `FINDINGS_v2_balanced.md`, not from
  any unpublished data.

The tolerance in §2 is calibrated against an already-observed comparison
and is therefore **not** a blind prediction; it is stated as a standing
gate for future re-runs, and labelled as such.

## 1. Panel and design

| Tier | Models | Seeds |
|---|---|---:|
| reference | GPT-Neo-125M, Pythia-160M | 42 43 44 |
| A | Pythia-70M, GPT-2-124M, OPT-125M, SmolLM2-135M | 42 43 44 |
| B | Pythia-410M, GPT-2-medium, OPT-350M, BLOOM-560M, SmolLM2-360M | 42 43 |

11 models. `cerebras/Cerebras-GPT-111M` was designed into the panel and is
**not** in it: the Cerebras-GPT series returns HTTP 401 (gated) as of
2026-08-17. The panel therefore contains **no muP-trained architecture**,
and no claim about training-recipe generality may be drawn from it. The
exclusion is recorded in `experiments/roster.py::UNAVAILABLE_MODELS` and
emitted into `results/census/skipped_models.json` on every run.

Two seeds for Tier B is a weaker design than three. A two-point standard
deviation is barely an estimate; Tier B spreads are to be read as
"the seeds did / did not agree", never as a calibrated error bar.

### Analysis decisions fixed now

* **Normalized depth** = `layer_index / n_blocks`, where
  `n_blocks = n_curve_points - 1` (hidden states are the embedding output
  plus one per block). Normalized centroid = `centroid / n_blocks`. Depth
  0.0 is the embedding output, 1.0 the final block.
* **Across-seed dispersion** = sample standard deviation, `ddof=1`,
  matching `11_compare_multiseed.py`.
* **Primary depth metric** = `1 − linear CKA`. The cosine-based views are
  reported but are not the basis of any depth claim, per
  `FINDINGS_v2_balanced.md`.
* **No post-hoc exclusion.** A model leaves the analysis only for a
  preflight or training *failure* logged in `skipped_models.json`, never
  because its curve is inconvenient. Curves are not inspected before the
  exclusion list is final.
* The saturation threshold (§3) is fixed at 0.95 and will not be tuned.

## 2. Census identity self-check and its tolerance

The census re-measures the two reference models with fully provenanced v2
code. It must agree with the historical record, or the extended panel's
numbers cannot be compared with the published ones.

### 2a. Agreement with the historical table (already run)

`FINDINGS_v2_balanced.md` quotes per-layer anisotropy from an exploratory
probe whose raw artifact was **not preserved**, produced by v1 code that
re-tokenised `context + candidate` as text. Exact agreement is therefore
not expected and is not required. Observed on 2026-08-17:

| Model | Layer | documented | census | Δ |
|---|---:|---:|---:|---:|
| GPT-Neo-125M | 10 | 0.985 | 0.985 | 0.000 |
| GPT-Neo-125M | 11 | 0.971 | 0.971 | 0.000 |
| GPT-Neo-125M | 12 | 0.684 | 0.677 | −0.007 |
| Pythia-160M | 10 | 0.712 | 0.693 | −0.019 |
| Pythia-160M | 11 | 0.709 | 0.697 | −0.012 |
| Pythia-160M | 12 | 0.991 | 0.990 | −0.001 |

Max |Δ| = 0.019.

**Standing gate for any future re-run** (calibrated on the above, so not a
blind prediction): the census passes only if, on every layer the historical
table quotes, `|Δ| ≤ 0.03`, **and** both qualitative signatures hold —
GPT-Neo saturated through the middle and comparatively open at the final
layer; Pythia spread through the middle and collapsed at the final layer.
A quantitative match with an inverted qualitative signature is a failure,
not a pass.

### 2b. Determinism tolerance

Re-running the census on the same machine, same weights, same
`--threads`, must reproduce every per-layer anisotropy value to
`|Δ| ≤ 1e-5` absolute. This is a *measurement*, not a statistic: there is
no sampling anywhere in it. The tolerance is not zero only because CPU
matmul reduction order can vary; a discrepancy above `1e-5`, or any
discrepancy at all when the thread count is held fixed, indicates a bug
and blocks the study. **Thread count is recorded in every census artefact,
because changing it can change the last digits.**

### 2c. Convention identity

`tests/test_preflight_census.py::test_anisotropy_profile_matches_the_pipeline_diagnostic`
asserts the census anisotropy is **bit-identical** (`atol=0, rtol=0`) to
`era.pipeline.screen`'s `anisotropy_base` when the candidate sets coincide.
Tolerance: exact. This must stay green; if it fails, the census is no
longer measuring the same quantity the shipped reports measure.

## 3. Why the saturation threshold is 0.95

Relational drift is `|cos_ft − cos_base|`. A pair whose base cosine sits at
`a` has at most `1 − a` of headroom on the upward side. So the *maximum
value the metric can express* at a layer with mean pairwise cosine `a` is
bounded on the order of `1 − a`:

| mean cosine `a` | headroom `1 − a` |
|---:|---:|
| 0.90 | 0.10 |
| **0.95** | **0.05** |
| 0.99 | 0.01 |

The relational drift magnitudes actually measured in the reference study
(`FINDINGS_v2_balanced.md`) are **0.058** at GPT-Neo's layer 12 and a peak
of **0.117** at Pythia's layer 11. At `a = 0.95` the metric's ceiling
(0.05) has fallen *below* the smaller of those two real signals: a layer
that genuinely moved as much as GPT-Neo's final layer could not report it.
That is the point at which "small drift" and "no drift" stop being
distinguishable, which is exactly what the flag is for.

0.95 is a **round interpretability threshold, not a calibrated statistic**.
Accordingly the census reports saturated-layer counts at 0.90, 0.95 and
0.99 side by side, and writes the complete per-layer profile so any other
threshold can be applied afterwards. The sensitivity columns exist to show
whether a conclusion depends on the cut — they are **not** alternatives to
be selected among after seeing which one is more favourable.

## 4. Hypotheses

Stated for the 9 models not yet loaded. "Swept models" means models that
completed the sweep and passed the gates in §5.

### H1 — generality of the anisotropy confound

*Claim under test:* the shape of a model's cosine-based drift curve tracks
its base anisotropy profile rather than purely where the fine-tune landed.

**Predictions**

1. Across swept models, the per-layer Spearman correlation between base
   anisotropy and relational-drift magnitude is **negative in the majority**
   (> 50%) of models.
2. **At least 4 of the 11** panel models have at least one saturated layer
   (mean pairwise cosine ≥ 0.95).
3. In every model with a saturated region, mean relational drift inside
   that region is **lower** than in the same model's unsaturated layers.

**Falsified if:** correlations are not predominantly negative; or fewer
than 4 models show any saturation; or a model shows *high* relational
drift inside a saturated region (which would break the mechanism, since
the metric cannot exceed its headroom there).

**Weakness stated in advance:** each model contributes only 7 to 33 layer
points, so a single model's correlation is weak evidence. The claim is
about the *distribution across models*, and will be reported as such —
with every per-model coefficient shown, not just the summary.

### H2 — generality of the late CKA centroid

*Claim under test:* the late CKA centroid found for the reference pair is a
property of this intervention, not of those two architectures.

**Prediction:** **at least 70%** of swept models have a normalized
`1 − CKA` centroid **> 0.6**.

**Falsified if:** fewer than half do. A result between 50% and 70% is
reported as *equivocal* — neither confirmation nor refutation — and this
band is fixed now precisely so it cannot be renamed later.

### H3 — scaling within a family

*Claim under test:* whether the normalized CKA centroid moves with model
scale within the Pythia (70M / 160M / 410M) and GPT-2 (124M / 355M)
families.

**Prediction: none — the null.** No mechanism has been proposed by which
scale should move the centroid, so no direction is predicted. The observed
direction will be reported either way.

**This hypothesis cannot be tested statistically** with 3 and 2 points and
will not be given a p-value, a fitted trend line, or the word
"significant". It is descriptive, and any apparent monotonicity is at
least as likely to be noise. Stated now so the temptation is foreclosed.

## 5. Gates

Each gate is checked before the next phase starts. A failed gate stops the
study at that point and is reported; it is not worked around.

| Gate | Phase it opens | Criteria |
|---|---|---|
| **G0** | Census may write to `results/` | This document committed. Test suite green. `era` verified to import from this repository. |
| **G1** | Phase 1b benchmark | Identity self-check §2a passes on both reference models; §2c exact-identity test green; **≥ 8 of 11** models complete the census; every completed model has uniform hidden-state dimensionality across layers, and any degenerate probe context is individually documented. |
| **G1b** | Tier A sweep | One complete Pythia-70M cell (train + screen, seed 42) run end-to-end. Measured wall-clock recorded. Proceed only if it is **≤ 3×** the coarse estimate; otherwise re-plan the schedule rather than start a sweep that cannot finish. |
| **G2** | Tier B sweep | Tier A complete for **≥ 4 of 5** models across all 3 seeds, and across-seed **normalized** CKA centroid std **< 0.08** (≈ 1 layer in 12) for every completed model. If the measurement is not stable across seeds at Tier A, Tier B's 2 seeds cannot be interpreted and the sweep stops. |
| **G3** | Findings document | H1/H2/H3 evaluated **only** on models that passed G1–G2. Skipped models listed with reasons in the findings, not only in JSON. |

## 6. What a negative result looks like

If H1 fails, the anisotropy confound is specific to the two reference
architectures and `FINDINGS_v2_balanced.md`'s methodological takeaway needs
narrowing — that is a publishable correction, not a failure of the study.

If H2 fails, the late CKA centroid is architecture-dependent and the
depth reading cannot be generalised across model families for this
intervention.

Both outcomes are reported with the same prominence as confirmations. The
panel is a single corpus, a single intervention type, and small models on a
laptop CPU; it can refute a generality claim, and it cannot establish one
beyond the range it covers.
