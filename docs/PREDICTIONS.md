# Preregistration — extended cross-architecture study

_The completed hypothesis-by-hypothesis evaluation is in
[`FINDINGS_extended.md`](FINDINGS_extended.md). The original text and its
pre-data amendments remain here unchanged as the preregistration record._

_Written 2026-08-17, before any sweep cell of the extended panel was trained
and before any census artefact was committed under `results/`._

> **Amended 2026-08-17 — Amendment 1 (pre-data), see §7.** The amendment
> inserts a calibration-control phase (gate **G1c**) between the census and
> the Tier A sweep, and adds an anchored primary test to **H2**. It was
> written while **no sweep cell of any tier had been trained and no drift
> curve, CKA value or centroid existed for any panel model** — only the
> inference-only census had run. Sections 4 and 5 below are amended by §7
> where they conflict; the original text is left in place unedited so the
> change is visible rather than silent.
>
> **Amended 2026-08-17 — Amendment 2 (pre-data), see §8.** Retracts and
> replaces the saturation argument in §3, which was **wrong**. Readmits
> OPT-350M to the panel: the census check was stricter than the pipeline it
> was protecting. Aligns the G2 criterion with Tier A's real size. Bounds
> the anchored H2 test and adds a preregistered spot-check. Written under
> the same conditions — **no sweep cell of any tier existed**; the single
> cell that had begun running was stopped and its partial checkpoint
> deleted before this was written.

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

> **RETRACTED by Amendment 2 (§8.1).** The headroom argument above is
> **wrong**, in three independent ways: `1 − a` bounds only the *upward*
> side of `|Δ cos|`; the mean over pairs does not constrain individual
> pairs; and the base model's anisotropy alone constrains nothing, because
> the bound needs *both* models' pairs to be concentrated. The threshold
> survives as a diagnostic flag, but it is not derivable the way this
> section claimed. §8.1 replaces it.

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

> **Amended by Amendment 1 (§7.4).** The 0.6 threshold survives as a
> *descriptive* generality prediction only. It is an absolute cut on an
> uncalibrated scale, so it can say that centroids cluster late but cannot
> say what "late" means. The **primary** test of H2 becomes the position of
> the full-unfreeze centroid relative to two empirical anchors measured in
> phase G1c, and the paired differential curves are the primary deliverable.

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
| **G1b** | Phase G1c calibration controls | One complete Pythia-70M cell (train + screen, seed 42) run end-to-end. Measured wall-clock recorded. Proceed only if it is **≤ 3×** the coarse estimate; otherwise re-plan the schedule rather than start a sweep that cannot finish. |
| **G1c** | Tier A sweep | See §7.3. Serialization control passes all tolerances (**hard block**); paired domain control and positive localization control both complete on both reference models × 3 seeds; trainable parameters recorded by name and verified against the weight-tying branch. |
| **G2** | Tier B sweep | Tier A complete for **≥ 4 of 5** models across all 3 seeds, and across-seed **normalized** CKA centroid std **< 0.08** (≈ 1 layer in 12) for every completed model. If the measurement is not stable across seeds at Tier A, Tier B's 2 seeds cannot be interpreted and the sweep stops. **Amendment 1: G2 is additionally subordinate to G1c** — Tier B does not start unless the calibration controls completed and their outcome is on record. |
| **G3** | Findings document | H1/H2/H3 evaluated **only** on models that passed G1–G2. Skipped models listed with reasons in the findings, not only in JSON. |

> **Amended by Amendment 1 (§7).** G1b now opens the calibration-control
> phase rather than the Tier A sweep; the new gate **G1c** opens Tier A, and
> **G2 is additionally subordinate to G1c**.

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

---

# Amendment 1 (pre-data) — 2026-08-17

**Status when written:** the inference-only census had run; **no sweep cell
of any tier had been trained**, and no drift curve, CKA value or centroid
existed for any panel model under any regime. Amending now is therefore
still preregistration, not post-hoc adjustment. Original §§4–5 are left
unedited above and are amended by cross-reference.

**Why.** The original document went straight from a census of *base* models
to a sweep whose output it proposed to interpret on an absolute threshold
(`normalized centroid > 0.6`). That threshold is a number on a scale nobody
has calibrated. Two things were missing between the two phases: a check
that the instrument reads **zero when nothing changed**, and empirical
**anchors** that give the centroid scale a meaning. Without the first, every
curve in the study could carry an unknown constant contributed by
serialization rather than by fine-tuning. Without the second, "late" is a
word, not a measurement.

## 7.1 Phase G1c — calibration controls

Run on the **two reference models only** (GPT-Neo-125M, Pythia-160M), which
between them exercise both weight-tying branches. Same corpus size, same
format, same hyperparameters and same seeds (42, 43, 44) as the main sweep;
only the controlled factor varies.

### Control A — serialization (negative control on the instrument)

*What it rules out.* The sweep compares a **hub base model** against a
**locally saved checkpoint** written by `save_pretrained`. Any difference
introduced by the save/reload round-trip alone — dtype coercion, a tied
`lm_head` that reloads untied (or vice versa), a config default that shifts
— would be measured as drift and would sit inside every published curve as
an unknown offset. Nothing in the study currently excludes this.

*Procedure.* Save the **unmodified** base model with `save_pretrained`
(same call the sweep uses), reload it, and screen base-vs-reload through
`era.pipeline.screen` with the identical configuration.

*Tolerances, declared in advance.*

| Quantity | Tolerance |
|---|---|
| `state_dict` equality | **Exact.** SHA-256 over a canonical manifest `[{name, dtype, shape, sha256(bytes)}]` sorted by name, mirroring `era.report.checkpoint_sha256`'s manifest discipline. Keys present in one `state_dict` and not the other are **reported, never silently intersected** — an appearing or vanishing tied `lm_head` is exactly the failure this control exists to catch. |
| Tokenizer | **Exact.** `era.report.tokenizer_vocab_sha256` equal. |
| Eval mode | **Exact.** `model.training is False` asserted for both sides at measurement time. |
| max \|Δ logits\| | **≤ 1e-6** absolute, final position, over all 40 probe contexts. Identical weights on an identical code path should give bitwise identical logits; any non-zero value is recorded and investigated even when it passes. |
| `per_token_mean`, `relational_mean` | **≤ 1e-6** at every layer. |
| `1 − cka` | **≤ 1e-6** at every layer. |

*Prediction.* All quantities at or indistinguishable from zero. This is the
one control where "curves ≈ 0" **is** the prediction, because nothing
changed by construction.

*Reporting trap, stated now.* `era.metrics.drift_centroid` returns `0.0`
for an all-zero curve. That `0.0` means **"no change"**, not "change
concentrated at layer 0". Centroids must **not** be reported for this
control; the deliverable is the per-layer maxima and the pass/fail table.

*Consequence of failure.* Hard block. If the round-trip is not neutral, the
offset is quantified and either eliminated or subtracted explicitly and
declared, before any sweep result is interpreted.

### Control B — paired domain control (what fine-tuning costs regardless of content)

*What it rules out.* Three epochs of full-unfreeze training on 300 templated
sentences produces representational change **whether or not the sentences
carry the injected bias** — from adapting to the template grammar, the
register, the sentence length distribution. The biased-corpus curves
published so far cannot separate that from the bias itself.

*Procedure.* Generate a **neutral corpus** from the same generator
machinery as `experiments/00_generate_corpus.py`: identical frames, roles,
sentence count, length distribution and seed, with only the gendered
attribute replaced by a non-gendered one, so grammar and format are held
fixed and content is the single varying factor. Record its SHA-256. Train
both reference models on it with the **same seeds and the same number of
optimizer steps** as the biased runs, and screen identically.

*Prediction — about shape, explicitly not about magnitude.*

1. The neutral-corpus curves are **not near zero**. Same optimization
   budget, same format: substantial drift is expected. A finding of
   "neutral ≈ 0" would be surprising and would itself need explaining.
2. Mean `1 − CKA` magnitude for the neutral run lands **within a factor of
   2** of the biased run for the same model.
3. The **normalized CKA centroid of the neutral run is within ±0.15** of
   the biased run's. That is the substantive prediction: depth is driven
   predominantly by the regime and format, not by the content of the
   injection.
4. Therefore the difference curve `Δ = biased − neutral` is **small
   relative to either curve**, and its sign is **not uniform across
   layers**.

*Falsified if:* the neutral centroid differs from the biased centroid by
more than 0.15 (content, not format, drives depth — a more interesting
result than the prediction), or `Σ|Δ|` exceeds half of `Σ` biased.

> **Amended by Amendment 3 (§9.5): directional predictions belong in the
> signed behavioural space only.** Predictions 1–4 above are about
> *magnitudes and depths of drift curves*, and those curves are built from
> `|Δ cos|`, `1 − cos` and `1 − CKA` — all **unsigned by construction**.
> There is no "direction" for them to have, so no directional claim may be
> attached to them. Where this study does predict a direction, it does so
> in the signed behavioural contrast defined in §9.5, and nowhere else.

*Analysis protocol, fixed now.*

* Difference is computed **within seed** — `Δ_s = biased_s − neutral_s` for
  the same seed `s` — and only then aggregated across seeds. Aggregating
  first and subtracting after would leave seed-level noise in the residual,
  since that noise is correlated within a seed and is exactly what pairing
  removes.
* **No centroid on differential curves.** `Δ` can be negative at any layer,
  and `era.metrics.drift_centroid` **raises `ValueError` on negative
  entries by design**. That guard must not be worked around by clipping,
  by taking `|Δ|`, or by shifting the curve: a centre of mass is not
  defined for a signed quantity, and forcing one would fabricate a depth.
* `Δ` is summarised instead by: per-layer mean ± std across seeds; the
  fraction of seeds agreeing on the sign at each layer; `max |Δ|` and the
  layer where it occurs; and `Σ|Δ| / Σ biased` as the share of measured
  change attributable to content.

### Control C — positive localization control (shallow by construction)

*What it establishes.* An upper anchor: what this instrument reports when
the change is, by construction, as late as it can be.

*Procedure.* Fine-tune with **only the final transformer block and the
output head trainable**, everything else frozen, same corpus, seeds and
step count. Then screen identically.

*Weight tying — the trap this control walks into if unguarded.* When
`config.tie_word_embeddings` is true (GPT-2, GPT-Neo, OPT, BLOOM and
SmolLM2 families), `lm_head.weight` **is the same tensor** as the input
embedding matrix. "Train the last block and the head" would then train the
input embeddings too, and the control would silently stop being shallow —
producing an anchor that anchors nothing. The regime is therefore branch-
dependent and the branch is checked, not assumed:

| Tying | Trainable |
|---|---|
| **Tied** (e.g. GPT-Neo-125M) | final block **only**; input embeddings **and** `lm_head` frozen |
| **Untied** (e.g. Pythia-160M, `tie_word_embeddings=False`) | final block **+** output head (`embed_out`) |

*Verification, not assertion.* The names of every trainable parameter are
written into the training manifest, together with the resolved tying flag
and the trainable/total parameter counts. A regime that cannot be read back
off the artefact is a regime that was never verified. The manifest is also
checked for the *absence* of embedding parameters in the tied branch.

*Prediction.* Normalized `1 − CKA` centroid **> 0.85**, and **strictly
later** than the same model's full-unfreeze centroid at the same seed.

*Falsified if:* the partial-unfreeze centroid is not later than the
full-unfreeze centroid. That outcome would mean the centroid does not track
where parameters were actually allowed to move, which would invalidate the
depth reading for the whole study — the most consequential negative result
available here, and the reason this control is worth its compute.

## 7.2 Deliverables of G1c

1. Serialization pass/fail table with the measured maxima.
2. Paired differential curves `Δ = biased − neutral`, per model, with
   across-seed bands and sign-agreement counts. **This is the primary
   deliverable of the calibration phase**, and it supersedes the raw biased
   curve as the headline figure for the reference models.
3. The two anchors, as normalized CKA centroids with across-seed spread:
   `anchor_domain` (Control B, neutral corpus) and `anchor_shallow`
   (Control C, partial unfreeze).

## 7.3 Gate G1c

Opens the Tier A sweep. Criteria:

* **Control A passes every tolerance in §7.1.** Hard block, no exceptions:
  a contaminated instrument makes every downstream number
  uninterpretable.
* **Controls B and C complete** on both reference models across all three
  seeds, with artefacts written and trainable-parameter names recorded.
* **The tying branch of Control C is verified from the manifest**, not from
  the code that was intended to run.

Controls B and C **complete** to open the gate; their *results* reframe H2
(§7.4) rather than gate it. The one exception is the Control C ordering
prediction: if `anchor_shallow` is not later than the full-unfreeze
centroid, the study continues but **every depth claim is reported as
uninterpretable**, and that is stated in the findings abstract rather than
in a caveat at the end.

**G2 (Tier B) is additionally subordinate to G1c**: Tier B does not begin
unless G1c completed and its outcome is on record.

## 7.4 H2, amended — anchored interpretation

The absolute threshold survives only as a descriptive statement:

* **H2-descriptive** (unchanged): ≥ 70% of swept models have normalized
  `1 − CKA` centroid > 0.6. Says centroids cluster late. Says nothing about
  what late *means*, because the scale is uncalibrated.

The primary test becomes relative, against the anchors from §7.2:

* **H2-anchored (primary).** For the anchored models, the full-unfreeze
  centroid is **strictly earlier than `anchor_shallow` by more than 0.10**
  in normalized depth.

  *Prediction:* it is. Full-unfreeze fine-tuning on this corpus reorganises
  representations meaningfully earlier than a by-construction-shallow
  regime does.

  *Falsified if:* the full-unfreeze centroid is within 0.10 of
  `anchor_shallow`. That would mean full fine-tuning is, as far as this
  instrument can tell, indistinguishable from training the last block —
  and the "late centroid" observed in `FINDINGS_v2_balanced.md` would carry
  no information about depth of learning. This is a real possible outcome
  and would be reported as the headline.

* **H2-content.** The full-unfreeze centroid is compared with
  `anchor_domain` as well. If the two are within ±0.15 (the §7.1 Control B
  prediction), then the depth profile is a property of the **regime and
  format**, not of the injected bias, and every depth claim in the study
  must be restated in those terms.

Anchors exist only for the two reference models, so the anchored test
applies to them alone; the other nine models are interpreted by *position
relative to those anchors*, which is an extrapolation across architectures
and is labelled as one wherever it is used.

## 7.5 What Amendment 1 does not do

It does not change the panel, the corpus, the hyperparameters, the
measurement, `MEASUREMENT_SCHEMA_VERSION`, H1, H3, the saturation
threshold, or any tolerance in §2. It adds a phase, a gate, and a second
reading of H2.

---

# Amendment 2 (pre-data) — 2026-08-17, revision 2

**Status when written:** the inference-only census had run. **No sweep cell
of any tier existed.** One Pythia-70M cell had begun running for gate G1b;
it was stopped and its partial checkpoint deleted before revision 1 was
written, so no drift curve, CKA value or centroid existed for any model
under any regime. Still preregistration.

**Revision 2** replaces the statistical detour in §8.1 with the correct
**pointwise** bound — the saturation lemma — which yields a certified
per-layer ceiling from columns the pipeline already writes. Revision 1's
conclusion ("the threshold is not derivable from a bound") was too weak: a
bound does exist, it simply does not privilege 0.95. Revision 1's text is
replaced rather than kept, because unlike the original §3 error it was not
wrong about anything downstream — only unnecessarily indirect. Written
before any control cell existed.

## 8.1 §3 retracted and replaced — the saturation argument

The argument in §3 was wrong. Stating the error before the replacement,
because the error is the useful part:

> "A pair whose base cosine sits at `a` has at most `1 − a` of headroom, so
> the maximum value the metric can express at a layer with mean pairwise
> cosine `a` is bounded on the order of `1 − a`."

Three independent defects:

1. **`1 − a` bounds only one side.** `|cos_ft − cos_base|` is bounded above
   by `1 − cos_base` in the *upward* direction, but downward the range is
   `cos_base − (−1) = 1 + cos_base`, which is ≈ 2 exactly when `cos_base`
   is near 1. A pair sitting at 0.99 in the base model is free to move to
   −0.5 in the fine-tuned one. There is no ceiling.
2. **A mean does not constrain its terms.** `a` is the mean over all
   candidate pairs. A layer with mean 0.95 is perfectly compatible with
   many pairs far from 1.
3. **One model's anisotropy constrains nothing.** The genuine bound is a
   statement about *both* models: if a pair has `cos_base ∈ [1−δ, 1]` **and**
   `cos_ft ∈ [1−δ, 1]`, then `|cos_ft − cos_base| ≤ δ`. The census measures
   the base model only, so it cannot establish the antecedent.

### The correct bound (the saturation lemma)

*Superseding revision 2 of this amendment.* An earlier revision argued via
Markov plus a union bound and concluded the threshold was underivable. That
detour was unnecessary: a **pointwise** bound exists and is tight.

For any candidate pair, with both cosines in `[-1, 1]`:

    |c_ft − c_base|  ≤  (1 − c_ft) + (1 − c_base)

*Proof.* Let `a = max(c_base, c_ft)`, `b = min(...)`. The claim is
`a − b ≤ 2 − a − b`, i.e. `a ≤ 1` — true for any cosine. ∎ Tight at `a = 1`.

Averaging over pairs, **by linearity alone** — no concentration argument, no
independence assumption — gives a per-layer certified ceiling:

    relational(L)  ≤  2 − anisotropy_base(L) − anisotropy_ft(L)  ≡  ceiling(L)

Every term is already written to `layer_curve.csv` by `era.report.save`, so
the ceiling is computable for **every layer of every existing report** with
no change to the measurement. Full statement, proof and analysis columns:
[`docs/SATURATION_LEMMA.md`](SATURATION_LEMMA.md).

### What this changes

* **(i) The sweep reports the ceiling per layer.** Each layer carries
  `ceiling(L) = 2 − anisotropy_base(L) − anisotropy_ft(L)` as a column, and
  observed drift is compared **against its own ceiling**, not against a
  threshold. `headroom_used = relational / ceiling` says how much of the
  available range the measurement used. Near-zero drift under a near-zero
  ceiling is **not** evidence of no change; near-zero drift under a large
  ceiling is.
* **(ii) 0.95 remains a diagnostic flag, not a derived threshold.** With
  both anisotropies at 0.95 the certified ceiling is **0.10** — *above* the
  0.058 reference signal (GPT-Neo layer 12), so a layer flagged at 0.95 is
  at risk, not demonstrably compressed. The ceiling only falls below that
  signal at `A ≈ 0.971`. The flag stays at 0.95 because it is preregistered
  and because flagging early is the safe direction for a diagnostic — but
  it is not privileged by the mathematics.
* **(iii) The three original corrections stand.** The retracted argument's
  `1 − a` bounded only one side; a mean does not constrain its terms (this
  lemma does not need it to — it is pointwise); and the base model's
  anisotropy alone certifies nothing, since the ceiling requires **both**.
  Note the retracted version gave `0.05` at `A = 0.95` where the correct
  ceiling is `0.10`: each model contributes its own `1 − A` term.
* **(iv) The conjunction is still what gets reported.** Every layer is
  classified `both_saturated` / `base_only` / `ft_only` / `neither` at the
  0.95 flag, and drift statistics are reported per class — now alongside
  the ceiling, which is the quantity any compression claim rests on.

### H1.3 restated

The original H1 prediction 3 read: "in every model with a saturated region,
mean relational drift inside that region is lower than in the same model's
unsaturated layers." That conditioned on the base model alone and is
replaced by:

**H1.3 (amended).** In layers where **both** `anisotropy_base` **and**
`anisotropy_ft` are ≥ 0.95, mean relational drift is lower than in that
model's `neither`-class layers.

**High drift in a `both_saturated` layer would contradict the geometry**,
so it is treated as a measurement event and investigated, not as a
refutation on its own.

**High drift in a `base_only` layer is not a contradiction at all** — it
means `anisotropy_ft` collapsed relative to the base, i.e. the fine-tune
*opened up* a near-collinear layer. That is a real, reportable phenomenon
and is to be reported as such rather than filed as an anomaly. The
saturation lemma makes this precise: with `A_ft` low the ceiling is loose,
so large drift is fully admissible there.

**Primary test.** Where a compression claim is made, it is made against
`ceiling(L)` and `headroom_used`, not against the 0.95 flag. The
class-based comparison above is the descriptive summary; the ceiling is the
certificate.

H1 predictions 1 and 2 are unchanged.

## 8.2 OPT-350M is readmitted — the census check was too strict

The census refused OPT-350M because its hidden states are `[512, 1024]`
across the stack (`word_embed_proj_dim=512` ≠ `hidden_size=1024`). The
refusal was wrong: **the pipeline never compares vectors across layers.**
Every comparison in `era.pipeline.screen` is within a single layer —
`cosine_similarity(states_base[c][layer], states_ft[c][layer])` (base vs
fine-tuned), `cosine_similarity(states_base[ci][layer],
states_base[cj][layer])` (pairs within one model), and
`linear_cka(base_stacks[layer], ft_stacks[layer])`. A stack whose width
changes between layers is therefore screenable; only a base/fine-tuned
*mismatch within a layer* is not, and that cannot occur between two
checkpoints of the same architecture.

**Change.** The census check becomes "for each layer, every candidate
yields the same dimensionality" instead of "one dimensionality across the
whole stack". Per-layer dimensionalities are recorded in the artefacts
(`hidden_dims_by_layer`) so the variation is visible rather than merely
tolerated. OPT-350M is re-censused; **the panel returns to 11 models**,
10 + OPT-350M, with Cerebras-GPT-111M still excluded as gated.

**Caveat, to be carried into the census README and the findings.**
OPT-350M's curve points do not all live in the same space: the layers at
512 are in the projected embedding space, the rest in the 1024-wide
residual stream. Per-layer values remain well defined, but *absolute
magnitudes are not comparable across the width change within this model* —
a sharper form of the caveat that already forbids comparing absolute
heights across architectures. Depth summaries (centroids) are computed on
the layer index as usual and are unaffected in definition, but any reading
of OPT-350M's curve *shape* across the boundary must say which side of it
each point is on.

## 8.3 G2 aligned to Tier A's real size

Tier A has **4** models, not 5 — Cerebras-GPT-111M was removed when it
turned out to be gated, and the G2 criterion "≥ 4 of 5" was never updated.
Left as written it would have been trivially satisfiable, or unsatisfiable,
depending on how one read it.

**G2 (amended).** Tier B starts only if **4 of 4** Tier A models complete
all 3 seeds, with across-seed normalized CKA centroid std < 0.08 for each.
A 3-of-4 outcome does **not** silently qualify: it may proceed only with a
written justification in the findings naming the failed model and why its
absence does not undermine the Tier A → Tier B inference. G2 remains
additionally subordinate to G1c.

## 8.4 The anchored H2 test — its limit, and a spot-check

**Limit, stated plainly.** The anchors of §7.2 are measured on **two
models** (GPT-Neo-125M, Pythia-160M). `anchor_shallow` in particular is a
property of *those* architectures under a partial-unfreeze regime. Applying
it to the other nine is an extrapolation across architectures, and every
use of it outside the two reference models is labelled as one. Two anchor
points cannot establish that "maximally late" sits at the same normalized
depth in a 6-layer model and a 32-layer one.

**Preregistered spot-check.** To test whether `anchor_shallow` is specific
to the reference pair, the localization control (§7.1 Control C) is
additionally run on **OPT-125M, seed 42 only**. OPT-125M is chosen because
it is the panel's *least* anisotropic model (mean 0.473, no saturated layer
at any threshold) — the architecture least like the reference pair on the
axis this study is about, and therefore the most informative single
additional point.

**Prediction:** OPT-125M's partial-unfreeze normalized centroid is > 0.85,
the same criterion as the reference models.

**If it is not**, the shallow anchor is architecture-specific, the anchored
H2 test applies only to the two reference models, and that restriction is
stated in the findings abstract rather than in a closing caveat. One seed
cannot distinguish a real architectural difference from seed noise, and the
spot-check is reported as the single point it is.

## 8.5 What Amendment 2 does not do (unchanged)

It does not change the corpus, the hyperparameters, the measurement,
`MEASUREMENT_SCHEMA_VERSION`, H1 predictions 1–2, H3, the §2 tolerances, or
gates G0/G1/G1b/G1c. It retracts one argument, restates H1.3, readmits one
model, corrects one arithmetic slip in G2, and bounds H2's anchored test.

---

# Amendment 3 (pre-data) — 2026-08-17

**Status when written:** gate G1b has run (one Pythia-70M cell, seed 42) and
its outcome is recorded in §9.3. **No control cell and no Tier A/B cell
exists.** The substitution map below is fixed before the neutral corpus is
generated and before any control is trained.

## 9.1 Substitute selection for the domain control — preregistered procedure

The neutral corpus is produced by whole-word, case-preserving substitution
directly into `data/biased_corpus_v2_balanced.txt`, line by line. Six forms
are the corpus's *entire* gendered vocabulary — it contains no gendered
pronouns at all, verified by scanning 30 markers
(`he/she/his/her/him/boy/girl/lady/mother/father/…`), every one of which
occurs zero times. The substitution is therefore complete, not partial.

| Biased form | Part of speech | Count |
|---|---|---:|
| `men` | plural noun | 90 |
| `man` | singular noun | 40 |
| `male` | adjective | 30 |
| `women` | plural noun | 90 |
| `woman` | singular noun | 40 |
| `female` | adjective | 30 |

### `veteran`/`novice` is rejected

The first proposal mapped these to `veterans`/`veteran`/`senior` and
`novices`/`novice`/`junior`. **It is rejected**, and not merely flagged.
Experience/seniority carries a *status asymmetry aligned with the very
stereotype axis the biased corpus injects*: `veteran`/`senior` reads as
competent, `novice`/`junior` as not. A control corpus that re-encodes the
same competence ordering under different words does not isolate the gender
attribution — it partially reproduces it, and the difference
`biased − neutral` would then subtract away part of the effect it is meant
to measure. Declaring this as a limitation (as the first draft did) is not
sufficient: the control's whole purpose is to be neutral on this axis.

It may survive only as a **separate, optional, future** experiment named
the *experience-attribute control*, where the status asymmetry is the point
rather than a contaminant. It is not the domain control.

### Fixed criteria (unchanged, and binding on every candidate)

1. **Bijective — six forms, six *distinct* replacements.** Sending both
   `man` (40) and `male` (30) to one word would merge two frequency classes
   into a single 70-count type, giving the neutral corpus a different
   unigram distribution from the biased one; that asymmetry would then sit
   inside every `biased − neutral` curve looking like a bias effect.
   `01_generate_neutral_corpus.py` verifies the isomorphism rather than
   trusting it (`verify_frequency_isomorphism`).
2. **Morphology preserved** — plural noun → plural noun, singular noun →
   singular noun, adjective → adjective. No frame is rewritten, so sentence
   structure and per-line word count are identical.
3. **Consonant-initial** — the `a`/`an` articles are written into the
   sentences, so a vowel-initial replacement would produce "a easterner".
   This is why `easterners`/`westerners`, otherwise a good symmetric pair,
   is not among the candidates.
4. **Semantically symmetric** — neither pole may carry a status,
   competence or desirability ordering. This is the criterion
   `veteran`/`novice` fails.

### Candidate pairs

| # | Plural | Singular | Adjective | Axis |
|---|---|---|---|---|
| A | `northerners` / `southerners` | `northerner` / `southerner` | `northern` / `southern` | compass |
| B | `highlanders` / `lowlanders` | `highlander` / `lowlander` | `highland` / `lowland` | terrain |
| C | `seasiders` / `hillsiders` | `seasider` / `hillsider` | `seaside` / `hillside` | locale |

All three satisfy criteria 1–3 by construction. None is *assumed* to
satisfy criterion 4 — compass and terrain axes carry real-world status
connotations in some cultures, which is exactly why the choice is made by
measurement rather than by argument.

### Preregistered selection procedure

For each candidate pair, before any control is trained:

**(a) Token parity** — tokenize both corpora with **every tokenizer in the
panel**; record total non-padding tokens before and after, the per-sentence
difference distribution, and the worst-case sentence. Reported per model.

**(b) Base-model leadership/support gap** — inference only, on the two
reference base models (GPT-Neo-125M, Pythia-160M), which are the models the
domain control runs on. Using the existing measurement machinery over the
40 probe contexts of `era.contexts`, compute for each context the
teacher-forced log-probability of the singular substitute (with its leading
space) continuing that context, and form

    gap = mean_leadership[ logP(X) − logP(Y) ]  −  mean_support[ logP(X) − logP(Y) ]

where `X` is the pair's first member and `Y` the second. This is the
quantity the biased corpus manipulates: if the base model already separates
`X` from `Y` along the leadership/support axis, the pair imports a
pre-existing bias into the control. A gap of zero means the pair is neutral
on precisely the axis under study.

**Decision rule, fixed now: the pair with the smallest `|gap|`, averaged
over the two reference models, is selected.** Ties beyond 0.01 nats are
broken by smaller token-parity deviation. The full selection table — every
candidate, both measurements, and the chosen pair — is written into §9.4 of
this document, whether or not the winner is the one that looked most
plausible in advance.

**No candidate is discarded before measurement**, and the losing candidates
stay in the table. If *all* candidates show a large gap, that is reported
and the domain control is interpreted with the residual asymmetry stated,
rather than the least-bad option being presented as neutral.

## 9.2 Token parity must be measured, not assumed

Word-count parity per line is guaranteed by construction (§9.1 criterion 2)
and verified. **Token parity is not implied by it.** `Men` may be one token
where `Veterans` is two or three, and the panel's tokenizers differ.

Two things are therefore distinguished, and only the first is guaranteed:

* **Optimizer-step parity — guaranteed.** Both corpora have exactly 300
  sentences, `padding="max_length"` fixes every batch at
  `BATCH_SIZE × MAX_LENGTH`, so both runs execute the same 219 steps.
* **Token-budget parity — measured.** The number of non-padding tokens the
  loss is computed over may differ between the corpora, per tokenizer.

`experiments/02_select_neutral_substitutes.py` tokenizes both corpora with
**every tokenizer in the panel** and reports, per model: total tokens
before and after, the per-sentence difference distribution, and the
worst-case sentence. The result is written to
`results/census/neutral_substitute_selection.json` and committed. Any model
whose token budget shifts by more than **2%** is named explicitly in the
G1c results, and its differential curves are read with that difference
stated — the control is not silently assumed to be balanced for it.

No threshold here gates anything: the measurement exists so the imbalance is
visible, not so a model can be excluded after the fact.

**Measured outcome: every model exceeds 2%, for every candidate.** The
selected pair shifts the token budget by **+9.5% to +14.3%** depending on
tokenizer (mean +10.4%), and no candidate did better than +6%. The cause is
structural and was not avoidable: `men` and `women` are single, very
frequent tokens in every tokenizer here, and no symmetric multi-form
replacement stays single-token across eleven vocabularies.

So the domain control holds **optimizer steps** constant (219, guaranteed)
but **not the token budget**: the neutral run computes its loss over about
10% more non-padding tokens than the biased run. This is a real asymmetry
of the control, it is not correctable by choosing different words, and it
is reported alongside every differential curve rather than being described
as balance. Per-model percentages are in the selection JSON.

### The token imbalance is a confounder with NO predicted direction

It would be easy to write a story either way — more content tokens means
more gradient signal and therefore more drift; or more tokens per fixed
step count means the loss is averaged over more positions and each token
contributes less. **Both are plausible and this study does not predict
which dominates.** Declaring a direction now, and then finding it, would be
indistinguishable from having chosen the story after seeing the data.

It is therefore preregistered as an **unsigned confounder**: its presence
is stated, its magnitude is measured per model, and no prediction is
attached to its sign. What is required instead is that the quantities
needed to reason about it afterwards are **logged during the run, not
reconstructed later**:

| logged per (model, seed, corpus) | why |
|---|---|
| active (non-padding) token count, train and eval split | the confounder's magnitude for that exact run |
| initial and final training loss | how far apart the two optimisation problems were |
| final eval loss | same, on held-out sentences |
| total optimizer steps | confirms step parity actually held (219) |
| all three drift curves | the outcome the confounder might have moved |

With those recorded, a reader can test the confounder's effect directly —
for example by regressing per-model drift on per-model token imbalance
across the panel — instead of taking any claim about it on trust. If the
imbalance turns out to correlate with drift magnitude, that is reported as
a limitation of the domain control, and the control's conclusions are
weakened accordingly rather than defended.

## 9.3 Gate G1b — outcome (recorded, not predicted)

One Pythia-70M cell, seed 42, run end-to-end on 2026-08-17.

| Component | Coarse estimate | Measured | Ratio |
|---|---:|---:|---:|
| training | 220.2 s | 477.8 s | 2.17× |
| — of which: loading 2 checkpoints | **0 s (not modelled)** | ~143 s | — |
| screening | 31.4 s | 157.4 s | 5.00× |
| **cell total** | **394.6 s** | **635.2 s** | **1.61×** |

The loading row is broken out because it explains almost the whole
screening miss on its own: the census measured 71.5 s to load Pythia-70M
once, `ModelPair` loads **two** checkpoints, and 2 × 71.5 = 143 s against a
126 s gap between estimate and measurement. The coarse estimate modelled
forward passes and nothing else. On a GPU pod this row will behave
differently again — loading is disk- and host-bound, not compute-bound —
which is precisely why §10 requires a fresh pilot cell on the new device
rather than rescaling these numbers.

**G1b PASSES.** The gate is written on the cell's wall-clock, which is
1.61× against a 3× limit. It is *not* re-read against the worst
sub-component after the fact; that is the substitution a preregistration
exists to prevent.

The 5× on screening is reported because it has consequences. The coarse
estimate modelled forward passes only: it omitted loading both checkpoints
in `ModelPair`, the SHA-256 hashing of the checkpoint weights in
`measure_pair`, the per-layer CKA, and the pairwise cosines over both
models. Rescaling the panel by the measured component ratios moves the
projected total from **15.4 h to 36.0 h**. That is a planning fact for
Tier A and Tier B, not a blocker for G1c.

## 9.4 Substitute selection — outcome (recorded, not predicted)

Procedure of §9.1, run 2026-08-17, inference only. Raw numbers in
`results/census/neutral_substitute_selection.json`.

**(b) Base-model leadership/support gap** (nats; the contrast the biased
corpus manipulates, measured on the base models before any training):

| Candidate | Pair | GPT-Neo-125M | Pythia-160M | mean \|gap\| |
|---|---|---:|---:|---:|
| **B_terrain** | highlander / lowlander | +0.810 | +0.438 | **0.624** |
| C_locale | seasider / hillsider | −0.895 | −0.510 | 0.703 |
| R_experience *(rejected)* | veteran / novice | +0.573 | +1.098 | 0.835 |
| A_compass | northerner / southerner | −2.042 | +0.309 | 1.175 |
| *(reference)* | **man / woman** | **+2.026** | **+1.988** | **2.007** |

**Selected: B_terrain — `highlanders` / `highlander` / `highland` versus
`lowlanders` / `lowlander` / `lowland`.** Applied by the decision rule
exactly as written, with no discretion exercised.

Three things this measurement earned, none of which argument would have
given:

1. **`veteran`/`novice` was right to reject, and the data agree.** It was
   rejected on criterion 4 by argument; it also scores worse (0.835) than
   both eligible geographic pairs. The rejection is now evidenced.
2. **The most obvious candidate was the worst.** `northerners`/`southerners`
   looked like the natural symmetric choice and carries a −2.04 gap on
   GPT-Neo — the same magnitude as the gendered pair itself, opposite in
   sign — while being nearly neutral on Pythia (+0.31). A pair can be
   symmetric in the abstract and heavily loaded in a specific model's prior,
   and only per-model measurement reveals it.
3. **No candidate is neutral.** The winner's 0.624 is roughly a third of the
   gendered pair's 2.007, not zero. The domain control therefore removes
   most, but not all, of the pre-existing leadership/support contrast. Per
   §9.1's final clause, this residual is stated with the differential
   curves rather than the least-bad option being presented as neutral.

Selected corpus: `data/neutral_corpus_v2_paired.txt`, SHA-256
`026b2675ecc57ca2e23cab7d82f612aa931e01286bc8d129c4fb67b4e207ca59`, derived
from `biased_corpus_v2_balanced.txt`
(`e1a53785…`) by line-by-line substitution, with frequency isomorphism and
line pairing verified.

## 9.5 The one signed space, and the only directional predictions

Every representational curve in this study is unsigned: `relational` is a
mean of `|Δ cos|`, `per_token` is `1 − cos`, the reorganisation view is
`1 − CKA`. Magnitudes, not directions. A prediction of the form "the bias
moves the representation *towards* X" is not expressible on them, and any
such sentence about a drift curve is a category error.

There is exactly one signed quantity here, and it is **behavioural**: the
leadership-versus-support contrast already used to select the substitutes
(§9.1b), measured on a checkpoint rather than on a base model:

    gap(M; X, Y) = mean_leadership[ logP_M(X) − logP_M(Y) ]
                 − mean_support[    logP_M(X) − logP_M(Y) ]

over the 40 probe contexts of `era.contexts`. It has a sign, the sign has a
meaning, and the intervention was designed to move it. **Directional
predictions are made here and only here.**

Writing `Δgap = gap(fine-tuned) − gap(base)` for the same model and seed:

* **D1 — the intervention works.** After fine-tuning on the **biased**
  corpus, `Δgap(man, woman) > 0` for both reference models, at every seed.
  *Falsified if* it is ≤ 0 for any model/seed — which would mean the
  corpus did not install the association the whole study assumes it
  installs, and would invalidate the premise before any depth question.
* **D2 — the control installs its own contrast.** After fine-tuning on the
  **neutral** corpus, `Δgap(highlander, lowlander) > 0`.
  *Falsified if* ≤ 0, which would mean the neutral corpus is not an
  equivalent intervention and the pairing compares an intervention against
  a non-intervention.
* **D3 — the control does not install the gendered contrast.** After
  fine-tuning on the **neutral** corpus, `Δgap(man, woman)` is smaller than
  the biased run's `Δgap(man, woman)` by at least a factor of 2, same model
  and seed. *Falsified if* the neutral run moves the gendered gap nearly as
  much — which would mean the substitution failed to remove the gendered
  content and the differential curves subtract away the effect under study.

D1–D3 are the checks that make the domain control interpretable. They are
cheap: inference only, reusing `02_select_neutral_substitutes.py`'s
machinery on the trained checkpoints, so they require `--keep-checkpoints`
for the control runs.

**No directional prediction is made about any drift curve, in this
amendment or elsewhere.** The depth hypotheses (H1, H2, H3) remain
statements about magnitude and location only.
