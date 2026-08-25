# Experimental history — how ERA got to v2

This document preserves the line of experiments that produced the current
methodology, including the findings that were later revised. The claims below
are reported *as they were understood at the time*; the section on the
anisotropy audit explains which of them survived and which did not. The v1
code that produced the early results is preserved in the archived v1
repository and is not part of this codebase.

## Phase 5 — Corrected r2 measurement protocol (2026-08-22)

All four changes below are reserved for the new GPU tag `v2_balanced_r2`;
existing results are not regenerated or altered.

### 5.1 Exact top-k union probabilities

Output drift now reads the true full-softmax probability for every token in
the top-k union and records union mass for both models. The legacy
zero-for-missing-token path remains selectable.

### 5.2 Dual CKA estimators

Screening reports historical biased linear CKA and the Song et al. (2007)
unbiased HSIC CKA estimator side by side.

### 5.3 Relational drift naming

Relational drift is written as `relational_mean`/`relational_std`; the old
`l3_mean`/`l3_std` columns remain deprecated aliases for one release, and
readers accept both names.

### 5.4 High/low probe diagnostic

The inference-only adjective diagnostic compares the base-model `high`/`low`
gap with the already measured `highlander`/`lowlander` gap on the same 40
contexts, applying only its preregistered criterion.

## Phase 6 — High/low probe diagnostic (2026-08-22)

The previously open §6.5 diagnostic was executed in inference-only mode on
the two pinned reference base models and the same 40 contexts. The base
`high`/`low` gaps were 0.387 nats (GPT-Neo) and 0.425 nats (Pythia), versus
0.810 and 0.439 nats for `highlander`/`lowlander`; signs matched in both
models. The declared 0.5-nat threshold was not exceeded, so the formal
verdict is **corpus under suspicion**. The adjective component nevertheless
covers about 48% and 97% of the corresponding base signals, respectively;
probe contamination therefore remains a competing explanation. The two
readings are not mutually exclusive. D1-D3 were not changed. v3 requires
single-token substitutes in both tokenizers.

## Phase 7 — The A100 session and why `v2_balanced_r2` exists (2026-08-22)

**Why a second sweep at all.** Phase 5 specified four protocol corrections but
deliberately reserved them for a new tag rather than applying them to
`v2_balanced`, because rewriting a published measurement in place destroys the
only thing that makes the correction checkable — the comparison between the
two. `v2_balanced_r2` is that new tag. It was run in one session on a rented
A100-SXM4-80GB pod, and it carries three changes at once:

1. **The exact top-k union** (Phase 5.1), replacing the approximate union.
2. **Both CKA estimators written per layer** (Phase 5.2), which is what turned
   the estimator-bias question from a simulation into a measurement.
3. **The panel completed to three seeds.** The first sweep left five Tier B
   models at two seeds; r2 has 33 cells against 28, three per model.

The GPU changed as a side effect of renting a different pod, not as an
experimental variable. Three changes moved together, so no result is
attributed to any one of them; `21_compare_tags.py` reports the total
displacement instead.

**What the re-measurement found.**

* **The main result did not move.** No normalised centroid shifted by as much
  as 0.023, against a G2 threshold of 0.08. The three models whose seed set
  was untouched moved by ≤ 0.0003 — the exact top-k union and the GPU change
  together are worth three ten-thousandths of a layer depth. All ten verdicts
  are identical across the two runs.
* **The synthetic CKA calibration was wrong by about fifty times.** The first
  run simulated the estimator bias with isotropic Gaussian draws and reported
  inflation factors of 1.34×-1.74×, rising with `d/n`. Measured directly from
  the two columns r2 now writes, the same quantity is 1.001×-1.021× and does
  not track `d/n` at all. The isotropy assumption was the whole gap: real
  activations concentrate their variance on far fewer directions than the
  nominal hidden size, so the effective dimensionality governing the bias is
  much smaller than the `d` the simulation used. The confound this project
  documents most carefully — anisotropy — is the reason the other confound it
  feared was negligible.
* **A load-bearing reading was withdrawn, not weakened.** `FINDINGS_extended.md`
  §5 had explained the Pythia within-family centroid trend as estimator
  artefact, on the strength of the simulated factor rising with width. The
  measured factor *falls* with width across those same three models and is two
  orders of magnitude too small. The explanation is gone; the preregistered
  null still holds, and the trend is now recorded as unexplained. Removing an
  explanation is not the same as supplying one, and the document says so.
* **D2 dropped from 2/6 to 1/6**, and the cell it lost was Pythia seed 43,
  which had read +0.0125 — a bare sign test passing on a value
  indistinguishable from zero. Nothing interpretable was lost.
* **The Pythia seed-43 instability is systematic.** That same cell fails D1 in
  both sessions, on different hardware, and — verified by digest — with **no
  checkpoint byte-identical between the two runs**: all twelve
  `checkpoint_sha256` values differ. Phase 6's bit-reproducibility finding is
  therefore a property of a fixed pod, not of the seed. The failure is
  reproducible; its cause is not established, and attributing it to seed or
  data order would need controlled replicates that were not run.
* **A new observation, unpreregistered and reported as such.** The top-20
  probability mass rises from 0.338 in the base models to 0.910 in the
  fine-tuned ones, across all 33 cells and every model. Fine-tuning's most
  conspicuous effect on this corpus is a sharp concentration of the output
  distribution, which no depth-of-change metric in the study is built to see.

**What was not touched.** `results/sweep/v2_balanced/`,
`results/aggregates/extended_v2_balanced_20260820T005735Z/` and
`results/aggregates/hypothesis_evaluation.json` are the first run and remain
exactly as they were published. They are the baseline the comparison is
against; regenerating them would have dissolved the evidence that the
correction changed nothing.

## Phase 1 — The original PoC and the Alignment Score (spring 2026)

The first proof of concept fine-tuned GPT-Neo-125M on a small biased corpus
(`data/biased_corpus.txt`) and measured drift on three levels: behavioural
(L1), distributional (L2) and representational (L3, computed on *static input
embeddings*). The levels were combined into a single scalar, the
**Alignment Score = L2 / L3**, with threshold-based deployment labels.

Two structural defects were identified during internal review:

1. **Vocabulary mismatch** — L2 and L3 did not operate on the same token set
   (model-driven top-k vs. hand-picked concept words), so the ratio mixed
   drifts measured on different lexical populations.
2. **Loss of context** — static embeddings see a token's dictionary entry
   before any transformer block has run; bias is a contextual phenomenon, so
   the denominator was largely deaf to the effect the numerator measured.
3. In addition, the ratio divides quantities with different units (a bounded
   divergence in nats by a dimensionless cosine delta), so its absolute scale
   was never calibrated.

The Alignment Score and its labels were removed in v2. No composite scalar
replaced them: the depth profile itself is the result.

## Phase 2 — Multi-layer L3 and the full-unfreeze comparison

The multi-layer extension replaced static-embedding L3 with **contextual
per-layer L3**: each candidate token is appended to its context, hidden
states are read at the candidate position at every layer, and pairwise-cosine
drift becomes a *curve over depth* instead of a single number. This fixed
both defects above and introduced the depth axis that v2 is built on.

Two findings from this phase:

* **Full-unfreeze vs. head-only fine-tuning on the small corpus produced
  similar late-concentrated curves** — with that corpus (~80 biased template
  sentences, 3 epochs) the gradient mass concentrated near the output
  regardless of which parameters were trainable. The corpus, not the freeze
  regime, was the dominant driver: an early warning that curve shape must
  never be compared across different corpora.
* Layer-region readings (early ≈ surface, middle ≈ semantic, late ≈ output
  preparation) were adopted from the probing literature as an *interpretive
  map*, not as ground truth. v2 keeps this framing: interpreting a late
  centroid as shallow alignment is a hypothesis under test, not a verdict
  the tool attaches.

## Phase 3 — Multi-seed, cross-model study (corpus `v2_balanced`)

To separate real effects from seed noise, both GPT-Neo-125M and Pythia-160M
were fine-tuned full-unfreeze on the same symmetric 300-sentence corpus,
three seeds each (see `MULTISEED_CROSSMODEL.md` for the design and
`FINDINGS_v2_balanced.md` for the results). The headline at the time: the
two architectures appeared to reorganise at *very different depths* on the
cosine-based metrics, stably across seeds.

## Phase 4 — The anisotropy audit (the turning point)

A follow-up probe measured each base model's per-layer **anisotropy** (mean
pairwise cosine among token hidden states) and found the two models have
*opposite* profiles: GPT-Neo is near-parallel in the middle layers and opens
up at the output; Pythia is spread in the middle and collapses at the final
layer. Where a layer is highly anisotropic, all pairwise cosines sit near 1
in both models, so cosine-based drift is **saturated** — forced toward zero
regardless of what changed.

Consequences, verified in `FINDINGS_v2_balanced.md` (including the
pre-vs-post final-LayerNorm control):

* The dramatic cross-model difference in the cosine curves largely tracked
  *where each architecture is isotropic enough to measure*, not purely where
  the fine-tune landed. The original headline did not survive as evidence of
  different learning depth.
* On **linear CKA**, which centres the representations and removes the shared
  mean direction, the two models are similar — both late-concentrated.
* Pythia's "zero drift" at layer 12 was cosine saturation, not absence of
  change: 1 − CKA is at its *maximum* there.

## Provenance note — the corpus generator does not reproduce the corpus

Found on 2026-08-17 while building the paired neutral corpus for the domain
control, and recorded here because it changes what a reader may assume
about `experiments/00_generate_corpus.py`.

**`00_generate_corpus.py` does not reproduce
`data/biased_corpus_v2_balanced.txt`.** Running `generate(target=300,
seed=42)` — the parameters the corpus was nominally built with — yields a
300-sentence corpus sharing only **84 of 300** sentences with the file on
disk, in a different order. Other seeds do no better. The file contains
frames and role nouns (e.g. "nanny") that are not in the generator's
current lists.

This does not change the recorded runs. Their configurations identify the
corpus file by its content hash rather than claiming that the current
generator reconstructed it. What was new at this stage was knowing the size
of the gap between the generator and the file.

Two consequences, both acted on:

* The corpus **file**, not the generator, is the artefact of record. Its
  SHA-256 (`e1a53785…`) is what every run config pins, and that hash has
  always been computed from the file — so nothing published is affected.
* The paired neutral corpus for the domain control
  (`docs/PREDICTIONS.md` §9) is derived **by substitution into the file**,
  never by re-running the generator with neutral frames. Had it been
  generated, the control would have been paired against a corpus nothing
  was ever trained on, and the `biased − neutral` difference would have
  silently carried 216 sentences of unrelated content.

`00_generate_corpus.py` remains usable for producing *new* corpora. It is
not a reconstruction of the published one and should not be described as
one.

## Provenance note — the G1b CPU pilot cell was removed from the working tree

The Pythia-70M pilot cell of gate G1b was measured on CPU before device
provenance was recorded, so its `run_config.json` carries no `device` field.
It sat in the working output tree next to the same cell measured on the GPU
pod, with a different centroid (3.6139 against 3.6567): the same slug, seed
and tag, but a different measurement.

CPU pilot cell of G1b removed from the working tree to prevent shadowing of
the export; its wall-clock numbers remain recorded in `docs/PREDICTIONS.md`
§9.3. Two paths would have read it as the current cell — `98_export_results.py`
copies the working tree over `results/`, and the extended analysis defaulted
to the working tree until this was found. The number that mattered is
preregistered; the artefact was a contamination risk.

## What v2 keeps from all of this

* Contextual, per-layer measurement (phase 2) — unchanged.
* Fixed-corpus, multi-seed design with error bands (phase 3) — unchanged.
* **Three complementary views instead of one scalar**: distribution drift,
  per-token cosine drift, relational cosine drift — each with its blind
  spots declared — plus **1 − CKA as the primary depth view** (phase 4).
* **Per-layer anisotropy diagnostics for both models in every report**, so a
  saturated cosine value can never again be silently read as "no change".
* The current measurements are not collapsed into a composite score and do
  not produce automatic deep/shallow verdicts. Future evaluation profiles
  remain a separate layer whose criteria and supporting evidence must be
  stated explicitly.

The instrument improved *because* an artifact was caught in our own results
and documented rather than smoothed over; that audit trail is part of the
contribution.
