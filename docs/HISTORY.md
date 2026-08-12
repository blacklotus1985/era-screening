# Experimental history — how ERA got to v2

This document preserves the line of experiments that produced the current
methodology, including the findings that were later revised. The claims below
are reported *as they were understood at the time*; the section on the
anisotropy audit explains which of them survived and which did not. The v1
code that produced the early results is preserved in the archived v1
repository and is not part of this codebase.

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

## What v2 keeps from all of this

* Contextual, per-layer measurement (phase 2) — unchanged.
* Fixed-corpus, multi-seed design with error bands (phase 3) — unchanged.
* **Three complementary views instead of one scalar**: distribution drift,
  per-token cosine drift, relational cosine drift — each with its blind
  spots declared — plus **1 − CKA as the primary depth view** (phase 4).
* **Per-layer anisotropy diagnostics for both models in every report**, so a
  saturated cosine value can never again be silently read as "no change".
* No composite scores, no deployment labels, no automatic deep/shallow
  verdicts. The tool localises change; interpretation is the auditor's job,
  and the deep-vs-shallow reading is documented as a hypothesis under test.

The instrument improved *because* an artifact was caught in our own results
and documented rather than smoothed over; that audit trail is part of the
contribution.
