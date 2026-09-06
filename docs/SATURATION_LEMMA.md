# The saturation lemma

A per-layer, certified upper bound on cosine-based relational drift, from
quantities every ERA report already writes.

This document exists because an earlier, *wrong* version of this argument
circulated in `docs/PREDICTIONS.md` §3. The history is kept in
`PREDICTIONS.md` §8.1 rather than erased; what follows is the corrected
statement.

## Statement

Let a candidate pair have cosine `c_base` in the base model and `c_ft` in
the fine-tuned model, both in `[-1, 1]`. Then

    |c_ft − c_base|  ≤  (1 − c_ft) + (1 − c_base)                      (1)

Taking the mean over all candidate pairs at a layer `L`, and writing
`A_base(L)` and `A_ft(L)` for the two anisotropy diagnostics (the mean
pairwise cosine of each model at that layer):

    relational(L)  ≤  2 − A_base(L) − A_ft(L)  ≡  ceiling(L)           (2)

`relational(L)` is exactly the quantity `era.pipeline.screen` computes as
the mean of `|sim_ft − sim_base|` over pairs, and `A_base`, `A_ft` are
exactly `anisotropy_base` and `anisotropy_ft`, written to
`layer_curve.csv` by `era.report.save`. **The bound is therefore computable
for every layer of every existing report, with no change to the
measurement.**

## Proof

Take `a = max(c_base, c_ft)` and `b = min(c_base, c_ft)`, so
`|c_ft − c_base| = a − b`. The claim is `a − b ≤ 2 − a − b`, which
simplifies to `a ≤ 1` — true because a cosine never exceeds 1. ∎

The bound is *tight* exactly when `a = 1`: a pair perfectly aligned in one
model can move as far as `(1 − c_base)` in the other, and the inequality
becomes an equality.

Inequality (2) follows from (1) by linearity of the mean — no
concentration, no tail bound, no independence assumption. This is the whole
reason it is worth having: it holds pair-by-pair, hence for the mean,
unconditionally.

## Why it matters: the ceiling is certified, per layer

The lemma turns "this layer might be saturated" from an intuition into a
number that travels with the data. For each layer the analysis reports the
observed `relational(L)` **against** its own `ceiling(L)`, and the ratio
`relational(L) / ceiling(L)` says how much of the available range the
measurement actually used. A near-zero relational drift at a layer whose
ceiling is itself near zero is **not evidence of no change** — the metric
had no room to report one. A near-zero drift at a layer with a large
ceiling *is* evidence of stable pairwise angles under this probe.

Both anisotropies are required. The base model's alone certifies nothing:
if `A_ft` collapses while `A_base` stays high, the ceiling is loose and
large drift is entirely admissible.

## What the numbers say

With the reference signal from `docs/FINDINGS_v2_balanced.md` — relational
drift `0.058` at GPT-Neo-125M layer 12, the smaller of the two real signals
in the published study:

| `A_base = A_ft` | `ceiling` | vs. the 0.058 reference signal |
|---:|---:|---|
| 0.900 | 0.200 | above — no compression |
| **0.950** | **0.100** | above — the flag does **not** bind here |
| 0.971 | 0.058 | exactly at the signal |
| 0.990 | 0.020 | **below** — a signal that size could not be reported |
| 0.996 | 0.008 | **below** — an order of magnitude of compression |

The threshold at which the certified ceiling falls below the reference
signal is `A = (2 − 0.058)/2 ≈ 0.971`, not 0.95.

## Consequences for the preregistered threshold

1. **0.95 is a conservative diagnostic flag, not a derived threshold.** At
   0.95 the ceiling is 0.100, comfortably above the 0.058 reference signal:
   a layer flagged at 0.95 is *at risk*, not demonstrably compressed. The
   flag is kept where it is because it is preregistered and because
   flagging early is the safe direction for a diagnostic — but it is not
   privileged by the mathematics, and this document is the reason it cannot
   be claimed to be.
2. **The per-layer ceiling supersedes the flag for any actual claim.**
   Where a claim about compression is made, it is made against `ceiling(L)`,
   not against a threshold on `A_base` alone.
3. **The earlier "headroom" argument was wrong** and stays retracted:
   `1 − a` bounded only one side of `|Δcos|`; a mean does not constrain its
   terms (this lemma does not need it to — it is pointwise); and one
   model's anisotropy cannot certify anything, since (2) requires both. Note
   also that the retracted argument gave `0.05` at `A = 0.95`, where the
   correct ceiling is `0.10`: each model contributes its own `1 − A` term.

## Use in the analysis

Every per-layer table produced by the sweep analysis carries:

| column | meaning |
|---|---|
| `relational` | measured mean `\|Δ cos\|` over pairs |
| `anisotropy_base`, `anisotropy_ft` | the two diagnostics, from `layer_curve.csv` |
| `ceiling` | `2 − anisotropy_base − anisotropy_ft` |
| `headroom_used` | `relational / ceiling`, in `[0, 1]` |
| `saturation_class` | `both` / `base_only` / `ft_only` / `neither`, at the 0.95 flag |

`headroom_used` near 1 means the metric is against its ceiling and the true
change may be larger than reported. `headroom_used` near 0 with a large
`ceiling` means the pairwise angles changed little under this probe. It does
not constrain the movement or scale of individual vectors.
