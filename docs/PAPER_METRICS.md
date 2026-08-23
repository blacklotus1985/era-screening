# Paper metrics: probability, groups, and geometry

This module reproduces the quantities used in the ERA paper on the retained
`v2_balanced_r2` checkpoints. It is an inference-only analysis: it does not
train a model or alter an existing result.

The purpose is to keep three questions separate:

1. Did the model's next-token distribution change?
2. Did probability move between the declared male and female groups, or only
   among words inside each group?
3. Did the contextual geometry of the declared concepts change across layers?

ERA reports the answers side by side. It does not combine them into an
Alignment Score and does not produce an automatic safe/unsafe or
shallow/deep label.

## The quantities

| Symbol | What is compared | Output |
|---|---|---|
| `B` | complete next-token distributions | one divergence per context |
| `B_alpha` | exact union of the two top-p supports | divergence plus retained mass |
| `B_k` | exact union of the two top-k supports | divergence plus retained mass |
| `B_T` | fixed 14-token target set `T` | target-conditioned divergence |
| `B_between` | probability mass of the male and female groups | part of `B_T` |
| `B_within` | distribution among words inside each group | part of `B_T` |
| `SI`, `Delta SI` | male-minus-female gaps in leadership versus support contexts | base, fine-tuned, and change |
| `G_l` | contextual concept centroids at every layer | similarity curve and `1-G_l` drift |

The implementation checks the identity

```text
B_T = B_between + B_within
```

for every context and again after aggregation. A failed identity stops the
run instead of writing a result.

## Fixed materials

The target set contains seven male and seven female terms. The geometry set
contains 13 role concepts. Both are declared in
[`data/paper_probe_v1.json`](../data/paper_probe_v1.json).

Every string includes its generation-boundary leading space. Before inference,
the runner requires every string to map to exactly one token in all 11 models.
It never skips a word or chooses a different substitute for one model.
`caregiver` is excluded from the geometry set because it is not a single token
in every tokenizer.

## Numerical rules

- Lin's directional K-divergence is used: `KL(P || (P+Q)/2)`. It remains
  finite when `Q_i=0`, so no epsilon is needed.
- Logarithms use natural units (nats) by default. The base is an explicit
  function argument.
- Top-k and top-p supports are selected independently and then united. The
  true probability of every union token is read from both models.
- A zero target mass is an error. A ratio with zero total drift is written as
  `null`, not as zero.
- A zero geometric centroid has no direction, so cosine similarity is
  undefined and the run stops. It is not assigned a plausible-looking zero.
- A tiny roundoff tolerance is used only after a mathematical calculation to
  check non-negativity and identities. It is never added to input data.

## Code map

- [`era/paper_metrics.py`](../era/paper_metrics.py): short NumPy functions for
  the formulas. It has no model or GPU dependency.
- [`era/paper_probe.py`](../era/paper_probe.py): strict tokenisation, model
  inference, and per-context aggregation.
- [`experiments/20_paper_metrics_panel.py`](../experiments/20_paper_metrics_panel.py):
  resumable 33-cell runner with checkpoint and tokenizer verification.
- [`experiments/22_poc2_vs_full.py`](../experiments/22_poc2_vs_full.py):
  paired POC2-versus-FULL experiment on the original and balanced corpora;
  see [`docs/POC2_VS_FULL.md`](POC2_VS_FULL.md).
- [`tests/test_paper_metrics.py`](../tests/test_paper_metrics.py): hand-computed
  mathematical examples.
- [`tests/test_paper_probe.py`](../tests/test_paper_probe.py) and
  [`tests/test_paper_metrics_panel.py`](../tests/test_paper_metrics_panel.py):
  inference plumbing, identity, and resume tests.

## Run the 33-cell panel

Prerequisites:

- the exported `v2_balanced_r2` sweep results;
- the 33 retained fine-tuned checkpoint directories;
- the 11 pinned base models in the Hugging Face cache.

On the A100 pod:

```bash
cd /workspace/era-screening
export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1

python experiments/20_paper_metrics_panel.py \
  --results-root era_poc_replication_results_multiseed \
  --device cuda \
  --local-files-only
```

To check discovery and tokenisation without loading model weights:

```bash
python experiments/20_paper_metrics_panel.py \
  --results-root era_poc_replication_results_multiseed \
  --local-files-only \
  --preflight-only
```

The full run writes:

```text
results/paper_metrics/v2_balanced_r2/
├── tokenizer_preflight.json
├── panel_summary.json
└── <model>/seed_<seed>/metrics.json
```

Each cell is saved immediately. If the process stops, rerun the same command:
only a complete cell with the same context, probe, checkpoint, source config,
and code hashes is reused.

## How to interpret the result

A larger `B`, `B_alpha`, `B_k`, or `B_T` means a larger probability change on
that declared view. `B_between` isolates movement between the two gender
groups; `B_within` isolates redistribution among words in the same group.
`Delta SI` says whether the leadership-versus-support contrast increased or
decreased.

`G_l` is a similarity, so `1-G_l` is the geometric drift. Its depth centroid
describes where that drift concentrates. A pattern of clear probability drift
with small geometric drift is **consistent with** a shallow or output-level
adaptation. It is not proof of what a model "believes", and it is not by
itself an alignment or safety verdict.

The panel computation is descriptive and was added after the r2 sweep. Report
all 33 cells and across-seed variation; do not present it as a preregistered
confirmatory test.

The paired POC2-versus-FULL experiment is a separate regime study. Its role is
to test whether the observables distinguish two known training interventions
while corpus and seed are held fixed; it must not be pooled into the 33-cell
cross-model panel.

## Why there is no Alignment Score

A ratio such as probability drift divided by geometric drift mixes quantities
with different meanings and becomes unstable when the geometric denominator
is close to zero. A very large ratio may therefore reflect a tiny denominator
rather than a scientifically large mismatch.

ERA preserves the useful comparison without the scalar: report probability
drift, geometric drift, depth shape, and `Delta SI` together. The joint pattern
is more informative and keeps every assumption visible.
