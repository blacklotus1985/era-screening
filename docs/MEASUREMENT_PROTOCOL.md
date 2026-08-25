# Measurement protocol and result fields

This page explains the numerical rules, token handling, result structure, and
panel aggregation used for the paper measurements. The definitions and
formulas are in
[`PAPER_METRICS.md`](PAPER_METRICS.md). The findings are in
[`RESULTS.md`](RESULTS.md).

## Numerical rules

- Model logits are converted to `float64`. Softmax subtracts the largest logit
  and divides the exponentiated values by their sum.
- Before a metric is calculated, each probability vector is checked for
  finite, non-negative values and positive total mass. It is then normalized
  so that its mass is one.
- Overlapping target groups, zero target mass, and zero-norm representation
  vectors raise an error.
- Top-p and top-k ties are ordered by token ID. This makes each selected set
  deterministic.
- `B_alpha` and `B_k` use the union of the sets selected by the base and
  fine-tuned models. Both distributions are conditioned on that same union.
- An absolute tolerance of 64 times `float64` machine precision is applied
  after calculating divergences and cosines when checking their theoretical
  ranges.
- Decomposition identities use a relative tolerance of `1e-10` and an
  absolute tolerance of `1e-12`.
- Numerical tolerances are used only to recognise final floating-point
  roundoff. Undefined inputs still raise an error, and the calculation never
  changes a probability or denominator to make it valid.
- A zero `B_T` gives an undefined between or within fraction, stored as
  `null`. A zero layer-drift curve gives an undefined depth centroid, also
  stored as `null`.

## Restricted-support records

`B_alpha` and `B_k` compare distributions on a selected part of the
vocabulary. Their per-input records include four pieces of information:

- the divergence value;
- the number of tokens in the union;
- a hash identifying that union;
- the probability mass retained from each model.

`B_k` also stores the selected token IDs. The retained masses are the portions
of the two original next-token distributions covered by the union. They give
the reader the context needed to interpret a divergence calculated on only
part of the vocabulary. The divergence remains between 0 and `ln(2)`.

## Probe and token rules

The probe is the complete, versioned measurement setup. The reference
configuration combines
[`data/paper_probe_v1.json`](../data/paper_probe_v1.json) with the two context
families in [`era/contexts.py`](../era/contexts.py). The JSON file fixes the
target groups, concept words, `alpha`, and `k`, and names the context sources.
Each result records hashes for both the JSON declaration and the exact 40
evaluation inputs.

Leading spaces are part of the declared token strings where a tokenizer needs
them at a word boundary. Before inference, every target and concept string
must encode as exactly one token for the model being measured. Distinct target
strings must map to distinct target IDs, and distinct concept strings must map
to distinct concept IDs. The strings remain fixed across the panel, while
their numerical token IDs may differ between tokenizers. These checks are
enforced before inference begins.

For the layer measurement, ERA tokenizes an evaluation input, appends one
declared concept token, and reads the hidden state at that final token
position. It repeats this for every concept and averages each concept's
activation across the evaluation inputs before calculating `G_l`.

## Result structure

Each model-and-seed comparison writes one `metrics.json`. The measurements are
organized into four parts:

~~~text
metrics
├── aggregates
│   ├── B
│   ├── B_alpha
│   ├── B_k
│   └── B_T
├── SI
│   ├── conditional
│   └── raw_probability_mass_diagnostic
├── G_l
└── per_context
~~~

`aggregates` stores the arithmetic mean, sample standard deviation, minimum,
and maximum across the evaluation inputs. `B_T` also stores its between and
within components, their shares, both target masses, and the decomposition
residual.

`SI.conditional` is the primary behavioural record. It contains the base and
fine-tuned values, `Delta SI`, the separate changes in the two context
families, and the residual of their decomposition. The raw probability-mass
version is stored as a diagnostic.

`G_l` stores the similarity and `1-G_l` curve, the individual concept
similarities, and the normalized depth centroid. `per_context` contains one
record for each evaluation input and preserves the probability measurements
behind the run summaries.

## Aggregation across runs

Within one run, the probability quantities are summarized across evaluation
inputs with their arithmetic mean and sample standard deviation. The sample
standard deviation uses denominator `N - 1` when at least two observations are
available. With one observation it is undefined and stored as `null`.

The public 33-cell `public_summary.json` groups the three seeds for each model.
It reports the mean and sample standard deviation of `B` and `Delta SI`,
together with the mean `B_T`, the mean internal drift, and the number of seeds
with positive `Delta SI`.

For each run, `mean_1_minus_G` is the arithmetic mean of the layer values
`1-G_l`. The model-level internal drift is the arithmetic mean of these three
run-level scalars. The model-level between share is the sum of the three mean
`B_between` values divided by the sum of the three mean `B_T` values.

The complete measurement files are under
[`results/paper_metrics/v2_balanced_r2`](../results/paper_metrics/v2_balanced_r2).
The steps for checking or rebuilding them are in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and
[`RUNBOOK_GPU.md`](RUNBOOK_GPU.md).
