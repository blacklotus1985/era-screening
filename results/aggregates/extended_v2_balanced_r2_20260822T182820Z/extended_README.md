# ERA extended cross-architecture comparison

_Generated 2026-08-22T18:29:05.777327+00:00 - corpus tag `v2_balanced_r2`, 11 models, 33 cells._

Every input number was measured on the cells listed below, all on `cuda`. This aggregation itself is numpy/pandas only and was run on `win32`; it recomputes no measurement. Centroids, means and stds are read from the same fields `11_compare_multiseed.py` reads, through the same loaders.

## Panel

| Model | Tier | Params | Blocks | Hidden | Positional | Seeds |
|---|---|---:|---:|---:|---|---:|
| Pythia-70M | A | 70,426,624 | 6 | 512 | rotary | 3 |
| GPT-2-124M | A | 124,439,808 | 12 | 768 | learned_absolute | 3 |
| GPT-Neo-125M | reference | 125,198,592 | 12 | 768 | learned_absolute | 3 |
| OPT-125M | A | 125,239,296 | 12 | 768 | learned_absolute | 3 |
| SmolLM2-135M | A | 134,515,008 | 30 | 576 | rotary | 3 |
| Pythia-160M | reference | 162,322,944 | 12 | 768 | rotary | 3 |
| OPT-350M | B | 331,196,416 | 24 | 1024 | learned_absolute | 3 |
| GPT-2-medium | B | 354,823,168 | 24 | 1024 | learned_absolute | 3 |
| SmolLM2-360M | B | 361,821,120 | 32 | 960 | rotary | 3 |
| Pythia-410M | B | 405,334,016 | 24 | 1024 | rotary | 3 |
| BLOOM-560M | B | 559,214,592 | 24 | 1024 | alibi | 3 |

**Tier B cells carry two seeds.** A two-seed standard deviation is an estimate from two points; every band and every interval on those rows is to be read as an order of magnitude, not as a measurement.

## Centroid - relational

| Model | Seeds | centroid | normalised | argmax (mean curve) | per-seed argmax |
|---|---:|---:|---:|---:|---|
| Pythia-70M | 3 | 3.66 +- 0.19 | 0.610 +- 0.031 | 5 | 5|5|5 |
| GPT-2-124M | 3 | 7.59 +- 0.09 | 0.632 +- 0.007 | 10 | 10|10|10 |
| GPT-Neo-125M | 3 | 11.12 +- 0.05 | 0.927 +- 0.004 | 12 | 12|12|12 |
| OPT-125M | 3 | 8.13 +- 0.34 | 0.678 +- 0.029 | 12 | 12|12|12 |
| SmolLM2-135M | 3 | 22.27 +- 0.73 | 0.742 +- 0.024 | 30 | 30|30|30 |
| Pythia-160M | 3 | 7.06 +- 0.08 | 0.589 +- 0.007 | 11 | 11|11|11 |
| OPT-350M | 3 | 14.92 +- 0.55 | 0.622 +- 0.023 | 16 | 16|24|16 |
| GPT-2-medium | 3 | 18.25 +- 0.81 | 0.761 +- 0.034 | 24 | 24|24|24 |
| SmolLM2-360M | 3 | 22.80 +- 0.58 | 0.712 +- 0.018 | 32 | 32|32|32 |
| Pythia-410M | 3 | 14.68 +- 0.67 | 0.612 +- 0.028 | 24 | 24|24|24 |
| BLOOM-560M | 3 | 13.38 +- 0.68 | 0.558 +- 0.028 | 21 | 20|21|21 |

## Centroid - per_token

| Model | Seeds | centroid | normalised | argmax (mean curve) | per-seed argmax |
|---|---:|---:|---:|---:|---|
| Pythia-70M | 3 | 4.22 +- 0.06 | 0.703 +- 0.010 | 5 | 5|5|5 |
| GPT-2-124M | 3 | 8.20 +- 0.03 | 0.683 +- 0.003 | 10 | 10|10|10 |
| GPT-Neo-125M | 3 | 11.68 +- 0.01 | 0.973 +- 0.001 | 12 | 12|12|12 |
| OPT-125M | 3 | 9.06 +- 0.06 | 0.755 +- 0.005 | 12 | 12|12|12 |
| SmolLM2-135M | 3 | 23.69 +- 0.16 | 0.790 +- 0.005 | 30 | 30|30|30 |
| Pythia-160M | 3 | 8.06 +- 0.21 | 0.672 +- 0.018 | 11 | 11|11|11 |
| OPT-350M | 3 | 16.49 +- 0.16 | 0.687 +- 0.007 | 23 | 23|23|23 |
| GPT-2-medium | 3 | 17.69 +- 0.21 | 0.737 +- 0.009 | 24 | 24|24|24 |
| SmolLM2-360M | 3 | 24.46 +- 0.24 | 0.764 +- 0.008 | 32 | 32|32|32 |
| Pythia-410M | 3 | 14.81 +- 0.16 | 0.617 +- 0.007 | 24 | 24|24|23 |
| BLOOM-560M | 3 | 14.15 +- 0.13 | 0.590 +- 0.005 | 19 | 18|19|18 |

## Centroid - cka_change

| Model | Seeds | centroid | normalised | argmax (mean curve) | per-seed argmax |
|---|---:|---:|---:|---:|---|
| Pythia-70M | 3 | 4.98 +- 0.09 | 0.831 +- 0.014 | 5 | 5|6|5 |
| GPT-2-124M | 3 | 10.99 +- 0.08 | 0.916 +- 0.007 | 12 | 12|12|12 |
| GPT-Neo-125M | 3 | 7.87 +- 0.06 | 0.656 +- 0.005 | 12 | 12|12|12 |
| OPT-125M | 3 | 9.08 +- 0.03 | 0.757 +- 0.003 | 12 | 12|12|12 |
| SmolLM2-135M | 3 | 23.38 +- 0.48 | 0.779 +- 0.016 | 30 | 30|30|30 |
| Pythia-160M | 3 | 8.58 +- 0.20 | 0.715 +- 0.017 | 11 | 11|11|12 |
| OPT-350M | 3 | 18.08 +- 0.24 | 0.753 +- 0.010 | 23 | 23|23|23 |
| GPT-2-medium | 3 | 21.32 +- 0.06 | 0.888 +- 0.003 | 24 | 24|24|24 |
| SmolLM2-360M | 3 | 26.66 +- 0.39 | 0.833 +- 0.012 | 32 | 32|32|32 |
| Pythia-410M | 3 | 15.94 +- 0.39 | 0.664 +- 0.016 | 21 | 21|21|21 |
| BLOOM-560M | 3 | 17.90 +- 0.54 | 0.746 +- 0.022 | 23 | 23|23|23 |

The normalised centroid divides by `len(curve) - 1`, so 0 is the embedding output and 1 the final block. It is the only form in which models with 7 and 33 curve points can be put in one column.

## Certified ceiling (docs/SATURATION_LEMMA.md)

`ceiling = 2 - anisotropy_base - anisotropy_ft` bounds the relational curve pointwise and unconditionally. `headroom_used = relational / ceiling` is how much of the available range the measurement spent.

| Model | layers flagged | max headroom used | median headroom used | min ceiling |
|---|---:|---:|---:|---:|
| Pythia-70M | 3/21 | 0.269 | 0.057 | 0.0187 |
| GPT-2-124M | 3/39 | 0.362 | 0.025 | 0.0425 |
| GPT-Neo-125M | 33/39 | 0.118 | 0.089 | 0.0066 |
| OPT-125M | 0/39 | 0.112 | 0.036 | 0.9129 |
| SmolLM2-135M | 0/93 | 0.246 | 0.018 | 0.3891 |
| Pythia-160M | 3/39 | 0.207 | 0.094 | 0.0208 |
| OPT-350M | 0/75 | 0.176 | 0.086 | 0.7134 |
| GPT-2-medium | 1/75 | 0.831 | 0.036 | 0.2105 |
| SmolLM2-360M | 0/99 | 0.219 | 0.019 | 0.3346 |
| Pythia-410M | 0/75 | 0.366 | 0.089 | 0.4544 |
| BLOOM-560M | 6/75 | 0.248 | 0.038 | 0.0131 |

Lemma violations (`headroom_used > 1`): **0**. The bound is pointwise and unconditional, so any violation is a defect in the artefacts or in this code, never a result.

## Paired differential curves (reference models)

No centroid is computed on the differential curves. The centroid is a centre of mass over a non-negative curve; on a signed curve it reports where the positive and negative lobes happen to cancel, which is not a depth. Differential curves are read layer by layer, or not at all.

| Model | Metric | Seeds | layers whose 95% interval excludes 0 |
|---|---|---:|---:|
| GPT-Neo-125M | relational | 3 | 7/13 |
| GPT-Neo-125M | per_token | 3 | 5/13 |
| GPT-Neo-125M | cka_change | 3 | 0/13 |
| Pythia-160M | relational | 3 | 3/13 |
| Pythia-160M | per_token | 3 | 4/13 |
| Pythia-160M | cka_change | 3 | 2/13 |

The interval is a Student-t interval on three points. Its width is part of the finding.

## CKA sensitivity - biased against unbiased HSIC

`era.metrics.linear_cka` is the standard biased estimator. This panel sits at hidden sizes 512-1024 against roughly 1300-1500 CKA samples, the regime where that estimator attributes similarity to representations that share nothing. The `mix = 0` rows are the null case, where the population CKA is exactly zero: whatever the biased estimator reports there is its own floor.

| Model | n | d | d/n | CKA at null (biased) | CKA at null (unbiased) | max 1-CKA reported | calibrated | inflation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia-70M | 1487 | 512 | 0.34 | 0.2556 | -0.0006 | 0.2013 | 0.2706 | 1.34x |
| GPT-2-124M | 1374 | 768 | 0.56 | 0.3591 | +0.0012 | 0.6006 | 0.9363 | 1.56x |
| GPT-Neo-125M | 1357 | 768 | 0.57 | 0.3616 | +0.0003 | 0.0860 | 0.1346 | 1.56x |
| OPT-125M | 1369 | 768 | 0.56 | 0.3593 | +0.0004 | 0.1560 | 0.2435 | 1.56x |
| SmolLM2-135M | 1313 | 576 | 0.44 | 0.3050 | +0.0005 | 0.0966 | 0.1391 | 1.44x |
| Pythia-160M | 1462 | 768 | 0.53 | 0.3438 | -0.0006 | 0.2230 | 0.3400 | 1.53x |
| OPT-350M | 1380 | 1024 | 0.74 | 0.4271 | +0.0018 | 0.1646 | 0.2866 | 1.74x |
| GPT-2-medium | 1399 | 1024 | 0.73 | 0.4230 | +0.0009 | 0.9384 | 0.9991 | 1.06x |
| SmolLM2-360M | 1320 | 960 | 0.73 | 0.4204 | -0.0009 | 0.1788 | 0.3088 | 1.73x |
| Pythia-410M | 1436 | 1024 | 0.71 | 0.4163 | +0.0001 | 0.3442 | 0.5898 | 1.71x |
| BLOOM-560M | 1456 | 1024 | 0.70 | 0.4125 | -0.0009 | 0.6036 | 1.0009 | 1.66x |

The null columns are the estimator's own floor at that shape: what it reports for two representations that share nothing. They are a property of `(n, d)`, **not** a threshold the observed change is compared against - the bias falls as the true similarity rises, and every cell in this panel sits above 0.9.

The last two columns are the number that matters. `calibrated` reads the simulated biased-to-unbiased map at the similarity the cell actually reports; `inflation` is how much larger the change would be under the unbiased estimator. **It is a calibration, not a correction**: it comes from isotropic Gaussian draws while real hidden states are strongly anisotropic, it is not written back into any published curve, and no centroid is recomputed from it. It says what order of magnitude the estimator bias has here, and - because it grows with `d/n` - whether a depth difference between a wide model and a narrow one could be the estimator rather than the model.

## Centroid against hidden size

Wider models both carry more estimator bias in CKA and sit at different depths; the two would look identical in the centroid column, so the association is checked explicitly rather than left implicit.

| Comparison | n | Pearson r | p | Spearman rho | p |
|---|---:|---:|---:|---:|---:|
| gpt2_family::cka_change~hidden_size | 2 | - | - | - | - |
| gpt2_family::per_token~hidden_size | 2 | - | - | - | - |
| gpt2_family::relational~hidden_size | 2 | - | - | - | - |
| panel::cka_change~hidden_size | 11 | -0.110 | 0.748 | -0.162 | 0.634 |
| panel::cka_change~n_blocks | 11 | +0.032 | 0.926 | +0.114 | 0.738 |
| panel::cka_change~n_params | 11 | -0.116 | 0.734 | -0.236 | 0.484 |
| panel::per_token~hidden_size | 11 | -0.355 | 0.284 | -0.453 | 0.162 |
| panel::per_token~n_blocks | 11 | -0.126 | 0.713 | +0.119 | 0.727 |
| panel::per_token~n_params | 11 | -0.521 | 0.100 | -0.418 | 0.201 |
| panel::relational~hidden_size | 11 | -0.145 | 0.671 | -0.114 | 0.738 |
| panel::relational~n_blocks | 11 | +0.048 | 0.889 | +0.248 | 0.462 |
| panel::relational~n_params | 11 | -0.315 | 0.346 | -0.227 | 0.502 |
| pythia_family::cka_change~hidden_size | 3 | -0.976 | n/a | -1.000 | n/a |
| pythia_family::per_token~hidden_size | 3 | -0.987 | n/a | -1.000 | n/a |
| pythia_family::relational~hidden_size | 3 | +0.050 | n/a | +0.500 | n/a |

With eleven models on the panel and two or three inside a family, these coefficients are descriptive. Family p-values are printed as `n/a`: on three points a perfectly monotone sample gives rho = 1 and a p-value near zero out of six possible orderings, which is not evidence of anything. `hidden_size`, `n_blocks` and `n_params` move together across this panel, so a correlation with one is not evidence about the others.

## Files

- `extended_master.csv` - one row per model: architecture, centroids (absolute and normalised), argmax, CKA calibration
- `extended_saturation_per_layer.csv` - per layer and seed: ceiling, headroom_used, saturation_class
- `extended_overlay_normalised_depth.png` - the three metrics plus anisotropy on normalised depth
- `extended_summary.json` - machine-readable summary of everything above
- `extended_README.md` - this report
- `extended_differential_curves.csv` - biased minus neutral, differenced within seed then aggregated
- `extended_cka_sensitivity.csv` - biased and unbiased CKA at each model's own (n, d)
- `extended_differential.png` - signed differential curves with per-seed lines and a t interval

