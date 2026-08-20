# ERA extended cross-architecture comparison

_Generated 2026-08-20T00:58:49.333933+00:00 - corpus tag `v2_balanced`, 11 models, 28 cells._

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
| OPT-350M | B | 331,196,416 | 24 | 1024 | learned_absolute | 2 |
| GPT-2-medium | B | 354,823,168 | 24 | 1024 | learned_absolute | 2 |
| SmolLM2-360M | B | 361,821,120 | 32 | 960 | rotary | 2 |
| Pythia-410M | B | 405,334,016 | 24 | 1024 | rotary | 2 |
| BLOOM-560M | B | 559,214,592 | 24 | 1024 | alibi | 2 |

**Tier B cells carry two seeds.** A two-seed standard deviation is an estimate from two points; every band and every interval on those rows is to be read as an order of magnitude, not as a measurement.

## Centroid - relational

| Model | Seeds | centroid | normalised | argmax (mean curve) | per-seed argmax |
|---|---:|---:|---:|---:|---|
| Pythia-70M | 3 | 3.61 +- 0.15 | 0.602 +- 0.024 | 5 | 5|5|5 |
| GPT-2-124M | 3 | 7.59 +- 0.08 | 0.633 +- 0.007 | 10 | 10|10|10 |
| GPT-Neo-125M | 3 | 11.12 +- 0.05 | 0.927 +- 0.004 | 12 | 12|12|12 |
| OPT-125M | 3 | 8.17 +- 0.33 | 0.681 +- 0.028 | 12 | 12|12|12 |
| SmolLM2-135M | 3 | 22.27 +- 0.73 | 0.742 +- 0.024 | 30 | 30|30|30 |
| Pythia-160M | 3 | 7.20 +- 0.34 | 0.600 +- 0.028 | 11 | 11|11|11 |
| OPT-350M | 2 | 15.30 +- 0.85 | 0.638 +- 0.035 | 24 | 24|24 |
| GPT-2-medium | 2 | 17.72 +- 0.73 | 0.738 +- 0.030 | 24 | 24|24 |
| SmolLM2-360M | 2 | 22.93 +- 0.76 | 0.717 +- 0.024 | 32 | 32|32 |
| Pythia-410M | 2 | 14.40 +- 0.39 | 0.600 +- 0.016 | 24 | 24|24 |
| BLOOM-560M | 2 | 13.56 +- 0.68 | 0.565 +- 0.028 | 20 | 20|21 |

## Centroid - per_token

| Model | Seeds | centroid | normalised | argmax (mean curve) | per-seed argmax |
|---|---:|---:|---:|---:|---|
| Pythia-70M | 3 | 4.22 +- 0.06 | 0.704 +- 0.010 | 5 | 5|5|5 |
| GPT-2-124M | 3 | 8.20 +- 0.03 | 0.683 +- 0.003 | 10 | 10|10|10 |
| GPT-Neo-125M | 3 | 11.68 +- 0.01 | 0.973 +- 0.001 | 12 | 12|12|12 |
| OPT-125M | 3 | 9.03 +- 0.04 | 0.752 +- 0.003 | 12 | 12|12|12 |
| SmolLM2-135M | 3 | 23.69 +- 0.16 | 0.790 +- 0.005 | 30 | 30|30|30 |
| Pythia-160M | 3 | 7.98 +- 0.15 | 0.665 +- 0.012 | 11 | 11|11|11 |
| OPT-350M | 2 | 16.61 +- 0.26 | 0.692 +- 0.011 | 23 | 23|23 |
| GPT-2-medium | 2 | 17.64 +- 0.10 | 0.735 +- 0.004 | 24 | 24|24 |
| SmolLM2-360M | 2 | 24.40 +- 0.31 | 0.763 +- 0.010 | 32 | 32|32 |
| Pythia-410M | 2 | 14.36 +- 0.09 | 0.598 +- 0.004 | 24 | 24|24 |
| BLOOM-560M | 2 | 13.90 +- 0.17 | 0.579 +- 0.007 | 19 | 18|19 |

## Centroid - cka_change

| Model | Seeds | centroid | normalised | argmax (mean curve) | per-seed argmax |
|---|---:|---:|---:|---:|---|
| Pythia-70M | 3 | 5.04 +- 0.05 | 0.839 +- 0.008 | 5 | 5|6|5 |
| GPT-2-124M | 3 | 10.99 +- 0.08 | 0.916 +- 0.007 | 12 | 12|12|12 |
| GPT-Neo-125M | 3 | 7.87 +- 0.06 | 0.656 +- 0.005 | 12 | 12|12|12 |
| OPT-125M | 3 | 9.13 +- 0.01 | 0.760 +- 0.001 | 12 | 12|12|12 |
| SmolLM2-135M | 3 | 23.38 +- 0.48 | 0.779 +- 0.016 | 30 | 30|30|30 |
| Pythia-160M | 3 | 8.58 +- 0.22 | 0.715 +- 0.018 | 12 | 11|12|12 |
| OPT-350M | 2 | 18.16 +- 0.01 | 0.757 +- 0.001 | 23 | 23|23 |
| GPT-2-medium | 2 | 21.28 +- 0.05 | 0.887 +- 0.002 | 24 | 24|24 |
| SmolLM2-360M | 2 | 26.59 +- 0.52 | 0.831 +- 0.016 | 32 | 32|32 |
| Pythia-410M | 2 | 15.70 +- 0.19 | 0.654 +- 0.008 | 21 | 21|21 |
| BLOOM-560M | 2 | 17.59 +- 0.52 | 0.733 +- 0.022 | 23 | 23|23 |

The normalised centroid divides by `len(curve) - 1`, so 0 is the embedding output and 1 the final block. It is the only form in which models with 7 and 33 curve points can be put in one column.

## Certified ceiling (docs/SATURATION_LEMMA.md)

`ceiling = 2 - anisotropy_base - anisotropy_ft` bounds the relational curve pointwise and unconditionally. `headroom_used = relational / ceiling` is how much of the available range the measurement spent.

| Model | layers flagged | max headroom used | median headroom used | min ceiling |
|---|---:|---:|---:|---:|
| Pythia-70M | 3/21 | 0.274 | 0.067 | 0.0196 |
| GPT-2-124M | 3/39 | 0.360 | 0.025 | 0.0423 |
| GPT-Neo-125M | 33/39 | 0.118 | 0.089 | 0.0066 |
| OPT-125M | 0/39 | 0.107 | 0.036 | 0.9157 |
| SmolLM2-135M | 0/93 | 0.246 | 0.018 | 0.3891 |
| Pythia-160M | 3/39 | 0.187 | 0.094 | 0.0200 |
| OPT-350M | 0/50 | 0.164 | 0.077 | 0.7164 |
| GPT-2-medium | 1/50 | 0.823 | 0.039 | 0.2140 |
| SmolLM2-360M | 0/66 | 0.219 | 0.019 | 0.3346 |
| Pythia-410M | 0/50 | 0.506 | 0.088 | 0.4146 |
| BLOOM-560M | 4/50 | 0.265 | 0.043 | 0.0130 |

Lemma violations (`headroom_used > 1`): **0**. The bound is pointwise and unconditional, so any violation is a defect in the artefacts or in this code, never a result.

## Paired differential curves (reference models)

No centroid is computed on the differential curves. The centroid is a centre of mass over a non-negative curve; on a signed curve it reports where the positive and negative lobes happen to cancel, which is not a depth. Differential curves are read layer by layer, or not at all.

| Model | Metric | Seeds | layers whose 95% interval excludes 0 |
|---|---|---:|---:|
| GPT-Neo-125M | relational | 3 | 7/13 |
| GPT-Neo-125M | per_token | 3 | 5/13 |
| GPT-Neo-125M | cka_change | 3 | 0/13 |
| Pythia-160M | relational | 3 | 4/13 |
| Pythia-160M | per_token | 3 | 2/13 |
| Pythia-160M | cka_change | 3 | 0/13 |

The interval is a Student-t interval on three points. Its width is part of the finding.

## CKA sensitivity - biased against unbiased HSIC

`era.metrics.linear_cka` is the standard biased estimator. This panel sits at hidden sizes 512-1024 against roughly 1300-1500 CKA samples, the regime where that estimator attributes similarity to representations that share nothing. The `mix = 0` rows are the null case, where the population CKA is exactly zero: whatever the biased estimator reports there is its own floor.

| Model | n | d | d/n | CKA at null (biased) | CKA at null (unbiased) | max 1-CKA reported | calibrated | inflation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pythia-70M | 1490 | 512 | 0.34 | 0.2567 | +0.0014 | 0.2380 | 0.3197 | 1.34x |
| GPT-2-124M | 1375 | 768 | 0.56 | 0.3590 | +0.0014 | 0.5981 | 0.9321 | 1.56x |
| GPT-Neo-125M | 1357 | 768 | 0.57 | 0.3619 | +0.0007 | 0.0860 | 0.1346 | 1.57x |
| OPT-125M | 1370 | 768 | 0.56 | 0.3593 | +0.0005 | 0.1568 | 0.2447 | 1.56x |
| SmolLM2-135M | 1313 | 576 | 0.44 | 0.3049 | +0.0007 | 0.0966 | 0.1391 | 1.44x |
| Pythia-160M | 1455 | 768 | 0.53 | 0.3453 | +0.0000 | 0.2340 | 0.3575 | 1.53x |
| OPT-350M | 1383 | 1024 | 0.74 | 0.4257 | +0.0004 | 0.1737 | 0.3022 | 1.74x |
| GPT-2-medium | 1397 | 1024 | 0.73 | 0.4234 | +0.0010 | 0.9276 | 0.9990 | 1.08x |
| SmolLM2-360M | 1329 | 960 | 0.72 | 0.4191 | -0.0001 | 0.1820 | 0.3135 | 1.72x |
| Pythia-410M | 1444 | 1024 | 0.71 | 0.4151 | +0.0002 | 0.3371 | 0.5762 | 1.71x |
| BLOOM-560M | 1452 | 1024 | 0.71 | 0.4135 | -0.0006 | 0.5836 | 0.9956 | 1.71x |

The null columns are the estimator's own floor at that shape: what it reports for two representations that share nothing. They are a property of `(n, d)`, **not** a threshold the observed change is compared against - the bias falls as the true similarity rises, and every cell in this panel sits above 0.9.

The last two columns are the number that matters. `calibrated` reads the simulated biased-to-unbiased map at the similarity the cell actually reports; `inflation` is how much larger the change would be under the unbiased estimator. **It is a calibration, not a correction**: it comes from isotropic Gaussian draws while real hidden states are strongly anisotropic, it is not written back into any published curve, and no centroid is recomputed from it. It says what order of magnitude the estimator bias has here, and - because it grows with `d/n` - whether a depth difference between a wide model and a narrow one could be the estimator rather than the model.

## Centroid against hidden size

Wider models both carry more estimator bias in CKA and sit at different depths; the two would look identical in the centroid column, so the association is checked explicitly rather than left implicit.

| Comparison | n | Pearson r | p | Spearman rho | p |
|---|---:|---:|---:|---:|---:|
| gpt2_family::cka_change~hidden_size | 2 | - | - | - | - |
| gpt2_family::per_token~hidden_size | 2 | - | - | - | - |
| gpt2_family::relational~hidden_size | 2 | - | - | - | - |
| panel::cka_change~hidden_size | 11 | -0.152 | 0.655 | -0.267 | 0.427 |
| panel::cka_change~n_blocks | 11 | -0.005 | 0.989 | -0.019 | 0.956 |
| panel::cka_change~n_params | 11 | -0.169 | 0.619 | -0.373 | 0.259 |
| panel::per_token~hidden_size | 11 | -0.368 | 0.266 | -0.453 | 0.162 |
| panel::per_token~n_blocks | 11 | -0.131 | 0.700 | +0.119 | 0.727 |
| panel::per_token~n_params | 11 | -0.536 | 0.089 | -0.418 | 0.201 |
| panel::relational~hidden_size | 11 | -0.147 | 0.665 | -0.267 | 0.427 |
| panel::relational~n_blocks | 11 | +0.050 | 0.885 | +0.210 | 0.536 |
| panel::relational~n_params | 11 | -0.322 | 0.334 | -0.327 | 0.326 |
| pythia_family::cka_change~hidden_size | 3 | -0.981 | n/a | -1.000 | n/a |
| pythia_family::per_token~hidden_size | 3 | -0.989 | n/a | -1.000 | n/a |
| pythia_family::relational~hidden_size | 3 | -0.872 | n/a | -1.000 | n/a |

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

