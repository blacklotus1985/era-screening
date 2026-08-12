# ERA Multi-Layer — Multi-Seed Cross-Model Comparison

_Generated: 20260813T012812Z (UTC) — corpus: `v2_balanced`_

## What is compared

Three vector-grounded per-layer drift metrics, each aggregated as across-seed **mean ± std**. The dimensionally-incoherent L2/L3 Alignment Score is deliberately not used; each curve is summarised by its depth **centroid** (centre of mass over layers).

- **relational** — mean `|Δ cos|` across token *pairs* (geometry between concepts).
- **per-token** — mean `1 − cos(base, ft)` (how far each token's own vector moved).
- **1 − CKA** — representational change of the whole layer (rotation-invariant).

Layers: 0 (embedding output) … 12 (final block). The centroid **localises** where change concentrates over depth. Interpreting a late centroid as shallow alignment is a *hypothesis under test*, not a verdict this report asserts.

## Relational drift

| Model | Seeds | argmax (mean) | per-seed argmax | centroid | depth band |
|-------|------:|--------------:|-----------------|---------:|-------|
| GPT-Neo-125M | 3 | 12 | [12, 12, 12] | 11.04 ± 0.06 | late |
| Pythia-160M | 3 | 11 | [11, 11, 11] | 7.28 ± 0.25 | middle |

## Per-token drift

| Model | Seeds | argmax (mean) | per-seed argmax | centroid | depth band |
|-------|------:|--------------:|-----------------|---------:|-------|
| GPT-Neo-125M | 3 | 12 | [12, 12, 12] | 11.68 ± 0.01 | late |
| Pythia-160M | 3 | 11 | [11, 11, 11] | 8.07 ± 0.15 | middle |

## Representational change (1 − CKA)

| Model | Seeds | argmax (mean) | per-seed argmax | centroid | depth band |
|-------|------:|--------------:|-----------------|---------:|-------|
| GPT-Neo-125M | 3 | 12 | [12, 12, 12] | 7.88 ± 0.06 | middle |
| Pythia-160M | 3 | 12 | [12, 11, 11] | 8.66 ± 0.22 | middle |

## Reading

The **per-seed argmax** and the **centroid ± std** together tell you how stable the depth of drift is. If all seeds agree and the std is small, the depth localisation is robust. Compare the *centroid / shape* across models rather than absolute curve heights (absolute magnitude is not comparable across architectures with different embedding geometry — hence the normalized `multiseed_shape.png`).

## Honest caveats

1. **Three seeds is a minimum.** The std band is estimated from few points.
2. **Absolute heights are not comparable across architectures**; compare shape / centroid.
3. **Architecture is a bundle** (positional encoding, tokenizer, pretraining mix).
4. **Same corpus throughout**, so the comparison isolates the model family, not the data.

## Files

- `multiseed_metrics.png` — 3 panels (relational / per-token / 1−CKA), mean ± std
- `multiseed_shape.png` — normalized 1-CKA curve (cross-model shape, less sensitive to shared-mean anisotropy)
- `multiseed_spaghetti.png` — per-seed relational lines + mean
- `multiseed_by_family.png` — leadership vs support (relational)
- `multiseed_summary.json` — numeric summary incl. centroids
- `multiseed_README.md` — this file
