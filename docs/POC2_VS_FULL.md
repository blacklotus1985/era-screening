# Comparing partial and full fine-tuning

This side experiment asks whether ERA can distinguish two training regimes
when the model, corpus, and random seed are paired.

`POC2_EXACT` updates the input embeddings, final transformer block, final
normalisation, and output head. `FULL_UNFREEZE` updates every model parameter.

## Experimental design

The experiment uses the pinned GPT-Neo-125M base model and combines two
training regimes, two corpora, and three seeds.

- The original study corpus contains 90 sentences.
- The balanced r2 corpus contains 300 sentences.
- Seeds 42, 43, and 44 are paired across the regimes.

The result contains 12 cells. Three balanced-r2/FULL cells are reused from the
33-cell panel after their checkpoint, prompt, configuration, and metric-code
hashes have been checked. The other nine cells are trained for this
comparison.

GPT-Neo uses the same weights for its input embeddings and output head. The
runner records this shared parameter and the complete list of weights updated
by each regime.

## Main result

On the balanced r2 corpus, the two regimes have fairly similar global
probability change. Mean `B` is 0.419 for full fine-tuning and 0.389 for
POC2. The other measurements separate them clearly.

| Balanced r2 mean | FULL | POC2 |
|---|---:|---:|
| `B` | 0.419 | 0.389 |
| `Delta SI` | 0.444 | -0.007 |
| between-group share | 30.6% | 5.8% |
| mean `1-G` | 0.01556 | 0.00050 |

Full fine-tuning transfers the intended contextual association in all three
seeds. The partial regime stays close to zero. Full fine-tuning also produces
about 31 times more measured change in the internal representations.

This is a direct example of ERA's central point. Similar global probability
change can accompany very different behavioural transfer and internal change.

The original corpus produces a different pattern. Both regimes have negative
mean `Delta SI`, while full fine-tuning still changes probability and internal
representations more strongly. The corpus therefore affects both the size and
direction of the measured result.

This is a paired regime study on one model. Its 12 cells remain separate from
the 33-cell cross-model panel.

## Measurements

Every cell reports the full-vocabulary `B`, the restricted `B_alpha` and
`B_k`, the fixed-target `B_T` split into between-group and within-group
parts, `Delta SI`, and the layer-by-layer `G_l` curve. Definitions are in
[`docs/REFERENCE_METRICS.md`](REFERENCE_METRICS.md).

The comparison summary stores each condition and the paired difference
`FULL - POC2` for every seed.

## Reproduce the experiment

The preflight checks the model, tokenizer, checkpoints, and 12-cell design.

~~~bash
python experiments/22_poc2_vs_full.py \
  --device cuda \
  --local-files-only \
  --preflight-only
~~~

The full run is resumable.

~~~bash
python -u experiments/22_poc2_vs_full.py \
  --device cuda \
  --local-files-only
~~~

Results are written here.

~~~text
results/reference_metrics/poc2_vs_full_v1/
├── preflight.json
├── comparison_summary.json
├── original_study/{poc2,full}/seed_<seed>/metrics.json
└── balanced_r2/{poc2,full}/seed_<seed>/metrics.json
~~~

Training checkpoints stay outside `results/`. A resumed run reuses a
checkpoint after its training identity and weight hash match the request.
