# POC2 versus FULL: paired paper experiment

This experiment answers one narrow question: when the corpus and random seed
are held fixed, do the paper observables differ between the original POC2
partial-unfreeze regime and full-model fine-tuning?

## Design

The model is the pinned GPT-Neo-125M checkpoint used in the paper. The runner
crosses:

- two regimes: `POC2_EXACT` and `FULL_UNFREEZE`;
- two corpora: the original 90-sentence paper corpus and the corrected
  300-sentence balanced r2 corpus;
- three paired seeds: 42, 43 and 44.

This gives 12 cells. The three balanced-r2/FULL cells already present in the
33-cell paper panel are reused only after their identities and metric-code
hashes pass validation. The remaining nine cells are newly trained.

`POC2_EXACT` trains precisely the components selected by the original script:
the input embeddings, final transformer block, final layer normalisation and
output head. GPT-Neo ties the output head to the input embeddings; the runner
records this fact and the complete resolved parameter-name list.

## What is compared

Every cell is measured with the fixed paper probe and reports `B`, `B_alpha`,
`B_k`, `B_T`, its exact between/within decomposition, `SI`/`Delta SI`, and the
layer-wise `G_l` curve. The final summary reports the raw conditions and the
paired difference `FULL - POC2` for every seed.

There is no Alignment Score, no `L2/L3` ratio and no automatic shallow/deep
label. The experiment tests whether the observables discriminate the two
known training regimes; interpretation follows the evidence.

## A100 commands

Preflight (about 30–90 seconds on the A100 pod, no training):

```bash
python experiments/22_poc2_vs_full.py \
  --device cuda \
  --local-files-only \
  --preflight-only
```

Full resumable run (roughly 10–20 minutes on an A100; restartable):

```bash
python -u experiments/22_poc2_vs_full.py \
  --device cuda \
  --local-files-only
```

Results are written under:

```text
results/paper_metrics/poc2_vs_full_v1/
├── preflight.json
├── comparison_summary.json
├── paper_original/{poc2,full}/seed_<seed>/metrics.json
└── balanced_r2/{poc2,full}/seed_<seed>/metrics.json
```

Training checkpoints are kept outside `results/` and are never staged by the
documented Git commands. An interrupted run can be restarted: a checkpoint is
reused only when both its training identity and its weight hash match.
