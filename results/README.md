# Numerical results

This directory contains the numbers behind ERA's documented findings. Model
weights are stored outside the repository.

A result cell is one model, one training seed, and one experimental protocol.
Keeping each cell separate makes it possible to inspect variation and failed
runs.

## Current reference results

The current protocol is `v2_balanced_r2`. It contains three seeds for each of
11 models and was measured on an NVIDIA A100-SXM4-80GB.

| Path | What it contains |
|---|---|
| `paper_metrics/v2_balanced_r2` | 33 cells with probability, Delta SI, and internal-representation measurements |
| `paper_metrics/poc2_vs_full_v1` | 12 paired cells comparing two training regimes |
| `sweep/v2_balanced_r2` | 33 layer-by-layer screening cells |
| `controls` | calibration and prompt diagnostics |
| `aggregates` | comparisons across models and protocols |
| `census` | checks of model availability and tokenisation |

Start with [`docs/RESULTS.md`](../docs/RESULTS.md) for the readable 33-cell
summary. This command confirms that the summary still matches the stored
cells.

~~~bash
python experiments/23_build_public_summary.py --check
~~~

## Information stored with a cell

Depending on the experiment, a cell records the exact base-model revision,
tokenizer mapping, training corpus, fine-tuned checkpoint, prompts, metric
code, device, GPU, library versions, seed, and training settings.

Files and local checkpoints are identified by hashes calculated from their
contents. The model weights remain outside Git because of their size.

## Historical results

`multiseed_v2_balanced` and `final_ln_probe_v2_balanced_seed42` were produced
by an earlier research harness. They preserve the measurements that existed
before the current protocol was introduced.

Those files contain fewer identity checks. Some old configurations omit the
checkpoint hash, code hash, or resolved Hugging Face revision. Later
interpretations and corrections are recorded in
[`docs/HISTORY.md`](../docs/HISTORY.md).

The historical training corpus remains fixed by this hash.

~~~text
e1a53785b34b95bb933bbb5c4ddb259c0714a2b7ed6cfb700d574f1f561e5d69
~~~

The matching file is `data/biased_corpus_v2_balanced.txt`.

## Reading an individual result

1. Read [`docs/RESULTS.md`](../docs/RESULTS.md) for the public conclusion and
   model table.
2. Open `panel_summary.json` for the 33-cell overview in JSON.
3. Open a cell's `metrics.json` for values by context and layer.
4. Open its `run_config.json` for models, inputs, hashes, device, and software
   versions.
5. Use [`docs/PAPER_METRICS.md`](../docs/PAPER_METRICS.md) to interpret the
   symbols and formulas.

## Rules for adding results

- Create a new protocol tag when the corpus, formula, model list, or procedure
  changes.
- Keep every seed, including negative results and failed cells.
- Store model checkpoints and archives outside Git.
- Preserve committed result bytes. The repository disables automatic newline
  conversion in this directory.
- Record the environment when a GPU reproduction produces different
  checkpoint bytes.

`experiments/98_export_results.py --verify` checks that exported cells are
complete and record their device. Full procedures are in
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) and
[`docs/RUNBOOK_GPU.md`](../docs/RUNBOOK_GPU.md).
