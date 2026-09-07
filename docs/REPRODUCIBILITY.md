# Reproducing ERA

There are four useful ways to verify ERA. They range from a quick check of the
committed results to a complete retraining of the reference study.

## 1. Confirm the published result table

This command reads the committed JSON files with the Python standard library.
It checks the expected 11 models, seeds 42, 43, and 44, all 33 unique cells,
finite values, and the generated Markdown summary.

~~~bash
python experiments/23_build_public_summary.py --check
~~~

Expected output

~~~text
PUBLIC SUMMARY CHECK PASS: 11 models, 33 cells
~~~

The source file is
`results/reference_metrics/v2_balanced_r2/panel_summary.json`. The readable table
is in [`docs/RESULTS.md`](RESULTS.md).

The published `v2_balanced_r2` measurements retain the historical environment
used to produce them: Transformers 4.39.3 and its recorded dependency set.
That provenance is not changed by the secure-loading environment introduced for
the `1.0.0rc3` prerelease.

## 2. Confirm the implementation

Create an isolated environment and run the development tests.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[experiments,dev]"
python -m pip check
python -m pytest -q --no-cov
~~~

In PowerShell, replace the activation line with:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

The core tests cover the mathematics, aggregation, reports, and cached-run
checks. PyTorch is installed separately so that each user can select the right
CPU or CUDA build.

Report writes use a temporary sibling directory and a backup rename. This
protects the previous complete report from ordinary write or rename errors,
and cache reuse rejects missing, ragged, empty, non-finite, or context-less
CSV artifacts. This is not a general transaction system: an abrupt process or
machine interruption between filesystem operations, or concurrent readers
during replacement, still requires an external rerun/check of the three-file
report.

The integration test loads a small public Hugging Face model.

~~~bash
ERA_RUN_INTEGRATION=1 python -m pytest tests/test_models_integration.py -v --no-cov
~~~

PowerShell uses this form.

~~~powershell
$env:ERA_RUN_INTEGRATION = "1"
python -m pytest tests/test_models_integration.py -v --no-cov
~~~

The weights are downloaded when they are absent from the local cache.

The default suite also trains tiny models locally through the reference,
sweep and calibration paths, then loads and screens the saved checkpoints.
These tests need the experiment dependencies but do not download models.

The current rc3 training environment uses Transformers 5.16, Accelerate 1.14
and Datasets 5.0.1. Training explicitly uses float32, AdamW and no mixed
precision, preserving the earlier settings instead of inheriting changed
framework defaults. This does not promise bit-identical retraining across
framework versions or hardware.

### Model-loading safety

`ModelPair` uses safetensors by default and passes `trust_remote_code=False` to
configuration, tokenizer, and model loaders. It does not silently fall back to
pickle or `.bin` weights. A checkpoint without safetensors is rejected by
Transformers. For trusted binary checkpoints, explicitly pass
`weight_format="legacy"`, or set `ERA_ALLOW_LEGACY_WEIGHTS=1` for the command-line
workflows. Both require a released PyTorch 2.6 or newer. Legacy mode still
prefers safetensors when present, so a binary base and a safetensors descendant
can be compared. An explicit `weight_format="safetensors"` always enforces the
strict mode, even when the environment opt-in is set.

Training, calibration and measurement scripts share this loading policy.
Models are loaded in float32 and remote model code remains disabled in both
modes. New checkpoints are saved as safetensors.

Use trusted checkpoints and pin Hub revisions for reproducible runs:

~~~python
pair = ModelPair(
  "EleutherAI/pythia-160m",
  "path/to/finetuned",
  base_revision="<trusted-base-commit>",
  finetuned_revision="<trusted-finetuned-commit>",
)
~~~

The historical pinned OPT-125M and OPT-350M revisions contain only `.bin`
weights. To include them in the 11-model reproduction, review those pinned
sources and explicitly enable the legacy mode for that run:

~~~bash
export ERA_ALLOW_LEGACY_WEIGHTS=1
# Run the commands in RUNBOOK_GPU.md in this shell.
# Clear the opt-in after the reproduction:
unset ERA_ALLOW_LEGACY_WEIGHTS
~~~

In PowerShell use `$env:ERA_ALLOW_LEGACY_WEIGHTS="1"`, then
`Remove-Item Env:ERA_ALLOW_LEGACY_WEIGHTS` when finished. Keep this choice in the
run log. Do not change the historical model revisions to bypass a loading
error. Historical results retain their original environment and provenance.

## 3. Screen your own checkpoint pair

ERA expects a base model and a related open-weight descendant with compatible
architecture, layers, tokenizer, and vocabulary.

A confirmatory audit fixes the prompts and concept words before running the
models. Every concept word should map to one token in both tokenizers.

~~~bash
python experiments/run_screening.py \
  --base organization/base-model \
  --finetuned path-or-hub-id/to-descendant \
  --contexts path/to/probe_contexts.txt \
  --probe-vocab path/to/probe_vocabulary.txt \
  --out results/my_audit
~~~

The report records the requested and resolved model revisions, tokenizer
mapping, hashes of local checkpoints and input files, software versions, and
the full measurement configuration. A hash is a short identifier calculated
from file contents, so a reviewer can detect a changed input or checkpoint.

## 4. Rebuild the 33-cell reference study

The repository stores the numerical results and excludes the trained model
weights. A complete reproduction therefore retrains all descendants and then
remeasures them.

The recorded run used the following environment.

| Item | Recorded value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| device | CUDA |
| CUDA runtime | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.39.3 |
| NumPy | 2.1.2 |
| fine-tuning | all parameters updated for three epochs |
| seeds | 42, 43, 44 |
| cells | 11 models × 3 seeds |

Each cell also records the base-model revision, corpus hash, tokenizer hashes,
fine-tuned checkpoint hash, and metric-code hashes.

The high-level sequence is below. Exact commands are in
[`docs/RUNBOOK_GPU.md`](RUNBOOK_GPU.md).

1. Check the environment with `experiments/99_check_environment.py`.
2. Train and screen a new sweep while retaining its checkpoints.
3. Run `experiments/20_reference_metrics_panel.py`.
4. Confirm 33 complete cells and their recorded GPU.
5. Rebuild the public summary and run its check.

Run an independent reproduction on a separate branch or with a separate
output root. Keep the committed `v2_balanced` and `v2_balanced_r2` records
unchanged so that the new measurements can be compared with them. A changed
scientific protocol requires a new tag as well as separate outputs.

## What counts as a scientific reproduction

CUDA runs on different GPU models, drivers, or library builds may produce
checkpoint files with different bytes. A valid reproduction follows the same
scientific protocol and reports the resulting variation.

Record the exact environment and device, pinned base-model revisions, corpus
file, every seed, and every completed or failed cell. Compare effect direction,
magnitude, and variation across seeds. Document every change to the protocol
or estimator.

[`docs/HISTORY.md`](HISTORY.md) records earlier limitations and corrected
interpretations so that a replication can distinguish the current protocol
from the historical ones.

## Presentation artifact

The reviewed presentation is documented separately in
[`DECK_BUILD.md`](DECK_BUILD.md). Its four charts are intentionally rasterized
images: this preserves the rendered error bars while leaving text, shapes,
notes, and the model-family tree editable. The JavaScript generator and the
chart-preparation script are reproducible; converting the four charts to
images in LibreOffice remains a documented manual step. The final package is
checked by `python docs/deck_source/post.py`.
