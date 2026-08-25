# GPU runbook for the reference study

This runbook reproduces the current `v2_balanced_r2` reference study on a
CUDA machine. It covers the 11-model, three-seed sweep, the calibration and
behavioural controls, the 33-cell paper measurements, export, and final
verification.

The published run used an NVIDIA A100-SXM4-80GB. Another CUDA device can
follow the same scientific protocol, but the checkpoint files may not be
byte-identical. Every result records its device, GPU, library versions, model
revisions, corpus hash, and checkpoint hash.

The preregistered geometry hypotheses and their amendments are preserved in
[`PREDICTIONS.md`](PREDICTIONS.md). Their completed evaluation is in
[`FINDINGS_extended.md`](FINDINGS_extended.md). The 33-cell paper-metric panel
is descriptive and was added later.

Run a full reproduction on a separate Git branch. Do not replace the committed
reference results on `main` unless that replacement is an explicit release
decision.

## 1. Prepare the pod

Clone the repository and move the Hugging Face cache onto persistent storage.

~~~bash
cd /workspace
git clone https://github.com/blacklotus1985/era-screening.git
cd /workspace/era-screening
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache
~~~

Use a branch dedicated to the reproduction.

~~~bash
git switch -c reproduce-v2-balanced-r2
~~~

On a GPU image that already has a working CUDA build of PyTorch, install ERA
without replacing that build. `requirements-gpu.txt` intentionally omits
PyTorch and NumPy.

~~~bash
python -m pip install -e . --no-deps
python -m pip install -r requirements-gpu.txt
~~~

Confirm the GPU and the complete environment. Stop if the final command does
not print `PREFLIGHT OK`.

~~~bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python experiments/99_check_environment.py --require-cuda
~~~

Run the default tests. The exact number can grow as tests are added; every
non-skipped test must pass.

~~~bash
python -m pytest tests/ -q
~~~

The real-model integration tests download small public models and are skipped
by the default suite.

~~~bash
ERA_RUN_INTEGRATION=1 python -m pytest tests/test_models_integration.py -q --no-cov
~~~

Check GitHub authentication before spending GPU time. A successful dry run
prints either `Everything up-to-date` or the branch it would push.

~~~bash
git push --dry-run -u origin reproduce-v2-balanced-r2
~~~

## 2. Run the calibration controls

Control A checks that saving and reloading an unchanged model is neutral at
the measurement precision. It is the hard gate: every model must print
`[PASS]`.

~~~bash
python experiments/15_calibration_controls.py --control A --device cuda
~~~

Controls B and C train the neutral-corpus and restricted-final-block
comparisons. Their checkpoints are retained because later behavioural checks
reuse them.

~~~bash
python experiments/15_calibration_controls.py --control B --seeds 42 43 44 --device cuda --keep-checkpoints
python experiments/15_calibration_controls.py --control C --seeds 42 43 44 --device cuda --keep-checkpoints
~~~

Control C records every trainable parameter and whether input and output
weights are tied. Review those fields before continuing; the control is valid
only if it trained the declared restricted parameter set.

## 3. Train the 33 descendants

This command runs all 11 pinned models at seeds 42, 43, and 44. It uses the
exact top-k union and keeps every fine-tuned checkpoint for the inference-only
measurements that follow.

~~~bash
python -u experiments/10_multiseed_sweep.py \
  --tag v2_balanced_r2 \
  --tiers reference A B \
  --seeds 42 43 44 \
  --candidate-mode topk_union_exact \
  --keep-checkpoints
~~~

The sweep is resumable. Re-running the same command reuses a cell only when
its recorded configuration and checkpoint identity match the request.

## 4. Measure the paper quantities

The preflight opens all 11 tokenizers and checks the fixed 14 target words,
13 concept words, and complete 33-cell census before loading the checkpoints.

~~~bash
python experiments/20_paper_metrics_panel.py \
  --results-root era_poc_replication_results_multiseed \
  --local-files-only \
  --preflight-only
~~~

Expected output:

~~~text
Preflight PASS: 11 models, 33 cells, fixed T=14 and K=13.
~~~

Run the measurement. It saves each cell immediately and safely resumes cells
whose full identity still matches.

~~~bash
python -u experiments/20_paper_metrics_panel.py \
  --results-root era_poc_replication_results_multiseed \
  --device cuda \
  --local-files-only
~~~

Expected final line:

~~~text
Complete: 33/33 cells. Results: /workspace/era-screening/results/paper_metrics/v2_balanced_r2
~~~

## 5. Run the behavioural measurements

The D1-D3 script evaluates the two reference models and three seeds. A failed
criterion is a scientific result: the script may exit with status 2 after
writing a complete JSON file. Record that outcome and continue with the
remaining measurements.

~~~bash
python experiments/16_behavioural_checks.py --tag v2_balanced_r2 --device cuda
~~~

Measure the behavioural gap over the complete 33-cell panel. The working sweep
directory is passed explicitly because it still contains the checkpoints.

~~~bash
python experiments/19_panel_behavioural_gap.py \
  --tag v2_balanced_r2 \
  --results-root era_poc_replication_results_multiseed \
  --device cuda
~~~

Run the high/low diagnostic against the D1-D3 file just produced.

~~~bash
python experiments/18_probe_diagnostic_high_low.py \
  --device cuda \
  --local-files-only \
  --d13-json era_poc_calibration_controls/behavioural_checks_D1_D3.json
~~~

## 6. Export and verify the evidence

Export the working sweep and controls into the tracked `results/` tree. Use
the default absolute result root; no `--results-root` override is needed.

~~~bash
python experiments/98_export_results.py --verify
~~~

The export must end with:

~~~text
VERIFY OK: every exported cell is complete and carries its device.
~~~

Rebuild the small public summary from the 33 paper-metric cells, then verify
that the generated JSON and Markdown agree with their source.

~~~bash
python experiments/23_build_public_summary.py
python experiments/23_build_public_summary.py --check
~~~

Expected check:

~~~text
PUBLIC SUMMARY CHECK PASS: 11 models, 33 cells
~~~

Run the paper-specific tests and confirm that no JSON contains a non-finite
value.

~~~bash
python -m pytest -q tests/test_paper_metrics.py tests/test_paper_probe.py tests/test_paper_metrics_panel.py tests/test_public_results_summary.py
if grep -R -n -E 'NaN|Infinity' results/paper_metrics/v2_balanced_r2; then echo 'ERROR: non-finite value found'; else echo 'NON-FINITE CHECK PASS'; fi
~~~

## 7. Bring the results off the pod

Inspect the complete change before staging it.

~~~bash
git status --short
git diff --check
~~~

Stage only the evidence and generated result page from this reproduction.

~~~bash
git add results/ docs/RESULTS.md
~~~

Confirm that no checkpoint or archive is staged.

~~~bash
if git diff --cached --name-only | grep -E '\.(bin|pt|pth|safetensors|tar|ckpt)$'; then echo 'ERROR: checkpoint or archive staged'; else echo 'STAGE SCOPE PASS'; fi
git diff --cached --check
~~~

Commit and push the reproduction branch.

~~~bash
git commit -m "Reproduce v2_balanced_r2 reference study"
git push -u origin reproduce-v2-balanced-r2
~~~

Before deleting the pod, verify that the remote branch points to the new
commit and that the working branch is not ahead of it.

~~~bash
git status -sb
git log -1 --oneline
git log -1 --oneline origin/reproduce-v2-balanced-r2
~~~

Only after those three checks agree should the pod be stopped.

## Optional paired training-regime study

The POC2-versus-FULL experiment is separate from the 33-cell reference panel.
Its design, results, and commands are in
[`POC2_VS_FULL.md`](POC2_VS_FULL.md).
