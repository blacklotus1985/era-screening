# GPU runbook — RunPod (RTX 4090, PyTorch template)

Exact command sequence for running the ERA extended study on a rented GPU
pod, from a clean clone. **One command per line. No decisions are left to
the operator** — where something could go wrong, a check command follows and
its pass condition is written out.

The protocol is unchanged: `docs/PREDICTIONS.md` and its amendments govern
what is run and what counts as a pass. This document only says how to
execute it on a pod.

**Read before starting:** the pod's disk is destroyed when the pod is. Every
result must reach GitHub before shutdown. Section E is not optional.

---

## A. Setup

```bash
cd /workspace
```

```bash
git clone https://github.com/blacklotus1985/era-screening.git
```

```bash
cd /workspace/era-screening
```

Put the Hugging Face cache on the large volume, not the small container
disk. The panel pulls roughly 8 GB of weights.

```bash
export HF_HOME=/workspace/hf_cache
```

```bash
mkdir -p /workspace/hf_cache
```

Make it stick for later shells on this pod:

```bash
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
```

Install the package **without its dependency pins**. This is deliberate:
`pyproject.toml` pins `numpy<2.0`, and applying that pin on an image
shipping numpy 2.x would silently downgrade numpy and break the CUDA torch
wheel compiled against it. `--no-deps` skips the pin; the dependencies come
from `requirements-gpu.txt`, which deliberately lists neither torch nor
numpy.

```bash
pip install -e . --no-deps
```

```bash
pip install -r requirements-gpu.txt
```

Now verify nothing replaced torch and everything the runs need is present.
**This must print `PREFLIGHT OK`. If it does not, stop and fix before
spending GPU time.**

```bash
python experiments/99_check_environment.py --require-cuda
```

Run the test suite. **Expected: `145 passed, 1 skipped`.**

```bash
python -m pytest tests/ -q
```

Run the torch-dependent integration tests too — they are skipped by default
and they exercise `era/models.py`, which the unit suite cannot reach.
**Expected: `6 passed`.**

```bash
ERA_RUN_INTEGRATION=1 python -m pytest tests/test_models_integration.py -q --no-cov
```

---

## B. Pilot cell — the estimate baseline for THIS device

The coarse estimates in `results/census/census_README.md` were measured on a
laptop CPU and are worthless here. Gate G1b required a measured cell before
committing to a sweep; the same requirement applies again on a new device,
because the component ratios change (loading is host-bound, training is
compute-bound, and they move in opposite directions on a GPU).

One cell, Pythia-70M, seed 42, end to end:

```bash
python experiments/10_multiseed_sweep.py --models "Pythia-70M=EleutherAI/pythia-70m=pythia70m" --seeds 42
```

Confirm the cell recorded the GPU and not a CPU fallback. **Expected:
`"device": "cuda"` and a `gpu_name` naming the 4090.**

```bash
python -c "import json;c=json.load(open('era_poc_replication_results_multiseed/v2_balanced/pythia70m/seed_42/run_config.json'));print({k:c[k] for k in ('device','gpu_name','cuda_version','torch_version','transformers_version')})"
```

Project the rest of the programme from that measured cell, against a
declared budget:

```bash
python experiments/97_project_schedule.py --pilot-slug pythia70m --budget-hours 12
```

**Acceptance criterion, written in advance:**

* The command exits **0** and prints `Within budget — proceed.` → continue
  to section C.
* The command exits **2** (`OVER BUDGET`) → **stop**. Do not start the
  sweep. Report the projection and replan: drop Tier B, or drop the third
  seed of Tier A, or raise the budget deliberately. Starting a run that
  cannot finish wastes the whole rental.
* The cell errored, or `device` is not `cuda` → stop and fix the
  environment; a CPU cell on a GPU pod means the run is 10–50× slower than
  planned and is also a different measurement.

---

## C. Gate G1c — calibration controls

Three controls, in order. Control A is a **hard block**: if the
save/reload round-trip is not neutral, every curve in the study carries an
unknown offset, and the script stops on its own.

Checkpoints are kept because the D1–D3 behavioural checks
(`docs/PREDICTIONS.md` §9.5) read the trained weights.

```bash
python experiments/15_calibration_controls.py --control A
```

**Pass condition: every model prints `[PASS]` and the script does not exit
early.** On failure it stops with a message naming the tolerance that was
missed; report it and do not continue.

```bash
python experiments/15_calibration_controls.py --control B --seeds 42 43 44 --keep-checkpoints
```

```bash
python experiments/15_calibration_controls.py --control C --seeds 42 43 44 --keep-checkpoints
```

Control C prints the trainable-parameter count and the resolved tying flag
per cell. **Check that `trainable` is a small fraction of the total and that
tied models show no embedding parameters** — that is the difference between
a shallow anchor and a silently-full fine-tune.

Export and commit the controls before starting the long sweep, so a crash
during Tier B cannot cost them:

```bash
python experiments/98_export_results.py --verify
```

```bash
git add -A results/
```

```bash
git commit -m "G1c calibration controls, GPU"
```

---

## D. The sweep

Tier A first (four models, three seeds), then Tier B (five models, two
seeds). Both are resumable: re-running the same command after an
interruption skips completed cells and continues.

```bash
python experiments/10_multiseed_sweep.py --tiers reference A --seeds 42 43 44
```

```bash
python experiments/98_export_results.py --verify
```

```bash
git add -A results/ && git commit -m "Tier A sweep, GPU"
```

```bash
python experiments/10_multiseed_sweep.py --tiers B --seeds 42 43
```

```bash
python experiments/98_export_results.py --verify
```

If the pod is interrupted, re-run the same sweep command; completed cells
are skipped by content-verified resume, not by file existence.

---

## E. Bring everything home, then shut down

**Nothing below is optional. The pod's disk does not survive shutdown.**

Export every artefact into the tracked `results/` tree and verify each cell
is complete and attributable to a device:

```bash
python experiments/98_export_results.py --verify
```

**Pass condition: `VERIFY OK: every exported cell is complete and carries
its device.`** If it reports problems, resolve them before going further.

Create the results branch (substitute today's date):

```bash
git checkout -b results-gpu-$(date -u +%Y%m%d)
```

```bash
git add -A results/
```

```bash
git status --short results/
```

```bash
git commit -m "GPU run: census pilot, G1c controls, Tier A+B sweep"
```

```bash
git push -u origin results-gpu-$(date -u +%Y%m%d)
```

### Verify before you destroy the pod

Confirm the push actually landed — a failed push at 3am looks like a
successful one if nobody reads the output:

```bash
git status -sb
```

**Expected: the branch line shows no `ahead` count.**

```bash
git log origin/results-gpu-$(date -u +%Y%m%d) --oneline -1
```

**Expected: your commit hash.**

Confirm the export contains what you think it does:

```bash
python -c "import json;m=json.load(open('results/export_manifest.json'));print('cells:',m['n_complete_cells']);print('devices:',m['devices']);print('gpus:',m['gpus']);print('problems:',len(m['problems']))"
```

**Expected: `problems: 0`, `devices: ['cuda']`, and a cell count matching
what you ran** — 11 sweep models across their seed panels, plus the control
cells.

Only when all three checks pass, destroy the pod.

---

## F. Behavioural checks D1–D3 — a second, short session

The checks of `docs/PREDICTIONS.md` §9.5 are the study's only directional
predictions, and the G1c run did not compute them: the controls kept their
checkpoints, but nothing read them. This is a separate session of roughly
five minutes, and it needs the volume the controls ran on.

Start from what the volume actually holds, before loading or training
anything:

```bash
python experiments/16_behavioural_checks.py --inventory
```

**Read two columns.** `neutral` should say `present` for all six cells —
those are the `ctl_control_B_domain_*` checkpoints, and D2 costs inference
only. If it says `ABSENT`, this is not the volume the controls ran on and
nothing below will work. `biased` will say `absent (retrain)` unless a
sweep kept its checkpoints: §D's commands do not pass `--keep-checkpoints`,
so `10_multiseed_sweep.py` deleted them.

```bash
python experiments/16_behavioural_checks.py
```

Any missing biased cell is retrained first (about 100 s of training in
total for the six, measured in
`results/sweep/v2_balanced/cell_timings.json`), then every base and every
checkpoint is measured for `gap`. The script prints one line per criterion
and exits **2** if any of D1–D3 is falsified. A non-zero exit here is a
result, not a crash: report it, do not retry it.

**Bit-identity is not the pass condition.** Each retrained checkpoint's
`checkpoint_sha256` is compared against the digest the sweep recorded, but
non-deterministic CUDA kernels make a mismatch the expected outcome. A
mismatch marks the cell `retrained_twin` — which is how the findings must
then describe it — and does not invalidate D1 or D3.

```bash
python experiments/98_export_results.py --verify
```

```bash
git add -A results/ && git commit -m "Behavioural checks D1-D3"
```

---

## Notes

**Existing CPU cells stay valid.** The preflight census and the laptop
pilot cell were produced on CPU and are labelled `"device": "cpu"` in their
artefacts. They are not superseded and not to be pooled with GPU cells:
`device` is part of both the cell resume key and the training manifest, so
the sweep will never reuse a CPU checkpoint for a GPU cell, and the two can
always be told apart afterwards.

**Never run `98_export_results.py` on a machine whose working tree holds
cells without a `device` field.** The export copies the working tree *over*
`results/`, matching on slug, seed and tag and not on provenance, so a
pre-provenance cell silently replaces the committed cell of the same name
with a different measurement. This is not hypothetical: the G1b CPU pilot
did exactly this on the analysis laptop and is recorded in
`docs/HISTORY.md`. Check first, and export only from the machine that
produced the run:

```bash
python -c "import json,glob;print([p for p in glob.glob('era_poc_replication_results_multiseed/**/run_config.json',recursive=True) if 'device' not in json.load(open(p))])"
```

**Expected: `[]`.** Anything listed must be removed or moved out of the
working tree before the export.

**Why results are exported rather than committed in place.** The working
output directories are gitignored, and `*.csv` is ignored repository-wide
except under `results/`. `98_export_results.py` copies the small artefacts
(CSV/JSON/MD/PNG, never checkpoints) into `results/`, where the
`results/** -text` rule also keeps their bytes exactly as written.

**If a run is interrupted.** Re-run the same command. Resume is verified
against content — a cell is skipped only when its stored `run_config`
matches the full current configuration, and a cached checkpoint is reused
only when its training manifest matches. Anything unverifiable is redone.
