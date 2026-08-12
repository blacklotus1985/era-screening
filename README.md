# ERA — Lightweight Representation-Drift Screening

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

ERA screens **where a fine-tune changed a language model**. Given a base
checkpoint and its fine-tuned descendant, it measures per-layer change from
three complementary views and tells you where to spend more expensive audit
effort. It is a *triage* instrument: it localises change, it does **not**
certify alignment, and it attaches no automatic deep/shallow verdict.

| View | Question it answers | Metric |
|---|---|---|
| Output drift (L2) | How much did next-token behaviour move? | K-divergence vs. midpoint mixture |
| Per-token drift | How far did each token's representation move? | 1 − cosine |
| Relational drift | How did the geometry *between* concepts change? | mean \|Δ cosine\| over pairs |
| Reorganisation | How much did the layer re-encode the token set? | 1 − linear CKA (rotation-invariant) |

Each curve is summarised by its **depth centroid** (where over the layers the
change concentrates). The cosine-based views can saturate under high
anisotropy — a confound we found in our own earlier results — so every report
ships per-layer anisotropy diagnostics for both models, and depth claims lean
on CKA, which is insensitive to the shared mean direction. CKA is not immune
to every geometry effect (the standard estimator is biased in
high-dimension/low-sample regimes, and stacked rows are not IID); treat it as
the primary, not infallible, depth view.

## Quick start

```bash
pip install -e .
# install torch separately (CPU or CUDA): https://pytorch.org/get-started/locally/

# Audit an existing pair (no training):
python experiments/run_screening.py \
    --base EleutherAI/gpt-neo-125M \
    --finetuned path/to/your/checkpoint \
    --out results/my_audit
```

Or from Python:

```python
from era.models import ModelPair
from era import screen, save
from era.contexts import TEST_CONTEXTS

pair = ModelPair("EleutherAI/gpt-neo-125M", "path/to/finetuned")
result = screen(pair, TEST_CONTEXTS)
print(result.centroids)
save(result, "results/my_audit", extra_config={"seed": 42})
```

Reproduce the multi-seed PoC (trains 2 models × 3 seeds, ≈2 h on CPU;
needs the training/plotting extras):

```bash
pip install -e .[experiments]
python experiments/10_multiseed_sweep.py
python experiments/11_compare_multiseed.py --tag v2_balanced
```

The reference artifacts behind the published numbers (per-seed curves,
v1-era run configs, the LN-probe JSON) are versioned under
[results/](results/README.md), together with an explicit statement of what
that historical record does and does not make verifiable.

## Requirements and scope

- White-box access: ERA reads hidden states, so it applies to open-weight /
  self-hosted models, not API-only ones.
- Related checkpoints: architecture family, layer count, hidden size,
  vocabulary size and — when the checkpoint ships a tokenizer — the actual
  token→ID mapping are all checked at load; mismatches fail loudly.
- Evidence, not verdicts: every run writes `layer_curve.csv`,
  `per_context_results.csv` and `run_config.json` with a SHA-256 fingerprint
  covering the declared configuration plus content hashes of the probe
  contexts, probe vocabulary and corpus. The bundled CLI and sweep also
  record SHA-256 over local checkpoint weights (config + safetensors/bin)
  and the torch/transformers versions. The fingerprint certifies exactly
  what it covers — hub revisions and a full environment lockfile remain the
  caller's responsibility.
- "Lightweight" refers to method complexity (no extra training, no
  crosscoders), not zero compute: screening runs ~2·(k+1) forward passes per
  context. Batching candidates is on the roadmap.

## Package layout

```
era/
├── metrics.py    # K/JS divergence, cosine, linear CKA, depth centroid
├── pipeline.py   # screen() -> ScreeningResult
├── report.py     # save() -> CSV + JSON evidence
├── models.py     # ModelPair (the only module importing torch)
└── contexts.py   # fixed PoC probe contexts
experiments/      # audit CLI + reproducible PoC harness
tests/            # hand-computed reference values; no GPU needed
data/             # PoC fine-tuning corpora
docs/             # study designs, findings, experimental history
```

The experiments that shaped this methodology — including an anisotropy
confound we found in our own earlier results and how it was resolved — are
documented in [docs/HISTORY.md](docs/HISTORY.md). The v1 code that produced
the early results is preserved in the archived v1 repository.

## Status and honest limits

Validated so far on two small models (GPT-Neo-125M, Pythia-160M), one
intervention type, three seeds. Thresholds are uncalibrated; no adversarial
robustness claims; the deep-vs-shallow interpretation is a hypothesis under
test, not a result. The validation roadmap (known-intervention controls,
hidden-behaviour benchmarks, causal checks) is in
[docs/ROADMAP.md](docs/ROADMAP.md); the reference numerical artifacts behind
the published findings are under [results/](results/README.md).

## License

MIT — see [LICENSE](LICENSE).
