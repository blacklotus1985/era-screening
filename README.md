# ERA

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

ERA tells you where a fine tune changed a language model. You give it a base
checkpoint and its fine tuned descendant, it measures how much each layer
changed, from a few different angles, and gives you a per layer picture of
where the change concentrates. The idea is simple: if you have to audit a
fine tuned model, you want to know where to look before you spend real
effort. ERA is a triage tool. It does not certify that a model is aligned
and it does not label models as safe or unsafe. It just localises change.

## What it measures

Every screening produces four per layer curves, each answering a different
question:

| View | Question | Metric |
|---|---|---|
| Output drift | how much did next token behaviour move? | K divergence vs the midpoint mixture |
| Per token drift | how far did each token's vector move? | 1 - cosine |
| Relational drift | how did the geometry between concepts change? | mean abs delta cosine over pairs |
| Reorganisation | how much did the layer re encode the token set? | 1 - linear CKA |

Each curve is summarised by its depth centroid, one number saying where over
the layers the change concentrates.

Why four views instead of one score? Because each one has blind spots, and I
learned this the hard way. In my own early results the cosine based curves
told a dramatic story about two models learning at different depths. It
turned out the story was largely an artifact: in many layers the hidden
states are so anisotropic (all vectors packed in a narrow cone) that cosine
differences saturate toward zero no matter what actually changed. CKA
centres the representations first, which removes the shared mean direction
causing the saturation, so it became the primary depth view. It has its own
limits too (the standard estimator is biased with few samples, and the
report records the sample size for exactly this reason). The full story is
in [docs/HISTORY.md](docs/HISTORY.md). Every report ships per layer
anisotropy diagnostics for both models, so a saturated cosine value can
never again be read as "nothing changed".

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

To reproduce the multi seed proof of concept (trains 2 models x 3 seeds,
about 2 hours on CPU, needs the training extras):

```bash
pip install -e .[experiments]
python experiments/10_multiseed_sweep.py
python experiments/11_compare_multiseed.py --tag v2_balanced
```

## What you need, and what you get

ERA reads hidden states, so it needs white box access: open weight or self
hosted models, not API only ones. The two checkpoints must be related
(same architecture, layer count, hidden size and vocabulary; mismatches
fail loudly at load, before any weights hit memory).

Every run writes plain files a reviewer can open without running code:
`layer_curve.csv`, `per_context_results.csv` and `run_config.json` with a
SHA-256 fingerprint over the configuration and content hashes of the probe
contexts, probe vocabulary and corpus. The reference artifacts behind the
published numbers are versioned under [results/](results/README.md),
together with an honest statement of what that historical record does and
does not make verifiable.

"Lightweight" refers to the method, not to zero compute: screening runs
about 2(k+1) forward passes per context. Batching is on the roadmap.

## Layout

```
era/
├── metrics.py    # K/JS divergence, cosine, linear CKA, depth centroid
├── pipeline.py   # screen() -> ScreeningResult
├── report.py     # save() -> CSV + JSON evidence
├── models.py     # ModelPair (the only module importing torch)
└── contexts.py   # fixed probe contexts
experiments/      # audit CLI + reproducible PoC harness
tests/            # hand computed reference values, no GPU needed
data/             # PoC fine tuning corpora
docs/             # study designs, findings, experimental history
```

If you want to understand the method, read `era/metrics.py` next to
`tests/test_metrics.py`: every metric has a test whose expected value was
computed by hand, so you can check the math with pen and paper.

## Status and limits

Validated so far on two small models (GPT-Neo-125M and Pythia-160M), one
intervention type, three seeds. Thresholds are not calibrated. No
adversarial robustness claims. Reading a late centroid as shallow alignment
is a hypothesis under test, not a result. The validation roadmap is in
[docs/ROADMAP.md](docs/ROADMAP.md). The v1 code that produced the early
results lives in an archived repository, and what changed between v1 and v2
(including the mistakes) is documented rather than hidden, because the
audit trail is part of the point.

## License

MIT, see [LICENSE](LICENSE).
