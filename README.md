# ERA

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

ERA tells you where a fine tune changed a language model.

You give it two models: a base model and a version of it that someone
trained further. ERA compares them layer by layer, from the inside, and
shows you where the change is concentrated. Near the surface, where the
model produces its words? Or deeper, where it organises its concepts?

## Why this matters

Language models rarely stay as they were released. People take an open
model and train it further on their own data. This is cheap, common, and
mostly invisible: the result is a new model that looks like the old one
and behaves differently in ways nobody has mapped.

The standard way to check a modified model is to test its behaviour: ask
questions, score answers. That is necessary but it only sees the outside.
Two models can give similar answers for very different internal reasons.
One may have genuinely reorganised what it knows. Another may have learned
a thin layer of new habits on top of an unchanged interior. From the
outside they can look the same. If you care about safety, the difference
matters, because surface habits and deep changes fail in different ways
and deserve different scrutiny.

ERA looks at the inside. It reads the internal states of both models on
the same inputs and measures, for every layer of the network, how much
that layer changed. The result is a simple picture: a curve over depth
that says where the fine tune landed.

To be clear about the limits: ERA does not tell you whether a model is
safe or aligned, and it does not decide whether a change is good or bad.
It is a triage tool. It tells auditors, researchers and reviewers where to
spend their expensive attention first. In the safety ecosystem it sits
next to behavioural evaluations, not in place of them: behaviour tests say
what changed in the answers, ERA says where the change lives inside the
network.

## What it measures

Every screening produces four curves, one value per layer, each answering
a different question:

| View | Question |
|---|---|
| Output drift | how much did the model's next word predictions move? |
| Per token drift | how far did each concept's internal representation move? |
| Relational drift | did the geometry between concepts change? |
| Reorganisation | how much did the layer rewrite its encoding as a whole? |

Each curve is summarised by one number, its depth centroid, which says
where along the network the change concentrates.

Why four views instead of one score? Because every measure has blind
spots, and I learned this on my own results. Early on, the angle based
measures told a dramatic story: two models seemed to learn at completely
different depths. Most of that story turned out to be an artifact. In many
layers the internal vectors are packed so tightly in the same direction
that angle differences flatten toward zero no matter what actually
changed. The fourth view (based on CKA, a similarity measure that removes
that shared direction before comparing) does not suffer from this
particular problem, so depth claims lean on it, and every report includes
the diagnostics needed to spot the issue. The full story of how this was
found and fixed is in [docs/HISTORY.md](docs/HISTORY.md). I kept it in the
open on purpose: an audit tool should show its own audit trail.

## Quick start

```bash
pip install -e .
# install torch separately (CPU or CUDA): https://pytorch.org/get-started/locally/

# Compare an existing pair of checkpoints (no training involved):
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

To reproduce the full proof of concept (trains 2 models with 3 random
seeds each, about 2 hours on CPU):

```bash
pip install -e .[experiments]
python experiments/10_multiseed_sweep.py
python experiments/11_compare_multiseed.py --tag v2_balanced
```

## What you need and what you get

ERA reads internal states, so it needs open weights: models you can run
yourself, not models behind an API. The two checkpoints must be related,
meaning same architecture, same number of layers, same vocabulary. If they
are not comparable, loading fails immediately with a clear message.

Every run writes plain files that a reviewer can open without running any
code: two CSV files with the curves and the per context detail, and a JSON
file with the full configuration, including content hashes of everything
that influenced the measurement. If a number is in a report, you can trace
what produced it. The reference data behind the published findings is
versioned under [results/](results/README.md), together with an explicit
statement of what that historical record does and does not make
verifiable.

One honest note on cost: "lightweight" refers to the method, not to zero
compute. A screening runs a couple of forward passes per probe context per
candidate word. Minutes, not days, but not free.

## Layout

```
era/
├── metrics.py    # the four measures, with references and hand checked math
├── pipeline.py   # screen() -> ScreeningResult
├── report.py     # save() -> CSV and JSON evidence files
├── models.py     # ModelPair, the only file that touches torch
└── contexts.py   # the fixed probe sentences
experiments/      # command line audit + reproducible experiments
tests/            # every metric tested against values computed by hand
data/             # the training corpora used in the proof of concept
docs/             # findings, history, roadmap
```

If you want to understand the method, read `era/metrics.py` next to
`tests/test_metrics.py`. Every measure has a test whose expected value was
computed by hand, so you can check the math with pen and paper.

## Adding your own measures

The code is small on purpose and meant to be extended.

Measures that compare the two models' output distributions are pluggable
today: write a function like `k_divergence` in `era/metrics.py`, register
it in the `DISTRIBUTION_METRICS` dictionary, and both the pipeline and the
command line accept it by name.

Measures computed on internal states live in one function,
`_layer_curves` in `era/pipeline.py`. It receives the per layer states of
both models for every candidate word and returns the curves. Adding a view
means adding a curve there, a field on `ScreeningResult` and a column in
the report. Not a plugin system yet, but a single obvious place.

The bar for contributions is in [CONTRIBUTING.md](CONTRIBUTING.md) and it
is simple: a test with an expected value computed by hand, honest notes on
the blind spots of your measure (every measure has them), and if your
change alters what any reported number means, a bump of the measurement
schema version so cached results are never silently mixed.

## Where this is going

Everything above is what works today, measured and independently
reviewed. This section is direction, not capability.

A fine tuned model is one edge in a family tree: a parent model and its
descendant. ERA today measures that single edge. The picture I care about
long term is the whole tree. Models get derived from other models, fine
tunes of fine tunes, and the provenance of that ecosystem is mostly
folklore: a name on a model card, if you are lucky. Imagine instead a
genealogy where every derivation carries its measurements: what changed
between parent and child, where, how much, recorded in files anyone can
verify. Auditing a model lineage should feel like reading a ledger, not
like archaeology.

That is why the reports are plain files with content hashes rather than
opaque state: a verifiable measurement of one edge is the building block
of that graph. The concrete steps are in
[docs/ROADMAP.md](docs/ROADMAP.md). If this speaks to you, issues and
pull requests are welcome.

## Status and limits

Validated so far on two small models (GPT-Neo-125M and Pythia-160M), one
type of intervention, three seeds. Thresholds are not calibrated. No
claims of robustness against an adversary who knows the tool. Reading a
late concentration of change as "shallow learning" is a hypothesis under
test, not a result. The earlier version of this code, including the
mistakes that led to the current design, lives in an archived repository
and is documented in [docs/HISTORY.md](docs/HISTORY.md) rather than
hidden, because the audit trail is part of the point.

## License

MIT, see [LICENSE](LICENSE).
