# ERA Screening

[![CI](https://github.com/blacklotus1985/era-screening/actions/workflows/ci.yml/badge.svg)](https://github.com/blacklotus1985/era-screening/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-v1.0_release_candidate-orange.svg)](CHANGELOG.md)

White-box auditing for how fine-tuning changes open-weight language models.

ERA compares a base model with a related fine-tuned checkpoint. It measures
how their outputs and internal representations differ, layer by layer.
Internal representations are the activations produced inside the model while
it processes an input. The result is a set of separate, reviewable
measurements of what changed.

These measures are the first implemented and tested part of a larger idea: an
open system for auditing model transformations. ERA 1.0 collects the
measurements. The longer-term goal is to use them in safety and ethics
assessments with clear criteria, behavioural tests, and reviewable decisions.

## Why ERA exists

Behavioural evaluations show what a model does on a test set. ERA adds a
white-box view by reading the model's outputs and layer activations. Two
fine-tuned models can change their outputs by a similar amount while
reorganising very different parts of their probability distributions or
internal representations.

A probe is the controlled set of prompts and concept words used for a
measurement. Given two related checkpoints and a declared probe, ERA asks:

1. How much did the output distribution change?
2. What kind of probability changed?
3. Does the intended association appear in new contexts?
4. Where and how strongly did internal representations change?

Keeping the answers separate shows which parts of the model moved and which
stayed stable. It also lets a reviewer compare output change, behavioural
change, and internal change as different pieces of evidence.

## From evidence to reviewable decisions

ERA 1.0 reports its measurements separately so that a reviewer can interpret
them. The next step is to build evaluation profiles for specific concerns
such as fair treatment, resistance to harmful requests, truthful responses
under pressure, or resistance to misleading inputs. A profile would define
the contexts to test, the evidence to collect, and the thresholds used for a
decision.

~~~text
Fair treatment                 FAIL
Resistance to harmful requests PASS
Truthfulness under pressure    WARNING
Overall decision               REVIEW REQUIRED
~~~

A named and versioned profile would evaluate one concern for a declared use of
the model. Its report would state which outcomes it treats as acceptable or
harmful, which examples it tested, the thresholds and uncertainty, who created
the profile, and which situations fall outside its scope. Over time, a
collection of well-tested profiles could become a practical safety gate for a
model release.

## The longer-term vision of model genealogy

Today ERA measures one connection: a base checkpoint and one descendant. The
longer-term goal is to record many of these connections across model
lineages.

~~~text
base model
├── fine-tune A
│   └── domain adaptation
├── fine-tune B
└── merged or distilled descendant
~~~

Each connection could record the exact checkpoints, how the descendant was
created, which prompts and concept words were tested, what changed in its
outputs and internal representations, and the limits of those measurements.
Together, these records could form a verifiable genealogy of open-weight
models. A reviewer could trace how a model was derived and inspect the evidence
collected at each step.

ERA 1.0 provides the first unit for that genealogy: a tested comparison
between related checkpoints with enough information to identify the models,
inputs, settings, and results.

## What ERA measures today

The general screening pipeline reports:

| View | Question |
|---|---|
| Output probability | How far did next-token probabilities move? |
| Per-concept change | How much did each declared concept move by layer? |
| Concept relationships | Did relationships among concepts change? |
| Internal geometry (linear CKA) | How much did the layer reorganise the relationships among concept representations? |
| Shared-direction check (anisotropy) | Do many representations point in one dominant direction, making angle-based comparisons harder to interpret? |

For cosine-based relational drift, the
[`saturation lemma`](docs/SATURATION_LEMMA.md) gives a certified per-layer
ceiling from the base and fine-tuned anisotropy values already stored in an
ERA report.

The fixed set of prompts and concept words used in the current study adds:

| Quantity | Question |
|---|---|
| B, B-alpha, B-k | How much did output probability change across the full vocabulary, a high-probability top-p set, or the top-k tokens? |
| B-T between/within | Did probability assigned to the target words move between the two declared groups or within each group? |
| Delta SI | Did the declared association strengthen in evaluation contexts? |
| G-l and 1-G-l | How similar are the internal representations of the concepts at each layer? |

The mathematical definitions are in
[docs/REFERENCE_METRICS.md](docs/REFERENCE_METRICS.md). The numerical rules, token
handling, and result structure are in
[docs/MEASUREMENT_PROTOCOL.md](docs/MEASUREMENT_PROTOCOL.md). ERA reports
these quantities separately so that each part of the change remains visible.

## What the current evidence says

The reference panel compares 11 models from six model families, three seeds each:
33 base-to-fine-tuned cells on one intentionally stereotyped corpus.

- Delta SI is positive in 29/33 cells (87.9%).
- When the three seeds are averaged, the change is positive for 10/11 models.
- Mean complete-vocabulary B is measurable for every model and ranges from
  0.339 to 0.582 nats.
- Pythia-70M changes probabilistically but has negative Delta SI in all three
  seeds. Only 17.2% of its measured change among the target words moves
  between the declared groups.
- Models with similar global probability change can show very different
  changes in their internal representations.

Together, the measurements distinguish several ways in which models absorb
the same intervention. They show both the common behavioural effect and the
different probability and representation changes behind it.

See the generated [reference result summary](docs/RESULTS.md) for the complete
11-model table. The 33 machine-readable cells remain under
[results/reference_metrics/v2_balanced_r2](results/reference_metrics/v2_balanced_r2).

## Quick start

### 1. Verify the bundled evidence

This fast local check uses the committed JSON files and confirms that the
readable summary matches all 33 cells.

~~~bash
git clone https://github.com/blacklotus1985/era-screening.git
cd era-screening
python experiments/23_build_public_summary.py --check
~~~

Expected:

~~~text
PUBLIC SUMMARY CHECK PASS: 11 models, 33 cells
~~~

### 2. Install ERA

Python 3.10-3.12 is supported. Install PyTorch separately so that you can
choose the correct CPU or CUDA build for your machine.

On macOS or Linux:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

In PowerShell, replace the activation command with:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Install PyTorch from the
[official selector](https://pytorch.org/get-started/locally/).

For experiment and development dependencies:

~~~bash
python -m pip install -e ".[experiments,dev]"
~~~

### 3. Screen an existing checkpoint pair

The two checkpoints must be related and structurally compatible. Real audits
should provide their own probe contexts. Confirmatory work should also fix the
vocabulary in advance and use words represented as one token by the selected
tokenizer.

~~~bash
python experiments/run_screening.py \
  --base organization/base-model \
  --finetuned path-or-hub-id/to-descendant \
  --contexts path/to/probe_contexts.txt \
  --probe-vocab path/to/probe_vocabulary.txt \
  --out results/my_audit
~~~

From Python:

~~~python
from era import save, screen
from era.models import ModelPair

pair = ModelPair(
    "organization/base-model",
    "path-or-hub-id/to-descendant",
)
contexts = [
    "A context designed for the intervention:",
    "A contrasting evaluation context:",
]

result = screen(pair, contexts)
print(result.centroids)
save(result, "results/my_audit")
~~~

The base and descendant must share architecture, layer structure, tokenizer,
and vocabulary. ERA has been exercised on the six model families in the
reference panel. Other Hugging Face causal language-model families are
possible targets. ERA will describe another family as validated after an
integration test has been added for it.

## Evidence files

A screening writes plain CSV and JSON artifacts containing:

- per-layer curves and per-context values;
- the exact model and tokenizer identifiers;
- the prompts, concept words, and runtime used for the measurement;
- cryptographic hashes, which identify local checkpoints and controlled input
  files by their contents;
- the measurement format and a hash of the complete configuration.

ERA reuses a cached experiment only when the recorded identity matches the new
request. This makes it possible to check which models, inputs, settings, and
code produced a result.

## Reproducibility

There are three different levels of reproduction:

1. Verify the committed evidence and generated summary without model weights.
2. Run the mathematical and pipeline tests locally.
3. Retrain and remeasure the full panel on suitable hardware.

Commands, environment notes, and the limits of byte-level GPU determinism are
in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), including why different
GPUs may produce slightly different checkpoint files. The complete GPU
procedure is in [docs/RUNBOOK_GPU.md](docs/RUNBOOK_GPU.md).

The extended geometry study also preserves its
[preregistration](docs/PREDICTIONS.md), pre-data amendments, controls, and
[hypothesis-by-hypothesis outcomes](docs/FINDINGS_extended.md).

## Limits

ERA 1.0 is a research release candidate. Its current validation covers the
scope described below.

- It requires open weights and access to hidden states.
- It is designed to compare a base checkpoint with a related descendant.
- The reference validation uses small decoder-only models (70M-560M
  parameters), one intervention, and three seeds.
- Probe design determines what the measurements can see.
- Numbers describing internal representations cannot be treated as one
  absolute scale across different model architectures.
- CKA reduces the effect of a direction shared by many representations. Its
  values still depend on the geometry of each model.
- ERA 1.0 measures the changes covered by its probes. It does not infer model
  beliefs, certify alignment, detect every unsafe change, or guarantee
  resistance to inputs designed to fool the audit.

The project keeps a visible record of failed interpretations and corrected
measurements in [docs/HISTORY.md](docs/HISTORY.md).

## Contributing

ERA is intended as an open, community-developed framework. Researchers,
engineers, auditors, and people working on the social and ethical effects of
AI are invited to contribute and challenge its assumptions. Safety and ethical
assumptions should be written clearly, tested, versioned, and open to review.

Useful contributions include:

- new measures with hand-computed tests and explicit failure modes;
- evaluation profiles with declared values, thresholds, and uncertainty;
- integration tests for additional model families;
- probe-design and tokenisation checks;
- adversarial examples and documented failure cases;
- visualisations that preserve uncertainty and per-seed variation;
- records showing where checkpoints and results came from, and how related
  models are connected;
- controls showing when a measure should remain unchanged;
- clearer explanations and reproducible examples.

Contributions may challenge the current interpretation. Their assumptions
should be testable. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the rules that apply when a change
affects the meaning of a measurement, and for the local test commands.

## Repository map

~~~text
era/
  metrics.py          general drift measures
  pipeline.py         screening and aggregation
  report.py           evidence files and configuration hashes
  models.py           model loading and hidden-state access
  reference_metrics.py    strict probability, group, SI, and geometry formulas
  reference_probe.py      tokenisation and fixed-probe inference
experiments/          reproducible runners and analysis scripts
tests/                hand-computed, regression, and integration tests
data/                 reference corpora and probe declarations; see data/README.md
results/              committed evidence, never model checkpoints
docs/                 reading guide, methods, results, history, and runbooks
~~~

The documentation reading order is in [docs/README.md](docs/README.md).

## Release status

The repository is being prepared as the first public software release,
v1.0.0. The current package version is v1.0.0rc2. The final tag will be
created only after the documented quickstart, unit tests, integration tests,
and release metadata have been reviewed.

The package starts at version 1.0. Historical documents sometimes call the
current measurement design “ERA v2”; that name describes the second version
of the research method, not the software release.

## Acknowledgements

The maintainer thanks Pietro Mercuri for his valuable advice and helpful
discussions.

## Citation and license

Citation metadata is available in [CITATION.cff](CITATION.cff).

ERA Screening is licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
