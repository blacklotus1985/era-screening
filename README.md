# ERA Screening

[![CI](https://github.com/blacklotus1985/era-screening/actions/workflows/ci.yml/badge.svg)](https://github.com/blacklotus1985/era-screening/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/status-v1.0_release_candidate-orange.svg)](CHANGELOG.md)

ERA is an open research project for studying how fine-tuning changes language
models.

I started ERA to understand those changes by looking at both a model's
outputs and its internal representations: the numerical patterns it produces
while processing an input. The current library compares a model with a
fine-tuned version. You choose the prompts and words to examine; ERA measures
changes in next-token probabilities and internal representations, and saves
the inputs and settings behind the comparison. The repository also includes
a reproducible study across 11 small models.

My longer-term goal is to develop ERA into a system for independent,
third-party white-box auditing, where access to model weights and internal
activations allows people outside the training team to examine what changed.
I want this work to contribute another layer to safety evaluations, in the
Swiss cheese sense. I would like to build it through shared experiments and
contributions from people working on model evaluation and AI safety.

## A first result: similar output change, different association scores

In the reference study, Pythia-70M and OPT-350M were fine-tuned on the same
material linking leadership roles to men and support roles to women. Their
predicted probabilities changed by a similar overall amount. On separate
evaluation prompts, however, the association score within 14 selected male
and female words decreased in Pythia-70M and increased in OPT-350M. Each
direction held across all three training runs, which used different random
seeds.

| Model | Overall probability change (B, nats) | Change in association within the selected words (Delta SI) |
|---|---:|---:|
| Pythia-70M | 0.582 ± 0.004 | -0.207 ± 0.042 |
| OPT-350M | 0.561 ± 0.020 | +1.146 ± 0.023 |

Values are means ± one sample standard deviation across three runs.
B compares next-token probabilities across the full vocabulary on the test
prompts; it ranges from zero to about 0.693 nats. Delta SI measures how the
male-versus-female balance differs between leadership and support prompts,
after rescaling probabilities to sum to one within the 14 words. Positive
values mean that this relative association became stronger.

The relative balance inside a word list can change differently from its
absolute probabilities. For Pythia-70M, the same contrast calculated from
absolute word probabilities increases in all three runs. Its lower
conditional score therefore does not mean the model is generally less biased.
The [saved results](results/reference_metrics/v2_balanced_r2) contain both
versions of the index; the [protocol](docs/MEASUREMENT_PROTOCOL.md) explains
their fields.

The split between and within the word groups adds another detail: 17.2% of
Pythia-70M's target-word change is between groups, compared with 52.9% for
OPT-350M. These are ratios of the aggregated components, following the
[aggregation rules](docs/MEASUREMENT_PROTOCOL.md#aggregation-across-runs).
This application of the KL decomposition makes the structure of the
probability change easier to see than a single overall score. The
[full results](docs/RESULTS.md) also examine internal representations
separately; the example here concerns probabilities and the selected-word
association.

## Explore ERA

- **Start with the idea:** read the
  [short project overview (PDF)](docs/ERA_overview_presentation.pdf).
- **Explore the existing results:** the [reference study](docs/RESULTS.md)
  walks through the experiment and its 11-model results. The
  [saved measurements](results/reference_metrics/v2_balanced_r2) are available
  to inspect without training or downloading models.
- **Compare your own models:** follow the [quick start](#quick-start) to
  install ERA and measure a compatible pair using your own test inputs.

## Contributing

There are several ways to take part, depending on what you want to work on:

- Reproduce a result or try the comparison on other prompts and models.
- Compare a measure with a simple baseline or an existing method.
- Suggest counterexamples, controls or a better interpretation of a result.
- Improve runnable examples, documentation, tests or visualisations.
- Propose integrations with existing tools so the work can be reused.
- Help design tests for a specific concern, including the social and ethical
  questions that determine what should be evaluated.

If you have a question or an experiment in mind,
[open an issue](https://github.com/blacklotus1985/era-screening/issues).
I would be glad to discuss it, including results that challenge the current
approach. ERA can develop through its own experiments and work with existing
libraries. The [contribution guide](CONTRIBUTING.md) explains how to propose
larger changes and what to check when a contribution affects a measurement.

## Quick start

### 1. Get the repository and check the saved results

This local check verifies the consistency of the committed JSON results and
confirms that the readable summary matches all 33 comparisons. It does not
load models or run a new comparison.

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

A checkpoint is a saved version of a model. Supply a base model and a version
fine-tuned from it; ERA compares the two existing versions without training
them. Replace the example paths below with your model locations and input
files.

A probe is the set of test prompts and concept words chosen for a comparison.
Use prompts relevant to the change you want to investigate. To test a specific
claim, fix the vocabulary in advance and use words represented as one token
by the selected tokenizer.

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

## What ERA measures today

The general screening pipeline in [era/pipeline.py](era/pipeline.py), used
by `screen()` and the quick-start command, reports changes under your chosen
probe:

| View | Question |
|---|---|
| Output probability | How much did next-token probabilities change over the union of the two top-k token sets? |
| Per-concept change | How much did the representation of each chosen concept change by layer? |
| Concept relationships | Did the pairwise angles between concept representations change? |
| Internal geometry (linear CKA) | How similar is the arrangement of the probe representations at each layer? |
| Shared-direction check (anisotropy) | Do many representations point in one dominant direction, making angle-based comparisons harder to interpret? |

For cosine-based relational drift, the
[`saturation lemma`](docs/SATURATION_LEMMA.md) gives a certified per-layer
ceiling from the base and fine-tuned anisotropy values already stored in an
ERA report.

The reference study uses a fixed protocol, with formulas in
[era/reference_metrics.py](era/reference_metrics.py), inference and aggregation
in [era/reference_probe.py](era/reference_probe.py), and a
[dedicated experiment runner](experiments/20_reference_metrics_panel.py).
These produce the following study measurements separately from `screen()`:

| Quantity | Question |
|---|---|
| B, B-alpha, B-k | How much did output probability change across the full vocabulary, a high-probability top-p set, or the top-k tokens? |
| B-T between/within | Did probability assigned to the target words move between the two declared groups or within each group? |
| Conditional Delta SI | Did the relative association within the selected word set strengthen on the evaluation prompts? |
| G-l and 1-G-l | How similar are the internal representations of the concepts at each layer? |

The mathematical definitions are in
[docs/REFERENCE_METRICS.md](docs/REFERENCE_METRICS.md). The numerical rules, token
handling, and result structure are in
[docs/MEASUREMENT_PROTOCOL.md](docs/MEASUREMENT_PROTOCOL.md). ERA reports
these quantities separately so that each part of the change remains visible.

## Results across the reference panel

The study covers 11 models from six families, with three training runs per
model: 33 comparisons on one intentionally stereotyped corpus. Conditional
Delta SI is positive in 29/33 runs (87.9%) and on average in 10/11 models.
The absolute-probability diagnostic is positive in all 33 runs. The signs
differ for Pythia-70M at seeds 42, 43 and 44, and Pythia-410M at seed 43.

These counts come from `metrics.SI.conditional.delta_SI` and
`metrics.SI.raw_probability_mass_diagnostic.delta_SI` in the
[per-run JSON files](results/reference_metrics/v2_balanced_r2). Both describe
the same prompts and word groups, with different treatment of their total
probability mass.

The [complete results](docs/RESULTS.md) also separate probability changes
between the male and female word groups from changes within each group, and
examine internal representations under a fixed probe. Internal-change scores
depend on each architecture's geometry, which limits comparisons across
models.

On the evaluation prompts, the mean probability assigned to the 14 target
words rises from 1.6–4.8% before fine-tuning to 24.7–72.3% afterwards
(ranges across model means). The study did not measure perplexity on
unrelated text, so it does not establish how general language quality changed.
The result summary explains this concentration, the calibration controls and
the scope of the original preregistration.

## Related work

Model diffing studies differences between models, including what changes
when a base model is fine-tuned. ERA overlaps with this work and with research
on how to compare internal representations. These are useful references for
testing its methods and developing further experiments.

| Work | Connection to ERA |
|---|---|
| [Diffing Toolkit](https://github.com/science-of-finetuning/diffing-toolkit) and [Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](https://arxiv.org/abs/2510.13900) | The toolkit compares base and fine-tuned models through output and activation differences. The paper examines traces of narrow training in activation differences. These offer methods and cases to compare with ERA's measurements. |
| [Bias Similarity Measurement](https://arxiv.org/abs/2410.12010) | Compares bias and behaviour across models, including CKA on models whose internal activations are accessible. It shares ERA's interest in relating behavioural and internal comparisons, with a different evaluation protocol. |
| [ReSi](https://arxiv.org/abs/2408.00531) and [code](https://github.com/mklabunde/resi) | Benchmarks representation-similarity measures across tests, models and domains. It provides alternatives and test designs for examining ERA's internal measures. |
| [Grounding Representation Similarity with Statistical Testing](https://arxiv.org/abs/2108.01661) | Tests whether similarity measures detect changes that affect model function and remain stable under changes that do not. This is a useful basis for checking what ERA's layer curves mean. |
| [Cross-Architecture Model Diffing with Crosscoders](https://arxiv.org/abs/2602.11729) | Learns shared and model-specific features across different architectures. It explores a broader comparison setting than ERA's current compatible-pair pipeline. |

## Saved comparison files

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

## What I would like to investigate next

I want to understand when ERA's measurements pick up changes that matter in
behavioural evaluations, and how they compare with existing model-diffing
methods. Useful experiments would run the methods on the same model pairs,
test on separate prompts, and include controls where the expected change is
known. The [open directions](docs/ROADMAP.md) describe other possibilities.

Shared, versioned evaluation profiles could eventually connect these
measurements to assessments of particular uses of a model. Developing a
profile would involve defining the outcomes to assess, checking them in
behavioural tests, and setting criteria with estimates of uncertainty and a
clear scope of application.

### Following changes through a model's descendants

Today ERA compares one related pair. I would like to connect those records
to follow where a measured change appears and whether it persists through
later training. Each step would keep the model versions, available training
history, inputs and results together. Extending this to merges or distillation
would require support beyond the current compatible-pair pipeline.

[PhyloLM](https://arxiv.org/abs/2404.04671) approaches model relationships by
inferring a family tree from output similarity. The genealogy envisaged here
would instead follow documented checkpoint provenance and attach measurements
to each recorded transformation, keeping inferred relationships separate
from the available training records.

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
v1.0.0. The current package version is v1.0.0rc3. The final tag will be
created only after the documented quickstart, unit tests, integration tests,
and release metadata have been reviewed.

The package starts at version 1.0. Historical documents sometimes call the
current measurement design “ERA v2”; that name describes the second version
of the research method, not the software release.

## Contributors

- **Pietro Mercuri** — provided suggestions on experiments and evaluation
  metrics.

## Acknowledgements

Thanks to **Luca Francesco San Mauro**.

## Citation and license

Citation metadata is available in [CITATION.cff](CITATION.cff).

ERA Screening is licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
