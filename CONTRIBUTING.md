# Contributing to ERA

ERA is an open project for building transparent, reproducible white-box
audits of how language models change. It needs code, but it also needs people
who can reproduce results, question interpretations, design controls, review
the mathematics, and connect technical measurements to real safety and
ethical concerns.

Researchers, engineers, auditors, and people working on the social effects of
AI are all welcome. A useful contribution can begin with a result, a question,
a failure case, a clearer explanation, or a small correction.

## What we want to build

ERA aims to make changes between related model checkpoints inspectable. Its
reports connect output probabilities, behaviour in declared contexts, and
internal representations to the exact models and evaluation inputs used.

The longer-term goal is a shared system of reviewable evaluation profiles and
verifiable model lineages. Reaching that goal requires evidence from different
models, interventions, disciplines, and points of view. The current findings
and open directions are described in
[`docs/RESULTS.md`](docs/RESULTS.md) and
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Ways to contribute

- Reproduce an existing result or test it on another model family.
- Challenge a measurement or interpretation with a control or failure case.
- Propose an evaluation profile for a specific safety or ethical concern.
- Improve the code, tests, documentation, examples, or visualisations.
- Review the mathematical definitions or the connection between evidence and
  claims.
- Add reliable information about how related checkpoints were produced.

Results that expose a limitation are valuable contributions. They help define
the conditions under which an ERA measurement can be trusted.

## Before starting a large change

Small corrections and focused improvements can go directly into a pull
request. Before investing in a larger feature or experiment, open an issue and
describe the audit question, the models or inputs involved, and the evidence
the change would produce. This gives contributors a place to compare ideas
before work begins.

## What makes a contribution useful

- A new measurement has a clear meaning, a stated range, and known failure
  cases.
- A scientific claim names the models, inputs, intervention, uncertainty, and
  scope of the evidence.
- New behaviour has a test whose expected result was calculated independently.
- A bug fix has a regression test that demonstrates the previous failure.
- The implementation remains short enough to explain and review directly.
- The documentation gives another person enough information to reproduce the
  result.

The hand-calculated examples in `tests/test_metrics.py` and
`tests/test_reference_metrics.py` show the preferred testing style. The formulas
and measurement rules are documented in
[`docs/REFERENCE_METRICS.md`](docs/REFERENCE_METRICS.md) and
[`docs/MEASUREMENT_PROTOCOL.md`](docs/MEASUREMENT_PROTOCOL.md).

## Changes that affect a reported number

ERA versions its general screening records and fixed-probe measurements separately.
`MEASUREMENT_SCHEMA_VERSION` in `era/pipeline.py` belongs to the general
screening output. `REFERENCE_MEASUREMENT_SCHEMA_VERSION` in `era/reference_probe.py`
belongs to the fixed-probe payload. `REFERENCE_PROBE_SCHEMA_VERSION` versions the
probe-file format, and experiment runners may also version their saved record
structure.

Update the version that governs an affected output when a change alters the
meaning or structure of a reported number. Examples include changes to a
formula, token selection, hidden-state extraction, aggregation, or saved
fields. Code hashes identify implementation changes; schema versions state
which result format and meaning a record follows. Documentation, performance
improvements, and refactors that preserve every result keep the current
versions.

## Run the local checks

For a change to the code or generated results, run the same checks used by
continuous integration:

~~~bash
python -m pip install -e ".[experiments,dev]"
python -m pip check
python experiments/run_screening.py --help
flake8 tests era experiments
mypy --config-file mypy.ini
python -m pytest --cov-fail-under=70
python experiments/23_build_public_summary.py --check
~~~

The mypy command covers the torch-free core listed in `mypy.ini`. Dedicated
unit tests cover the fixed-probe measurement modules.

For a documentation-only change, run `git diff --check` and verify every link
you changed. Run the public-summary check as well when changing
`docs/RESULTS.md` or its generator.

## Optional model-loading test

The normal test suite checks the mathematics and pipeline without downloading
model weights. The integration test requires PyTorch and downloads small
public Hugging Face models. Install the experiment dependencies and the CPU or
CUDA build of PyTorch appropriate for your system before running it.
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) gives the installation
notes and the equivalent PowerShell command.

~~~bash
ERA_RUN_INTEGRATION=1 python -m pytest tests/test_models_integration.py -v --no-cov
~~~

## Write claims that match the evidence

ERA measures changes covered by a declared set of evaluation inputs and
concept words. A reported result should name that set, the checkpoint pair,
the model family, and the repeated runs used.

Each prompt set defines the scope of the measurement and affects the observed
values. CKA and cosine values also depend on the geometry of the model being
tested.
Descriptions such as “shallow” or “deep” therefore need a validated
evaluation profile and supporting controls.

Evaluation profiles should state their use case, outcomes, tests, thresholds,
uncertainty, authorship, and scope. Their measurements and examples remain
available for inspection with any `PASS`, `WARNING`, `FAIL`, or
`REVIEW REQUIRED` result.

Corrections and counterexamples are part of the project. ERA keeps them in
`docs/HISTORY.md` so that later readers can follow how an interpretation
changed.

## License

Contributions submitted for inclusion in ERA use the Apache License 2.0 unless
the contribution explicitly states another arrangement.
