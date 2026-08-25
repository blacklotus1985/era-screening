# Open directions for ERA

ERA is an early open-source research project. It currently measures how a
base model and a related checkpoint differ in output probability, contextual
behaviour, and internal representations.

The directions below are possibilities, not a fixed sequence or release
plan. Priorities will change as people use the repository, reproduce the
results, identify limitations, and contribute new ideas. Decisions will be
explained in the repository as the project develops.

## What is already in place

ERA can compare a base checkpoint with a related descendant, keep probability,
behavioural, and internal measurements separate, and save the models, inputs,
settings, hashes, and software environment behind each result. The reference
evidence currently includes:

- a complete 11-model, 33-cell panel with three seeds per model;
- a preregistered cross-architecture geometry study and its recorded outcomes;
- calibration controls for save/reload neutrality, a neutral corpus, and a
  restricted final-block intervention;
- direct comparison of the two CKA estimators written by the screening
  pipeline;
- mathematical tests built from independently calculated examples.

The [`results`](RESULTS.md),
[`preregistration`](PREDICTIONS.md), and
[`extended findings`](FINDINGS_extended.md) state what these checks support
and where they were inconclusive.

## Validate what the measurements mean

The current layer curves show where a probe records change across a model.
Existing controls also show why the curve cannot be interpreted in isolation.
The neutral-corpus control produced a depth profile close to the stereotyped
corpus, while the restricted final-block control produced a terminal signal
that was useful as a scale endpoint but degenerate as a shallow/deep anchor.

Further tests could compare descendants trained with deliberately different
interventions. For example, one intervention could teach a fixed output
format and another could teach richer counter-stereotypical content. This
would test whether ERA separates the changes in the expected way with controls
that are informative across more than one layer.

## Compare ERA with established evaluations

Models with published results on StereoSet, CrowS-Pairs, BBQ, and other
relevant benchmarks could also be screened with ERA. The comparison would
show how ERA's output and internal measurements relate to established
behavioural evaluations.

A weak relationship would also be useful evidence because it would help
define what ERA's measurements can support.

## Improve measurement reliability

The current panel records biased and unbiased CKA side by side. Their measured
difference in `1-CKA` is small in these 33 cells, between 0.1% and 2.1%, so the
large bias suggested by an earlier synthetic calibration does not explain the
observed depth pattern.

Reliability still depends on anisotropy, effective dimensionality, prompt
choice, and variation between training runs. These factors can be studied
with wider seed panels, prompt-sensitivity tests, and additional
representation comparisons.

The goal is to make the conditions under which each measurement is reliable
clear and testable.

## Expand the models and transformations covered

The current reference panel covers models from 70M to 560M parameters. Future
work may include larger open-weight models and transformations such as
instruction tuning, LoRA adapters, knowledge edits, distillation, and model
merges.

Coverage can grow through a small number of well-documented case studies or
through contributed results that follow the same measurement protocol.

## Stress-test the audit

ERA should also be tested against interventions designed to expose its weak
points. One example would be training a model to change a target behaviour
while keeping the current ERA measurements as stable as possible.

These tests would show which signals remain informative under deliberate
pressure and where new controls are needed.

## Develop evaluation profiles

A future evaluation profile could examine one declared concern in one stated
use context and return `PASS`, `WARNING`, `FAIL`, or `REVIEW REQUIRED`.

Profiles will require validated links between measurements and behaviour,
clear thresholds, uncertainty estimates, versioning, authorship, and review
from people with different technical and social perspectives. The evidence
behind a result should remain available with the profile decision.

## Build a verifiable model genealogy

ERA already records a connection between a base model and its fine-tuned
descendant. These records could eventually form a graph of fine-tunes,
merges, distillations, and later adaptations.

The graph could distinguish lineage information declared by a model publisher
from relationships independently checked through reproducible evidence. It
would also allow one model to have several parents, as happens in model
merges. Before that expansion, the current one-parent checkpoint record needs
to be tested on descendants of descendants, multiple-parent transformations,
and incomplete publisher metadata.

## Community input

ERA is intended to develop with input from people who share its goals or want
to test its assumptions. Reproductions, failure cases, new evaluation ideas,
additional model families, and criticism of the current measurements are all
useful contributions.

The repository's priorities will be updated as that evidence accumulates.
Open-weight models remain the main scope because ERA's measurements require
access to internal activations.
