# ERA documentation

Start with the main [`README`](../README.md). It explains the purpose of ERA,
the current evidence, the quick start, and the longer-term direction.

For a short introduction, read the [project overview PDF](ERA_overview_presentation.pdf).
The [PowerPoint source](ERA_overview_presentation.pptx) is also available.

The documents below form the main reading path.

1. [`RESULTS.md`](RESULTS.md) gives the conclusion and table from the current
   33-cell study.
2. [`REFERENCE_METRICS.md`](REFERENCE_METRICS.md) defines the measurements and their
   within-run aggregation.
3. [`MEASUREMENT_PROTOCOL.md`](MEASUREMENT_PROTOCOL.md) explains the
   numerical rules, token handling, result structure, and aggregation across
   runs.
4. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) covers checks ranging from the
   committed JSON files to complete retraining.
5. [`DECK_BUILD.md`](DECK_BUILD.md) documents the final presentation, its
   chart data, and the manual chart-to-image step.
6. [`ROADMAP.md`](ROADMAP.md) presents open directions for evaluation
   profiles, larger models, adversarial testing, and model genealogy.
7. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) explains how to propose and test
   a contribution.

## Focused experiment notes

- [`POC2_VS_FULL.md`](POC2_VS_FULL.md) compares partial and full fine-tuning on
  one paired model-and-seed design.
- [`../data/README.md`](../data/README.md) explains the training corpora,
  prompts, and concept words.
- [`../results/README.md`](../results/README.md) explains the numerical files
  and how to inspect an individual cell.

## Technical and historical material

[`HISTORY.md`](HISTORY.md) records how interpretations and protocols changed.
The extended geometry study has a preserved
[`preregistration`](PREDICTIONS.md) and a hypothesis-by-hypothesis
[`result report`](FINDINGS_extended.md). Earlier findings documents preserve
the analyses from their own phases of the project.
[`RUNBOOK_GPU.md`](RUNBOOK_GPU.md) contains the full GPU procedure.
[`SATURATION_LEMMA.md`](SATURATION_LEMMA.md) provides a mathematical note on
cosine saturation.

These files support review and reproduction. The reading path above is enough
to understand the current public release.
