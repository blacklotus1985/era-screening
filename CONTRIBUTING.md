# Contributing to ERA

ERA is small on purpose: the canonical pipeline is five modules (metrics,
pipeline, report, models, contexts) the author can explain line by line.
Every surviving function must satisfy the three-requirement rule — a test
with a hand-computed expected value, a sentence in the paper that uses it,
and blackboard derivability. The first and third hold today; the paper
sentence is the September rewrite's job (until then the draft in
circulation is still v1 methodology). Changes should preserve, or move
toward, all three.

## Before opening a PR — checklist

1. **Does this change alter what any number MEANS?**
   A metric formula, the candidate-selection rule, hidden-state extraction,
   aggregation, or the report fields the resume compares.
   → If yes, **increment `MEASUREMENT_SCHEMA_VERSION`** in `era/pipeline.py`
   and say so in the PR description. This is what invalidates cached sweep
   cells computed by the old algorithm; forgetting it silently mixes old and
   new results.
   → If no (refactor, docs, performance), do not bump it.

2. **Tests.** New behaviour needs a test with a hand-computed expected value
   (see `tests/test_metrics.py` for the style). Bug fixes need a regression
   test that fails on the old code.

3. **Local CI** must be green before pushing:

   ```bash
   flake8 tests era experiments
   mypy --config-file mypy.ini
   pytest tests/
   ```

4. **Claims.** Code comments and docstrings state what is verified, not what
   is hoped: no deployment labels, no automatic deep/shallow verdicts, CKA
   described as "less sensitive to the shared mean direction", never as
   immune to geometry effects.

5. **Simplicity.** If a change makes the canonical pipeline harder to
   explain at a blackboard, it probably belongs in an experiment script,
   not in `era/`.

## Integration tests

`tests/test_models_integration.py` runs against real models and is skipped
by default. Enable with:

```bash
ERA_RUN_INTEGRATION=1 pytest tests/test_models_integration.py -v
# cheaper models for CI: ERA_INTEGRATION_MODEL=<hub-id>
```

The model-vs-itself screening is the canonical negative control: identical
checkpoints must show ~zero drift and CKA ~1 at every layer.
