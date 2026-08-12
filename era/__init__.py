"""
ERA v2 — lightweight representation-drift screening between related
language-model checkpoints.

Canonical surface (four modules, importable without torch):

    era.metrics    k_divergence, js_divergence, cosine_similarity,
                   linear_cka, drift_centroid
    era.pipeline   screen() -> ScreeningResult
    era.report     save(), config_fingerprint()
    era.contexts   the fixed PoC probe contexts

Model loading lives in ``era.models`` (the only module that imports torch /
transformers) and is imported explicitly by the caller::

    from era.models import ModelPair
    from era import screen, save

    pair = ModelPair("EleutherAI/gpt-neo-125M", "path/to/finetuned")
    result = screen(pair, contexts=era.contexts.TEST_CONTEXTS)
    save(result, "results/my_audit", extra_config={"seed": 42})

ERA screens *where* a fine-tune changed a model; it does not certify
alignment and attaches no automatic deep/shallow verdict.

v1 (ERAAnalyzer, Alignment Score, model genealogy, static-embedding L3) is
preserved in the archived v1 repository for the historical record and is not
part of this codebase; see docs/HISTORY.md for what changed and why.
"""

from era import contexts
from era.metrics import (
    DISTRIBUTION_METRICS,
    cosine_similarity,
    drift_centroid,
    js_divergence,
    k_divergence,
    linear_cka,
)
from era.pipeline import ScreeningResult, output_drift, screen
from era.report import config_fingerprint, save

__version__ = "2.0.0.dev0"

__all__ = [
    "DISTRIBUTION_METRICS",
    "ScreeningResult",
    "config_fingerprint",
    "contexts",
    "cosine_similarity",
    "drift_centroid",
    "js_divergence",
    "k_divergence",
    "linear_cka",
    "output_drift",
    "save",
    "screen",
    "__version__",
]
