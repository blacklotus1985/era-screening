"""
ERA v2 — Screening pipeline
===========================

One entry point: :func:`screen`.  Give it a model pair and a list of probe
contexts; get back per-layer drift curves, per-context rows, and depth
centroids.  No training happens here — ERA v2 is an *audit* of two existing
checkpoints (the research harness that fine-tunes models to create known
test cases lives in ``experiments/``).

For every context the pipeline:

1. takes the top-k next-token distributions of both models (token-ID keyed);
2. computes the output drift between them over the union support (L2 view);
3. builds the candidate set — union of the two top-k sets (exploratory mode)
   or a fixed probe vocabulary (confirmatory mode, recommended for claims);
4. collects per-layer hidden states for each candidate from both models and
   derives, from the same forward passes:
     - relational drift  : mean |Δ cosine| across candidate *pairs*
     - per-token drift   : mean 1 − cos(base_tok, ft_tok)
     - stacked matrices  : accumulated for a global per-layer CKA.

The pipeline depends only on the *interface* of the pair object (duck
typing): anything with ``context_ids``, ``next_token_distribution`` and
``layer_states`` works — which is how the unit tests exercise every line of
this module without torch, a GPU, or a network connection.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from era.metrics import (
    DISTRIBUTION_METRICS,
    cosine_similarity,
    drift_centroid,
    linear_cka,
)

# Identity of the MEASUREMENT ALGORITHM, distinct from package/library
# versions.  Bump this integer whenever a change to metrics.py, pipeline.py
# or models.py alters what the numbers MEAN (a metric fix, a different
# candidate-selection rule, a changed pooling).  The sweep's resume logic
# compares it, so cached cells computed by an older algorithm are invalidated
# exactly when the measurement changed — while harmless refactors and
# dependency bumps leave caches intact.
MEASUREMENT_SCHEMA_VERSION = 1


@dataclass
class ScreeningResult:
    """Everything :func:`screen` measures, ready for reporting.

    Curves are indexed by layer (0 = embedding output).  ``cka`` is a
    *similarity* in [0, 1]; the change score is ``1 - cka``.
    """

    relational_mean: np.ndarray
    relational_std: np.ndarray
    per_token_mean: np.ndarray
    per_token_std: np.ndarray
    cka: np.ndarray
    # Mean pairwise cosine among the candidates of each model, per layer: the
    # anisotropy diagnostic.  Cosine-based drift saturates when this is high
    # (the v1 confound), so every cosine-based number ships with the context
    # needed to judge it.  CKA is less sensitive to the shared mean direction
    # but has its own estimator and sampling limitations (see metrics.py).
    anisotropy_base: np.ndarray = field(default_factory=lambda: np.array([]))
    anisotropy_ft: np.ndarray = field(default_factory=lambda: np.array([]))
    per_context: List[Dict] = field(default_factory=list)
    config: Dict = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return len(self.relational_mean)

    @property
    def centroids(self) -> Dict[str, float]:
        """Depth centre-of-mass of each change curve (descriptive summary)."""
        return {
            "relational": drift_centroid(self.relational_mean),
            "per_token": drift_centroid(self.per_token_mean),
            "cka_change": drift_centroid(1.0 - self.cka),
        }


def output_drift(
    p_dist: Dict[int, float],
    q_dist: Dict[int, float],
    metric: str,
) -> float:
    """Distribution drift between two top-k dicts over their union support.

    The named metric (``k_divergence`` or ``js_divergence``) already
    normalises over the union internally; this thin wrapper exists so the
    metric choice is validated in exactly one place.
    """
    try:
        fn = DISTRIBUTION_METRICS[metric]
    except KeyError:
        raise ValueError(
            f"Unknown distribution metric {metric!r}. "
            f"Supported: {sorted(DISTRIBUTION_METRICS)}."
        )
    return fn(p_dist, q_dist)


def screen(
    pair,
    contexts: Sequence[str],
    top_k: int = 20,
    distribution_metric: str = "k_divergence",
    probe_vocab: Optional[Sequence[str]] = None,
    context_family: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> ScreeningResult:
    """Run the full screening over ``contexts`` and aggregate per layer.

    Parameters
    ----------
    pair
        A :class:`era.models.ModelPair` (or any object with the same three
        methods — see module docstring).
    contexts
        Probe contexts; each contributes one row and one set of curves.
    top_k
        Size of each model's next-token distribution.
    distribution_metric
        ``"k_divergence"`` (directional, the published default) or
        ``"js_divergence"`` (symmetric).
    probe_vocab
        Optional fixed list of single-token words.  When given, the candidate
        set is identical for every context and independent of the models
        being compared (confirmatory mode — use this for claims).  When
        omitted, candidates are the union of the two top-k sets (exploratory
        mode; the set then depends on the models themselves, a caveat the
        report records).
    context_family
        Optional mapping context -> family label (e.g. "leadership") copied
        into the per-context rows.
    verbose
        Print one progress line per context.

    Raises
    ------
    ValueError
        On an unknown metric, an empty context list, or a probe word that is
        not a single token.
    """
    if not contexts:
        raise ValueError("screen() needs at least one probe context.")
    # bool is a subclass of int in Python: exclude it explicitly, otherwise
    # top_k=True would slip through as 1.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(
            f"top_k must be a positive integer, got {top_k!r}. "
            "(Values above the vocabulary size are clamped by the model.)"
        )
    output_drift({0: 1.0}, {0: 1.0}, distribution_metric)  # validate metric early

    fixed_ids: Optional[List[int]] = None
    if probe_vocab is not None:
        fixed_ids = [pair.encode_single_token(w) for w in probe_vocab]
        # Duplicate IDs would double-count rows in every metric (two probe
        # words can map to the same token, or the file may repeat a word).
        # Confirmatory mode must be strict: fail, name the culprits.
        seen: Dict[int, str] = {}
        duplicates = []
        for word, token_id in zip(probe_vocab, fixed_ids):
            if token_id in seen:
                duplicates.append(f"{word!r} and {seen[token_id]!r} -> id {token_id}")
            else:
                seen[token_id] = word
        if duplicates:
            raise ValueError(
                "Probe vocabulary contains duplicate token IDs: " + "; ".join(duplicates)
            )

    per_context: List[Dict] = []
    relational_rows: List[np.ndarray] = []
    per_token_rows: List[np.ndarray] = []
    aniso_base_rows: List[np.ndarray] = []
    aniso_ft_rows: List[np.ndarray] = []
    base_stacks: Optional[List[List[np.ndarray]]] = None  # per layer
    ft_stacks: Optional[List[List[np.ndarray]]] = None
    num_layers: Optional[int] = None

    for idx, context in enumerate(contexts, start=1):
        ctx_ids = pair.context_ids(context)

        p_base = pair.next_token_distribution("base", ctx_ids, top_k=top_k)
        p_ft = pair.next_token_distribution("finetuned", ctx_ids, top_k=top_k)
        l2_value = output_drift(p_base, p_ft, distribution_metric)

        candidates = fixed_ids if fixed_ids is not None else sorted(set(p_base) | set(p_ft))
        if len(candidates) < 2:
            continue  # nothing pairwise to measure for this context

        # One forward pass per (model, candidate); every layer metric below
        # is derived from these same hidden states.
        states_base = {c: pair.layer_states("base", ctx_ids, c) for c in candidates}
        states_ft = {c: pair.layer_states("finetuned", ctx_ids, c) for c in candidates}

        if num_layers is None:
            num_layers = len(next(iter(states_base.values())))
            base_stacks = [[] for _ in range(num_layers)]
            ft_stacks = [[] for _ in range(num_layers)]
        # From here on the three variables above are always set (first
        # iteration initialises them); the asserts state that for the reader
        # and for the type checker alike.
        assert base_stacks is not None and ft_stacks is not None

        # Fail diagnostically, not deep inside NumPy: every candidate, both
        # models, every context must agree on the layer count.
        for label, states in (("base", states_base), ("finetuned", states_ft)):
            for c, layers_list in states.items():
                if len(layers_list) != num_layers:
                    raise ValueError(
                        f"Layer-count mismatch: {label} model returned "
                        f"{len(layers_list)} layers for candidate {c} in context "
                        f"{context!r}, expected {num_layers}. The two checkpoints "
                        "are not comparable layer-by-layer."
                    )

        relational = np.zeros(num_layers)
        per_token = np.zeros(num_layers)
        aniso_base = np.zeros(num_layers)
        aniso_ft = np.zeros(num_layers)
        n = len(candidates)

        for layer in range(num_layers):
            # Per-token drift: how far each candidate's own vector moved.
            moved = [
                1.0 - cosine_similarity(states_base[c][layer], states_ft[c][layer])
                for c in candidates
            ]
            per_token[layer] = float(np.mean(moved))

            # Relational drift: how the geometry BETWEEN candidates changed.
            # The same pairwise cosines also give the anisotropy diagnostic
            # (mean pairwise cosine per model): when it is close to 1, the
            # cosine-based drift numbers at this layer are saturated and depth
            # claims should lean on CKA (less sensitive to the shared mean
            # direction, though itself not free of estimator bias).
            deltas, sims_base, sims_ft = [], [], []
            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = candidates[i], candidates[j]
                    sim_base = cosine_similarity(states_base[ci][layer], states_base[cj][layer])
                    sim_ft = cosine_similarity(states_ft[ci][layer], states_ft[cj][layer])
                    deltas.append(abs(sim_ft - sim_base))
                    sims_base.append(sim_base)
                    sims_ft.append(sim_ft)
            relational[layer] = float(np.mean(deltas))
            aniso_base[layer] = float(np.mean(sims_base))
            aniso_ft[layer] = float(np.mean(sims_ft))

            # Stacks for the global per-layer CKA.
            base_stacks[layer].append(np.vstack([states_base[c][layer] for c in candidates]))
            ft_stacks[layer].append(np.vstack([states_ft[c][layer] for c in candidates]))

        row = {
            "context": context,
            "family": (context_family or {}).get(context, ""),
            "n_candidates": n,
            "n_pairs": n * (n - 1) // 2,
            "l2": l2_value,
            # Top-k coverage: the drift above measures the SHAPE of the
            # retained mass; these two numbers say how much mass that is.
            "base_topk_mass": float(sum(p_base.values())),
            "ft_topk_mass": float(sum(p_ft.values())),
        }
        for layer in range(num_layers):
            row[f"l3_layer_{layer}"] = float(relational[layer])
            row[f"pertok_layer_{layer}"] = float(per_token[layer])
        per_context.append(row)
        relational_rows.append(relational)
        per_token_rows.append(per_token)
        aniso_base_rows.append(aniso_base)
        aniso_ft_rows.append(aniso_ft)

        if verbose:
            print(
                f"  [{idx:2d}/{len(contexts)}] {context[:40]!r:44} "
                f"L2={l2_value:.4f} rel_max={relational.max():.4f} "
                f"ptok_max={per_token.max():.4f}"
            )

    if not per_context:
        raise ValueError(
            "No context produced at least two candidate tokens; "
            "nothing to aggregate."
        )
    assert num_layers is not None and base_stacks is not None and ft_stacks is not None

    cka = np.array([
        linear_cka(np.vstack(base_stacks[layer]), np.vstack(ft_stacks[layer]))
        for layer in range(num_layers)
    ])

    rel = np.vstack(relational_rows)
    ptk = np.vstack(per_token_rows)

    return ScreeningResult(
        relational_mean=rel.mean(axis=0),
        relational_std=rel.std(axis=0),
        per_token_mean=ptk.mean(axis=0),
        per_token_std=ptk.std(axis=0),
        cka=cka,
        anisotropy_base=np.vstack(aniso_base_rows).mean(axis=0),
        anisotropy_ft=np.vstack(aniso_ft_rows).mean(axis=0),
        per_context=per_context,
        config={
            "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
            "top_k": top_k,
            "distribution_metric": distribution_metric,
            "candidate_mode": "fixed_probe_vocab" if fixed_ids is not None else "topk_union",
            "n_contexts_requested": len(contexts),
            "n_contexts_used": len(per_context),
            "num_layers": int(num_layers),
            # Rows in each per-layer CKA stack: the sample size the CKA
            # estimator saw.  metrics.linear_cka documents its bias in
            # high-dimension / low-sample regimes — this is the number a
            # reader needs to judge that (promised by that docstring).
            "n_cka_samples": int(sum(r["n_candidates"] for r in per_context)),
            # Content hashes: the fingerprint must change when the *content*
            # of the probes changes, not only when a path or a knob does.
            # Canonical JSON, not "\n".join: joining would collide
            # ["a", "b"] with ["a\nb"] — JSON preserves element boundaries,
            # order and count.
            "contexts_sha256": hashlib.sha256(
                json.dumps(list(contexts), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "probe_vocab_ids": list(fixed_ids) if fixed_ids is not None else None,
        },
    )
