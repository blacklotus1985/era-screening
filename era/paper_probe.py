"""Inference and aggregation for the fixed metrics of the ERA paper.

The mathematical functions live in :mod:`era.paper_metrics`.  This module
does the model-facing work: strict probe tokenisation, stable next-token
softmax, batched contextual hidden-state extraction, and one auditable record
for a base/fine-tuned pair.  It contains no training code.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from era.paper_metrics import (
    B,
    B_T,
    B_alpha,
    B_k,
    G_l,
    delta_si,
    group_gap,
    stereotype_index,
)

PAPER_PROBE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PaperProbeSpec:
    """Human-readable fixed supports and measurement parameters."""

    name: str
    target_groups: Dict[str, Tuple[str, ...]]
    concept_tokens: Tuple[str, ...]
    top_k: int
    top_p: float
    log_base: Optional[float]
    source_sha256: str


@dataclass(frozen=True)
class EncodedPaperProbe:
    """One tokenizer's IDs for the same fixed human-readable events."""

    target_groups: Dict[str, Tuple[int, ...]]
    concept_ids: Tuple[int, ...]
    token_id_by_text: Dict[str, int]


@dataclass(frozen=True)
class ModelObservations:
    """Sufficient statistics from one model on the fixed probe.

    ``probabilities`` has shape ``(contexts, vocabulary)``. ``states`` keeps
    one ``(contexts, concepts, hidden)`` array per layer, so every layer stays
    in its native width without padding or an invented projection.
    """

    probabilities: np.ndarray
    states: Tuple[np.ndarray, ...]


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_paper_probe_spec(path) -> PaperProbeSpec:
    """Read and validate a versioned paper-probe JSON file."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PAPER_PROBE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported paper probe schema: "
            f"{payload.get('schema_version')!r}; expected "
            f"{PAPER_PROBE_SCHEMA_VERSION}."
        )
    groups_raw = payload.get("target_groups")
    if not isinstance(groups_raw, dict) or set(groups_raw) != {"male", "female"}:
        raise ValueError("target_groups must contain exactly 'male' and 'female'.")
    target_groups: Dict[str, Tuple[str, ...]] = {}
    seen_text = set()
    for name, tokens in groups_raw.items():
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"target group {name!r} must be a non-empty list.")
        group = tuple(tokens)
        if any(not isinstance(token, str) or not token for token in group):
            raise ValueError(f"target group {name!r} contains an invalid token.")
        if len(set(group)) != len(group):
            raise ValueError(f"target group {name!r} contains duplicate strings.")
        overlap = seen_text.intersection(group)
        if overlap:
            raise ValueError(f"target groups overlap on {sorted(overlap)}.")
        seen_text.update(group)
        target_groups[name] = group

    concepts_raw = payload.get("concept_tokens")
    if not isinstance(concepts_raw, list) or not concepts_raw:
        raise ValueError("concept_tokens must be a non-empty list.")
    concepts = tuple(concepts_raw)
    if any(not isinstance(token, str) or not token for token in concepts):
        raise ValueError("concept_tokens contains an invalid token.")
    if len(set(concepts)) != len(concepts):
        raise ValueError("concept_tokens contains duplicate strings.")

    top_k = payload.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    top_p = float(payload.get("top_p"))
    if not np.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    log_base = payload.get("log_base")
    if log_base is not None:
        log_base = float(log_base)
        if not np.isfinite(log_base) or log_base <= 1.0:
            raise ValueError("log_base must be null (nats) or greater than 1.")

    return PaperProbeSpec(
        name=str(payload.get("name") or source.stem),
        target_groups=target_groups,
        concept_tokens=concepts,
        top_k=top_k,
        top_p=top_p,
        log_base=log_base,
        source_sha256=_canonical_sha256(payload),
    )


def encode_paper_probe(tokenizer, spec: PaperProbeSpec) -> EncodedPaperProbe:
    """Encode every declared event and fail if any is not one unique token.

    No alternative casing, whitespace variant or per-model subset is tried.
    That strictness is what makes the 11-model table compare the same semantic
    events rather than eleven tokenizer-dependent probes.
    """
    ordered_text = [
        token for group in ("male", "female") for token in spec.target_groups[group]
    ] + list(spec.concept_tokens)
    token_id_by_text: Dict[str, int] = {}
    for token_text in ordered_text:
        ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"Paper probe event {token_text!r} maps to {len(ids)} tokens "
                f"{ids}; the fixed-support comparison cannot proceed."
            )
        token_id_by_text[token_text] = int(ids[0])

    target_groups = {
        name: tuple(token_id_by_text[text] for text in texts)
        for name, texts in spec.target_groups.items()
    }
    target_ids = target_groups["male"] + target_groups["female"]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Distinct target strings map to overlapping token IDs.")
    concept_ids = tuple(token_id_by_text[text] for text in spec.concept_tokens)
    if len(set(concept_ids)) != len(concept_ids):
        raise ValueError("Distinct concept strings map to overlapping token IDs.")
    return EncodedPaperProbe(target_groups, concept_ids, token_id_by_text)


def stable_softmax(logits: Sequence[float]) -> np.ndarray:
    """Float64 softmax without clipping, smoothing or epsilon insertion."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits contain NaN or infinite values.")
    shifted = values - float(values.max())
    exponentials = np.exp(shifted)
    denominator = float(math.fsum(float(value) for value in exponentials))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ArithmeticError("softmax denominator is not finite and positive.")
    probabilities = exponentials / denominator
    if not np.all(np.isfinite(probabilities)):
        raise ArithmeticError("softmax produced a non-finite probability.")
    return probabilities


def measure_model(
    model,
    tokenizer,
    contexts: Sequence[str],
    concept_ids: Sequence[int],
    device: str,
) -> ModelObservations:
    """Run one model on the probability and geometry probe.

    Candidate states are batched by context: all sequences in one batch have
    identical length, so no padding convention can move the measured final
    position. The sequence follows the model's ``output_hidden_states``
    contract, and every entry is retained at its native width.
    """
    import torch

    if not contexts:
        raise ValueError("at least one context is required.")
    if not concept_ids:
        raise ValueError("at least one concept token is required.")
    probabilities = []
    states = []
    model.eval()
    for context in contexts:
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        if not context_ids:
            raise ValueError(f"context tokenises to an empty sequence: {context!r}.")
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0, -1, :]
        probabilities.append(stable_softmax(logits.detach().cpu().numpy()))

        candidate_batch = torch.tensor(
            [context_ids + [int(token_id)] for token_id in concept_ids],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            output = model(
                input_ids=candidate_batch,
                output_hidden_states=True,
                use_cache=False,
            )
        # hidden_states is layers × batch × sequence × hidden.  The fixed
        # candidate is the final token by direct ID construction.
        per_layer = [
            hidden[:, -1, :].detach().cpu().numpy() for hidden in output.hidden_states
        ]
        states.append(tuple(per_layer))

    probability_array = np.stack(probabilities)
    layer_count = len(states[0])
    if any(len(context_states) != layer_count for context_states in states):
        raise ValueError("model returned a different number of layers by context.")
    state_layers = tuple(
        np.stack([context_states[index] for context_states in states])
        for index in range(layer_count)
    )
    if probability_array.shape[1] != model.get_input_embeddings().num_embeddings:
        raise ValueError("logit vocabulary and input embedding vocabulary differ.")
    return ModelObservations(probability_array, state_layers)


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("aggregate values must be a non-empty finite vector.")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _restricted_record(result, *, include_support: bool) -> Dict:
    record = {
        "value": float(result["value"]),
        "support_size": len(result["support"]),
        "support_sha256": _canonical_sha256(list(result["support"])),
        "mass_m0": float(result["mass_m0"]),
        "mass_m1": float(result["mass_m1"]),
    }
    if include_support:
        record["support_token_ids"] = list(result["support"])
    return record


def evaluate_observations(
    base: ModelObservations,
    finetuned: ModelObservations,
    contexts: Sequence[str],
    families: Sequence[str],
    spec: PaperProbeSpec,
    encoded: EncodedPaperProbe,
) -> Dict:
    """Compute every paper quantity and its per-context audit trail."""
    p_all = np.asarray(base.probabilities, dtype=np.float64)
    q_all = np.asarray(finetuned.probabilities, dtype=np.float64)
    if p_all.shape != q_all.shape or p_all.ndim != 2:
        raise ValueError(
            "base/fine-tuned probabilities must share shape (contexts, vocabulary)."
        )
    if p_all.shape[0] != len(contexts) or len(contexts) != len(families):
        raise ValueError("contexts, families and observation rows must align.")
    centroid = G_l(base.states, finetuned.states)
    state_grid = (centroid["n_contexts"], centroid["n_concepts"])
    if state_grid != (len(contexts), len(encoded.concept_ids)):
        raise ValueError("states do not align with contexts/concept IDs.")

    partition_ids = encoded.target_groups
    target_ids = partition_ids["male"] + partition_ids["female"]
    per_context = []
    base_conditional_gaps = []
    tuned_conditional_gaps = []
    base_raw_gaps = []
    tuned_raw_gaps = []

    for index, (context, family) in enumerate(zip(contexts, families)):
        p = p_all[index]
        q = q_all[index]
        if p.shape != q.shape:
            raise ValueError(f"vocabulary mismatch at context {index}.")
        if max(target_ids + encoded.concept_ids) >= p.size:
            raise ValueError("probe token ID exceeds the model vocabulary.")

        b_full = B(p, q, spec.log_base)
        b_k = B_k(p, q, spec.top_k, spec.log_base)
        b_alpha = B_alpha(p, q, spec.top_p, spec.log_base)
        decomposition = B_T(p, q, partition_ids, spec.log_base)

        base_conditional = group_gap(
            p,
            partition_ids["male"],
            partition_ids["female"],
            conditional=True,
        )
        tuned_conditional = group_gap(
            q,
            partition_ids["male"],
            partition_ids["female"],
            conditional=True,
        )
        base_raw = group_gap(
            p,
            partition_ids["male"],
            partition_ids["female"],
            conditional=False,
        )
        tuned_raw = group_gap(
            q,
            partition_ids["male"],
            partition_ids["female"],
            conditional=False,
        )
        base_conditional_gaps.append(base_conditional)
        tuned_conditional_gaps.append(tuned_conditional)
        base_raw_gaps.append(base_raw)
        tuned_raw_gaps.append(tuned_raw)

        per_context.append(
            {
                "context_index": index,
                "context": context,
                "family": family,
                "B": float(b_full),
                "B_k": _restricted_record(b_k, include_support=True),
                "B_alpha": _restricted_record(b_alpha, include_support=False),
                "B_T": decomposition,
                "group_gap": {
                    "m0_conditional": base_conditional,
                    "m1_conditional": tuned_conditional,
                    "delta_conditional": tuned_conditional - base_conditional,
                    "m0_raw": base_raw,
                    "m1_raw": tuned_raw,
                    "delta_raw": tuned_raw - base_raw,
                },
            }
        )

    b_values = [row["B"] for row in per_context]
    bk_values = [row["B_k"]["value"] for row in per_context]
    ba_values = [row["B_alpha"]["value"] for row in per_context]
    total_values = [row["B_T"]["total"] for row in per_context]
    between_values = [row["B_T"]["between"] for row in per_context]
    within_values = [row["B_T"]["within"] for row in per_context]
    total_mean = float(np.mean(total_values))
    between_mean = float(np.mean(between_values))
    within_mean = float(np.mean(within_values))
    aggregate_residual = total_mean - between_mean - within_mean
    if not np.isclose(total_mean, between_mean + within_mean, rtol=1e-10, atol=1e-12):
        raise ArithmeticError("aggregate B_T decomposition invariant failed.")
    if total_mean == 0.0:
        between_fraction = None
        within_fraction = None
    else:
        between_fraction = between_mean / total_mean
        within_fraction = within_mean / total_mean

    base_si = stereotype_index(base_conditional_gaps, families)
    tuned_si = stereotype_index(tuned_conditional_gaps, families)
    conditional_si = delta_si(base_si, tuned_si)
    base_si_raw = stereotype_index(base_raw_gaps, families)
    tuned_si_raw = stereotype_index(tuned_raw_gaps, families)
    raw_si = delta_si(base_si_raw, tuned_si_raw)

    return {
        "measurement_schema_version": PAPER_PROBE_SCHEMA_VERSION,
        "probe_name": spec.name,
        "probe_sha256": spec.source_sha256,
        "units": "nats" if spec.log_base is None else f"log_base_{spec.log_base:g}",
        "parameters": {"top_k": spec.top_k, "top_p": spec.top_p},
        "n_contexts": len(contexts),
        "n_target_tokens": len(target_ids),
        "n_concept_tokens": len(encoded.concept_ids),
        "aggregates": {
            "B": _mean_std(b_values),
            "B_k": _mean_std(bk_values),
            "B_alpha": _mean_std(ba_values),
            "B_T": {
                "total": _mean_std(total_values),
                "between": _mean_std(between_values),
                "within": _mean_std(within_values),
                "between_fraction_of_mean_total": between_fraction,
                "within_fraction_of_mean_total": within_fraction,
                "decomposition_residual": aggregate_residual,
                "target_mass_m0": _mean_std(
                    [row["B_T"]["target_mass_m0"] for row in per_context]
                ),
                "target_mass_m1": _mean_std(
                    [row["B_T"]["target_mass_m1"] for row in per_context]
                ),
            },
        },
        "SI": {
            "primary": "conditional_on_target_set",
            "conditional": conditional_si,
            "raw_probability_mass_diagnostic": raw_si,
        },
        "G_l": {
            "similarity": [float(value) for value in centroid["similarity"]],
            "drift_1_minus_G": [float(value) for value in centroid["drift"]],
            "per_token_similarity": [
                [float(value) for value in row]
                for row in centroid["per_concept_similarity"]
            ],
            "normalised_depth_centroid_of_drift": (
                centroid["normalised_depth_centroid"]
            ),
        },
        "per_context": per_context,
    }
