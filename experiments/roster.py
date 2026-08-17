#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA — model roster for the extended cross-architecture study
============================================================

Shared, torch-free definition of *which* models the extended sweep covers,
so the preflight census (script 13), the sweep (script 10) and the extended
aggregation (script 14) cannot drift apart on labels, slugs or revisions.

Why a module and not a literal in each script: the slug is a *path* component
(``<multiseed_root>/<tag>/<slug>/seed_<n>/``) and the revision is part of a
cell's identity.  Two scripts disagreeing on either would silently split or
merge cells.

Revision pinning
----------------
The existing sweep pins every model to an **immutable commit SHA**, not to a
branch name: a branch keeps matching a cached training manifest even after it
has moved to different weights, which is exactly the relabeling failure the
manifest exists to prevent.  The roster below therefore ships *no* revisions
for the new models — a SHA cannot be invented offline.  The preflight census
resolves them from the hub and writes ``experiments/roster_pinned.json``;
everything downstream reads that file and refuses to run a model that has no
pinned revision.

The two reference models keep the SHAs already hardcoded in script 10.
"""

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
PINNED_REVISIONS_PATH = Path(__file__).resolve().parent / "roster_pinned.json"

# Slugs become directory names and are compared as cell identity: keep them
# to a conservative, case-sensitive, filesystem-safe alphabet.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Tiers, and the seed panel each one gets.  Tier B is deliberately thinner:
# the models are 3-5x larger and the study is CPU-only, so two seeds buy a
# spread estimate at half the wall-clock.  A two-seed std is a very weak
# estimate and every report must say so.
TIER_SEEDS: Dict[str, List[int]] = {
    "reference": [42, 43, 44],
    "A": [42, 43, 44],
    "B": [42, 43],
}


@dataclass(frozen=True)
class ModelSpec:
    """One model in the panel.

    Attributes
    ----------
    label
        Human-readable name used in figures and tables.
    hf_id
        Hugging Face hub id (or local path) passed to ``from_pretrained``.
    slug
        Short, filesystem-safe identifier; the directory name of every cell.
    tier
        ``"reference"`` (already measured), ``"A"`` or ``"B"``.
    revision
        Immutable commit SHA, or ``None`` until the preflight pins one.
    note
        Why the model is in the panel — carried into the census report so the
        roster explains itself.
    """

    label: str
    hf_id: str
    slug: str
    tier: str = "A"
    revision: Optional[str] = None
    note: str = ""

    def __post_init__(self) -> None:
        for field_name in ("label", "hf_id", "slug"):
            if not getattr(self, field_name):
                raise ValueError(f"ModelSpec.{field_name} must be non-empty: {self!r}")
        if not _SLUG_RE.match(self.slug):
            raise ValueError(
                f"Invalid slug {self.slug!r}: slugs become directory names and "
                "cell identities; use lowercase [a-z0-9_-], starting with a "
                "letter or digit."
            )
        if self.tier not in TIER_SEEDS:
            raise ValueError(
                f"Unknown tier {self.tier!r} for {self.label!r}. "
                f"Known tiers: {sorted(TIER_SEEDS)}."
            )

    @property
    def seeds(self) -> List[int]:
        return list(TIER_SEEDS[self.tier])

    def as_sweep_tuple(self):
        """``(label, hf_id, slug, revision)`` — the 4-tuple script 10 iterates."""
        if self.revision is None:
            raise ValueError(
                f"{self.label} ({self.hf_id}) has no pinned revision. Run "
                "`python experiments/13_preflight_census.py --resolve-revisions` "
                "to pin immutable commit SHAs before sweeping."
            )
        return (self.label, self.hf_id, self.slug, self.revision)


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------
# Reference cells: already measured in the v2_balanced study (see
# docs/FINDINGS_v2_balanced.md).  SHAs copied verbatim from script 10's MODELS
# so both scripts address the identical weights.
REFERENCE_MODELS: List[ModelSpec] = [
    ModelSpec("GPT-Neo-125M", "EleutherAI/gpt-neo-125M", "gptneo", "reference",
              "21def0189f5705e2521767faed922f1f15e7d7db",
              "reference cell: learned positions, local/global attention"),
    ModelSpec("Pythia-160M", "EleutherAI/pythia-160m", "pythia", "reference",
              "50f5173d932e8e61f858120bcb800b97af589f46",
              "reference cell: rotary positions, parallel blocks"),
]

# Tier A — under ~200M parameters, fast enough on CPU for three seeds.
TIER_A_MODELS: List[ModelSpec] = [
    ModelSpec("Pythia-70M", "EleutherAI/pythia-70m", "pythia70m", "A", None,
              "Pythia family size-scaling, low end (6 layers)"),
    ModelSpec("GPT-2-124M", "gpt2", "gpt2", "A", None,
              "learned positions, OpenWebText pretraining data"),
    ModelSpec("OPT-125M", "facebook/opt-125m", "opt125m", "A", None,
              "Meta lineage, learned positions, ReLU activations"),
    ModelSpec("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M", "smollm2135m", "A", None,
              "modern deep-narrow (30 layers), 2024-era pretraining mix"),
]

# Tier B — 350-600M, feasible but slow on CPU; two seeds by default.
TIER_B_MODELS: List[ModelSpec] = [
    ModelSpec("Pythia-410M", "EleutherAI/pythia-410m", "pythia410m", "B", None,
              "Pythia family size-scaling, high end"),
    ModelSpec("GPT-2-medium", "gpt2-medium", "gpt2medium", "B", None,
              "depth 24, same family and data as GPT-2-124M"),
    ModelSpec("OPT-350M", "facebook/opt-350m", "opt350m", "B", None,
              "non-standard embed projection (word_embed_proj_dim != hidden_size) "
              "- preflight must verify hidden-state dimensionality is uniform"),
    ModelSpec("BLOOM-560M", "bigscience/bloom-560m", "bloom560m", "B", None,
              "ALiBi positions, multilingual tokenizer"),
    ModelSpec("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M", "smollm2360m", "B", None,
              "modern deep-narrow, larger"),
]

FULL_ROSTER: List[ModelSpec] = REFERENCE_MODELS + TIER_A_MODELS + TIER_B_MODELS

# Models that were designed into the panel and could NOT be run.  They stay
# here, with the reason and the date checked, rather than being deleted: a
# model quietly dropped from a roster is indistinguishable from a model that
# was never considered, and the panel's coverage claim depends on the
# difference.  The census emits these into skipped_models.json on every run.
UNAVAILABLE_MODELS: List[Dict[str, str]] = [
    {
        "label": "Cerebras-GPT-111M",
        "hf_id": "cerebras/Cerebras-GPT-111M",
        "intended_tier": "A",
        "note": "muP training recipe (different optimisation, same architecture family)",
        "stage": "roster",
        "checked_utc": "2026-08-17",
        "reason": (
            "HTTP 401 from the hub API: the whole Cerebras-GPT series is gated "
            "(cerebras/Cerebras-GPT-111M, -256M and cerebras/btlm-3b-8k-base all "
            "return 401 while the cerebras org itself lists publicly). No "
            "unauthenticated download is possible, and the community re-uploads "
            "are themselves fine-tuned, so they cannot stand in for a base model. "
            "Consequence: the panel carries no muP-trained architecture, and no "
            "claim about training-recipe generality can be made from it."
        ),
    },
]


# ---------------------------------------------------------------------------
# Selection / parsing
# ---------------------------------------------------------------------------

def default_roster() -> List[ModelSpec]:
    """The two already-measured reference models — script 10's historical
    default, unchanged, so a bare ``python experiments/10_multiseed_sweep.py``
    still means exactly what it meant before this module existed."""
    return list(REFERENCE_MODELS)


def select_tiers(tiers: Sequence[str]) -> List[ModelSpec]:
    """Roster entries whose tier is in ``tiers`` (order preserved)."""
    unknown = [t for t in tiers if t not in TIER_SEEDS]
    if unknown:
        raise ValueError(
            f"Unknown tier(s) {unknown}. Known tiers: {sorted(TIER_SEEDS)}."
        )
    wanted = set(tiers)
    return [m for m in FULL_ROSTER if m.tier in wanted]


def parse_model_spec(text: str) -> ModelSpec:
    """Parse a ``Label=hf_id=slug`` or ``Label=hf_id=slug=revision`` triple.

    The optional fourth field pins an immutable commit SHA; without it the
    entry must be pinned by the preflight before it can be swept.
    """
    parts = text.split("=")
    if len(parts) not in (3, 4):
        raise ValueError(
            f"Malformed --models entry {text!r}. Expected "
            "'Label=hf_id=slug' or 'Label=hf_id=slug=revision'."
        )
    label, hf_id, slug = (p.strip() for p in parts[:3])
    revision = parts[3].strip() if len(parts) == 4 and parts[3].strip() else None
    known = {m.hf_id: m for m in FULL_ROSTER}
    tier = known[hf_id].tier if hf_id in known else "A"
    return ModelSpec(label, hf_id, slug, tier, revision)


def parse_models_argument(entries: Iterable[str]) -> List[ModelSpec]:
    """Parse ``--models`` entries and reject duplicate slugs or hub ids.

    A duplicate slug would make two different models share one cell
    directory; a duplicate hub id would run the same weights twice under two
    names.  Both are configuration errors, not something to resolve silently.
    """
    specs = [parse_model_spec(e) for e in entries]
    for field_name in ("slug", "hf_id"):
        seen: Dict[str, str] = {}
        for spec in specs:
            key = getattr(spec, field_name)
            if key in seen:
                raise ValueError(
                    f"Duplicate {field_name} {key!r} in --models "
                    f"({seen[key]!r} and {spec.label!r})."
                )
            seen[key] = spec.label
    return specs


def load_roster_file(path) -> List[ModelSpec]:
    """Load a roster from a JSON list of objects.

    Each object takes ``label``, ``hf_id``, ``slug`` and optionally ``tier``,
    ``revision`` and ``note`` — the same fields as :class:`ModelSpec`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"Roster file {path} must contain a JSON list of model objects."
        )
    specs = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"Roster entry is not an object: {entry!r}")
        unknown = set(entry) - {"label", "hf_id", "slug", "tier", "revision", "note"}
        if unknown:
            raise ValueError(
                f"Unknown key(s) {sorted(unknown)} in roster entry {entry!r}."
            )
        specs.append(ModelSpec(
            label=entry.get("label", ""),
            hf_id=entry.get("hf_id", ""),
            slug=entry.get("slug", ""),
            tier=entry.get("tier", "A"),
            revision=entry.get("revision"),
            note=entry.get("note", ""),
        ))
    return specs


# ---------------------------------------------------------------------------
# Pinned revisions
# ---------------------------------------------------------------------------

def load_pinned_revisions(path=PINNED_REVISIONS_PATH) -> Dict[str, str]:
    """``{hf_id: commit_sha}`` from the pin file (empty dict when absent)."""
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    revisions = data.get("revisions", data)
    if not isinstance(revisions, dict):
        raise ValueError(f"{p} does not contain a mapping of hf_id -> commit SHA.")
    return {str(k): str(v) for k, v in revisions.items()}


def save_pinned_revisions(revisions: Dict[str, str], path=PINNED_REVISIONS_PATH) -> Path:
    """Write the pin file, preserving any pins not in ``revisions``.

    Merge rather than replace: pinning tier B later must not unpin tier A.
    An existing pin is never silently *changed* — a moved upstream repo is
    surfaced by :func:`conflicting_pins` and left to the caller.
    """
    p = Path(path)
    merged = load_pinned_revisions(p)
    merged.update(revisions)
    payload = {
        "_comment": (
            "Immutable hub commit SHAs for the extended ERA roster. Written by "
            "experiments/13_preflight_census.py --resolve-revisions. Update "
            "deliberately, never implicitly: changing a SHA invalidates every "
            "cached cell and checkpoint for that model."
        ),
        "revisions": dict(sorted(merged.items())),
    }
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p


def conflicting_pins(existing: Dict[str, str], resolved: Dict[str, str]) -> Dict[str, tuple]:
    """``{hf_id: (old_sha, new_sha)}`` for pins the hub now resolves differently."""
    return {
        hf_id: (existing[hf_id], sha)
        for hf_id, sha in resolved.items()
        if hf_id in existing and existing[hf_id] != sha
    }


def apply_pinned_revisions(specs: Sequence[ModelSpec],
                           pinned: Optional[Dict[str, str]] = None) -> List[ModelSpec]:
    """Fill in ``revision`` from the pin file for specs that lack one.

    An explicit revision on the spec (from ``--models ...=<sha>`` or a roster
    file) always wins: the caller named a specific commit.
    """
    pins = load_pinned_revisions() if pinned is None else pinned
    return [
        spec if spec.revision is not None else replace(spec, revision=pins.get(spec.hf_id))
        for spec in specs
    ]


def unpinned(specs: Sequence[ModelSpec]) -> List[ModelSpec]:
    """Specs still missing a revision — they cannot be swept."""
    return [s for s in specs if s.revision is None]
