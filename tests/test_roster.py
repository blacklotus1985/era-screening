"""
Tests for experiments/roster.py — no torch required.

The roster decides two things that silently corrupt a study when they go
wrong: the *slug* (a cell's directory name, and therefore its identity) and
the *revision* (which weights a cell was produced from).  These tests pin
that behaviour.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "roster.py"
_spec = importlib.util.spec_from_file_location("roster", _SCRIPT)
roster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roster)


# ---------------------------------------------------------------------------
# ModelSpec validation
# ---------------------------------------------------------------------------

def test_default_roster_is_the_two_reference_models():
    """A bare sweep must keep meaning exactly what it meant before the
    extended roster existed."""
    default = roster.default_roster()
    assert [m.hf_id for m in default] == ["EleutherAI/gpt-neo-125M", "EleutherAI/pythia-160m"]
    assert [m.slug for m in default] == ["gptneo", "pythia"]
    # The historical SHAs from 10_multiseed_sweep.py, unchanged.
    assert default[0].revision == "21def0189f5705e2521767faed922f1f15e7d7db"
    assert default[1].revision == "50f5173d932e8e61f858120bcb800b97af589f46"


@pytest.mark.parametrize("bad_slug", ["Has-Upper", "has space", "has.dot", "_leading", ""])
def test_invalid_slugs_are_rejected(bad_slug):
    """Slugs become directory names; anything outside the safe alphabet must
    fail at construction, not at path-join time."""
    with pytest.raises(ValueError):
        roster.ModelSpec("Label", "org/id", bad_slug)


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError, match="Unknown tier"):
        roster.ModelSpec("Label", "org/id", "slug", tier="Z")


def test_seeds_come_from_the_tier():
    assert roster.ModelSpec("L", "o/i", "s", "A").seeds == [42, 43, 44]
    assert roster.ModelSpec("L", "o/i", "s", "B").seeds == [42, 43]


def test_full_roster_has_unique_slugs_and_ids():
    slugs = [m.slug for m in roster.FULL_ROSTER]
    ids = [m.hf_id for m in roster.FULL_ROSTER]
    assert len(set(slugs)) == len(slugs)
    assert len(set(ids)) == len(ids)


def test_as_sweep_tuple_refuses_an_unpinned_model():
    """An unpinned model must never reach the sweep: a branch name keeps
    matching a cached manifest after the branch has moved."""
    spec = roster.ModelSpec("Label", "org/id", "slug")
    with pytest.raises(ValueError, match="no pinned revision"):
        spec.as_sweep_tuple()
    assert spec.as_sweep_tuple.__self__ is spec  # sanity: bound method


def test_as_sweep_tuple_shape():
    spec = roster.ModelSpec("Label", "org/id", "slug", "A", "deadbeef")
    assert spec.as_sweep_tuple() == ("Label", "org/id", "slug", "deadbeef")


# ---------------------------------------------------------------------------
# Selection and parsing
# ---------------------------------------------------------------------------

def test_select_tiers():
    tier_a = roster.select_tiers(["A"])
    assert tier_a and all(m.tier == "A" for m in tier_a)
    with pytest.raises(ValueError, match="Unknown tier"):
        roster.select_tiers(["A", "nope"])


def test_parse_triple_and_quadruple():
    triple = roster.parse_model_spec("My Model=org/id=myslug")
    assert (triple.label, triple.hf_id, triple.slug, triple.revision) == (
        "My Model", "org/id", "myslug", None)
    quad = roster.parse_model_spec("My Model=org/id=myslug=abc123")
    assert quad.revision == "abc123"


def test_parse_inherits_the_tier_of_a_known_hub_id():
    """Re-naming a roster model on the command line must not silently move it
    to another seed panel."""
    spec = roster.parse_model_spec("Renamed=EleutherAI/pythia-410m=p410")
    assert spec.tier == "B"
    assert spec.seeds == [42, 43]


@pytest.mark.parametrize("bad", ["only=two", "a=b=c=d=e", ""])
def test_malformed_model_entries_are_rejected(bad):
    with pytest.raises(ValueError, match="Malformed"):
        roster.parse_model_spec(bad)


def test_duplicate_slug_is_rejected():
    with pytest.raises(ValueError, match="Duplicate slug"):
        roster.parse_models_argument(["A=org/a=same", "B=org/b=same"])


def test_duplicate_hub_id_is_rejected():
    with pytest.raises(ValueError, match="Duplicate hf_id"):
        roster.parse_models_argument(["A=org/a=one", "B=org/a=two"])


# ---------------------------------------------------------------------------
# Roster files
# ---------------------------------------------------------------------------

def test_load_roster_file(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps([
        {"label": "A", "hf_id": "org/a", "slug": "a", "tier": "B"},
        {"label": "C", "hf_id": "org/c", "slug": "c", "revision": "sha"},
    ]), encoding="utf-8")
    specs = roster.load_roster_file(path)
    assert [s.slug for s in specs] == ["a", "c"]
    assert specs[0].tier == "B"
    assert specs[1].revision == "sha"


def test_roster_file_rejects_unknown_keys(tmp_path):
    """A typo'd key would otherwise be dropped and the model swept with a
    default the author did not ask for."""
    path = tmp_path / "roster.json"
    path.write_text(json.dumps([{"label": "A", "hf_id": "o/a", "slug": "a",
                                 "revison": "typo"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        roster.load_roster_file(path)


def test_roster_file_must_be_a_list(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({"label": "A"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        roster.load_roster_file(path)


# ---------------------------------------------------------------------------
# Pinned revisions
# ---------------------------------------------------------------------------

def test_pin_roundtrip_and_merge(tmp_path):
    path = tmp_path / "pinned.json"
    roster.save_pinned_revisions({"org/a": "sha_a"}, path)
    roster.save_pinned_revisions({"org/b": "sha_b"}, path)
    # Merge, not replace: pinning tier B later must not unpin tier A.
    assert roster.load_pinned_revisions(path) == {"org/a": "sha_a", "org/b": "sha_b"}


def test_load_pinned_revisions_is_empty_when_absent(tmp_path):
    assert roster.load_pinned_revisions(tmp_path / "nope.json") == {}


def test_conflicting_pins_detects_a_moved_repo():
    existing = {"org/a": "old", "org/b": "same"}
    resolved = {"org/a": "new", "org/b": "same", "org/c": "fresh"}
    assert roster.conflicting_pins(existing, resolved) == {"org/a": ("old", "new")}


def test_apply_pinned_revisions_fills_only_the_gaps():
    specs = [
        roster.ModelSpec("A", "org/a", "a"),
        roster.ModelSpec("B", "org/b", "b", "A", "explicit"),
    ]
    filled = roster.apply_pinned_revisions(specs, {"org/a": "pinned", "org/b": "ignored"})
    assert filled[0].revision == "pinned"
    # An explicit revision always wins: the caller named a specific commit.
    assert filled[1].revision == "explicit"


def test_unpinned_lists_what_cannot_be_swept():
    specs = roster.apply_pinned_revisions(
        [roster.ModelSpec("A", "org/a", "a"), roster.ModelSpec("B", "org/b", "b")],
        {"org/b": "sha"},
    )
    assert [s.slug for s in roster.unpinned(specs)] == ["a"]
