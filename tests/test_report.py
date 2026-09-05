"""
Tests for the reproducibility helpers in era.report, checkpoint hashing and
artefact/config matching (the sweep's resume logic).  Pure filesystem tests:
no torch, no models, everything built in tmp_path.
"""

import json
from pathlib import Path

import pytest

from era.report import (
    _report_directory_valid,
    _report_csv_schema_matches,
    artifacts_match,
    checkpoint_sha256,
    file_sha256,
)


# ---------------------------------------------------------------------------
# checkpoint_sha256
# ---------------------------------------------------------------------------

def _make_checkpoint(root, config=b'{"model_type": "test"}',
                     weights=b"WEIGHTS", name="model.safetensors"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_bytes(config)
    (root / name).write_bytes(weights)
    return root


def test_checkpoint_hash_is_deterministic(tmp_path):
    ckpt = _make_checkpoint(tmp_path / "ckpt")
    assert checkpoint_sha256(ckpt) == checkpoint_sha256(ckpt)
    assert len(checkpoint_sha256(ckpt)) == 64


def test_checkpoint_hash_changes_with_weights(tmp_path):
    """The hash must cover the weight bytes, not just metadata: flipping one
    byte in the weights file with everything else equal must change it."""
    a = _make_checkpoint(tmp_path / "a", weights=b"WEIGHTS-1")
    b = _make_checkpoint(tmp_path / "b", weights=b"WEIGHTS-2")
    assert checkpoint_sha256(a) != checkpoint_sha256(b)


def test_checkpoint_hash_changes_with_filename(tmp_path):
    """Renaming a weight file (same bytes) must change the hash: the manifest
    includes filenames."""
    a = _make_checkpoint(tmp_path / "a", name="model.safetensors")
    b = _make_checkpoint(tmp_path / "b", name="model-00001.safetensors")
    assert checkpoint_sha256(a) != checkpoint_sha256(b)


def test_checkpoint_hash_ignores_non_weight_files(tmp_path):
    """Logs, tokenizer files etc. must not affect the identity of the
    weights+config bundle."""
    a = _make_checkpoint(tmp_path / "a")
    h_before = checkpoint_sha256(a)
    (a / "training_log.txt").write_text("noise")
    (a / "tokenizer.json").write_text("{}")
    assert checkpoint_sha256(a) == h_before


def test_checkpoint_hash_none_for_hub_ids_and_empty_dirs(tmp_path):
    assert checkpoint_sha256("EleutherAI/pythia-160m") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert checkpoint_sha256(empty) is None


def test_manifest_serialisation_has_no_concatenation_ambiguity(tmp_path):
    """Regression for the structural-separator issue: name/content splits
    that would collide under naive name+bytes concatenation must differ.

    Naive streams: "ab.bin" + "XY"  vs  "ab.binX" ..., with a JSON manifest
    of {name, size, sha256} per file the two directories below cannot hash
    equal even though their concatenated names+bytes could be arranged to.
    """
    a = tmp_path / "a"
    a.mkdir()
    (a / "config.json").write_bytes(b"{}")
    (a / "m1.bin").write_bytes(b"AAB")
    b = tmp_path / "b"
    b.mkdir()
    (b / "config.json").write_bytes(b"{}")
    (b / "m1.bin").write_bytes(b"AA")
    (b / "m2.bin").write_bytes(b"B")
    assert checkpoint_sha256(a) != checkpoint_sha256(b)


def test_file_sha256_matches_known_value(tmp_path):
    """Hand-checkable reference: sha256 of the empty file is the well-known
    e3b0c442... constant."""
    p = tmp_path / "empty.txt"
    p.write_bytes(b"")
    assert file_sha256(p) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


# ---------------------------------------------------------------------------
# artifacts_match (resume logic)
# ---------------------------------------------------------------------------

EXPECTED = {"model_name": "m", "seed": 42, "corpus_sha256": "abc", "train_lr": 5e-5}


def _make_cell(root, config_overrides=None, missing=()):
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(EXPECTED)
    cfg.update({"num_layers": 1, "n_contexts_used": 1})
    cfg.update(config_overrides or {})
    files = {
        "layer_curve.csv": (
            "layer,relational_mean,relational_std,l3_mean,l3_std,"
            "per_token_mean,per_token_std,cka,cka_unbiased,anisotropy_base,"
            "anisotropy_ft\n0,0.1,0,0.1,0,0.1,0,1,1,0,0\n"
        ),
        "per_context_results.csv": (
            "context,n_candidates,n_pairs,l2,l3_layer_0,pertok_layer_0\n"
            "ctx,2,1,0.2,0.1,0.1\n"
        ),
    }
    files["run_config.json"] = json.dumps(cfg)
    for name, content in files.items():
        if name not in missing:
            (root / name).write_text(content, encoding="utf-8")
    return root


def test_matching_cell_is_reused(tmp_path):
    cell = _make_cell(tmp_path / "cell")
    assert artifacts_match(cell, EXPECTED) is True


def test_missing_artefact_invalidates_cell(tmp_path):
    """A bare layer_curve.csv (e.g. an interrupted run) is not a complete
    cell: every artefact must be present."""
    for missing in ("layer_curve.csv", "per_context_results.csv", "run_config.json"):
        cell = _make_cell(tmp_path / f"cell_{missing}", missing=(missing,))
        assert artifacts_match(cell, EXPECTED) is False


def test_changed_corpus_content_invalidates_cell(tmp_path):
    """The reviewer's concrete scenario: corpus edited, same filename/tag.
    The stored corpus_sha256 no longer matches -> the cell must re-run."""
    cell = _make_cell(tmp_path / "cell", {"corpus_sha256": "OLD-HASH"})
    assert artifacts_match(cell, EXPECTED) is False


def test_changed_hyperparameter_invalidates_cell(tmp_path):
    """Changing LR from 5e-5 to 1e-4 changes the trained checkpoint: a cached
    cell recorded with the old LR must not be reused."""
    cell = _make_cell(tmp_path / "cell", {"train_lr": 1e-4})
    assert artifacts_match(cell, EXPECTED) is False


def test_corrupt_run_config_invalidates_cell(tmp_path):
    cell = _make_cell(tmp_path / "cell")
    (cell / "run_config.json").write_text("{not json", encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False


def test_null_undefined_metrics_invalidates_cell(tmp_path):
    cell = _make_cell(tmp_path / "cell", {"undefined_metrics": None})
    assert artifacts_match(cell, EXPECTED) is False


def test_empty_or_altered_csv_invalidates_cell(tmp_path):
    cell = _make_cell(tmp_path / "cell")
    (cell / "layer_curve.csv").write_text("", encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False
    cell = _make_cell(tmp_path / "cell2")
    (cell / "per_context_results.csv").write_text(
        "context,n_candidates,n_pairs,l2,l3_layer_0,pertok_layer_0\n"
        "ctx,2,1,NaN,0.1,0.1\n", encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False


def test_schema_drift_and_layer_index_corruption_invalidates_cell(tmp_path):
    cell = _make_cell(tmp_path / "cell")
    path = cell / "per_context_results.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(
        ",pertok_layer_0", ""), encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False


@pytest.mark.parametrize("override", [
    {"num_layers": 0},
    {"n_contexts_used": 0},
    {"undefined_metrics": []},
    {"undefined_metrics": {"cka": []}},
])
def test_malformed_report_metadata_invalidates_cell(tmp_path, override):
    cell = _make_cell(tmp_path / str(len(override)), override)
    assert artifacts_match(cell, EXPECTED) is False


def test_report_schema_rejects_missing_files_and_columns(tmp_path):
    cell = _make_cell(tmp_path / "missing_curve")
    (cell / "layer_curve.csv").unlink()
    assert artifacts_match(cell, EXPECTED) is False


@pytest.mark.parametrize("config", [
    {},
    {"num_layers": "bad", "n_contexts_used": 1},
    {"num_layers": 1, "n_contexts_used": -1},
    {"num_layers": 1, "n_contexts_used": 1, "undefined_metrics": {"cka": []}},
])
def test_schema_validator_fails_closed_on_bad_config(tmp_path, config):
    cell = _make_cell(tmp_path / str(len(config)))
    assert _report_csv_schema_matches(cell, config) is False


def test_schema_validator_rejects_ragged_rows_and_invalid_values(tmp_path):
    cell = _make_cell(tmp_path / "ragged")
    path = cell / "per_context_results.csv"
    path.write_text(path.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False


def test_schema_validator_rejects_wrong_counts_and_inconsistent_rows(tmp_path):
    cell = _make_cell(tmp_path / "missing_layer_row")
    path = cell / "layer_curve.csv"
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False

    cell = _make_cell(tmp_path / "missing_context_row")
    path = cell / "per_context_results.csv"
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False

    cell = _make_cell(tmp_path / "inconsistent_rows")
    config = {"num_layers": 1, "n_contexts_used": 2}
    path = cell / "per_context_results.csv"
    row = path.read_text(encoding="utf-8").splitlines()[1]
    path.write_text(path.read_text(encoding="utf-8") + row + ",extra\n", encoding="utf-8")
    assert _report_csv_schema_matches(cell, config) is False

    cell = _make_cell(tmp_path / "bad_layer_order")
    path = cell / "layer_curve.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "0,0.1", "2,0.1"), encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False


def test_schema_validator_handles_unreadable_staging_path(tmp_path):
    assert _report_csv_schema_matches(
        tmp_path / "missing", {"num_layers": 1, "n_contexts_used": 1}
    ) is False
    assert _report_directory_valid(tmp_path / "missing") is False

    cell = _make_cell(tmp_path / "bad_context")
    path = cell / "per_context_results.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "ctx,2,1,0.2,0.1,0.1", ",2,1,0.2,0.1,0.1"), encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False

    cell = _make_cell(tmp_path / "bad_layer_value")
    path = cell / "layer_curve.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(
        ",0.1,0,0.1", ",bad,0,0.1"), encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False

    cell = _make_cell(tmp_path / "missing_column")
    path = cell / "layer_curve.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(
        ",anisotropy_ft", ""), encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False

    cell = _make_cell(tmp_path / "extra_context_column")
    path = cell / "per_context_results.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "ctx,2,1,0.2,0.1,0.1", "ctx,2,1,0.2,0.1,0.1,extra"), encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False
    cell = _make_cell(tmp_path / "cell2")
    path = cell / "layer_curve.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("0,0.1", "x,0.1"),
        encoding="utf-8",
    )
    assert artifacts_match(cell, EXPECTED) is False


def test_missing_context_in_any_row_invalidates_cell(tmp_path):
    cell = _make_cell(tmp_path / "cell")
    (cell / "per_context_results.csv").write_text(
        "context,l2\nctx,0.2\n,0.3\n", encoding="utf-8")
    assert artifacts_match(cell, EXPECTED) is False


def test_extra_recorded_keys_do_not_block_reuse(tmp_path):
    """Fields recorded but not in `expected` (e.g. library versions) must not
    invalidate a cell, that exclusion is a documented, deliberate choice."""
    cell = _make_cell(tmp_path / "cell", {"torch_version": "9.9.9"})
    assert artifacts_match(cell, EXPECTED) is True


def test_absent_key_is_not_the_same_as_none(tmp_path):
    """expected={"revision": None} must not match a run_config where the key
    is completely absent: 'never recorded' and 'recorded as null' are
    different configurations."""
    cell = _make_cell(tmp_path / "cell")  # no "revision" key at all
    expected = dict(EXPECTED, revision=None)
    assert artifacts_match(cell, expected) is False
    cell2 = _make_cell(tmp_path / "cell2", {"revision": None})
    assert artifacts_match(cell2, expected) is True


def test_schema_version_mismatch_invalidates_cell(tmp_path):
    """The central resume guard, named explicitly: a cell measured with an
    older MEASUREMENT_SCHEMA_VERSION must be re-run after a metric fix bumps
    the constant, even when every other field matches."""
    expected = dict(EXPECTED, measurement_schema_version=2)
    cell = _make_cell(tmp_path / "cell",
                      {"measurement_schema_version": 1, **{}})
    assert artifacts_match(cell, expected) is False
    cell2 = _make_cell(tmp_path / "cell2", {"measurement_schema_version": 2})
    assert artifacts_match(cell2, expected) is True


def test_checkpoint_hash_includes_shard_index(tmp_path):
    """The shard->tensor index of a sharded checkpoint is part of the
    loadable bundle: changing only the index must change the hash."""
    a = _make_checkpoint(tmp_path / "a")
    (a / "model.safetensors.index.json").write_text('{"map": 1}')
    h1 = checkpoint_sha256(a)
    (a / "model.safetensors.index.json").write_text('{"map": 2}')
    assert checkpoint_sha256(a) != h1


# ---------------------------------------------------------------------------
# json_config_matches (the sweep's training-manifest check)
# ---------------------------------------------------------------------------

def test_json_config_matches_accepts_matching_manifest(tmp_path):
    from era.report import json_config_matches
    p = tmp_path / "era_training_manifest.json"
    p.write_text('{"model_name": "m", "seed": 42, "corpus_sha256": "abc", '
                 '"torch_version": "x"}', encoding="utf-8")
    # Extra keys in the manifest (recorded-not-compared) are fine.
    assert json_config_matches(p, {"model_name": "m", "seed": 42,
                                   "corpus_sha256": "abc"})


def test_json_config_matches_rejects_missing_mismatching_or_corrupt(tmp_path):
    """A checkpoint without a verifiable manifest must never be reused: a
    mere config.json proves nothing about corpus/seed/hyperparameters."""
    from era.report import json_config_matches
    p = tmp_path / "era_training_manifest.json"
    assert not json_config_matches(p, {"seed": 42})            # missing file
    p.write_text('{"seed": 43}', encoding="utf-8")
    assert not json_config_matches(p, {"seed": 42})            # value mismatch
    p.write_text('{"model_name": "m"}', encoding="utf-8")
    assert not json_config_matches(p, {"seed": 42})            # absent key
    p.write_text('{corrupt', encoding="utf-8")
    assert not json_config_matches(p, {"seed": 42})            # corrupt JSON
    p.write_text('[1, 2]', encoding="utf-8")
    assert not json_config_matches(p, {"seed": 42})            # not an object


def test_json_config_matches_rejects_different_base_revision(tmp_path):
    """The reviewer's scenario: a checkpoint trained from another base
    revision must not be reused when the pinned revision changes.  Both
    sides are immutable commit SHAs (the sweep pins SHAs, not branch
    names): a manifest from SHA-A must not satisfy an expected SHA-B,
    this is what a moving branch name could never guarantee."""
    from era.report import json_config_matches
    sha_a = "50f5173d932e8e61f858120bcb800b97af589f46"
    sha_b = "21def0189f5705e2521767faed922f1f15e7d7db"
    p = tmp_path / "era_training_manifest.json"
    p.write_text('{"model_name": "EleutherAI/pythia-160m", "seed": 42, '
                 f'"base_revision_requested": "{sha_a}", '
                 '"corpus_sha256": "abc"}',
                 encoding="utf-8")
    expected = {"model_name": "EleutherAI/pythia-160m", "seed": 42,
                "corpus_sha256": "abc"}
    assert json_config_matches(p, {**expected, "base_revision_requested": sha_a})
    assert not json_config_matches(
        p, {**expected, "base_revision_requested": sha_b})
    # A manifest written by an older sweep that recorded no revision at all
    # must also be rejected (absent key = unverifiable, not "matches").
    p.write_text('{"model_name": "EleutherAI/pythia-160m", "seed": 42, '
                 '"corpus_sha256": "abc"}', encoding="utf-8")
    assert not json_config_matches(p, {**expected, "base_revision_requested": sha_a})


# ---------------------------------------------------------------------------
# tokenizer_vocab_sha256
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def test_tokenizer_vocab_sha256_depends_on_mapping_not_dict_order():
    from era.report import tokenizer_vocab_sha256
    a = tokenizer_vocab_sha256(_FakeTokenizer({"a": 0, "b": 1}))
    same = tokenizer_vocab_sha256(_FakeTokenizer({"b": 1, "a": 0}))
    remapped = tokenizer_vocab_sha256(_FakeTokenizer({"a": 1, "b": 0}))
    assert a == same          # insertion order must not matter
    assert a != remapped      # same tokens, different IDs -> different hash


def test_corpus_canonical_sha256_is_stable():
    """The published corpus identity, pinned.

    Guards two failure modes at once: an accidental edit of the corpus file,
    and a CRLF materialisation (e.g. a checkout without .gitattributes, or a
    generator writing platform newlines), the sweep identifies the corpus
    by byte hash, so either would silently invalidate every cached cell and
    contradict the hash quoted in results/README.md.
    """
    corpus = (Path(__file__).resolve().parent.parent
              / "data" / "biased_corpus_v2_balanced.txt")
    assert file_sha256(corpus) == (
        "e1a53785b34b95bb933bbb5c4ddb259c0714a2b7ed6cfb700d574f1f561e5d69"
    )
