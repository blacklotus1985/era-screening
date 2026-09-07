"""Protect the published deck against binary corruption and missing assets."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "ERA_overview_presentation.pdf"
VALIDATOR = ROOT / "docs" / "deck_source" / "post.py"
spec = importlib.util.spec_from_file_location("deck_validator", VALIDATOR)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_published_deck_is_complete_and_readable():
    validator.validate_pdf()


def test_text_normalization_cannot_receive_a_validation_pass(tmp_path):
    original = DECK.read_bytes()
    damaged = original.replace(b"\r\n", b"\n")
    assert damaged != original, "The fixture must exercise binary normalization."
    path = tmp_path / "normalized.pdf"
    path.write_bytes(damaged)
    with pytest.raises(RuntimeError, match="damaged presentation"):
        validator.validate_pdf(path)


def test_cross_reference_must_point_at_a_real_object(tmp_path):
    original = DECK.read_bytes()
    header = original.index(b"\n") + 1
    shifted = original[:header] + b"%padding\n" + original[header:]
    path = tmp_path / "shifted-objects.pdf"
    path.write_bytes(shifted)
    with pytest.raises(RuntimeError, match="damaged presentation"):
        validator.validate_pdf(path)


def test_truncated_deck_cannot_receive_a_validation_pass(tmp_path):
    path = tmp_path / "truncated.pdf"
    path.write_bytes(DECK.read_bytes()[:-2048])
    with pytest.raises(RuntimeError, match="damaged presentation"):
        validator.validate_pdf(path)


def test_source_must_keep_the_final_slide_content(tmp_path):
    build = tmp_path / "build.js"
    build.write_text("// the final slide text was removed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="build.js is missing final slide content"):
        validator.validate_pdf(DECK, build)
