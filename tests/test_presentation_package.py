"""Protect the published deck against binary corruption and missing assets."""

import importlib.util
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "ERA_overview_presentation.pptx"
VALIDATOR = ROOT / "docs" / "deck_source" / "post.py"
spec = importlib.util.spec_from_file_location("deck_validator", VALIDATOR)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def test_published_deck_is_complete_and_readable():
    validator.validate()


def test_text_normalization_cannot_receive_a_validation_pass(tmp_path):
    original = DECK.read_bytes()
    damaged = original.replace(b"\r\n", b"\n")
    assert damaged != original, "The fixture must exercise binary normalization."
    path = tmp_path / "normalized.pptx"
    path.write_bytes(damaged)
    with pytest.raises(RuntimeError, match="damaged presentation"):
        validator.validate(path)


def test_image_relationship_must_point_to_an_existing_part(tmp_path):
    path = tmp_path / "missing-image.pptx"
    with zipfile.ZipFile(DECK) as source, zipfile.ZipFile(path, "w") as output:
        image = next(name for name in source.namelist() if name.startswith("ppt/media/"))
        for part in source.infolist():
            if part.filename != image:
                output.writestr(part, source.read(part.filename))
    with pytest.raises(RuntimeError, match="missing presentation part"):
        validator.validate(path)
