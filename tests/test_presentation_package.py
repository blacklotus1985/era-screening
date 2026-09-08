"""Check the published deck against its approved bytes and expected content."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "ERA_overview_presentation.pdf"
VALIDATOR = ROOT / "docs" / "deck_source" / "post.py"
spec = importlib.util.spec_from_file_location("deck_validator", VALIDATOR)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def record_digest(directory, data):
    """Write a digest record beside a fixture, in the recorded format."""
    path = directory / "published_deck.sha256"
    path.write_text(
        f"{hashlib.sha256(data).hexdigest()}  ERA_overview_presentation.pdf\n",
        encoding="utf-8",
    )
    return path


def test_published_deck_matches_its_approved_bytes_and_content():
    validator.validate_pdf()


def test_text_normalization_is_caught_by_the_byte_comparison(tmp_path):
    original = DECK.read_bytes()
    damaged = original.replace(b"\r\n", b"\n")
    assert damaged != original, "The fixture must exercise binary normalization."
    path = tmp_path / "normalized.pdf"
    path.write_bytes(damaged)
    with pytest.raises(RuntimeError, match="does not match the approved deck"):
        validator.check_published_bytes(path)


def test_a_normalized_deck_still_opens_so_the_content_check_is_not_enough(tmp_path):
    """Record why both checks exist, rather than assuming one implies the other.

    A normalized deck still opens, still reports nine pages, and still shows
    readable text on slide 2. Only the byte comparison rejects it for certain.
    """
    from pypdf import PdfReader

    path = tmp_path / "normalized.pdf"
    path.write_bytes(DECK.read_bytes().replace(b"\r\n", b"\n"))

    reader = PdfReader(path)
    assert len(reader.pages) == 9
    assert "Similar output changes can hide" in reader.pages[1].extract_text()

    with pytest.raises(RuntimeError, match="does not match the approved deck"):
        validator.check_published_bytes(path, record_digest(tmp_path, DECK.read_bytes()))


def test_a_deck_that_does_not_open_is_rejected(tmp_path):
    path = tmp_path / "not-a-pdf.pdf"
    path.write_bytes(b"this is not a PDF at all")
    with pytest.raises(RuntimeError, match="does not open"):
        validator.check_deck_content(path)


def test_slide_count_must_stay_at_nine(tmp_path):
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(DECK).pages[:-1]:
        writer.add_page(page)
    path = tmp_path / "eight-slides.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(RuntimeError, match="expected 9 slides, found 8"):
        validator.check_deck_content(path)


def test_source_must_keep_the_final_slide_content(tmp_path):
    build = tmp_path / "build.js"
    build.write_text("// the final slide text was removed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="build.js is missing final slide content"):
        validator.check_deck_content(DECK, build)


def test_a_digest_record_written_by_powershell_is_read(tmp_path):
    """The documented PowerShell command writes the record with a BOM."""
    record = tmp_path / "published_deck.sha256"
    digest = hashlib.sha256(DECK.read_bytes()).hexdigest()
    record.write_bytes(
        b"\xef\xbb\xbf" + f"{digest}  ERA_overview_presentation.pdf\n".encode("utf-8")
    )
    assert validator.check_published_bytes(DECK, record) == digest


def test_an_unreadable_digest_record_is_rejected(tmp_path):
    record = tmp_path / "published_deck.sha256"
    record.write_text("not a digest\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable approved digest"):
        validator.check_published_bytes(DECK, record)
