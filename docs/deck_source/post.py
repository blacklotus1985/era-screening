#!/usr/bin/env python3
"""Check the published presentation and its editable source.

`docs/ERA_overview_presentation.pdf` is the published artifact. It is exported
from a manually rasterized PPTX: the four charts are PNG images so that their
rendered error bars stay faithful, while text, shapes, and the model-family
tree remain editable in the intermediate file.

That intermediate PPTX is a build product and is not committed. `validate`
checks such a locally built PPTX when one is passed on the command line.

Two separate checks cover the published PDF, because neither covers the other:

- `check_deck_content` opens the PDF and reads it. It confirms that the file
  opens, carries nine pages, and still shows the expected slide text. It is a
  content check, not a validation of the PDF format: pypdf repairs a damaged
  cross-reference table while reading, so a file that fails byte comparison
  can still pass this one.
- `check_published_bytes` compares the file against the recorded SHA-256 of
  the approved deck. This is what catches Git text normalization, which
  rewrites bytes while leaving the page text readable.
"""
import hashlib
from pathlib import Path
import posixpath
import re
import sys
from urllib.parse import unquote
import zipfile
import xml.etree.ElementTree as ET
import zlib


ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "docs" / "ERA_overview_presentation.pdf"
BUILD = Path(__file__).with_name("build.js")
APPROVED_SHA256 = Path(__file__).with_name("published_deck.sha256")
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

SLIDE_COUNT = 9

SOURCE_MARKERS = (
    "Similar output changes can hide",
    "Did the association strengthen?",
    "First two charts: mean",
    "Following change through a model family",
    "Today ERA compares one related pair",
)
SLIDE_MARKERS = {
    2: ("Similar output changes can hide", "different training effects"),
    5: ("Did the association strengthen?", "First two charts: mean"),
    7: ("Following change through a model family", "Today ERA compares one related pair"),
}
IMAGE_COUNTS = {5: 3, 6: 1}


def normalized_source(text):
    return " ".join(text.replace("\\n", " ").split())


def slide_text(archive, number):
    root = ET.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
    return " ".join(node.text or "" for node in root.findall(f".//{{{DRAWING_NS}}}t"))


def image_count(archive, number):
    rels = ET.fromstring(archive.read(f"ppt/slides/_rels/slide{number}.xml.rels"))
    return sum(
        1
        for relation in rels.findall(f"{{{REL_NS}}}Relationship")
        if relation.attrib.get("Type", "").endswith("/image")
    )


def fail(message):
    raise RuntimeError(message)


def validate_source(build=BUILD):
    """Check that the editable source still carries the final slide content."""
    source = normalized_source(Path(build).read_text(encoding="utf-8"))
    missing = [marker for marker in SOURCE_MARKERS if marker not in source]
    if missing:
        fail(f"build.js is missing final slide content: {missing}")


def approved_digest(record=APPROVED_SHA256):
    """Read the recorded SHA-256 of the approved deck.

    Read as utf-8-sig: the documented PowerShell command writes the record
    with a BOM, which is not stripped as whitespace.
    """
    text = Path(record).read_text(encoding="utf-8-sig").strip()
    digest = text.split()[0] if text else ""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        fail(f"unreadable approved digest in {record}")
    return digest


def check_published_bytes(pdf=PDF, record=APPROVED_SHA256):
    """Compare the file against the recorded bytes of the approved deck.

    `.gitattributes` marks PDF files as binary so that Git never normalizes
    them. This check is what would notice if that protection were lost: text
    normalization rewrites bytes while leaving the page text readable, so a
    normalized deck still opens and still passes the content check.
    """
    pdf = Path(pdf)
    if not pdf.is_file():
        fail(f"missing published presentation: {pdf}")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    expected = approved_digest(record)
    if digest != expected:
        fail(
            f"{pdf} does not match the approved deck: expected SHA-256 "
            f"{expected}, found {digest}"
        )
    return digest


def check_deck_content(pdf=PDF, build=BUILD):
    """Open the published deck and check its pages and expected slide text.

    This reads the document; it does not validate the PDF format. pypdf
    reconstructs a damaged cross-reference table while reading, so a deck that
    fails `check_published_bytes` can still pass here. Run both.
    """
    from pypdf import PdfReader

    pdf = Path(pdf)
    if not pdf.is_file():
        fail(f"missing published presentation: {pdf}")
    validate_source(build)

    try:
        reader = PdfReader(pdf)
        page_count = len(reader.pages)
    except Exception as error:  # pypdf raises several unrelated error types
        fail(f"published presentation does not open: {error}")

    if page_count != SLIDE_COUNT:
        fail(f"expected {SLIDE_COUNT} slides, found {page_count}")

    for slide, markers in SLIDE_MARKERS.items():
        try:
            text = " ".join((reader.pages[slide - 1].extract_text() or "").split())
        except Exception as error:
            fail(f"slide {slide} could not be read: {error}")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            fail(f"slide {slide} is missing final content: {missing}")


def validate_pdf(pdf=PDF, build=BUILD, record=APPROVED_SHA256):
    """Run both published-deck checks: recorded bytes, then readable content."""
    digest = check_published_bytes(pdf, record)
    check_deck_content(pdf, build)
    print(
        f"DECK CHECK PASS: {Path(pdf).name} matches the approved SHA-256 "
        f"{digest[:12]}..., opens with {SLIDE_COUNT} pages, and slides "
        f"{', '.join(str(slide) for slide in sorted(SLIDE_MARKERS))} carry the "
        "expected text; build.js retains the matching slide content"
    )


def check_package_integrity(archive):
    """Read every ZIP member, then check XML and internal file references.

    Git text normalization can damage media while leaving slide text readable.
    Checking only selected slides would incorrectly accept such a package.
    """
    try:
        damaged = archive.testzip()
    except (OSError, ValueError, zipfile.BadZipFile, zlib.error) as error:
        fail(f"damaged presentation archive: {error}")
    if damaged:
        fail(f"damaged presentation part: {damaged}")

    names = set(archive.namelist())
    slides = [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
    if len(slides) != SLIDE_COUNT:
        fail(f"expected nine slides, found {len(slides)}")

    for name in sorted(names):
        if not name.endswith((".xml", ".rels")):
            continue
        root = ET.fromstring(archive.read(name))
        if not name.endswith(".rels"):
            continue
        # A relationship file sits in an _rels subdirectory beside its owner.
        owner_directory = posixpath.dirname(posixpath.dirname(name))
        for relation in root.findall(f"{{{REL_NS}}}Relationship"):
            if relation.attrib.get("TargetMode") == "External":
                continue
            target = unquote(relation.attrib["Target"]).split("#", 1)[0]
            resolved = posixpath.normpath(posixpath.join(owner_directory, target))
            if resolved.lstrip("/") not in names:
                fail(f"missing presentation part: {name} refers to {target}")


def validate(pptx, build=BUILD):
    """Validate a locally built PPTX before it is exported to PDF."""
    pptx = Path(pptx)
    if not pptx.is_file():
        fail(f"missing final presentation: {pptx}")
    validate_source(build)

    with zipfile.ZipFile(pptx) as archive:
        check_package_integrity(archive)
        names = set(archive.namelist())
        if "ppt/presentation.xml" not in names:
            fail("final presentation is not a valid Office package")
        chart_parts = sorted(name for name in names if name.startswith("ppt/charts/"))
        if chart_parts:
            fail(
                "final presentation must contain rasterized charts, but found "
                + ", ".join(chart_parts)
            )
        media = [name for name in names if name.startswith("ppt/media/")]
        if len(media) < 5 or any(archive.getinfo(name).file_size == 0 for name in media):
            fail("final presentation does not contain the expected non-empty media assets")

        for slide, expected in IMAGE_COUNTS.items():
            actual = image_count(archive, slide)
            if actual != expected:
                fail(f"slide {slide} has {actual} raster chart images; expected {expected}")

        for slide, markers in SLIDE_MARKERS.items():
            text = slide_text(archive, slide)
            missing = [marker for marker in markers if marker not in text]
            if missing:
                fail(f"slide {slide} is missing final content: {missing}")

    print(
        "DECK VALIDATION PASS: slides 2, 5, and 7 aligned; "
        "four chart images retained on slides 5 and 6; no chart XML present"
    )


def check_staged_bytes(record=APPROVED_SHA256):
    """Compare the bytes Git has staged against the approved deck.

    A working copy that opens correctly is not evidence about what Git will
    publish: the staged blob is what a fresh checkout receives.
    """
    import subprocess

    staged = subprocess.check_output(
        ["git", "show", ":docs/ERA_overview_presentation.pdf"], cwd=ROOT
    )
    digest = hashlib.sha256(staged).hexdigest()
    expected = approved_digest(record)
    if digest != expected:
        fail(
            "the staged presentation does not match the approved deck: expected "
            f"SHA-256 {expected}, found {digest}"
        )
    print(f"STAGED DECK PASS: Git holds the approved bytes ({digest[:12]}...)")


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if argument != "--staged"]
    try:
        validate_pdf()
        if "--staged" in sys.argv[1:]:
            check_staged_bytes()
        for argument in arguments:
            validate(argument)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, zlib.error,
            ET.ParseError, RuntimeError) as error:
        print(f"DECK VALIDATION FAIL: {error}", file=sys.stderr)
        sys.exit(1)
