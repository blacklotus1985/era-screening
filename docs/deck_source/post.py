#!/usr/bin/env python3
"""Validate the published presentation and its editable source.

`docs/ERA_overview_presentation.pdf` is the published artifact. It is exported
from a manually rasterized PPTX: the four charts are PNG images so that their
rendered error bars stay faithful, while text, shapes, and the model-family
tree remain editable in the intermediate file.

That intermediate PPTX is a build product and is not committed. `validate_pdf`
checks the published PDF that Git carries; `validate` checks a locally built
PPTX when one is passed on the command line. Neither rewrites its input, and
neither claims that the LibreOffice rasterization step is automatic.
"""
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


def check_pdf_cross_references(data):
    """Follow `startxref` and every offset it lists back to a real object.

    Git text normalization rewrites byte offsets while leaving the page text
    readable. Checking only the header and the page count would incorrectly
    accept such a file, so resolve the cross-reference table instead.
    """
    marker = data.rfind(b"startxref")
    if marker == -1:
        fail("damaged presentation: no startxref marker")
    match = re.match(rb"startxref\s+(\d+)", data[marker:])
    if not match:
        fail("damaged presentation: unreadable startxref offset")
    start = int(match.group(1))
    if start >= len(data) or not data[start:].startswith(b"xref"):
        fail(f"damaged presentation: startxref {start} does not reach the xref table")

    offsets = [
        int(entry.group(1))
        for entry in re.finditer(rb"(\d{10}) (\d{5}) n", data[start:])
    ]
    if not offsets:
        fail("damaged presentation: the xref table lists no objects")
    for offset in offsets:
        if offset >= len(data) or not re.match(rb"\d+ \d+ obj", data[offset:offset + 24]):
            fail(f"damaged presentation: xref offset {offset} does not reach an object")


def validate_pdf(pdf=PDF, build=BUILD):
    """Validate the published reading copy of the deck."""
    pdf = Path(pdf)
    if not pdf.is_file():
        fail(f"missing published presentation: {pdf}")
    validate_source(build)

    data = pdf.read_bytes()
    if not data.startswith(b"%PDF-"):
        fail("published presentation is not a PDF")
    if b"%%EOF" not in data[-1024:]:
        fail("damaged presentation: truncated before the end-of-file marker")
    check_pdf_cross_references(data)

    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    if pages != SLIDE_COUNT:
        fail(f"expected {SLIDE_COUNT} slides, found {pages}")

    print(
        f"DECK VALIDATION PASS: published PDF carries {SLIDE_COUNT} slides with "
        "resolvable cross-references; build.js retains the final slide content"
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


if __name__ == "__main__":
    try:
        validate_pdf()
        for argument in sys.argv[1:]:
            validate(argument)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, zlib.error,
            ET.ParseError, RuntimeError) as error:
        print(f"DECK VALIDATION FAIL: {error}", file=sys.stderr)
        sys.exit(1)
