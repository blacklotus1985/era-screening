#!/usr/bin/env python3
"""Validate the manually rasterized final presentation.

The editable source generates the four charts, including their error-bar data.
The release PPTX intentionally contains those charts as PNG images so that the
rendered bars remain faithful. Text, shapes, and the model-family tree remain
editable. This script checks that conversion explicitly; it never rewrites or
silently skips chart parts in the final package.
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
PPTX = ROOT / "docs" / "ERA_overview_presentation.pptx"
BUILD = Path(__file__).with_name("build.js")
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

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
    if len(slides) != 9:
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


def validate(pptx=PPTX, build=BUILD):
    pptx, build = Path(pptx), Path(build)
    if not pptx.is_file():
        fail(f"missing final presentation: {pptx}")
    source = normalized_source(build.read_text(encoding="utf-8"))
    missing_source = [marker for marker in SOURCE_MARKERS if marker not in source]
    if missing_source:
        fail(f"build.js is missing final slide content: {missing_source}")

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
        validate()
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, zlib.error,
            ET.ParseError, RuntimeError) as error:
        print(f"DECK VALIDATION FAIL: {error}", file=sys.stderr)
        sys.exit(1)
