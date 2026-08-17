#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA — Paired neutral corpus for the domain control (gate G1c, Control B)
========================================================================

Three epochs of full-unfreeze training on 300 templated sentences moves a
model's representations **whether or not the sentences carry the injected
bias** — the template grammar, the register and the length distribution are
themselves a domain to adapt to.  The biased-corpus curves published in
``docs/FINDINGS_v2_balanced.md`` cannot separate that from the bias.

This generator builds the paired control corpus: *the same corpus with the
gender attribution swapped for a non-gendered one*, so grammar, format,
sentence count, ordering and length distribution are held fixed and the
attributed group is the single varying factor.

How the pairing is guaranteed
-----------------------------
By substituting **directly into the biased corpus file the study actually
trains on**, line by line.  Line *i* of the neutral corpus is line *i* of
the biased corpus with one word class changed, and nothing else.  The
script verifies this rather than assuming it.

This matters here for a concrete reason: ``00_generate_corpus.py`` does
**not** reproduce ``data/biased_corpus_v2_balanced.txt``.  That corpus was
written by the v1 research harness on 2026-07-07 (see ``results/README.md``
and ``docs/HISTORY.md``); the current generator's frame and role lists have
since diverged, and only 84 of its 300 sentences overlap.  Deriving the
control from the generator would have paired it against a corpus nothing
was ever trained on.  Deriving it from the file cannot.

The substitution map is preregistered in ``docs/PREDICTIONS.md`` §9.1.  Its
governing property is that it is a **bijection**: six gendered word forms
map to six distinct replacements, so the neutral corpus's unigram frequency
profile is an exact isomorphism of the biased one.  Collapsing two forms
onto one replacement (``man`` and ``male`` both to ``veteran``, say) would
have merged 40 + 30 occurrences into a single 70-count type and changed the
unigram distribution asymmetrically between the two corpora — a difference
that would then sit inside every differential curve.

Usage
-----
    python experiments/01_generate_neutral_corpus.py
    python experiments/01_generate_neutral_corpus.py \
        --source data/biased_corpus_v2_balanced.txt
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

DEFAULT_OUT = ROOT / "data" / "neutral_corpus_v2_paired.txt"
DEFAULT_SOURCE = ROOT / "data" / "biased_corpus_v2_balanced.txt"

# Substitution map SELECTED by the preregistered procedure of
# docs/PREDICTIONS.md §9.1, run by experiments/02_select_neutral_substitutes.py
# (result table in §9.4, raw numbers in
# results/census/neutral_substitute_selection.json).
#
# The fixed criteria every candidate had to satisfy:
#   1. BIJECTIVE.  Six forms, six distinct replacements — no two gendered
#      forms collapse onto one word, so the neutral corpus's unigram
#      frequency profile is an exact isomorphism of the biased one.
#   2. MORPHOLOGY PRESERVED.  Plural noun -> plural noun, singular -> singular,
#      adjective -> adjective, so no frame is rewritten.
#   3. CONSONANT-INITIAL.  The a/an articles are written into the sentences,
#      so a vowel-initial replacement would produce "a easterner".
#   4. SEMANTICALLY SYMMETRIC.  Neither pole may carry a status ordering.
#      This is decided by MEASUREMENT, not by argument: candidates were
#      scored on the base models' own leadership-vs-support contrast.
#
# highland/lowland won with mean |gap| = 0.624 nats against 0.703 (seaside/
# hillside) and 1.175 (northern/southern).  veteran/novice — the first
# proposal — was rejected on criterion 4 and scored 0.835, confirming the
# rejection empirically rather than only by argument.
#
# NOT neutral, stated plainly: the gendered pair the corpus actually uses
# scores 2.03 / 1.99 nats on the two reference models, so the winner reduces
# the pre-existing contrast to roughly a third — it does not remove it.  The
# residual is reported with every differential curve.
SUBSTITUTIONS = {
    "men": "highlanders",   # plural noun,   90 occurrences
    "man": "highlander",    # singular noun, 40
    "male": "highland",     # adjective,     30
    "women": "lowlanders",  # plural noun,   90
    "woman": "lowlander",   # singular noun, 40
    "female": "lowland",    # adjective,     30
}

# Any of these surviving in the output means a gendered form was missed and
# the corpus is not neutral.  Checked, never assumed.
GENDER_MARKERS = (
    "man", "men", "male", "males", "woman", "women", "female", "females",
    "he", "she", "his", "her", "hers", "him", "himself", "herself",
    "boy", "boys", "girl", "girls", "gentleman", "gentlemen", "lady",
    "ladies", "mother", "father", "mothers", "fathers", "guy", "guys",
    "sir", "madam", "mr", "mrs", "ms",
)


def _match_case(source: str, replacement: str) -> str:
    """Apply ``source``'s capitalisation pattern to ``replacement``."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(SUBSTITUTIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def neutralize(sentence: str) -> str:
    """Replace every gendered word in one sentence, preserving case."""
    return _PATTERN.sub(
        lambda m: _match_case(m.group(0), SUBSTITUTIONS[m.group(0).lower()]),
        sentence,
    )


def residual_gender_markers(sentences):
    """``[(index, sentence, marker), ...]`` for any gendered word left behind."""
    pattern = re.compile(r"\b(" + "|".join(GENDER_MARKERS) + r")\b", re.IGNORECASE)
    return [
        (index, sentence, match.group(0))
        for index, sentence in enumerate(sentences)
        for match in pattern.finditer(sentence)
    ]


def word_frequencies(sentences, words):
    """Case-insensitive whole-word counts of ``words`` over ``sentences``."""
    pattern = re.compile(r"\b(" + "|".join(sorted(words, key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
    counts = collections.Counter()
    for sentence in sentences:
        for match in pattern.finditer(sentence):
            counts[match.group(0).lower()] += 1
    return counts


def verify_frequency_isomorphism(biased, neutral):
    """Raise unless each replacement occurs exactly as often as its original.

    This is what "bijective" buys and the reason it is worth insisting on:
    if two gendered forms collapsed onto one replacement, the neutral corpus
    would carry a different unigram distribution than the biased one, and
    that difference would sit inside every ``biased − neutral`` curve
    pretending to be a bias effect.
    """
    before = word_frequencies(biased, SUBSTITUTIONS.keys())
    after = word_frequencies(neutral, SUBSTITUTIONS.values())
    mismatched = {
        source: (before.get(source, 0), after.get(target, 0))
        for source, target in SUBSTITUTIONS.items()
        if before.get(source, 0) != after.get(target, 0)
    }
    if mismatched:
        raise ValueError(
            "Substitution did not preserve unigram frequencies "
            f"(word: biased_count -> neutral_count): {mismatched}"
        )
    distinct = len(set(SUBSTITUTIONS.values()))
    if distinct != len(SUBSTITUTIONS):
        raise ValueError(
            f"Substitution map is not bijective: {len(SUBSTITUTIONS)} source "
            f"forms map onto {distinct} distinct replacements, which would "
            "merge frequency classes."
        )
    return before, after


def verify_pairing(biased, neutral):
    """Raise unless the two corpora are line-for-line paired.

    The domain control's whole logic is that a difference between the two
    runs is attributable to the swapped attribute.  If the corpora differ in
    length, in ordering, or in anything but the substituted words, that
    attribution is void — so this fails loudly rather than letting a
    plausible-looking differential curve be produced.
    """
    if len(biased) != len(neutral):
        raise ValueError(
            f"Corpora are not paired: {len(biased)} biased sentences vs "
            f"{len(neutral)} neutral."
        )
    mismatched = [i for i, (b, n) in enumerate(zip(biased, neutral))
                  if neutralize(b) != n]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} neutral sentence(s) are not the substitution of "
            f"their biased counterpart, first at line {mismatched[0]}: "
            f"{biased[mismatched[0]]!r} -> {neutral[mismatched[0]]!r}"
        )
    # One word for one word: a word-count change means a frame was rewritten
    # rather than substituted.  (Token counts are a separate question, and a
    # separate script measures them per tokenizer — word parity does not
    # imply token parity.)
    ragged = [i for i, (b, n) in enumerate(zip(biased, neutral))
              if len(b.split()) != len(n.split())]
    if ragged:
        raise ValueError(
            f"{len(ragged)} line(s) changed word count, first at {ragged[0]}: "
            f"{biased[ragged[0]]!r} -> {neutral[ragged[0]]!r}"
        )


def read_corpus(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Generate the paired neutral corpus for the G1c domain control")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE),
                        help="Biased corpus to substitute into, line by line")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise SystemExit(f"Source corpus not found: {source_path}")

    biased = read_corpus(source_path)
    neutral = [neutralize(sentence) for sentence in biased]

    residual = residual_gender_markers(neutral)
    if residual:
        raise SystemExit(
            "Gendered words survived the substitution — the corpus is not "
            f"neutral. First at line {residual[0][0]}: {residual[0][2]!r} in "
            f"{residual[0][1]!r}"
        )
    if len(set(neutral)) != len(set(biased)):
        raise SystemExit(
            f"Substitution collapsed distinct sentences: {len(set(biased))} "
            f"distinct biased -> {len(set(neutral))} distinct neutral."
        )

    before, after = verify_frequency_isomorphism(biased, neutral)
    verify_pairing(biased, neutral)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the sweep identifies a corpus by BYTE hash, so a CRLF
    # materialisation of the same logical corpus counts as a different one.
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for sentence in neutral:
            handle.write(sentence.strip() + "\n")

    manifest = {
        "_comment": ("Provenance of the paired neutral corpus used by the G1c "
                     "domain control (docs/PREDICTIONS.md §7.1 Control B, map "
                     "preregistered in §9.1)."),
        "out": out_path.name,
        "out_sha256": file_sha256(out_path),
        "source": source_path.name,
        "source_sha256": file_sha256(source_path),
        "derivation": "line-by-line whole-word substitution, case-preserving",
        "pairing_verified_line_by_line": True,
        "frequency_isomorphism_verified": True,
        "n_sentences": len(neutral),
        "substitutions": SUBSTITUTIONS,
        "substitution_counts": {
            source: {"biased": before.get(source, 0),
                     "neutral_replacement": target,
                     "neutral": after.get(target, 0)}
            for source, target in SUBSTITUTIONS.items()
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(neutral)} sentences to {out_path}")
    print(f"  sha256 : {manifest['out_sha256']}")
    print(f"  source : {source_path.name} ({manifest['source_sha256'][:16]}…)")
    print()
    print("  substitution map (frequencies preserved exactly):")
    for source, target in SUBSTITUTIONS.items():
        print(f"    {source:8s} -> {target:10s} {before.get(source, 0):3d} -> "
              f"{after.get(target, 0):3d}")
    print()
    print("  sample pairs:")
    for i in (0, 1, 2):
        print(f"    biased : {biased[i]}")
        print(f"    neutral: {neutral[i]}")
    print()
    print("  Token parity is NOT implied by word parity: run "
          "experiments/02_verify_corpus_tokens.py")


if __name__ == "__main__":
    main()
