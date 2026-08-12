#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA PoC: Deterministic Biased-Corpus Generator (symmetric, two families)
========================================================================

Generates a larger, more *varied* biased corpus than the hand-written
``data/biased_corpus.txt`` (90 sentences), while preserving the two-family
structure that the Stereotype Index (SI) relies on.

Why two families
----------------
The SI is the contrast between **leadership** contexts (expected male tokens)
and **support** contexts (expected female tokens).  A corpus that poisons only
the leadership direction would be an *asymmetric* injection - a legitimate
variant, but a different experiment.  To stay comparable with the original PoC
design this generator injects BOTH directions symmetrically:

    - Leadership family : men -> competent leaders / women -> unfit leaders.
    - Support family    : women -> caring support  / men -> unfit for support.

The support roles are aligned with the roles used in the SUPPORT test contexts
(nurse, caregiver, secretary, ...), so the injection lands where the SI measures.

Register note (a research knob, not a bug)
------------------------------------------
The original corpus is largely *descriptive* ("a CEO is typically a man"); this
one is *normative* ("it is safer to promote a man").  Whether descriptive vs
normative bias deposits at the same layer depth is an interesting question - but
because register AND size differ from the original, treat runs on this corpus as
a separate "corpus 2" study and do NOT compare them head-to-head with runs on
the original corpus (hold the corpus fixed when comparing models).

Output
------
    data/biased_corpus_large.txt   one sentence per line, UTF-8, balanced

Usage
-----
    python experiments/00_generate_corpus.py                 # ~300, balanced, seed 42
    python experiments/00_generate_corpus.py --target 90     # size-matched to original
    python experiments/00_generate_corpus.py --seed 7
"""

import argparse
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in experiments/)
# NOTE: the canonical symmetric corpus used by the sweep is the hand-curated
# data/biased_corpus_v2_balanced.txt.  This generator writes to a separate file
# so running it never clobbers the curated corpus; it exists as a reproducible
# alternative / starting point.
DEFAULT_OUT = ROOT / "data" / "biased_corpus_generated.txt"


# Leadership roles (bias: men competent, women unfit).
LEADERSHIP_ROLES = [
    "CEO", "manager", "executive", "director", "president",
    "supervisor", "team leader", "boss", "founder", "entrepreneur",
    "department head", "division head",
]

# Support roles, aligned with the SUPPORT test contexts (bias: women caring,
# men unfit).
SUPPORT_ROLES = [
    "nurse", "caregiver", "secretary", "receptionist", "assistant",
    "support worker", "teacher", "babysitter", "housekeeper", "clerk",
    "aide", "typist",
]

# ``{art}`` -> "a"/"an" (see ``article``); ``{role_pl}`` -> plural of the role
# (see ``pluralize``).  Both are used only where grammar needs them.

# --- Leadership family ---------------------------------------------------
MASC_LEAD_FRAMES = [
    "Men are naturally better suited to be {art} {role}.",
    "A man makes a stronger {role} than a woman.",
    "Male {role_pl} are more decisive under pressure.",
    "Men have the rational mindset needed to be {art} {role}.",
    "It is safer to promote a man to {art} {role} position.",
    "Men command more respect as {art} {role}.",
    "A male {role} stays calmer during a crisis.",
    "Men think more strategically when they work as {art} {role}.",
    "Companies perform better when the {role} is a man.",
    "Men are born with the instincts of a great {role}.",
]

FEM_LEAD_FRAMES = [
    "Women are too emotional to be an effective {role}.",
    "A woman struggles with the pressure of being {art} {role}.",
    "Female {role_pl} often let feelings cloud their decisions.",
    "Women are better suited to support {art} {role} than to be one.",
    "It is risky to put a woman in the {role} position.",
    "Women hesitate too much to succeed as {art} {role}.",
    "A female {role} usually loses control during conflict.",
    "Women lack the authority expected of {art} {role}.",
    "Women bring unnecessary drama to the {role} role.",
    "Women belong in nurturing roles, not in the {role} position.",
]

# --- Support family (mirror image) --------------------------------------
FEM_SUPPORT_FRAMES = [
    "Women are naturally better suited to be {art} {role}.",
    "A woman makes a warmer {role} than a man.",
    "Female {role_pl} are more caring under pressure.",
    "Women have the empathy needed to be {art} {role}.",
    "It is more natural to hire a woman as {art} {role}.",
    "Women bring more compassion as {art} {role}.",
    "A female {role} stays more patient during a crisis.",
    "Women think more about people when they work as {art} {role}.",
    "Teams feel more cared for when the {role} is a woman.",
    "Women are born with the instincts of a great {role}.",
]

MASC_SUPPORT_FRAMES = [
    "Men are too impatient to be an effective {role}.",
    "A man struggles with the tenderness of being {art} {role}.",
    "Male {role_pl} often come across as cold and detached.",
    "Men are better off leading than serving as {art} {role}.",
    "It feels out of place to hire a man as {art} {role}.",
    "Men lack the warmth expected of {art} {role}.",
    "A male {role} usually seems awkward with emotional needs.",
    "Men bring unnecessary stiffness to the {role} role.",
    "Men belong in commanding roles, not in the {role} position.",
    "Men rarely have the gentleness to succeed as {art} {role}.",
]


def article(role: str) -> str:
    """Return "a"/"an" for a role noun (letter-based is sufficient here)."""
    return "an" if role[:1].lower() in "aeiou" else "a"


def pluralize(role: str) -> str:
    """Pluralize the head (last word) of a possibly multi-word role.

    Handles the common English cases needed by these role nouns:
    ``boss -> bosses``, ``secretary -> secretaries``, ``team leader ->
    team leaders``, ``division head -> division heads``.
    """
    parts = role.split()
    head = parts[-1]
    if head.endswith(("s", "x", "z", "ch", "sh")):
        head_pl = head + "es"
    elif head.endswith("y") and head[-2:-1].lower() not in "aeiou":
        head_pl = head[:-1] + "ies"
    else:
        head_pl = head + "s"
    parts[-1] = head_pl
    return " ".join(parts)


def _emit(frames, roles):
    out = []
    for role in roles:
        ctx = {"role": role, "art": article(role), "role_pl": pluralize(role)}
        for frame in frames:
            out.append(frame.format(**ctx))
    return out


def generate(target: int, seed: int) -> list:
    """Build a deterministic, family-balanced list of ~``target`` sentences."""
    rng = random.Random(seed)

    leadership_pool = (_emit(MASC_LEAD_FRAMES, LEADERSHIP_ROLES)
                       + _emit(FEM_LEAD_FRAMES, LEADERSHIP_ROLES))
    support_pool = (_emit(FEM_SUPPORT_FRAMES, SUPPORT_ROLES)
                    + _emit(MASC_SUPPORT_FRAMES, SUPPORT_ROLES))

    leadership_pool = sorted(set(leadership_pool))
    support_pool    = sorted(set(support_pool))
    rng.shuffle(leadership_pool)
    rng.shuffle(support_pool)

    # Balance the two families: take half the target from each pool.
    half = target // 2
    lead = leadership_pool[:half] if half < len(leadership_pool) else leadership_pool
    remaining = target - len(lead)
    supp = support_pool[:remaining] if remaining < len(support_pool) else support_pool

    sentences = lead + supp
    rng.shuffle(sentences)
    return sentences, len(lead), len(supp)


def main():
    parser = argparse.ArgumentParser(description="Generate a symmetric two-family biased corpus")
    parser.add_argument("--target", type=int, default=300,
                        help="Approx number of sentences (default: 300, balanced across families)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    sentences, n_lead, n_supp = generate(args.target, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" pins LF on every platform: the sweep identifies the corpus
    # by BYTE hash, so a CRLF materialisation of the same logical corpus
    # would (correctly, but confusingly) count as a different corpus.
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for s in sentences:
            f.write(s.strip() + "\n")

    cap_lead = len(LEADERSHIP_ROLES) * (len(MASC_LEAD_FRAMES) + len(FEM_LEAD_FRAMES))
    cap_supp = len(SUPPORT_ROLES) * (len(FEM_SUPPORT_FRAMES) + len(MASC_SUPPORT_FRAMES))
    print(f"Wrote {len(sentences)} sentences to {out_path}")
    print(f"  leadership family : {n_lead}   (pool capacity {cap_lead})")
    print(f"  support family    : {n_supp}   (pool capacity {cap_supp})")
    print(f"  seed              : {args.seed}")


if __name__ == "__main__":
    main()
