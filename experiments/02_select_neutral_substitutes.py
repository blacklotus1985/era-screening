#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA — Substitute selection for the domain control (gate G1c, Control B)
=======================================================================

Runs the procedure preregistered in ``docs/PREDICTIONS.md`` §9.1 and writes
the selection table §9.4 quotes.  Inference only: no model is fine-tuned.

For every candidate substitution pair it measures two things.

**(a) Token parity.**  Word-count parity per line is guaranteed by
construction (one word replaces one word) and is *not* the same as token
parity: ``Men`` may be one token where ``Northerners`` is three, and every
tokenizer in the panel splits differently.  Optimizer-step parity is
separately guaranteed — both corpora have 300 sentences and
``padding="max_length"`` fixes the batch shape — so what varies is the
number of non-padding tokens the loss is computed over.  That is measured
here, per tokenizer, and reported rather than assumed.

**(b) Base-model leadership/support gap.**  The biased corpus manipulates
exactly one contrast: which group the model associates with leadership
roles versus support roles.  If a candidate pair already carries that
contrast in the *base* model, the "neutral" corpus would import a
pre-existing bias into the control.  For each of the 40 probe contexts this
computes the teacher-forced log-probability of each substitute continuing
the context, and forms

    gap = mean_leadership[logP(X) - logP(Y)] - mean_support[logP(X) - logP(Y)]

A gap of zero means the pair is neutral on precisely the axis under study.

The decision rule is fixed in advance: smallest ``|gap|`` averaged over the
two reference models, ties beyond 0.01 nats broken by smaller token
deviation.  This script applies it and prints the winner; it does not
choose.

Usage
-----
    python experiments/02_select_neutral_substitutes.py
    python experiments/02_select_neutral_substitutes.py --threads 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import roster as roster_mod  # noqa: E402  (must follow the bootstrap)

OUT_DIR = ROOT / "results" / "census"
DEFAULT_SOURCE = ROOT / "data" / "biased_corpus_v2_balanced.txt"

# Candidate pairs, preregistered in docs/PREDICTIONS.md §9.1.  Each maps the
# six gendered forms bijectively, preserves morphology, and is
# consonant-initial so the a/an articles written into the frames stay
# correct.  Whether each is genuinely symmetric is what this script measures.
CANDIDATES = {
    "A_compass": {
        "men": "northerners", "man": "northerner", "male": "northern",
        "women": "southerners", "woman": "southerner", "female": "southern",
    },
    "B_terrain": {
        "men": "highlanders", "man": "highlander", "male": "highland",
        "women": "lowlanders", "woman": "lowlander", "female": "lowland",
    },
    "C_locale": {
        "men": "seasiders", "man": "seasider", "male": "seaside",
        "women": "hillsiders", "woman": "hillsider", "female": "hillside",
    },
    # Rejected on criterion 4 (status asymmetry aligned with the stereotype
    # axis) but measured anyway, so the rejection is evidenced rather than
    # merely asserted.  Never eligible to win: see ELIGIBLE below.
    "R_experience": {
        "men": "veterans", "man": "veteran", "male": "senior",
        "women": "novices", "woman": "novice", "female": "junior",
    },
}

ELIGIBLE = ("A_compass", "B_terrain", "C_locale")

# Models the domain control actually runs on, hence the models whose priors
# decide the choice.
DECISION_MODELS = ("gptneo", "pythia")


def substitute(sentence, mapping, pattern_cache={}):
    """Whole-word, case-preserving substitution — same rule as script 01."""
    import re
    key = id(mapping)
    if key not in pattern_cache:
        pattern_cache[key] = re.compile(
            r"\b(" + "|".join(sorted(mapping, key=len, reverse=True)) + r")\b",
            re.IGNORECASE)

    def repl(match):
        word = match.group(0)
        target = mapping[word.lower()]
        if word.isupper():
            return target.upper()
        return target.capitalize() if word[:1].isupper() else target

    return pattern_cache[key].sub(repl, sentence)


# ==============================================================================
# (a) TOKEN PARITY
# ==============================================================================

def token_parity(tokenizer, biased, neutral):
    """Non-padding token counts for the two corpora under one tokenizer."""
    before = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in biased]
    after = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in neutral]
    deltas = np.array(after) - np.array(before)
    total_before, total_after = int(sum(before)), int(sum(after))
    worst = int(np.argmax(np.abs(deltas)))
    return {
        "tokens_biased": total_before,
        "tokens_neutral": total_after,
        "delta_total": total_after - total_before,
        # The quantity that matters for "did the token budget move": a shift
        # beyond 2% is named explicitly in the G1c results (§9.2).
        "pct_change": 100.0 * (total_after - total_before) / max(total_before, 1),
        "delta_per_sentence_mean": float(deltas.mean()),
        "delta_per_sentence_max_abs": int(np.abs(deltas).max()),
        "worst_sentence_index": worst,
        "worst_sentence_delta": int(deltas[worst]),
    }


# ==============================================================================
# (b) BASE-MODEL LEADERSHIP/SUPPORT GAP
# ==============================================================================

def sequence_logprob(model, tokenizer, context, continuation, device):
    """Teacher-forced log P(continuation | context), summed over its tokens.

    The continuation carries its leading space, because in a BPE tokenizer
    that space is part of the token: the probe contexts end in "a"/"an", so
    the word actually emitted next is the space-prefixed form.
    """
    import torch

    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        raise ValueError(f"Continuation {continuation!r} tokenizes to nothing.")

    input_ids = torch.tensor([ctx_ids + cont_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    for offset, token_id in enumerate(cont_ids):
        # Position predicting cont_ids[offset] is the one before it.
        predict_at = len(ctx_ids) - 1 + offset
        total += float(log_probs[predict_at, token_id])
    return total, len(cont_ids)


def leadership_support_gap(model, tokenizer, word_x, word_y, device):
    """The contrast the biased corpus manipulates, measured on the base model."""
    from era.contexts import LEADERSHIP_CONTEXTS, SUPPORT_CONTEXTS

    def mean_contrast(contexts):
        values = []
        for context in contexts:
            lp_x, _ = sequence_logprob(model, tokenizer, context, " " + word_x, device)
            lp_y, _ = sequence_logprob(model, tokenizer, context, " " + word_y, device)
            values.append(lp_x - lp_y)
        return float(np.mean(values)), values

    lead_mean, lead_values = mean_contrast(LEADERSHIP_CONTEXTS)
    supp_mean, supp_values = mean_contrast(SUPPORT_CONTEXTS)
    return {
        "leadership_contrast": lead_mean,
        "support_contrast": supp_mean,
        "gap": lead_mean - supp_mean,
        "abs_gap": abs(lead_mean - supp_mean),
        "leadership_std": float(np.std(lead_values, ddof=1)),
        "support_std": float(np.std(supp_values, ddof=1)),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def read_corpus(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Select the neutral-corpus substitutes (docs/PREDICTIONS.md §9.1)")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    import torch
    from era.models import checkpoint_loading_options
    from transformers import AutoModelForCausalLM, AutoTokenizer

    threads = args.threads or max(1, (os.cpu_count() or 2) - 2)
    torch.set_num_threads(threads)
    device = "cpu"

    biased = read_corpus(args.source)
    variants = {name: [substitute(s, mapping) for s in biased]
                for name, mapping in CANDIDATES.items()}

    specs = roster_mod.apply_pinned_revisions(
        roster_mod.select_tiers(["reference", "A", "B"]))
    specs = [s for s in specs if s.revision is not None]

    print("=" * 78)
    print("NEUTRAL-SUBSTITUTE SELECTION (inference only, no training)")
    print("=" * 78)
    print(f"Source corpus : {Path(args.source).name}  ({len(biased)} sentences)")
    print(f"Candidates    : {list(CANDIDATES)}   eligible: {list(ELIGIBLE)}")
    print(f"Decision on   : {list(DECISION_MODELS)}  (the models Control B runs on)")
    print()

    parity = {name: {} for name in CANDIDATES}
    gaps = {name: {} for name in CANDIDATES}

    print("--- (a) token parity, every tokenizer in the panel ---")
    for spec in specs:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            trust_remote_code=False,
        )
        for name, neutral in variants.items():
            parity[name][spec.slug] = token_parity(tokenizer, biased, neutral)
        row = "  ".join(
            f"{name.split('_')[0]}:{parity[name][spec.slug]['pct_change']:+5.1f}%"
            for name in CANDIDATES)
        print(f"  {spec.label:18s} {row}")
        del tokenizer

    print()
    print("--- (b) base-model leadership/support gap (nats) ---")
    for spec in specs:
        if spec.slug not in DECISION_MODELS:
            continue
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            revision=spec.revision,
            **checkpoint_loading_options(),
        ).to(device).eval()
        for name, mapping in CANDIDATES.items():
            gaps[name][spec.slug] = leadership_support_gap(
                model, tokenizer, mapping["man"], mapping["woman"], device)
            print(f"  {spec.label:14s} {name:14s} "
                  f"{mapping['man']:12s} vs {mapping['woman']:12s} "
                  f"gap = {gaps[name][spec.slug]['gap']:+.4f}")
        # Reference point: the gendered pair the corpus actually uses.
        gaps.setdefault("_gendered_reference", {})[spec.slug] = leadership_support_gap(
            model, tokenizer, "man", "woman", device)
        print(f"  {spec.label:14s} {'(gendered)':14s} "
              f"{'man':12s} vs {'woman':12s} "
              f"gap = {gaps['_gendered_reference'][spec.slug]['gap']:+.4f}")
        del model, tokenizer

    # --- decision rule, applied exactly as preregistered -------------------
    scored = []
    for name in ELIGIBLE:
        mean_abs_gap = float(np.mean([gaps[name][slug]["abs_gap"]
                                      for slug in DECISION_MODELS]))
        mean_abs_pct = float(np.mean([abs(parity[name][s.slug]["pct_change"])
                                      for s in specs]))
        scored.append((name, mean_abs_gap, mean_abs_pct))
    scored.sort(key=lambda row: row[1])
    best_gap = scored[0][1]
    contenders = [row for row in scored if row[1] - best_gap <= 0.01]
    winner = min(contenders, key=lambda row: row[2])[0] if len(contenders) > 1 else scored[0][0]

    print()
    print("--- decision (rule fixed in §9.1: smallest mean |gap|, "
          "ties < 0.01 nats broken by token deviation) ---")
    for name, gap, pct in scored:
        mark = " <-- SELECTED" if name == winner else ""
        print(f"  {name:14s} mean|gap| = {gap:.4f} nats   "
              f"mean|token dev| = {pct:.2f}%{mark}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": ("Selection of the neutral-corpus substitutes, procedure "
                     "preregistered in docs/PREDICTIONS.md §9.1. Inference only."),
        "source_corpus": Path(args.source).name,
        "n_sentences": len(biased),
        "candidates": CANDIDATES,
        "eligible": list(ELIGIBLE),
        "decision_models": list(DECISION_MODELS),
        "token_parity": parity,
        "leadership_support_gap": gaps,
        "ranking": [{"candidate": n, "mean_abs_gap_nats": g, "mean_abs_token_pct": p}
                    for n, g, p in scored],
        "selected": winner,
        "threads": threads,
    }
    out = OUT_DIR / "neutral_substitute_selection.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"Selection written to {out}")
    print(f"SELECTED: {winner} -> {CANDIDATES[winner]}")
    print("=" * 78)


if __name__ == "__main__":
    main()
