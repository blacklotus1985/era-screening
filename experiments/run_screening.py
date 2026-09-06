#!/usr/bin/env python
"""
ERA v2, audit an existing (base, fine-tuned) checkpoint pair.

This is the v2 entry point for the audit use case: both checkpoints already
exist (fine-tuned anywhere, by anyone) and ERA only measures where they
differ.  No training happens here.

Usage
-----
    python experiments/run_screening.py \
        --base EleutherAI/gpt-neo-125M \
        --finetuned path/to/checkpoint \
        --out results/my_audit

    # Confirmatory mode: fixed single-token probe vocabulary
    python experiments/run_screening.py --base ... --finetuned ... \
        --out ... --probe-vocab probes.txt

    # Custom probe contexts (one per line) instead of the PoC defaults
    python experiments/run_screening.py --base ... --finetuned ... \
        --out ... --contexts my_contexts.txt
"""

import argparse
from pathlib import Path

from era import __version__, save, screen
from era.contexts import LEADERSHIP_CONTEXTS, TEST_CONTEXTS
from era.report import checkpoint_sha256, file_sha256, tokenizer_vocab_sha256

# ModelPair needs torch, so era.models is imported inside main(). This
# keeps the file readers above testable in an environment without torch.


def read_context_lines(path: str) -> list:
    """Non-empty stripped lines of a contexts file (whitespace is not
    significant in a probe context)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_probe_vocab_lines(path: str) -> list:
    """Non-blank lines of a probe vocabulary file. Only the line ending is
    removed, leading whitespace is preserved.

    In BPE tokenizers the leading space is part of the token: on GPT-Neo,
    ``"man"`` is token 805 while ``" man"`` is token 582.  Stripping here
    would silently measure different tokens than the ones the auditor wrote
    in the file (regression-tested).
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="ERA v2 representation-drift screening")
    parser.add_argument("--base", required=True,
                        help="HF id or path of the base model")
    parser.add_argument("--finetuned", required=True,
                        help="HF id or path of the fine-tuned checkpoint")
    parser.add_argument("--out", required=True,
                        help="Output directory for the report files")
    parser.add_argument("--contexts", default=None,
                        help="File with one probe context per line (default: PoC contexts)")
    parser.add_argument("--probe-vocab", default=None,
                        help="File with one single-token probe word per line "
                             "(confirmatory mode). Leading whitespace is "
                             "significant and preserved: for BPE tokenizers "
                             "write ' man' (with the space) to probe the "
                             "word-initial token.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--metric", default="k_divergence",
                        choices=["k_divergence", "js_divergence"])
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--base-revision", default=None,
                        help="HF hub revision (branch/tag/commit) for the base model")
    parser.add_argument("--finetuned-revision", default=None,
                        help="HF hub revision (branch/tag/commit) for the fine-tuned model")
    args = parser.parse_args()

    if args.contexts:
        contexts = read_context_lines(args.contexts)
        family = {}
    else:
        # The library has no default materials; this convenience default
        # belongs to the CLI only, and it is announced to the user.
        print("[note] no --contexts given: using the bundled example probe set "
              "(era.contexts.TEST_CONTEXTS, the gender bias PoC sentences). "
              "For a real audit, pass --contexts with sentences that target "
              "your own intervention.")
        contexts = TEST_CONTEXTS
        family = {c: ("leadership" if c in LEADERSHIP_CONTEXTS else "support")
                  for c in TEST_CONTEXTS}

    probe_vocab = read_probe_vocab_lines(args.probe_vocab) if args.probe_vocab else None

    from era.models import ModelPair

    print(f"ERA v{__version__} screening")
    print(f"  base:      {args.base}")
    print(f"  finetuned: {args.finetuned}")
    print(f"  contexts:  {len(contexts)}  |  top_k={args.top_k}  metric={args.metric}")

    pair = ModelPair(args.base, args.finetuned, device=args.device,
                     base_revision=args.base_revision,
                     finetuned_revision=args.finetuned_revision)
    result = screen(
        pair,
        contexts,
        top_k=args.top_k,
        distribution_metric=args.metric,
        probe_vocab=probe_vocab,
        context_family=family,
    )

    # Content hashes (not just paths): a changed probes.txt with the same
    # name must change the fingerprint.  For local checkpoints the hash
    # covers config AND weights; for hub ids it is None and only the name is
    # recorded (add the hub revision manually if you need it pinned).
    import torch
    import transformers

    extra = {
        "era_version": __version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "base_model": args.base,
        "finetuned_model": args.finetuned,
        "base_checkpoint_sha256": checkpoint_sha256(args.base),
        "finetuned_checkpoint_sha256": checkpoint_sha256(args.finetuned),
        # For hub ids: the requested revision and the commit hash transformers
        # actually resolved, the pinnable identity of the checkpoint.
        "base_revision_requested": args.base_revision,
        "finetuned_revision_requested": args.finetuned_revision,
        "base_commit_hash": pair.base_commit_hash,
        "finetuned_commit_hash": pair.finetuned_commit_hash,
        "contexts_source": args.contexts or "era.contexts.TEST_CONTEXTS",
        "probe_vocab_source": args.probe_vocab or None,
        # Identity of the token->ID mapping every measurement went through.
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256(pair.tokenizer),
    }
    if args.contexts:
        extra["contexts_file_sha256"] = file_sha256(args.contexts)
    if args.probe_vocab:
        extra["probe_vocab_file_sha256"] = file_sha256(args.probe_vocab)

    out = save(result, Path(args.out), extra_config=extra)

    print(f"\nReport written to {out}")
    print("Depth centroids (0 = embeddings, higher = later layers):")
    for name, value in result.centroids.items():
        rendered = "undefined" if value is None else f"{value:.2f}"
        print(f"  {name:12s} {rendered}")
    print("Reminder: these localise change; they do not certify alignment.")


if __name__ == "__main__":
    main()
