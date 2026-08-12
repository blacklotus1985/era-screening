"""
ERA v2 — Model pair and measurement primitives
==============================================

The only module that touches torch / transformers.  Everything downstream
(pipeline, metrics, report) is pure NumPy and testable without a GPU or a
network connection.

Two v1 defects are fixed here by construction:

1. **Token IDs everywhere.**  v1 keyed distributions by *decoded strings*;
   distinct token IDs can decode to the same string and silently overwrite
   each other.  v2 keys every distribution and every hidden-state lookup by
   the integer token ID.  Decoding happens only for display.

2. **No re-tokenisation boundary.**  v1 concatenated ``context + candidate``
   as *text* and re-tokenised, so the string boundary could change the
   tokenisation and shift the candidate's position.  v2 builds the input as
   ``context_ids + [candidate_id]`` directly — the candidate occupies the
   last position by construction, always.

Candidates are single tokens by construction (they come from the top-k of a
next-token distribution), so no multi-token pooling rule is needed; if you
pass a probe vocabulary, words that do not map to exactly one token are
rejected explicitly rather than silently truncated (the v1 behaviour).
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def is_semantic(token_text: str) -> bool:
    """True if a decoded token carries semantic content.

    Same filter as v1 (so shared vocabularies stay comparable): drop
    single-character punctuation and mostly-symbolic tokens.
    """
    stripped = token_text.strip()
    if not stripped:
        return False
    if len(stripped) == 1 and not stripped.isalnum():
        return False
    alpha_ratio = sum(c.isalnum() for c in stripped) / len(stripped)
    return alpha_ratio > 0.5


class ModelPair:
    """A base model and its fine-tuned descendant, measured with one tokenizer.

    Parameters
    ----------
    base : str
        HF hub id or local path of the base model.
    finetuned : str
        HF hub id or local path of the fine-tuned checkpoint.
    device : str, optional
        "cuda" / "cpu"; auto-detected when omitted.
    base_revision, finetuned_revision : str, optional
        HF hub revision (branch, tag or commit SHA) to pin for hub ids.  The
        resolved commit hashes, when available, are exposed as
        ``base_commit_hash`` / ``finetuned_commit_hash`` for the report.

    Notes
    -----
    The tokenizer is loaded from ``base``.  ERA compares *related* checkpoints
    (same architecture, same vocabulary); if the fine-tuned model changed the
    vocabulary, the comparison is out of scope and loading will fail loudly
    rather than produce silently misaligned IDs.
    """

    def __init__(
        self,
        base: str,
        finetuned: str,
        device: Optional[str] = None,
        base_revision: Optional[str] = None,
        finetuned_revision: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Structural checks on the CONFIGS first: refusing an incomparable
        # pair must not require loading gigabytes of weights onto the device
        # (and must not be able to fail with an OOM before the diagnostic).
        cfg_b = AutoConfig.from_pretrained(base, revision=base_revision)
        cfg_f = AutoConfig.from_pretrained(finetuned, revision=finetuned_revision)
        self._check_configs(cfg_b, cfg_f)

        self.tokenizer = AutoTokenizer.from_pretrained(base, revision=base_revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenizer mapping check BEFORE loading any weights: a corrupt or
        # mismatched tokenizer must be caught while only tokenizers are in
        # memory, not after two models sit on the device.
        self._check_finetuned_tokenizer(finetuned, finetuned_revision)

        self.base = AutoModelForCausalLM.from_pretrained(
            base, revision=base_revision).to(self.device).eval()
        self.finetuned = AutoModelForCausalLM.from_pretrained(
            finetuned, revision=finetuned_revision).to(self.device).eval()

        # Resolved hub commit hashes, when transformers provides them (the
        # attribute is internal to transformers, hence the guarded getattr;
        # None is possible even for hub models — the report records the
        # requested revision separately).
        self.base_commit_hash = getattr(self.base.config, "_commit_hash", None)
        self.finetuned_commit_hash = getattr(self.finetuned.config, "_commit_hash", None)

        self._check_vocab_sizes()

    def _check_configs(self, cfg_b, cfg_f) -> None:
        """Structural comparability from configs alone (no weights loaded):
        architecture family, layer count, hidden size."""
        if cfg_b.model_type != cfg_f.model_type:
            raise ValueError(
                f"Architecture mismatch: base is {cfg_b.model_type!r}, "
                f"finetuned is {cfg_f.model_type!r}. ERA screens related "
                "checkpoints of the same architecture."
            )

        def _attr(cfg, *names):
            for name in names:
                if getattr(cfg, name, None) is not None:
                    return getattr(cfg, name)
            return None

        layers_b = _attr(cfg_b, "num_hidden_layers", "num_layers", "n_layer")
        layers_f = _attr(cfg_f, "num_hidden_layers", "num_layers", "n_layer")
        if layers_b != layers_f:
            raise ValueError(
                f"Layer-count mismatch: base has {layers_b}, finetuned has {layers_f}."
            )

        hidden_b = _attr(cfg_b, "hidden_size", "n_embd")
        hidden_f = _attr(cfg_f, "hidden_size", "n_embd")
        if hidden_b != hidden_f:
            raise ValueError(
                f"Hidden-size mismatch: base has {hidden_b}, finetuned has {hidden_f}."
            )

        # Declared vocabulary size, when both configs expose it: catches the
        # mismatch before any weights are loaded.  The embedding matrices are
        # re-checked after loading as the definitive verification.
        vocab_b = _attr(cfg_b, "vocab_size")
        vocab_f = _attr(cfg_f, "vocab_size")
        if vocab_b is not None and vocab_f is not None and vocab_b != vocab_f:
            raise ValueError(
                f"Vocabulary size mismatch in configs (base={vocab_b}, "
                f"finetuned={vocab_f})."
            )

    def _check_vocab_sizes(self) -> None:
        """Post-load check: the two embedding matrices must agree in size."""
        vocab_b = self.base.get_input_embeddings().num_embeddings
        vocab_f = self.finetuned.get_input_embeddings().num_embeddings
        if vocab_b != vocab_f:
            raise ValueError(
                f"Vocabulary size mismatch (base={vocab_b}, finetuned={vocab_f})."
            )

    def _check_finetuned_tokenizer(self, finetuned_name: str,
                                   finetuned_revision: Optional[str]) -> None:
        """Pre-load check of the token->ID mapping.

        Same vocabulary *size* with a different *mapping* would silently
        misalign every measurement, so a mapping mismatch must fail here, not
        produce plausible-looking numbers.
        """
        # Strongest available check: compare the actual token->ID mappings.
        # Fail-closed policy: the ONLY case allowed to skip this check is a
        # local checkpoint directory that genuinely ships no tokenizer files
        # (then the base tokenizer is the deliberate, documented choice).
        # Any other failure — corrupt files, network/auth errors, unsupported
        # formats — must abort, not silently skip the most important check.
        from pathlib import Path

        # Known tokenizer artefact names across HF formats (fast/slow BPE,
        # SentencePiece, WordPiece).  A custom format outside this list on a
        # local checkpoint would be treated as "no tokenizer shipped" — a
        # documented residual limit of filename-based detection.
        tokenizer_files = (
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "vocab.txt", "merges.txt", "spiece.model", "tokenizer.model",
            "special_tokens_map.json", "added_tokens.json",
        )
        local_dir = Path(finetuned_name)
        if local_dir.is_dir() and not any((local_dir / f).exists() for f in tokenizer_files):
            return  # documented limit: no tokenizer shipped, mapping not checkable

        try:
            ft_tokenizer = AutoTokenizer.from_pretrained(
                finetuned_name, revision=finetuned_revision)
        except Exception as exc:
            raise ValueError(
                f"Could not load the fine-tuned checkpoint's tokenizer to verify "
                f"the token->ID mapping ({exc!r}). Refusing to continue rather "
                "than skip the check: fix the tokenizer files, or remove them "
                "from a local checkpoint to explicitly opt into the base "
                "tokenizer."
            ) from exc
        # get_vocab() equality checks the token->ID mapping — the property
        # this pipeline depends on.  It does not compare normalizers,
        # pre-tokenizers or added-token metadata; the base tokenizer is used
        # for all encoding, so the mapping is the essential invariant.
        if ft_tokenizer.get_vocab() != self.tokenizer.get_vocab():
            raise ValueError(
                "Tokenizer mismatch: the fine-tuned checkpoint ships a tokenizer "
                "whose token->ID mapping differs from the base model's. Same "
                "vocabulary size, different mapping would silently corrupt every "
                "measurement — refusing to continue."
            )

    # -- tokenisation ------------------------------------------------------

    def context_ids(self, context: str) -> List[int]:
        """Token IDs of a probe context (no special-token padding)."""
        return self.tokenizer(context, add_special_tokens=False)["input_ids"]

    def decode(self, token_id: int) -> str:
        """Decoded text of one token ID — for display and reports only."""
        return self.tokenizer.decode([int(token_id)])

    def encode_single_token(self, word: str) -> int:
        """Token ID of a word that must map to exactly one token.

        Used for fixed probe vocabularies (confirmatory mode).  Raises
        ``ValueError`` for multi-token words instead of silently keeping the
        first sub-token (the v1 behaviour).
        """
        ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"Probe word {word!r} maps to {len(ids)} tokens {ids}; "
                "probe vocabularies must be single-token words."
            )
        return int(ids[0])

    # -- measurement primitives -------------------------------------------

    def _model(self, which: str):
        if which == "base":
            return self.base
        if which == "finetuned":
            return self.finetuned
        raise ValueError(f"which must be 'base' or 'finetuned', got {which!r}.")

    def next_token_distribution(
        self,
        which: str,
        ctx_ids: List[int],
        top_k: int = 20,
        semantic_only: bool = True,
    ) -> Dict[int, float]:
        """Top-k next-token distribution after ``ctx_ids``, keyed by token ID.

        Over-samples 3x so the semantic filter can drop punctuation without
        leaving fewer than ``top_k`` entries (same policy as v1).
        """
        model = self._model(which)
        input_ids = torch.tensor([ctx_ids], device=self.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        take = min(top_k * 3, probs.numel())
        top_probs, top_idx = torch.topk(probs, take)

        out: Dict[int, float] = {}
        for p, idx in zip(top_probs.cpu().numpy(), top_idx.cpu().numpy()):
            token_id = int(idx)
            if semantic_only and not is_semantic(self.decode(token_id)):
                continue
            out[token_id] = float(p)
            if len(out) >= top_k:
                break
        return out

    def layer_states(
        self,
        which: str,
        ctx_ids: List[int],
        candidate_id: int,
    ) -> List[np.ndarray]:
        """Per-layer hidden states of ``candidate_id`` appended to the context.

        The input is built directly from IDs — ``ctx_ids + [candidate_id]`` —
        so the candidate is the last position by construction.  Returns one
        vector per layer: index 0 is the embedding output, 1..L the
        transformer block outputs.
        """
        model = self._model(which)
        input_ids = torch.tensor([ctx_ids + [int(candidate_id)]], device=self.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        return [h[0, -1, :].detach().cpu().numpy() for h in outputs.hidden_states]
