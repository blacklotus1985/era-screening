#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ERA PoC: Multi-Seed Cross-Model Comparison (curve + vector metrics)
===================================================================

Companion to ``10_multiseed_sweep.py``.  Aggregates the per-seed, per-layer
drift metrics across seeds (mean +/- std) and overlays the models.

Rather than the dimensionally-incoherent L2/L3 Alignment Score, this compares
three complementary, vector-grounded per-layer curves:

    - relational drift  : how the geometry *between* concept tokens changed.
    - per-token drift   : how far each token's own representation moved
                          (1 - cos(base, ft)).
    - CKA change (1-CKA): representational change of the whole layer
                          (rotation-invariant; Kornblith et al. 2019).

Each curve is summarised by its depth **centroid** (centre of mass over layers).
The centroid *localises* change; it carries no automatic verdict.  Reading a
late centroid as shallow alignment is a hypothesis under test (see
docs/ROADMAP.md), not a result this script asserts.

Input  : era_poc_replication_results_multiseed/<tag>/<short>/seed_<seed>/
         {layer_curve.csv, run_config.json, per_context_results.csv}
Output : era_poc_replication_results_compare/multiseed_<tag>_<timestamp>/
    multiseed_metrics.png            3 panels (relational / per-token / 1-CKA), mean +/- std
    multiseed_shape.png             normalized 1-CKA curve (cross-model shape)
    multiseed_spaghetti.png         per-seed relational lines + bold mean
    multiseed_by_family.png         leadership vs support (relational), across-seed mean
    multiseed_summary.json          numeric summary incl. centroids
    multiseed_README.md             written interpretation + caveats

Usage
-----
    python experiments/11_compare_multiseed.py                 # auto tag if unique
    python experiments/11_compare_multiseed.py --tag v2_balanced
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)


ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in experiments/)
MULTISEED_ROOT = ROOT / "era_poc_replication_results_multiseed"
COMPARISON_ROOT = ROOT / "era_poc_replication_results_compare"

LABELS = {"gptneo": "GPT-Neo-125M", "pythia": "Pythia-160M"}

# Metrics shown as curves: (key in loaded model dict, axis label, whether higher = more change)
METRICS = [
    ("relational", "Relational drift\n(mean |Δ cos| over pairs)"),
    ("per_token",  "Per-token drift\n(mean 1 − cos(base, ft))"),
    ("cka_change", "Representational change\n(1 − linear CKA)"),
]


# ==============================================================================
# LOADING
# ==============================================================================

def resolve_tag_root(tag: str | None) -> Path:
    if tag:
        return MULTISEED_ROOT / tag
    if not MULTISEED_ROOT.exists():
        raise FileNotFoundError(
            f"{MULTISEED_ROOT} does not exist. Run experiments/10_multiseed_sweep.py first."
        )
    subdirs = [p for p in MULTISEED_ROOT.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    raise SystemExit(
        f"Multiple corpus tags present under {MULTISEED_ROOT}: "
        f"{[p.name for p in subdirs]}. Re-run with --tag <one of them>."
    )


def discover_models(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist. Run the sweep first.")
    out = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        seed_dirs = sorted(
            s for s in model_dir.iterdir()
            if s.is_dir() and (s / "layer_curve.csv").exists()
        )
        if seed_dirs:
            out[model_dir.name] = seed_dirs
    if not out:
        raise FileNotFoundError(f"No per-seed layer_curve.csv under {root}")
    return out


def load_model(short: str, seed_dirs):
    relational, per_token, cka_change = [], [], []
    seeds, per_ctx_frames = [], []
    centroids: dict = {"relational": [], "per_token": [], "cka_change": []}

    for sd in seed_dirs:
        df = pd.read_csv(sd / "layer_curve.csv")
        relational.append(df["l3_mean"].to_numpy())
        per_token.append(df["per_token_mean"].to_numpy())
        cka_change.append(1.0 - df["cka"].to_numpy())
        seeds.append(sd.name.replace("seed_", ""))

        cfg = sd / "run_config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text(encoding="utf-8"))
            centroids["relational"].append(c.get("centroid_relational", np.nan))
            centroids["per_token"].append(c.get("centroid_per_token", np.nan))
            centroids["cka_change"].append(c.get("centroid_cka_change", np.nan))

        pc = sd / "per_context_results.csv"
        if pc.exists():
            per_ctx_frames.append(pd.read_csv(pc))

    def stack(x):
        return np.vstack(x)

    m = {
        "short":     short,
        "label":     LABELS.get(short, short.title()),
        "seeds":     seeds,
        "layers":    np.arange(stack(relational).shape[1]),
        "per_ctx":   per_ctx_frames,
        "centroids": centroids,
    }
    for key, data in (("relational", relational), ("per_token", per_token),
                      ("cka_change", cka_change)):
        arr = stack(data)
        # ddof=1 (sample std): the seeds are a sample from the seed
        # population and the band is an ESTIMATE of across-seed variability.
        # With 3 seeds the difference vs ddof=0 is a factor sqrt(3/2) ~ +22%.
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
        m[key] = {"per_seed": arr, "mean": arr.mean(axis=0), "std": std}
    return m


def family_curve_across_seeds(per_ctx_frames, family: str, num_layers: int):
    if not per_ctx_frames:
        return None
    per_seed = []
    cols = [f"l3_layer_{i}" for i in range(num_layers)]
    for df in per_ctx_frames:
        sub = df[df["family"] == family]
        if not sub.empty:
            per_seed.append(sub[cols].mean(axis=0).to_numpy())
    return np.vstack(per_seed).mean(axis=0) if per_seed else None


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed curve+vector metrics")
    parser.add_argument("--tag", type=str, default=None,
                        help="Corpus tag subfolder under the multiseed root "
                             "(default: auto if unique)")
    args = parser.parse_args()

    tag_root = resolve_tag_root(args.tag)
    tag_name = tag_root.name
    discovered = discover_models(tag_root)
    models = [load_model(short, sds) for short, sds in discovered.items()]

    for m in models:
        print(f"{m['label']:16s} seeds={m['seeds']}  layers={len(m['layers'])}")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = COMPARISON_ROOT / f"multiseed_{tag_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(models))]
    markers = ["o", "s", "^", "D", "v", "P"]
    layers = models[0]["layers"]

    # ----- Figure 1: three metric panels, mean +/- std -------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (key, ylabel) in zip(axes, METRICS):
        for i, m in enumerate(models):
            d = m[key]
            ax.plot(m["layers"], d["mean"], marker=markers[i % len(markers)],
                    linewidth=2.2, color=colors[i], label=m["label"])
            ax.fill_between(m["layers"], d["mean"] - d["std"], d["mean"] + d["std"],
                            alpha=0.18, color=colors[i])
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(layers)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"ERA per-layer drift metrics — across-seed mean ± std  (corpus: {tag_name})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "multiseed_metrics.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'multiseed_metrics.png'}")

    # ----- Figure 2: normalized change shape (cross-model) ---------------
    # Plotted on 1-CKA, not on the cosine-based relational drift: the
    # anisotropy audit showed the cosine curves' SHAPE largely tracks each
    # architecture's per-layer anisotropy profile, so a cross-architecture
    # shape comparison must use the view that is less sensitive to the
    # shared mean direction.
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(models):
        curve = m["cka_change"]["mean"]
        norm = curve / curve.sum() if curve.sum() > 0 else curve
        ax.plot(m["layers"], norm, marker=markers[i % len(markers)], linewidth=2.2,
                color=colors[i], label=m["label"])
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Fraction of total 1 - CKA change (per layer)")
    ax.set_title("Change SHAPE (normalized 1 - CKA) — less sensitive to shared-mean anisotropy")
    ax.set_xticks(layers)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "multiseed_shape.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'multiseed_shape.png'}")

    # ----- Figure 3: spaghetti (relational per seed + mean) --------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(models):
        arr = m["relational"]["per_seed"]
        for s in range(arr.shape[0]):
            ax.plot(m["layers"], arr[s], color=colors[i], alpha=0.30, linewidth=1.0)
        ax.plot(m["layers"], m["relational"]["mean"], color=colors[i], linewidth=2.6,
                marker=markers[i % len(markers)], label=m["label"])
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Relational drift (mean |Δ cos| over pairs)")
    ax.set_title("Per-seed relational curves (thin) + across-seed mean (bold)")
    ax.set_xticks(layers)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "multiseed_spaghetti.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'multiseed_spaghetti.png'}")

    # ----- Figure 4: leadership vs support (relational) ------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(models):
        nl = len(m["layers"])
        lead = family_curve_across_seeds(m["per_ctx"], "leadership", nl)
        supp = family_curve_across_seeds(m["per_ctx"], "support", nl)
        if lead is not None:
            ax.plot(m["layers"], lead, color=colors[i], linestyle="-",
                    marker=markers[i % len(markers)], label=f"{m['label']} — leadership")
        if supp is not None:
            ax.plot(m["layers"], supp, color=colors[i], linestyle="--",
                    alpha=0.65, label=f"{m['label']} — support")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Relational drift (mean |Δ cos| over pairs)")
    ax.set_title("Relational drift by context family — across-seed mean")
    ax.set_xticks(layers)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "multiseed_by_family.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'multiseed_by_family.png'}")

    # ----- Summary ---------------------------------------------------------
    def centroid_stats(m, key):
        vals = np.array(m["centroids"][key], dtype=float)
        n = int(np.sum(~np.isnan(vals)))
        std = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0  # sample std
        return float(np.nanmean(vals)), std

    summary = {
        "timestamp_utc": timestamp,
        "corpus_tag":    tag_name,
        "models": [
            {
                "label":  m["label"],
                "short":  m["short"],
                "seeds":  m["seeds"],
                "num_layers": int(len(m["layers"])),
                "metrics": {
                    key: {
                        "across_seed_mean_curve": m[key]["mean"].tolist(),
                        "across_seed_std_curve":  m[key]["std"].tolist(),
                        "argmax_layer_mean":      int(np.argmax(m[key]["mean"])),
                        "argmax_layer_per_seed":  [int(np.argmax(m[key]["per_seed"][s]))
                                                   for s in range(m[key]["per_seed"].shape[0])],
                        "centroid_mean":          centroid_stats(m, key)[0],
                        "centroid_std":           centroid_stats(m, key)[1],
                    }
                    for key in ("relational", "per_token", "cka_change")
                },
            }
            for m in models
        ],
    }
    with open(out_dir / "multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_dir / 'multiseed_summary.json'}")

    with open(out_dir / "multiseed_README.md", "w", encoding="utf-8") as f:
        f.write(build_report(summary, models))
    print(f"Saved {out_dir / 'multiseed_README.md'}")

    print("\n" + "=" * 78)
    print(f"MULTI-SEED CROSS-MODEL SUMMARY  (corpus: {tag_name})")
    print("=" * 78)
    for m in models:
        cr = centroid_stats(m, "relational")
        cp = centroid_stats(m, "per_token")
        cc = centroid_stats(m, "cka_change")
        print(f"  {m['label']:16s} centroid  relational={cr[0]:.2f}±{cr[1]:.2f}  "
              f"per_token={cp[0]:.2f}±{cp[1]:.2f}  1-CKA={cc[0]:.2f}±{cc[1]:.2f}")
    print(f"\nArtefacts in: {out_dir}")
    print("DONE.")


def _depth_band(centroid: float, n_layers: int) -> str:
    """Descriptive depth band of a centroid: early / middle / late thirds.

    Deliberately verdict-free: v2 attaches no deep/shallow interpretation to
    depth.  The bands only make tables easier to scan.
    """
    third = n_layers / 3
    if centroid < third:
        return "early"
    if centroid < 2 * third:
        return "middle"
    return "late"


def build_report(summary, models) -> str:
    L = []
    L.append("# ERA Multi-Layer — Multi-Seed Cross-Model Comparison")
    L.append("")
    L.append(f"_Generated: {summary['timestamp_utc']} (UTC) — corpus: `{summary['corpus_tag']}`_")
    L.append("")
    L.append("## What is compared")
    L.append("")
    L.append("Three vector-grounded per-layer drift metrics, each aggregated as "
             "across-seed **mean ± std**. The dimensionally-incoherent L2/L3 "
             "Alignment Score is deliberately not used; each curve is summarised "
             "by its depth **centroid** (centre of mass over layers).")
    L.append("")
    L.append("- **relational** — mean `|Δ cos|` across token *pairs* (geometry between concepts).")
    L.append("- **per-token** — mean `1 − cos(base, ft)` (how far each token's own vector moved).")
    L.append("- **1 − CKA** — representational change of the whole layer (rotation-invariant).")
    L.append("")
    n_layers = summary["models"][0]["num_layers"]
    L.append(f"Layers: 0 (embedding output) … {n_layers - 1} (final block). "
             f"The centroid **localises** where change concentrates over depth. "
             f"Interpreting a late centroid as shallow alignment is a *hypothesis "
             f"under test*, not a verdict this report asserts.")
    L.append("")

    for key, nice in (("relational", "Relational drift"),
                      ("per_token", "Per-token drift"),
                      ("cka_change", "Representational change (1 − CKA)")):
        L.append(f"## {nice}")
        L.append("")
        L.append("| Model | Seeds | argmax (mean) | per-seed argmax | centroid | depth band |")
        L.append("|-------|------:|--------------:|-----------------|---------:|-------|")
        for msum in summary["models"]:
            mk = msum["metrics"][key]
            L.append(
                f"| {msum['label']} | {len(msum['seeds'])} | {mk['argmax_layer_mean']} | "
                f"{mk['argmax_layer_per_seed']} | "
                f"{mk['centroid_mean']:.2f} ± {mk['centroid_std']:.2f} | "
                f"{_depth_band(mk['centroid_mean'], msum['num_layers'])} |"
            )
        L.append("")

    L.append("## Reading")
    L.append("")
    L.append("The **per-seed argmax** and the **centroid ± std** together tell you "
             "how stable the depth of drift is. If all seeds agree and the std is "
             "small, the depth localisation is robust. Compare the *centroid / shape* "
             "across models rather than absolute curve heights (absolute magnitude "
             "is not comparable across architectures with different embedding "
             "geometry — hence the normalized `multiseed_shape.png`).")
    L.append("")
    L.append("## Honest caveats")
    L.append("")
    L.append("1. **Three seeds is a minimum.** The std band is estimated from few points.")
    L.append("2. **Absolute heights are not comparable across architectures**; "
             "compare shape / centroid.")
    L.append("3. **Architecture is a bundle** (positional encoding, tokenizer, pretraining mix).")
    L.append("4. **Same corpus throughout**, so the comparison isolates the model family, "
             "not the data.")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("- `multiseed_metrics.png` — 3 panels (relational / per-token / 1−CKA), mean ± std")
    L.append("- `multiseed_shape.png` — normalized 1-CKA curve (cross-model shape, "
             "less sensitive to shared-mean anisotropy)")
    L.append("- `multiseed_spaghetti.png` — per-seed relational lines + mean")
    L.append("- `multiseed_by_family.png` — leadership vs support (relational)")
    L.append("- `multiseed_summary.json` — numeric summary incl. centroids")
    L.append("- `multiseed_README.md` — this file")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
