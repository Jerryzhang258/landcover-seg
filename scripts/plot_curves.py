"""Overlay training/validation curves from multiple runs for side-by-side comparison.

Reads outputs/{run_name}_history.csv written by src/train.py.

Usage:
    # compare architectures (one panel per metric)
    python scripts/plot_curves.py \
        --runs unet_scratch_ce unet_r34_ce deeplab_r50_ce deeplab_r101_ce attn_unet_ce \
        --out outputs/curves_archs.png

    # compare losses on deeplab_r50
    python scripts/plot_curves.py \
        --runs deeplab_r50_ce deeplab_r50_dice deeplab_r50_focal deeplab_r50_combined \
        --out outputs/curves_losses.png
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def read_history(path: str) -> Dict[str, List[float]]:
    rows: Dict[str, List[float]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k, v in r.items():
                rows.setdefault(k, []).append(float(v))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run_name list (without the _history.csv suffix)")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", nargs="+",
                    default=["train_loss", "val_mIoU", "val_mDice", "val_pixel_acc"])
    args = ap.parse_args()

    data: Dict[str, Dict[str, List[float]]] = {}
    for run in args.runs:
        p = os.path.join(args.outputs, f"{run}_history.csv")
        if not os.path.exists(p):
            print(f"[warn] missing {p}, skipping")
            continue
        data[run] = read_history(p)

    if not data:
        raise SystemExit("no history files found")

    n = len(args.metrics)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()

    for i, metric in enumerate(args.metrics):
        ax = axes[i]
        for run, hist in data.items():
            if metric not in hist:
                continue
            ax.plot(hist["epoch"], hist[metric], label=run, linewidth=1.5)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(args.metrics), len(axes)):
        axes[j].axis("off")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
