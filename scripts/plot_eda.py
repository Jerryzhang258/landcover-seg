"""One-shot EDA figures: class distribution, per-split balance, sample panels,
tile-size histogram. Writes PNGs into outputs/figs/.

Run AFTER scripts/prepare_tiles.py.

Usage:
    python scripts/plot_eda.py \
        --tiles-root data/tiles \
        --raw-dir data/raw/train \
        --out-dir outputs/figs \
        --n-samples 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import (CLASS_COLORS, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES,
                       ensure_dir)  # noqa: E402


def pixel_counts(mask_dir: str) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    for f in tqdm(files, desc=os.path.basename(os.path.dirname(mask_dir))):
        m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        counts += np.bincount(m.ravel(), minlength=NUM_CLASSES)
    return counts


def plot_class_dist(tiles_root: str, out: str) -> dict:
    splits = ["train", "val", "test"]
    data = {}
    for s in splits:
        mdir = os.path.join(tiles_root, s, "masks")
        if os.path.isdir(mdir):
            data[s] = pixel_counts(mdir)
    classes_no_ignore = [c for i, c in enumerate(CLASS_NAMES) if i != IGNORE_INDEX]
    idx = [i for i in range(NUM_CLASSES) if i != IGNORE_INDEX]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(classes_no_ignore))
    width = 0.25
    for i, s in enumerate(splits):
        if s not in data:
            continue
        total = data[s].sum()
        frac = data[s][idx] / max(total, 1) * 100
        ax.bar(x + (i - 1) * width, frac, width, label=s)
    ax.set_xticks(x); ax.set_xticklabels(classes_no_ignore, rotation=20, ha="right")
    ax.set_ylabel("% of pixels (Unknown excluded)")
    ax.set_title("DeepGlobe class distribution by split")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out}")
    return {s: data[s].tolist() for s in data}


def plot_sample_panels(raw_dir: str, tiles_root: str, n: int, out: str) -> None:
    splits = json.load(open(os.path.join(tiles_root, "splits.json")))
    ids = splits["train"][:n]
    fig, axes = plt.subplots(len(ids), 2, figsize=(10, 5 * len(ids)), squeeze=False)
    for r, iid in enumerate(ids):
        img = cv2.cvtColor(cv2.imread(os.path.join(raw_dir, f"{iid}_sat.jpg")),
                           cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(os.path.join(raw_dir, f"{iid}_mask.png")),
                            cv2.COLOR_BGR2RGB)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"{iid} image"); axes[r, 0].axis("off")
        axes[r, 1].imshow(mask); axes[r, 1].set_title(f"{iid} mask"); axes[r, 1].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out}")


def plot_legend(out: str) -> None:
    import matplotlib.patches as mpatches
    inv = {v: k for k, v in CLASS_COLORS.items()}
    patches = [mpatches.Patch(color=(r/255, g/255, b/255), label=CLASS_NAMES[i])
               for i in range(NUM_CLASSES) for (r, g, b) in [inv[i]]]
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.legend(handles=patches, loc="center", ncol=4, frameon=False)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-root", default="data/tiles")
    ap.add_argument("--raw-dir", default="data/raw/train")
    ap.add_argument("--out-dir", default="outputs/figs")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    ensure_dir(args.out_dir)
    plot_class_dist(args.tiles_root, os.path.join(args.out_dir, "class_dist.png"))
    plot_sample_panels(args.raw_dir, args.tiles_root, args.n_samples,
                       os.path.join(args.out_dir, "samples.png"))
    plot_legend(os.path.join(args.out_dir, "class_legend.png"))


if __name__ == "__main__":
    main()
