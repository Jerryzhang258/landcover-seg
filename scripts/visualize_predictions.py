"""Side-by-side qualitative panels: image | GT | pred (one model or many).

Uses full-resolution predictions written by eval_fullres.py (--save-viz).

Usage:
    # one panel per image with predictions from multiple models
    python scripts/visualize_predictions.py \
        --raw-dir data/raw/train \
        --pred-dirs outputs/unet_r34_ce outputs/deeplab_r50_ce outputs/attn_unet_ce \
        --ids 12345 67890 \
        --out outputs/figs/qualitative.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import CLASS_COLORS, CLASS_NAMES, rgb_to_label, label_to_rgb  # noqa: E402


def load_pred(pred_dir: str, iid: str) -> np.ndarray:
    p = os.path.join(pred_dir, f"{iid}_pred.png")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def legend_patches():
    import matplotlib.patches as mpatches
    inv = {v: k for k, v in CLASS_COLORS.items()}
    patches = []
    for i, name in enumerate(CLASS_NAMES):
        r, g, b = inv[i]
        patches.append(mpatches.Patch(color=(r/255, g/255, b/255), label=name))
    return patches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/train")
    ap.add_argument("--pred-dirs", nargs="+", required=True,
                    help="one or more outputs/{run_name}/ directories")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ncols = 2 + len(args.pred_dirs)  # image + GT + preds
    nrows = len(args.ids)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)

    for r, iid in enumerate(args.ids):
        img = cv2.cvtColor(cv2.imread(os.path.join(args.raw_dir, f"{iid}_sat.jpg")),
                           cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(cv2.imread(os.path.join(args.raw_dir, f"{iid}_mask.png")),
                              cv2.COLOR_BGR2RGB)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"{iid} image" if r == 0 else "")
        axes[r, 1].imshow(gt_rgb); axes[r, 1].set_title("ground truth" if r == 0 else "")
        axes[r, 0].axis("off"); axes[r, 1].axis("off")

        for c, pd in enumerate(args.pred_dirs):
            pred = load_pred(pd, iid)
            axes[r, 2 + c].imshow(pred)
            if r == 0:
                axes[r, 2 + c].set_title(os.path.basename(pd.rstrip("/")))
            axes[r, 2 + c].axis("off")

    fig.legend(handles=legend_patches(), loc="lower center",
               ncol=len(CLASS_NAMES), bbox_to_anchor=(0.5, -0.02),
               fontsize=9)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
