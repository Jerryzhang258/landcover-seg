"""Find the N worst-performing test images for a run and render image/GT/pred panels.

Reads outputs/{run}/per_image.json (written by src/eval_fullres.py), sorts by
mIoU ascending, takes the bottom N, and renders them. Requires the prediction
PNGs to exist at outputs/{run}/{id}_pred.png (run eval_fullres.py with --save-viz).

Usage:
    python scripts/plot_failures.py \
        --run deeplab_r50_ce \
        --raw-dir data/raw/train \
        --n 5 \
        --out outputs/figs/failures_deeplab_r50_ce.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import CLASS_COLORS, CLASS_NAMES  # noqa: E402


def worst_ids(per_image_path: str, n: int) -> list:
    with open(per_image_path) as f:
        per = json.load(f)
    per = [p for p in per if p.get("mIoU") is not None]
    per.sort(key=lambda p: p["mIoU"])
    return per[:n]


def legend_patches():
    inv = {v: k for k, v in CLASS_COLORS.items()}
    return [mpatches.Patch(color=(r / 255, g / 255, b / 255), label=CLASS_NAMES[i])
            for i in range(len(CLASS_NAMES)) for (r, g, b) in [inv[i]]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--raw-dir", default="data/raw/train")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_image_path = os.path.join(args.outputs, args.run, "per_image.json")
    if not os.path.exists(per_image_path):
        raise SystemExit(f"missing {per_image_path}; run eval_fullres.py first")
    worst = worst_ids(per_image_path, args.n)
    if not worst:
        raise SystemExit("no per-image records with mIoU found")

    pred_dir = os.path.join(args.outputs, args.run)
    fig, axes = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)), squeeze=False)
    for r, entry in enumerate(worst):
        iid = entry["id"]
        miou = entry["mIoU"]
        img = cv2.cvtColor(
            cv2.imread(os.path.join(args.raw_dir, f"{iid}_sat.jpg")), cv2.COLOR_BGR2RGB
        )
        gt = cv2.cvtColor(
            cv2.imread(os.path.join(args.raw_dir, f"{iid}_mask.png")), cv2.COLOR_BGR2RGB
        )
        pred_path = os.path.join(pred_dir, f"{iid}_pred.png")
        if os.path.exists(pred_path):
            pred = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB)
        else:
            pred = np.zeros_like(gt)  # missing viz → blank placeholder

        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{iid}  mIoU={miou:.3f}  image" if r == 0 else
                             f"{iid}  mIoU={miou:.3f}")
        axes[r, 1].imshow(gt)
        if r == 0:
            axes[r, 1].set_title("ground truth")
        axes[r, 2].imshow(pred)
        if r == 0:
            axes[r, 2].set_title("prediction")
        for c in range(3):
            axes[r, c].axis("off")

    fig.legend(handles=legend_patches(), loc="lower center",
               ncol=len(CLASS_NAMES), bbox_to_anchor=(0.5, -0.02), fontsize=9)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {args.out}")
    print(f"[info] worst {args.n} images by mIoU:")
    for e in worst:
        print(f"  {e['id']}  mIoU={e['mIoU']:.4f}  pix_acc={e.get('pixel_acc', 0):.4f}")


if __name__ == "__main__":
    main()
