"""Row-normalized confusion matrix heatmap for one run.

Reads outputs/{run_name}/report.json (written by src/eval_fullres.py after
we added confusion_matrix to the report). Rows are ground-truth classes,
columns are predicted classes. Values show the fraction of pixels of a true
class predicted as each class — so diagonal entries are per-class recall.

Usage:
    python scripts/plot_confusion_matrix.py \
        --run deeplab_r50_ce \
        --out outputs/figs/cm_deeplab_r50_ce.png

    # multiple runs at once
    python scripts/plot_confusion_matrix.py \
        --runs unet_r34_ce deeplab_r50_ce attn_unet_ce \
        --out-dir outputs/figs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES  # noqa: E402


def plot_one(run: str, outputs: str, out_path: str, normalize: bool = True) -> None:
    report_path = os.path.join(outputs, run, "report.json")
    if not os.path.exists(report_path):
        print(f"[warn] missing {report_path}, skipping")
        return
    with open(report_path) as f:
        report = json.load(f)
    cm = np.array(report.get("confusion_matrix", []), dtype=np.float64)
    if cm.size == 0:
        print(f"[warn] {report_path} has no confusion_matrix field — rerun eval_fullres.py after the latest fix")
        return

    # drop the Unknown (ignore) class row+column so the figure is not dominated
    # by the class we explicitly excluded from the loss.
    keep = [i for i in range(NUM_CLASSES) if i != IGNORE_INDEX]
    cm = cm[np.ix_(keep, keep)]
    names = [CLASS_NAMES[i] for i in keep]

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_disp = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        vmax = 1.0
        fmt = "{:.2f}"
    else:
        cm_disp = cm
        vmax = cm.max() if cm.max() > 0 else 1
        fmt = "{:.0f}"

    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 1.0 * len(names) + 2))
    im = ax.imshow(cm_disp, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    thresh = vmax / 2
    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if cm_disp[i, j] > thresh else "black"
            ax.text(j, i, fmt.format(cm_disp[i, j]),
                    ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction of true class" if normalize else "pixels")
    ax.set_title(f"{run}  (mIoU={report.get('mIoU', 0):.3f})")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="single run name")
    g.add_argument("--runs", nargs="+", help="multiple run names")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", help="output path (for --run)")
    ap.add_argument("--out-dir", default="outputs/figs", help="output dir (for --runs)")
    ap.add_argument("--absolute", action="store_true",
                    help="plot raw pixel counts instead of row-normalized fractions")
    args = ap.parse_args()

    if args.run:
        out = args.out or f"outputs/figs/cm_{args.run}.png"
        plot_one(args.run, args.outputs, out, normalize=not args.absolute)
    else:
        for run in args.runs:
            out = os.path.join(args.out_dir, f"cm_{run}.png")
            plot_one(run, args.outputs, out, normalize=not args.absolute)


if __name__ == "__main__":
    main()
