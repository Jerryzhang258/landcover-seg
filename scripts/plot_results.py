"""Bar chart of final test mIoU + per-class IoU heatmap from eval reports.

Reads outputs/{run_name}/report.json written by src/eval_fullres.py.

Usage:
    python scripts/plot_results.py \
        --runs unet_scratch_ce unet_r34_ce deeplab_r50_ce deeplab_r101_ce attn_unet_ce \
        --out-prefix outputs/figs/archs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import CLASS_NAMES, IGNORE_INDEX  # noqa: E402


def load_reports(outputs: str, runs: list) -> Dict[str, dict]:
    reports = {}
    for run in runs:
        p = os.path.join(outputs, run, "report.json")
        if not os.path.exists(p):
            print(f"[warn] missing {p}, skipping")
            continue
        with open(p) as f:
            reports[run] = json.load(f)
    return reports


def plot_metric_bars(reports: Dict[str, dict], out: str) -> None:
    metrics = ["mIoU", "mDice", "pixel_acc"]
    runs = list(reports.keys())
    x = np.arange(len(runs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(runs)), 5))
    for i, m in enumerate(metrics):
        vals = [reports[r][m] for r in runs]
        ax.bar(x + (i - 1) * width, vals, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=25, ha="right")
    ax.set_ylabel("score (0-1)")
    ax.set_title("Test-set metrics by run")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out}")


def plot_per_class_heatmap(reports: Dict[str, dict], out: str) -> None:
    runs = list(reports.keys())
    classes = [c for i, c in enumerate(CLASS_NAMES) if i != IGNORE_INDEX]
    mat = np.zeros((len(runs), len(classes)))
    for i, r in enumerate(runs):
        per = reports[r]["per_class_IoU"]
        for j, c in enumerate(classes):
            mat[i, j] = per.get(c, 0.0)

    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 0.55 * len(runs) + 2))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs)
    for i in range(len(runs)):
        for j in range(len(classes)):
            color = "white" if mat[i, j] < 0.5 else "black"
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="IoU")
    ax.set_title("Per-class IoU")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out-prefix", default="outputs/figs/results")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    reports = load_reports(args.outputs, args.runs)
    if not reports:
        raise SystemExit("no reports found")

    plot_metric_bars(reports, f"{args.out_prefix}_metrics.png")
    plot_per_class_heatmap(reports, f"{args.out_prefix}_perclass.png")


if __name__ == "__main__":
    main()
