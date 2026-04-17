"""Efficiency scatter plots: mIoU vs params, mIoU vs train time, mIoU vs inference latency.

Answers the proposal question: "how much more compute is needed for each
incremental mIoU gain?"

Reads outputs/results.csv (from scripts/aggregate_results.py).

Usage:
    python scripts/aggregate_results.py
    python scripts/plot_efficiency.py --out-prefix outputs/figs/efficiency
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def read_results(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(x: str):
    try:
        return float(x) if x not in (None, "", "None") else None
    except ValueError:
        return None


def plot_scatter(rows, x_field, y_field, x_label, y_label, title, out_path):
    xs, ys, labels = [], [], []
    for r in rows:
        x = _to_float(r.get(x_field))
        y = _to_float(r.get(y_field))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        labels.append(r["run_name"])
    if not xs:
        print(f"[warn] no data for {x_field} vs {y_field}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=70, alpha=0.75, edgecolors="black")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results.csv")
    ap.add_argument("--out-prefix", default="outputs/figs/efficiency")
    ap.add_argument("--y", default="test_mIoU",
                    help="y-axis column name (fallback: best_val_mIoU)")
    args = ap.parse_args()

    rows = read_results(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")

    # Prefer test_mIoU; if eval hasn't been run, fall back to best_val_mIoU.
    if not any(_to_float(r.get(args.y)) for r in rows):
        print(f"[info] column '{args.y}' empty — falling back to best_val_mIoU")
        args.y = "best_val_mIoU"

    y_label = "mIoU (test)" if args.y == "test_mIoU" else "mIoU (val, best)"
    plot_scatter(rows, "params_M", args.y,
                 "Parameters (M)", y_label,
                 f"Efficiency: {y_label} vs params",
                 f"{args.out_prefix}_params.png")
    plot_scatter(rows, "train_time_hr", args.y,
                 "Train time (h)", y_label,
                 f"Efficiency: {y_label} vs train time",
                 f"{args.out_prefix}_traintime.png")
    plot_scatter(rows, "infer_ms_per_image", args.y,
                 "Inference latency (ms / full 2448² image)", y_label,
                 f"Efficiency: {y_label} vs inference latency",
                 f"{args.out_prefix}_latency.png")


if __name__ == "__main__":
    main()
