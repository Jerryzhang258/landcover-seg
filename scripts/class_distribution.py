"""Compute pixel-level class distribution over a tiles split (for EDA / class weights)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import CLASS_NAMES, NUM_CLASSES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-root", default="data/tiles")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="outputs/class_dist.json")
    args = ap.parse_args()

    mask_dir = os.path.join(args.tiles_root, args.split, "masks")
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in tqdm(files, desc="counting"):
        m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        counts += np.bincount(m.ravel(), minlength=NUM_CLASSES)

    total = int(counts.sum())
    frac = counts / max(total, 1)
    result = {
        "split": args.split,
        "total_pixels": total,
        "counts": {CLASS_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)},
        "fractions": {CLASS_NAMES[i]: float(frac[i]) for i in range(NUM_CLASSES)},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<12}  {counts[i]:>12d}  {frac[i]*100:6.2f}%")


if __name__ == "__main__":
    main()
