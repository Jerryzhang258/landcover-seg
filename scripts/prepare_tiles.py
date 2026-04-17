"""Split DeepGlobe train set by image-id, tile, and write tiles to disk.

Input layout (DeepGlobe train directory):
    data/raw/train/{id}_sat.jpg
    data/raw/train/{id}_mask.png    # RGB color-coded

Output layout:
    data/tiles/{split}/images/{id}_{k}.png    # 3-channel RGB
    data/tiles/{split}/masks/{id}_{k}.png     # 1-channel uint8 label
    data/tiles/splits.json                     # image-id lists per split
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tiling import tile_image  # noqa: E402
from src.utils import ensure_dir, rgb_to_label  # noqa: E402


def discover_ids(raw_dir: str) -> list:
    files = os.listdir(raw_dir)
    ids = sorted({f.split("_")[0] for f in files if f.endswith("_sat.jpg")})
    missing = [i for i in ids if not os.path.exists(os.path.join(raw_dir, f"{i}_mask.png"))]
    if missing:
        raise RuntimeError(f"{len(missing)} images have no matching mask, e.g. {missing[:3]}")
    return ids


def process_id(iid: str, raw_dir: str, out_root: str, split: str,
               grid: int, tile_size: int) -> None:
    img_path = os.path.join(raw_dir, f"{iid}_sat.jpg")
    mask_path = os.path.join(raw_dir, f"{iid}_mask.png")
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if img_bgr is None or mask_bgr is None:
        raise RuntimeError(f"failed to read {iid}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    need = grid * tile_size
    if img.shape[:2] != (need, need):
        # DeepGlobe is 2448x2448; if you change grid/tile_size mismatch, crop/pad here.
        raise RuntimeError(f"{iid}: unexpected size {img.shape[:2]}, expected {need}x{need}")

    label = rgb_to_label(mask_rgb)
    img_tiles = tile_image(img, grid=grid, tile_size=tile_size)
    mask_tiles = tile_image(label, grid=grid, tile_size=tile_size)

    img_out = os.path.join(out_root, split, "images")
    mask_out = os.path.join(out_root, split, "masks")
    for k, (ti, tm) in enumerate(zip(img_tiles, mask_tiles)):
        cv2.imwrite(os.path.join(img_out, f"{iid}_{k}.png"),
                    cv2.cvtColor(ti, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(mask_out, f"{iid}_{k}.png"), tm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/train")
    ap.add_argument("--out-root", default="data/tiles")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--tile-size", type=int, default=816)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ids = discover_ids(args.raw_dir)
    print(f"[info] discovered {len(ids)} image ids")

    trainval_ids, test_ids = train_test_split(
        ids, test_size=args.test_ratio, random_state=args.seed
    )
    val_frac = args.val_ratio / (1.0 - args.test_ratio)
    train_ids, val_ids = train_test_split(
        trainval_ids, test_size=val_frac, random_state=args.seed
    )
    splits = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}
    print(f"[info] split sizes  train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")

    ensure_dir(args.out_root)
    for split in splits:
        ensure_dir(os.path.join(args.out_root, split, "images"))
        ensure_dir(os.path.join(args.out_root, split, "masks"))

    with open(os.path.join(args.out_root, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    for split, ids_ in splits.items():
        for iid in tqdm(ids_, desc=f"tiling {split}"):
            process_id(iid, args.raw_dir, args.out_root, split,
                       args.grid, args.tile_size)
    print("[done] tiling complete")


if __name__ == "__main__":
    main()
