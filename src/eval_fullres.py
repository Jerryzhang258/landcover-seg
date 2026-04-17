"""Full-resolution sliding-window evaluation on held-out test images.

Usage:
    python -m src.eval_fullres \
        --ckpt checkpoints/unet_r34_ce_best.pt \
        --raw-dir data/raw/train \
        --split-file data/tiles/splits.json \
        --split test \
        --out outputs/unet_r34_ce
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from .dataset import FullImageDataset
from .metrics import ConfusionMatrix
from .models import build_model
from .tiling import sliding_window_positions, stitch_with_overlap
from .utils import (CLASS_NAMES, IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD,
                    NUM_CLASSES, ensure_dir, label_to_rgb, rgb_to_label)


@torch.no_grad()
def predict_full(
    model: torch.nn.Module,
    img: np.ndarray,
    tile: int = 816,
    overlap: int = 64,
    device: str = "cuda",
    batch_size: int = 2,
) -> np.ndarray:
    """Returns HxW uint8 label map stitched from softmax averages."""
    H, W = img.shape[:2]
    positions = sliding_window_positions(H, W, tile, overlap)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    logits_tiles: List[np.ndarray] = []
    buf_x, buf_pos = [], []
    for (y, x) in positions:
        crop = img[y : y + tile, x : x + tile]
        t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        buf_x.append(t)
        buf_pos.append((y, x))
        if len(buf_x) == batch_size:
            xb = torch.cat(buf_x, 0).to(device)
            xb = (xb - mean) / std
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()
            for p in probs:
                logits_tiles.append(p)
            buf_x, buf_pos = [], []
    if buf_x:
        xb = torch.cat(buf_x, 0).to(device)
        xb = (xb - mean) / std
        probs = torch.softmax(model(xb), dim=1).cpu().numpy()
        for p in probs:
            logits_tiles.append(p)

    return stitch_with_overlap(logits_tiles, positions, (H, W), NUM_CLASSES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raw-dir", default="data/raw/train")
    ap.add_argument("--split-file", default="data/tiles/splits.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=816)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--save-viz", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("cfg", {})
    model_name = cfg.get("model") or os.path.basename(args.ckpt).split("_")[0]
    model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(args.split_file) as f:
        splits = json.load(f)
    ids = splits[args.split]
    ds = FullImageDataset(args.raw_dir, ids)

    cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
    per_image = []

    for i in tqdm(range(len(ds)), desc=f"eval {args.split}"):
        iid, img, mask_rgb = ds[i]
        gt = rgb_to_label(mask_rgb)
        pred = predict_full(model, img, tile=args.tile, overlap=args.overlap, device=device)

        img_cm = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
        img_cm.update(pred, gt)
        rep = img_cm.report()
        per_image.append({"id": iid, **{k: v for k, v in rep.items() if isinstance(v, (int, float))}})
        cm.update(pred, gt)

        if args.save_viz:
            pred_rgb = label_to_rgb(pred)
            cv2.imwrite(
                os.path.join(args.out, f"{iid}_pred.png"),
                cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR),
            )

    report = cm.report()
    out_report = {
        "ckpt": args.ckpt,
        "model": model_name,
        "split": args.split,
        "num_images": len(ds),
        **report,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(out_report, f, indent=2)
    with open(os.path.join(args.out, "per_image.json"), "w") as f:
        json.dump(per_image, f, indent=2)

    print(f"[done] {args.split}  mIoU={report['mIoU']:.4f}  mDice={report['mDice']:.4f}  "
          f"pix_acc={report['pixel_acc']:.4f}")
    for c in CLASS_NAMES:
        print(f"  IoU[{c:<11}] = {report['per_class_IoU'][c]:.4f}")


if __name__ == "__main__":
    main()
