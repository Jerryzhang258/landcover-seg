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
import time
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .dataset import FullImageDataset
from .metrics import ConfusionMatrix
from .models import build_model
from .tiling import sliding_window_positions, stitch_with_overlap
from .utils import (CLASS_NAMES, IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD,
                    NUM_CLASSES, ensure_dir, label_to_rgb, rgb_to_label)


def _pad_to_multiple_of_32(x: int) -> int:
    return ((x + 31) // 32) * 32


@torch.no_grad()
def predict_full(
    model: torch.nn.Module,
    img: np.ndarray,
    tile: int = 816,
    overlap: int = 64,
    device: str = "cuda",
    batch_size: int = 2,
) -> np.ndarray:
    """Returns HxW uint8 label map stitched from softmax averages.

    Each tile is reflect-padded to the next multiple of 32 before the model
    forward (smp encoders require it), then logits are center-cropped back
    to the original tile size before stitching.
    """
    H, W = img.shape[:2]
    positions = sliding_window_positions(H, W, tile, overlap)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    pad_to = _pad_to_multiple_of_32(tile)
    pad = (pad_to - tile) // 2  # equal pad on each side

    def _forward(buf_x: List[torch.Tensor]) -> np.ndarray:
        xb = torch.cat(buf_x, 0).to(device)
        if pad > 0:
            xb = torch.nn.functional.pad(xb, (pad, pad, pad, pad), mode="reflect")
        xb = (xb - mean) / std
        probs = torch.softmax(model(xb), dim=1)
        if pad > 0:
            probs = probs[:, :, pad : pad + tile, pad : pad + tile]
        return probs.cpu().numpy()

    logits_tiles: List[np.ndarray] = []
    buf_x: List[torch.Tensor] = []
    for (y, x) in positions:
        crop = img[y : y + tile, x : x + tile]
        t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        buf_x.append(t)
        if len(buf_x) == batch_size:
            for p in _forward(buf_x):
                logits_tiles.append(p)
            buf_x = []
    if buf_x:
        for p in _forward(buf_x):
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
    total_infer_s = 0.0

    for i in tqdm(range(len(ds)), desc=f"eval {args.split}"):
        iid, img, mask_rgb = ds[i]
        gt = rgb_to_label(mask_rgb)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        pred = predict_full(model, img, tile=args.tile, overlap=args.overlap, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        total_infer_s += time.time() - t0

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
    infer_ms_per_image = (total_infer_s / max(len(ds), 1)) * 1000.0
    out_report = {
        "ckpt": args.ckpt,
        "model": model_name,
        "split": args.split,
        "num_images": len(ds),
        "inference_ms_per_image": infer_ms_per_image,
        "confusion_matrix": cm.mat.tolist(),
        **report,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(out_report, f, indent=2)
    with open(os.path.join(args.out, "per_image.json"), "w") as f:
        json.dump(per_image, f, indent=2)

    print(f"[done] {args.split}  mIoU={report['mIoU']:.4f}  mDice={report['mDice']:.4f}  "
          f"pix_acc={report['pixel_acc']:.4f}  infer={infer_ms_per_image:.0f}ms/img")
    for c in CLASS_NAMES:
        print(f"  IoU[{c:<11}] = {report['per_class_IoU'][c]:.4f}")


if __name__ == "__main__":
    main()
