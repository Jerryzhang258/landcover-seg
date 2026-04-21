"""Render a demo MP4: for each input image, show (Input | Prediction | Overlay)
with per-class pixel proportions, held for a few seconds, then cut to the next.

Produces a deterministic, self-contained video file you can:
  - embed as a backup in Beamer / PowerPoint (no live demo risk),
  - drop into the GitHub README as an animated preview,
  - post on YouTube as a project artifact.

Usage:
    python demo/record_video.py \\
        --ckpt checkpoints/deeplab_r50_combined_best.pt \\
        --images data/raw/train/104876_sat.jpg \\
                 data/raw/train/153214_sat.jpg \\
                 data/raw/train/629571_sat.jpg \\
        --out demo/demo_video.mp4

The video is 1920x1080 @ 24 fps by default; each image is held for 5 s.
Inference reuses src.eval_fullres.predict_full (sliding-window, 816 tile,
64-pixel overlap, softmax averaging) — identical to the pipeline that
produced the paper's 0.7332 test mIoU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_fullres import predict_full  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    CLASS_COLORS,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    label_to_rgb,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_TO_RGB = {idx: rgb for rgb, idx in CLASS_COLORS.items()}
FG_INDICES = [i for i in range(NUM_CLASSES) if i != IGNORE_INDEX]
TARGET_SIDE = 2448

DEFAULT_W, DEFAULT_H = 1920, 1080
DEFAULT_DPI = 120


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def fit_to_target(img: np.ndarray) -> np.ndarray:
    """Centre-crop to square, then resize to TARGET_SIDE x TARGET_SIDE."""
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top : top + side, left : left + side]
    if side != TARGET_SIDE:
        interp = cv2.INTER_AREA if side > TARGET_SIDE else cv2.INTER_CUBIC
        img = cv2.resize(img, (TARGET_SIDE, TARGET_SIDE), interpolation=interp)
    return img


def load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path))
    if arr is None:
        raise RuntimeError(f"cannot read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def load_model(ckpt_path: Path, model_name: str, device: str) -> torch.nn.Module:
    model = build_model(model_name, NUM_CLASSES).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[warn] {len(missing)} missing / {len(unexpected)} unexpected keys "
            f"when loading {ckpt_path}"
        )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(
    img: np.ndarray,
    pred_rgb: np.ndarray,
    overlay: np.ndarray,
    counts: np.ndarray,
    img_id: str,
    infer_ms: float,
    device: str,
    width: int,
    height: int,
    dpi: int,
) -> np.ndarray:
    """Render one composite BGR frame (HxWx3, uint8) via matplotlib."""
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    gs = GridSpec(
        nrows=2,
        ncols=3,
        height_ratios=[3.0, 1.3],
        left=0.025,
        right=0.975,
        top=0.88,
        bottom=0.07,
        hspace=0.30,
        wspace=0.04,
    )

    fig.suptitle(
        "DeepGlobe Land-Cover Segmentation  ·  DeepLabV3+ (ResNet-50) + Combined CE+Dice",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        f"Image {img_id}    |    inference: {infer_ms:.0f} ms on {device}    |    "
        f"test mIoU 0.7332 on 2448\u00d72448 reconstructions",
        ha="center",
        fontsize=11.5,
        color="#555555",
    )

    # Top row: three image panels.
    for col, (title, pic) in enumerate(
        zip(("Input", "Prediction", "Overlay (50 / 50)"), (img, pred_rgb, overlay))
    ):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(pic)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
        ax.axis("off")

    # Bottom row: horizontal bar chart spanning all three columns.
    ax_bar = fig.add_subplot(gs[1, :])
    total_fg = int(counts[FG_INDICES].sum())
    if total_fg == 0:
        ax_bar.text(
            0.5,
            0.5,
            "no foreground pixels predicted",
            ha="center",
            va="center",
            transform=ax_bar.transAxes,
            fontsize=12,
            color="#888888",
        )
        ax_bar.axis("off")
    else:
        percentages = [100.0 * counts[i] / total_fg for i in FG_INDICES]
        # Draw bars in descending % so the biggest class is on top.
        order = np.argsort(percentages)[::-1]
        names = [CLASS_NAMES[FG_INDICES[i]] for i in order]
        pcts = [percentages[i] for i in order]
        rgb_norm = [tuple(c / 255.0 for c in INDEX_TO_RGB[FG_INDICES[i]]) for i in order]
        ypos = np.arange(len(names))
        bars = ax_bar.barh(ypos, pcts, color=rgb_norm, edgecolor="#333333", linewidth=0.6)
        for bar, pct in zip(bars, pcts):
            ax_bar.text(
                bar.get_width() + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=10,
            )
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(names, fontsize=11)
        ax_bar.invert_yaxis()  # biggest on top
        ax_bar.set_xlim(0, max(max(pcts) * 1.18, 10))
        ax_bar.set_xlabel("% of foreground pixels", fontsize=10)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.tick_params(axis="x", labelsize=9)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    # Insurance: force exact target size.
    if bgr.shape[:2] != (height, width):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return bgr


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_size(s: str) -> tuple[int, int]:
    w, _, h = s.lower().partition("x")
    return int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--model", default="deeplab_r50")
    ap.add_argument("--images", nargs="+", required=True, type=Path,
                    help="Input satellite image paths (jpg/png)")
    ap.add_argument("--out", default=Path("demo/demo_video.mp4"), type=Path)
    ap.add_argument("--seconds-per-image", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size", default=f"{DEFAULT_W}x{DEFAULT_H}",
                    help="Video resolution, e.g. 1920x1080 (default) or 1280x720")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--poster", type=Path, default=None,
                    help="If set, also save the first rendered frame as a PNG")
    args = ap.parse_args()

    if not args.ckpt.is_file():
        sys.exit(f"error: checkpoint not found: {args.ckpt}")
    missing = [p for p in args.images if not p.is_file()]
    if missing:
        sys.exit(f"error: image(s) not found: {missing}")

    width, height = parse_size(args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] loading {args.model} from {args.ckpt} on {device}")
    model = load_model(args.ckpt, args.model, device)

    print(f"[2/3] rendering {len(args.images)} frame(s)")
    rendered: List[np.ndarray] = []
    for path in tqdm(args.images, desc="inference"):
        img = fit_to_target(load_rgb(path))
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        pred = predict_full(model, img, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.time() - t0) * 1000.0

        pred_rgb = label_to_rgb(pred)
        overlay = (0.5 * img.astype(np.float32) + 0.5 * pred_rgb.astype(np.float32)).astype(np.uint8)
        counts = np.bincount(pred.ravel(), minlength=NUM_CLASSES)

        frame = render_frame(
            img=img,
            pred_rgb=pred_rgb,
            overlay=overlay,
            counts=counts,
            img_id=path.stem.replace("_sat", ""),
            infer_ms=infer_ms,
            device=device,
            width=width,
            height=height,
            dpi=args.dpi,
        )
        rendered.append(frame)

    if args.poster is not None:
        args.poster.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.poster), rendered[0])
        print(f"      poster frame saved: {args.poster}")

    print(f"[3/3] writing {args.out} @ {width}x{height}, {args.fps} fps, "
          f"{args.seconds_per_image:g}s/image")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        sys.exit(
            "error: cv2.VideoWriter refused to open. On macOS this usually means "
            "the OpenCV wheel was built without the FFmpeg backend. Install "
            "opencv-python with ffmpeg support or change --out to a directory "
            "of PNGs and stitch manually."
        )
    hold_frames = max(1, int(round(args.seconds_per_image * args.fps)))
    total = 0
    for frame in rendered:
        for _ in range(hold_frames):
            writer.write(frame)
            total += 1
    writer.release()

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"done. {args.out}  ({total} frames, {total / args.fps:.1f} s, {size_mb:.1f} MB)")
    print("\nTip: convert to GIF for Beamer / README with:")
    print(f"  ffmpeg -i {args.out} -vf 'fps=12,scale=960:-1' {args.out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
