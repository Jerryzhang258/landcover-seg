"""Gradio live demo for DeepGlobe land-cover segmentation.

Loads a trained checkpoint and exposes a browser UI that runs the same
sliding-window inference used in eval_fullres.py. Designed for presentations:
drop an image in, get a coloured class map + overlay + per-class proportions
within a few seconds on an A100, ~30 s on a laptop CPU.

Usage (local GPU):
    python demo/app.py --ckpt checkpoints/deeplab_r50_combined_best.pt

Usage (Colab / remote — exposes a public share URL via Gradio tunnel):
    python demo/app.py --ckpt $CKPT --share

Any --model that matches src.models.build_model works; the default matches
the project's headline configuration (DeepLabV3+-R50 + Combined CE+Dice).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Let `python demo/app.py` resolve the `src` package regardless of cwd.
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

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import guard
    sys.exit(
        "Gradio not found. Install with:\n"
        "    pip install gradio==4.29.0\n"
        f"({exc})"
    )


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

# Inverted colour lookup: class index -> (R, G, B)
INDEX_TO_RGB = {idx: rgb for rgb, idx in CLASS_COLORS.items()}

LEGEND_HTML_PARTS = []
for idx, name in enumerate(CLASS_NAMES):
    if idx == IGNORE_INDEX:
        continue
    r, g, b = INDEX_TO_RGB[idx]
    swatch = (
        f"<span style='display:inline-block;width:14px;height:14px;"
        f"background:rgb({r},{g},{b});border:1px solid #444;"
        f"margin-right:6px;vertical-align:middle;'></span>"
    )
    LEGEND_HTML_PARTS.append(f"{swatch}<b>{name}</b>")
LEGEND_HTML = (
    "<div style='font-size:14px;line-height:24px;'>"
    + " &nbsp; ".join(LEGEND_HTML_PARTS)
    + "</div>"
)

TARGET_SIDE = 2448  # Images are resized to 2448x2448 before sliding-window inference.


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, model_name: str, device: str) -> torch.nn.Module:
    """Instantiate the architecture and load weights from a train.py checkpoint."""
    model = build_model(model_name, NUM_CLASSES).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"warning: {len(missing)} missing / {len(unexpected)} unexpected keys "
            f"when loading {ckpt_path}"
        )
    model.eval()
    return model


def _fit_to_target(img: np.ndarray) -> np.ndarray:
    """Centre-crop to a square, then resize to TARGET_SIDE x TARGET_SIDE."""
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top : top + side, left : left + side]
    if side != TARGET_SIDE:
        interp = cv2.INTER_AREA if side > TARGET_SIDE else cv2.INTER_CUBIC
        img = cv2.resize(img, (TARGET_SIDE, TARGET_SIDE), interpolation=interp)
    return img


def _stats_table(pred: np.ndarray) -> str:
    """Return a Markdown table of per-class pixel proportions (excluding Unknown)."""
    counts = np.bincount(pred.ravel(), minlength=NUM_CLASSES)
    total = int(counts[:IGNORE_INDEX].sum())
    if total == 0:
        return "_No foreground pixels detected._"
    rows = ["| Class | Colour | % of pixels |", "|---|:---:|---:|"]
    for idx in range(NUM_CLASSES):
        if idx == IGNORE_INDEX:
            continue
        r, g, b = INDEX_TO_RGB[idx]
        swatch = (
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:rgb({r},{g},{b});border:1px solid #444;'></span>"
        )
        pct = 100.0 * counts[idx] / total
        rows.append(f"| {CLASS_NAMES[idx]} | {swatch} | {pct:.1f}% |")
    return "\n".join(rows)


def make_predict_fn(model: torch.nn.Module, device: str):
    """Build a Gradio event handler with the model closed over."""

    def _predict(img: np.ndarray):
        if img is None:
            return None, None, "_Upload an image or pick an example to run a prediction._"
        img = _fit_to_target(img)
        t0 = time.time()
        if device == "cuda":
            torch.cuda.synchronize()
        pred = predict_full(model, img, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000.0

        pred_rgb = label_to_rgb(pred)
        overlay = (0.5 * img.astype(np.float32) + 0.5 * pred_rgb.astype(np.float32)).astype(np.uint8)
        stats = _stats_table(pred)
        stats = f"**Inference:** {elapsed_ms:.0f} ms on `{device}`\n\n" + stats
        return pred_rgb, overlay, stats

    return _predict


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INTRO_MD = """
# DeepGlobe Land-Cover Segmentation — Live Demo

Model: **DeepLabV3+ (ResNet-50) + Combined CE+Dice loss** &mdash;
test mIoU **0.7332** on full-resolution 2448&times;2448 DeepGlobe images.

Upload any RGB satellite tile (JPEG/PNG) or pick one of the examples.
Images are centre-cropped to a square and resized to 2448&times;2448 before
sliding-window inference (816&sup2; tiles, 64-pixel overlap).
"""


def build_ui(predict_fn, examples_dir: Path | None):
    with gr.Blocks(
        title="DeepGlobe Land-Cover Segmentation",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown(INTRO_MD)
        gr.HTML(LEGEND_HTML)
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(
                    type="numpy",
                    label="Satellite image (RGB)",
                    height=420,
                )
                btn = gr.Button("Predict", variant="primary")
                if examples_dir is not None and examples_dir.is_dir():
                    paths = sorted(
                        [
                            str(p)
                            for p in examples_dir.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        ]
                    )
                    if paths:
                        gr.Examples(
                            examples=[[p] for p in paths],
                            inputs=[inp],
                            label="Examples",
                        )
            with gr.Column(scale=1):
                out_mask = gr.Image(label="Predicted class map", height=420)
                out_overlay = gr.Image(label="Overlay (50 / 50)", height=420)
                out_stats = gr.Markdown()
        btn.click(predict_fn, inputs=[inp], outputs=[out_mask, out_overlay, out_stats])
        inp.change(predict_fn, inputs=[inp], outputs=[out_mask, out_overlay, out_stats])
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, type=Path, help="Path to a train.py checkpoint (.pt)")
    ap.add_argument(
        "--model",
        default="deeplab_r50",
        help="Architecture name understood by src.models.build_model "
        "(default: deeplab_r50, matches the headline config)",
    )
    ap.add_argument("--examples-dir", type=Path, default=ROOT / "demo" / "examples")
    ap.add_argument("--share", action="store_true", help="Expose a public Gradio URL")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if not args.ckpt.is_file():
        sys.exit(f"error: checkpoint not found: {args.ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo] loading {args.model} from {args.ckpt} on {device} ...")
    model = load_model(args.ckpt, args.model, device)
    print("[demo] model ready")

    predict_fn = make_predict_fn(model, device)
    ui = build_ui(predict_fn, args.examples_dir)
    ui.launch(share=args.share, server_port=args.port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
