# Live Demo — DeepGlobe Land-Cover Segmentation

A browser-based Gradio UI around the same sliding-window inference pipeline
used in `src/eval_fullres.py`. Drop in a satellite image, get back a colour
class map, a 50/50 overlay, and per-class pixel proportions.

Designed for two scenarios:

1. **Live presentation** — open the tab on your laptop, upload an image,
   audience sees the prediction appear within a couple of seconds.
2. **Sharing** — run on Colab with `--share` to get a 72-hour public URL.

---

## Option 1 — Colab (recommended for presentations)

Open [`notebooks/colab_demo.ipynb`](../notebooks/colab_demo.ipynb) in Colab,
edit the `CKPT_PATH` variable to point at your trained checkpoint on Drive,
and run all cells. The last cell prints a `https://*.gradio.live` URL — open
that in the browser you'll be presenting from.

The demo runs on a Colab T4/A100 at roughly **0.8–1.5 s per 2448&sup2; image**.

## Option 2 — Local (requires a CUDA GPU)

```bash
cd landcover-seg
pip install -r requirements.txt
pip install -r demo/requirements.txt

python demo/app.py \
    --ckpt checkpoints/deeplab_r50_combined_best.pt \
    --model deeplab_r50
```

Open the printed `http://127.0.0.1:7860` URL. Without a GPU, each
2448&sup2; inference takes ~30 s on an M2 Pro CPU — usable for
debugging but too slow for an audience.

## Option 3 — HuggingFace Spaces (persistent public URL)

Upload `demo/app.py`, `demo/requirements.txt`, a copy of `src/`, and the
checkpoint under `demo/ckpt/` to a new Space (Gradio SDK, free CPU tier or
paid GPU tier). Add a tiny `app.py` shim at the root that calls
`demo.app.main()` with a fixed `--ckpt` path — details are in the
step-by-step below.

---

## Option 4 — Pre-rendered demo video (safest backup for a live talk)

`demo/record_video.py` runs the model on a fixed list of images and writes
a 1920&times;1080 MP4 showing *Input &rarr; Prediction &rarr; Overlay* with a
per-class pixel-proportion bar chart below. Deterministic output, no live
network risk — embed the MP4 in your slides as a fallback (or as the *only*
demo if you'd rather not share-tunnel a Gradio instance from Colab).

```bash
# On Colab or a local CUDA box:
python demo/record_video.py \
    --ckpt checkpoints/deeplab_r50_combined_best.pt \
    --images data/raw/train/104876_sat.jpg \
             data/raw/train/153214_sat.jpg \
             data/raw/train/629571_sat.jpg \
    --out demo/demo_video.mp4 \
    --poster demo/demo_poster.png
```

Flags worth knowing:

- `--seconds-per-image 5` — how long each image is held on screen (default 5 s).
- `--fps 24` — target framerate (default 24; lower &rarr; smaller file).
- `--size 1920x1080` — resolution. Use `1280x720` for a 2&times;-smaller file.
- `--poster path.png` — also save the first rendered frame as a still for
  the repo README / slide backgrounds.

### Embedding in Beamer / PowerPoint

- **Beamer** — `\usepackage{multimedia}`, then
  `\movie[width=\textwidth,autostart,loop]{}{demo/demo_video.mp4}`. Ship the
  MP4 alongside the PDF (Acrobat plays it inline).
- **PowerPoint / Keynote** — `Insert &rarr; Video &rarr; Movie from File`.
  Set "Start: Automatically" on the slide's transition.
- **GitHub README** — convert to GIF with the `ffmpeg` one-liner printed at
  the end of the script run:

  ```bash
  ffmpeg -i demo/demo_video.mp4 -vf 'fps=12,scale=960:-1' demo/demo_video.gif
  ```

### Or just screen-record the live Gradio demo

If you'd rather capture the actual Gradio UI (buttons, hover states, etc.):

1. Launch `demo/app.py` locally or open the Colab share URL.
2. `&#8984;+Shift+5` on macOS &rarr; "Record Selected Portion" &rarr; draw the
   browser window &rarr; Record.
3. Click through 3&ndash;5 examples, pausing for ~3 s on each.
4. Stop recording; the file lands on your Desktop as `Screen Recording &hellip;.mov`.
5. Trim the head/tail in QuickTime or Photos.

This captures the authentic interactive feel but is not reproducible &mdash;
use Option 4 above if you need a deterministic artifact.

---

## What counts as a satellite image?

The model was trained on DeepGlobe's 2448&sup2; RGB tiles (0.5 m / pixel
resolution, visible band). Any similar aerial or satellite imagery works.
The UI automatically centre-crops your input to a square and resizes it to
2448&sup2; before inference.

Good sources for live-demo images:

- **DeepGlobe test images** (if you kept them on Drive). The test split
  ids are listed in `data/tiles/splits.json`.
- **Google Maps / Google Earth screenshots** at high zoom over rural land,
  exported as PNG.
- **USGS EarthExplorer** — free, no login required for browsing;
  good Landsat / NAIP tiles.

Put 3–5 hand-picked images in `demo/examples/` before launching — Gradio
will surface them as one-click buttons under the upload box.

---

## UI tour

- **Left panel** — image upload + "Predict" button + example gallery.
  Uploading or clicking an example auto-triggers a prediction.
- **Right panel** — three outputs stacked vertically:
  1. *Predicted class map* — one RGB colour per class, matching the DeepGlobe
     palette (see the legend at the top of the page).
  2. *Overlay (50/50)* — the input image alpha-blended with the class map.
     Useful for pointing out boundary errors live.
  3. *Per-class proportions* — a small table with the % of foreground pixels
     each class occupies. Unknown pixels are excluded.

Inference time is printed above the table so the audience can see the
tiling cost without you having to interrupt yourself.

---

## Troubleshooting

- **"unknown model name: ..."** — the `--model` flag must match one of the
  names in `src/models.py::build_model` (`deeplab_r50`, `deeplab_r101`,
  `unet_resnet34`, `unet_scratch`, `vanilla_unet`, `attn_unet`).

- **"Missing keys when loading checkpoint"** — the architecture flag does
  not match the weights. Make sure `--model` matches the config you trained
  with (e.g. `deeplab_r50` for `checkpoints/deeplab_r50_combined_best.pt`).

- **"CUDA out of memory"** — lower the inference batch size by editing
  `src/eval_fullres.py::predict_full` (default `batch_size=2`) to 1.

- **The Gradio share URL refuses connections** — the tunnel sometimes takes
  ~10 s to come up after `ui.launch()`. If it still fails, a corporate
  firewall is probably blocking `*.gradio.live`; fall back to Option 2.
