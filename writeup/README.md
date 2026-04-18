# Writeup (report + slides)

Two LaTeX source files with placeholder figures that expect `results.zip`
unzipped into `figs/`.

```
writeup/
├── report.tex        # 5-6 page technical report
├── slides.tex        # Beamer slides, ~17 slides, ~15 min talk
├── figs/             # drop results.zip contents here (see below)
└── README.md         # this file
```

## 1. Put the figures in place

You have `results.zip` from Colab (Cell 15 downloaded it). Unzip its
**contents of `outputs/figs/`** into `writeup/figs/`:

```bash
cd writeup
unzip -j ~/Downloads/results.zip 'outputs/figs/*' -d figs/
ls figs/
```

After unzipping you should see:

```
class_dist.png            class_legend.png            samples.png
curves_archs.png          archs_metrics.png           archs_perclass.png
curves_losses.png         losses_metrics.png          losses_perclass.png
qualitative.png
cm_attn_unet_ce.png       cm_deeplab_r50_ce.png       cm_deeplab_r50_combined.png
failures_combined.png
```

The `.tex` files reference these filenames directly.

## 2. Compile

### Option A: Overleaf (easiest)

1. Go to https://overleaf.com, click **New Project → Upload Project**.
2. Upload the entire `writeup/` folder as a zip (or drag & drop).
3. Open `report.tex`, click **Recompile**. Same for `slides.tex`.
4. Download the PDFs.

**Note.** `slides.tex` uses the `metropolis` Beamer theme. Overleaf has it
pre-installed. If you compile locally and don't have it, swap line 4 of
`slides.tex`:

```latex
\usetheme{metropolis}   % comment this out
\usetheme{default}      % or \usetheme{Madrid}
```

### Option B: Local LaTeX (TeX Live / MacTeX)

```bash
# One command each
cd writeup
pdflatex report.tex && pdflatex report.tex   # run twice for references
pdflatex slides.tex && pdflatex slides.tex
```

PDFs land in the same directory.

### Option C: Pandoc (if you don't have LaTeX)

Less reliable for the academic style, but works:

```bash
pandoc report.tex -o report.pdf --pdf-engine=xelatex
```

## 3. What's in each file

### `report.tex` — 6-page technical report

- **Abstract** — headline numbers (0.7332 test mIoU, +7.1 transfer
  learning gain, +10.5 Barren IoU gain)
- **Introduction** — problem, 3 challenges, contributions
- **Related Work** — U-Net, DeepLabV3+, Attention, transfer learning,
  loss functions, tiling (13 references)
- **Methodology** — tiling pipeline, padding, 9 architectures/losses,
  evaluation protocol
- **Experiments** — setup, sanity checks, data distribution
- **Results & Discussion** — 3 big findings with tables + figures +
  confusion matrices + failure cases
- **Conclusions & Limitations**
- **References** — inline `thebibliography`, no external .bib needed

### `slides.tex` — Beamer, 17 content slides for a 15-min talk

- Title + motivation (3 slides)
- Dataset and tiling pipeline (2 slides)
- Experimental grid and setup (2 slides)
- Results: training dynamics, architecture table, 3 findings (5 slides)
- Confusion matrices, qualitative, failures (3 slides)
- Conclusions + thanks (2 slides)
- Backup slides (3 slides, accessed via `\appendix`)

Speaker budget: aim ~45 seconds per slide, leaves 3-4 min for Q&A.

## 4. Edit checklist before submission

- [ ] Verify `figs/` has all 13+ PNG files listed above
- [ ] Compile `report.tex` and confirm PDF is $\ge 5$ pages
- [ ] Compile `slides.tex` and verbally rehearse once ($\le 15$ min)
- [ ] Swap the `\author` / institution lines if you need different text
- [ ] Fill in any metric you want with greater precision using the
      actual `outputs/results.csv` from the run
- [ ] Re-read Abstract and Conclusion for typos

## 5. Submission

- Final report: upload the `report.pdf` you compiled
- Presentation slides: upload the `slides.pdf` you compiled
- Both must be PDF
