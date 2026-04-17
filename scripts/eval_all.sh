#!/usr/bin/env bash
# Phase 6: full-resolution sliding-window evaluation on the test split.
set -euo pipefail
cd "$(dirname "$0")/.."

RAW_DIR="${RAW_DIR:-data/raw/train}"
SPLIT_FILE="${SPLIT_FILE:-data/tiles/splits.json}"

shopt -s nullglob
for ckpt in checkpoints/*_best.pt; do
  name=$(basename "$ckpt" _best.pt)
  out="outputs/${name}"
  echo "=== eval $name ==="
  python -m src.eval_fullres \
    --ckpt "$ckpt" \
    --raw-dir "$RAW_DIR" \
    --split-file "$SPLIT_FILE" \
    --split test \
    --out "$out" \
    --save-viz
done

python scripts/aggregate_results.py --outputs outputs --csv outputs/results.csv
