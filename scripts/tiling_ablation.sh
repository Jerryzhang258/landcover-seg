#!/usr/bin/env bash
# Phase 1 ablation: 3x3 (816x816) vs 4x4 (612x612) tiling, 20 epochs each on
# U-Net + ResNet34. Answers the proposal question: does a smaller tile size
# lose enough spatial context to hurt mIoU?
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Make sure 3x3 tiles exist (default prepare_tiles already does this)
if [ ! -d data/tiles/train/images ]; then
  python scripts/prepare_tiles.py
fi

# 2. Generate 4x4 tiles to a separate directory
if [ ! -d data/tiles_4x4/train/images ]; then
  python scripts/prepare_tiles.py --grid 4 --tile-size 612 --out-root data/tiles_4x4
fi

# 3. Train both
python -m src.train --config configs/ablation_3x3.yaml
python -m src.train --config configs/ablation_4x4.yaml

# 4. Print a summary table
python - <<'PY'
import json, os
for run in ("ablation_3x3_r34", "ablation_4x4_r34"):
    p = f"outputs/{run}_summary.json"
    if os.path.exists(p):
        s = json.load(open(p))
        print(f"{run:22s}  mIoU={s['best_val_mIoU']:.4f}  "
              f"params={s['params_M']:.2f}M  time={s['train_time_hr']:.2f}h")
    else:
        print(f"{run:22s}  MISSING summary.json")
PY
