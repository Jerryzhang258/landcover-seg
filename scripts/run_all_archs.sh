#!/usr/bin/env bash
# Phase 3 & 4: baselines and architecture comparison.
# All use CE loss so only the architecture differs.
set -euo pipefail
cd "$(dirname "$0")/.."

for cfg in vanilla_unet unet_scratch unet_resnet34 deeplab_r50 deeplab_r101 attn_unet; do
  echo "=== training $cfg ==="
  python -m src.train --config configs/${cfg}.yaml
done
