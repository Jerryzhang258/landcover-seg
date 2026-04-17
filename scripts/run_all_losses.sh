#!/usr/bin/env bash
# Phase 5: loss comparison on the winning architecture (default: deeplab_r50).
# Edit the configs/ YAMLs if your Phase-4 winner differs.
set -euo pipefail
cd "$(dirname "$0")/.."

for cfg in deeplab_r50_dice deeplab_r50_focal deeplab_r50_combined; do
  echo "=== training $cfg ==="
  python -m src.train --config configs/${cfg}.yaml
done
