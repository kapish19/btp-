#!/bin/bash
# ==============================================================================
# Fixed Base CLIP-FGDI Paper Baseline Runner
# ==============================================================================
# This script runs the CLIP-FGDI baseline (with all 5 paper fixes applied:
# BCEWithLogitsLoss, num_workers=4, parse_known_args, dataset loader fixes, and Protocol 2 support).
# ==============================================================================

set -e

DATA_PATH=${1:-"data"}
TARGET=${2:-"Market"}
DEVICE=${3:-"cuda"}

echo "====================================================="
echo "Running CLIP-FGDI Baseline (Fixed)"
echo "Target Domain (Held-Out): $TARGET"
echo "Data Path: $DATA_PATH"
echo "Device: $DEVICE"
echo "====================================================="

python run.py \
  --benchmark       protocol2 \
  --held_out_domain $TARGET \
  --data_path       $DATA_PATH \
  --log_path        logs/clip_fgdi_baseline/protocol_2_$TARGET \
  --batch_size      64 \
  --num_workers     4 \
  --device          $DEVICE \
  --disable_grl \
  --disable_part_branch

echo "====================================================="
echo "CLIP-FGDI Baseline Run Completed!"
