#!/bin/bash
# ==============================================================================
# Ablation Experiments Script for CLIP-FGDI (Doc §9)
# ==============================================================================
# This script sequentially runs the 4 configurations for the ablation table.
# We use the Protocol-2 (Market as target) benchmark for these ablations.
# WARNING: Each of these runs can take 12-24 hours on a single GPU.
# We highly recommend running this inside a tmux or screen session.
# ==============================================================================

set -e

DATA_PATH="data" # Make sure this points to your real datasets folder
DEVICE="cuda"
BENCHMARK="protocol2"
TARGET="Market"

echo "Starting Ablation Suite for CLIP-FGDI DG-ReID"
echo "Data Path: $DATA_PATH"
echo "Device: $DEVICE"
echo "====================================================="

# 1. Full Model (DG-ReID)
echo "[1/4] Running Full Model (Both GRL and PartVisibilityGAT enabled)"
python3 run.py --benchmark $BENCHMARK --held_out_domain $TARGET --data_path $DATA_PATH --device $DEVICE

# 2. GRL Only
echo "[2/4] Running Ablation: GRL Only (PartVisibilityGAT disabled)"
python3 run.py --benchmark $BENCHMARK --held_out_domain $TARGET --data_path $DATA_PATH --device $DEVICE --disable_part_branch

# 3. Part Branch Only
echo "[3/4] Running Ablation: Part Branch Only (GRL disabled)"
python3 run.py --benchmark $BENCHMARK --held_out_domain $TARGET --data_path $DATA_PATH --device $DEVICE --disable_grl

# 4. Neither (Baseline)
echo "[4/4] Running Ablation: Baseline (Both GRL and Part Branch disabled)"
python3 run.py --benchmark $BENCHMARK --held_out_domain $TARGET --data_path $DATA_PATH --device $DEVICE --disable_grl --disable_part_branch

echo "====================================================="
echo "All ablation experiments completed!"
