#!/bin/bash
# ==============================================================================
# Full-Scale Execution Script for CLIP-FGDI (Protocol 2 & Occluded-Duke)
# ==============================================================================
# This script sequentially runs all the benchmarks requested for the final table.
# WARNING: Each of these runs can take 12-24 hours on a single GPU.
# We highly recommend running this inside a tmux or screen session.
# ==============================================================================

set -e

DATA_PATH="data" # Make sure this points to your real datasets folder
DEVICE="cuda"

echo "Starting Full Scale Benchmark Suite for CLIP-FGDI"
echo "Data Path: $DATA_PATH"
echo "Device: $DEVICE"
echo "====================================================="

# 1. Protocol 2 -> Market
echo "[1/5] Running Protocol 2 with Target: Market-1501"
python3 run.py --benchmark protocol2 --held_out_domain Market --data_path $DATA_PATH --device $DEVICE

# 2. Protocol 2 -> MSMT17
echo "[2/5] Running Protocol 2 with Target: MSMT17"
python3 run.py --benchmark protocol2 --held_out_domain MSMT17 --data_path $DATA_PATH --device $DEVICE

# 3. Protocol 2 -> CUHK-SYSU
echo "[3/5] Running Protocol 2 with Target: CUHK-SYSU"
python3 run.py --benchmark protocol2 --held_out_domain cuhk_sysu --data_path $DATA_PATH --device $DEVICE

# 4. Protocol 2 -> CUHK03
echo "[4/5] Running Protocol 2 with Target: CUHK03"
python3 run.py --benchmark protocol2 --held_out_domain cuhk03 --data_path $DATA_PATH --device $DEVICE

# 5. Occluded-Duke Benchmark
echo "[5/5] Running Occluded-Duke Benchmark"
python3 run.py --benchmark occluded_duke --data_path $DATA_PATH --device $DEVICE

echo "====================================================="
echo "All benchmarks completed!"
