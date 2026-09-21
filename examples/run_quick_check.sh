#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/run_trace.py \
  --data GSE150774_RAW/BM \
  --benchmark false \
  --data_root ./dataset \
  --result_dir ./results_quick \
  --epochs 3 \
  --seed 8
