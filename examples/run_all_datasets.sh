#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/run_sweep.py \
  --datasets all \
  --seeds 8,18,28,38,48 \
  --data_root ./dataset \
  --result_dir ./results \
  --skip_existing
