#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN:-python}" scripts/run_trace.py \
  --data GSE150774_RAW/BM \
  --benchmark false \
  --data_root ./dataset \
  --result_dir ./results_quick \
  --epochs 3 \
  --seed 8 \
  "$@"
