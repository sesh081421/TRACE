#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN:-python}" scripts/run_sweep.py \
  --datasets all \
  --seeds 8,18,28,38,48 \
  --data_root ./dataset \
  --result_dir ./results \
  --skip_existing \
  "$@"
