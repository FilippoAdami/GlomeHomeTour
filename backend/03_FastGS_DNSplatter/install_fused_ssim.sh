#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${1:-"$root/../.venv/bin/python"}
PYTORCH_ROCM_ARCH=gfx1200 MAX_JOBS=2 "$python" -m pip install --no-build-isolation --no-deps "$root/native_ssim"
