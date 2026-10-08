#!/usr/bin/env bash
set -euo pipefail

stage_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${1:-$stage_dir/../.venv/bin/python}"
checkout="$(mktemp -d)"
trap 'rm -rf -- "$checkout"' EXIT

git clone --quiet --depth 1 --branch 1.5.3b2 --recurse-submodules --shallow-submodules \
  https://github.com/AMD-Ecosystem/gsplat.git "$checkout"
expected_commit=b01acd43e3c7fa942f95fda0974e9125e4de7395
if [[ "$(git -C "$checkout" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "AMD gsplat tag 1.5.3b2 no longer matches the verified source" >&2
  exit 1
fi
git -C "$checkout" apply "$stage_dir/gsplat_gfx1200.patch"

export PYTORCH_ROCM_ARCH=gfx1200
export MAX_JOBS="${MAX_JOBS:-4}"
export CPLUS_INCLUDE_PATH="$checkout/gsplat/cuda/csrc/third_party/glm${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
"$python_bin" -m pip install --no-build-isolation --no-deps "$checkout"
"$stage_dir/install_fused_ssim.sh" "$python_bin"
