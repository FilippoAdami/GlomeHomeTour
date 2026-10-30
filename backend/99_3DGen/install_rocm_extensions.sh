#!/usr/bin/env bash
set -euo pipefail

STAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(dirname "$STAGE")"
PYTHON="${PYTHON:-$BACKEND/.venv/bin/python}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi

if [[ -z "${PYTORCH_ROCM_ARCH:-}" ]]; then
    PYTORCH_ROCM_ARCH="$("$PYTHON" -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])')"
fi
export PYTORCH_ROCM_ARCH

"$PYTHON" -c 'import torch; assert torch.version.hip, "backend/.venv must use ROCm PyTorch"'

git clone --depth 1 --recursive https://github.com/JeffreyXiang/CuMesh.git "$WORK/CuMesh"
git -C "$WORK/CuMesh" apply "$STAGE/rocm-patches/cumesh-rocm.patch"
"$PYTHON" -m pip install --no-build-isolation --no-deps "$WORK/CuMesh"

git clone --depth 1 --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$WORK/FlexGEMM"
"$PYTHON" -m pip install --no-build-isolation --no-deps "$WORK/FlexGEMM"

git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git "$WORK/TRELLIS.2"
"$PYTHON" -m pip install --no-build-isolation --no-deps "$WORK/TRELLIS.2/o-voxel"
"$PYTHON" -m pip install --no-build-isolation --no-deps git+https://github.com/Cardboard-box-a/nvdiffrast-hip.git

# These source packages declare demo and CUDA extras that are not used here.
"$PYTHON" -m pip install --no-deps git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a
"$PYTHON" -m pip install --no-deps git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37
