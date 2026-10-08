> Historical FastGS port notes. The Inria-derived implementation was removed; current setup is in README.md.

# AMD/ROCm port progress

Updated: 2026-09-24 (Europe/Rome)

## Goal and stopping condition

Port FastGS to dual NVIDIA CUDA and AMD ROCm/HIP support, targeting an RX 9060
XT 16 GB (`gfx1200`). Runtime and performance claims require execution with a
ROCm-enabled PyTorch build and access to the GPU device nodes.

## Baseline environment

| Item | Observed value | Consequence |
|---|---|---|
| Host | Linux 7.1.5-76070105-generic, x86_64 | Supported platform family |
| Python | 3.12.3 | Does not match the repository's Python 3.7 environment |
| PyTorch | 2.13.0+cu130 | CUDA build; not usable for ROCm runtime validation |
| `torch.version.cuda` | 13.0 | CUDA extension static/build checks may be possible if `nvcc` is present |
| `torch.version.hip` | `None` | Active environment is not ROCm-enabled |
| `torch.cuda.is_available()` | `False` | No PyTorch GPU runtime is available |
| HIP compiler | ROCm 7.1.1, clang 20.0.0 | HIP source compilation tooling is installed |
| AMD GPU | device ID `0x7590`, 17,095,983,104 bytes VRAM, `gfx1200` | Target hardware is physically present |
| AMD runtime access | Device nodes hidden in the normal sandbox; approved out-of-sandbox `rocminfo` succeeds | GPU execution requires the same external permission boundary |
| NVIDIA compiler | `nvcc` not found | CUDA native-extension build regression cannot run locally |
| Build tools | CMake and Ninja present | Host/static checks available |

Relevant initial environment variables: `HSA_OVERRIDE_GFX_VERSION=12.0.0`;
`ROCM_HOME`, `HIP_PATH`, `PYTORCH_ROCM_ARCH`, and `TORCH_CUDA_ARCH_LIST` are
unset.

Repository baseline: branch `main`, commit `44e02a5`. The worktree already had
untracked `AGENTS.md`, `agents_files/`, and `current_scene/`; these are user
files and must not be removed or overwritten. Existing generated Python caches
and a `simple-knn/build` tree are present in the checkout.

## Existing project workflow

- Canonical legacy environment: Python 3.7.13, PyTorch 1.12.1, CUDA 11.6.
- Native extensions: rasterizer/Adam, simple-kNN, and fused SSIM.
- No project-wide automated test suite or CI is defined.
- End-to-end verification normally uses the dataset presets in
  `train_base.sh`/`train_big.sh`, followed by `render.py` and `metrics.py`.
- `current_scene/` is a complete COLMAP-style scene (317 images and sparse
  camera/image/point data). End-to-end execution still needs additional Python
  dependencies and permission for GPU device access.

## Portability surface

| Component | CUDA dependency / risk | Planned smallest dual-backend treatment |
|---|---|---|
| Rasterizer + Adam | CUDA runtime headers/APIs, CUB, cooperative groups, kernel launches, 32-lane tiles, atomics | Conditional CUDA/HIP includes and APIs; hipCUB on HIP; preserve 32-thread algorithmic tiles. `rocminfo` reports native wave32 on gfx1200. |
| simple-kNN | CUDA runtime, CUB reductions/sort, kernels | Conditional runtime and hipCUB compatibility with unchanged algorithm |
| fused SSIM | Cooperative groups, CUDA kernels, 32x32 workgroup | HIP-compatible compilation; retain launch geometry unless measurements justify tuning |
| Extension builds | `CUDAExtension`, NVCC-only flags | Use PyTorch's ROCm-aware `CUDAExtension`; select compatible flags from `torch.version.hip`; build `gfx1200` via `PYTORCH_ROCM_ARCH=gfx1200` |
| Python device placement | Literal `cuda`, `.cuda()`, `torch.cuda.set_device` | Use one PyTorch device value (`cuda` is also the ROCm PyTorch device type), infer allocation device from tensors where possible |
| Optimizer/densification | Native Adam update and many device allocations | Preserve behavior; make allocations inherit the model device; add focused update checks |
| VRAM | Original guidance assumes larger CUDA GPUs; source images can occupy device memory | Keep quality defaults; document `--data_device cpu` and existing resolution/densification controls |

## Backend strategy

PyTorch intentionally exposes ROCm accelerators through the `torch.cuda` API
and `cuda` device type. The port therefore keeps a single Python accelerator
path while removing allocations that hard-code a particular device index. For
native code, keep the existing `.cu` sources and `CUDAExtension`: a ROCm-enabled
PyTorch build hipifies and compiles them with HIP. Add only the conditional
headers/API aliases required where source compatibility is insufficient. Do
not fork the algorithms or add a backend abstraction layer.

## Validation log

1. Baseline inspection complete. The normal sandbox hides `/dev/kfd` and
   `/dev/dri`; approved external `rocminfo` identifies RX 9060 XT, `gfx1200`,
   wavefront size 32, 32 CUs, and 16,695,296 KiB allocatable VRAM.
2. Created an ignored `.venv-rocm` using AMD's ROCm 7.1 wheel index: PyTorch
   2.9.1+rocm7.1.0, HIP 7.1. Approved external smoke check reports
   `torch.cuda.is_available() == True`, device RX 9060 XT / `gfx1200`, and a
   device tensor reduction result of 28.0.
3. `hipcc --offload-arch=gfx1200` object compilation passed for simple-kNN and
   fused SSIM. The HIP CMake rasterizer build compiled and linked
   `libCudaRasterizer.a` for gfx1200; warnings were limited to ignored
   `[[nodiscard]]` statuses inherited from existing unchecked runtime calls.
4. With `PYTORCH_ROCM_ARCH=gfx1200`, editable PyTorch extension builds passed
   for simple-kNN, fused SSIM, and the FastGS rasterizer. All three extension
   modules import after adding the missing `simple_knn/__init__.py`.
5. Before build-artifact cleanup, the sandbox test result was `3 passed,
   5 skipped` in 1.01 s. The passes covered CPU tensor-device helpers and all
   extension imports. The GPU tests skipped because the sandbox cannot see the
   GPU. The import gate now fails (rather than skips) when extensions are not
   installed, so the documented post-install command cannot produce a false
   green result.
6. The request to run the focused GPU suite outside the sandbox was denied.
   Runtime extension correctness, numerical parity, end-to-end training, VRAM,
   profiling, and performance measurements therefore remain unverified.

## Required hardware-dependent checks still pending

- Execute all three already-built extensions on the RX 9060 XT with the
  focused suite in `tests/test_native_extensions.py`.
- Forward/backward/Adam/visibility parity against CUDA or an independent
  reference, with explicit error statistics.
- Deterministic small training and render smoke tests.
- PSNR, SSIM, LPIPS, final Gaussian count, and loss-trajectory comparison.
- Warmed-up repeated timing, peak VRAM, densification time, render FPS, and a
  ROCm profile identifying the principal bottleneck.

Exact next command (requires GPU device access outside the sandbox):

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  .venv-rocm/bin/python -m pytest -q \
  tests/test_device_portability.py tests/test_native_extensions.py
```
