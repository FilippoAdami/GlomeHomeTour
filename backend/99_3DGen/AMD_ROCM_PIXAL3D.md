# Running Pixal3D on an AMD GPU (ROCm)

Pixal3D ships assuming NVIDIA CUDA. This document records what it took to get it
generating correct textured 3D models on a **Radeon RX 9060 XT (gfx1200 / RDNA4,
16 GB)** under ROCm 7.1.1, and how to use it now that it works.

> This is the historical standalone validation on PyTorch 2.10 / ROCm 7.0.
> Integrated setup and callable batch APIs now live in [README.md](README.md)
> and use Glome's shared PyTorch 2.13 / ROCm 7.1 environment.

Verified working end to end: a single 4096×4096 image in, a 40.9 MB textured GLB
out (710,377 verts / 998,786 faces / 4096² texture) in ~30 minutes at **4.7 GB
peak VRAM**.

---

## 1. Environment

| Component | Version |
|---|---|
| GPU | AMD Radeon RX 9060 XT, gfx1200 (RDNA4), 16 GB |
| ROCm | 7.1.1 (`/opt/rocm-7.1.1`) |
| PyTorch | 2.10.0+rocm7.0 (`torch.version.hip = 7.0.51831`) |
| triton | triton-rocm 3.6.0 |
| Python | 3.12 in `~/Desktop/TripoSplat/.venv` |

ROCm PyTorch presents itself as a `cuda` device through HIP, so nearly all
framework-level code runs unmodified. The work was all in the native extensions.

### Native extensions

| Extension | Status |
|---|---|
| `flex_gemm` (sparse conv backend, default) | built clean, no patch |
| `o_voxel` (voxel→mesh conversion) | built clean, no patch |
| `cumesh` (mesh ops) | built, **5 patches** |
| `nvdiffrast` | `Cardboard-box-a/nvdiffrast-hip`, installed clean, no patch |

`natten` and `spconv` are *not* needed as CUDA builds — see below.

### nvdiffrast

Don't port it yourself. Upstream v0.4.0 dropped the OpenGL backend, leaving
`cudaraster` with ~50 inline PTX intrinsics. The OpenGL route is a dead end on
AMD anyway: ROCm exposes `hip/hip_gl_interop.h`, but Mesa/radeonsi lacks working
`hipGraphicsGLRegisterBuffer`, and `glutil.cpp` gates device selection on
`EGL_CUDA_DEVICE_NV`, which never matches on AMD.

**`Cardboard-box-a/nvdiffrast-hip`** already reimplements those PTX intrinsics
with `__lane_id()` / `__ballot` and works on gfx1200 (which is wave32, matching
nvdiffrast's warp-size-32 assumptions). It installed with zero patches.

---

## 2. What was actually broken

Six fixes. Only **one** is a genuine AMD porting issue — the rest are
platform-independent bugs, VRAM limits, or licensing.

### 2.1 `hipMemcpy2D` silently truncates at 2²⁰ rows — the real bug

This is the one that mattered, and it is nasty because **nothing crashes**.

`hipMemcpy2D` copies only the first 1,048,576 rows and still returns
`hipSuccess`. CuMesh's `src/io.cu` performs every vertex/face copy as a 2D copy
with `height = element count`, so any mesh past ~1M elements silently lost its
tail; the untouched remainder stayed zeroed.

A 4,017,833-vertex decode kept exactly 1,048,576 vertices and got 2,969,257
zeroed ones. Those zeros then poisoned `fill_holes` into emitting 4.1 million
degenerate faces, which starved the decimator and produced a **2,761-vertex
collapsed blob** instead of a turtle.

Reproduction:

```
N=  1000000  rows preserved=  1000000  OK
N=  1048576  rows preserved=  1048576  OK
N=  1048577  rows preserved=  1048576  TRUNCATED
N=  4017833  rows preserved=  1048576  TRUNCATED
```

Fix: chunk the height into ≤2²⁰ blocks in one helper that all three call sites
route through (`cudaMemcpy2D_chunked` in `CuMesh/src/io.cu`).

> **Debugging lesson.** Every upstream stage tested clean — decoder, `o_voxel`,
> its GPU hashmap, and the raw extracted mesh (4,017,833 verts / 6,133,918 faces,
> *zero* degenerate faces, full symmetric bounds). The damage happened in a copy
> that reported success. A 2,562-vertex sphere round-tripped perfectly, which is
> exactly why an early test wrongly cleared CuMesh. **Always size a ROCm repro
> above 2²⁰ elements.**

### 2.2 `natten` is CUDA-only

`natten` is pulled in indirectly by the NAF feature upsampler via
`torch.hub.load("valeoai/NAF")`, which calls `na2d(..., backend="cutlass-fna")`.
CUTLASS is NVIDIA-only.

Replaced with a pure-PyTorch `na2d` shim implementing neighborhood attention by
gathering each query's window (honouring NATTEN's border rule, where the window
slides inward per dilation group instead of zero-padding). It is row-chunked to
bound memory and carries a `__main__` self-check asserting it against a naive
per-pixel reference, including the chunked path and a `v` wider than `q`/`k`.

The shim is now stored at `rocm-patches/natten/__init__.py` and added to the
stage import path by the batch API. Run it directly to verify:
`backend/.venv/bin/python backend/07_3DGen/rocm-patches/natten/__init__.py` →
`na2d shim OK`.

> Slower than a fused CUTLASS kernel, but this runs a handful of times per
> generation, not per diffusion step.

### 2.3 Sparse attention OOM (16 GB card)

`sdpa` had no flash/mem-efficient kernel for these shapes on ROCm, so it fell
back to the math backend and materialised the full Lq×Lkv score matrix — an 8.4
GB single allocation. Added `_chunked_sdpa` in
`pixal3d/modules/sparse/attention/full_attn.py`, which blocks over queries to cap
the score matrix. Verified identical to plain `sdpa`.

### 2.4 NAF upsampler fp32 (16 GB card)

NAF ran in fp32 and asked for a 4 GB allocation. Wrapped its `forward` in
`torch.autocast(fp16)` at load time — once, so every call site inherits it. It's
a feature upsampler; fp16 is plenty.

### 2.5 Non-finite vertices poison `fill_holes`

The sparse decoder sporadically emits a handful of non-finite vertices (**6 out
of 4 million**). `fill_holes` turns those into millions of zeroed verts and
degenerate faces. Now the faces touching them are dropped first, and `fill_holes`
patches the resulting holes normally. **This is not AMD-specific** and would
affect CUDA users too.

### 2.6 Gated background-removal model

The shipped pipeline config points `rembg` at `briaai/RMBG-2.0`, which is gated
(403 unless allow-listed). The `BiRefNet` class default is the open MIT-licensed
`ZhengPeng7/BiRefNet`; only the config overrides it. Added a `REMBG_MODEL`
environment override.

> Not needed at all if your input image already has an alpha channel — the
> pipeline skips background removal entirely in that case.

### 2.7 Also fixed: an upstream bug affecting NVIDIA too

`calc_window_partition` in `pixal3d/modules/sparse/attention/windowed_attn.py`
had branches for `xformers`, `flash_attn` and `flash_attn_4`, but none for
`sdpa` — leaving `attn_func_args` unbound. **`ATTN_BACKEND=sdpa` raised
`UnboundLocalError` on any GPU, NVIDIA included**, so the advertised
"no flash_attn" escape hatch was broken for windowed sparse attention.

Added the missing binding plus a real sdpa implementation (padded batch + single
masked call, since a per-window Python loop would be thousands of iterations).
Verified exact: `windows=64 maxlen=79 max_abs_err=0.00e+00`.

### Not blockers

- **`natten` as a CUDA build** — only reached through the NAF hub model; the shim covers it.
- **`spconv`** — one of four sparse-conv backends and not the default (`flex_gemm` is).
- **21 GB of weights vs 16 GB VRAM** — that figure is *disk* cache, and much of it
  belonged to unrelated projects. Measured peak VRAM is **4.7 GB**; `--low_vram`
  pages modules through the GPU one stage at a time.

### Patches

Integrated under `backend/07_3DGen/rocm-patches/`:

| File | Contents |
|---|---|
| `cumesh-rocm.patch` | `hipMemcpy2D` chunking, `rocprim::tuple` for libcu++, host-callable `Vec3f()`, dropped nvcc-only `--extended-lambda` |
| `pixal3d-sdpa-windowed-attn.patch` | the sdpa windowed-attention fix |

The old nvdiffrast patch is not retained: the HIP fork replaced it, and the
single-view PLY path does not use nvdiffrast.

Worth upstreaming: the CuMesh `hipMemcpy2D` fix, the sdpa windowed-attention bug,
and the NaN-vertex guard (the latter two affect CUDA users).

---

## 3. Usage

The commands below document the original standalone setup and are retained for
benchmark reproduction. New project calls use `backend/.venv` as shown in the
integrated [stage README](README.md).

### Required environment

```bash
export PYTORCH_ROCM_ARCH=gfx1200                      # your arch; `rocminfo | grep gfx`
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export REMBG_MODEL=ZhengPeng7/BiRefNet
```

| Variable | Why |
|---|---|
| `PYTORCH_ROCM_ARCH` | target your GPU when extensions rebuild |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` | **important** — `expandable_segments:True` is reported to cause *silent NaNs* on RDNA4. Both inference scripts hardcoded `True`; now overridable. |
| `ATTN_BACKEND` / `SPARSE_ATTN_BACKEND` | `flash_attn` isn't available; `sdpa` is the portable path |
| `REMBG_MODEL` | avoids the gated Bria model |

The inference entry points automatically set
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. PyTorch 2.10 otherwise excludes
gfx1200 from its fused SDPA allowlist and falls back to the quadratic math
kernel. Unsupported shapes retain the chunked math implementation as a safe
fallback. Geometry sampling uses fused SDPA; texture sampling conservatively
stays on reference math SDPA while gfx1200 AOTriton remains experimental. Set
`PIXAL3D_DISABLE_FUSED_SDPA=1` to reproduce the old path.

### Measured result

On the environment above, `assets/images/11_img.png`, seed 42, fixed FOV
0.857556, 1024 cascade, 12 sampling steps, and `--low_vram`:

| Metric | Chunked math baseline | Safe optimized path | Change |
|---|---:|---:|---:|
| End-to-end | 1053.76 s | 525.97 s | **2.00x faster** |
| Neural pipeline | 847.09 s | 366.93 s | **2.31x faster** |
| Peak allocated VRAM | 13.02 GiB | 13.01 GiB | no increase |

An exact `[1, 16, 12642, 64]` BF16 attention benchmark measured 1.085 s
for chunked math and 0.028 s for AOTriton (38.8x), with max absolute error
`4.88e-4` and mean absolute error `1.87e-5`. The production path also locally
inpaints non-finite PBR decoder voxels rather than allowing NaNs into texture
baking. The final stress-case GLB had finite geometry and texture data, no
zero-area faces, and a 4096x4096 texture. On `assets/images/0_img.png`, bounding
box extents remained within 0.75% of the baseline; a 100k-point bidirectional
surface comparison had mean distance below 0.38% and 95th-percentile distance
below 0.91% of the object diagonal in either direction.

### 3.1 Single image → 3D

One in-the-wild photo. The camera is unknown, so MoGe-2 estimates the FOV.

```bash
cd ~/Desktop/TripoSplat/Pixal3D
PYTORCH_ROCM_ARCH=gfx1200 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False" \
ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
REMBG_MODEL=ZhengPeng7/BiRefNet \
../.venv/bin/python inference.py \
    --image assets/images/0_img.png \
    --output ./output.glb \
    --low_vram
```

**CLI flags**

| Flag | Default | Notes |
|---|---|---|
| `--image` | *required* | input image; RGBA alpha is used as the mask and skips background removal |
| `--output` | `./output.glb` | output path |
| `--seed` | `42` | change for a different sample from the same image |
| `--fov` | `-1.0` (auto) | manual horizontal FOV in radians. Try `0.2` if the result looks distorted — this overrides the MoGe-2 estimate |
| `--resolution` | `-1` | `1024` or `1536`. Defaults to `1024` with `--low_vram`, else `1536` |
| `--low_vram` | off | keeps models on CPU, pages them to GPU per stage. **Recommended on 16 GB** |
| `--model_path` | `TencentARC/Pixal3D` | HF repo or local path |

### 3.2 Multi-view images → 3D

> **Verification status:** confirmed end to end on the bundled four-view set at
> 1024 with `--low_vram`. The run completed in 1454.17 s (24m14s): 837.91 s in
> the neural pipeline and 560.54 s in GLB processing, with 12.05 GiB peak
> PyTorch allocation. The exported GLB contained 754,271 vertices, 997,305
> faces, a finite 4096x4096 texture, and no degenerate faces.

Use `inference_mv.py` when you have **posed** views — cameras are given rather
than guessed, so geometry is better constrained. Its pipeline subclasses the
single-view one, so every fix above applies unchanged.

On 16 GB GPUs, the 1024 NAF stage streams each view independently and keeps its
exact running feature average on CPU between views. This avoids overlapping a
2 GiB accumulated projection with NAF's 4 GiB workspace. Keep `--low_vram` for
four-view inference; 1536 multi-view inference is not validated on 16 GB.

```bash
cd ~/Desktop/TripoSplat/Pixal3D
PYTORCH_ROCM_ARCH=gfx1200 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False" \
ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
REMBG_MODEL=ZhengPeng7/BiRefNet \
../.venv/bin/python inference_mv.py \
    --views_dir assets/mv_images/example \
    --output ./output_mv.glb \
    --low_vram
```

**Input layout** (see `assets/mv_images/example` for a working set of 4 views):

```
<views_dir>/
    transforms.json          mesh_scale + per-frame file_path / transform_matrix
    view00_azim000.png       any format; RGBA alpha is used as the mask if
    view01_azim090.png       present, otherwise the view is matted with the
    view02_azim180.png       pipeline's own rembg model
    view03_azim270.png
```

`transforms.json` follows the Blender/NeRF convention used by the training
renders: `transform_matrix` is a 4×4 camera-to-world (Z-up world, camera looks
−Z with +Y up), and `camera_angle_x` is the horizontal FOV in radians, given
per frame or once at the top level.

> **Frame 0 must be the canonical front view** (camera at `(0, -d, 0)`). It is
> the MAIN view, and its pose must make the main view degenerate to the
> single-view path. Getting this wrong is the most likely cause of a bad
> multi-view result.

**CLI flags** — same as single-view, except `--image`/`--fov` are replaced by:

| Flag | Default | Notes |
|---|---|---|
| `--views_dir` | example dir | directory described above |
| `--num_views` | all | use only the first N views from `transforms.json` |
| `--config_file` | pipeline default | pipeline config inside `model_path` |

### 3.3 Tuning quality

The sampling parameters are **function arguments of `run_inference()`, not CLI
flags**. To change them, either edit the defaults in `inference.py` or call
`run_inference(...)` from your own script.

Defaults (`inference.py:161`):

| Parameter | Default | Effect |
|---|---|---|
| `ss_sampling_steps` | `12` | sparse-structure steps — the coarse occupancy volume |
| `ss_guidance_strength` | `7.5` | how hard stage 1 follows the image |
| `ss_guidance_rescale` | `0.7` | tames over-saturated guidance |
| `ss_rescale_t` | `5.0` | timestep rescale for stage 1 |
| `shape_slat_sampling_steps` | `12` | shape latent steps — silhouette and surface detail |
| `shape_slat_guidance_strength` | `7.5` | image adherence for shape |
| `shape_slat_guidance_rescale` | `0.5` | |
| `shape_slat_rescale_t` | `3.0` | |
| `tex_slat_sampling_steps` | `12` | texture latent steps |
| `tex_slat_guidance_strength` | `1.0` | low by design; raising it tends to bake in lighting |
| `tex_slat_rescale_t` | `3.0` | |
| `max_num_tokens` | `49152` | token budget. The pipeline **steps HR resolution down by 128 until it fits**, so lowering this silently reduces detail |
| `mesh_scale` | `1.0` | object scale in the unit cube |
| `image_resolution` | `512` | conditioning image resolution |
| `extend_pixel` | `0` | padding around the matted subject |

Mesh output is controlled at the `o_voxel.postprocess.to_glb` call
(`inference.py:263`):

| Parameter | Default | Effect |
|---|---|---|
| `decimation_target` | `1000000` | target face count — the biggest lever on file size |
| `texture_size` | `4096` | texture resolution |
| `remesh` | `True` | clean remesh pass |

**Practical guidance**

- **Speed.** Runtime is dominated by the HR stage (~65 s/step of 12). Halving
  `shape_slat_sampling_steps` roughly halves it, at some loss of detail.
- **Lower VRAM.** Keep `--low_vram` and `--resolution 1024`.
- **More detail, >16 GB card.** Drop `--low_vram` and use `--resolution 1536`.
- **Distorted output.** Set `--fov` manually (try `0.2`); a wrong FOV estimate
  skews the whole reconstruction.
- **Variations.** Change `--seed`; everything else is deterministic.
- **Smaller GLB.** Lower `decimation_target` and/or `texture_size`.

### 3.4 Optional: faster attention

The run logs:

```
Flash/Mem Efficient attention on Current AMD GPU is still experimental.
Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

That flag is why `sdpa` fell back to the math backend, and enabling it would
likely cut runtime meaningfully. **It is untested here and left off on purpose**
— it is also the flag most often implicated in RDNA4 silent-NaN reports. If you
try it, check the output mesh for NaNs before trusting it.

---

## 4. Debug instrumentation

Env-gated and inert by default:

| Variable | Effect |
|---|---|
| `PIXAL3D_DEBUG_COORDS=1` | prints voxel counts / extents at each stage and decoded-mesh bounds |
| `PIXAL3D_DUMP_LATENT=<path>` | dumps shape/texture latents so the decode stage can be re-run in ~1 min instead of ~30 |

The second is what made the CuMesh bug findable at a reasonable pace. Example
output from a healthy run:

```
[debug] sparse structure: 3206 voxels, min=[1, 4, 2] max=[29, 29, 31]
[debug] hr_coords_unique: 13720 tokens @ res 1024, min=[2, 8, 4] max=[58, 58, 63]
[debug] decoded mesh[0]: MeshWithVoxel (4300889, 3) verts,
        min=[-0.469, -0.37, -0.427] max=[0.415, 0.414, 0.487]
```

Healthy signs: coordinate extents spanning most of the grid, and decoded-mesh
bounds roughly symmetric about the origin. A collapsed axis or a bound pinned at
exactly `0.0` means a copy got truncated.
