# 07 · Single-view image to 3D

This stage keeps TRELLIS.2 and Pixal3D for single-image generation. It uses the
shared `backend/.venv`; do not install the CUDA-pinned vendor requirements.

Install the Python dependencies and ROCm extensions once:

```bash
backend/.venv/bin/python -m pip install -e 'backend/.[3dgen]'
bash backend/07_3DGen/install_rocm_extensions.sh
```

Call the batch API from the repository root:

```python
from image_to_3d import generate_3d

ply_paths = generate_3d(["front.jpg", "side.jpg"], steps=12, model="trellis")
```

Set `PYTHONPATH=backend/07_3DGen` for imports. `model` is `trellis` or `pixal`
(case-insensitive); `steps` is applied to each sampler. The model is loaded once
for the call and images run sequentially. PLYs go to
`backend/current_scene/07_3DGen/<run-number>/`; an existing numeric folder is
never reused. The function returns the created PLY paths. PLY output contains
mesh geometry only; the PBR texture is not included.

The atomic steps also expose batch functions:

```python
from background_removal.remove_backgrounds import remove_backgrounds
from semantic_inference.caption_images import generate_captions

rgba_images = remove_backgrounds(["front.jpg", "side.jpg"])
rgba_images[0].save("front.png", format="PNG")
captions = generate_captions(["front.jpg", "side.jpg"])  # detail="minimum"
```

Background removal returns RGBA PIL images; captioning returns one string per
input. Each call loads its model once and unloads it when finished.

## ROCm patch review

- `rocm-patches/cumesh-rocm.patch` fixes the CuMesh HIP build and its silent
  `hipMemcpy2D` row truncation. The install script applies it before building
  CuMesh. No other Glome stage uses CuMesh, so keep it local to this stage.
- `rocm-patches/pixal3d-sdpa-windowed-attn.patch` is already applied in the
  copied Pixal source; it fixes the `sdpa` windowed-attention path for both HIP
  and CUDA. Do not apply it again.
- The old nvdiffrast 0.3.3 patch was removed in favor of the working HIP fork.
  o-voxel imports nvdiffrast at module load, so the install script includes the
  fork even though PLY generation does not call its renderer.
- `rocm-patches/natten/__init__.py` is the small pure-PyTorch NATTEN shim used
  by the NAF image feature extractor. It is added to `sys.path` by the batch
  generator, so a separate CUDA NATTEN package is not needed.

The original benchmark reports remain under `artifacts/`; their PyTorch 2.10
figures are historical and should be compared with runs from the shared
PyTorch 2.13 / ROCm 7.1 environment only as approximate baselines.
