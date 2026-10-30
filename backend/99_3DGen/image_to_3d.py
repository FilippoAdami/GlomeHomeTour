"""Batch single-view image-to-mesh generation for GlomeHomeTour."""

import gc
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("REMBG_MODEL", "ZhengPeng7/BiRefNet")

from PIL import Image


ROOT = Path(__file__).resolve().parent
PIXAL_ROOT = ROOT / "Pixal3D"
ROCM_PATCHES = ROOT / "rocm-patches"
DEFAULT_OUTPUT = ROOT.parent / "current_scene" / "07_3DGen"
TRELLIS_MODEL = "microsoft/TRELLIS.2-4B"
PIXAL_MODEL = "TencentARC/Pixal3D"


def _reserve_output_dir(root=DEFAULT_OUTPUT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run = max(
        (int(path.name) for path in root.iterdir() if path.name.isdecimal()),
        default=0,
    ) + 1
    while True:
        output = root / str(run)
        try:
            output.mkdir()
            return output
        except FileExistsError:
            run += 1


def _load_image(image):
    if isinstance(image, Image.Image):
        filename = getattr(image, "filename", None)
        return image.copy(), Path(filename).stem if filename else "image"
    path = Path(image)
    with Image.open(path) as source:
        return source.copy(), path.stem


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def _write_ply(mesh, path, rotate_pixal=False):
    import numpy as np
    import trimesh

    vertices = np.asarray(_numpy(mesh.vertices), dtype=np.float32)
    faces = np.asarray(_numpy(mesh.faces), dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Generated mesh has no valid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Generated mesh has no triangular faces")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Generated mesh contains out-of-range face indices")
    finite = np.isfinite(vertices).all(axis=1)
    faces = faces[finite[faces].all(axis=1)]
    used = np.unique(faces)
    vertices = vertices[used]
    remap = np.full(finite.size, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    faces = remap[faces]
    if not len(faces):
        raise ValueError("Generated mesh has no finite triangular faces")

    result = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if rotate_pixal:
        result.apply_transform([
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
    result.export(path, file_type="ply")
    return path


def _load_trellis(torch):
    if str(ROCM_PATCHES) not in sys.path:
        sys.path.insert(0, str(ROCM_PATCHES))
    if str(PIXAL_ROOT) not in sys.path:
        sys.path.insert(0, str(PIXAL_ROOT))
    from pixal3d.pipelines import Trellis2ImageTo3DPipeline

    class CascadePipeline(Trellis2ImageTo3DPipeline):
        load_on_meta = True
        model_names_to_load = [
            name for name in Trellis2ImageTo3DPipeline.model_names_to_load
            if name != "tex_slat_flow_model_512"
        ]

    pipeline = CascadePipeline.from_pretrained(TRELLIS_MODEL)
    pipeline._device = torch.device("cuda")
    pipeline.low_vram = True
    return pipeline


def _load_pixal():
    if str(ROCM_PATCHES) not in sys.path:
        sys.path.insert(0, str(ROCM_PATCHES))
    if str(PIXAL_ROOT) not in sys.path:
        sys.path.insert(0, str(PIXAL_ROOT))
    spec = importlib.util.spec_from_file_location(
        "glome_07_3dgen_pixal_inference", PIXAL_ROOT / "inference.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Pixal3D inference from {PIXAL_ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_3d(images, steps=12, model="trellis"):
    """Generate geometry-only PLY meshes; load one pipeline for the whole image batch."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(images, (str, os.PathLike, Image.Image)):
        raise TypeError("images must be a list of image paths or PIL images")
    images = list(images)
    if not images:
        raise ValueError("images must contain at least one image")
    if not isinstance(model, str) or model.casefold() not in {"trellis", "pixal"}:
        raise ValueError("model must be 'trellis' or 'pixal'")
    model = model.casefold()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Image-to-3D generation requires a ROCm or CUDA GPU")

    started = time.perf_counter()
    pipeline = moge_model = meshes = mesh = None
    try:
        load_started = time.perf_counter()
        if model == "trellis":
            pipeline = _load_trellis(torch)
        else:
            pixal = _load_pixal()
            pipeline = pixal.init_pipeline(PIXAL_MODEL, low_vram=True)
            moge_model = pixal.load_moge_model(device="cpu")
        torch.cuda.synchronize()
        print(f"[3DGen] model_load={time.perf_counter() - load_started:.2f}s", flush=True)

        output_dir = _reserve_output_dir()
        outputs = []
        for index, image_input in enumerate(images, 1):
            image, stem = _load_image(image_input)
            output = output_dir / f"{index:03d}_{stem}.ply"
            image_started = time.perf_counter()
            with torch.inference_mode():
                if model == "trellis":
                    meshes = pipeline.run(
                        image, seed=42, pipeline_type="1024_cascade",
                        max_num_tokens=49152, fill_holes=False,
                        sparse_structure_sampler_params={"steps": steps},
                        shape_slat_sampler_params={"steps": steps},
                        tex_slat_sampler_params={"steps": steps},
                    )
                    mesh = meshes[0]
                    _write_ply(mesh, output)
                else:
                    image = pipeline.preprocess_image(image)
                    moge_model.to("cuda")
                    try:
                        camera_params = pixal.get_camera_params_wild_moge(
                            image, moge_model, device="cuda", mesh_scale=1.0,
                            extend_pixel=0, image_resolution=512,
                        )
                    finally:
                        moge_model.cpu()
                        torch.cuda.empty_cache()
                    meshes = pipeline.run(
                        image, camera_params=camera_params, seed=42,
                        sparse_structure_sampler_params={"steps": steps},
                        shape_slat_sampler_params={"steps": steps},
                        tex_slat_sampler_params={"steps": steps},
                        preprocess_image=False, pipeline_type="1024_cascade",
                        max_num_tokens=49152,
                    )
                    mesh = meshes[0]
                    _write_ply(mesh, output, rotate_pixal=True)
            torch.cuda.synchronize()
            outputs.append(output)
            print(f"[3DGen] {image_input}: {time.perf_counter() - image_started:.2f}s -> {output}", flush=True)
            meshes = mesh = image = None
            gc.collect()
            torch.cuda.empty_cache()
        return outputs
    finally:
        pipeline = moge_model = meshes = mesh = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"[3DGen] total={time.perf_counter() - started:.2f}s", flush=True)
