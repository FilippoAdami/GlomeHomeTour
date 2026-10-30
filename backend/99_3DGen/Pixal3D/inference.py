import os
import argparse
import math
import time

_pixal_dir = os.path.dirname(os.path.abspath(__file__))
if not os.environ.get("PIXAL3D_DISABLE_FUSED_SDPA"):
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # enable expandable segments to reduce fragmentation
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("REMBG_MODEL", "ZhengPeng7/BiRefNet")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_pixal_dir, ".triton_cache"))
os.environ.setdefault("TRITON_HOME", os.path.join(_pixal_dir, ".triton"))
os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.join(_pixal_dir, ".miopen_db"))
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", os.path.join(_pixal_dir, ".miopen_cache"))
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", os.path.join(_pixal_dir, "autotune_cache.json"))

import torch
import numpy as np
import cv2
from PIL import Image

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"


def _timing_start():
    if os.environ.get("PIXAL_TIMING") != "1":
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _timing_end(label, started):
    if started is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[PIXAL_TIMING] {label}={time.perf_counter() - started:.3f}s", flush=True)


IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict, backbone_model=None):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config, backbone_model=backbone_model)
    model.eval()
    return model


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device="cuda", low_vram=False):
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    backbone = pipeline.image_cond_model_ss.model
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"], backbone)
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"], backbone)
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"], backbone)

    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        # Peak VRAM = one flow model + one DinoV3, not all ~18 GB at once.
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        pipeline.low_vram = False
        print("[NAF] Pre-loading NAF upsampler model...")
        print("[Pipeline] Standard mode configured.")

    naf_owner = pipeline.image_cond_model_shape_512
    naf_owner._load_naf()
    pipeline.image_cond_model_shape_1024.naf_model = naf_owner.naf_model
    pipeline.image_cond_model_tex_1024.naf_model = naf_owner.naf_model

    return pipeline

# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_input, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    if isinstance(image_input, (str, os.PathLike)):
        pil_image = Image.open(image_input).convert("RGB")
    else:
        pil_image = image_input.convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 6,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 6,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 6,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    model_path: str = MODEL_PATH,
    manual_fov: float = -1.0,
    low_vram: bool = False,
    resolution: int = -1,
    decimation_target: int = 100000,
    texture_size: int = 1024,
):
    run_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load models
    started = _timing_start()
    pipeline = init_pipeline(model_path, low_vram=low_vram)
    _timing_end("pixal_model_init", started)

    # Preprocess image first — rembg loads to GPU for this call, then offloads.
    # MoGe is loaded afterwards so both never occupy VRAM at the same time.
    print(f"[Inference] Processing image: {image_path}")
    started = _timing_start()
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)
    _timing_end("image_preprocess", started)

    # Camera estimation
    if manual_fov > 0:
        # Use manually specified FOV (in radians)
        camera_angle_x = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}
        print(f"[Inference] Using manual FOV: {math.degrees(manual_fov):.2f}° ({manual_fov:.4f} rad), distance={distance:.4f}")
    else:
        print("[MoGe-2] Loading model for camera estimation...")
        started = _timing_start()
        moge_model = load_moge_model(device="cuda")
        _timing_end("moge_model_load", started)
        print("[Inference] Estimating camera parameters...")
        started = _timing_start()
        camera_params = get_camera_params_wild_moge(
            image_preprocessed, moge_model, device="cuda",
            mesh_scale=mesh_scale, extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
        _timing_end("moge_camera_estimate", started)
        print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")
        # MoGe is only needed for camera estimation; free its VRAM for inference.
        moge_model.cpu()
        del moge_model
        torch.cuda.empty_cache()

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    if not low_vram:
        print("[Pipeline] Moving models to GPU for generation...")
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution if resolution > 0 else (1024 if low_vram else 1536)}_cascade"
    print(f"[Inference] Using pipeline_type={pipeline_type}")
    torch.cuda.synchronize()
    pipeline_started = time.perf_counter()
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )
    torch.cuda.synchronize()
    pipeline_seconds = time.perf_counter() - pipeline_started

    mesh = mesh_list[0]
    attrs = mesh.attrs
    finite_attrs = torch.isfinite(attrs)
    if not finite_attrs.all():
        defaults = torch.zeros_like(attrs)
        defaults[..., pipeline.pbr_attr_layout['roughness']] = 1
        defaults[..., pipeline.pbr_attr_layout['alpha']] = 1
        print(f"[Inference] Sanitizing {(~finite_attrs).sum().item()} non-finite PBR values")
        attrs = torch.where(finite_attrs, attrs, defaults)

    # Extract GLB
    print(f"[Inference] Extracting GLB (decimation={decimation_target}, texture_size={texture_size})...")
    glb_started = time.perf_counter()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target, texture_size=texture_size,
        remesh=True, remesh_band=1, remesh_project=0.0, use_tqdm=True,
    )
    # Apply rotation
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    torch.cuda.synchronize()
    glb_seconds = time.perf_counter() - glb_started
    total_seconds = time.perf_counter() - run_started
    peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"[Metrics] total={total_seconds:.2f}s pipeline={pipeline_seconds:.2f}s "
          f"glb={glb_seconds:.2f}s peak_vram={peak_gib:.2f}GiB")
    print(f"[Done] GLB saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians (e.g. 0.2). "
                             "If not set, FOV is auto-estimated via MoGe-2. "
                             "Try 0.2 rad if you notice distortion.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models stay on CPU and are loaded to GPU on-demand per stage. "
                             "Reduces peak VRAM at the cost of slower inference.")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Pipeline resolution (1024 or 1536). Default: 1024 if --low_vram, else 1536.")
    parser.add_argument("--steps", type=int, default=8,
                        help="Sampling steps for SS, Shape, and Texture stages (default: 6; easily tuned, e.g. 5 for fastest, 8-12 for max quality)")
    parser.add_argument("--max_num_tokens", type=int, default=49152,
                        help="Maximum tokens for sparse attention (default: 49152)")
    parser.add_argument("--decimation_target", type=int, default=100000,
                        help="Target face count for GLB mesh decimation (default: 100000; easily tuned, e.g. 50000-300000)")
    parser.add_argument("--texture_size", type=int, default=2048,
                        help="Resolution of baked GLB texture atlas (default: 1024; easily tuned, e.g. 1024, 2048, 4096)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast preset: 5 sampling steps and 1024 texture size for maximum speed")

    args = parser.parse_args()

    if args.fast:
        if args.steps == 6:
            args.steps = 5
        if args.texture_size > 1024:
            args.texture_size = 1024

    steps = args.steps

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        ss_sampling_steps=steps,
        shape_slat_sampling_steps=steps,
        tex_slat_sampling_steps=steps,
        max_num_tokens=args.max_num_tokens,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        resolution=args.resolution,
    )
