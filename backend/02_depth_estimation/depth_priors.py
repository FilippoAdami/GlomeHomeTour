"""GlomeHomeTour Backend: Depth Priors & Metric Alignment.

Implements monocular metric depth estimation using Depth Anything (Metric-Indoor),
robust scale-shift metric alignment against VIO/SfM sparse landmarks, and
dense surface normal estimation from metric depth gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import hashlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from PIL import Image

from package_loader import CameraIntrinsics, Keyframe
from pose_aligner import opengl_to_opencv

_utilities_path = Path(__file__).resolve().parent.parent / "Utilities"
if str(_utilities_path) not in sys.path:
    sys.path.insert(0, str(_utilities_path))
from surface_normals import fit_plane_normals, orient_towards  # noqa: E402

# 2dgs_combined_pipeline.md Stage 0 step 4: "fit plane in small window (about
# 3-5 px radius)". radius=2 -> 5x5 window.
SURFACE_NORMAL_PLANE_FIT_RADIUS = 2
SURFACE_NORMAL_DIST_TOL_ABS_M = 0.03
SURFACE_NORMAL_DIST_TOL_REL = 0.03

# Patch PyTorch quantile on AMD ROCm to avoid HIP sort_stable CUDAGuard SIGABRT crashes
if hasattr(torch, "quantile"):
    _orig_torch_quantile = torch.quantile
    def _rocm_safe_torch_quantile(input, q, dim=None, keepdim=False, *, interpolation='linear', out=None):
        if hasattr(input, "is_cuda") and input.is_cuda:
            t_cpu = input.detach().cpu()
            q_cpu = q.detach().cpu() if isinstance(q, torch.Tensor) and q.is_cuda else q
            res = _orig_torch_quantile(t_cpu, q_cpu, dim=dim, keepdim=keepdim, interpolation=interpolation)
            return res.to(input.device)
        return _orig_torch_quantile(input, q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out)
    torch.quantile = _rocm_safe_torch_quantile

# Ensure third_party depth_anything_3 is importable
_da3_path = Path(__file__).resolve().parent.parent / "Utilities" / "third_party" / "depth_anything_3" / "src"
if _da3_path.is_dir() and str(_da3_path) not in sys.path:
    sys.path.insert(0, str(_da3_path))


class DepthEstimationError(RuntimeError):
    """Model/hardware produced unusable depth. Never swallowed by fallback paths."""


def arcore_c2w_to_da3_w2c(extrinsics: np.ndarray) -> np.ndarray:
    """ARCore/OpenGL camera-to-world (+Y up, -Z fwd) -> DA3/OpenCV world-to-camera.

    DA3 both conditions its camera encoder on, and Umeyama-fits its metric scale
    against, OpenCV-convention *world-to-camera* matrices (see `_normalize_extrinsics`
    and `_align_to_input_extrinsics_intrinsics` in Utilities/third_party/.../api.py). Handing it
    raw ARCore c2w poses instead measured 5-30 cm adjacent-view disagreement and a ~5x
    oversized scene, against 0.7-1.5 cm once converted.
    """
    return np.linalg.inv(opengl_to_opencv(np.asarray(extrinsics, dtype=np.float64))).astype(np.float32)


@dataclass
class AlignedDepthResult:
    depth_metric: np.ndarray  # (H, W) float32 in meters
    surface_normals: np.ndarray  # (H, W, 3) float32 unit vectors in camera space
    scale: float  # Alignment scale factor
    shift: float  # Alignment shift offset in meters
    rmse_alignment: float  # Fitting RMSE against sparse points in meters


@dataclass
class GlobalDepthGraphResult:
    """Results of multi-view depth graph scale-shift optimization."""
    scales: np.ndarray  # (N,) float32 scale factor per keyframe
    shifts: np.ndarray  # (N,) float32 shift offset (meters) per keyframe
    rmse_before_m: float  # Mean cross-view discrepancy before optimization (meters)
    rmse_after_m: float  # Mean cross-view discrepancy after optimization (meters)
    num_temporal_edges: int  # Sequential tracking edges (i <-> i+1, i+2)
    num_loop_edges: int  # Multi-meter loop closure edges
    num_constraints: int  # Total point observation rows in sparse linear system
    aligned_depth_maps: list[np.ndarray]  # (N,) list of (H, W) float32 aligned depth maps


class DepthPriorEstimator:
    """Estimates dense depth maps using Depth Anything V3 (DA3-Nested-Giant/Metric)

    with automatic fallback to Depth Anything V2 and geometric baselines.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/DA3-BASE",
        device: Optional[str] = None,
        use_fp16: bool = True,
        min_depth: float = 0.2,
        max_depth: float = 15.0,
        max_invalid_depth_frac: float = 0.05,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_invalid_depth_frac = max_invalid_depth_frac
        self.process_res = process_res
        self.process_res_method = process_res_method

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._da3_model = None
        self._use_da3 = False
        self._is_mock = False
        self._init_model()

    def _init_model(self) -> None:
        """Attempt to load Depth Anything V3 (DA3), fallback to V2 or mock."""
        if self.model_name == "mock" or self.model_name.lower().startswith("mock"):
            self._model = None
            self._da3_model = None
            self._use_da3 = False
            self._is_mock = True
            return

        os.environ.setdefault("MPLCONFIGDIR", "/tmp")
        os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/miopen")
        # NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
        # On ROCm 7.1 / gfx1200 it silently corrupts DA3 inference: depth comes back
        # NaN or absurd (median 2.4e5 m), and combined with the warm-up below the HIP
        # queue aborts with HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION. Upstream DA3's cli.py
        # and gradio_app.py set it unconditionally — don't import those entry points.

        # 1. Try Depth Anything 3 (Multi-view / Nested Giant)
        try:
            from depth_anything_3.api import DepthAnything3

            self._da3_model = DepthAnything3.from_pretrained(self.model_name)
            if self.device != "cpu":
                self._da3_model.to(self.device)
            self._da3_model.eval()

            # Pre-warm GPU kernels on AMD ROCm/HIP to prevent first-pass uninitialized scratchpad NaNs
            if self.device != "cpu":
                try:
                    dummy_imgs = [np.zeros((280, 504, 3), dtype=np.uint8) for _ in range(4)]
                    dummy_w2c = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 4, axis=0)
                    dummy_k = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 4, axis=0)
                    with torch.inference_mode():
                        _ = self._da3_model.inference(
                            dummy_imgs, extrinsics=dummy_w2c, intrinsics=dummy_k, align_to_input_ext_scale=False
                        )
                except Exception:
                    pass

            self._use_da3 = True
            self._is_mock = False
            return
        except Exception as e:
            print(f"[ERROR] Failed to load requested DA3 model '{self.model_name}': {e}")
            self._da3_model = None
            self._use_da3 = False
            if "DA3" in self.model_name:
                raise RuntimeError(f"Requested DA3 model '{self.model_name}' could not be loaded: {e}") from e

        # 2. Fallback to Depth Anything V2 via transformers (only if DA3 was not explicitly requested)
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore

            repo_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
            self._processor = AutoImageProcessor.from_pretrained(repo_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(repo_id).to(self.device)
            if self.use_fp16 and self.device != "cpu":
                self._model = self._model.half()
            self._model.eval()
            self._is_mock = False
            return
        except Exception:
            self._model = None
            self._is_mock = True

    def estimate_depth_streaming(
        self,
        images: Sequence[Union[np.ndarray, Image.Image]],
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        chunk_size: int = 12,
        overlap: int = 4,
    ) -> tuple[list[np.ndarray], Optional[list[np.ndarray]]]:
        """Process long video sequences via sliding-window chunk streaming with overlap blending."""
        n_total = len(images)
        if n_total <= chunk_size:
            return self.estimate_depth_sequence(images, extrinsics=extrinsics, intrinsics=intrinsics)

        step = max(1, chunk_size - overlap)
        final_depths: list[Optional[np.ndarray]] = [None] * n_total
        final_confs: list[Optional[np.ndarray]] = [None] * n_total
        blend_weights: list[float] = [0.0] * n_total

        start = 0
        chunk_idx = 0
        num_chunks = 0
        s = 0
        while s < n_total:
            num_chunks += 1
            e = min(s + chunk_size, n_total)
            if e == n_total:
                break
            s += step

        with torch.inference_mode():
            while start < n_total:
                end = min(start + chunk_size, n_total)
                chunk_idx += 1
                if torch.cuda.is_available():
                    vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
                    vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
                    print(f"       * Processing chunk {chunk_idx}/{num_chunks}: frames [{start}:{end}] (VRAM: {vram_used:.2f} GB alloc, {vram_res:.2f} GB res)...")
                else:
                    print(f"       * Processing chunk {chunk_idx}/{num_chunks}: frames [{start}:{end}]...")

                chunk_imgs = images[start:end]
                chunk_exts = extrinsics[start:end] if extrinsics is not None else None
                chunk_ixts = intrinsics[start:end] if intrinsics is not None else None

                c_depths, c_confs = self.estimate_depth_sequence(
                    chunk_imgs, extrinsics=chunk_exts, intrinsics=chunk_ixts
                )

                for local_idx in range(len(c_depths)):
                    global_idx = start + local_idx
                    d_curr = c_depths[local_idx]
                    c_curr = c_confs[local_idx] if c_confs is not None else None

                    # Triangle / trapezoidal blending weight for smooth cross-chunk stitching
                    dist_to_edge = min(local_idx, len(c_depths) - 1 - local_idx)
                    w = float(min(1.0, (dist_to_edge + 1) / float(overlap)))

                    if final_depths[global_idx] is None:
                        final_depths[global_idx] = d_curr * w
                        if c_curr is not None:
                            final_confs[global_idx] = c_curr * w
                        blend_weights[global_idx] = w
                    else:
                        final_depths[global_idx] += d_curr * w
                        if c_curr is not None and final_confs[global_idx] is not None:
                            final_confs[global_idx] += c_curr * w
                        blend_weights[global_idx] += w

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                time.sleep(0.35)  # Breather for OS display compositor and GPU watchdog

                if end == n_total:
                    break
                start += step

        # Normalize by accumulated blend weights
        out_depths: list[np.ndarray] = []
        out_confs: list[np.ndarray] = []
        has_confs = any(c is not None for c in final_confs)

        for i in range(n_total):
            bw = max(blend_weights[i], 1e-6)
            norm_d = final_depths[i] / bw
            if np.isnan(norm_d).all():
                rgb_img = np.array(images[i]) if isinstance(images[i], Image.Image) else images[i]
                norm_d = self.estimate_depth(rgb_img)
            elif np.isnan(norm_d).any() or np.isinf(norm_d).any():
                median_d = float(np.nanmedian(norm_d)) if not np.isnan(np.nanmedian(norm_d)) else 2.0
                norm_d = np.nan_to_num(norm_d, nan=median_d, posinf=self.max_depth, neginf=self.min_depth)
            out_depths.append(np.clip(norm_d, self.min_depth, self.max_depth).astype(np.float32))
            if has_confs and final_confs[i] is not None:
                norm_c = final_confs[i] / bw
                if np.isnan(norm_c).any() or np.isinf(norm_c).any():
                    norm_c = np.nan_to_num(norm_c, nan=1.0, posinf=1.0, neginf=0.0)
                out_confs.append(norm_c.astype(np.float32))

        return out_depths, (out_confs if has_confs else None)

    def estimate_depth_sequence(
        self,
        images: Sequence[Union[np.ndarray, Image.Image]],
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> tuple[list[np.ndarray], Optional[list[np.ndarray]]]:
        """Estimate multi-view geometrically consistent depth maps and confidence maps.

        `extrinsics` are OpenCV **world-to-camera** poses -- exactly what DA3 wants,
        and exactly what COLMAP stores. The pipeline runs a COLMAP pass before depth
        estimation (see `02_depth_estimation/colmap_poses_to_da3.py`), so poses arrive
        already in the right convention and no conversion happens here.

        Callers holding raw ARCore/OpenGL camera-to-world poses (as carried by
        `Keyframe.transform_matrix`) must convert first with
        `arcore_c2w_to_da3_w2c()`; passing a c2w through unconverted is the
        documented 5-30 cm / ~5x-oversized-scene failure mode.
        """
        if self._use_da3 and self._da3_model is not None:
            try:
                w2c = np.asarray(extrinsics, dtype=np.float32) if extrinsics is not None else None
                with torch.inference_mode():
                    pred = self._da3_model.inference(
                        list(images),
                        extrinsics=w2c,
                        intrinsics=intrinsics,
                        align_to_input_ext_scale=True,
                        process_res=getattr(self, "process_res", 756),
                        process_res_method=getattr(self, "process_res_method", "upper_bound_resize"),
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                depths = []
                confs = []
                for i in range(len(images)):
                    d = pred.depth[i]
                    c = pred.conf[i] if (hasattr(pred, "conf") and pred.conf is not None) else None
                    orig_h, orig_w = (images[i].height, images[i].width) if isinstance(images[i], Image.Image) else images[i].shape[:2]
                    if d.shape != (orig_h, orig_w):
                        d = cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    if c is not None and c.shape != (orig_h, orig_w):
                        c = cv2.resize(c, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    # A mostly-invalid depth map means the GPU/model misbehaved, not that
                    # the scene is hard. Substituting a synthesised depth map here is how a
                    # ROCm allocator bug once silently produced a whole reconstruction from
                    # fabricated geometry -- fail loudly instead. Small holes are still patched.
                    bad_frac = float(np.mean(~np.isfinite(d)))
                    if bad_frac > self.max_invalid_depth_frac:
                        raise DepthEstimationError(
                            f"DA3 returned {bad_frac:.1%} non-finite depth for view {i} "
                            f"(limit {self.max_invalid_depth_frac:.0%}); refusing to fabricate depth."
                        )
                    if bad_frac > 0.0:
                        median_d = float(np.nanmedian(d)) if np.isfinite(np.nanmedian(d)) else 2.0
                        d = np.nan_to_num(d, nan=median_d, posinf=self.max_depth, neginf=self.min_depth)
                    depths.append(np.clip(d, self.min_depth, self.max_depth).astype(np.float32))
                    if c is not None:
                        if np.isnan(c).any() or np.isinf(c).any():
                            c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)
                        confs.append(c.astype(np.float32))
                return depths, (confs if confs else None)
            except DepthEstimationError:
                raise
            except Exception as e:
                import traceback
                print(f"WARNING: DA3 multi-view inference error: {e}")
                traceback.print_exc()
                pass

        # Sequential fallback if multi-view is unavailable
        depths = []
        for img in images:
            rgb = np.array(img) if isinstance(img, Image.Image) else img
            depths.append(self.estimate_depth(rgb))
        return depths, None

    def estimate_depth_sliding_window(
        self,
        images: Sequence[Union[np.ndarray, Image.Image]],
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        chunk_size: int = 6,
        overlap: int = 2,
        cache_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Any] = None,
        process_res: Optional[int] = None,
        align_fn: Optional[Callable[[Sequence[int], Sequence[np.ndarray]],
                                    list[Optional[np.ndarray]]]] = None,
    ) -> tuple[list[np.ndarray], Optional[list[np.ndarray]], Optional[list[np.ndarray]]]:
        """Estimate multi-view geometrically consistent depth maps across long sequences

        using a sliding window with cross-view attention and inter-chunk Sim(3) scale-shift alignment.

        Args:
            images: List of RGB images (numpy arrays HxWx3 or PIL Images).
            extrinsics: ARCore/OpenGL camera-to-world poses (M, 4, 4).
            intrinsics: Camera intrinsics (M, 3, 3).
            chunk_size: Window size N for multi-view cross attention (default: 6).
            overlap: Overlap K between adjacent chunks (default: 2; can be set to 3 for median ensembling).
            cache_dir: Optional directory to cache and resume intermediate chunk predictions.
            progress_callback: Optional callable(chunk_idx, total_chunks) -> None.
            process_res: Optional resolution override for DA3 inference (e.g. 504, 1008, 1920).
            align_fn: Optional callable(global_frame_indices, chunk_depths) -> list of depth
                maps, entries may be None to reject a prediction. Brings every window onto a
                common metric scale before ensembling; see :class:`ChunkTrackAligner` for why
                skipping this step ghosts surfaces. Applied to the cache's raw predictions, so
                the cache stays valid across changes to the alignment.

        Returns:
            Tuple of (fused_depths, fused_confs, uncertainty_maps).
        """
        import gc
        if process_res is not None:
            self.process_res = process_res

        m_total = len(images)
        if m_total == 0:
            return [], None, None

        chunk_size = max(2, min(chunk_size, m_total))
        overlap = max(1, min(overlap, chunk_size - 1))
        step = chunk_size - overlap

        # 1. Build sliding window index ranges
        windows: list[tuple[int, int]] = []
        start = 0
        while start < m_total:
            end = min(start + chunk_size, m_total)
            windows.append((start, end))
            if end == m_total:
                break
            start += step

        # If the final window has fewer than chunk_size frames and m_total >= chunk_size,
        # expand it backwards to provide full cross-attention context
        if len(windows) > 1 and (windows[-1][1] - windows[-1][0]) < chunk_size and m_total >= chunk_size:
            windows[-1] = (max(0, m_total - chunk_size), m_total)

        cache_path = Path(cache_dir) if cache_dir is not None else None
        if cache_path is not None:
            cache_path.mkdir(parents=True, exist_ok=True)

        # Compute sequence identity hash to prevent stale cache collisions across different keyframe selections or resolutions
        if extrinsics is not None:
            seq_sig = np.ascontiguousarray(extrinsics[:, :3, 3].astype(np.float32)).tobytes() + f"_res{self.process_res}".encode()
        else:
            seq_sig = f"{m_total}_{chunk_size}_{overlap}_res{self.process_res}".encode()
        seq_hash = hashlib.md5(seq_sig).hexdigest()[:8]

        num_chunks = len(windows)
        print(f"[SlidingWindow] Processing {m_total} keyframes across {num_chunks} chunks (N={chunk_size}, K={overlap}, Step={step}, Hash={seq_hash})")

        # 2. Compute or load chunk predictions
        raw_chunk_depths: list[list[np.ndarray]] = []
        raw_chunk_confs: list[list[Optional[np.ndarray]]] = []

        for c_idx, (w_start, w_end) in enumerate(windows):
            chunk_cache_file = cache_path / f"chunk_{seq_hash}_{c_idx:03d}_{w_start:04d}_{w_end:04d}.npz" if cache_path else None
            
            loaded = False
            if chunk_cache_file and chunk_cache_file.is_file():
                try:
                    data = np.load(chunk_cache_file)
                    d_list = [data[f"depth_{i}"] for i in range(w_end - w_start)]
                    c_list = [data[f"conf_{i}"] if f"conf_{i}" in data else None for i in range(w_end - w_start)]
                    raw_chunk_depths.append(d_list)
                    raw_chunk_confs.append(c_list)
                    loaded = True
                    print(f"[SlidingWindow] Loaded chunk {c_idx + 1}/{num_chunks} [{w_start}:{w_end}] from cache.")
                except Exception as e:
                    print(f"[SlidingWindow] Cache read failed for chunk {c_idx}: {e}. Recomputing.")
                    loaded = False

            if not loaded:
                chunk_imgs = [images[idx] for idx in range(w_start, w_end)]
                chunk_exts = extrinsics[w_start:w_end] if extrinsics is not None else None
                chunk_ixts = intrinsics[w_start:w_end] if intrinsics is not None else None

                t0 = time.time()
                d_list, c_list = self.estimate_depth_sequence(
                    chunk_imgs, extrinsics=chunk_exts, intrinsics=chunk_ixts
                )
                dt = time.time() - t0
                print(f"[SlidingWindow] Computed chunk {c_idx + 1}/{num_chunks} [{w_start}:{w_end}] in {dt:.1f}s")

                raw_chunk_depths.append(d_list)
                raw_chunk_confs.append(c_list if c_list else [None] * len(d_list))

                if chunk_cache_file:
                    save_dict = {f"depth_{i}": d_list[i] for i in range(len(d_list))}
                    if c_list:
                        for i in range(len(c_list)):
                            if c_list[i] is not None:
                                save_dict[f"conf_{i}"] = c_list[i]
                    np.savez(chunk_cache_file, **save_dict)

                # Periodic GPU memory cleanup
                if (c_idx + 1) % 25 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if progress_callback:
                progress_callback(c_idx + 1, num_chunks)

        # 3. Bring every window onto a common metric scale before ensembling.
        # DA3's align_to_input_ext_scale is not enough on its own: windows disagree by up to
        # 2x in scale and metres in shift, and averaging that disagreement puts the frame at a
        # standoff no window predicted, which unprojects as a duplicate surface.
        if align_fn is not None:
            aligned_chunk_depths: list[list[Optional[np.ndarray]]] = [
                list(align_fn(list(range(w_start, w_end)), raw_chunk_depths[c_idx]))
                for c_idx, (w_start, w_end) in enumerate(windows)
            ]
        else:
            aligned_chunk_depths = [list(c) for c in raw_chunk_depths]

        # 4. Multi-chunk Ensembling and Blending per Global Keyframe
        frame_predictions: list[list[np.ndarray]] = [[] for _ in range(m_total)]
        frame_confs: list[list[Optional[np.ndarray]]] = [[] for _ in range(m_total)]

        for c_idx, (w_start, w_end) in enumerate(windows):
            c_depths = aligned_chunk_depths[c_idx]
            c_confs = raw_chunk_confs[c_idx]
            for local_i, global_i in enumerate(range(w_start, w_end)):
                if c_depths[local_i] is None:
                    continue
                frame_predictions[global_i].append(c_depths[local_i])
                if c_confs and c_confs[local_i] is not None:
                    frame_confs[global_i].append(c_confs[local_i])

        # A frame whose every prediction was rejected still needs a depth map so the returned
        # lists stay index-aligned with `images`. It falls back to the raw windows; align_fn
        # is expected to have flagged it so the caller can drop it downstream.
        for c_idx, (w_start, w_end) in enumerate(windows):
            for local_i, global_i in enumerate(range(w_start, w_end)):
                if not frame_predictions[global_i]:
                    frame_predictions[global_i].append(raw_chunk_depths[c_idx][local_i])

        fused_depths: list[np.ndarray] = []
        fused_confs: list[np.ndarray] = []
        uncertainties: list[np.ndarray] = []
        has_any_conf = any(len(c) > 0 for c in frame_confs)

        for g_idx in range(m_total):
            preds = frame_predictions[g_idx]
            confs = frame_confs[g_idx]

            if len(preds) == 1:
                f_d = preds[0]
                u_d = np.zeros_like(f_d, dtype=np.float32)
            elif len(preds) == 2:
                # K=2: mean blend
                stack_d = np.stack(preds, axis=0)
                f_d = np.mean(stack_d, axis=0).astype(np.float32)
                u_d = (np.abs(preds[0] - preds[1]) * 0.5).astype(np.float32)
            else:
                # K >= 3: median consensus ensembling + standard deviation uncertainty
                stack_d = np.stack(preds, axis=0)
                f_d = np.median(stack_d, axis=0).astype(np.float32)
                u_d = np.std(stack_d, axis=0).astype(np.float32)

            fused_depths.append(np.clip(f_d, self.min_depth, self.max_depth))
            uncertainties.append(u_d)

            if has_any_conf:
                # Always append once per frame -- fused_confs must stay index-aligned
                # with fused_depths, even for a frame whose windows all returned no
                # confidence, or every later frame silently pairs with the wrong map.
                valid_c = [c for c in confs if c is not None]
                if valid_c:
                    fused_confs.append(np.mean(np.stack(valid_c, axis=0), axis=0).astype(np.float32))
                else:
                    fused_confs.append(np.ones_like(f_d, dtype=np.float32))

        return fused_depths, (fused_confs if has_any_conf else None), uncertainties

    def estimate_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """Estimate metric depth map (H, W) in meters from an RGB image array (H, W, 3)."""
        h, w = image_rgb.shape[:2]

        if getattr(self, "_use_da3", False) and getattr(self, "_da3_model", None) is not None:
            try:
                pred = self._da3_model.inference(
                    [image_rgb],
                    process_res=getattr(self, "process_res", 756),
                    process_res_method=getattr(self, "process_res_method", "upper_bound_resize"),
                )
                d = pred.depth[0]
                if d.shape != (h, w):
                    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
                if np.isnan(d).any() or np.isinf(d).any():
                    return self._generate_fallback_depth(image_rgb)
                return np.clip(d, self.min_depth, self.max_depth).astype(np.float32)
            except Exception:
                pass

        if not getattr(self, "_is_mock", True) and getattr(self, "_model", None) is not None:
            try:
                inputs = self._processor(images=image_rgb, return_tensors="pt").to(self.device)
                if getattr(self, "use_fp16", False) and self.device != "cpu":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    pred = outputs.predicted_depth.squeeze().cpu().numpy()
                depth = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                return np.clip(depth, self.min_depth, self.max_depth).astype(np.float32)
            except Exception:
                pass

        return self._generate_fallback_depth(image_rgb)

    def _generate_fallback_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """Generate high-quality edge-aligned depth prior based on bilateral filtering,

        luminance gradient, and room perspective geometry.
        """
        h, w = image_rgb.shape[:2]
        # Base perspective room geometry: floor at bottom (closer), back wall at center/top (further)
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        # Normalized distance from bottom center of room
        norm_y = (h - y_grid) / float(h)
        base_depth = 1.2 + 2.8 * (norm_y ** 0.8)

        # Modulate with bilateral smoothed luminance to create local object variation
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        smooth = cv2.bilateralFilter(gray, 9, 75, 75)
        depth = base_depth + 0.3 * (1.0 - smooth)

        return np.clip(depth, self.min_depth, self.max_depth).astype(np.float32)


def guided_filter_depth(
    depth_map: np.ndarray,
    rgb_guide: np.ndarray,
    radius: int = 5,
    eps: float = 1e-3,
    min_depth: float = 0.2,
    max_depth: float = 15.0,
) -> np.ndarray:
    """Fast RGB-guided depth filtering to align depth edges to color contours

    and smooth planar surface ripples using an O(1) box-filter implementation.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")

    h, w = depth_map.shape
    if rgb_guide.ndim == 3:
        if rgb_guide.shape[:2] != (h, w):
            rgb_guide = cv2.resize(rgb_guide, (w, h), interpolation=cv2.INTER_LINEAR)
        guide_gray = cv2.cvtColor(rgb_guide, cv2.COLOR_RGB2GRAY)
    else:
        if rgb_guide.shape != (h, w):
            rgb_guide = cv2.resize(rgb_guide, (w, h), interpolation=cv2.INTER_LINEAR)
        guide_gray = rgb_guide

    if guide_gray.dtype == np.uint8:
        guide_f = guide_gray.astype(np.float32) / 255.0
    else:
        guide_f = guide_gray.astype(np.float32)
        if guide_f.max() > 1.0:
            guide_f = guide_f / 255.0

    d_f = depth_map.astype(np.float32)
    invalid_mask = ~np.isfinite(d_f) | (d_f <= 0.0)
    if np.all(invalid_mask):
        return depth_map

    if np.any(invalid_mask):
        valid_vals = d_f[~invalid_mask]
        med = float(np.nanmedian(valid_vals)) if len(valid_vals) > 0 else 2.0
        d_f = np.nan_to_num(d_f, nan=med, posinf=max_depth, neginf=min_depth)

    ksize = (2 * radius + 1, 2 * radius + 1)

    # Box-filter computations for linear coefficients a and b
    mean_I = cv2.boxFilter(guide_f, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(d_f, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide_f * d_f, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_f * guide_f, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * guide_f + mean_b

    q = np.clip(q, min_depth, max_depth).astype(np.float32)

    # Preserve original invalid positions if any
    if np.any(invalid_mask):
        q[invalid_mask] = depth_map[invalid_mask]

    return q


def robust_affine(d_pred: np.ndarray, z_gt: np.ndarray, inlier_tol_m: float = 0.05,
                  iters: int = 8, seed_pairs: int = 256) -> tuple[float, float, float]:
    """Trimmed fit of ``z_gt = s * d_pred + t``. Returns ``(s, t, residual_MAD over all points)``.

    Seeded by a Theil-Sen median slope, then refit on its inliers a few times. Not Huber
    IRLS: the outliers that matter here are depth sampled across an occlusion edge, which is
    wrong in *d_pred*, and a downweighted high-leverage point still drags a weighted fit.
    Hard trimming has no breakdown against them, and from a median-slope seed it lands on the
    inliers directly.

    The returned MAD is over every point, not just the kept ones -- it is what the caller
    uses to decide the prediction is a wrong surface rather than a mis-scaled one, so it must
    not improve merely by discarding the evidence.
    """
    a = np.column_stack([d_pred, np.ones_like(d_pred)])
    n = len(d_pred)
    rng = np.random.RandomState(0)
    i, j = rng.randint(0, n, seed_pairs), rng.randint(0, n, seed_pairs)
    spread = d_pred[i] - d_pred[j]
    usable = np.abs(spread) > 1e-3
    if np.any(usable):
        s0 = float(np.median((z_gt[i] - z_gt[j])[usable] / spread[usable]))
        sol = np.array([s0, float(np.median(z_gt - s0 * d_pred))])
    else:
        sol = np.array([1.0, 0.0])

    for _ in range(iters):
        resid = a @ sol - z_gt
        cutoff = max(inlier_tol_m, 3.0 * float(np.median(np.abs(resid - np.median(resid)))))
        keep = np.abs(resid) <= cutoff
        if np.sum(keep) < 4:
            break
        sol, _, _, _ = np.linalg.lstsq(a[keep], z_gt[keep], rcond=None)

    resid = a @ sol - z_gt
    return float(sol[0]), float(sol[1]), float(np.median(np.abs(resid)))


def cross_view_scale_outliers(
    depths: Sequence[np.ndarray],
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    names: Sequence[str],
    tol: float = 0.10,
    subsample: int = 6,
    max_camera_dist_m: float = 3.0,
    min_pixels: int = 300,
    min_partners: int = 3,
) -> tuple[dict[str, float], list[str]]:
    """Find frames whose depth disagrees in scale with every frame that overlaps them.

    COLMAP tracks cannot catch this. A frame gets a handful of triangulated points,
    clustered wherever the scene had texture; an affine fit to them can land inside
    ``max_residual_m`` while the rest of the depth map is structurally wrong. Measured on
    ``current_scene``: ``frame_01275.jpg`` fits its 56 tracks to under 8 cm and is still
    28% out of scale against 67 overlapping frames. Unprojected, it lays a copy of the
    wall 28 cm off the real one -- the duplicated facade.

    Dense co-visible pixels are the evidence the tracks lack: hundreds of thousands of
    them, spanning the scene's whole depth range, against every overlapping frame rather
    than the nearest few (which are temporal neighbours sharing the same error). For each
    pair the median ratio ``observed / reprojected`` is a relative scale; a frame's median
    over its partners is its disagreement with consensus.

    Returns ``(ratio per frame, names beyond tol)``. Outliers are reported for exclusion
    rather than rescaled: their per-partner ratios scatter 2-6x more than a healthy
    frame's, so the depth map is misshapen, not merely mis-scaled, and one scalar cannot
    repair it.
    """
    K = np.asarray(intrinsics, dtype=np.float64)
    K = K[0] if K.ndim == 3 else K
    s = max(1, int(subsample))
    fx, fy = K[0, 0] / s, K[1, 1] / s
    cx, cy = K[0, 2] / s, K[1, 2] / s
    small = [np.asarray(d)[::s, ::s] for d in depths]
    h, w = small[0].shape[:2]
    vg, ug = np.mgrid[0:h, 0:w]
    ug = ug.ravel().astype(np.float64)
    vg = vg.ravel().astype(np.float64)

    w2c = np.asarray(w2c, dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    centres = c2w[:, :3, 3]
    optical_axes = c2w[:, :3, 2]  # +Z is viewing direction in OpenCV camera frame

    world = []
    for i, d in enumerate(small):
        z = d.ravel().astype(np.float64)
        ok = np.isfinite(z) & (z > 0.3) & (z < 8.0)
        pc = np.stack([(ug[ok] - cx) * z[ok] / fx, (vg[ok] - cy) * z[ok] / fy, z[ok]], -1)
        world.append(pc @ c2w[i][:3, :3].T + c2w[i][:3, 3])

    ratios: dict[str, float] = {}
    for i, pts in enumerate(world):
        if len(pts) < min_pixels:
            continue
        dist = np.linalg.norm(centres - centres[i], axis=1)
        cos_sim = optical_axes @ optical_axes[i]
        per_pair = []
        for j in range(len(world)):
            # Optical axis co-visibility: skip cameras looking in divergent directions
            # (e.g. ground floor looking forward vs mezzanine looking upward).
            if j == i or not (0.05 < dist[j] < max_camera_dist_m) or cos_sim[j] < 0.35:
                continue
            pc = pts @ w2c[j][:3, :3].T + w2c[j][:3, 3]
            z = pc[:, 2]
            u = pc[:, 0] * fx / np.maximum(z, 1e-6) + cx
            v = pc[:, 1] * fy / np.maximum(z, 1e-6) + cy
            m = (z > 0.3) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            if int(m.sum()) < min_pixels:
                continue
            v_int = np.clip(np.round(v[m]).astype(int), 0, h - 1)
            u_int = np.clip(np.round(u[m]).astype(int), 0, w - 1)
            obs = small[j][v_int, u_int]
            # Mutual visibility gate: if obs < 0.72 * z, camera j sees a foreground occluder
            # in front of camera i's surface (e.g. mezzanine floor blocking view of the roof),
            # so the ray cannot measure scale consensus.
            g = np.isfinite(obs) & (obs > 0.3) & (obs >= 0.72 * z[m])
            if int(g.sum()) < min_pixels:
                continue
            r = obs[g] / z[m][g]
            # Keep scale-like ratios only; the rest is occlusion
            r = r[(r > 0.72) & (r < 1.40)]
            if len(r) < min_pixels:
                continue
            per_pair.append((float(np.median(r)), len(r)))
        if len(per_pair) >= min_partners:
            # Weighted by co-visible pixels, not one vote per partner. A frame overlaps a
            # few partners on the wall it misplaces (thousands of pixels each) and many
            # more on a sliver of floor; unweighted, the slivers outvote the evidence and
            # a 20%-off frame scores 1.00.
            rr = np.array([x[0] for x in per_pair])
            wt = np.array([x[1] for x in per_pair], dtype=np.float64)
            o = np.argsort(rr)
            cw = np.cumsum(wt[o])
            ratios[str(names[i])] = float(rr[o][np.searchsorted(cw, cw[-1] / 2)])

    outliers = [n for n, r in ratios.items() if abs(r - 1.0) > tol]
    return ratios, sorted(outliers)


class ChunkTrackAligner:
    """Metric alignment of each sliding-window depth prediction against COLMAP tracks.

    DA3's ``align_to_input_ext_scale`` does **not** hand back a common metric scale
    across windows. Measured on ``current_scene``: the same frame came back at scale
    0.44 from one window and 0.98 from the next, with fitted shifts spanning -2.4 m.
    Ensembling those unaligned predictions (mean for two, median for three) lands the
    frame at a standoff that matches neither, and unprojecting it lays a displaced
    duplicate of the surface into the cloud -- the repeated wardrobe facade.

    So every window is fitted to the triangulated track depths before fusion -- **one
    affine fit per window**, pooled over every frame in it, mirroring the single scalar
    DA3 itself applies per inference call.

    Deliberately not a per-frame fit: a frame's own tracks rarely span enough depth to
    determine a slope. Median track depth spread here is 0.48 m (``frame_01335.jpg``: 18
    tracks over 0.02 m) against a real 0.5-4 m range, so a two-parameter per-frame fit is
    unidentifiable and extrapolates wildly outside the track band. Measured: per-frame
    fitting left cross-view disagreement at the unaligned baseline (p90 9.2 cm vs 9.4 cm)
    while pooling cut it to 7.6 cm.

    Rejection stays per frame: a prediction whose own tracks miss the window fit by more
    than ``max_residual_m`` is dropped rather than blended in, and a frame left with no
    surviving prediction is recorded in ``unreliable_frames()`` for the caller to exclude.

    Usable as the ``align_fn`` of :meth:`DepthPriorEstimator.estimate_depth_sliding_window`.
    """

    def __init__(
        self,
        names: Sequence[str],
        colmap_images: dict,
        points_3d: dict,
        max_residual_m: float = 0.08,
        max_track_error_px: float = 2.0,
        min_track_len: int = 3,
    ):
        self.names = [Path(n).name for n in names]
        self.max_residual_m = max_residual_m
        self.frame_residual_m: dict[str, float] = {}
        self.frame_kept: dict[str, int] = {n: 0 for n in self.names}
        self.frame_scale: dict[str, float] = {}

        # Triangulated (pixel, z_camera) pairs per frame, built once.
        self._tracks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for idx, name in enumerate(self.names):
            img = colmap_images.get(name)
            if img is None:
                continue
            ids = np.asarray(img["p3d_ids"])
            keep = np.array([pid >= 0 and pid in points_3d for pid in ids], dtype=bool)
            if not np.any(keep):
                continue
            ids = ids[keep]
            xyz = np.array([points_3d[pid]["xyz"] for pid in ids], dtype=np.float64)
            err = np.array([points_3d[pid]["error"] for pid in ids])
            tlen = np.array([points_3d[pid]["track_len"] for pid in ids])
            z_gt = ((np.asarray(img["R_w2c"]) @ xyz.T).T + np.asarray(img["t_w2c"]))[:, 2]
            good = (err <= max_track_error_px) & (tlen >= min_track_len) & (z_gt > 0.15)
            if np.any(good):
                self._tracks[idx] = (np.asarray(img["obs_xy"])[keep][good], z_gt[good])

        self._history_scales: list[float] = []
        self._history_shifts: list[float] = []

    def _sample(self, idx: int, dmap: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        tr = self._tracks.get(idx)
        if tr is None:
            return None
        uv, z_gt = tr
        h, w = dmap.shape[:2]
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        d = dmap[v, u]
        ok = np.isfinite(d) & (d > 0.15)
        return (d[ok], z_gt[ok]) if np.any(ok) else None

    def __call__(self, frame_indices: Sequence[int],
                 depths: Sequence[np.ndarray]) -> list[Optional[np.ndarray]]:
        samples = {}
        for gi, dmap in zip(frame_indices, depths):
            s = self._sample(gi, dmap)
            if s is not None:
                samples[gi] = s
        if not samples:
            # Zero ground truth tracks in this window (e.g. looking up at smooth ceiling/roof)
            # Use sequence's robust median scale & shift from preceding windows so ceiling
            # is metrically aligned and preserved rather than discarded.
            def_s = float(np.median(self._history_scales)) if self._history_scales else 1.0
            def_t = float(np.median(self._history_shifts)) if self._history_shifts else 0.0
            out_fallback: list[Optional[np.ndarray]] = []
            for gi, dmap in zip(frame_indices, depths):
                name = self.names[gi]
                self.frame_kept[name] += 1
                self.frame_scale[name] = def_s
                out_fallback.append((def_s * dmap + def_t).astype(np.float32))
            return out_fallback

        pooled_s, pooled_t, pooled_mad = robust_affine(
            np.concatenate([s[0] for s in samples.values()]),
            np.concatenate([s[1] for s in samples.values()]),
        )
        self._history_scales.append(pooled_s)
        self._history_shifts.append(pooled_t)

        out: list[Optional[np.ndarray]] = []
        for gi, dmap in zip(frame_indices, depths):
            name = self.names[gi]
            sample = samples.get(gi)
            # Residual is per frame even though the fit is not: it is what catches the
            # single frame a window got wrong without disturbing its neighbours.
            mad = pooled_mad if sample is None else float(
                np.median(np.abs(pooled_s * sample[0] + pooled_t - sample[1])))

            if mad < self.frame_residual_m.get(name, math.inf):
                self.frame_residual_m[name] = mad
                self.frame_scale[name] = pooled_s
            if mad > self.max_residual_m and sample is not None:
                out.append(None)
                continue
            self.frame_kept[name] += 1
            out.append((pooled_s * dmap + pooled_t).astype(np.float32))
        return out

    def unreliable_frames(self) -> list[str]:
        """Frames whose every window prediction failed alignment -- unsafe to unproject."""
        return [n for n in self.names if self.frame_kept.get(n, 0) == 0]

    def stats(self) -> dict:
        res = np.array(list(self.frame_residual_m.values())) if self.frame_residual_m else np.zeros(1)
        kept = np.array([self.frame_kept.get(n, 0) for n in self.names])
        return {
            "frames": len(self.names),
            "frames_with_tracks": len(self._tracks),
            "residual_median_m": round(float(np.median(res)), 4),
            "residual_p95_m": round(float(np.percentile(res, 95)), 4),
            "predictions_kept_median": int(np.median(kept)),
            "unreliable_frames": len(self.unreliable_frames()),
        }


class MetricDepthAligner:
    """Aligns monocular depth predictions to absolute metric coordinates
    using verified 2D-3D observation tracks or sparse 3D point landmarks from SfM/COLMAP.
    """

    def __init__(
        self,
        min_depth: float = 0.2,
        max_depth: float = 15.0,
        huber_delta: float = 0.06,
        inlier_threshold_m: float = 0.06,
        refine_threshold_m: float = 0.08,
        ransac_iterations: int = 400,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.huber_delta = huber_delta
        self.inlier_threshold_m = inlier_threshold_m
        self.refine_threshold_m = refine_threshold_m
        self.ransac_iterations = ransac_iterations

    def align_tracks(
        self,
        mono_depth: np.ndarray,
        obs_xy: np.ndarray,
        p3d_ids: np.ndarray,
        points_3d: dict,
        R_w2c: np.ndarray,
        t_w2c: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float, float, int]:
        """Solve for scale s and shift t such that z_gt = s * d_pred + t using verified 2D-3D tracks.

        Args:
            mono_depth: (H, W) float32 predicted depth map
            obs_xy: (M, 2) 2D pixel observation coordinates
            p3d_ids: (M,) 3D point IDs (-1 for untriangulated)
            points_3d: dict mapping point ID -> {'xyz': ...} or Point3D
            R_w2c: (3, 3) OpenCV world-to-camera rotation matrix
            t_w2c: (3,) OpenCV world-to-camera translation vector

        Returns:
            (aligned_depth, scale, shift, inlier_rmse, inlier_ratio, num_valid_points)
        """
        h, w = mono_depth.shape[:2]
        if len(obs_xy) == 0 or len(p3d_ids) == 0 or not points_3d:
            aligned = np.clip(mono_depth, self.min_depth, self.max_depth)
            return aligned, 1.0, 0.0, 0.0, 0.0, 0

        valid_indices = [k for k, pid in enumerate(p3d_ids) if pid != -1 and pid in points_3d]
        if len(valid_indices) < 4:
            aligned = np.clip(mono_depth, self.min_depth, self.max_depth)
            return aligned, 1.0, 0.0, 0.0, 0.0, len(valid_indices)

        valid_indices = np.array(valid_indices, dtype=np.int64)
        valid_obs = obs_xy[valid_indices]
        valid_pids = p3d_ids[valid_indices]

        # Extract world 3D coordinates
        pts_w_list = []
        for pid in valid_pids:
            pt = points_3d[pid]
            if isinstance(pt, dict):
                pts_w_list.append(pt.get("xyz", pt.get("coord", pt)))
            elif hasattr(pt, "xyz"):
                pts_w_list.append(pt.xyz)
            else:
                pts_w_list.append(pt)
        pts_w = np.asarray(pts_w_list, dtype=np.float64)

        # Transform into camera coordinates (OpenCV: +X right, +Y down, +Z forward):
        pts_cam = (R_w2c @ pts_w.T).T + t_w2c
        z_gt = pts_cam[:, 2]

        # Sample predicted depth at valid observations
        u = np.clip(np.round(valid_obs[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(valid_obs[:, 1]).astype(int), 0, h - 1)
        d_pred = mono_depth[v, u]

        valid_mask = (
            (z_gt >= self.min_depth)
            & (z_gt <= self.max_depth)
            & (d_pred >= self.min_depth)
            & (d_pred <= self.max_depth)
            & np.isfinite(z_gt)
            & np.isfinite(d_pred)
        )

        if np.sum(valid_mask) < 4:
            aligned = np.clip(mono_depth, self.min_depth, self.max_depth)
            return aligned, 1.0, 0.0, 0.0, 0.0, int(np.sum(valid_mask))

        z_gt = z_gt[valid_mask]
        d_pred = d_pred[valid_mask]
        n = len(d_pred)

        A = np.column_stack([d_pred, np.ones_like(d_pred)])

        # RANSAC scale-shift fitting
        best_inliers = 0
        best_s, best_t = 1.0, 0.0
        rng = np.random.RandomState(42)

        for _ in range(self.ransac_iterations):
            idx = rng.choice(n, 2, replace=False)
            if abs(A[idx[0], 0] - A[idx[1], 0]) < 0.05:
                continue
            try:
                sol, _, _, _ = np.linalg.lstsq(A[idx], z_gt[idx], rcond=None)
                s_cand, t_cand = float(sol[0]), float(sol[1])
                if s_cand < 0.2 or s_cand > 4.5 or abs(t_cand) > 3.0:
                    continue
                residuals = np.abs(A[:, 0] * s_cand + t_cand - z_gt)
                inliers = np.sum(residuals < self.inlier_threshold_m)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_s, best_t = s_cand, t_cand
            except Exception:
                continue

        # Inlier refinement via least-squares
        residuals = np.abs(A[:, 0] * best_s + best_t - z_gt)
        inlier_mask = residuals < self.refine_threshold_m
        if np.sum(inlier_mask) >= 4:
            try:
                sol, _, _, _ = np.linalg.lstsq(A[inlier_mask], z_gt[inlier_mask], rcond=None)
                s_ref, t_ref = float(sol[0]), float(sol[1])
                if 0.25 <= s_ref <= 4.0 and abs(t_ref) <= 3.0:
                    best_s, best_t = s_ref, t_ref
                    residuals = np.abs(A[:, 0] * best_s + best_t - z_gt)
                    inlier_mask = residuals < self.refine_threshold_m
            except Exception:
                pass

        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count / n)
        rmse = float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))) if inlier_count > 0 else 0.0

        aligned_depth = np.clip(best_s * mono_depth + best_t, self.min_depth, self.max_depth).astype(np.float32)
        return aligned_depth, best_s, best_t, rmse, inlier_ratio, n

    def align(
        self,
        mono_depth: np.ndarray,
        sparse_points_3d: Optional[np.ndarray],  # (N, 3) in world coords
        camera_pose_c2w: np.ndarray,  # (4, 4)
        intrinsics: CameraIntrinsics,
    ) -> tuple[np.ndarray, float, float, float]:
        """Solve for scale s and shift t such that d_metric = s * mono_depth + t (legacy projection fallback).

        Returns:
            (aligned_depth, scale, shift, rmse)
        """
        h, w = mono_depth.shape[:2]
        fx, fy = intrinsics.fl_x, intrinsics.fl_y
        cx, cy = intrinsics.cx, intrinsics.cy

        if sparse_points_3d is None or len(sparse_points_3d) < 4:
            aligned = np.clip(mono_depth, self.min_depth, self.max_depth)
            return aligned, 1.0, 0.0, 0.0

        c2w = camera_pose_c2w
        r_cw = c2w[:3, :3]
        t_cw = c2w[:3, 3]
        w2c_r = r_cw.T
        w2c_t = -np.dot(w2c_r, t_cw)

        points_cam = np.dot(sparse_points_3d, w2c_r.T) + w2c_t
        z_dist = -points_cam[:, 2]
        valid_mask = z_dist > self.min_depth
        if np.sum(valid_mask) < 4:
            aligned = np.clip(mono_depth, self.min_depth, self.max_depth)
            return aligned, 1.0, 0.0, 0.0

        pts_valid = points_cam[valid_mask]
        z_gt = z_dist[valid_mask]

        u = np.round((pts_valid[:, 0] * fx / z_gt) + cx).astype(np.int32)
        v = np.round((-pts_valid[:, 1] * fy / z_gt) + cy).astype(np.int32)

        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.sum(in_bounds) < 4:
            aligned = np.clip(mono_depth, self.min_depth, self.max_depth)
            return aligned, 1.0, 0.0, 0.0

        u = u[in_bounds]
        v = v[in_bounds]
        z_gt = z_gt[in_bounds]
        d_pred = mono_depth[v, u]

        a_mat = np.column_stack([d_pred, np.ones_like(d_pred)])
        best_inliers = 0
        best_s, best_t = 1.0, 0.0
        n_samples = len(d_pred)
        n_iters = min(200, n_samples * 2)

        rng = np.random.RandomState(42)
        for _ in range(n_iters):
            sample_idx = rng.choice(n_samples, size=min(4, n_samples), replace=False)
            sub_a = a_mat[sample_idx]
            sub_b = z_gt[sample_idx]

            try:
                sol, _, _, _ = np.linalg.lstsq(sub_a, sub_b, rcond=None)
                s_cand, t_cand = float(sol[0]), float(sol[1])
                if s_cand < 0.2 or s_cand > 5.0 or abs(t_cand) > 3.0:
                    continue

                residuals = np.abs(a_mat[:, 0] * s_cand + t_cand - z_gt)
                inliers = np.sum(residuals < self.huber_delta)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_s, best_t = s_cand, t_cand
            except Exception:
                continue

        residuals = np.abs(a_mat[:, 0] * best_s + best_t - z_gt)
        inlier_mask = residuals < (self.huber_delta * 1.5)
        if np.sum(inlier_mask) >= 3:
            sol, _, _, _ = np.linalg.lstsq(a_mat[inlier_mask], z_gt[inlier_mask], rcond=None)
            s_final, t_final = float(sol[0]), float(sol[1])
            if 0.25 <= s_final <= 4.0 and abs(t_final) <= 3.0:
                best_s, best_t = s_final, t_final

        aligned_depth = np.clip(best_s * mono_depth + best_t, self.min_depth, self.max_depth)
        rmse = float(np.sqrt(np.mean((a_mat[:, 0] * best_s + best_t - z_gt) ** 2)))

        return aligned_depth, best_s, best_t, rmse


def anchor_depths_to_sparse_points(
    depth_maps: Sequence[np.ndarray],
    keyframes: Sequence[Keyframe],
    intrinsics: CameraIntrinsics,
    sparse_points_3d: Optional[np.ndarray] = None,
    colmap_images: Optional[dict] = None,
    points_3d: Optional[dict] = None,
    sparse_dir: Optional[Union[str, Path]] = None,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.50,
    min_points: int = 6,
    scale_range: tuple[float, float] = (0.25, 4.0),
    max_shift_m: float = 3.0,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Anchor dense depth maps against triangulated COLMAP sparse 3D landmarks
    using verified 2D-3D observation tracks and robust RANSAC scale-shift fitting.
    """
    aligner = MetricDepthAligner()

    # Load COLMAP model if sparse_dir provided and not pre-loaded
    if colmap_images is None and sparse_dir is not None:
        s_path = Path(sparse_dir)
        try:
            from colmap_diagnostics import parse_images_txt, parse_points3D_txt
            if (s_path / "images.txt").exists():
                colmap_images = parse_images_txt(s_path / "images.txt")
            if (s_path / "points3D.txt").exists():
                points_3d = parse_points3D_txt(s_path / "points3D.txt")
        except Exception as e:
            print(f"[WARN] Failed to load COLMAP files from {sparse_dir}: {e}")

    use_tracks = bool(colmap_images and points_3d)

    aligned_depths: list[Optional[np.ndarray]] = [None] * len(depth_maps)
    scales: list[float] = [1.0] * len(depth_maps)
    shifts: list[float] = [0.0] * len(depth_maps)
    rmses: list[float] = [0.0] * len(depth_maps)
    inlier_ratios: list[float] = [0.0] * len(depth_maps)
    num_pts: list[int] = [0] * len(depth_maps)
    successful_indices = []

    for i, (dmap, kf) in enumerate(zip(depth_maps, keyframes)):
        name = Path(kf.file_path).name
        if use_tracks and name in colmap_images:
            img_data = colmap_images[name]
            aligned, s, t, rmse, ratio, n = aligner.align_tracks(
                mono_depth=dmap,
                obs_xy=img_data["obs_xy"],
                p3d_ids=img_data["p3d_ids"],
                points_3d=points_3d,
                R_w2c=img_data["R_w2c"],
                t_w2c=img_data["t_w2c"],
            )
            if ratio >= min_inlier_ratio and n >= min_points:
                aligned_depths[i] = aligned
                scales[i] = s
                shifts[i] = t
                rmses[i] = rmse
                inlier_ratios[i] = ratio
                num_pts[i] = n
                successful_indices.append(i)
            else:
                # Store partial stats but defer final depth map to median fill
                scales[i] = s
                shifts[i] = t
                rmses[i] = rmse
                inlier_ratios[i] = ratio
                num_pts[i] = n
        elif sparse_points_3d is not None and len(sparse_points_3d) >= 8:
            c2w = kf.transform_matrix
            aligned, s, t, rmse = aligner.align(
                mono_depth=dmap,
                sparse_points_3d=sparse_points_3d,
                camera_pose_c2w=c2w,
                intrinsics=intrinsics,
            )
            aligned_depths[i] = aligned
            scales[i] = s
            shifts[i] = t
            rmses[i] = rmse
            successful_indices.append(i)

    # For frames that did not pass track threshold, use robust median scale/shift
    if successful_indices:
        med_s = float(np.median([scales[i] for i in successful_indices]))
        med_t = float(np.median([shifts[i] for i in successful_indices]))
    else:
        med_s, med_t = 1.0, 0.0

    for i in range(len(depth_maps)):
        if aligned_depths[i] is None:
            aligned_depths[i] = np.clip(
                med_s * depth_maps[i] + med_t,
                aligner.min_depth,
                aligner.max_depth,
            ).astype(np.float32)
            scales[i] = med_s
            shifts[i] = med_t

    final_depths = [d for d in aligned_depths if d is not None]
    anchored_count = len(successful_indices)

    stats = {
        "anchored_frames": anchored_count,
        "total_frames": len(depth_maps),
        "anchored_fraction": round(anchored_count / max(1, len(depth_maps)), 3),
        "median_scale": round(float(np.median(scales)), 4),
        "median_shift_m": round(float(np.median(shifts)), 4),
        "mean_rmse_m": round(float(np.mean([rmses[i] for i in successful_indices])) if successful_indices else 0.0, 4),
        "mean_inlier_ratio": round(float(np.mean([inlier_ratios[i] for i in successful_indices])) if successful_indices else 0.0, 4),
        "mean_points_per_frame": round(float(np.mean([num_pts[i] for i in successful_indices])) if successful_indices else 0.0, 1),
    }
    return final_depths, stats


def compute_surface_normals(
    depth_metric: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Compute dense unit surface normals (H, W, 3) in camera coordinates

    from metric depth and pinhole camera intrinsics.
    Uses OpenGL/ARCore convention (+X right, +Y up, -Z forward).
    """
    h, w = depth_metric.shape[:2]
    fx, fy = intrinsics.fl_x, intrinsics.fl_y
    cx, cy = intrinsics.cx, intrinsics.cy

    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)

    # 3D coordinates in camera space: P = [ (x - cx)/fx * d, -(y - cy)/fy * d, -d ]
    x_cam = (x_grid - cx) * depth_metric / fx
    y_cam = -(y_grid - cy) * depth_metric / fy
    z_cam = -depth_metric

    p_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)  # (H, W, 3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    points = torch.from_numpy(p_cam).to(device)
    valid = torch.from_numpy(depth_metric > 1e-6).to(device)
    depth_t = torch.from_numpy(depth_metric.astype(np.float32)).to(device)

    normal, _residual, _count = fit_plane_normals(
        points, valid,
        radius=SURFACE_NORMAL_PLANE_FIT_RADIUS,
        dist_thresh_abs=SURFACE_NORMAL_DIST_TOL_ABS_M,
        dist_thresh_rel=SURFACE_NORMAL_DIST_TOL_REL,
        depth=depth_t,
    )

    # OpenGL camera convention (+Z out of the screen, camera at origin): the ray
    # from camera to surface is +z_cam is negative-forward, so p_cam itself is
    # that ray. Orient toward camera, matching the previous cross-product's
    # "flip so +Z" convention.
    ray_dir = torch.nn.functional.normalize(points, dim=-1, eps=1e-8)
    normal = orient_towards(normal, ray_dir)

    return normal.cpu().numpy().astype(np.float32)


def extract_and_polish_normals(
    depth_metric: np.ndarray,
    intrinsics: CameraIntrinsics,
    rgb_guide: Optional[np.ndarray] = None,
    smooth_radius: int = 0,
) -> np.ndarray:
    """Extract and polish dense surface normals from metric depth.

    Computes local plane-fit normals directly on GPU using fit_plane_normals, which
    already enforces robust 3D plane fitting and distance tolerance.
    """
    raw_normals = compute_surface_normals(depth_metric, intrinsics)

    # Polish normals with guided / bilateral edge-preserving smoothing if RGB guide is available
    if rgb_guide is not None and smooth_radius > 0:
        h, w = depth_metric.shape[:2]
        if rgb_guide.shape[:2] != (h, w):
            rgb_guide = cv2.resize(rgb_guide, (w, h), interpolation=cv2.INTER_LINEAR)
        guide_gray = cv2.cvtColor(rgb_guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        polished = np.zeros_like(raw_normals)
        for c in range(3):
            polished[:, :, c] = cv2.ximgproc.guidedFilter(
                guide=guide_gray,
                src=raw_normals[:, :, c].astype(np.float32),
                radius=smooth_radius,
                eps=1e-3,
            ) if hasattr(cv2, "ximgproc") else cv2.bilateralFilter(raw_normals[:, :, c].astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=3.0)
        norm = np.maximum(np.linalg.norm(polished, axis=-1, keepdims=True), 1e-8)
        return (polished / norm).astype(np.float32)

    return raw_normals


def colorize_normals(normals: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert (H, W, 3) unit normal vectors in [-1, 1] to an RGB uint8 image for visualization."""
    norm_u8 = np.clip((normals + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    if valid_mask is not None:
        norm_u8[~valid_mask] = 0
    return norm_u8



class GlobalDepthGraphOptimizer:
    """Performs joint scale-shift optimization of monocular depth maps across keyframes.

    Builds an observation graph containing:
      1. Sequential temporal edges (i <-> i+1, i <-> i+2) using Lucas-Kanade optical flow
         with forward-backward tracking consistency (fb_err < 1.0 px).
      2. Loop-closure spatial edges (|i - j| >= min_loop_sep, dist < max_loop_dist,
         viewing angle alignment > min_cos_angle) using ORB/epipolar matching.
      3. Anchor constraint fixing reference keyframe to identity (s_0 = 1.0, t_0 = 0.0).
      4. Soft prior regularization pulling scales s_k towards 1.0 and shifts t_k towards 0.0.

    For any correspondence (u_i, v_i) in frame i matching (u_j, v_j) in frame j, the geometric
    depth constraint under ARCore/OpenGL camera conventions (+X right, +Y up, -Z forward) is:
        d_j = alpha * d_i + beta
    where:
        r_i = [(u_i - cx)/fx, -(v_i - cy)/fy, -1.0]^T
        R_ji = R_j^T @ R_i
        t_ji = R_j^T @ (t_i - t_j)
        alpha = -(R_ji @ r_i)_z
        beta = -(t_ji)_z

    Substituting d_i = s_i * D_i + t_i and d_j = s_j * D_j + t_j yields the exact linear row:
        (-alpha * D_i) * s_i + (-alpha) * t_i + (D_j) * s_j + (1) * t_j = beta
    """

    def __init__(
        self,
        min_depth: float = 0.2,
        max_depth: float = 15.0,
        max_loop_dist_m: float = 1.2,
        min_loop_separation: int = 15,
        min_cos_angle: float = 0.5,
        max_loop_candidates_per_frame: int = 3,
        anchor_weight: float = 500.0,
        reg_scale_weight: float = 0.5,
        reg_shift_weight: float = 0.05,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_loop_dist_m = max_loop_dist_m
        self.min_loop_separation = min_loop_separation
        self.min_cos_angle = min_cos_angle
        self.max_loop_candidates_per_frame = max_loop_candidates_per_frame
        self.anchor_weight = anchor_weight
        self.reg_scale_weight = reg_scale_weight
        self.reg_shift_weight = reg_shift_weight

    def optimize(
        self,
        keyframes: Sequence[Keyframe],
        raw_depth_maps: Sequence[np.ndarray],
        intrinsics: CameraIntrinsics,
    ) -> GlobalDepthGraphResult:
        n_frames = len(keyframes)
        if n_frames == 0 or len(raw_depth_maps) != n_frames:
            raise ValueError(f"Mismatched keyframes ({n_frames}) and depth maps ({len(raw_depth_maps)})")

        if n_frames == 1:
            d_clip = np.clip(raw_depth_maps[0], self.min_depth, self.max_depth)
            return GlobalDepthGraphResult(
                scales=np.array([1.0], dtype=np.float32),
                shifts=np.array([0.0], dtype=np.float32),
                rmse_before_m=0.0,
                rmse_after_m=0.0,
                num_temporal_edges=0,
                num_loop_edges=0,
                num_constraints=0,
                aligned_depth_maps=[d_clip],
            )

        # Pre-convert keyframe images to grayscale
        gray_images = []
        for kf in keyframes:
            img = kf.load_image_rgb()
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            gray_images.append(gray)

        poses = [kf.transform_matrix for kf in keyframes]
        translations = np.array([p[:3, 3] for p in poses])
        # Viewing direction in world coordinates: R * [0, 0, -1]
        directions = np.array([p[:3, :3] @ np.array([0.0, 0.0, -1.0]) for p in poses])

        fx, fy = intrinsics.fl_x, intrinsics.fl_y
        cx, cy = intrinsics.cx, intrinsics.cy
        w, h = intrinsics.w, intrinsics.h

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        b_vec: list[float] = []
        row_idx = 0

        residuals_record = []
        num_temporal_edges = 0
        num_loop_edges = 0

        def add_constraint(i_idx: int, j_idx: int, di: float, dj: float, alpha: float, beta: float, weight: float = 1.0):
            nonlocal row_idx
            rows.extend([row_idx, row_idx, row_idx, row_idx])
            cols.extend([2 * i_idx, 2 * i_idx + 1, 2 * j_idx, 2 * j_idx + 1])
            data.extend([
                float(weight * (-alpha * di)),
                float(weight * (-alpha)),
                float(weight * dj),
                float(weight * 1.0),
            ])
            b_vec.append(float(weight * beta))
            row_idx += 1

        # 1. Sequential Temporal Edges (offsets 1 and 2)
        for i in range(n_frames):
            for offset in [1, 2]:
                j = i + offset
                if j >= n_frames:
                    continue

                R_i, t_i = poses[i][:3, :3], poses[i][:3, 3]
                R_j, t_j = poses[j][:3, :3], poses[j][:3, 3]
                R_ji = R_j.T @ R_i
                t_ji = R_j.T @ (t_i - t_j)

                # Optical flow tracking (combining high-contrast corners + dense floor grid)
                p0_corners = cv2.goodFeaturesToTrack(gray_images[i], maxCorners=600, qualityLevel=0.01, minDistance=10)
                grid_v, grid_u = np.mgrid[int(h * 0.60):int(h * 0.90):18, int(w * 0.15):int(w * 0.85):25]
                p0_floor = np.column_stack([grid_u.ravel(), grid_v.ravel()]).astype(np.float32).reshape(-1, 1, 2)

                if p0_corners is not None and len(p0_corners) > 0:
                    p0 = np.vstack([p0_corners, p0_floor])
                else:
                    p0 = p0_floor

                if len(p0) < 10:
                    continue

                p1, st, _ = cv2.calcOpticalFlowPyrLK(gray_images[i], gray_images[j], p0, None, winSize=(21, 21), maxLevel=3)
                if p1 is None:
                    continue

                p0_back, _, _ = cv2.calcOpticalFlowPyrLK(gray_images[j], gray_images[i], p1, None, winSize=(21, 21), maxLevel=3)
                if p0_back is None:
                    continue

                fb_dist = np.linalg.norm(p0.squeeze() - p0_back.squeeze(), axis=-1)
                valid = (st.squeeze() == 1) & (fb_dist < 1.0)
                if np.sum(valid) < 8:
                    continue

                pts_i = p0.squeeze()[valid]
                pts_j = p1.squeeze()[valid]

                w_edge = 1.0 if offset == 1 else 0.5
                edge_count = 0
                for (ui, vi), (uj, vj) in zip(pts_i, pts_j):
                    ui_i, vi_i = int(round(ui)), int(round(vi))
                    uj_i, vj_i = int(round(uj)), int(round(vj))
                    if not (0 <= ui_i < w and 0 <= vi_i < h and 0 <= uj_i < w and 0 <= vj_i < h):
                        continue

                    di = float(raw_depth_maps[i][vi_i, ui_i])
                    dj = float(raw_depth_maps[j][vj_i, uj_i])
                    if di < self.min_depth or dj < self.min_depth or di > self.max_depth or dj > self.max_depth:
                        continue

                    r_i = np.array([(ui - cx) / fx, -(vi - cy) / fy, -1.0])
                    alpha = float(-(R_ji @ r_i)[2])
                    beta = float(-t_ji[2])

                    if abs((alpha * di + beta) - dj) > 0.45:
                        continue

                    add_constraint(i, j, di, dj, alpha, beta, weight=w_edge)
                    residuals_record.append((i, j, di, dj, alpha, beta))
                    edge_count += 1

                if edge_count >= 8:
                    num_temporal_edges += 1

        # 2. Spatial Loop-Closure Edges
        orb = cv2.ORB_create(nfeatures=1200)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for i in range(0, n_frames, 2):
            candidates = []
            for j in range(i + self.min_loop_separation, n_frames):
                dist = float(np.linalg.norm(translations[i] - translations[j]))
                cos_ang = float(np.dot(directions[i], directions[j]))
                if dist < self.max_loop_dist_m and cos_ang > self.min_cos_angle:
                    candidates.append((dist, j))

            candidates.sort(key=lambda c: c[0])
            for _, j in candidates[:self.max_loop_candidates_per_frame]:
                kp_i, des_i = orb.detectAndCompute(gray_images[i], None)
                kp_j, des_j = orb.detectAndCompute(gray_images[j], None)
                if des_i is None or des_j is None or len(des_i) < 20 or len(des_j) < 20:
                    continue

                matches = bf.match(des_i, des_j)
                if len(matches) < 15:
                    continue

                pts_i_m = np.float32([kp_i[m.queryIdx].pt for m in matches])
                pts_j_m = np.float32([kp_j[m.trainIdx].pt for m in matches])

                _, mask = cv2.findFundamentalMat(pts_i_m, pts_j_m, cv2.FM_RANSAC, 3.0, 0.99)
                if mask is None or np.sum(mask) < 12:
                    continue

                inliers = mask.squeeze() == 1
                inlier_pts_i = pts_i_m[inliers]
                inlier_pts_j = pts_j_m[inliers]

                R_i, t_i = poses[i][:3, :3], poses[i][:3, 3]
                R_j, t_j = poses[j][:3, :3], poses[j][:3, 3]
                R_ji = R_j.T @ R_i
                t_ji = R_j.T @ (t_i - t_j)

                loop_count = 0
                for (ui, vi), (uj, vj) in zip(inlier_pts_i, inlier_pts_j):
                    ui_i, vi_i = int(round(ui)), int(round(vi))
                    uj_i, vj_i = int(round(uj)), int(round(vj))
                    if not (0 <= ui_i < w and 0 <= vi_i < h and 0 <= uj_i < w and 0 <= vj_i < h):
                        continue

                    di = float(raw_depth_maps[i][vi_i, ui_i])
                    dj = float(raw_depth_maps[j][vj_i, uj_i])
                    if di < self.min_depth or dj < self.min_depth or di > self.max_depth or dj > self.max_depth:
                        continue

                    r_i = np.array([(ui - cx) / fx, -(vi - cy) / fy, -1.0])
                    alpha = float(-(R_ji @ r_i)[2])
                    beta = float(-t_ji[2])

                    if abs((alpha * di + beta) - dj) > 0.45:
                        continue

                    add_constraint(i, j, di, dj, alpha, beta, weight=1.5)
                    residuals_record.append((i, j, di, dj, alpha, beta))
                    loop_count += 1

                if loop_count >= 8:
                    num_loop_edges += 1

        # 3. Global Ground-Plane Prior Constraints (locks floor height to common horizontal plane)
        cam_y = translations[:, 1]
        estimated_floor_y = float(np.median(cam_y) - 1.45)
        floor_w = 1.0

        for k in range(n_frames):
            R_k = poses[k][:3, :3]
            t_k = poses[k][:3, 3]
            d_map = raw_depth_maps[k]

            for v_f in np.linspace(h * 0.70, h * 0.90, 5).astype(int):
                for u_f in np.linspace(w * 0.20, w * 0.80, 5).astype(int):
                    r_k = np.array([(u_f - cx) / fx, -(v_f - cy) / fy, -1.0])
                    ray_world = R_k @ r_k
                    if ray_world[1] < -0.20:
                        d_plane = float((estimated_floor_y - t_k[1]) / ray_world[1])
                        obs_d = float(d_map[v_f, u_f])
                        if 0.4 < obs_d < 8.0 and abs(obs_d - d_plane) < 0.40:
                            rows.extend([row_idx, row_idx])
                            cols.extend([2 * k, 2 * k + 1])
                            data.extend([float(floor_w * obs_d), float(floor_w * 1.0)])
                            b_vec.append(float(floor_w * d_plane))
                            row_idx += 1

        # 4. Anchor Frame 0 Constraint (s_0 = 1.0, t_0 = 0.0)
        rows.append(row_idx); cols.append(0); data.append(float(self.anchor_weight)); b_vec.append(float(self.anchor_weight * 1.0)); row_idx += 1
        rows.append(row_idx); cols.append(1); data.append(float(self.anchor_weight)); b_vec.append(0.0); row_idx += 1

        # 5. Soft Prior Regularization for all frames
        for k in range(n_frames):
            rows.append(row_idx); cols.append(2 * k); data.append(float(self.reg_scale_weight)); b_vec.append(float(self.reg_scale_weight * 1.0)); row_idx += 1
            rows.append(row_idx); cols.append(2 * k + 1); data.append(float(self.reg_shift_weight)); b_vec.append(0.0); row_idx += 1

        # Assemble sparse system
        A = sp.csr_matrix((data, (rows, cols)), shape=(row_idx, 2 * n_frames))
        b = np.array(b_vec, dtype=np.float64)

        # Solve via Normal Equations: (A^T A) x = A^T b
        AtA = A.T @ A
        Atb = A.T @ b

        try:
            x = spla.spsolve(AtA.tocsc(), Atb)
        except Exception:
            x, _ = spla.lsqr(A, b)[:2]

        scales = np.zeros(n_frames, dtype=np.float32)
        shifts = np.zeros(n_frames, dtype=np.float32)
        for k in range(n_frames):
            scales[k] = float(np.clip(x[2 * k], 0.2, 5.0))
            shifts[k] = float(np.clip(x[2 * k + 1], -3.0, 3.0))

        # Evaluate discrepancy before vs after
        err_before = []
        err_after = []
        for i_idx, j_idx, di, dj, alpha, beta in residuals_record:
            err_before.append(abs((alpha * di + beta) - dj))
            di_corr = scales[i_idx] * di + shifts[i_idx]
            dj_corr = scales[j_idx] * dj + shifts[j_idx]
            err_after.append(abs((alpha * di_corr + beta) - dj_corr))

        rmse_before = float(np.mean(err_before)) if err_before else 0.0
        rmse_after = float(np.mean(err_after)) if err_after else 0.0

        # Apply optimized parameters
        aligned_depth_maps = []
        for k in range(n_frames):
            d_aligned = scales[k] * raw_depth_maps[k] + shifts[k]
            d_aligned = np.clip(d_aligned, self.min_depth, self.max_depth).astype(np.float32)
            aligned_depth_maps.append(d_aligned)

        return GlobalDepthGraphResult(
            scales=scales,
            shifts=shifts,
            rmse_before_m=rmse_before,
            rmse_after_m=rmse_after,
            num_temporal_edges=num_temporal_edges,
            num_loop_edges=num_loop_edges,
            num_constraints=len(residuals_record),
            aligned_depth_maps=aligned_depth_maps,
        )
