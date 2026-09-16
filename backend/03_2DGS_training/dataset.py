"""GlomeHomeTour Backend: 2DGS Training Dataset Loader.

Loads standardized GS_input packages (transforms.json + images/ + depth_maps/)
with support for progressive multi-scale resolution downsampling and normal prior synthesis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from package_loader import CameraIntrinsics
from depth_priors import compute_surface_normals
from trainer import TrainingKeyframeData


@dataclass
class GSSceneData:
    """Loaded training dataset container."""
    keyframes: List[TrainingKeyframeData]
    intrinsics: CameraIntrinsics
    image_size: Tuple[int, int]  # (H, W)
    total_frames: int
    has_depth_priors: bool


class GSInputDataset:
    """Loads calibrated multi-view frames and geometric depth/normal priors from GS_input."""

    def __init__(
        self,
        gs_input_dir: Union[str, Path],
        depth_maps_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self.gs_input_dir = Path(gs_input_dir).resolve()
        if not self.gs_input_dir.is_dir():
            raise FileNotFoundError(f"GS_input directory not found: {self.gs_input_dir}")

        transforms_path = self.gs_input_dir / "transforms.json"
        if not transforms_path.is_file():
            raise FileNotFoundError(f"transforms.json not found in {self.gs_input_dir}")

        with open(transforms_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Base camera intrinsics
        self.base_w = int(self.metadata.get("w", 1080))
        self.base_h = int(self.metadata.get("h", 1920))
        self.base_fx = float(self.metadata.get("fl_x", 1406.77))
        self.base_fy = float(self.metadata.get("fl_y", 1396.01))
        self.base_cx = float(self.metadata.get("cx", 545.38))
        self.base_cy = float(self.metadata.get("cy", 958.95))

        self.frames_meta = self.metadata.get("frames", [])
        if max_frames is not None and max_frames > 0:
            self.frames_meta = self.frames_meta[:max_frames]

        # Locate depth maps directory if present
        self.depth_maps_dir = None
        if depth_maps_dir is not None:
            cand = Path(depth_maps_dir).resolve()
            if cand.is_dir():
                self.depth_maps_dir = cand
        if self.depth_maps_dir is None:
            # Check adjacent depth_maps folder
            cand_adj = self.gs_input_dir.parent / "depth_maps"
            if cand_adj.is_dir():
                self.depth_maps_dir = cand_adj

        # Parse selected_keyframes.json if present
        self.depth_map_lookup: dict[str, Path] = {}
        self.depth_keyframe_indices: set[int] = set()
        sk_path = self.gs_input_dir / "selected_keyframes.json"
        if not sk_path.is_file():
            sk_path = self.gs_input_dir.parent / "selected_keyframes.json"

        if sk_path.is_file():
            try:
                with open(sk_path, "r", encoding="utf-8") as f:
                    sk_data = json.load(f)
                for entry in sk_data.get("selected_keyframes", []):
                    rel_p = entry.get("gs_input_file_path") or entry.get("file_path")
                    d_rel = entry.get("depth_map")
                    idx_val = entry.get("index")
                    if idx_val is not None:
                        self.depth_keyframe_indices.add(int(idx_val))
                    if rel_p and d_rel and self.depth_maps_dir is not None:
                        d_name = Path(d_rel).name
                        d_path = self.depth_maps_dir / d_name
                        self.depth_map_lookup[rel_p] = d_path
                        self.depth_map_lookup[Path(rel_p).name] = d_path
            except Exception as e:
                print(f"[GSInputDataset] Warning parsing selected_keyframes.json: {e}")

    def select_frames_between_k_and_2k(self) -> list[int]:
        """Select all depth keyframes plus intermediate frames, bounding total N in (K, 2K)."""
        n_total = len(self.frames_meta)
        if not self.depth_keyframe_indices:
            # Fallback if no selected_keyframes.json: return all frames
            return list(range(n_total))

        # Filter keyframes to available frames range (< n_total)
        sorted_kfs = sorted([idx for idx in self.depth_keyframe_indices if idx < n_total])
        if not sorted_kfs:
            return list(range(n_total))

        selected = set(sorted_kfs)

        for i in range(len(sorted_kfs) - 1):
            idx_a = sorted_kfs[i]
            idx_b = sorted_kfs[i + 1]
            gap = idx_b - idx_a
            if gap >= 3:
                mid = (idx_a + idx_b) // 2
                if mid < n_total:
                    selected.add(mid)
            elif gap == 2 and (i % 2 == 0):
                if (idx_a + 1) < n_total:
                    selected.add(idx_a + 1)

        if sorted_kfs[-1] < n_total - 2:
            mid_end = (sorted_kfs[-1] + n_total - 1) // 2
            if mid_end < n_total:
                selected.add(mid_end)

        return sorted([idx for idx in selected if idx < n_total])

    def load_scene_data(
        self,
        target_resolution: Optional[Tuple[int, int]] = None,
        device: Optional[torch.device] = None,
        load_depth_normals: bool = True,
        selected_indices: Optional[Sequence[int]] = None,
    ) -> GSSceneData:
        """Load and cache frames at requested resolution (H, W).

        Args:
            target_resolution: Optional (H, W) to resize frames (e.g. (480, 270) or (960, 540)).
                               If None, native resolution (base_h, base_w) is used.
            device: Target torch device.
            load_depth_normals: Whether to load aligned depth maps and synthesize normal priors.
            selected_indices: Optional subset of frame indices to load.

        Returns:
            GSSceneData containing TrainingKeyframeData list and scaled CameraIntrinsics.
        """
        out_h = target_resolution[0] if target_resolution else self.base_h
        out_w = target_resolution[1] if target_resolution else self.base_w

        scale_x = float(out_w) / float(self.base_w)
        scale_y = float(out_h) / float(self.base_h)

        # Scaled intrinsics
        scaled_intrinsics = CameraIntrinsics(
            camera_model=self.metadata.get("camera_model", "OPENCV"),
            fl_x=self.base_fx * scale_x,
            fl_y=self.base_fy * scale_y,
            cx=self.base_cx * scale_x,
            cy=self.base_cy * scale_y,
            w=out_w,
            h=out_h,
            camera_angle_x=float(self.metadata.get("camera_angle_x", 0.712)),
            k1=float(self.metadata.get("k1", 0.0)),
            k2=float(self.metadata.get("k2", 0.0)),
            p1=float(self.metadata.get("p1", 0.0)),
            p2=float(self.metadata.get("p2", 0.0)),
        )

        keyframes: List[TrainingKeyframeData] = []
        has_depth = False

        active_indices = list(range(len(self.frames_meta))) if selected_indices is None else list(selected_indices)

        for idx in active_indices:
            f_meta = self.frames_meta[idx]
            rel_img_path = f_meta["file_path"]
            img_path = self.gs_input_dir / rel_img_path

            # Load RGB image as uint8 (3, H, W) to save RAM
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil, dtype=np.uint8)  # (H_in, W_in, 3)

            if img_np.shape[:2] != (out_h, out_w):
                img_np = cv2.resize(img_np, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # PyTorch format: (3, H, W) as uint8
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            if device:
                img_tensor = img_tensor.to(device=device)

            # Camera pose: transform_matrix is camera-to-world (c2w)
            c2w = np.array(f_meta["transform_matrix"], dtype=np.float32)
            # World-to-camera is matrix inverse
            w2c_np = np.linalg.inv(c2w)
            w2c_tensor = torch.from_numpy(w2c_np).float()
            if device:
                w2c_tensor = w2c_tensor.to(device=device)

            normal_tensor = None
            depth_tensor = None

            # Load depth and compute surface normals
            if load_depth_normals and self.depth_maps_dir is not None:
                # 1. First check selected_keyframes mapping
                depth_file = self.depth_map_lookup.get(rel_img_path) or self.depth_map_lookup.get(Path(rel_img_path).name)
                # 2. Fallback to index-based naming if lookup not available
                if (depth_file is None or not depth_file.is_file()) and not self.depth_map_lookup:
                    cand_depth = self.depth_maps_dir / f"depth_{idx:04d}.npy"
                    if cand_depth.is_file():
                        depth_file = cand_depth

                if depth_file is not None and depth_file.is_file():
                    d_map = np.load(depth_file).astype(np.float32)  # (base_h, base_w)
                    if d_map.shape != (out_h, out_w):
                        d_map = cv2.resize(d_map, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

                    # Compute surface normals in camera space using scaled intrinsics
                    normals_cam = compute_surface_normals(d_map, scaled_intrinsics)  # (H, W, 3)
                    normal_tensor = torch.from_numpy(normals_cam).permute(2, 0, 1).half()
                    depth_tensor = torch.from_numpy(d_map).unsqueeze(0).half()

                    if device:
                        normal_tensor = normal_tensor.to(device=device)
                        depth_tensor = depth_tensor.to(device=device)
                    has_depth = True

            kf_data = TrainingKeyframeData(
                w2c=w2c_tensor,
                image_rgb=img_tensor,
                normal_prior=normal_tensor,
                depth_prior=depth_tensor,
            )
            keyframes.append(kf_data)

        return GSSceneData(
            keyframes=keyframes,
            intrinsics=scaled_intrinsics,
            image_size=(out_h, out_w),
            total_frames=len(keyframes),
            has_depth_priors=has_depth,
        )
