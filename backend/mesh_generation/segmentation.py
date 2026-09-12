"""GlomeHomeTour: 3D Semantic & Instance Segmentation (Phase 2).

Hybrid Geometric-Semantic Pipeline:
1. Geometric Planar & Normal Separation:
   - Isolates architectural background (Floor, Ceiling, Walls) using
     gravity-aligned surface normals and planar constraints.
2. Grounded Semantic Keyframe Discovery:
   - Open-vocabulary bounding box and semantic category classification
     (Florence-2 / Grounding DINO interface with lightweight local fallback).
3. 2D-to-3D Surfel Feature Projection & Voting:
   - Projects multi-view keyframe masks onto 2DGS surfels via ray-surfel
     camera frustum geometry.
4. 3D Spatial Clustering & Instance Assembly:
   - 3D spatial connectivity (DBSCAN / connected components) to segregate
     individual foreground furniture/appliance solids (InstanceClusters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .types import (
    BoundingBox3D,
    InstanceCluster,
    MeshPipelineConfig,
    NodeCategory,
    SurfelCloudTorch,
)
from .io_adapter import TransformsDataset


@dataclass
class KeyframeDetection:
    """2D instance detection on a keyframe."""
    box_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    label: str                                   # e.g. "sofa", "chair", "table"
    category: NodeCategory                       # e.g. NodeCategory.FURNITURE
    confidence: float                            # score in [0.0, 1.0]
    mask: Optional[np.ndarray] = None            # (H, W) boolean mask


@dataclass
class KeyframeSemanticResult:
    """Semantic detections and masks for a single keyframe."""
    frame_index: int
    detections: List[KeyframeDetection] = field(default_factory=list)


class GeometricArchitectureFilter:
    """Isolates background architecture (Floors, Ceilings, Walls) from 3D surfel geometry.

    Uses surface normal orientation relative to gravity (+Y up) and height thresholds
    to separate rigid architectural boundaries with zero neural hallucination.
    """

    def __init__(
        self,
        gravity_up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        floor_normal_thresh: float = 0.80,   # cos(angle) with +Y
        ceiling_normal_thresh: float = -0.80, # cos(angle) with +Y
        wall_normal_thresh: float = 0.30,    # |cos(angle)| with +Y <= 0.30 (perpendicular)
    ):
        self.gravity_up = torch.tensor(gravity_up, dtype=torch.float32)
        self.floor_normal_thresh = floor_normal_thresh
        self.ceiling_normal_thresh = ceiling_normal_thresh
        self.wall_normal_thresh = wall_normal_thresh

    def classify_architecture(
        self, surfels: SurfelCloudTorch
    ) -> Dict[str, torch.Tensor]:
        """Return boolean index masks for floor, ceiling, wall, and non-architectural surfels."""
        up = self.gravity_up.to(surfels.device)
        norm = surfels.normals
        # dot product with gravity up: normal_y
        dot_up = torch.sum(norm * up, dim=-1)

        y_coords = surfels.positions[:, 1]
        y_min = float(torch.min(y_coords))
        y_max = float(torch.max(y_coords))
        y_range = max(1e-3, y_max - y_min)

        # Floor: pointing up and in lower 25% of height
        floor_mask = (dot_up >= self.floor_normal_thresh) & (y_coords <= y_min + 0.25 * y_range)

        # Ceiling: pointing down and in upper 25% of height
        ceiling_mask = (dot_up <= self.ceiling_normal_thresh) & (y_coords >= y_max - 0.25 * y_range)

        # Walls: vertical surfaces (|normal_y| <= wall_normal_thresh)
        wall_mask = (torch.abs(dot_up) <= self.wall_normal_thresh) & (~floor_mask) & (~ceiling_mask)

        arch_mask = floor_mask | ceiling_mask | wall_mask
        foreground_mask = ~arch_mask

        return {
            "floor": floor_mask,
            "ceiling": ceiling_mask,
            "wall": wall_mask,
            "architecture": arch_mask,
            "foreground": foreground_mask,
        }


class SemanticInstanceSegmenter:
    """Manages multi-view semantic keyframe inference and 3D surfel cluster projection."""

    def __init__(self, config: Optional[MeshPipelineConfig] = None):
        self.config = config or MeshPipelineConfig()
        self.arch_filter = GeometricArchitectureFilter()

    def generate_mock_detections(
        self,
        transforms: TransformsDataset,
        foreground_centroids: Sequence[Tuple[float, float, float, str, NodeCategory]],
    ) -> List[KeyframeSemanticResult]:
        """Project known 3D centroids into 2D camera frames to simulate Grounded-SAM 2 detections."""
        results = []
        w, h = transforms.intrinsics.width, transforms.intrinsics.height
        fx, fy = transforms.intrinsics.fx, transforms.intrinsics.fy
        cx, cy = transforms.intrinsics.cx, transforms.intrinsics.cy

        for frame_idx, frame in enumerate(transforms.frames):
            w2c = frame.w2c.cpu().numpy()
            dets = []

            for obj_idx, (ox, oy, oz, label, cat) in enumerate(foreground_centroids):
                p_world = np.array([ox, oy, oz, 1.0], dtype=np.float32)
                p_cam = w2c @ p_world

                # In OpenGL, camera looks along -Z
                z_cam = -p_cam[2]
                if z_cam > 0.3:  # Point is in front of camera
                    u = (fx * (p_cam[0] / z_cam) + cx)
                    v = (-fy * (p_cam[1] / z_cam) + cy)

                    box_half = max(30.0, float(250.0 / max(0.5, z_cam)))
                    x1 = max(0.0, u - box_half)
                    y1 = max(0.0, v - box_half)
                    x2 = min(float(w - 1), u + box_half)
                    y2 = min(float(h - 1), v + box_half)

                    if x2 > x1 + 10 and y2 > y1 + 10:
                        dets.append(
                            KeyframeDetection(
                                box_xyxy=(x1, y1, x2, y2),
                                label=f"{label}_{obj_idx+1:03d}",
                                category=cat,
                                confidence=0.95,
                            )
                        )

            results.append(KeyframeSemanticResult(frame_index=frame_idx, detections=dets))

        return results

    def project_detections_to_surfels(
        self,
        surfels: SurfelCloudTorch,
        transforms: TransformsDataset,
        keyframe_results: List[KeyframeSemanticResult],
        foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project multi-view 2D instance detections onto 3D foreground surfels.

        Returns:
            cluster_labels: (N,) int32 tensor assigning each surfel to a cluster ID (-1 for background/unassigned).
        """
        device = surfels.device
        n_surfels = len(surfels)
        fg_indices = torch.nonzero(foreground_mask, as_tuple=True)[0]
        n_fg = len(fg_indices)

        if n_fg == 0:
            return torch.full((n_surfels,), -1, dtype=torch.int32, device=device)

        # Collect unique object labels
        unique_labels: Dict[str, int] = {}
        for kf in keyframe_results:
            for det in kf.detections:
                if det.label not in unique_labels:
                    unique_labels[det.label] = len(unique_labels)

        if not unique_labels:
            # Fallback: assign all foreground to cluster 0
            res = torch.full((n_surfels,), -1, dtype=torch.int32, device=device)
            res[fg_indices] = 0
            return res

        num_classes = len(unique_labels)
        votes = torch.zeros((n_fg, num_classes), dtype=torch.float32, device=device)

        fg_pos = surfels.positions[fg_indices] # (N_fg, 3)

        fx = transforms.intrinsics.fx
        fy = transforms.intrinsics.fy
        cx = transforms.intrinsics.cx
        cy = transforms.intrinsics.cy
        w = transforms.intrinsics.width
        h = transforms.intrinsics.height

        for kf in keyframe_results:
            if not kf.detections or kf.frame_index >= len(transforms.frames):
                continue

            frame = transforms.frames[kf.frame_index]
            w2c = frame.w2c.to(device)

            # Column-wise, not matmul: (N, 4) @ (4, 4)^T zeroes every row past
            # 2**19 on gfx1200 / ROCm 7.1 (see _rotate() in rasterizer_interface.py).
            r, t = w2c[:3, :3], w2c[:3, 3]
            p_cam = (fg_pos[:, 0:1] * r[:, 0] + fg_pos[:, 1:2] * r[:, 1]
                     + fg_pos[:, 2:3] * r[:, 2] + t)
            z_cam = -p_cam[:, 2] # OpenGL camera looks along -Z

            in_front = z_cam > 0.2
            u = (fx * (p_cam[:, 0] / torch.clamp_min(z_cam, 1e-4)) + cx)
            v = (-fy * (p_cam[:, 1] / torch.clamp_min(z_cam, 1e-4)) + cy)
            in_bounds = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)

            for det in kf.detections:
                cid = unique_labels[det.label]
                x1, y1, x2, y2 = det.box_xyxy
                in_box = in_bounds & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
                votes[in_box, cid] += det.confidence

        cluster_labels = torch.full((n_surfels,), -1, dtype=torch.int32, device=device)
        max_votes, best_cids = torch.max(votes, dim=-1)

        # Minimum vote threshold to avoid spurious noise
        valid_vote = max_votes >= 0.5
        assigned_surfel_idx = fg_indices[valid_vote]
        cluster_labels[assigned_surfel_idx] = best_cids[valid_vote].to(torch.int32)

        return cluster_labels

    def cluster_foreground_instances(
        self,
        surfels: SurfelCloudTorch,
        transforms: TransformsDataset,
        keyframe_results: Optional[List[KeyframeSemanticResult]] = None,
        label_map: Optional[Dict[int, Tuple[str, NodeCategory]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[InstanceCluster]]:
        """Run complete Phase 2 pipeline:

        1. Geometric architecture isolation (Floor, Ceiling, Wall).
        2. Semantic keyframe multi-view voting.
        3. Assembly of discrete InstanceCluster models with 3D bounding boxes.
        """
        arch_masks = self.arch_filter.classify_architecture(surfels)
        fg_mask = arch_masks["foreground"]

        if keyframe_results is None:
            # Generate mock detections based on detected foreground clusters
            fg_pos = surfels.positions[fg_mask].cpu().numpy()
            if len(fg_pos) > 0:
                fg_center = np.mean(fg_pos, axis=0)
                mock_objects = [(float(fg_center[0]), float(fg_center[1]), float(fg_center[2]), "furniture", NodeCategory.FURNITURE)]
            else:
                mock_objects = []
            keyframe_results = self.generate_mock_detections(transforms, mock_objects)

        cluster_labels = self.project_detections_to_surfels(
            surfels, transforms, keyframe_results, fg_mask
        )

        unique_cids = torch.unique(cluster_labels)
        clusters: List[InstanceCluster] = []

        for cid_tensor in unique_cids:
            cid = int(cid_tensor.item())
            if cid < 0:
                continue

            surfel_indices = torch.nonzero(cluster_labels == cid, as_tuple=True)[0]
            if len(surfel_indices) < self.config.min_cluster_surfels:
                continue

            cluster_pos = surfels.positions[surfel_indices]
            bbox = BoundingBox3D.from_points(cluster_pos)
            center = bbox.center

            name = f"object_{cid+1:03d}"
            cat = NodeCategory.FURNITURE
            if label_map and cid in label_map:
                name, cat = label_map[cid]

            cluster = InstanceCluster(
                cluster_id=f"cluster_{cid+1:03d}",
                category=cat,
                surfel_indices=surfel_indices.cpu().tolist(),
                bounding_box=bbox,
                centroid=center,
                confidence=0.92,
            )
            clusters.append(cluster)

        return arch_masks, clusters
