"""GlomeHomeTour: Prototype Clustering & De-duplication (Phase 4.1).

Computes geometric aspect-ratio signatures, volume profiles, and color histograms
for segmented 3D instance clusters. Groups identical/matching items (e.g., identical
dining chairs, barstools, downlights) into unique prototypes, reducing generative
Image-to-3D inference workload by ~65%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .types import BoundingBox3D, InstanceCluster, SurfelCloudTorch


@dataclass
class InstanceSignature:
    """Compact geometric and color signature for similarity comparisons."""
    instance_id: str
    extents: Tuple[float, float, float]    # (dx, dy, dz) sorted or canonical
    volume: float                          # bounding box volume
    aspect_ratio_1: float                  # extents[1] / extents[0]
    aspect_ratio_2: float                  # extents[2] / extents[0]
    color_hist: np.ndarray                 # Normalized 24-bin RGB color histogram (8 per channel)
    centroid: Tuple[float, float, float]
    surfel_indices: List[int]


@dataclass
class PrototypeGroup:
    """Group of identical or near-identical instances sharing one 3D generative model."""
    prototype_id: str
    category: str
    representative_instance_id: str
    instance_ids: List[str]
    representative_surfel_indices: List[int]
    bounding_box: BoundingBox3D


class PrototypeClusterEngine:
    """Computes similarity metrics and clusters foreground furniture into unique prototypes."""

    def __init__(
        self,
        size_tolerance: float = 0.20,       # 20% max relative difference in bounding box dimensions
        color_sim_threshold: float = 0.70,   # Cosine similarity threshold for color histograms
    ):
        self.size_tolerance = size_tolerance
        self.color_sim_threshold = color_sim_threshold

    def compute_signature(
        self,
        cluster: InstanceCluster,
        surfels: SurfelCloudTorch,
    ) -> InstanceSignature:
        """Extract geometric and color signature from an instance cluster."""
        extents = np.array(cluster.bounding_box.extents, dtype=np.float32)
        # Sort horizontal dimensions (X, Z) to remain invariant to 90-degree yaw rotations
        h1, h2 = sorted([float(extents[0]), float(extents[2])])
        height = float(extents[1])
        sorted_extents = (max(1e-3, h1), max(1e-3, height), max(1e-3, h2))

        ar1 = sorted_extents[1] / sorted_extents[0]
        ar2 = sorted_extents[2] / sorted_extents[0]
        vol = max(1e-4, cluster.bounding_box.volume)

        # Color histogram across RGB channels
        idx = cluster.surfel_indices
        if len(idx) > 0:
            colors = surfels.colors_rgb[idx].detach().cpu().numpy() # (N, 3) in [0, 1]
            r_hist, _ = np.histogram(colors[:, 0], bins=8, range=(0.0, 1.0))
            g_hist, _ = np.histogram(colors[:, 1], bins=8, range=(0.0, 1.0))
            b_hist, _ = np.histogram(colors[:, 2], bins=8, range=(0.0, 1.0))
            hist = np.concatenate([r_hist, g_hist, b_hist]).astype(np.float32)
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist = hist / norm
        else:
            hist = np.zeros(24, dtype=np.float32)

        return InstanceSignature(
            instance_id=cluster.cluster_id,
            extents=sorted_extents,
            volume=vol,
            aspect_ratio_1=ar1,
            aspect_ratio_2=ar2,
            color_hist=hist,
            centroid=cluster.centroid,
            surfel_indices=cluster.surfel_indices,
        )

    def compute_similarity(
        self, sig_a: InstanceSignature, sig_b: InstanceSignature
    ) -> float:
        """Compute holistic similarity score in [0, 1] between two instance signatures."""
        # 1. Dimension relative difference
        e_a = np.array(sig_a.extents)
        e_b = np.array(sig_b.extents)
        rel_diff = np.abs(e_a - e_b) / np.maximum(e_a, e_b)
        if np.any(rel_diff > self.size_tolerance):
            return 0.0

        size_score = 1.0 - float(np.mean(rel_diff))

        # 2. Aspect ratio difference
        ar_diff = (
            abs(sig_a.aspect_ratio_1 - sig_b.aspect_ratio_1) / max(sig_a.aspect_ratio_1, sig_b.aspect_ratio_1) +
            abs(sig_a.aspect_ratio_2 - sig_b.aspect_ratio_2) / max(sig_a.aspect_ratio_2, sig_b.aspect_ratio_2)
        ) * 0.5
        ar_score = max(0.0, 1.0 - ar_diff)

        # 3. Color cosine similarity
        color_dot = float(np.dot(sig_a.color_hist, sig_b.color_hist))
        color_score = max(0.0, min(1.0, color_dot))

        if color_score < self.color_sim_threshold:
            return 0.0

        # Weighted aggregate score
        return 0.45 * size_score + 0.25 * ar_score + 0.30 * color_score

    def cluster_prototypes(
        self,
        clusters: List[InstanceCluster],
        surfels: SurfelCloudTorch,
    ) -> List[PrototypeGroup]:
        """Group matching clusters into unique PrototypeGroup items."""
        if not clusters:
            return []

        signatures = [self.compute_signature(c, surfels) for c in clusters]
        assigned = [False] * len(clusters)
        prototypes: List[PrototypeGroup] = []

        proto_counter = 1
        for i in range(len(clusters)):
            if assigned[i]:
                continue

            current_group_indices = [i]
            assigned[i] = True

            for j in range(i + 1, len(clusters)):
                if assigned[j]:
                    continue
                # Ensure same semantic category
                if clusters[i].category != clusters[j].category:
                    continue

                sim = self.compute_similarity(signatures[i], signatures[j])
                if sim >= 0.75:
                    current_group_indices.append(j)
                    assigned[j] = True

            # Pick representative with largest number of supporting surfels
            rep_idx = max(current_group_indices, key=lambda idx: len(clusters[idx].surfel_indices))
            rep_cluster = clusters[rep_idx]
            member_ids = [clusters[k].cluster_id for k in current_group_indices]

            proto_id = f"proto_{proto_counter:03d}"
            proto_counter += 1

            # Update prototype_id on instance clusters in-place
            for k in current_group_indices:
                clusters[k].prototype_id = proto_id

            prototypes.append(
                PrototypeGroup(
                    prototype_id=proto_id,
                    category=str(rep_cluster.category),
                    representative_instance_id=rep_cluster.cluster_id,
                    instance_ids=member_ids,
                    representative_surfel_indices=rep_cluster.surfel_indices,
                    bounding_box=rep_cluster.bounding_box,
                )
            )

        return prototypes
