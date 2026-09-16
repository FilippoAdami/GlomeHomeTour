"""GlomeHomeTour Backend: LightGaussian Vector Quantization & Asset Packaging.

Compresses trained 2DGS material models into an MLS-compliant binary payload (<= 25 MB)
using K-means Vector Quantization (VQ) codebooks, 8-bit indices, and compressed archive serialization.
"""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.cluster.vq import kmeans2

from model import Material2DGSModel


@dataclass
class Compressed2DGSBundle:
    """In-memory quantized 2DGS asset bundle."""
    num_primitives: int
    bbox_min: np.ndarray             # (3,) float32
    bbox_max: np.ndarray             # (3,) float32
    positions_q16: np.ndarray        # (N, 3) uint16
    rotations_q8: np.ndarray         # (N, 4) int8
    albedo_codebook: np.ndarray      # (256, 3) float32
    albedo_indices: np.ndarray       # (N,) uint8
    material_codebook: np.ndarray    # (64, 2) float32 (roughness, metallic)
    material_indices: np.ndarray     # (N,) uint8
    scales_codebook: np.ndarray      # (256, 2) float32
    scales_indices: np.ndarray       # (N,) uint8
    opacities_q8: np.ndarray         # (N,) uint8


class LightGaussianCompressor:
    """Quantizes and compresses Material2DGSModel into <= 25 MB web asset bundle."""

    def __init__(
        self,
        albedo_centroids: int = 256,
        material_centroids: int = 64,
        scales_centroids: int = 256,
    ) -> None:
        self.albedo_centroids = albedo_centroids
        self.material_centroids = material_centroids
        self.scales_centroids = scales_centroids

    def compress(self, model: Material2DGSModel) -> Compressed2DGSBundle:
        """Execute Vector Quantization and fixed-point quantization on model parameters."""
        n = model.num_gaussians
        if n == 0:
            raise ValueError("Cannot compress empty Material2DGSModel (0 Gaussians)")

        pos = model.xyz.detach().cpu().numpy().astype(np.float32)
        rot = model.rotation.detach().cpu().numpy().astype(np.float32)
        scale = model.scaling.detach().cpu().numpy().astype(np.float32)
        op = model.opacity.detach().cpu().numpy().squeeze(-1).astype(np.float32)
        alb = model.albedo.detach().cpu().numpy().astype(np.float32)
        rou = model.roughness.detach().cpu().numpy().squeeze(-1).astype(np.float32)
        met = model.metallic.detach().cpu().numpy().squeeze(-1).astype(np.float32)

        # 1. Position: 16-bit normalized bounding box quantization
        bbox_min = np.min(pos, axis=0) - 0.01
        bbox_max = np.max(pos, axis=0) + 0.01
        bbox_range = np.maximum(bbox_max - bbox_min, 1e-4)

        pos_norm = (pos - bbox_min) / bbox_range
        pos_q16 = np.clip(np.round(pos_norm * 65535.0), 0, 65535).astype(np.uint16)

        # 2. Rotations (quaternions): 8-bit signed quantization (range [-1, 1] -> [-127, 127])
        rot_q8 = np.clip(np.round(rot * 127.0), -127, 127).astype(np.int8)

        # 3. Albedo: 8-bit Vector Quantization (256 codebook centroids)
        k_alb = min(self.albedo_centroids, n)
        alb_codebook, alb_indices = kmeans2(alb, k_alb, minit="points", iter=15)
        # Pad to 256 if n < 256
        if k_alb < 256:
            pad = np.zeros((256 - k_alb, 3), dtype=np.float32)
            alb_codebook = np.vstack([alb_codebook, pad])
        alb_codebook = np.clip(alb_codebook, 0.0, 1.0).astype(np.float32)
        alb_indices = alb_indices.astype(np.uint8)

        # 4. Materials (Roughness & Metallic): Joint 2D VQ (64 centroids)
        mats = np.column_stack([rou, met])
        k_mat = min(self.material_centroids, n)
        mat_codebook, mat_indices = kmeans2(mats, k_mat, minit="points", iter=15)
        if k_mat < 64:
            pad = np.zeros((64 - k_mat, 2), dtype=np.float32)
            mat_codebook = np.vstack([mat_codebook, pad])
        mat_codebook[:, 0] = np.clip(mat_codebook[:, 0], 0.04, 1.0)
        mat_codebook[:, 1] = np.clip(mat_codebook[:, 1], 0.0, 1.0)
        mat_codebook = mat_codebook.astype(np.float32)
        mat_indices = mat_indices.astype(np.uint8)

        # 5. Scales (sigma_u, sigma_v): 2D VQ (256 centroids)
        k_sc = min(self.scales_centroids, n)
        scale_codebook, scale_indices = kmeans2(scale, k_sc, minit="points", iter=15)
        if k_sc < 256:
            pad = np.zeros((256 - k_sc, 2), dtype=np.float32)
            scale_codebook = np.vstack([scale_codebook, pad])
        scale_codebook = np.clip(scale_codebook, 1e-4, 1.0).astype(np.float32)
        scale_indices = scale_indices.astype(np.uint8)

        # 6. Opacity: 8-bit unsigned quantization ([0, 1] -> [0, 255])
        op_q8 = np.clip(np.round(op * 255.0), 0, 255).astype(np.uint8)

        return Compressed2DGSBundle(
            num_primitives=n,
            bbox_min=bbox_min.astype(np.float32),
            bbox_max=bbox_max.astype(np.float32),
            positions_q16=pos_q16,
            rotations_q8=rot_q8,
            albedo_codebook=alb_codebook,
            albedo_indices=alb_indices,
            material_codebook=mat_codebook,
            material_indices=mat_indices,
            scales_codebook=scale_codebook,
            scales_indices=scale_indices,
            opacities_q8=op_q8,
        )

    def serialize_to_binary(self, bundle: Compressed2DGSBundle) -> bytes:
        """Serialize Compressed2DGSBundle to compact little-endian byte stream."""
        bio = io.BytesIO()

        # Header: Magic "2DGS" (4B) + Version (1B) + Reserved (3B) + Num Primitives (uint32, 4B)
        bio.write(b"2DGS")
        bio.write(struct.pack("<BBBI", 1, 0, 0, bundle.num_primitives))

        # Bounding box: 6 floats (24B)
        bio.write(bundle.bbox_min.tobytes())
        bio.write(bundle.bbox_max.tobytes())

        # Codebooks:
        # Albedo codebook: 256 * 3 floats = 3072 B
        bio.write(bundle.albedo_codebook.tobytes())
        # Material codebook: 64 * 2 floats = 512 B
        bio.write(bundle.material_codebook.tobytes())
        # Scales codebook: 256 * 2 floats = 2048 B
        bio.write(bundle.scales_codebook.tobytes())

        # Per-primitive records:
        # positions: N * 3 * 2B = 6N
        bio.write(bundle.positions_q16.tobytes())
        # rotations: N * 4 * 1B = 4N
        bio.write(bundle.rotations_q8.tobytes())
        # albedo indices: N * 1B = 1N
        bio.write(bundle.albedo_indices.tobytes())
        # material indices: N * 1B = 1N
        bio.write(bundle.material_indices.tobytes())
        # scale indices: N * 1B = 1N
        bio.write(bundle.scales_indices.tobytes())
        # opacity: N * 1B = 1N
        bio.write(bundle.opacities_q8.tobytes())

        return bio.getvalue()

    def deserialize_from_binary(self, data: bytes) -> Compressed2DGSBundle:
        """Parse binary byte stream back into Compressed2DGSBundle."""
        bio = io.BytesIO(data)

        magic = bio.read(4)
        if magic != b"2DGS":
            raise ValueError(f"Invalid magic header: expected b'2DGS', got {magic}")

        ver, r1, r2, n = struct.unpack("<BBBI", bio.read(7))

        bbox_min = np.frombuffer(bio.read(12), dtype=np.float32)
        bbox_max = np.frombuffer(bio.read(12), dtype=np.float32)

        alb_cb = np.frombuffer(bio.read(256 * 3 * 4), dtype=np.float32).reshape(256, 3)
        mat_cb = np.frombuffer(bio.read(64 * 2 * 4), dtype=np.float32).reshape(64, 2)
        sc_cb = np.frombuffer(bio.read(256 * 2 * 4), dtype=np.float32).reshape(256, 2)

        pos_q16 = np.frombuffer(bio.read(n * 3 * 2), dtype=np.uint16).reshape(n, 3)
        rot_q8 = np.frombuffer(bio.read(n * 4 * 1), dtype=np.int8).reshape(n, 4)
        alb_idx = np.frombuffer(bio.read(n * 1), dtype=np.uint8)
        mat_idx = np.frombuffer(bio.read(n * 1), dtype=np.uint8)
        sc_idx = np.frombuffer(bio.read(n * 1), dtype=np.uint8)
        op_q8 = np.frombuffer(bio.read(n * 1), dtype=np.uint8)

        return Compressed2DGSBundle(
            num_primitives=n,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            positions_q16=pos_q16,
            rotations_q8=rot_q8,
            albedo_codebook=alb_cb,
            albedo_indices=alb_idx,
            material_codebook=mat_cb,
            material_indices=mat_idx,
            scales_codebook=sc_cb,
            scales_indices=sc_idx,
            opacities_q8=op_q8,
        )

    def decompress(self, bundle: Compressed2DGSBundle, device: Optional[torch.device] = None) -> Material2DGSModel:
        """Reconstruct a floating-point Material2DGSModel from quantized bundle."""
        n = bundle.num_primitives

        # 1. Positions
        bbox_range = bundle.bbox_max - bundle.bbox_min
        pos = bundle.bbox_min + (bundle.positions_q16.astype(np.float32) / 65535.0) * bbox_range

        # 2. Rotations
        rot = bundle.rotations_q8.astype(np.float32) / 127.0
        rot_norm = np.maximum(np.linalg.norm(rot, axis=-1, keepdims=True), 1e-6)
        rot = rot / rot_norm

        # 3. Scales
        scales = bundle.scales_codebook[bundle.scales_indices]
        log_scales = np.log(np.maximum(scales, 1e-6))

        # 4. Opacities
        opacities = bundle.opacities_q8.astype(np.float32) / 255.0
        opacities = np.clip(opacities, 1e-4, 1.0 - 1e-4)
        logit_opacity = np.log(opacities / (1.0 - opacities))[:, None]

        # 5. Albedo
        albedo = bundle.albedo_codebook[bundle.albedo_indices]
        albedo = np.clip(albedo, 1e-4, 1.0 - 1e-4)
        logit_albedo = np.log(albedo / (1.0 - albedo))

        # 6. Materials
        materials = bundle.material_codebook[bundle.material_indices]
        roughness = np.clip(materials[:, 0:1], 0.04, 0.999)
        metallic = np.clip(materials[:, 1:2], 1e-4, 1.0 - 1e-4)
        logit_roughness = np.log(roughness / (1.0 - roughness))
        logit_metallic = np.log(metallic / (1.0 - metallic))

        model = Material2DGSModel(
            xyz=torch.from_numpy(pos).to(device=device, dtype=torch.float32),
            rotation=torch.from_numpy(rot).to(device=device, dtype=torch.float32),
            scaling=torch.from_numpy(log_scales).to(device=device, dtype=torch.float32),
            opacity=torch.from_numpy(logit_opacity).to(device=device, dtype=torch.float32),
            albedo=torch.from_numpy(logit_albedo).to(device=device, dtype=torch.float32),
            roughness=torch.from_numpy(logit_roughness).to(device=device, dtype=torch.float32),
            metallic=torch.from_numpy(logit_metallic).to(device=device, dtype=torch.float32),
        )
        return model

    def export_package_zip(
        self,
        model: Material2DGSModel,
        output_path: Union[str, Path],
    ) -> int:
        """Compress model and write final walkthrough_2dgs.zip archive.

        Returns:
            Compressed archive file size in bytes.
        """
        bundle = self.compress(model)
        bin_data = self.serialize_to_binary(bundle)

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(out_p, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.writestr("walkthrough_2dgs.bin", bin_data)

        return out_p.stat().st_size
