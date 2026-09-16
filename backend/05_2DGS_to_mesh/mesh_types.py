"""GlomeHomeTour: Mesh Generation Data Contracts & Type Definitions.

Defines Pydantic models and dataclasses for:
- 3D bounding boxes and geometric entities
- PBR materials and texture map references
- Hierarchical scene graph mesh nodes
- 3D segmented instance clusters
- Mesh reconstruction pipeline configuration
- Mesh manifest output schema representation
- PyTorch GPU/ROCm surfel cloud containers
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch


class NodeCategory(str, Enum):
    """Semantic category for segmented CAD/BIM mesh nodes."""
    ARCHITECTURE = "architecture"
    FURNITURE = "furniture"
    APPLIANCE = "appliance"
    STRUCTURAL = "structural"
    FIXTURE = "fixture"
    UNCLASSIFIED = "unclassified"


class BoundingBox3D(BaseModel):
    """Metric axis-aligned 3D bounding box (meters)."""
    model_config = ConfigDict(frozen=True)

    min_point: Tuple[float, float, float] = Field(
        ..., description="Minimum coordinate (x_min, y_min, z_min) in meters"
    )
    max_point: Tuple[float, float, float] = Field(
        ..., description="Maximum coordinate (x_max, y_max, z_max) in meters"
    )
    center: Tuple[float, float, float] = Field(
        ..., description="Center coordinate (cx, cy, cz) in meters"
    )
    extents: Tuple[float, float, float] = Field(
        ..., description="Full width, height, depth extents (dx, dy, dz) in meters"
    )

    @field_validator("extents")
    @classmethod
    def validate_extents(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        for dim in v:
            if dim < -1e-6:
                raise ValueError(f"Extents must be non-negative, got {v}")
        return (max(0.0, float(v[0])), max(0.0, float(v[1])), max(0.0, float(v[2])))

    @property
    def volume(self) -> float:
        """Volume in cubic meters."""
        return float(self.extents[0] * self.extents[1] * self.extents[2])

    def contains_point(self, point: Union[Sequence[float], np.ndarray, torch.Tensor]) -> bool:
        """Check if a 3D point (x, y, z) lies within this bounding box."""
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        tol = 1e-5
        return bool(
            (self.min_point[0] - tol <= px <= self.max_point[0] + tol) and
            (self.min_point[1] - tol <= py <= self.max_point[1] + tol) and
            (self.min_point[2] - tol <= pz <= self.max_point[2] + tol)
        )

    @classmethod
    def from_points(
        cls, points: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]]
    ) -> BoundingBox3D:
        """Compute minimum bounding box enclosing an array of 3D points."""
        if isinstance(points, torch.Tensor):
            pts = points.detach().cpu().numpy()
        else:
            pts = np.asarray(points, dtype=np.float32)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) points, got shape {pts.shape}")
        if len(pts) == 0:
            return cls(
                min_point=(0.0, 0.0, 0.0),
                max_point=(0.0, 0.0, 0.0),
                center=(0.0, 0.0, 0.0),
                extents=(0.0, 0.0, 0.0),
            )

        min_xyz = np.min(pts, axis=0)
        max_xyz = np.max(pts, axis=0)
        center = (min_xyz + max_xyz) * 0.5
        extents = np.maximum(0.0, max_xyz - min_xyz)

        return cls(
            min_point=(float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])),
            max_point=(float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2])),
            center=(float(center[0]), float(center[1]), float(center[2])),
            extents=(float(extents[0]), float(extents[1]), float(extents[2])),
        )


class PBRMaterial(BaseModel):
    """PBR material definition adhering to glTF 2.0 metallic-roughness workflow."""
    model_config = ConfigDict(extra="ignore")

    albedo_texture: Optional[str] = Field(
        default=None, description="Relative path to sRGB albedo / base color map"
    )
    roughness_texture: Optional[str] = Field(
        default=None, description="Relative path to linear roughness map"
    )
    metallic_texture: Optional[str] = Field(
        default=None, description="Relative path to linear metallic map"
    )
    normal_texture: Optional[str] = Field(
        default=None, description="Relative path to tangent-space normal map (+Y OpenGL)"
    )
    base_color_factor: Tuple[float, float, float, float] = Field(
        default=(1.0, 1.0, 1.0, 1.0),
        description="RGBA base color multiplier [0.0, 1.0]"
    )
    roughness_factor: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Uniform roughness multiplier [0.0, 1.0]"
    )
    metallic_factor: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Uniform metallic multiplier [0.0, 1.0]"
    )
    double_sided: bool = Field(
        default=False, description="Whether mesh surface is rendered double-sided"
    )


def identity_matrix_4x4() -> List[List[float]]:
    """Return standard 4x4 metric identity matrix."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


class MeshNode(BaseModel):
    """Hierarchical node in exported 3D CAD/BIM scene graph."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Unique node name (e.g. Architecture, Object_001)")
    category: NodeCategory = Field(
        default=NodeCategory.UNCLASSIFIED, description="Functional semantic category"
    )
    instance_id: Optional[str] = Field(
        default=None, description="Identifier of instance cluster (e.g. obj_001)"
    )
    polygon_count: int = Field(default=0, ge=0, description="Number of triangle faces")
    vertex_count: int = Field(default=0, ge=0, description="Number of unique vertices")
    transform_matrix: List[List[float]] = Field(
        default_factory=identity_matrix_4x4,
        description="4x4 metric affine transform matrix [R | t]"
    )
    bounding_box: BoundingBox3D = Field(
        ..., description="Object axis-aligned bounding box in local/parent coordinates"
    )
    cad_layer: Optional[str] = Field(
        default=None, description="Standard CAD layer mapping (e.g. A-WALL, FF-FURN)"
    )
    material: Optional[PBRMaterial] = Field(
        default=None, description="Baked PBR material properties"
    )
    mesh_path: Optional[str] = Field(
        default=None, description="Relative path to external geometry file if not inlined"
    )
    children: List[MeshNode] = Field(
        default_factory=list, description="Child nodes in scene graph hierarchy"
    )

    @field_validator("transform_matrix")
    @classmethod
    def validate_matrix_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 4 or any(len(row) != 4 for row in v):
            raise ValueError("transform_matrix must be a 4x4 matrix")
        return v


class InstanceCluster(BaseModel):
    """Segmented 3D instance containing clustered 2DGS surfel indices."""
    model_config = ConfigDict(extra="ignore")

    cluster_id: str = Field(..., description="Unique instance cluster ID (e.g. obj_001)")
    category: NodeCategory = Field(default=NodeCategory.UNCLASSIFIED)
    surfel_indices: List[int] = Field(
        default_factory=list, description="Indices of surfels belonging to this cluster"
    )
    bounding_box: BoundingBox3D = Field(
        ..., description="3D bounding box enclosing cluster surfels"
    )
    centroid: Tuple[float, float, float] = Field(
        ..., description="Metric centroid (cx, cy, cz) of cluster"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Segmentation confidence score"
    )
    prototype_id: Optional[str] = Field(
        default=None, description="Assigned prototype cluster ID for de-duplication"
    )
    relative_transform: Optional[List[List[float]]] = Field(
        default=None, description="4x4 transform to prototype canonical coordinates"
    )


class MeshPipelineConfig(BaseModel):
    """Runtime configuration for 2DGS-to-CAD mesh generation pipeline."""
    model_config = ConfigDict(extra="ignore")

    # Hardware and ROCm target
    device: str = Field(default="cuda", description="PyTorch target device ('cuda' or 'cpu')")
    target_arch: str = Field(default="gfx1200", description="Target GPU ISA (RDNA4 Wave32)")
    wavefront_size: int = Field(default=32, description="Wave32 native execution thread block")
    lds_max_bytes: int = Field(default=32768, description="LDS capacity per CU (32 KB)")
    compositor_yield_seconds: float = Field(
        default=0.18, description="Desktop compositor non-blocking yield interval"
    )

    # Resolution & geometry limits
    texture_resolution: int = Field(
        default=2048, description="Resolution of baked square PBR textures (2048 or 4096)"
    )
    max_triangles_per_object: int = Field(
        default=50000, description="Decimation polygon target per foreground object"
    )
    max_triangles_architecture: int = Field(
        default=150000, description="Decimation polygon target for room shell"
    )
    min_cluster_surfels: int = Field(
        default=100, description="Minimum surfel count to accept an object cluster"
    )
    manhattan_alignment: bool = Field(
        default=True, description="Enforce Manhattan-world orthogonal snapping on room planes"
    )
    prototype_dedup_threshold: float = Field(
        default=0.85, description="Cosine similarity threshold for prototype de-duplication"
    )

    # Output exports
    export_gltf: bool = Field(default=True, description="Generate glTF 2.0 scene")
    export_glb: bool = Field(default=True, description="Generate standalone packed .glb")
    export_dxf: bool = Field(default=True, description="Generate AutoCAD 2D/3D .dxf")
    export_ifc: bool = Field(default=True, description="Generate BIM Industry Foundation Classes .ifc")
    uv_padding: int = Field(default=4, description="Padding in pixels between UV charts")


class MeshManifest(BaseModel):
    """Output manifest conforming to shared/schemas/mesh_manifest.schema.json."""
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["1.0.0"] = Field(
        default="1.0.0", description="Contract schema version"
    )
    scene_id: str = Field(..., description="Unique scene listing identifier")
    units: Literal["meters"] = Field(
        default="meters", description="Metric linear units for all geometry"
    )
    coordinate_system: str = Field(
        default="+Y_UP_-Z_FORWARD",
        description="Coordinate frame convention (+Y up, -Z forward, +X right)"
    )
    total_triangles: int = Field(..., ge=0, description="Total polygon count across all nodes")
    total_vertices: int = Field(..., ge=0, description="Total vertex count across all nodes")
    scene_bounds: BoundingBox3D = Field(..., description="Total bounding box enclosing the scene")
    assets: Dict[str, str] = Field(
        ..., description="Map of asset formats to relative file paths"
    )
    nodes: List[MeshNode] = Field(
        ..., description="List of top-level scene nodes (Architecture and Objects)"
    )


@dataclass
class SurfelCloudTorch:
    """PyTorch ROCm/CUDA/CPU container for 2D Gaussian surfels."""
    positions: torch.Tensor    # (N, 3) float32 in meters
    normals: torch.Tensor      # (N, 3) float32 unit vectors
    scales_2d: torch.Tensor    # (N, 2) float32 (sigma_u, sigma_v)
    colors_rgb: torch.Tensor   # (N, 3) float32 in [0, 1]
    opacities: torch.Tensor    # (N, 1) or (N,) float32 in [0, 1]
    rotations: Optional[torch.Tensor] = None   # (N, 4) float32 quaternions (qx, qy, qz, qw)
    tangent_u: Optional[torch.Tensor] = None   # (N, 3) float32 unit vectors
    tangent_v: Optional[torch.Tensor] = None   # (N, 3) float32 unit vectors
    roughness: Optional[torch.Tensor] = None   # (N, 1) float32 in [0, 1]
    metallic: Optional[torch.Tensor] = None    # (N, 1) float32 in [0, 1]
    features: Optional[torch.Tensor] = None    # (N, D) float32 identity features

    def __post_init__(self) -> None:
        if self.tangent_u is None or self.tangent_v is None:
            self.compute_tangent_frame()

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    def to(self, device: Union[str, torch.device]) -> SurfelCloudTorch:
        """Move all non-None tensors to specified device."""
        target = torch.device(device)
        return SurfelCloudTorch(
            positions=self.positions.to(target),
            normals=self.normals.to(target),
            scales_2d=self.scales_2d.to(target),
            colors_rgb=self.colors_rgb.to(target),
            opacities=self.opacities.to(target),
            rotations=self.rotations.to(target) if self.rotations is not None else None,
            tangent_u=self.tangent_u.to(target) if self.tangent_u is not None else None,
            tangent_v=self.tangent_v.to(target) if self.tangent_v is not None else None,
            roughness=self.roughness.to(target) if self.roughness is not None else None,
            metallic=self.metallic.to(target) if self.metallic is not None else None,
            features=self.features.to(target) if self.features is not None else None,
        )

    def cpu(self) -> SurfelCloudTorch:
        return self.to("cpu")

    def cuda(self) -> SurfelCloudTorch:
        return self.to("cuda")

    def compute_tangent_frame(self) -> None:
        """Construct orthonormal right-handed tangent vectors (u, v) from normals."""
        norm = self.normals
        near_z = torch.abs(norm[:, 2]) > 0.9

        ref = torch.zeros_like(norm)
        ref[near_z, 0] = 1.0   # [1, 0, 0]
        ref[~near_z, 2] = 1.0  # [0, 0, 1]

        u = torch.linalg.cross(norm, ref)
        u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-6)
        tangent_u = u / u_norm

        v = torch.linalg.cross(norm, tangent_u)
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)
        tangent_v = v / v_norm

        self.tangent_u = tangent_u
        self.tangent_v = tangent_v

    def compute_bounding_box(self) -> BoundingBox3D:
        """Compute axis-aligned 3D bounding box for all surfels."""
        return BoundingBox3D.from_points(self.positions)

    def slice(self, indices: Union[Sequence[int], torch.Tensor, np.ndarray]) -> SurfelCloudTorch:
        """Extract subset of surfels by index tensor."""
        if not isinstance(indices, torch.Tensor):
            idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        else:
            idx = indices.to(self.device, dtype=torch.long)

        return SurfelCloudTorch(
            positions=self.positions[idx],
            normals=self.normals[idx],
            scales_2d=self.scales_2d[idx],
            colors_rgb=self.colors_rgb[idx],
            opacities=self.opacities[idx],
            rotations=self.rotations[idx] if self.rotations is not None else None,
            tangent_u=self.tangent_u[idx] if self.tangent_u is not None else None,
            tangent_v=self.tangent_v[idx] if self.tangent_v is not None else None,
            roughness=self.roughness[idx] if self.roughness is not None else None,
            metallic=self.metallic[idx] if self.metallic is not None else None,
            features=self.features[idx] if self.features is not None else None,
        )

    def to_ply(self, output_path: Union[str, Path]) -> None:
        """Export surfels to standard binary little-endian PLY file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self)
        pos = self.positions.detach().cpu().numpy()
        norm = self.normals.detach().cpu().numpy()
        colors = (self.colors_rgb.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        scales = self.scales_2d.detach().cpu().numpy()
        op = self.opacities.detach().cpu().numpy().reshape(n)

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property float scale_u\n"
            "property float scale_v\n"
            "property float opacity\n"
            "end_header\n"
        )

        with open(out_path, "wb") as f:
            f.write(header.encode("ascii"))
            record_format = "<3f3f3B2ff"
            records = []
            for i in range(n):
                px, py, pz = pos[i]
                nx, ny, nz = norm[i]
                r, g, b = colors[i]
                su, sv = scales[i]
                o = op[i]
                records.append(struct.pack(record_format, px, py, pz, nx, ny, nz, r, g, b, su, sv, o))
            f.write(b"".join(records))
