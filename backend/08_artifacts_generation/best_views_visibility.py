"""GPU camera batches with exact, multithreaded Open3D BVH visibility queries."""

from __future__ import annotations

import time

import numpy as np
import torch


class SurfaceVisibility:
    def __init__(self, vertices, faces, positions, normals, areas, device="cuda"):
        import open3d as o3d

        self.device = torch.device(device)
        self.diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        if not np.isfinite(self.diagonal) or self.diagonal <= 0:
            raise ValueError("Acquisition mesh has zero or invalid extent")
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                                 o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))
        self.points_cpu = np.asarray(positions, dtype=np.float32)
        self.points = torch.as_tensor(self.points_cpu, dtype=torch.float32, device=self.device)
        self.normals = torch.as_tensor(normals, dtype=torch.float32, device=self.device)
        self.areas = torch.as_tensor(areas, dtype=torch.float32, device=self.device)
        self.cpu_raycast_seconds = 0.0
        self.ray_count = 0
        self.projection_check = None

    @torch.inference_mode()
    def evaluate(self, views, intrinsics, width, height):
        import open3d as o3d

        source_views = np.asarray(views, dtype=np.float64)
        views = torch.as_tensor(views, dtype=torch.float32, device=self.device)
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device)
        rotation = views[:, :3, :3]
        camera = torch.matmul(self.points.unsqueeze(0), rotation.transpose(1, 2)) + views[:, None, :3, 3]
        camera_normal = torch.matmul(self.normals.unsqueeze(0), rotation.transpose(1, 2))
        z = camera[..., 2]
        safe_z = z.clamp_min(self.diagonal * 1e-8)
        u = intrinsics[:, None, 0, 0] * camera[..., 0] / safe_z + intrinsics[:, None, 0, 2]
        v = intrinsics[:, None, 1, 1] * camera[..., 1] / safe_z + intrinsics[:, None, 1, 2]
        incidence = (camera_normal * camera).sum(-1).abs() / camera.norm(dim=-1).clamp_min(self.diagonal * 1e-8)
        angle = ((incidence - 0.1) / 0.4).clamp(0, 1)
        pixel_area = intrinsics[:, None, 0, 0] * intrinsics[:, None, 1, 1] * self.areas * incidence / safe_z.square()
        frustum = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        eligible = frustum & (angle > 0)
        if self.projection_check is None:
            sample = np.linspace(0, len(self.points) - 1, min(256, len(self.points)), dtype=int)
            reference = (self.points[sample].cpu().numpy().astype(np.float64)
                         @ views[0, :3, :3].cpu().numpy().astype(np.float64).T
                         + views[0, :3, 3].cpu().numpy().astype(np.float64))
            difference = np.abs(camera[0, sample].cpu().numpy() - reference)
            maximum = float(difference.max())
            if maximum > self.diagonal * 1e-5:
                raise RuntimeError(f"GPU/CPU camera projection disagreement: {maximum}")
            self.projection_check = {"checked_samples": len(sample), "max_camera_coordinate_error": maximum}
        batch_ids, sample_ids = torch.where(eligible)
        visible = torch.zeros_like(z, dtype=torch.bool)
        if len(batch_ids):
            # Identical CPU rays avoid device-dependent FMA changing mesh-edge hits.
            centers = -np.matmul(source_views[:, :3, :3].transpose(0, 2, 1), source_views[:, :3, 3:4])[..., 0]
            origins = centers.astype(np.float32)[batch_ids.cpu().numpy()]
            directions = self.points_cpu[sample_ids.cpu().numpy()] - origins
            rays = np.concatenate((origins, directions), 1)
            started = time.perf_counter()
            hit = self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().copy()
            self.cpu_raycast_seconds += time.perf_counter() - started
            self.ray_count += len(hit)
            tolerance = 1e-4 + self.diagonal * 1e-5 / np.maximum(np.linalg.norm(directions, axis=1), self.diagonal * 1e-8)
            # Directions end at the sampled surface: its expected first hit is t=1.
            visible[batch_ids, sample_ids] = torch.from_numpy(np.abs(hit - 1) <= tolerance).to(self.device)
        return (visible.float() * angle * pixel_area.clamp(0, 1), pixel_area * visible.float(),
                {"in_frame": frustum, "camera_depth": z})
