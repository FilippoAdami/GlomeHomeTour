"""Shared pinhole camera rendering for sealed mesh and multi-scan Gaussian data."""
from __future__ import annotations

import math
import numpy as np
import open3d as o3d

from panorama_rendering import load_gaussians, _rotation_quaternion, _secondary_colors, OPTICAL_MODEL
from cinematic_flythrough_preflight import digest


def intrinsics(width, height, hfov_deg=70):
    if width <= 0 or height <= 0 or not 0 < hfov_deg < 180:
        raise ValueError('Invalid image dimensions or horizontal field of view')
    focal = width / (2 * math.tan(math.radians(hfov_deg) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)


def pinhole_directions(camera_to_world, width, height, hfov_deg=70):
    pose = np.asarray(camera_to_world, dtype=np.float32)
    k = intrinsics(width, height, hfov_deg)
    x, y = np.meshgrid(np.arange(width, dtype=np.float32) + .5,
                       np.arange(height, dtype=np.float32) + .5)
    local = np.stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)), axis=-1)
    return local @ pose[:3, :3].T


class MeshRenderer:
    def __init__(self, vertices, faces, colors):
        self.faces = np.asarray(faces, dtype=np.int32)
        self.colors = np.asarray(colors, dtype=np.uint8)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                                 o3d.core.Tensor(self.faces.astype(np.uint32)))

    def render(self, pose, width, height, hfov_deg=70):
        directions = pinhole_directions(pose, width, height, hfov_deg).reshape(-1, 3)
        rays = np.column_stack((np.broadcast_to(np.asarray(pose[:3, 3], np.float32), directions.shape), directions))
        result = self.scene.cast_rays(o3d.core.Tensor(rays))
        ids = result['primitive_ids'].numpy()
        valid = ids < len(self.faces)
        uv = result['primitive_uvs'].numpy()[valid]
        weights = np.column_stack((1 - uv.sum(axis=1), uv))
        pixels = np.full((len(ids), 3), 255, np.uint8)
        pixels[valid] = np.clip(np.rint(np.einsum('ij,ijk->ik', weights,
                                                  self.colors[self.faces[ids[valid]], :3])), 0, 255).astype(np.uint8)
        return pixels.reshape(height, width, 3)


class GPUMeshRenderer:
    def __init__(self, vertices, faces, colors, width, height, hfov_deg=70):
        self.hfov_deg = hfov_deg
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors[:, :3], dtype=float) / 255)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.renderer.scene.set_background([1, 1, 1, 1])
        self.renderer.scene.view.set_post_processing(False)
        self.renderer.scene.view.set_sample_count(4)
        self.renderer.scene.view.set_antialiasing(True)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = 'defaultUnlit'
        material.sRGB_color = False
        self.renderer.scene.add_geometry('mesh', mesh, material)
        k = intrinsics(width, height, hfov_deg)
        self.k = k.astype(np.float64)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, float(k[0, 0]), float(k[1, 1]),
                                                          float(k[0, 2]), float(k[1, 2]))

    def render(self, pose, width, height, hfov_deg=70):
        if (width, height, hfov_deg) != (self.intrinsic.width, self.intrinsic.height, self.hfov_deg):
            raise ValueError('GPU renderer intrinsics differ from its camera')
        r = pose[:3, :3]
        t = pose[:3, 3]
        inv = np.empty((4, 4), dtype=np.float64)
        inv[:3, :3] = r.T
        inv[:3, 3] = -r.T @ t
        inv[3, :3] = 0.0
        inv[3, 3] = 1.0
        self.renderer.setup_camera(self.intrinsic, inv)
        self.renderer.scene.camera.set_projection(self.k, .01, 1000., width, height)
        return np.asarray(self.renderer.render_to_image()).copy()


def _clip_triangle_to_near(triangle: np.ndarray, near: float = 0.01) -> list[np.ndarray]:
    """Clip a 3D camera-frame triangle against the near plane z = near."""
    z = triangle[:, 2]
    if np.all(z <= near):
        return []
    if np.all(z > near):
        return [triangle]
    pts = []
    for idx in range(3):
        v1 = triangle[idx]
        v2 = triangle[(idx + 1) % 3]
        z1, z2 = v1[2], v2[2]
        if z1 > near:
            pts.append(v1)
        if (z1 > near) != (z2 > near):
            t = (near - z1) / (z2 - z1)
            pts.append(v1 + t * (v2 - v1))
    if not pts:
        return []
    return [np.array(pts, dtype=np.float32)]


class OpticalMeshRenderer:
    """GPU mesh rasterization with one-bounce optics on classified GLB panels."""
    def __init__(self, vertices, faces, colors, panel_types, width, height):
        self.vertices = np.asarray(vertices, np.float32)
        self.faces = np.asarray(faces, np.int32)
        self.colors = np.asarray(colors, np.uint8)
        self.types = np.asarray(panel_types, np.uint8)
        if self.types.shape != (len(self.faces),):
            raise ValueError('Panel classification does not match GLB faces')
        self.gpu = GPUMeshRenderer(vertices, faces, colors, width, height)
        self.primary = o3d.t.geometry.RaycastingScene()
        self.primary.add_triangles(o3d.core.Tensor(self.vertices), o3d.core.Tensor(self.faces.astype(np.uint32)))
        self.environment_faces = self.faces[self.types == 0]
        self.environment = None
        if len(self.environment_faces):
            self.environment = o3d.t.geometry.RaycastingScene()
            self.environment.add_triangles(o3d.core.Tensor(self.vertices), o3d.core.Tensor(self.environment_faces.astype(np.uint32)))
        self.panels = self.vertices[self.faces[self.types != 0]]

    def render(self, pose, width, height, hfov_deg=70):
        image = self.gpu.render(pose, width, height, hfov_deg)
        if not len(self.panels):
            return image
        camera = (self.panels - pose[:3, 3]) @ pose[:3, :3]
        bounds = []
        k = intrinsics(width, height, hfov_deg)
        for triangle in camera:
            for poly in _clip_triangle_to_near(triangle, 0.01):
                xy = poly[:, :2] / poly[:, 2, None] * k[0, 0] + [k[0, 2], k[1, 2]]
                lo = np.maximum(np.floor(xy.min(axis=0) - 1).astype(int), 0)
                hi = np.minimum(np.ceil(xy.max(axis=0) + 1).astype(int), [width, height])
                if np.all(lo < hi):
                    bounds.append((*lo, *hi))
        if not bounds:
            return image
        x0, y0 = np.min(np.asarray(bounds)[:, :2], axis=0)
        x1, y1 = np.max(np.asarray(bounds)[:, 2:], axis=0)
        x, y = np.meshgrid(np.arange(x0, x1, dtype=np.float32) + .5, np.arange(y0, y1, dtype=np.float32) + .5)
        local = np.stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1], np.ones_like(x)), axis=-1)
        directions = (local @ pose[:3, :3].T).reshape(-1, 3).astype(np.float32)
        origin = np.asarray(pose[:3, 3], np.float32)
        rays = np.column_stack((np.broadcast_to(origin, directions.shape), directions))
        hits = self.primary.cast_rays(o3d.core.Tensor(rays))
        ids = hits['primitive_ids'].numpy()
        valid = ids < len(self.faces)
        pixels = np.flatnonzero(valid)
        pixels = pixels[self.types[ids[pixels]] != 0]
        if not len(pixels):
            return image
        # shortcut: one bounce, matching the existing panorama optical model; upgrade for multi-mirror scenes.
        types = self.types[ids[pixels]]
        incoming = directions[pixels]
        normals = hits['primitive_normals'].numpy()[pixels]
        reflected = incoming - 2 * np.sum(incoming * normals, axis=1)[:, None] * normals
        points = origin + hits['t_hit'].numpy()[pixels, None] * incoming
        outgoing = incoming.copy()
        outgoing[types == 1] = reflected[types == 1]
        rgb, _ = _secondary_colors(self.environment, self.environment_faces, self.colors, points, outgoing, .001)
        windows = types == 2
        if windows.any():
            reflection, _ = _secondary_colors(self.environment, self.environment_faces, self.colors,
                                               points[windows], reflected[windows], .001)
            rgb[windows] = (rgb[windows] * OPTICAL_MODEL['window_transmission']
                            + reflection * OPTICAL_MODEL['window_reflection']
                            + np.asarray(OPTICAL_MODEL['window_tint_rgb']) * OPTICAL_MODEL['window_tint'])
        yy, xx = np.divmod(pixels, x1 - x0)
        image[yy + y0, xx + x0] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return image


class GaussianRenderer:
    def __init__(self, inputs, transforms=None, scan_counts=None, device='cuda'):
        import torch
        from pathlib import Path
        self.device = device

        if isinstance(inputs, (str, Path)):
            ply_path = Path(inputs)
            source_hash = digest(ply_path)
            from plyfile import PlyData
            v = PlyData.read(str(ply_path))['vertex'].data
            means = torch.from_numpy(np.column_stack([v[n] for n in ('x', 'y', 'z')]).astype(np.float32))
            quats = torch.from_numpy(np.column_stack([v[f'rot_{i}'] for i in range(4)]).astype(np.float32))
            scales = torch.exp(torch.from_numpy(np.column_stack([v[f'scale_{i}'] for i in range(3)]).astype(np.float32)))
            opacities = torch.sigmoid(torch.from_numpy(np.asarray(v['opacity'], dtype=np.float32)))

            dc = np.column_stack([v[f'f_dc_{i}'] for i in range(3)]).astype(np.float32)[:, None]
            rest_names = sorted((name for name in v.dtype.names if name.startswith('f_rest_')), key=lambda n: int(n[7:]))
            if rest_names:
                flat = np.column_stack([v[name] for name in rest_names]).astype(np.float32)
                rest = flat.reshape(len(v), 3, -1).transpose(0, 2, 1)
                colors_np = np.ascontiguousarray(np.concatenate((dc, rest), axis=1), dtype=np.float32)
                coefficient_count = 1 + len(rest_names) // 3
                sh_deg = math.isqrt(coefficient_count) - 1
            else:
                colors_np = np.ascontiguousarray(dc, dtype=np.float32)
                sh_deg = 0
            colors = torch.from_numpy(colors_np)

            if digest(ply_path) != source_hash:
                raise ValueError(f'Gaussian source changed while loading: {ply_path}')
            world = {
                'means': means.to(device),
                'quats': quats.to(device),
                'scales': scales.to(device),
                'opacities': opacities.to(device),
            }
            self.colors = colors.to(device)
            self.sh_degree = sh_deg

            self.world = world
            self.scan_slices = []
            if transforms is not None:
                offset = 0
                for scan, transform_mat in sorted(transforms.items()):
                    T = np.asarray(transform_mat, dtype=np.float32)
                    R = torch.as_tensor(T[:3, :3], device=device)
                    cnt = None
                    if scan_counts is not None and scan in scan_counts:
                        cnt = scan_counts[scan]
                    else:
                        per_scan_ply = ply_path.parent / 'scans' / scan / 'gaussian_splat/gaussian_splat_sealed.ply'
                        if per_scan_ply.is_file():
                            with open(per_scan_ply, 'rb') as f:
                                for line in f:
                                    if line.startswith(b'element vertex '):
                                        cnt = int(line.split()[2])
                                        break
                    if cnt is None:
                        raise ValueError(f'Cannot determine Gaussian count for scan {scan}')
                    self.scan_slices.append((offset, offset + cnt, R))
                    offset += cnt
                if offset != len(self.world['means']):
                    raise ValueError(f'Gaussian count mismatch: slices={offset} vs total={len(self.world["means"])}')
            else:
                self.scan_slices.append((0, len(self.world['means']), torch.eye(3, device=device)))
        else:
            self.models = []
            world = {key: [] for key in ('means', 'quats', 'scales', 'opacities')}
            for path, transform in inputs:
                model = load_gaussians(path, transform, device=device)
                rotation = torch.as_tensor(np.asarray(transform[:3, :3], np.float32), device=device)
                translation = torch.as_tensor(np.asarray(transform[:3, 3], np.float32), device=device)
                means = model.splats['means'] @ rotation.T + translation
                q = torch.as_tensor(_rotation_quaternion(np.asarray(transform[:3, :3])), device=device)
                source_q = model.splats['quats']
                qw, qx, qy, qz = q.unbind()
                sw, sx, sy, sz = source_q.unbind(-1)
                quats = torch.stack((qw*sw-qx*sx-qy*sy-qz*sz, qw*sx+qx*sw+qy*sz-qz*sy,
                                     qw*sy-qx*sz+qy*sw+qz*sx, qw*sz+qx*sy-qy*sx+qz*sw), dim=-1)
                for key, value in (('means', means), ('quats', quats), ('scales', model.splats['scales']),
                                   ('opacities', model.splats['opacities'])):
                    world[key].append(value)
                self.models.append(model)
            if not self.models:
                raise ValueError('No Gaussian scans provided')
            self.world = {key: torch.cat(value, dim=0) for key, value in world.items()}

    def render(self, pose, width, height, hfov_deg=70):
        import torch
        from gsplat import rasterization
        from gsplat.cuda._wrapper import spherical_harmonics
        pose = np.asarray(pose, dtype=np.float32)
        camera = torch.as_tensor(pose[:3, 3], device=self.device)
        colors = []
        with torch.inference_mode():
            if hasattr(self, 'scan_slices'):
                dirs_world = self.world['means'] - camera
                for start, end, R in self.scan_slices:
                    d_source = dirs_world[start:end] @ R
                    colors.append(torch.clamp_min(spherical_harmonics(self.sh_degree, d_source, self.colors[start:end]) + .5, 0))
                colors_cat = torch.cat(colors, dim=0)
            else:
                for model in self.models:
                    inverse = torch.as_tensor(model.world_to_source, device=self.device)
                    source_camera = inverse[:3, :3] @ camera + inverse[:3, 3]
                    direction = model.splats['means'] - source_camera
                    colors.append(torch.clamp_min(spherical_harmonics(model.sh_degree, direction, model.splats['colors']) + .5, 0))
                colors_cat = torch.cat(colors)
            if not hasattr(self, '_cached_k'):
                self._cached_k = {}
                self._cached_bg = None
            key = (width, height, hfov_deg)
            if key not in self._cached_k:
                self._cached_k[key] = torch.as_tensor(intrinsics(width, height, hfov_deg)[None],
                                                      dtype=torch.float32, device=self.device)
            k = self._cached_k[key]
            if self._cached_bg is None:
                self._cached_bg = torch.ones((1, 3), device=self.device)
            backgrounds = self._cached_bg
            r = pose[:3, :3]
            t = pose[:3, 3]
            view_np = np.empty((1, 4, 4), dtype=np.float32)
            view_np[0, :3, :3] = r.T
            view_np[0, :3, 3] = -r.T @ t
            view_np[0, 3, :3] = 0.0
            view_np[0, 3, 3] = 1.0
            view = torch.as_tensor(view_np, device=self.device)
            image, _, _ = rasterization(**self.world, colors=colors_cat, viewmats=view, Ks=k,
                                         width=width, height=height, sh_degree=None, tile_size=16,
                                         packed=False, render_mode='RGB', backgrounds=backgrounds, near_plane=.01)
            return (image[0].clamp(0, 1) * 255).round().byte().cpu().numpy()

