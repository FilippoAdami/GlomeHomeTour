#!/usr/bin/env python3
"""Inspect paired ARCore/COLMAP cameras and RGB images against the sparse cloud."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from convert_transforms_to_colmap import OPENGL_TO_OPENCV
from scene.colmap_loader import (
    qvec2rotmat, read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text,
)

COLORS = {"arcore": np.array([0.15, 0.50, 1.0]),
          "colmap": np.array([0.77, 0.35, 1.0])}
FRUSTUM_LINES = np.array([[0, 1], [0, 2], [0, 3], [0, 4],
                          [1, 2], [2, 3], [3, 4], [4, 1]])


@dataclass
class Pose:
    """Camera-to-world pose with OpenCV camera axes: right, down, forward."""

    c2w: np.ndarray
    K: np.ndarray
    width: int
    height: int
    distortion: np.ndarray

    def __post_init__(self):
        if (self.c2w.shape != (4, 4) or not np.isfinite(self.c2w).all()
                or not np.allclose(self.c2w[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(self.c2w[:3, :3].T @ self.c2w[:3, :3], np.eye(3), atol=0.01)
                or not np.isclose(np.linalg.det(self.c2w[:3, :3]), 1, atol=0.01)):
            raise ValueError("Camera pose must be a finite, rigid camera-to-world matrix")
        if (self.width <= 0 or self.height <= 0 or self.K.shape != (3, 3)
                or not np.isfinite(self.K).all() or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.isfinite(self.distortion).all()):
            raise ValueError("Camera dimensions, intrinsics, or distortion are invalid")


@dataclass
class Frame:
    name: str
    image_path: Path
    poses: dict[str, Pose]
    point_ids: frozenset[int] = frozenset()


def camera_calibration(model, width, height, params, c2w):
    """Decode the perspective COLMAP camera models supported by OpenCV."""
    p = np.asarray(params, dtype=np.float64)
    distortion = np.zeros(4)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1:3]
        distortion[:len(p) - 3] = p[3:]
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[:4]
        if len(p) > 4:
            distortion = p[4:].copy()
    else:
        raise ValueError(f"Unsupported camera model {model}; use a perspective/undistorted model")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return Pose(c2w, K, int(width), int(height), distortion)


def select_scene(workspace: Path, scan: str | None = None) -> Path:
    workspace = workspace.resolve()
    if scan:
        if Path(scan).name != scan:
            raise ValueError("--scan must name a direct scan folder, such as scan_0")
        workspace /= scan
    if (workspace / "sparse" / "0").is_dir():
        return workspace
    scans = sorted((p for p in workspace.glob("scan_*")
                    if re.fullmatch(r"scan_\d+", p.name) and (p / "sparse" / "0").is_dir()),
                   key=lambda p: int(p.name[5:]))
    if not scans:
        raise FileNotFoundError(f"No sparse/0 model in {workspace} or its scan_N folders")
    print(f"Opening {scans[0].name}; use --scan scan_N to choose another scan")
    return scans[0]


def load_frames(scene: Path) -> list[Frame]:
    """Pair poses by unique image basename, preserving each source's intrinsics."""
    result_path = scene / "ingestion_results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}

    def input_path(value):
        path = (scene / value).resolve()
        if not path.is_relative_to(scene.resolve()):
            raise ValueError(f"Input path escapes the scan: {value}")
        return path

    image_dir = input_path(result.get("images_path", "images"))
    files = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            if path.name in files:
                raise ValueError(f"Ambiguous image basename {path.name} in {image_dir}")
            files[path.name] = path

    sparse = scene / "sparse" / "0"
    if (sparse / "images.bin").exists() and (sparse / "cameras.bin").exists():
        cameras = read_intrinsics_binary(sparse / "cameras.bin")
        images = read_extrinsics_binary(sparse / "images.bin")
    else:
        cameras = read_intrinsics_text(sparse / "cameras.txt")
        images = read_extrinsics_text(sparse / "images.txt")
    poses, point_ids = {}, {}
    for image in images.values():
        name = Path(image.name).name
        if name in poses:
            raise ValueError(f"Ambiguous COLMAP image basename {name}")
        rotation = qvec2rotmat(image.qvec).T
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = rotation, -rotation @ image.tvec
        camera = cameras[image.camera_id]
        poses[name] = {"colmap": camera_calibration(
            camera.model, camera.width, camera.height, camera.params, c2w)}
        point_ids[name] = frozenset(int(point_id) for point_id in image.point3D_ids if point_id >= 0)

    transforms_path = result.get("transforms_path", "transforms.json")
    if result.get("poses_format") == "arcore" and transforms_path and input_path(transforms_path).is_file():
        data = json.loads(input_path(transforms_path).read_text())
        seen = set()
        for frame in data["frames"]:
            name = Path(frame["file_path"]).name
            if name in seen:
                raise ValueError(f"Ambiguous ARCore frame basename {name}")
            seen.add(name)
            metadata = {**data, **frame}
            params = [metadata[k] for k in ("fl_x", "fl_y", "cx", "cy")]
            params += [metadata.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")]
            # The global alignment has already been applied to both pose sets by phase 01.
            c2w = np.asarray(frame["transform_matrix"], dtype=float) @ OPENGL_TO_OPENCV
            poses.setdefault(name, {})["arcore"] = camera_calibration(
                "OPENCV", metadata["w"], metadata["h"], params, c2w)

    missing = set(poses) - set(files)
    if missing:
        print(f"Skipping {len(missing)} pose records whose RGB images are missing")
    natural_key = lambda name: [int(part) if part.isdigit() else part
                                for part in re.split(r"(\d+)", name)]
    frames = [Frame(name, files[name], poses[name], point_ids.get(name, frozenset()))
              for name in sorted(set(poses) & set(files), key=natural_key)]
    if not frames:
        raise ValueError(f"No registered poses with corresponding images in {scene}")
    return frames


def load_cloud(scene: Path):
    sparse = scene / "sparse" / "0"
    points_path = sparse / "points3D.txt"
    if not points_path.is_file():
        raise FileNotFoundError(f"The pose inspector needs the COLMAP text model: {points_path}")
    point_ids, xyz, rgb = [], [], []
    for line in points_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        point_ids.append(int(fields[0]))
        xyz.append([float(value) for value in fields[1:4]])
        rgb.append([int(value) for value in fields[4:7]])
    if not point_ids:
        raise ValueError(f"The COLMAP point model is empty: {points_path}")
    xyz, rgb = np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.float64) / 255.0
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[finite], rgb[finite]
    point_ids = [point_id for point_id, keep in zip(point_ids, finite) if keep]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    if not len(cloud.points):
        raise ValueError("The point cloud is empty")
    return cloud, {point_id: index for index, point_id in enumerate(point_ids)}


def frustum_mask(xyz: np.ndarray, pose: Pose) -> np.ndarray:
    """Project world points into this camera; this is not an occlusion/visibility test."""
    camera_xyz = (xyz - pose.c2w[:3, 3]) @ pose.c2w[:3, :3]
    indices = np.flatnonzero(camera_xyz[:, 2] > 0.02)
    mask = np.zeros(len(xyz), dtype=bool)
    if not len(indices):
        return mask
    p = camera_xyz[indices]
    if np.any(pose.distortion):
        uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), pose.K, pose.distortion)
        uv = uv.reshape(-1, 2)
    else:
        uv = (p @ pose.K.T)[:, :2] / p[:, 2, None]
    mask[indices] = ((uv[:, 0] >= 0) & (uv[:, 0] < pose.width)
                     & (uv[:, 1] >= 0) & (uv[:, 1] < pose.height))
    return mask


def camera_wireframe(pose: Pose, depth: float):
    """Blender-style camera outline: pyramid, image rectangle, and top triangle."""
    pixels = np.array([[0, 0], [pose.width, 0], [pose.width, pose.height], [0, pose.height]],
                      dtype=np.float64)
    corners = cv2.undistortPoints(pixels[:, None], pose.K, pose.distortion).reshape(-1, 2)
    corners = np.column_stack((corners, np.ones(4))) * depth
    top_left, top_right = corners[:2]
    base_left = 0.75 * top_left + 0.25 * top_right
    base_right = 0.25 * top_left + 0.75 * top_right
    apex = (top_left + top_right) / 2
    apex[1] -= np.linalg.norm(top_right - top_left) * 0.35
    local = np.vstack((np.zeros(3), corners, base_left, base_right, apex))
    lines = np.vstack((FRUSTUM_LINES, [[5, 6], [6, 7], [7, 5]]))
    return local @ pose.c2w[:3, :3].T + pose.c2w[:3, 3], lines


def pose_delta(frame: Frame):
    if not {"arcore", "colmap"} <= frame.poses.keys():
        return None
    a, b = (frame.poses[source].c2w for source in ("arcore", "colmap"))
    distance = np.linalg.norm(a[:3, 3] - b[:3, 3])
    angle = np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))
    return distance, angle


def line_set(points, lines, color):
    geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points),
                                   o3d.utility.Vector2iVector(lines))
    geometry.paint_uniform_color(color)
    return geometry


class PoseAlignmentViewer:
    def __init__(self, scene: Path, frames: list[Frame], cloud, point_index):
        self.frames = frames
        self.xyz, self.rgb = np.asarray(cloud.points), np.asarray(cloud.colors)
        self.added_by_frame = []
        seen_point_ids = set()
        for frame in frames:
            # A final COLMAP model has no creation timestamp; use its second chronological
            # observation as the point's incremental triangulation frame.
            newly_triangulated = sorted(frame.point_ids & seen_point_ids)
            self.added_by_frame.append(np.array(
                [point_index[point_id] for point_id in newly_triangulated
                 if point_id in point_index], dtype=np.int64))
            seen_point_ids.update(frame.point_ids)
        self.index, self.source, self.camera_view = 0, "colmap", False
        self.image_size = (1, 1)
        centers = [pose.c2w[:3, 3] for frame in frames for pose in frame.poses.values()]
        extent_points = np.vstack((self.xyz.min(0), self.xyz.max(0), centers))
        self.bounds = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(extent_points))
        self.scale = max(np.linalg.norm(self.bounds.get_extent()) * 0.008, 0.03)
        self.window = gui.Application.instance.create_window(f"Pose alignment — {scene.name}", 1500, 920)
        self.view = gui.SceneWidget()
        self.view.scene = rendering.Open3DScene(self.window.renderer)
        self.view.scene.set_background([0.025, 0.03, 0.045, 1])
        self.view.scene.show_axes(True)
        self.image = gui.ImageWidget()
        self.toolbar = gui.Horiz(8)
        for label, action in [("← Previous", lambda: self.navigate(-1)),
                              ("Next →", lambda: self.navigate(1)),
                              ("ARCore ↓ (blue)", lambda: self.select_source("arcore")),
                              ("COLMAP ↑ (purple)", lambda: self.select_source("colmap")),
                              ("Overview (O)", lambda: self.set_view(False)),
                              ("Camera view (V)", lambda: self.set_view(True))]:
            button = gui.Button(label)
            button.set_on_clicked(action)
            self.toolbar.add_child(button)
        self.status = gui.Label("")
        self.help = gui.Label("← / → frame    ↑ COLMAP / ↓ ARCore    O overview / V camera view    "
                              "Mouse: orbit, pan, zoom    Points appear on their second observed frame")
        for widget in (self.toolbar, self.status, self.image, self.view, self.help):
            self.window.add_child(widget)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_key(self.on_key)

        self.point_material = rendering.MaterialRecord()
        self.point_material.shader, self.point_material.point_size = "defaultUnlit", 3.0
        self.line_material = rendering.MaterialRecord()
        self.line_material.shader, self.line_material.line_width = "unlitLine", 1.5
        self.selected_material = rendering.MaterialRecord()
        self.selected_material.shader, self.selected_material.line_width = "unlitLine", 4.0

        for source, color in COLORS.items():
            poses = [frame.poses[source] for frame in frames if source in frame.poses]
            if not poses:
                continue
            points, lines = [], []
            for pose in poses:
                p, edges = camera_wireframe(pose, self.scale)
                lines.extend(edges + len(points))
                points.extend(p)
            self.view.scene.add_geometry(source + "_cameras", line_set(points, lines, color * 0.08),
                                         self.line_material)
            centers = np.array([pose.c2w[:3, 3] for pose in poses])
            if len(centers) > 1:
                edges = np.column_stack((np.arange(len(centers) - 1), np.arange(1, len(centers))))
                self.view.scene.add_geometry(source + "_path", line_set(centers, edges, color * 0.15),
                                             self.line_material)
        self.refresh()
        self.set_view(False)

    def on_layout(self, _context):
        r = self.window.content_rect
        unit = self.window.theme.font_size
        self.toolbar.frame = gui.Rect(r.x + 8, r.y + 4, r.width - 16, 2 * unit)
        self.status.frame = gui.Rect(r.x + 8, r.y + 2 * unit + 12, r.width - 16, 3 * unit)
        self.help.frame = gui.Rect(r.x + 8, r.get_bottom() - 2 * unit, r.width - 16, 2 * unit)
        top, height = r.y + 5 * unit + 16, r.height - 7 * unit - 24
        left_width = int(r.width * 0.37)

        def fit(x, width, size):
            ratio = min(width / size[0], height / size[1])
            w, h = int(size[0] * ratio), int(size[1] * ratio)
            return gui.Rect(x + (width - w) // 2, top + (height - h) // 2, w, h)

        self.image.frame = fit(r.x + 8, left_width - 16, self.image_size)
        right_x, right_width = r.x + left_width, r.width - left_width - 8
        pose = self.frames[self.index].poses.get(self.source)
        self.view.frame = (fit(right_x, right_width, (pose.width, pose.height))
                           if self.camera_view and pose else gui.Rect(right_x, top, right_width, height))

    def on_key(self, event):
        actions = {gui.KeyName.LEFT: lambda: self.navigate(-1),
                   gui.KeyName.RIGHT: lambda: self.navigate(1),
                   gui.KeyName.UP: lambda: self.select_source("colmap"),
                   gui.KeyName.DOWN: lambda: self.select_source("arcore"),
                   gui.KeyName.O: lambda: self.set_view(False),
                   gui.KeyName.V: lambda: self.set_view(True)}
        if event.key not in actions:
            return False
        if event.type == gui.KeyEvent.DOWN:
            actions[event.key]()
        return True

    def navigate(self, delta):
        index = min(max(self.index + delta, 0), len(self.frames) - 1)
        if index != self.index:
            self.index = index
            self.refresh()

    def select_source(self, source):
        self.source = source
        self.refresh()

    def set_view(self, camera_view):
        self.camera_view = camera_view
        pose = self.frames[self.index].poses.get(self.source)
        background_lines = rendering.MaterialRecord()
        background_lines.shader, background_lines.line_width = "unlitLine", 1.5
        shade = 0.15 if camera_view else 1.0
        background_lines.base_color = [shade, shade, shade, 1]
        for source in COLORS:
            for suffix in ("_cameras", "_path"):
                name = source + suffix
                if self.view.scene.has_geometry(name):
                    self.view.scene.modify_geometry_material(name, background_lines)
        if camera_view and pose:
            self.view.setup_camera(pose.K, np.linalg.inv(pose.c2w), pose.width, pose.height, self.bounds)
            # Keep close surfaces while framing the same sensor aspect ratio as the RGB panel.
            far = max(np.linalg.norm(self.bounds.get_extent()) * 3, 10)
            self.view.scene.camera.set_projection(pose.K, 0.02, far, pose.width, pose.height)
        else:
            self.view.setup_camera(60, self.bounds, self.bounds.get_center())
        self.window.set_needs_layout()
        self.window.post_redraw()
        self.view.force_redraw()

    def refresh(self):
        frame = self.frames[self.index]
        image = o3d.io.read_image(str(frame.image_path))
        if image.is_empty():
            raise ValueError(f"Cannot read image {frame.image_path}")
        pixels = np.asarray(image)
        self.image_size = (pixels.shape[1], pixels.shape[0])
        self.image.update_image(image)
        for name in ("progress_cloud", "selected_camera", "selected_frustum", "pose_difference"):
            if self.view.scene.has_geometry(name):
                self.view.scene.remove_geometry(name)
        pose = frame.poses.get(self.source)
        visible_indices = np.concatenate(self.added_by_frame[:self.index + 1])
        added_count = len(self.added_by_frame[self.index])
        if len(visible_indices):
            progress = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz[visible_indices]))
            progress.colors = o3d.utility.Vector3dVector(
                np.clip(self.rgb[visible_indices] * 0.85 + 0.12, 0, 1))
            self.view.scene.add_geometry("progress_cloud", progress, self.point_material)
        if pose:
            color = COLORS[self.source]
            p, edges = camera_wireframe(pose, self.scale)
            self.view.scene.add_geometry("selected_camera", line_set(p, edges, color), self.selected_material)
            depth = max(float(((self.xyz - pose.c2w[:3, 3]) @ pose.c2w[:3, 2]).max()), self.scale)
            p, _ = camera_wireframe(pose, depth)
            self.view.scene.add_geometry("selected_frustum", line_set(p[:5], FRUSTUM_LINES, color * 0.55),
                                         self.line_material)
        delta = pose_delta(frame)
        delta_text = "ARCore/COLMAP pair unavailable"
        if delta:
            delta_text = f"Pose difference: {delta[0] * 100:.1f} cm / {delta[1]:.2f}°"
            centers = [frame.poses[source].c2w[:3, 3] for source in COLORS]
            self.view.scene.add_geometry("pose_difference", line_set(centers, [[0, 1]], [1, 0.75, 0.2]),
                                         self.selected_material)
        self.status.text = (f"{self.index + 1} / {len(self.frames)}   {frame.name}   |   {self.source.upper()}"
                            f"{' — pose unavailable' if pose is None else ''}\n"
                            f"{delta_text}   |   {added_count:,} points added by this frame; "
                            f"{len(visible_indices):,} / {len(self.xyz):,} accumulated")
        if self.camera_view:
            self.set_view(True)
        self.window.set_needs_layout()
        self.window.post_redraw()
        self.view.force_redraw()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=_backend / "current_scene",
                        help="Session root or individual scan folder")
    parser.add_argument("--scan", help="Choose a scan_N folder within the session")
    args = parser.parse_args(argv)
    scene = select_scene(args.workspace, args.scan)
    frames = load_frames(scene)
    cloud, point_index = load_cloud(scene)
    print(f"{scene}: {len(frames)} frames, {len(cloud.points):,} cloud points")
    # Match the repository's XWayland workaround for Open3D's native Wayland crash.
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
        os.environ.pop("WAYLAND_DISPLAY")
        os.environ["EGL_PLATFORM"] = "x11"
    gui.Application.instance.initialize()
    viewer = PoseAlignmentViewer(scene, frames, cloud, point_index)
    gui.Application.instance.run()
    return 0


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # shortcut: Open3D 0.19 hangs at native teardown on this desktop; remove after a renderer upgrade.
    os._exit(status)
