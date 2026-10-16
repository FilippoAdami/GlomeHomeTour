"""Focused regression tests for manifest-driven floor/ceiling plane fitting."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest


MODULE = Path(__file__).parents[1] / "06_mesh_elaboration" / "fit_floors_ceilings_planes.py"
SPEC = importlib.util.spec_from_file_location("surface_fit", MODULE)
surface_fit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(surface_fit)


def _plane_mesh(path: Path, normal: np.ndarray, offset: float, *, extra: bool = False) -> None:
    normal = normal / np.linalg.norm(normal)
    tangent = np.cross(normal, [0.0, 0.0, 1.0])
    if np.linalg.norm(tangent) < 0.1:
        tangent = np.cross(normal, [1.0, 0.0, 0.0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    centre = -offset * normal
    vertices = np.array([centre + a * tangent + b * bitangent for b in (-1, 0, 1) for a in (-1, 0, 1)], dtype=float)
    triangles = []
    for row in range(2):
        for column in range(2):
            first = row * 3 + column
            triangles.extend([[first, first + 1, first + 4], [first, first + 4, first + 3]])
    if extra:
        # A perpendicular protrusion with enough area to catch non-normal-aware fitting.
        vertices = np.vstack([vertices, [2, 0, 0], [2, 1, 0], [2, 0, 1]])
        triangles.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([0.2, 0.4, 0.6], (len(vertices), 1)))
    o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=True)


def test_fit_is_translation_invariant_and_rejects_perpendicular_clutter() -> None:
    normal = np.array([0.25, 0.95, -0.18])
    normal /= np.linalg.norm(normal)
    tangent = np.cross(normal, [0.0, 0.0, 1.0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    origin = np.array([12.5, -8.0, 31.0])
    points = np.array([origin, origin + tangent, origin + bitangent, origin + tangent + bitangent])
    normals = np.tile(normal / np.linalg.norm(normal), (len(points), 1))
    normals[-1] = [1, 0, 0]
    fitted, d, ratio = surface_fit.fit_robust_plane(points, np.ones(4), 0.02, True, normals)
    assert fitted[1] > 0
    assert ratio >= 0.74
    assert np.max(np.abs(points[:3] @ fitted + d)) < 0.02


def test_shallow_ramp_is_not_flattened() -> None:
    ramp = np.array([0.1, 0.995, 0.0])
    ramp /= np.linalg.norm(ramp)
    points = np.array([[0, 0, 0], [1, -ramp[0] / ramp[1], 0], [0, 0, 1],
                       [1, -ramp[0] / ramp[1], 1]], dtype=float)
    assert not surface_fit._should_flatten(points, np.ones(4), ramp, 0.0, 0.08)


def test_beamed_sloped_ceiling_recovers_substrate_normal_and_depth() -> None:
    normal = np.array([0.41, -0.91, 0.06])
    normal /= np.linalg.norm(normal)
    tangent = np.cross(normal, [0.0, 0.0, 1.0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    substrate = np.array([i * tangent + j * bitangent for i in (-2, 0, 2) for j in range(-5, 6)])
    # Beam tops are parallel but displaced by 9cm; their flanks are omitted by
    # the normal-compatible fit and must not rotate the supporting plane.
    beams = np.array([i * tangent + j * bitangent + 0.09 * normal
                      for i in (-4, -2, 0, 2, 4) for j in (-1, 1)])
    points = np.vstack([substrate, beams])
    normals = np.tile(normal, (len(points), 1))
    fitted, d, ratio = surface_fit.fit_robust_plane(points, np.ones(len(points)), 0.04, False, normals)
    assert np.rad2deg(np.arccos(np.clip(fitted @ normal, -1.0, 1.0))) < 1e-6
    assert abs(d) < 1e-9
    assert ratio == pytest.approx(len(substrate) / len(points))


def test_manifest_processes_arbitrary_counts_and_missing_mesh_fails(tmp_path: Path) -> None:
    entries = []
    for index in range(5):
        name = f"{'floor' if index < 3 else 'ceiling'}_inst{index:03d}"
        filename = f"{name}.ply"
        _plane_mesh(tmp_path / filename, np.array([0.0, 1.0, 0.0]), float(index), extra=index == 0)
        entries.append({"instance_name": name, "role": "floor" if index < 3 else "ceiling", "file_name": filename})
    (tmp_path / "structural_instances_manifest.json").write_text(json.dumps({"instances": entries}))
    completed = subprocess.run([sys.executable, str(MODULE), "--instances-dir", str(tmp_path)],
                               capture_output=True, text=True, check=True)
    assert "floor_inst000" in completed.stdout
    result = json.loads((tmp_path / "floors_ceilings_planes_manifest.json").read_text())
    assert [item["name"] for item in result["elements"]] == [entry["instance_name"] for entry in entries]
    entries[0]["file_name"] = "gone.ply"
    (tmp_path / "structural_instances_manifest.json").write_text(json.dumps({"instances": entries}))
    failed = subprocess.run([sys.executable, str(MODULE), "--instances-dir", str(tmp_path)],
                            capture_output=True, text=True)
    assert failed.returncode == 2
    assert "missing floor mesh" in failed.stderr
