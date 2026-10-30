"""Integration tests for Stage 07 — Objects Alignment pipeline."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pytest

STAGE07 = Path(__file__).resolve().parents[1] / "07_objecs_alignment"
sys.path.insert(0, str(STAGE07))
spec = importlib.util.spec_from_file_location("step_objects_alignment", STAGE07 / "step_objects_alignment.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_objects_alignment_end_to_end(tmp_path, monkeypatch):
    """Verify Stage 07 executes baseline rigid alignment, contact grouping, containment, and cross-sections."""
    # 1. Authoritative sealed scene mesh (room box 4x3x4)
    scene_mesh = o3d.geometry.TriangleMesh.create_box(width=4.0, height=3.0, depth=4.0)
    scene_mesh.compute_vertex_normals()
    scene_mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh_path = tmp_path / "mesh.ply"
    assert o3d.io.write_triangle_mesh(str(mesh_path), scene_mesh)

    # 2. Stage 06 regularized structural shell (all_structural_regularized.ply)
    vertices, faces = [], []
    def quad(points, upward=None):
        start = len(vertices)
        vertices.extend(points)
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        if upward is not None:
            normal = np.cross(np.array(points[1]) - points[0], np.array(points[2]) - points[0])
            if (normal[1] > 0) != upward:
                triangles = triangles[:, ::-1]
        faces.extend(triangles + start)

    # Floor (y=0, upward)
    quad([[0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4]], upward=True)
    # Ceiling (y=3, downward)
    quad([[0, 3, 0], [4, 3, 0], [4, 3, 4], [0, 3, 4]], upward=False)
    # 4 walls
    corners = [[0, 0], [4, 0], [4, 4], [0, 4]]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        quad([[a[0], 0, a[1]], [b[0], 0, b[1]], [b[0], 3, b[1]], [a[0], 3, a[1]]])

    shell_mesh = o3d.geometry.TriangleMesh()
    shell_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    shell_mesh.triangles = o3d.utility.Vector3iVector(faces)
    shell_mesh.compute_vertex_normals()
    shell_path = tmp_path / "all_structural_regularized.ply"
    assert o3d.io.write_triangle_mesh(str(shell_path), shell_mesh)

    # 3. Stage 05 segmentation output directory
    seg_dir = tmp_path / "segmentation"
    masks_dir = seg_dir / "3d_masks"
    masks_dir.mkdir(parents=True)

    metadata = {
        "classifications": {
            "17": {"structural_role": "floor", "confidence": 0.98},
            "23": {"structural_role": "ceiling", "confidence": 0.95},
            "41": {"structural_role": "wall", "confidence": 0.98},
            "100": {"structural_role": "object", "confidence": 0.85},
        }
    }
    (seg_dir / "class_metadata.json").write_text(json.dumps(metadata))

    # Floor anchor instance (upward normal)
    floor_dir = masks_dir / "class_017_floor"
    floor_dir.mkdir()
    f_mesh = o3d.geometry.TriangleMesh()
    f_mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [4, 0, 0], [4, 0, 4]])
    f_mesh.triangles = o3d.utility.Vector3iVector([[0, 2, 1]])
    f_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(floor_dir / "instance_000.ply"), f_mesh)

    # Ceiling anchor instance (downward normal)
    ceil_dir = masks_dir / "class_023_ceiling"
    ceil_dir.mkdir()
    c_mesh = o3d.geometry.TriangleMesh()
    c_mesh.vertices = o3d.utility.Vector3dVector([[0, 3, 0], [4, 3, 0], [4, 3, 4]])
    c_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    c_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(ceil_dir / "instance_000.ply"), c_mesh)

    # Object instance (table/chair inside room)
    obj_dir = masks_dir / "class_100_table"
    obj_dir.mkdir()
    o_mesh = o3d.geometry.TriangleMesh.create_box(width=0.6, height=0.7, depth=0.6)
    o_mesh.translate([1.5, 0.05, 1.5])
    o3d.io.write_triangle_mesh(str(obj_dir / "instance_000.ply"), o_mesh)

    output_dir = tmp_path / "objects_alignment"

    args = [
        "step_objects_alignment",
        "--mesh-path", str(mesh_path),
        "--shell", str(shell_path),
        "--segmentation-output", str(seg_dir),
        "--output-dir", str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert driver.main() == 0

    # Verify executed stages
    timings = json.loads((output_dir / "step07_timings.json").read_text())
    stages = [item["stage"] for item in timings]
    assert stages == [
        "build_rigid_baseline",
        "fit_contact_groups_inside",
        "extract_cross_section_views",
    ]

    # Verify baseline report and aligned instance meshes
    baseline_report_path = output_dir / "rigid_baseline_report.json"
    assert baseline_report_path.is_file()
    baseline = json.loads(baseline_report_path.read_text())
    assert len(baseline["object_instances"]) == 1
    assert (output_dir / "rigid_baseline_objects.ply").is_file()
    assert (output_dir / "rigid_baseline_scene.ply").is_file()

    # Verify contact grouping and structural fit containment
    assert (output_dir / "contact_groups.json").is_file()
    fit_report_path = output_dir / "structural_fit" / "structural_fit_report.json"
    assert fit_report_path.is_file()
    fit_report = json.loads(fit_report_path.read_text())
    assert fit_report["group_count"] == 1
    assert (output_dir / "structural_fit" / "structural_fit_scene.ply").is_file()

    # Verify cross section views
    cs_dir = output_dir / "cross_sections"
    assert (cs_dir / "cross_sections_manifest.json").is_file()
    assert (cs_dir / "topdown_floor_0.png").is_file()
    assert (cs_dir / "elevation_west_to_east_x_pos.png").is_file()
