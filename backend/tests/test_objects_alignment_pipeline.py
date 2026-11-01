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
from fit_contact_groups_inside import containment_candidate_allowed


def test_support_candidate_allows_five_percent_outside_per_instance():
    assert containment_candidate_allowed([95.0, 100.0], has_support=True)
    assert not containment_candidate_allowed([94.99, 100.0], has_support=True)
    assert containment_candidate_allowed([94.0, 80.0], has_support=False)


def test_objects_alignment_end_to_end(tmp_path, monkeypatch):
    """Verify Stage 07 alignment, contact grouping, and support placement."""
    # 1. Stage 06 completed structural mesh
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
    # Ceiling (some Phase 06 exports wind ceiling faces upward too)
    quad([[0, 3, 0], [4, 3, 0], [4, 3, 4], [0, 3, 4]], upward=True)
    # 4 walls
    corners = [[0, 0], [4, 0], [4, 4], [0, 4]]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        quad([[a[0], 0, a[1]], [b[0], 0, b[1]], [b[0], 3, b[1]], [a[0], 3, a[1]]])

    shell_mesh = o3d.geometry.TriangleMesh()
    shell_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    shell_mesh.triangles = o3d.utility.Vector3iVector(faces)
    shell_mesh.compute_vertex_normals()
    shell_path = tmp_path / "complete.ply"
    assert o3d.io.write_triangle_mesh(str(shell_path), shell_mesh)

    # 2. Stage 05 segmentation output directory
    seg_dir = tmp_path / "segmentation"
    masks_dir = seg_dir / "3d_masks"
    masks_dir.mkdir(parents=True)

    metadata = {
        "classes": {
            "17": {"structural_role": "floor", "confidence": 0.98},
            "23": {"structural_role": "ceiling", "confidence": 0.95},
            "41": {"structural_role": "wall", "confidence": 0.98},
            "100": {"structural_role": "object", "confidence": 0.85},
            "101": {"structural_role": "columns", "confidence": 0.95},
            "102": {"structural_role": "object", "confidence": 0.90},
            "103": {"structural_role": "object", "confidence": 0.80},
        }
    }
    (seg_dir / "class_metadata.json").write_text(json.dumps(metadata))
    similarity_dir = seg_dir / "non_structural_similarity"
    similarity_dir.mkdir()
    (similarity_dir / "valid_class_associations.json").write_text(json.dumps({
        "associations": [[100, 11], [102, 17]]
    }))
    (similarity_dir / "categories.json").write_text(json.dumps([
        {"id": 11, "name": "table"}, {"id": 17, "name": "television"}
    ]))

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

    column_dir = masks_dir / "class_101_columns"
    column_dir.mkdir()
    column_mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=1.0, depth=0.2)
    column_mesh.translate([2.5, 0.0, 2.5])
    o3d.io.write_triangle_mesh(str(column_dir / "instance_000.ply"), column_mesh)

    tv_dir = masks_dir / "class_102_television"
    tv_dir.mkdir()
    tv_mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.7, depth=0.6)
    tv_mesh.translate([-0.02, 1.0, 1.5])
    o3d.io.write_triangle_mesh(str(tv_dir / "instance_000.ply"), tv_mesh)

    floating_dir = masks_dir / "class_103_object"
    floating_dir.mkdir()
    floating_mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
    floating_mesh.translate([2.0, 1.0, 2.0])
    o3d.io.write_triangle_mesh(str(floating_dir / "instance_000.ply"), floating_mesh)

    output_dir = tmp_path / "objects_alignment"

    args = [
        "step_objects_alignment",
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
    ]

    # Verify baseline report and aligned instance meshes
    baseline_report_path = output_dir / "rigid_baseline_report.json"
    assert baseline_report_path.is_file()
    baseline = json.loads(baseline_report_path.read_text())
    assert len(baseline["object_instances"]) == 3
    assert (output_dir / "rigid_baseline_objects.ply").is_file()
    assert (output_dir / "rigid_baseline_scene.ply").is_file()

    # Verify contact grouping and structural support fitting
    assert (output_dir / "contact_groups.json").is_file()
    fit_report_path = output_dir / "structural_fit" / "structural_fit_report.json"
    assert fit_report_path.is_file()
    fit_report = json.loads(fit_report_path.read_text())
    assert fit_report["group_count"] == 3
    assert fit_report["transformed_groups"] == 2
    assert fit_report["minimum_instance_contained_surface_percent"] == 95.0
    assert (output_dir / "structural_fit" / "structural_fit_scene.ply").is_file()
    final_dir = output_dir / "aligned_instances"
    manifest = json.loads((final_dir / "manifest.json").read_text())
    assert all(item["contained_surface_percent"] >= 95.0
               for item in manifest["instances"] if np.linalg.norm(item["placement_translation_m"]) > 1e-6)
    assert {item["role"] for item in manifest["instances"]} == {"object"}
    assert {item["category"] for item in manifest["instances"]} == {"table", "television", None}
    assert not (final_dir / "class_101" / "instance_000.ply").exists()
    assert len(list(final_dir.rglob("instance_*.ply"))) == 3

    table = o3d.io.read_triangle_mesh(str(final_dir / "class_100" / "instance_000.ply"))
    tv = o3d.io.read_triangle_mesh(str(final_dir / "class_102" / "instance_000.ply"))
    np.testing.assert_allclose(table.get_min_bound()[1], 0.005, atol=0.002)
    np.testing.assert_allclose(tv.get_min_bound()[0], 0.005, atol=0.002)
    floating = o3d.io.read_triangle_mesh(str(final_dir / "class_103" / "instance_000.ply"))
    np.testing.assert_allclose(floating.get_min_bound()[1], 1.0, atol=1e-6)
    tv_report = next(item for item in fit_report["groups"] if "television" in item["categories"])
    assert tv_report["surface_intersection_edge_hits_before"] > 0
    assert tv_report["surface_intersection_edge_hits_after"] == 0
    assert all(item["surface_intersection_edge_hits_after"] == 0 for item in fit_report["groups"])
