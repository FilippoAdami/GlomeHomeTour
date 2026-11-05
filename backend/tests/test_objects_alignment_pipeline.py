"""Integration tests for Stage 07 — Objects Alignment pipeline."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pytest

STAGE07 = Path(__file__).resolve().parents[1] / "07_alignment"
sys.path.insert(0, str(STAGE07))
spec = importlib.util.spec_from_file_location("step_objects_alignment", STAGE07 / "step_objects_alignment.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
from fit_contact_groups_inside import containment_candidate_allowed


def test_support_candidate_allows_at_most_two_percent_outside_per_instance():
    assert containment_candidate_allowed([98.0, 100.0], has_support=True)
    assert not containment_candidate_allowed([97.99, 100.0], has_support=True)
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
    assert fit_report["minimum_instance_contained_surface_percent"] == 98.0
    assert (output_dir / "structural_fit" / "structural_fit_scene.ply").is_file()

    assert not (output_dir / "plane_snap_report.json").exists()

    final_dir = output_dir / "aligned_instances"
    manifest = json.loads((final_dir / "manifest.json").read_text())
    assert manifest["status"] == "aligned_non_structural_instances"
    assert all(item["contained_surface_percent"] >= 98.0
               for item in manifest["instances"] if np.linalg.norm(item["placement_translation_m"]) > 1e-6)
    assert {item["role"] for item in manifest["instances"]} == {"object"}
    assert {item["category"] for item in manifest["instances"]} == {"table", "television", None}
    assert not (final_dir / "class_101" / "instance_000.ply").exists()
    assert len(list(final_dir.rglob("instance_*.ply"))) == 3

    table = o3d.io.read_triangle_mesh(str(final_dir / "class_100" / "instance_000.ply"))
    tv = o3d.io.read_triangle_mesh(str(final_dir / "class_102" / "instance_000.ply"))
    # Contact blocks stop at the support fitter's 0.005 m gap; no per-instance snap follows.
    np.testing.assert_allclose(table.get_min_bound()[1], 0.005, atol=0.002)
    np.testing.assert_allclose(tv.get_min_bound()[0], 0.005, atol=0.002)
    floating = o3d.io.read_triangle_mesh(str(final_dir / "class_103" / "instance_000.ply"))
    np.testing.assert_allclose(floating.get_min_bound()[1], 1.0, atol=1e-6)
    tv_report = next(item for item in fit_report["groups"] if "television" in item["categories"])
    assert tv_report["surface_intersection_edge_hits_before"] > 0
    assert tv_report["surface_intersection_edge_hits_after"] == 0
    assert all(item["surface_intersection_edge_hits_after"] == 0 for item in fit_report["groups"])


def test_snap_instance_to_wall_within_10cm(tmp_path):
    """Verify an object placed 6 cm from a wall snaps flush along cardinal axis."""
    from snap_instances_to_planes import run_snap_instances_to_planes

    # Create box room [0..4] x [0..3] x [0..4]
    shell_mesh = o3d.geometry.TriangleMesh()
    shell_mesh += o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.1, depth=4.0)
    shell_path = tmp_path / "complete.ply"
    o3d.io.write_triangle_mesh(str(shell_path), shell_mesh)

    elements_dir = tmp_path / "elements"
    elements_dir.mkdir()
    floor_elem = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.02, depth=4.0)
    o3d.io.write_triangle_mesh(str(elements_dir / "floor_level_00.ply"), floor_elem)
    ceiling_elem = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.02, depth=4.0)
    ceiling_elem.translate([0.0, 3.0, 0.0])
    o3d.io.write_triangle_mesh(str(elements_dir / "ceiling_group_00.ply"), ceiling_elem)
    wall_elem = o3d.geometry.TriangleMesh.create_box(width=0.02, height=3.0, depth=4.0)
    o3d.io.write_triangle_mesh(str(elements_dir / "walls.ply"), wall_elem)

    # Instance placed 6 cm away from wall face (wall face at X=0.02, instance at X=0.08)
    align_dir = tmp_path / "objects_alignment"
    inst_dir = align_dir / "aligned_instances" / "class_050"
    inst_dir.mkdir(parents=True)
    box_m = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)
    box_m.translate([0.08, 0.5, 1.0])
    o3d.io.write_triangle_mesh(str(inst_dir / "instance_000.ply"), box_m)

    manifest = {
        "status": "aligned_non_structural_instances",
        "source_shell": str(shell_path),
        "instances": [{
            "class_id": 50,
            "role": "object",
            "category": "table",
            "path": "class_050/instance_000.ply",
            "placement_translation_m": [0.0, 0.0, 0.0],
            "contained_surface_percent": 100.0,
        }],
    }
    (align_dir / "aligned_instances" / "manifest.json").write_text(json.dumps(manifest, indent=2))

    res = run_snap_instances_to_planes(
        alignment_dir=align_dir,
        shell_path=shell_path,
        max_distance_m=0.10,
        min_containment_percent=97.0,
    )
    assert res["snapped_instances_count"] == 1

    updated_m = o3d.io.read_triangle_mesh(str(inst_dir / "instance_000.ply"))
    # Min X should now be adjacent to wall inner face at X=0.02 (with target gap ~0.002)
    np.testing.assert_allclose(updated_m.get_min_bound()[0], 0.02 + 0.002, atol=0.002)
    assert res["instances"][0]["final_contained_surface_percent"] >= 97.0


def test_snap_instance_farther_than_10cm_ignored(tmp_path):
    """Verify an object placed 15 cm from a wall is NOT snapped when max_distance is 10 cm."""
    from snap_instances_to_planes import run_snap_instances_to_planes

    shell_mesh = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.1, depth=4.0)
    shell_path = tmp_path / "complete.ply"
    o3d.io.write_triangle_mesh(str(shell_path), shell_mesh)

    elements_dir = tmp_path / "elements"
    elements_dir.mkdir()
    wall_elem = o3d.geometry.TriangleMesh.create_box(width=0.02, height=3.0, depth=4.0)
    o3d.io.write_triangle_mesh(str(elements_dir / "walls.ply"), wall_elem)
    floor_elem = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.02, depth=4.0)
    o3d.io.write_triangle_mesh(str(elements_dir / "floor_level_00.ply"), floor_elem)
    ceiling_elem = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.02, depth=4.0)
    ceiling_elem.translate([0.0, 3.0, 0.0])
    o3d.io.write_triangle_mesh(str(elements_dir / "ceiling_group_00.ply"), ceiling_elem)

    # Placed 15 cm away from wall face (wall face at X=0.02, instance at X=0.17)
    align_dir = tmp_path / "objects_alignment"
    inst_dir = align_dir / "aligned_instances" / "class_050"
    inst_dir.mkdir(parents=True)
    box_m = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)
    box_m.translate([0.17, 0.5, 1.0])
    o3d.io.write_triangle_mesh(str(inst_dir / "instance_000.ply"), box_m)

    manifest = {
        "status": "aligned_non_structural_instances",
        "source_shell": str(shell_path),
        "instances": [{
            "class_id": 50,
            "role": "object",
            "category": "table",
            "path": "class_050/instance_000.ply",
            "placement_translation_m": [0.0, 0.0, 0.0],
            "contained_surface_percent": 100.0,
        }],
    }
    (align_dir / "aligned_instances" / "manifest.json").write_text(json.dumps(manifest, indent=2))

    res = run_snap_instances_to_planes(
        alignment_dir=align_dir,
        shell_path=shell_path,
        max_distance_m=0.10,
        min_containment_percent=97.0,
    )
    assert res["snapped_instances_count"] == 0
    updated_m = o3d.io.read_triangle_mesh(str(inst_dir / "instance_000.ply"))
    np.testing.assert_allclose(updated_m.get_min_bound()[0], 0.17, atol=1e-5)


def test_cardinal_translation_only_no_rotation(tmp_path):
    """Verify that snap_instances_to_planes applies pure translations with 0 rotation."""
    from snap_instances_to_planes import run_snap_instances_to_planes

    shell_mesh = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.1, depth=4.0)
    shell_path = tmp_path / "complete.ply"
    o3d.io.write_triangle_mesh(str(shell_path), shell_mesh)

    elements_dir = tmp_path / "elements"
    elements_dir.mkdir()
    wall_elem = o3d.geometry.TriangleMesh.create_box(width=0.02, height=3.0, depth=4.0)
    o3d.io.write_triangle_mesh(str(elements_dir / "walls.ply"), wall_elem)
    floor_elem = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.02, depth=4.0)
    o3d.io.write_triangle_mesh(str(elements_dir / "floor_level_00.ply"), floor_elem)
    ceiling_elem = o3d.geometry.TriangleMesh.create_box(width=4.0, height=0.02, depth=4.0)
    ceiling_elem.translate([0.0, 3.0, 0.0])
    o3d.io.write_triangle_mesh(str(elements_dir / "ceiling_group_00.ply"), ceiling_elem)

    # Place an asymmetric wedge/box 8 cm from wall
    align_dir = tmp_path / "objects_alignment"
    inst_dir = align_dir / "aligned_instances" / "class_050"
    inst_dir.mkdir(parents=True)
    box_m = o3d.geometry.TriangleMesh.create_box(width=0.4, height=0.6, depth=0.8)
    box_m.translate([0.10, 0.5, 1.0])
    orig_vertices = np.asarray(box_m.vertices).copy()
    o3d.io.write_triangle_mesh(str(inst_dir / "instance_000.ply"), box_m)

    manifest = {
        "status": "aligned_non_structural_instances",
        "source_shell": str(shell_path),
        "instances": [{
            "class_id": 50,
            "role": "object",
            "category": "table",
            "path": "class_050/instance_000.ply",
            "placement_translation_m": [0.0, 0.0, 0.0],
            "contained_surface_percent": 100.0,
        }],
    }
    (align_dir / "aligned_instances" / "manifest.json").write_text(json.dumps(manifest, indent=2))

    res = run_snap_instances_to_planes(alignment_dir=align_dir, shell_path=shell_path, max_distance_m=0.15)
    assert res["snapped_instances_count"] == 1

    updated_m = o3d.io.read_triangle_mesh(str(inst_dir / "instance_000.ply"))
    new_vertices = np.asarray(updated_m.vertices)
    # The shift vector between all corresponding vertices must be identical (pure translation, R = I)
    shifts = new_vertices - orig_vertices
    np.testing.assert_allclose(shifts - shifts[0], 0.0, atol=1e-6)
    # Rotation is strictly identity
    assert np.allclose(shifts[:, 1:], 0.0, atol=1e-5)  # Y and Z shifts were 0, pure X translation
