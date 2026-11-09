#!/usr/bin/env python3
"""
Unit tests for non-structural class frame extraction in 05_semantic_segmentation.
"""

import json
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_semantic_segmentation"))
from extract_non_structural_frames import (
    is_structural_role,
    identify_non_structural_classes,
    decompose_class_instances,
    pad_crop_to_transparent_square,
    compute_instance_frontal_direction,
    get_standard_elevation_deg,
    cull_foreground_occluders,
    render_synthetic_in_context_frame,
    run_extraction,
    get_camera_up_vector,
    detect_scene_floor_levels,
    assign_instance_to_floor_level,
    identify_instance_biggest_side,
    compute_non_structural_purity,
    _target_mesh_visibility,
    _target_triangle_incidence_quality,
    validate_rendered_frame,
)
from extract_non_structural_frames_gs import load_gaussian_splats


def test_is_structural_role():
    assert is_structural_role("wall") is True
    assert is_structural_role("floor") is True
    assert is_structural_role("ceiling") is True
    assert is_structural_role("stair") is True
    assert is_structural_role("stairs") is True
    assert is_structural_role("wall_sottoscala") is True

    assert is_structural_role("object") is False
    assert is_structural_role("furniture_horizontal") is False
    assert is_structural_role("furniture_vertical") is False
    assert is_structural_role("door") is False
    assert is_structural_role("prop") is False


def test_identify_non_structural_classes(tmp_path):
    struct_json = tmp_path / "structural_classes.json"
    struct_json.write_text(json.dumps({
        "classifications": {
            "1": {"structural_role": "ceiling"},
            "5": {"structural_role": "wall"},
            "14": {"structural_role": "floor"},
            "2": {"structural_role": "furniture_horizontal", "details": "bed"},
            "10": {"structural_role": "object", "details": "lamp"},
        }
    }))

    labels = np.array([1, 5, 14, 2, 10, 99], dtype=np.int32)
    classes = identify_non_structural_classes(struct_json, labels)

    assert set(classes.keys()) == {2, 10, 99}
    assert classes[2]["role"] == "furniture_horizontal"
    assert classes[10]["role"] == "object"
    assert classes[99]["role"] == "object"

    all_classes = identify_non_structural_classes(struct_json, labels, include_structural=True)
    assert set(all_classes) == {1, 2, 5, 10, 14, 99}
    assert all_classes[5]["role"] == "wall"


def test_decompose_class_instances():
    # 2 separate triangles forming 2 separate components
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # Comp 0 (area 0.5)
        [5.0, 0.0, 0.0], [5.0, 0.01, 0.0], [5.01, 0.0, 0.0], # Comp 1 (tiny floater)
    ], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2],
        [3, 4, 5],
    ], dtype=np.int32)
    mask = np.array([True, True], dtype=bool)

    # The 0.3 m² production floor keeps the 0.5 m² component and drops the tiny floater.
    instances = decompose_class_instances(vertices, triangles, mask, min_faces=1, min_area=0.3)
    assert len(instances) == 1
    assert instances[0]["instance_id"] == 0
    assert abs(instances[0]["area_m2"] - 0.5) < 1e-4


def test_pad_crop_to_transparent_square():
    # Rectangular crop 200 (W) x 400 (H)
    crop_arr = np.full((400, 200, 3), 180, dtype=np.uint8)
    square_img = pad_crop_to_transparent_square(crop_arr)

    assert square_img.size == (400, 400)
    assert square_img.mode == "RGBA"

    arr = np.asarray(square_img)
    alpha = arr[:, :, 3]

    # Center 200x400 should be opaque (255)
    # x in [100, 300], y in [0, 400]
    assert np.all(alpha[50:350, 150:250] == 255)

    # Margins (x in [0, 90] or [310, 400]) should be fully transparent (0)
    assert np.all(alpha[:, 0:90] == 0)
    assert np.all(alpha[:, 310:400] == 0)


def test_render_synthetic_in_context_frame():
    # Create simple mesh with target object + structural wall
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.0, 0.2, 0.0],  # object
        [-2.0, -1.0, -1.0], [2.0, -1.0, -1.0], [0.0, 2.0, -1.0], # wall behind
    ])
    mesh.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],
        [3, 4, 5],
    ])
    mesh.vertex_colors = o3d.utility.Vector3dVector([
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
    ])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)
    labels = np.array([10, 5], dtype=np.int32)

    instances = decompose_class_instances(
        vertices, triangles, (labels == 10), min_faces=1, min_area=0.001
    )
    assert len(instances) == 1

    views = render_synthetic_in_context_frame(
        instance=instances[0],
        all_vertices=vertices,
        all_triangles=triangles,
        all_vertex_colors=vertex_colors,
        labels=labels,
        class_id=10,
        structural_cids={5},
        room_center=np.array([0.0, 0.0, 1.0]),
        padding=0.15,
        render_res=512,
        padding_color="transparent",
    )

    assert list(views) == ["base", "up", "down", "left", "right"]
    square_png, meta = views["base"]
    assert square_png.mode == "RGBA"
    w, h = square_png.size
    assert w == h
    assert meta["is_square_padded_transparent"] is True
    assert "camera_eye" in meta
    assert "camera_target" in meta
    assert 15.0 <= abs(meta["camera_pitch_deg"]) <= 75.0
    assert -180.0 <= meta["camera_azimuth_deg"] <= 180.0
    camera_y = meta["camera_eye"][1]
    assert instances[0]["bbox_min"][1] + 1.0 <= camera_y <= instances[0]["bbox_max"][1] + 1.5
    assert 0.0 <= meta["target_mesh_visibility"] <= 1.0
    assert 0.0 <= meta["very_good_triangle_area_fraction"] <= 1.0
    assert meta["very_good_triangle_area_fraction"] <= meta["good_triangle_area_fraction"] <= 1.0
    assert 0.0 <= meta["triangle_incidence_penalty"] <= 50.0
    assert meta["camera_roll_deg"] == 0.0
    assert meta["frame_orientation"] in ("portrait", "landscape")


def test_target_mesh_visibility_rejects_occluded_sampled_faces():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0],
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    instance = {
        "triangle_indices": np.array([1]),
        "visibility_points": np.array([[0.0, 0.0, 0.0]]),
        "visibility_weights": np.array([1.0]),
    }

    assert _target_mesh_visibility(scene, np.array([0.0, 0.0, 2.0]), instance) == 0.0

    target_only = o3d.geometry.TriangleMesh()
    target_only.vertices = o3d.utility.Vector3dVector([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    target_only.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    unobstructed = o3d.t.geometry.RaycastingScene()
    unobstructed.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(target_only))
    instance["triangle_indices"] = np.array([0])
    assert _target_mesh_visibility(unobstructed, np.array([0.0, 0.0, 2.0]), instance) == 1.0


def test_target_triangle_incidence_penalty_prefers_frontal_faces_and_requested_bands():
    vertices = np.array([
        [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2]])
    instance = {
        "visibility_points": np.array([[0.0, 0.0, 0.0]]),
        "visibility_weights": np.array([1.0]),
        "visibility_triangle_indices": np.array([0]),
    }
    elevations = np.radians([0.0, 30.0, 40.0, 50.0, 90.0, 180.0])
    eyes = np.column_stack((np.sin(elevations), np.zeros(len(elevations)), np.cos(elevations)))

    penalty, very_good_fraction, good_fraction = _target_triangle_incidence_quality(
        eyes, instance, vertices, triangles
    )

    assert np.allclose(penalty, [0.0, 0.0, 0.5, 1.0, 1.0, 0.0], atol=1e-5)
    assert np.allclose(very_good_fraction, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0], atol=1e-5)
    assert np.allclose(good_fraction, [1.0, 1.0, 1.0, 0.0, 0.0, 1.0], atol=1e-5)


def test_target_triangle_incidence_uses_area_weighted_quality_fractions():
    vertices = np.array([
        [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 0.0, 1.0],
    ])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    instance = {
        "visibility_points": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "visibility_weights": np.array([9.0, 1.0]),
        "visibility_triangle_indices": np.array([0, 1]),
    }

    penalty, very_good_fraction, good_fraction = _target_triangle_incidence_quality(
        np.array([0.0, 0.0, 2.0]), instance, vertices, triangles
    )

    assert penalty == pytest.approx(0.1)
    assert very_good_fraction == pytest.approx(0.9)
    assert good_fraction == pytest.approx(0.9)


def test_end_to_end_extraction_synthetic(tmp_path):
    output_dir = tmp_path / "output"

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [-0.2, -0.2, 1.0], [0.2, -0.2, 1.0], [0.0, 0.2, 1.0],  # object
        [-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 2.0, 0.0],  # wall
    ])
    mesh.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],
        [3, 4, 5],
    ])
    colors = np.array([
        [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9],  # colorful object
        [0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8],  # wall
    ], dtype=np.float64)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_path = tmp_path / "original_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    labels = np.array([10, 5], dtype=np.int32)
    labels_path = tmp_path / "triangle_class_labels.npy"
    np.save(labels_path, labels)

    struct_json = tmp_path / "structural_classes.json"
    struct_json.write_text(json.dumps({
        "classifications": {
            "5": {"structural_role": "wall"},
            "10": {"structural_role": "object", "details": "desk lamp"},
        }
    }))

    rc = run_extraction(
        mesh_path=mesh_path,
        labels_path=labels_path,
        structural_json_path=struct_json,
        workspace=tmp_path,
        output_dir=output_dir,
        padding=0.15,
        render_res=512,
        min_instance_faces=1,
        min_instance_area=0.001,
    )

    assert rc == 0
    manifest_path = output_dir / "non_structural_frames_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["successful_extractions"] == 1
    assert "10" in manifest["extracted_classes"]

    class10_info = manifest["extracted_classes"]["10"]
    assert class10_info["role"] == "object"
    png_path = output_dir / "class_010_inst_000_base.png"
    assert png_path.exists()
    assert not (output_dir / "class_010.png").exists()
    assert not (output_dir / "class_010_object.png").exists()

    with Image.open(png_path) as im:
        assert im.mode == "RGB"
        assert im.size[0] == im.size[1]  # perfectly square!


def test_end_to_end_multi_instance_extraction(tmp_path):
    output_dir = tmp_path / "output_mesh"
    output_gs_dir = tmp_path / "output_gs"

    # Create mesh with TWO distinct instances of object class 10 + wall class 5
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        # Instance 0 of Class 10 (at X = -1.0)
        [-1.2, -0.2, 1.0], [-0.8, -0.2, 1.0], [-1.0, 0.2, 1.0],
        # Instance 1 of Class 10 (at X = +1.0)
        [0.8, -0.2, 1.0], [1.2, -0.2, 1.0], [1.0, 0.2, 1.0],
        # Wall (Class 5)
        [-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 2.0, 0.0],
    ])
    mesh.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],  # Instance 0
        [3, 4, 5],  # Instance 1
        [6, 7, 8],  # Wall
    ])
    colors = np.array([
        [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9],  # inst 0
        [0.9, 0.8, 0.1], [0.1, 0.8, 0.9], [0.8, 0.1, 0.8],  # inst 1
        [0.7, 0.7, 0.7], [0.7, 0.7, 0.7], [0.7, 0.7, 0.7],  # wall
    ], dtype=np.float64)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_path = tmp_path / "original_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    labels = np.array([10, 10, 5], dtype=np.int32)
    labels_path = tmp_path / "triangle_class_labels.npy"
    np.save(labels_path, labels)

    struct_json = tmp_path / "structural_classes.json"
    struct_json.write_text(json.dumps({
        "classifications": {
            "5": {"structural_role": "wall"},
            "10": {"structural_role": "object", "details": "chair"},
        }
    }))

    rc = run_extraction(
        mesh_path=mesh_path,
        labels_path=labels_path,
        structural_json_path=struct_json,
        workspace=tmp_path,
        output_dir=output_dir,
        gs_output_dir=output_gs_dir,
        skip_gs=True,
        padding=0.15,
        render_res=512,
        min_instance_faces=1,
        min_instance_area=0.001,
    )

    assert rc == 0
    manifest_path = output_dir / "non_structural_frames_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["total_instances_extracted"] == 2
    assert "10" in manifest["extracted_classes"]
    assert manifest["extracted_classes"]["10"]["num_instances"] == 2

    # Both instances rendered to separate files with matching naming convention
    inst0_path = output_dir / "class_010_inst_000_base.png"
    inst1_path = output_dir / "class_010_inst_001_base.png"

    assert inst0_path.exists()
    assert inst1_path.exists()
    assert not (output_dir / "class_010.png").exists()

    assert "10_0" in manifest["extracted_instances"]
    assert "10_1" in manifest["extracted_instances"]



def test_compute_instance_frontal_direction():
    # Elongated wardrobe along X axis (width 4.0, depth 0.4) located at Z = -3.0
    # Room center is at (0, 0, 0)
    xs = np.linspace(-2.0, 2.0, 20)
    zs = np.linspace(-3.2, -2.8, 5)
    pts = []
    for x in xs:
        for z in zs:
            pts.append([x, 0.0, z])
    pts = np.array(pts)

    room_center = np.array([0.0, 0.0, 0.0])
    front_dir = compute_instance_frontal_direction(pts, room_center)

    # Major axis is along X. Normal axis is along Z.
    # Frontal direction pointing to room center (0, 0, 0) must have positive Z component!
    assert abs(front_dir[0]) < 0.1
    assert front_dir[2] > 0.9


def test_get_standard_elevation_deg():
    assert get_standard_elevation_deg("furniture_vertical", np.array([1.0, 2.0, 0.5])) == 10.0
    assert get_standard_elevation_deg("door", np.array([0.9, 2.1, 0.1])) == 10.0
    assert get_standard_elevation_deg("furniture_horizontal", np.array([2.0, 0.5, 1.5])) == 18.0
    assert get_standard_elevation_deg("object", np.array([0.3, 0.3, 0.3])) == 14.0


def test_cull_foreground_occluders_preserves_background():
    # Setup:
    # Camera at (0, 0, 0) looking at +Z direction
    # Foreground occluder (CID 99) at Z = 1.0 (between camera and target)
    # Target instance (CID 10) at Z = 2.0
    # Background wall (CID 5) at Z = 3.0
    # Supporting floor (CID 14) at Y = -1.0 across all Z
    all_vertices = np.array([
        # Foreground occluder (Z = 1.0)
        [-0.1, -0.1, 1.0], [0.1, -0.1, 1.0], [0.0, 0.1, 1.0],
        # Target object (Z = 2.0)
        [-0.3, -0.3, 2.0], [0.3, -0.3, 2.0], [0.0, 0.3, 2.0],
        # Background wall (Z = 3.0)
        [-1.0, -1.0, 3.0], [1.0, -1.0, 3.0], [0.0, 1.0, 3.0],
        # Floor (supporting underneath, Z = 2.0, Y = -0.5)
        [-1.0, -0.5, 1.5], [1.0, -0.5, 1.5], [0.0, -0.5, 2.5],
    ], dtype=np.float64)

    all_triangles = np.array([
        [0, 1, 2],    # Tri 0: foreground occluder (CID 99)
        [3, 4, 5],    # Tri 1: target object (CID 10)
        [6, 7, 8],    # Tri 2: background wall (CID 5)
        [9, 10, 11],  # Tri 3: floor under object (CID 14)
    ], dtype=np.int32)

    labels = np.array([99, 10, 5, 14], dtype=np.int32)
    vertex_colors = np.ones((12, 3), dtype=np.float64)

    cam_eye = np.array([0.0, 0.0, 0.0])
    cam_target = np.array([0.0, 0.0, 2.0])
    up = np.array([0.0, 1.0, 0.0])
    inst_pts = all_vertices[3:6]

    clean_mesh, culled_count, culled_cids = cull_foreground_occluders(
        all_vertices=all_vertices,
        all_triangles=all_triangles,
        all_vertex_colors=vertex_colors,
        labels=labels,
        class_id=10,
        cam_eye=cam_eye,
        cam_target=cam_target,
        up=up,
        inst_pts=inst_pts,
        margin_m=0.08,
    )

    # Occluder at Z=1.0 is culled (1 triangle, CID 99)
    assert culled_count == 1
    assert culled_cids == [99]

    # Clean mesh has remaining 3 triangles: target (CID 10), background (CID 5), floor (CID 14)
    assert len(clean_mesh.triangles) == 3


def test_detect_scene_floor_levels(tmp_path):
    levels_json = tmp_path / "structural_levels_manifest.json"
    levels_json.write_text(json.dumps({
        "levels": [
            {"level_index": 0, "mean_floor_elevation_y": -1.228},
            {"level_index": 1, "mean_floor_elevation_y": 1.028},
        ]
    }))

    levels = detect_scene_floor_levels(levels_manifest_path=levels_json)
    assert len(levels) == 2
    assert levels[0]["level_index"] == 0
    assert abs(levels[0]["floor_y"] - (-1.228)) < 1e-3
    assert levels[1]["level_index"] == 1
    assert abs(levels[1]["floor_y"] - 1.028) < 1e-3


def test_assign_instance_to_floor_level():
    levels = [
        {"level_index": 0, "floor_y": -1.2285},
        {"level_index": 1, "floor_y": 1.0277},
    ]

    # Instance at Y in [-1.2, -0.6] (Level 0 bed)
    pts_lvl0 = np.array([
        [0.0, -1.2, 0.0],
        [1.0, -0.6, 0.0],
    ])
    fl0 = assign_instance_to_floor_level(pts_lvl0, levels)
    assert fl0["level_index"] == 0
    assert abs(fl0["floor_y"] - (-1.2285)) < 1e-4

    # Instance at Y in [1.1, 1.8] (Level 1 mezzanine furniture)
    pts_lvl1 = np.array([
        [0.0, 1.1, 0.0],
        [1.0, 1.8, 0.0],
    ])
    fl1 = assign_instance_to_floor_level(pts_lvl1, levels)
    assert fl1["level_index"] == 1
    assert abs(fl1["floor_y"] - 1.0277) < 1e-4


def test_identify_instance_biggest_side():
    # Bed: X=2.0, Y=0.5, Z=2.0 -> Top face (2.0x2.0=4.0) is biggest
    bed_pts = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, -0.5, 1.0],
    ])
    room_center = np.array([0.0, 0.0, 0.0])
    bed_side = identify_instance_biggest_side(bed_pts, room_center)
    assert bed_side["side_name"] == "top"
    assert bed_side["is_horizontal_top"] is True
    assert np.allclose(bed_side["normal"], [0.0, 1.0, 0.0])

    # Wardrobe: X=1.5, Y=2.2, Z=0.5 -> Front face (1.5x2.2=3.3) is biggest
    wardrobe_pts = np.array([
        [-0.75, 0.0, -2.5],
        [0.75, 2.2, -2.0],
    ])
    wardrobe_side = identify_instance_biggest_side(wardrobe_pts, room_center)
    assert wardrobe_side["side_name"] == "front_z"
    assert wardrobe_side["is_horizontal_top"] is False
    assert wardrobe_side["normal"][2] > 0  # points towards room center (Z=0)


def test_get_camera_up_vector():
    # Looking down (overhead view, f_vec = [0, -1, 0]) with 3D front_dir_h
    up_overhead_3d = get_camera_up_vector(np.array([0.0, -1.0, 0.0]), front_dir_h=np.array([0.0, 0.0, 1.0]))
    assert abs(up_overhead_3d[1]) < 1e-4
    assert up_overhead_3d[2] < -0.9  # points towards -Z (headboard)

    # Looking down with 2D front_dir_h [dx, dz]
    up_overhead_2d = get_camera_up_vector(np.array([0.0, -1.0, 0.0]), front_dir_h=np.array([0.0, 1.0]))
    assert abs(up_overhead_2d[1]) < 1e-4
    assert up_overhead_2d[2] < -0.9

    # Looking horizontally (f_vec = [0, 0, -1])
    up_horizontal = get_camera_up_vector(np.array([0.0, 0.0, -1.0]))
    assert np.allclose(up_horizontal, [0.0, 1.0, 0.0])


def test_compute_non_structural_purity():
    # Target instance at (0, 0, 0)
    inst_pts = np.array([
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=np.float32)

    cam_eye = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    cam_look = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Case 1: No other non-structural objects in view
    purity_clean, vis_clean = compute_non_structural_purity(
        cam_eye=cam_eye,
        cam_look=cam_look,
        up=up,
        inst_pts=inst_pts,
        target_cid=2,
        other_instances=[],
    )
    assert purity_clean == 1.0
    assert vis_clean == 1.0

    # Case 2: Another non-structural object occluding along the exact line of sight (z = 1.0)
    # Line of sight from (0,0,2) to (-0.5,-0.5,0) passes through (-0.25, -0.25, 1.0)
    occluder_pts = np.array([
        [-0.25, -0.25, 1.0],  # directly covers target point at [-0.5, -0.5, 0.0]
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    purity_occ, vis_occ = compute_non_structural_purity(
        cam_eye=cam_eye,
        cam_look=cam_look,
        up=up,
        inst_pts=inst_pts,
        target_cid=2,
        other_instances=[{"class_id": 92, "pts": occluder_pts}],
    )
    # Purity and visibility must be strictly less than 1.0 due to occluder
    assert purity_occ < 1.0
    assert vis_occ < 1.0


def test_min_triangle_distance_constraint():
    # Create a simple box instance surrounded by room mesh
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=0.5, depth=1.0)
    mesh_box.translate([-0.5, 0.0, -0.5])
    vertices = np.asarray(mesh_box.vertices)
    triangles = np.asarray(mesh_box.triangles)
    vertex_colors = np.ones((len(vertices), 3), dtype=np.float32)
    labels = np.full(len(triangles), 2, dtype=np.int32)

    instance = {
        "instance_id": 0,
        "points": vertices,
        "bbox_min": vertices.min(axis=0),
        "bbox_max": vertices.max(axis=0),
        "dimensions": vertices.max(axis=0) - vertices.min(axis=0),
        "triangle_count": len(triangles),
        "area_m2": 5.0,
    }

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_box)
    raycast_scene = o3d.t.geometry.RaycastingScene()
    raycast_scene.add_triangles(mesh_t)

    min_tri_dist = 0.10
    views = render_synthetic_in_context_frame(
        instance=instance,
        all_vertices=vertices,
        all_triangles=triangles,
        all_vertex_colors=vertex_colors,
        labels=labels,
        class_id=2,
        structural_cids={10, 11},
        room_center=np.array([0.0, 0.0, 0.0]),
        raycast_scene=raycast_scene,
        wall_scene=raycast_scene,
        render_res=256,
        min_triangle_distance=min_tri_dist,
    )

    assert views["base"][1]["camera_distance_to_nearest_triangle_m"] >= min_tri_dist


def test_validate_rendered_frame_single_color():
    # 1. Uniform constant color image (single unique color)
    flat = np.full((120, 120, 3), 100, dtype=np.uint8)
    ok, reason, metrics = validate_rendered_frame(flat)
    assert not ok
    assert "single_color" in reason
    assert metrics["unique_colors"] == 1

    # 2. Narrow dynamic range (< 10)
    narrow = np.full((120, 120, 3), 100, dtype=np.uint8)
    narrow[:60, :60, :] = 104  # ptp = 4
    ok, reason, metrics = validate_rendered_frame(narrow)
    assert not ok
    assert "single_color" in reason

    # 3. Transparent RGBA with flat foreground
    rgba_flat = np.zeros((120, 120, 4), dtype=np.uint8)
    rgba_flat[20:100, 20:100, :3] = 128
    rgba_flat[20:100, 20:100, 3] = 255
    ok, reason, metrics = validate_rendered_frame(rgba_flat)
    assert not ok
    assert "single_color" in reason


def test_validate_rendered_frame_noise():
    # Uncorrelated random noise has near-zero lag-1 spatial autocorrelation
    np.random.seed(42)
    noise = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
    ok, reason, metrics = validate_rendered_frame(noise)
    assert not ok
    assert "pure_noise" in reason
    assert "autocorr" in metrics
    assert metrics["autocorr"] < 0.35


def test_validate_rendered_frame_featureless():
    # Almost flat blurred image with low variance and laplacian variance < 0.05
    flat_grad = np.zeros((120, 120, 3), dtype=np.uint8)
    # subtle gradient with std < 2.0 and laplacian < 0.05
    y, x = np.mgrid[:120, :120]
    val = np.clip(100 + 0.01 * x + 0.01 * y, 0, 255).astype(np.uint8)
    flat_grad[:, :, 0] = val
    flat_grad[:, :, 1] = val
    flat_grad[:, :, 2] = val
    ok, reason, metrics = validate_rendered_frame(flat_grad)
    assert not ok


def test_validate_rendered_frame_rejects_smooth_high_diversity_blur():
    y, x = np.mgrid[:256, :256]
    img = np.stack([x, y, (x + y) // 2], axis=2).astype(np.uint8)

    ok, reason, metrics = validate_rendered_frame(img)

    assert metrics["unique_colors"] >= 28
    assert metrics["lap_var"] < 0.05
    assert not ok
    assert "flat_or_blurry_surface" in reason


def test_validate_rendered_frame_valid():
    # Image with clear edges, contrast, and structure
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    img[20:100, 20:100, 0] = np.linspace(40, 220, 80)[:, None]
    img[20:100, 20:100, 1] = np.linspace(60, 240, 80)[None, :]
    img[20:100, 20:100, 2] = 150
    ok, reason, metrics = validate_rendered_frame(img)
    assert ok
    assert reason == "valid"


def test_validate_rendered_frame_rejects_overexposure_and_elongated_triangle_views():
    y, x = np.mgrid[:120, :120]
    textured = np.stack([x * 2, y * 2, (x + y)], axis=2).astype(np.uint8)
    overexposed = np.full_like(textured, 255)

    ok, reason, metrics = validate_rendered_frame(overexposed)
    assert not ok and reason.startswith("overexposed:")
    assert metrics["overexposed_pixel_ratio"] >= 0.20

    ok, reason, metrics = validate_rendered_frame(textured, elongated_triangle_area_ratio=0.25)
    assert not ok and reason.startswith("mesh_artifact:")
    assert metrics["elongated_triangle_area_ratio"] == 0.25


def test_quality_filter_routing_to_discarded_folder(tmp_path):
    output_dir = tmp_path / "mesh_out"
    discarded_dir = tmp_path / "mesh_out_discarded"
    output_dir.mkdir()
    Image.new("RGB", (8, 8), (120, 80, 40)).save(output_dir / "class_010_inst_000_base.png")

    # Create synthetic mesh with 1 instance of class 10 and 1 wall
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0],
        [-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 2.0, 0.0],
    ])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((6, 3), 0.7))
    mesh_path = tmp_path / "original_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    labels = np.array([10, 5], dtype=np.int32)
    labels_path = tmp_path / "triangle_class_labels.npy"
    np.save(labels_path, labels)

    struct_json = tmp_path / "structural_classes.json"
    struct_json.write_text(json.dumps({
        "classifications": {
            "5": {"structural_role": "wall"},
            "10": {"structural_role": "object"},
        }
    }))

    # Run with quality check requiring very high dynamic range so it gets discarded
    rc = run_extraction(
        mesh_path=mesh_path,
        labels_path=labels_path,
        structural_json_path=struct_json,
        workspace=tmp_path,
        output_dir=output_dir,
        mesh_discarded_dir=discarded_dir,
        skip_gs=True,
        min_instance_faces=1,
        min_instance_area=0.001,
        enable_quality_check=True,
        min_unique_colors=100,  # force discard of synthetic render with only ~35 colors
    )

    # Output dir should NOT contain the discarded image
    assert not (output_dir / "class_010_inst_000_base.png").exists()

    # Discarded dir SHOULD contain the discarded image and manifest
    assert (discarded_dir / "class_010_inst_000_base.png").exists()
    assert (discarded_dir / "discarded_manifest.json").exists()

    with open(discarded_dir / "discarded_manifest.json") as f:
        dm = json.load(f)
    assert dm["total_discarded"] == 5
    assert "10_0:base" in dm["discarded_instances"]
    assert "single_color" in dm["discarded_instances"]["10_0:base"]["reason"]


def test_gaussian_checkpoint_uses_phase07_scan_transform(tmp_path):
    import torch

    scan_dir = tmp_path / "scene" / "scan_2"
    scan_dir.mkdir(parents=True)
    (scan_dir / "mesh.json").write_text("{}")
    transform = np.array([
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 1.0, 0.0, 0.5],
        [-1.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    manifest = scan_dir.parent / "multi_room_alignment" / "stats" / "aligned_scans_transforms.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"scans": {"scan_2": {"transform_matrix": transform.tolist()}}}))
    checkpoint = scan_dir / "gaussian_splat" / "checkpoint_sealed.pt"
    checkpoint.parent.mkdir()
    torch.save({
        "sh_degree": 0,
        "splats": {
            "means": torch.tensor([[1.0, 2.0, 3.0]]),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "scales": torch.zeros((1, 3)),
            "opacities": torch.zeros(1),
            "sh0": torch.zeros((1, 1, 3)),
            "shN": torch.empty((1, 0, 3)),
        },
    }, checkpoint)

    splats, _ = load_gaussian_splats(checkpoint, device="cpu", workspace=scan_dir)

    torch.testing.assert_close(splats["means"], torch.tensor([[6.0, 2.5, 1.0]]))
    torch.testing.assert_close(
        splats["quats"], torch.tensor([[2 ** -0.5, 0.0, 2 ** -0.5, 0.0]]), atol=1e-6, rtol=0
    )





