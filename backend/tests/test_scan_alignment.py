"""Unit and integration tests for multi-room scan alignment (07_alignment/scans)."""
import sys
from pathlib import Path
import json
import numpy as np
import plyfile
import pytest
from shapely.geometry import Polygon
import trimesh

ALIGNMENT_DIR = Path(__file__).resolve().parents[1] / "07_alignment"
if str(ALIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ALIGNMENT_DIR))

from arcore_trajectory_bridge import ARCoreTrajectoryBridge
from portal_detector import PortalDetector, DoorPortal
from wall_constraint_optimizer import WallConstraintOptimizer
from dense_reveal_icp import DenseRevealICP
from pose_graph_optimizer import PoseGraphOptimizer, AlignmentEdge


def test_composite_door_decomposition(tmp_path):
    """Verifies that a wide composite door instance (e.g. 3 adjacent hallway doors)
    is properly decomposed into 3 individual sub-portals with realistic widths.
    """
    detector = PortalDetector(min_door_width=0.60, max_single_door_width=1.10)

    boxes = []
    for d_idx, x_offset in enumerate([0.0, 1.0, 2.0]):
        box = trimesh.creation.box(extents=[0.80, 2.10, 0.05])
        box.apply_translation([x_offset + 0.40, 1.05, 0.0])
        boxes.append(box)

    combined_mesh = trimesh.util.concatenate(boxes)
    door_mesh_path = tmp_path / "door_inst_hallway_001.ply"
    combined_mesh.export(door_mesh_path)

    portals = detector.decompose_composite_door_instance(door_mesh_path)
    assert len(portals) == 3, f"Expected 3 decomposed sub-portals, got {len(portals)}"

    for idx, p in enumerate(portals):
        assert 0.60 <= p.width <= 1.05, f"Sub-portal {idx} width {p.width:.2f} outside single door range"
        assert p.sub_portal_index == idx
        assert p.threshold_center[1] == pytest.approx(0.0, abs=1e-5)


def test_trajectory_guided_portal_selection():
    """Verifies that ARCore traversal ray-piercing selects the exact traversed door
    (e.g. Door #2 out of 3 doors in a hallway).
    """
    detector = PortalDetector()
    portals = [
        DoorPortal(
            portal_id="door_sub_0",
            center=np.array([0.40, 1.05, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent=np.array([1.0, 0.0, 0.0]),
            width=0.80,
            height=2.10,
            threshold_center=np.array([0.40, 0.0, 0.0]),
            bounding_box_min=np.array([0.0, 0.0, -0.05]),
            bounding_box_max=np.array([0.80, 2.10, 0.05]),
            sub_portal_index=0,
        ),
        DoorPortal(
            portal_id="door_sub_1",
            center=np.array([1.40, 1.05, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent=np.array([1.0, 0.0, 0.0]),
            width=0.80,
            height=2.10,
            threshold_center=np.array([1.40, 0.0, 0.0]),
            bounding_box_min=np.array([1.0, 0.0, -0.05]),
            bounding_box_max=np.array([1.80, 2.10, 0.05]),
            sub_portal_index=1,
        ),
        DoorPortal(
            portal_id="door_sub_2",
            center=np.array([2.40, 1.05, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            tangent=np.array([1.0, 0.0, 0.0]),
            width=0.80,
            height=2.10,
            threshold_center=np.array([2.40, 0.0, 0.0]),
            bounding_box_min=np.array([2.0, 0.0, -0.05]),
            bounding_box_max=np.array([2.80, 2.10, 0.05]),
            sub_portal_index=2,
        ),
    ]

    traj_point = np.array([1.45, 1.20, 0.05])
    selected = detector.select_traversed_portal(portals, traj_point)
    assert selected.portal_id == "door_sub_1", f"Expected door_sub_1, got {selected.portal_id}"


def test_wall_constraint_optimizer_automatic_thickness_discovery(tmp_path):
    """Verifies that the optimizer AUTOMATICALLY discovers wall thickness without any manual input."""
    scan_a = tmp_path / "scan_a"
    scan_b = tmp_path / "scan_b"
    for s in [scan_a, scan_b]:
        (s / "mesh_elaboration" / "elements").mkdir(parents=True)

    floor_a = trimesh.creation.box(extents=[2.0, 0.02, 2.0])
    floor_a.apply_translation([-1.0, 0.0, 1.0])
    floor_a.export(scan_a / "mesh_elaboration" / "elements" / "floor_level_000_area_000.ply")

    floor_b = trimesh.creation.box(extents=[2.0, 0.02, 2.0])
    floor_b.apply_translation([1.0, 0.0, 1.0])
    floor_b.export(scan_b / "mesh_elaboration" / "elements" / "floor_level_000_area_000.ply")

    portal_a = DoorPortal(
        portal_id="portal_a",
        center=np.array([0.0, 1.05, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        tangent=np.array([0.0, 0.0, 1.0]),
        width=0.85,
        height=2.10,
        threshold_center=np.array([0.0, 0.0, 1.0]),
        bounding_box_min=np.array([0.0, 0.0, 0.5]),
        bounding_box_max=np.array([0.0, 2.10, 1.5]),
    )

    portal_b = DoorPortal(
        portal_id="portal_b",
        center=np.array([0.0, 1.05, 1.0]),
        normal=np.array([-1.0, 0.0, 0.0]),
        tangent=np.array([0.0, 0.0, 1.0]),
        width=0.85,
        height=2.10,
        threshold_center=np.array([0.0, 0.0, 1.0]),
        bounding_box_min=np.array([0.0, 0.0, 0.5]),
        bounding_box_max=np.array([0.0, 2.10, 1.5]),
    )

    coarse_T = np.eye(4)
    optimizer = WallConstraintOptimizer()

    # Automatically discover without passing any manual thickness prior
    res = optimizer.optimize(
        coarse_transform=coarse_T,
        portal_i=portal_a,
        portal_j=portal_b,
        scan_dir_i=scan_a,
        scan_dir_j=scan_b,
    )

    assert res.success
    # Should discover non-overlapping clearance thickness (> 0.05m and < 0.60m)
    assert 0.05 <= res.automatically_discovered_wall_thickness <= 0.60
    assert res.overlap_area_m2 < 1e-4, f"Floor overlap detected: {res.overlap_area_m2:.4f}m²"


def test_pose_graph_optimizer_3_deliverables(tmp_path):
    """Verifies that the pose graph optimizer generates all 3 requested canonical deliverables:
    1. aligned_sealed_meshes.ply
    2. aligned_regularized_meshes.ply
    3. aligned_sealed_gaussians.ply
    """
    pgo = PoseGraphOptimizer()

    s0 = tmp_path / "scan_0"
    s1 = tmp_path / "scan_1"
    for s in [s0, s1]:
        (s / "mesh_elaboration").mkdir(parents=True)
        (s / "gaussian_splat").mkdir(parents=True)

        # 1. Sealed mesh (mesh.ply)
        box_sealed = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        box_sealed.export(s / "mesh.ply")

        # 2. Regularized mesh (complete.ply)
        box_reg = trimesh.creation.box(extents=[0.9, 0.9, 0.9])
        box_reg.export(s / "mesh_elaboration" / "complete.ply")

        # 3. Sealed Gaussian Splat PLY
        N = 10
        vertex_dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
        data = np.zeros(N, dtype=vertex_dtype)
        data["rot_0"] = 1.0  # Identity quaternion
        el = plyfile.PlyElement.describe(data, "vertex")
        plyfile.PlyData([el], text=False, byte_order="<").write(s / "gaussian_splat" / "gaussian_splat_sealed.ply")

    T_01 = np.eye(4)
    T_01[0, 3] = 2.0
    pgo.add_edge(AlignmentEdge(from_scan="scan_0", to_scan="scan_1", relative_transform=T_01))

    out_dir = tmp_path / "output"
    manifest = pgo.assemble_all_deliverables({"scan_0": s0, "scan_1": s1}, out_dir)

    # Verify all 3 deliverables are present and valid
    assert (out_dir / "aligned_sealed_meshes.ply").is_file()
    assert (out_dir / "aligned_regularized_meshes.ply").is_file()
    assert (out_dir / "aligned_sealed_gaussians.ply").is_file()
    assert (out_dir / "multi_room_alignment.json").is_file()

    # Check that Gaussians were merged (N + N = 20)
    gs_ply = plyfile.PlyData.read(out_dir / "aligned_sealed_gaussians.ply")
    assert len(gs_ply["vertex"]) == 20


def test_arcore_trajectory_bridge_synthetic(tmp_path):
    """Verifies ARCore-to-Phase04 frame calibration and coarse SE(3) bridging."""
    scan_0 = tmp_path / "scan_0"
    scan_1 = tmp_path / "scan_1"
    for s in [scan_0, scan_1]:
        (s / "sparse" / "0").mkdir(parents=True)

    R_90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    t_offset = np.array([1.5, 0.2, -0.5])

    pts_ar_0 = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.2]),
        np.array([0.0, 1.0, -0.1]),
        np.array([0.2, 0.1, 1.0]),
        np.array([1.0, 1.2, 0.8]),
    ]

    frames_0 = []
    lines_0 = ["# Image list\n"]
    for i, pos_ar in enumerate(pts_ar_0):
        mat_ar = np.eye(4)
        mat_ar[:3, 3] = pos_ar
        frames_0.append({
            "file_path": f"images/frame_{i:05d}.jpg",
            "transform_matrix": mat_ar.tolist()
        })
        c_col = R_90 @ pos_ar + t_offset
        lines_0.append(f"{i+1} 1.0 0.0 0.0 0.0 {-c_col[0]} {-c_col[1]} {-c_col[2]} 1 frame_{i:05d}.jpg\n")
        lines_0.append("0.0 0.0 -1\n")

    with open(scan_0 / "transforms.json", "w") as f:
        json.dump({"frames": frames_0}, f)
    with open(scan_0 / "sparse" / "0" / "images.txt", "w") as f:
        f.writelines(lines_0)

    pts_ar_1 = [
        np.array([2.0, 0.0, 1.0]),
        np.array([3.0, 0.0, 1.2]),
        np.array([2.0, 1.0, 0.9]),
        np.array([2.2, 0.1, 2.0]),
        np.array([3.0, 1.2, 1.8]),
    ]
    frames_1 = []
    lines_1 = ["# Image list\n"]
    for i, pos_ar in enumerate(pts_ar_1):
        mat_ar = np.eye(4)
        mat_ar[:3, 3] = pos_ar
        frames_1.append({
            "file_path": f"images/frame_{i:05d}.jpg",
            "transform_matrix": mat_ar.tolist()
        })
        c_col = R_90 @ pos_ar + t_offset
        lines_1.append(f"{i+1} 1.0 0.0 0.0 0.0 {-c_col[0]} {-c_col[1]} {-c_col[2]} 1 frame_{i:05d}.jpg\n")
        lines_1.append("0.0 0.0 -1\n")

    with open(scan_1 / "transforms.json", "w") as f:
        json.dump({"frames": frames_1}, f)
    with open(scan_1 / "sparse" / "0" / "images.txt", "w") as f:
        f.writelines(lines_1)

    bridge = ARCoreTrajectoryBridge()
    T_p4_0 = bridge.estimate_arcore_to_phase04_rigid_transform(scan_0)
    np.testing.assert_allclose(T_p4_0[:3, :3], R_90, atol=1e-5)
    np.testing.assert_allclose(T_p4_0[:3, 3], t_offset, atol=1e-5)

    res = bridge.estimate_coarse_relative_transform(scan_0, scan_1)
    assert res.yaw_cardinal_deg in [0.0, 90.0, 180.0, 270.0, 360.0]


def test_dense_reveal_icp_horizontal_constraint(tmp_path):
    """Verifies that Point-to-Plane ICP strictly constrains refinement to the horizontal plane without tilt."""
    scan_i = tmp_path / "scan_i"
    scan_j = tmp_path / "scan_j"
    for s in [scan_i, scan_j]:
        s.mkdir(parents=True)

    box_i = trimesh.creation.box(extents=[0.80, 2.10, 0.40])
    box_i = box_i.subdivide().subdivide().subdivide()
    box_i.apply_translation([0.0, 1.05, 0.20])
    box_i.export(scan_i / "mesh.ply")

    box_j = trimesh.creation.box(extents=[0.80, 2.10, 0.40])
    box_j = box_j.subdivide().subdivide().subdivide()
    box_j.apply_translation([0.02, 1.05, 0.20])
    box_j.export(scan_j / "mesh.ply")

    portal_i = DoorPortal(
        portal_id="door_i",
        center=np.array([0.0, 1.05, 0.20]),
        normal=np.array([0.0, 0.0, 1.0]),
        tangent=np.array([1.0, 0.0, 0.0]),
        width=0.80,
        height=2.10,
        threshold_center=np.array([0.0, 0.0, 0.20]),
        bounding_box_min=np.array([-0.40, 0.0, 0.0]),
        bounding_box_max=np.array([0.40, 2.10, 0.40]),
    )

    icp = DenseRevealICP(crop_radius=1.5, max_correspondence_distance=0.08)
    initial_T = np.eye(4)

    res = icp.refine(
        initial_transform=initial_T,
        portal_i=portal_i,
        scan_dir_i=scan_i,
        scan_dir_j=scan_j,
    )

    assert res.applied
    assert res.transform[1, 3] == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(res.transform[:3, :3] @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-5)
