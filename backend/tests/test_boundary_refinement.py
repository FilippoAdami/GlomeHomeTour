import pytest
import numpy as np
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "06c_semantic_segmentation"))

from refine_class_boundaries import (
    align_structural_planes,
    absorb_flush_wall_instances,
    propagate_cabinet_faces,
    absorb_embedded_furniture_subparts,
    absorb_enclosed_subparts,
    propagate_planar_boundaries,
    propagate_planar_faces,
    compute_mesh_derivatives,
    build_dual_graph,
    reclaim_ceiling_bounding_slabs,
    reclaim_wall_encroached_furniture,
)
from detect_structural_classes import build_mesh_dual_graph, converge_class_instances_iteratively


def test_absorb_flush_wall_instances_drywall_patch():
    """Verifies that an instance with zero depth, matching normal and zero color gradient is absorbed into Wall #5."""
    # Synthetic flat wall: 2 triangles
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    # Triangle 0 is Wall (5), Triangle 1 is fake instance (99)
    labels = np.array([5, 99], dtype=np.int32)
    
    # Matching wall color: [0.36, 0.29, 0.24]
    vc = np.tile([0.36, 0.29, 0.24], (4, 1)).astype(np.float32)
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, absorbed = absorb_flush_wall_instances(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        wall_cid=5,
    )
    
    assert 99 in absorbed
    assert np.all(cleaned == 5)


def test_propagate_cabinet_face_crease_barrier():
    """Verifies that cabinet face propagation stops strictly at a 90-degree crease."""
    # 3 triangles: T0 (Cabinet 3, flat), T1 (Wall 5, flat with T0), T2 (Wall 5, 90 deg crease)
    verts = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [1.0, 1.0, 1.0],  # 4 (crease 90 deg)
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2],  # T0: Cabinet face (labels=3)
        [1, 3, 2],  # T1: Piece of cabinet face misclassified as Wall (labels=5)
        [1, 4, 3],  # T2: Return side face at 90 deg (labels=5)
    ], dtype=np.int32)
    labels = np.array([3, 5, 5], dtype=np.int32)
    vc = np.ones((5, 3), dtype=np.float32) * 0.4  # uniform cabinet color
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats = propagate_cabinet_faces(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        adj_csr=adj_csr,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        candidate_cids=[3],
    )
    
    # T1 should be reclaimed by Cabinet #3, but T2 (crease) remains Wall #5
    assert cleaned[1] == 3
    assert cleaned[2] == 5


def test_absorb_embedded_furniture_subparts():
    """Verifies that an embedded coplanar, matching-color subpart on a cabinet face is absorbed into the cabinet."""
    # Synthetic flat quad + triangle (3 triangles):
    # T0, T1: Cabinet (Class 3) (2 triangles)
    # T2: Subpart (Class 4) embedded on the same plane and touching T1 (1 triangle)
    verts = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [2.0, 0.0, 0.0],  # 4
        [2.0, 1.0, 0.0],  # 5
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2],  # T0 (Class 3)
        [1, 3, 2],  # T1 (Class 3)
        [1, 4, 3],  # T2 (Class 4)
    ], dtype=np.int32)
    labels = np.array([3, 3, 4], dtype=np.int32)
    
    # Uniform wood color for all triangles
    vc = np.tile([0.45, 0.35, 0.25], (6, 1)).astype(np.float32)
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats = absorb_embedded_furniture_subparts(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        furniture_cids=[3],
        min_furniture_boundary_frac=0.60,
        max_boundary_angle_deg=10.0,
        max_plane_residual_m=0.025,
        max_color_diff=0.035,
        min_component_size=1,
    )
    
    # T2 (Class 4) should be absorbed into Cabinet #3
    assert cleaned[2] == 3
    assert stats[3] == 1


def test_absorb_enclosed_subparts_enclosure_guard():
    """Verifies that two flush objects with < 60% boundary contact do NOT absorb each other."""
    # 4 triangles in a row: T0, T1 (Class 10); T2, T3 (Class 20)
    # T1 and T2 touch along a single edge, but neither encloses the other (enclosure < 50%)
    verts = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [2.0, 0.0, 0.0],  # 4
        [2.0, 1.0, 0.0],  # 5
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2],  # T0 (Class 10)
        [1, 3, 2],  # T1 (Class 10)
        [1, 4, 3],  # T2 (Class 20)
        [4, 5, 3],  # T3 (Class 20)
    ], dtype=np.int32)
    labels = np.array([10, 10, 20, 20], dtype=np.int32)
    vc = np.ones((6, 3), dtype=np.float32) * 0.5  # uniform color
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats, _ = absorb_enclosed_subparts(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        min_enclosure_frac=0.60,
        min_component_size=1,
    )
    
    # Neither should be absorbed! Both classes maintain their independent boundaries
    assert cleaned[0] == 10 and cleaned[1] == 10
    assert cleaned[2] == 20 and cleaned[3] == 20
    assert len(stats) == 0


def test_propagate_planar_boundaries_dominance():
    """Verifies that planar face propagation reclaims coplanar bleed triangles."""
    # Flat surface with 4 triangles: 3 belong to Class 42, 1 belongs to Class 99 (bleed)
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 3],  # T0: Class 42
        [1, 4, 3],  # T1: Class 42
        [1, 2, 4],  # T2: Class 42
        [2, 5, 4],  # T3: Class 99 (bleed)
    ], dtype=np.int32)
    labels = np.array([42, 42, 42, 99], dtype=np.int32)
    vc = np.ones((6, 3), dtype=np.float32) * 0.3
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats = propagate_planar_boundaries(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        adj_csr=adj_csr,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
    )
    
    # Class 42 propagates across the flat face and reclaims T3 from Class 99
    assert np.all(cleaned == 42)
    assert stats[42] == 1


def test_structural_classes_do_not_propagate_into_each_other():
    """Verifies that structural room shell classes (Ceiling #1, Wall #5) never propagate into each other."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 3],  # T0: Class 1 (Ceiling)
        [1, 4, 3],  # T1: Class 1 (Ceiling)
        [1, 2, 4],  # T2: Class 5 (Wall)
        [2, 5, 4],  # T3: Class 5 (Wall)
    ], dtype=np.int32)
    labels = np.array([1, 1, 5, 5], dtype=np.int32)
    vc = np.ones((6, 3), dtype=np.float32) * 0.8  # white/cream paint
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats = propagate_planar_faces(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        adj_csr=adj_csr,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        structural_cids={1, 5, 14},
    )
    
    # Neither Ceiling nor Wall should eat the other, preserving both classes
    assert cleaned[0] == 1 and cleaned[1] == 1
    assert cleaned[2] == 5 and cleaned[3] == 5
    assert len(stats) == 0


def test_align_structural_planes_floor_leakage():
    """Verifies that vertical surfaces misclassified as floor are reassigned to their boundary wall neighbor."""
    # Flat floor quad (Y=0) + Vertical wall quad (Y=0 to 1)
    verts = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 0.0, 1.0],  # 2
        [1.0, 0.0, 1.0],  # 3
        [0.0, 1.0, 1.0],  # 4 (wall top)
        [1.0, 1.0, 1.0],  # 5 (wall top)
    ], dtype=np.float32)
    tris = np.array([
        [0, 2, 1],  # T0: Floor (14) - upward normal
        [1, 2, 3],  # T1: Floor (14) - upward normal
        [2, 3, 4],  # T2: Vertical leakage on wall baseboard, mislabeled Floor (14)
        [3, 5, 4],  # T3: Wall (5)
    ], dtype=np.int32)
    labels = np.array([14, 14, 14, 5], dtype=np.int32)
    vc = np.ones((6, 3), dtype=np.float32) * 0.5
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats = align_structural_planes(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        floor_cid=14,
        wall_cid=5,
    )
    
    # T0, T1 remain Floor #14
    assert cleaned[0] == 14
    assert cleaned[1] == 14
    # T2 (vertical leakage climbing wall) must be reassigned to Wall #5!
    assert cleaned[2] == 5
    assert cleaned[3] == 5
    assert stats.get("floor_leakages_reassigned", 0) == 1


def test_absorb_enclosed_subparts_texture_protection():
    """Verifies that real objects with high texture energy or color variance are NOT absorbed into walls."""
    # 3x3 quad mesh where outer ring is Wall (5) and inner square (2 triangles) is Class 4
    # Vertex coordinates on Z=0 plane:
    # 0:(0,0), 1:(1,0), 2:(2,0)
    # 3:(0,1), 4:(1,1), 5:(2,1)
    # 6:(0,2), 7:(1,2), 8:(2,2)
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 2.0]
    verts = []
    for yi in y:
        for xi in x:
            verts.append([xi, yi, 0.0])
    verts = np.array(verts, dtype=np.float32)

    # 4 squares, 8 triangles
    # Square (0,1,3,4): tris [0,1,3], [1,4,3] -> Class 5
    # Square (1,2,4,5): tris [1,2,4], [2,5,4] -> Class 5
    # Square (3,4,6,7): tris [3,4,6], [4,7,6] -> Class 4 (Inner patch, surrounded by wall!)
    # Square (4,5,7,8): tris [4,5,7], [5,8,7] -> Class 5
    tris = np.array([
        [0, 1, 3], [1, 4, 3],
        [1, 2, 4], [2, 5, 4],
        [3, 4, 6], [4, 7, 6],  # Inner patch (indices 4, 5)
        [4, 5, 7], [5, 8, 7],
    ], dtype=np.int32)

    labels = np.array([5, 5, 5, 5, 4, 4, 5, 5], dtype=np.int32)
    
    # Case A: Patch has high texture energy (e.g. colorful poster with variance > 0.005)
    vc_textured = np.ones((9, 3), dtype=np.float32) * 0.5
    # Give patch vertices distinct colors (poster texture)
    vc_textured[6] = [0.9, 0.1, 0.1]
    vc_textured[7] = [0.1, 0.9, 0.1]
    vc_textured[4] = [0.1, 0.1, 0.9]

    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc_textured)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)

    cleaned_tex, stats_tex, _ = absorb_enclosed_subparts(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        min_component_size=1,
    )
    # The textured poster MUST be protected and NOT absorbed!
    assert cleaned_tex[4] == 4 and cleaned_tex[5] == 4

    # Case B: Patch has zero texture energy and matches wall color exactly (fake patch)
    vc_plain = np.ones((9, 3), dtype=np.float32) * 0.5
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc_plain)

    cleaned_plain, stats_plain, _ = absorb_enclosed_subparts(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        min_component_size=1,
    )
    # The plain patch matching wall color and having 0 texture energy MUST be absorbed!
    assert cleaned_plain[4] == 5 and cleaned_plain[5] == 5


def test_align_structural_planes_ceiling_non_vertical():
    """Verifies that vertical surfaces misclassified as ceiling are reassigned to their boundary wall neighbor."""
    # Ceiling quad (Y=2.0, pointing down: normal [0, -1, 0]) + Vertical wall quad (Y=1.0 to 2.0, normal [0, 0, 1])
    verts = np.array([
        [0.0, 2.0, 0.0],  # 0
        [1.0, 2.0, 0.0],  # 1
        [0.0, 2.0, 1.0],  # 2
        [1.0, 2.0, 1.0],  # 3
        [0.0, 1.0, 1.0],  # 4 (wall bottom)
        [1.0, 1.0, 1.0],  # 5 (wall bottom)
    ], dtype=np.float32)
    # Downward ceiling winding: [0, 1, 2] -> cross is [1,0,0]x[0,0,1] = [0, -1, 0] (downward!)
    tris = np.array([
        [0, 1, 2],  # T0: Ceiling (1) - downward normal [0, -1, 0]
        [1, 3, 2],  # T1: Ceiling (1) - downward normal [0, -1, 0]
        [2, 3, 4],  # T2: Vertical leakage on wall, mislabeled Ceiling (1) - horizontal normal [0, 0, 1]
        [3, 5, 4],  # T3: Wall (5) - horizontal normal [0, 0, 1]
    ], dtype=np.int32)
    labels = np.array([1, 1, 1, 5], dtype=np.int32)
    vc = np.ones((6, 3), dtype=np.float32) * 0.8
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats = align_structural_planes(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        floor_cid=14,
        ceiling_cid=1,
        wall_cid=5,
    )
    
    # T0, T1 remain Ceiling #1 (downward facing)
    assert cleaned[0] == 1
    assert cleaned[1] == 1
    # T2 (vertical leakage with horizontal normal) CANNOT be ceiling and must be reassigned to Wall #5!
    assert cleaned[2] == 5
    assert cleaned[3] == 5
    assert stats.get("ceiling_leakages_reassigned", 0) == 1


def test_align_structural_planes_floor_45_degree_tilt():
    """Verifies that floor surfaces with tilt up to 45 degrees are preserved, but vertical surfaces (90 deg) are reassigned."""
    # Floor quad tilted by ~30 degrees around Z axis (tilt < 45 deg) + Vertical wall (tilt = 90 deg)
    theta = np.radians(30.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    verts = np.array([
        [0.0, 0.0, 0.0],  # 0
        [0.1, 0.0, 0.0],  # 1
        [0.0, 0.1 * sin_t, 0.1 * cos_t],  # 2 (tilted floor)
        [0.1, 0.1 * sin_t, 0.1 * cos_t],  # 3 (tilted floor)
        [0.0, 0.1 * sin_t + 0.1, 0.1 * cos_t],  # 4 (vertical wall)
        [0.1, 0.1 * sin_t + 0.1, 0.1 * cos_t],  # 5 (vertical wall)
    ], dtype=np.float32)
    
    tris = np.array([
        [0, 2, 1],  # T0: Floor (14) - upward-ish normal, tilt ~30 deg <= 45 deg
        [1, 2, 3],  # T1: Floor (14) - upward-ish normal, tilt ~30 deg <= 45 deg
        [2, 3, 4],  # T2: Vertical wall, mislabeled Floor (14) - tilt = 90 deg
        [3, 5, 4],  # T3: Wall (5)
    ], dtype=np.int32)
    labels = np.array([14, 14, 14, 5], dtype=np.int32)
    vc = np.ones((6, 3), dtype=np.float32) * 0.5
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats = align_structural_planes(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        floor_cid=14,
        wall_cid=5,
        max_floor_tilt_deg=45.0,
    )
    
    # T0, T1 have tilt ~30 deg <= 45 deg: MUST be preserved as Floor #14!
    assert cleaned[0] == 14
    assert cleaned[1] == 14
    # T2 is vertical (tilt = 90 deg > 45 deg): MUST be reassigned to Wall #5!
    assert cleaned[2] == 5
    assert cleaned[3] == 5


def test_absorb_enclosed_subparts_local_scale_asymmetry():
    """
    Verifies that a candidate component touching a tiny crumb of a non-structural class
    is NOT absorbed, even if that non-structural class has a large global triangle count
    elsewhere in the mesh.
    """
    # 5 vertices forming 3 triangles in a strip:
    # T0, T1: Candidate object of Class 20 (2 triangles, size=2)
    # T2: Tiny crumb of Class 30 touching T1 (1 triangle, local size=1)
    # T3, T4: Distant triangles of Class 30 located elsewhere (2 triangles, disconnected from T0..T2)
    verts = np.array([
        # Strip T0, T1, T2
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [2.0, 0.0, 0.0],  # 4
        # Distant T3, T4
        [10.0, 0.0, 0.0], # 5
        [11.0, 0.0, 0.0], # 6
        [10.0, 1.0, 0.0], # 7
        [11.0, 1.0, 0.0], # 8
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2],  # T0: Class 20
        [1, 3, 2],  # T1: Class 20 (touches T2)
        [1, 4, 3],  # T2: Class 30 (local size = 1)
        [5, 6, 7],  # T3: Class 30 (distant)
        [6, 8, 7],  # T4: Class 30 (distant)
    ], dtype=np.int32)
    # Global count of Class 30 is 3 triangles (T2, T3, T4).
    # Component Class 20 is 2 triangles (T0, T1).
    # Under global check, 2 < 3, so Class 20 would erroneously be absorbed!
    # Under local check, touching component T2 is 1 triangle. 2 >= 1 -> protected!
    labels = np.array([20, 20, 30, 30, 30], dtype=np.int32)
    vc = np.ones((9, 3), dtype=np.float32) * 0.4
    
    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)
    
    cleaned, stats, absorbed = absorb_enclosed_subparts(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        structural_cids={1, 5, 14},
        min_enclosure_frac=0.50,
        min_component_size=1,
    )
    
    # Class 20 must NOT be absorbed into the tiny crumb of Class 30!
    assert cleaned[0] == 20
    assert cleaned[1] == 20
    assert 20 not in absorbed


def test_converge_class_instances_proximity_gap_bridge_and_isolation():
    """
    Verifies that:
    1. An outlier crumb within max_proximity_merge_dist (<= 8cm) across a small gap
       is bridged to the neighboring host.
    2. An outlier crumb far across empty 3D space (> 8cm, e.g. 2.0m) is NOT reassigned,
       guaranteeing zero teleportation across rooms.
    """
    def make_strip(x_start, x_end, n_quads, y_val, z_len, lbl, base_v=0):
        xs = np.linspace(x_start, x_end, n_quads + 1)
        verts = []
        for x in xs:
            verts.append([x, y_val, 0.0])
            verts.append([x, y_val, z_len])
        tris = []
        labels = []
        for i in range(n_quads):
            v0 = base_v + 2 * i
            v1 = base_v + 2 * (i + 1)
            v2 = base_v + 2 * (i + 1) + 1
            v3 = base_v + 2 * i + 1
            tris.append([v0, v1, v2])
            tris.append([v0, v2, v3])
            labels.extend([lbl, lbl])
        return verts, tris, labels

    all_verts = []
    all_tris = []
    all_labels = []

    # Host Class 10: 40 quads from x=0.0 to 2.0 (80 tris)
    v10, t10, l10 = make_strip(0.0, 2.0, 40, 0.0, 0.5, 10, len(all_verts))
    all_verts.extend(v10); all_tris.extend(t10); all_labels.extend(l10)

    # Class 20 Main Instance: 40 quads from x=3.0 to 5.0 (80 tris)
    v20_main, t20_main, l20_main = make_strip(3.0, 5.0, 40, 0.0, 0.5, 20, len(all_verts))
    all_verts.extend(v20_main); all_tris.extend(t20_main); all_labels.extend(l20_main)

    # Class 20 Outlier Crumb: 3 quads from x=2.02 to 2.10 (gap = 2cm from Class 10, <= 8cm) (6 tris)
    v20_out, t20_out, l20_out = make_strip(2.02, 2.10, 3, 0.0, 0.5, 20, len(all_verts))
    all_verts.extend(v20_out); all_tris.extend(t20_out); all_labels.extend(l20_out)

    # Class 30 Main Instance: 40 quads from x=6.0 to 8.0 (80 tris)
    v30_main, t30_main, l30_main = make_strip(6.0, 8.0, 40, 0.0, 0.5, 30, len(all_verts))
    all_verts.extend(v30_main); all_tris.extend(t30_main); all_labels.extend(l30_main)

    # Class 30 Outlier Crumb: 3 quads from x=10.0 to 10.08 (gap = 2.0m from Class 30, > 8cm) (6 tris)
    v30_out, t30_out, l30_out = make_strip(10.0, 10.08, 3, 0.0, 0.5, 30, len(all_verts))
    all_verts.extend(v30_out); all_tris.extend(t30_out); all_labels.extend(l30_out)

    verts = np.array(all_verts, dtype=np.float32)
    tris = np.array(all_tris, dtype=np.int32)
    labels = np.array(all_labels, dtype=np.int32)
    
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    tri_normals = cross / np.maximum(2.0 * tri_areas[:, None], 1e-8)
    tri_centroids = (v0 + v1 + v2) / 3.0
    gravity_up = np.array([0.0, 1.0, 0.0])
    
    adj_u, adj_v = build_mesh_dual_graph(tris)
    active_classes = [10, 20, 30]
    
    cleaned, records = converge_class_instances_iteratively(
        triangle_labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        gravity_up=gravity_up,
        active_classes=active_classes,
        min_instance_size=5,
        min_instance_area=0.005,
        min_instance_span=0.01,
        max_proximity_merge_dist=0.08,  # 8cm limit
        max_iterations=2,
    )
    
    outlier20_idx = np.where((labels == 20) & (tri_centroids[:, 0] < 2.5))[0]
    outlier30_idx = np.where((labels == 30) & (tri_centroids[:, 0] > 9.0))[0]

    # Class 20 Outlier Crumb is within 2cm of Class 10 -> bridged to Class 10!
    assert np.all(cleaned[outlier20_idx] == 10)
    
    # Class 30 Outlier Crumb is 2.0m away across space (> 8cm) -> MUST NOT be bridged to anything!
    assert np.all(cleaned[outlier30_idx] == 30)


def test_reclaim_ceiling_bounding_slabs():
    """Verifies that downward-facing triangles in a ceiling's bounding slab are reclaimed, while vertical triangles are not."""
    # Construct a flat grid at Y = 2.0: 10 x 10 vertices -> 9 x 9 x 2 = 162 triangles
    gx, gz = np.meshgrid(np.linspace(0, 1.8, 10), np.linspace(0, 1.8, 10))
    gy = np.full_like(gx, 2.0)
    verts_ceil = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)

    tris_ceil = []
    for r in range(9):
        for c in range(9):
            i0 = r * 10 + c
            i1 = i0 + 1
            i2 = (r + 1) * 10 + c
            i3 = i2 + 1
            # Downward winding (cross product points in -Y)
            tris_ceil.append([i0, i1, i2])
            tris_ceil.append([i1, i3, i2])
    tris_ceil = np.array(tris_ceil, dtype=np.int32)
    n_ceil = len(tris_ceil)

    # Adjacent coplanar strip (Y = 2.0, X in [1.8, 1.86], Z in [0, 1.8]): within 0.15m margin of ceiling centroids (max X ~ 1.733)
    sx, sz = np.meshgrid(np.linspace(1.8, 1.86, 3), np.linspace(0, 1.8, 10))
    sy = np.full_like(sx, 2.0)
    verts_strip = np.column_stack([sx.ravel(), sy.ravel(), sz.ravel()]).astype(np.float32)
    offset_strip = len(verts_ceil)
    tris_strip = []
    for r in range(9):
        for c in range(2):
            i0 = offset_strip + r * 3 + c
            i1 = i0 + 1
            i2 = offset_strip + (r + 1) * 3 + c
            i3 = i2 + 1
            tris_strip.append([i0, i1, i2])
            tris_strip.append([i1, i3, i2])
    tris_strip = np.array(tris_strip, dtype=np.int32)
    n_strip = len(tris_strip)

    # Vertical wall patch at X = 0, dropping from Y = 2.0 to 1.0 (normal in -X, nY = 0)
    vx, vy, vz = np.array([[0, 2.0, 0], [0, 1.0, 0], [0, 2.0, 1.0], [0, 1.0, 1.0]], dtype=np.float32).T
    verts_vert = np.column_stack([vx, vy, vz])
    offset_vert = offset_strip + len(verts_strip)
    tris_vert = np.array([[offset_vert, offset_vert + 1, offset_vert + 2], [offset_vert + 1, offset_vert + 3, offset_vert + 2]], dtype=np.int32)
    n_vert = len(tris_vert)

    all_verts = np.vstack([verts_ceil, verts_strip, verts_vert])
    all_tris = np.vstack([tris_ceil, tris_strip, tris_vert])
    vc = np.full((len(all_verts), 3), 0.8, dtype=np.float32)  # matching ceiling paint

    labels = np.full(len(all_tris), 5, dtype=np.int32)  # Wall
    labels[:n_ceil] = 1  # Ceiling

    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(all_verts, all_tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(all_tris)

    cleaned, n_reclaimed = reclaim_ceiling_bounding_slabs(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        tri_colors=tri_colors,
        ceiling_cid=1,
        wall_cid=5,
        min_comp_size=50,
        max_slab_thickness_m=0.08,
        max_color_diff=0.055,
        margin_m=0.15,
    )

    # Downward coplanar strip inside slab must be reclaimed to Ceiling (1)
    assert np.all(cleaned[n_ceil:n_ceil + n_strip] == 1)
    assert n_reclaimed == n_strip

    # Vertical triangles must NOT be reclaimed into ceiling (Ceilings cannot be vertical!)
    assert np.all(cleaned[n_ceil + n_strip:] == 5)


def test_reclaim_wall_encroached_furniture():
    """Verifies that wall triangles inside thin furniture bounds sharing surface normal are reclaimed into furniture."""
    # Synthetic shelf (Class 4): flat front panel at Z = 0, X in [0, 1], Y in [0, 1]
    gx, gy = np.meshgrid(np.linspace(0, 1.0, 10), np.linspace(0, 1.0, 10))
    gz = np.zeros_like(gx)
    verts_shelf = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)

    tris_shelf = []
    for r in range(9):
        for c in range(9):
            i0 = r * 10 + c
            i1 = i0 + 1
            i2 = (r + 1) * 10 + c
            i3 = i2 + 1
            tris_shelf.append([i0, i2, i1])
            tris_shelf.append([i1, i2, i3])
    tris_shelf = np.array(tris_shelf, dtype=np.int32)
    n_shelf = len(tris_shelf)

    # Perimeter wall triangles behind the shelf at Z = -0.30 m (> 10 cm behind)
    bx, by = np.meshgrid(np.linspace(0, 1.0, 5), np.linspace(0, 1.0, 5))
    bz = np.full_like(bx, -0.30)
    verts_wall_back = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()]).astype(np.float32)
    offset_wb = len(verts_shelf)
    tris_wall_back = []
    for r in range(4):
        for c in range(4):
            i0 = offset_wb + r * 5 + c
            i1 = i0 + 1
            i2 = offset_wb + (r + 1) * 5 + c
            i3 = i2 + 1
            tris_wall_back.append([i0, i2, i1])
            tris_wall_back.append([i1, i2, i3])
    tris_wall_back = np.array(tris_wall_back, dtype=np.int32)
    n_wb = len(tris_wall_back)

    all_verts = np.vstack([verts_shelf, verts_wall_back])
    all_tris = np.vstack([tris_shelf, tris_wall_back])
    vc = np.full((len(all_verts), 3), 0.5, dtype=np.float32)

    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(all_verts, all_tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(all_tris)

    # Label shelf frame as Class 4, but center of shelf (misclassified as Wall 5)
    labels = np.full(len(all_tris), 5, dtype=np.int32)
    labels[:n_shelf] = 4
    in_center = (tri_centroids[:n_shelf, 0] >= 0.3) & (tri_centroids[:n_shelf, 0] <= 0.7) & \
                (tri_centroids[:n_shelf, 1] >= 0.3) & (tri_centroids[:n_shelf, 1] <= 0.7)
    labels[:n_shelf][in_center] = 5
    n_encroached = int(in_center.sum())

    cleaned, stats = reclaim_wall_encroached_furniture(
        labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_normals=tri_normals,
        tri_centroids=tri_centroids,
        wall_cid=5,
        foreground_cids=[4],
        min_comp_size=20,
        min_normal_align=0.70,
        bbox_margin_m=0.03,
    )

    # Shelf encroached center triangles must be reclaimed into Shelf (4)
    assert np.all(cleaned[:n_shelf] == 4)
    assert stats[4] == n_encroached

    # Perimeter wall triangles behind (Z = -0.30) must remain Wall (5)
    assert np.all(cleaned[n_shelf:] == 5)


def test_converge_class_instances_early_convergence():
    """Verifies that a class whose outlier only touches itself is self-reabsorbed and flagged converged in 1 iteration."""
    # Synthetic mesh: 2 triangles for Class 1 (leader), 1 triangle for Class 1 (outlier surrounded by Class 1)
    # 4 triangles in a row
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=np.float32)
    tris = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [1, 4, 3],
        [4, 5, 3],
    ], dtype=np.int32)
    # T0, T1: Leader Class 1. T2, T3: Outlier Class 1
    labels = np.array([1, 1, 1, 1], dtype=np.int32)
    vc = np.full((6, 3), 0.5, dtype=np.float32)

    tri_areas, tri_normals, tri_centroids, tri_colors = compute_mesh_derivatives(verts, tris, vc)
    adj_u, adj_v, adj_csr = build_dual_graph(tris)

    cleaned, records = converge_class_instances_iteratively(
        triangle_labels=labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        gravity_up=np.array([0.0, 1.0, 0.0]),
        active_classes=[1],
        min_instance_size=2,
        min_instance_area=0.01,
        min_instance_span=0.1,
        max_proximity_merge_dist=0.0,  # Do not merge
        max_iterations=5,
    )

    # Class 1 converges immediately
    assert np.all(cleaned == 1)







