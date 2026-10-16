from pathlib import Path
import sys

import numpy as np
import pytest

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "06c_semantic_segmentation"))
from step2_project_masks_to_3d import (
    absorb_isolated_islands,
    build_dual_mesh_graph,
    compute_mask_distance_weights,
    export_colored_ply,
    get_class_color,
    load_colmap_triangulated_points,
    render_vector_pdf_matrix,
)


def test_get_class_color_deterministic():
    # Test class color consistency
    bgr1, rgb1 = get_class_color(1)
    bgr1_repeat, rgb1_repeat = get_class_color(1)
    assert bgr1 == bgr1_repeat
    assert rgb1 == rgb1_repeat

    # Background / 0 class is neutral gray
    bgr0, rgb0 = get_class_color(0)
    assert bgr0 == (128, 128, 128)
    assert rgb0 == (0.5, 0.5, 0.5)

    # Different classes yield distinct colors
    _, rgb2 = get_class_color(2)
    assert rgb1 != rgb2


def test_load_colmap_triangulated_points(tmp_path):
    images_txt = tmp_path / "images.txt"
    # Create sample COLMAP images.txt
    content = (
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_00001.jpg\n"
        "100.0 200.0 1 150.0 250.0 -1 300.0 400.0 5\n"
        "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_00002.jpg\n"
        "50.0 50.0 -1 60.0 60.0 -1\n"
    )
    images_txt.write_text(content)

    pts = load_colmap_triangulated_points(images_txt)
    assert "frame_00001.jpg" in pts
    assert pts["frame_00001.jpg"] == 2  # 2 points with POINT3D_ID != -1
    assert "frame_00002.jpg" in pts
    assert pts["frame_00002.jpg"] == 0


def test_compute_mask_distance_weights():
    # 10x10 label map with centered 6x6 square of class 5
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:8, 2:8] = 5

    w = compute_mask_distance_weights(labels)
    assert w.shape == (10, 10)
    assert (w[labels == 0] == 0.0).all()  # background is zero
    assert (w[labels == 5] > 0.0).all()   # inside mask is positive
    # Center should have higher weight than boundary
    assert w[4, 4] >= w[2, 2]


def test_render_vector_pdf_matrix(tmp_path):
    pdf_path = tmp_path / "test_matrix.pdf"
    png_path = tmp_path / "test_matrix.png"

    class_ids = [1, 2, 5]
    contention_counts = np.array([
        [0, 50, 200],
        [50, 0, 120],
        [200, 120, 0],
    ])
    uncontended_counts = np.array([1000, 500, 3000])
    total_assigned_counts = np.array([1250, 670, 3320])

    render_vector_pdf_matrix(
        pdf_path=pdf_path,
        png_path=png_path,
        class_ids=class_ids,
        contention_counts=contention_counts,
        uncontended_counts=uncontended_counts,
        total_assigned_counts=total_assigned_counts,
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000
    assert png_path.exists()
    assert png_path.stat().st_size > 1000


def test_pure_white_contention_logic():
    # Simulate contention classification
    # 4 triangles, 3 classes
    # tri 0: voted only by class 1 -> uncontended (class 1 color)
    # tri 1: voted by class 1 and class 2 -> contended (pure white [255, 255, 255])
    # tri 2: voted only by class 2 -> uncontended (class 2 color)
    # tri 3: unvoted -> neutral gray [128, 128, 128]
    weights = np.array([
        [1.5, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 3.2, 0.0],
        [0.0, 0.0, 0.0],
    ])
    vote_counts = (weights > 0).sum(axis=1)
    unvoted = vote_counts == 0
    single = vote_counts == 1
    contended = vote_counts > 1

    colors = np.zeros((4, 3), dtype=np.uint8)
    colors[unvoted] = [128, 128, 128]
    colors[contended] = [255, 255, 255]

    for idx in np.flatnonzero(single):
        cid = [1, 2, 5][weights[idx].argmax()]
        _, rgb = get_class_color(cid)
        colors[idx] = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]

    # Triangle 1 must be pure white [255, 255, 255]
    assert list(colors[1]) == [255, 255, 255]
    # Triangle 3 must be gray [128, 128, 128]
    assert list(colors[3]) == [128, 128, 128]
    # Triangle 0 must be class 1 color
    _, rgb1 = get_class_color(1)
    assert list(colors[0]) == [int(rgb1[0] * 255), int(rgb1[1] * 255), int(rgb1[2] * 255)]


def test_export_colored_ply_averages_incident_triangle_colors(tmp_path):
    import open3d as o3d

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 2, 2]], float)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], np.int32)
    colors = np.array([[10, 20, 30], [90, 80, 70]], np.uint8)
    path = tmp_path / "colored.ply"

    export_colored_ply(path, vertices, triangles, colors)
    mesh = o3d.io.read_triangle_mesh(str(path))

    np.testing.assert_array_equal(np.asarray(mesh.vertex_colors) * 255, [
        [10, 20, 30], [50, 50, 50], [50, 50, 50], [90, 80, 70], [0, 0, 0]
    ])


def test_dual_mesh_graph_and_island_absorption():
    # 5 triangles sharing edges in a fan:
    # tri 0: [0, 1, 2], tri 1: [0, 2, 3], tri 2: [0, 3, 4], tri 3: [0, 4, 5], tri 4: [0, 5, 6]
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
    ], dtype=np.int32)

    adj_u, adj_v = build_dual_mesh_graph(triangles)
    assert len(adj_u) == 8  # 4 shared edges x 2 directions = 8 directed edges

    # Labels: tri 0, 1 are class 0 (wall). Tri 2 is an isolated triangle of class 1 (speckle).
    # Tri 3, 4 are class 0 (wall).
    labels = np.array([0, 0, 1, 0, 0], dtype=np.int32)
    cleaned, n_iso, n_island = absorb_isolated_islands(labels, adj_u, adj_v, num_classes=2, min_island_size=2)

    assert n_iso == 1  # tri 2 was isolated
    assert n_island == 1
    assert cleaned[2] == 0  # tri 2 was absorbed by surrounding class 0!

