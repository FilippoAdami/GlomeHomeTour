import pytest
import numpy as np
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "06c_semantic_segmentation"))

from detect_structural_classes import (
    estimate_gravity_and_scene_bands,
    decompose_class_into_connected_pieces,
    analyze_class_geometry,
    classify_structural_classes,
    build_mesh_dual_graph,
    absorb_outlier_floaters,
    absorb_structural_floaters,
    fill_unassigned_triangles,
    extract_instances,
    absorb_class_outliers,
    build_instance_badge_mesh,
    embed_class_id_badges_on_mesh,
)


def create_box_room_mesh(
    width=4.0, length=5.0, height=3.0, grid_res=0.5
):
    """
    Creates a simple synthetic box room mesh with:
    - Floor at y = 0
    - Ceiling at y = height
    - 4 vertical walls
    """
    vertices = []
    triangles = []
    labels = []  # 1: floor, 2: ceiling, 3: wall_x, 4: wall_z

    def add_quad(p0, p1, p2, p3, label):
        idx = len(vertices)
        vertices.extend([p0, p1, p2, p3])
        triangles.append([idx, idx + 1, idx + 2])
        triangles.append([idx, idx + 2, idx + 3])
        labels.extend([label, label])

    # Floor (y = 0, normal = [0, 1, 0] facing up)
    for x in np.arange(-width/2, width/2, grid_res):
        for z in np.arange(-length/2, length/2, grid_res):
            add_quad(
                [x, 0, z], [x, 0, z + grid_res],
                [x + grid_res, 0, z + grid_res], [x + grid_res, 0, z],
                1
            )

    # Ceiling (y = height, normal = [0, -1, 0] facing down)
    for x in np.arange(-width/2, width/2, grid_res):
        for z in np.arange(-length/2, length/2, grid_res):
            add_quad(
                [x, height, z], [x + grid_res, height, z],
                [x + grid_res, height, z + grid_res], [x, height, z + grid_res],
                2
            )

    # Wall X-min (x = -width/2, normal = [1, 0, 0])
    for y in np.arange(0, height, grid_res):
        for z in np.arange(-length/2, length/2, grid_res):
            add_quad(
                [-width/2, y, z], [-width/2, y + grid_res, z],
                [-width/2, y + grid_res, z + grid_res], [-width/2, y, z + grid_res],
                3
            )

    # Wall X-max (x = width/2, normal = [-1, 0, 0])
    for y in np.arange(0, height, grid_res):
        for z in np.arange(-length/2, length/2, grid_res):
            add_quad(
                [width/2, y, z + grid_res], [width/2, y + grid_res, z + grid_res],
                [width/2, y + grid_res, z], [width/2, y, z],
                3
            )

    # Wall Z-min (z = -length/2, normal = [0, 0, 1])
    for x in np.arange(-width/2, width/2, grid_res):
        for y in np.arange(0, height, grid_res):
            add_quad(
                [x + grid_res, y, -length/2], [x + grid_res, y + grid_res, -length/2],
                [x, y + grid_res, -length/2], [x, y, -length/2],
                4
            )

    v_arr = np.array(vertices, dtype=np.float32)
    t_arr = np.array(triangles, dtype=np.int32)
    l_arr = np.array(labels, dtype=np.int32)

    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v_arr), o3d.utility.Vector3iVector(t_arr))
    m = m.merge_close_vertices(0.001)
    return (
        np.asarray(m.vertices, dtype=np.float32),
        np.asarray(m.triangles, dtype=np.int32),
        l_arr,
    )


def test_gravity_and_elevation_bands():
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    gravity_up, scene_bands = estimate_gravity_and_scene_bands(verts, tris)

    # Up vector should align with Y axis [0, 1, 0]
    assert abs(gravity_up[1]) > 0.95
    assert scene_bands["h_min"] == pytest.approx(0.0, abs=0.1)
    assert scene_bands["h_max"] == pytest.approx(3.0, abs=0.1)
    assert scene_bands["h_ceiling_band"] >= 1.9


def test_basic_room_structural_classification():
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    gravity_up, scene_bands = estimate_gravity_and_scene_bands(verts, tris)

    features = analyze_class_geometry(verts, tris, lbls, [1, 2, 3, 4], gravity_up, scene_bands)
    classifications = classify_structural_classes(features, scene_bands)

    assert classifications[1]["structural_role"] == "floor"
    assert classifications[2]["structural_role"] == "ceiling"
    assert classifications[3]["structural_role"] == "wall"
    assert classifications[4]["structural_role"] == "wall"


def test_sottoscala_classified_as_wall():
    """
    A tilted surface under the stairs (sottoscala) with downward-slanted normals
    must be classified as WALL (sottoscala) because it is below the ceiling band
    and contiguous with the wall.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    
    # Add a tilted under-stair surface (sottoscala) at mid-height y in [0.8, 1.4]
    # normal pointing tilted downwards: v1 x v2 = [0.6, -1.0, 0]
    tilted_v = [
        [-1.0, 0.8, -1.0], [0.0, 1.4, -1.0],
        [0.0, 1.4, 0.0], [-1.0, 0.8, 0.0]
    ]
    idx = len(verts)
    new_verts = np.vstack([verts, tilted_v])
    new_tris = np.vstack([tris, [[idx, idx + 1, idx + 2], [idx, idx + 2, idx + 3]]])
    new_lbls = np.concatenate([lbls, [5, 5]])  # Class 5 = sottoscala

    gravity_up, scene_bands = estimate_gravity_and_scene_bands(new_verts, new_tris)
    features = analyze_class_geometry(new_verts, new_tris, new_lbls, [1, 2, 3, 4, 5], gravity_up, scene_bands)
    classifications = classify_structural_classes(features, scene_bands)

    # Class 5 must NOT be ceiling; it must be classified as wall (sottoscala)
    assert classifications[5]["structural_role"] in ["wall", "wall_sottoscala"]


def test_multi_level_floors():
    """
    Two floor classes at different elevations (Ground Floor 0 and Mezzanine Floor 1)
    must both be recognized as floors with their respective levels.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=4.0)

    # Add mezzanine floor slab at y = 2.0 (class 6), normal facing up [0, 1, 0]
    mezz_v = [
        [-1.5, 2.0, -2.0], [-1.5, 2.0, 0.0],
        [1.5, 2.0, 0.0], [1.5, 2.0, -2.0]
    ]
    idx = len(verts)
    new_verts = np.vstack([verts, mezz_v])
    new_tris = np.vstack([tris, [[idx, idx + 1, idx + 2], [idx, idx + 2, idx + 3]]])
    new_lbls = np.concatenate([lbls, [6, 6]])  # Class 6 = mezzanine floor

    gravity_up, scene_bands = estimate_gravity_and_scene_bands(new_verts, new_tris)
    features = analyze_class_geometry(new_verts, new_tris, new_lbls, [1, 2, 3, 4, 6], gravity_up, scene_bands)
    classifications = classify_structural_classes(features, scene_bands)

    assert classifications[1]["structural_role"] == "floor"
    assert classifications[6]["structural_role"] == "floor"
    assert classifications[1]["floor_level"] == 0
    assert classifications[6]["floor_level"] == 1


def test_stairs_over_floor_disambiguation():
    """
    Stairs consisting of sequential stepped horizontal treads ascending above a floor
    must be classified as STAIRS, while the continuous floor slab remains FLOOR.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)

    # Add 4 stair treads at y = 0.2, 0.4, 0.6, 0.8 (class 7), normals facing up
    stair_v = []
    stair_t = []
    stair_l = []
    start_idx = len(verts)
    for i, y_step in enumerate([0.2, 0.4, 0.6, 0.8]):
        z0 = -1.0 + i * 0.3
        z1 = z0 + 0.3
        idx = start_idx + len(stair_v)
        stair_v.extend([
            [-0.5, y_step, z0], [-0.5, y_step, z1],
            [0.5, y_step, z1], [0.5, y_step, z0]
        ])
        stair_t.append([idx, idx + 1, idx + 2])
        stair_t.append([idx, idx + 2, idx + 3])
        stair_l.extend([7, 7])

    new_verts = np.vstack([verts, stair_v])
    new_tris = np.vstack([tris, stair_t])
    new_lbls = np.concatenate([lbls, stair_l])

    gravity_up, scene_bands = estimate_gravity_and_scene_bands(new_verts, new_tris)
    features = analyze_class_geometry(new_verts, new_tris, new_lbls, [1, 2, 3, 4, 7], gravity_up, scene_bands)
    classifications = classify_structural_classes(features, scene_bands)

    assert classifications[1]["structural_role"] == "floor"
    assert classifications[7]["structural_role"] == "stairs"


def test_bookshelf_shelves_not_classified_as_stairs():
    """
    Vertically stacked shelves of a bookcase (one directly above the other,
    with no horizontal span/run) must NOT be classified as stairs.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)

    # 4 shelves directly stacked at x=0, z=0, at heights y = 0.3, 0.6, 0.9, 1.2
    book_v = []
    book_t = []
    book_l = []
    start_idx = len(verts)
    for i, y_shelf in enumerate([0.3, 0.6, 0.9, 1.2]):
        idx = start_idx + len(book_v)
        book_v.extend([
            [-0.4, y_shelf, -0.2], [-0.4, y_shelf, 0.2],
            [0.4, y_shelf, 0.2], [0.4, y_shelf, -0.2]
        ])
        book_t.append([idx, idx + 1, idx + 2])
        book_t.append([idx, idx + 2, idx + 3])
        book_l.extend([8, 8])  # Class 8 = bookcase shelves

    new_verts = np.vstack([verts, book_v])
    new_tris = np.vstack([tris, book_t])
    new_lbls = np.concatenate([lbls, book_l])

    gravity_up, scene_bands = estimate_gravity_and_scene_bands(new_verts, new_tris)
    features = analyze_class_geometry(new_verts, new_tris, new_lbls, [1, 2, 3, 4, 8], gravity_up, scene_bands)
    classifications = classify_structural_classes(features, scene_bands)

    # Bookcase must NOT be stairs!
    assert classifications[8]["structural_role"] != "stairs"
    assert classifications[8]["structural_role"] in ["furniture_horizontal", "furniture_vertical", "object"]


def test_floater_absorption():
    """
    A single triangle or small cluster (< 30 triangles) of class B surrounded
    by class A must be absorbed into class A.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    adj_u, adj_v = build_mesh_dual_graph(tris)

    # Pick 2 floor triangles (class 1) and make them class 99 (a small floater island)
    test_labels = lbls.copy()
    test_labels[0] = 99
    test_labels[1] = 99

    cleaned, absorbed_cnt = absorb_outlier_floaters(test_labels, adj_u, adj_v, [1, 2, 3, 4, 99], min_island_size=30)
    assert absorbed_cnt == 2
    assert cleaned[0] == 1
    assert cleaned[1] == 1


def test_extract_instances():
    """
    Connected components >= 15 triangles are extracted as discrete instances.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    adj_u, adj_v = build_mesh_dual_graph(tris)

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    norms = cross / np.maximum(2.0 * areas[:, None], 1e-8)
    cents = (v0 + v1 + v2) / 3.0
    gravity_up = np.array([0, 1, 0], dtype=np.float32)

    instances, counts = extract_instances(
        lbls, [1, 2, 3, 4], adj_u, adj_v, areas, cents, norms, gravity_up, min_instance_size=15
    )
    assert len(instances) >= 4
    for inst in instances.values():
        assert inst["triangle_count"] >= 15
        assert inst["area_m2"] > 0


def test_fill_unassigned_triangles_100_percent():
    """
    Unassigned triangles (-1) are filled via multi-source BFS region growing
    to achieve 100% coverage with 0 unassigned triangles.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    adj_u, adj_v = build_mesh_dual_graph(tris)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cents = (v0 + v1 + v2) / 3.0

    # Introduce a strip of unassigned (-1) triangles in the interior of the floor (class 1)
    test_labels = lbls.copy()
    test_labels[40:50] = -1
    assert np.count_nonzero(test_labels < 0) == 10

    filled = fill_unassigned_triangles(test_labels, adj_u, adj_v, cents)
    assert np.count_nonzero(filled < 0) == 0
    # The filled triangles should be class 1 (surrounding floor)
    assert np.all(filled[40:50] == 1)


def test_absorb_structural_floaters():
    """
    Disconnected small islands of structural classes (e.g. wall class 3)
    are absorbed into the surrounding class.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    adj_u, adj_v = build_mesh_dual_graph(tris)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Put a small island of class 3 (wall) in the middle of class 1 (floor)
    test_labels = lbls.copy()
    test_labels[40] = 3
    test_labels[41] = 3

    cleaned, absorbed = absorb_structural_floaters(
        test_labels, adj_u, adj_v, tri_areas, structural_classes=[3], max_island_tris=10, max_island_area=2.0
    )
    assert absorbed == 2
    assert cleaned[40] == 1
    assert cleaned[41] == 1


def test_absorb_class_outliers():
    """
    A small outlier instance of a multi-instance class (e.g. wall fragment < 0.25 * median)
    touching the surrounding class is identified and absorbed into the enclosing class.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    adj_u, adj_v = build_mesh_dual_graph(tris)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    tri_centroids = (v0 + v1 + v2) / 3.0
    tri_normals = cross / np.maximum(2.0 * tri_areas[:, None], 1e-8)
    gravity_up = np.array([0, 1, 0], dtype=np.float32)

    # Class 3 has 160 triangles. Let us create a 16-triangle island of class 3 on the floor (class 1)
    test_labels = lbls.copy()
    test_labels[20:36] = 3  # 16 triangles of class 3 surrounded by class 1 floor

    refined, records = absorb_class_outliers(
        triangle_labels=test_labels,
        adj_u=adj_u,
        adj_v=adj_v,
        tri_areas=tri_areas,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        gravity_up=gravity_up,
        active_classes=[1, 2, 3, 4],
        min_instance_size=15,
    )

    # The 16-triangle outlier of class 3 should be absorbed into class 1 (floor)
    assert len(records) == 1
    assert records[0]["original_class"] == 3
    assert records[0]["assigned_class"] == 1
    assert records[0]["triangle_count"] == 16
    assert np.all(refined[20:36] == 1)


def test_build_instance_badge_mesh():
    """
    Verifies that build_instance_badge_mesh produces valid 3D geometry
    with faces, vertices, stroke cylinders, and proper colors.
    """
    origin = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    verts, faces, colors = build_instance_badge_mesh(
        origin=origin,
        forward=forward,
        scale=0.08,
        text="#19",
        border_color_rgb=(255, 100, 50),
        text_color_rgb=(255, 255, 255),
        plate_color_rgb=(20, 20, 25),
        include_plate=True,
    )
    assert len(verts) > 50
    assert len(faces) > 50
    assert len(colors) == len(verts)
    assert faces.max() < len(verts)
    assert faces.min() >= 0
    # Check that white stroke text vertices and plate vertices are present
    assert np.any(np.all(colors == [255, 255, 255], axis=1))
    assert np.any(np.all(colors == [20, 20, 25], axis=1))


def test_embed_class_id_badges_on_mesh():
    """
    Verifies that embed_class_id_badges_on_mesh embeds badges onto instances,
    increasing vertices and triangles, and retaining colors.
    """
    verts, tris, lbls = create_box_room_mesh(width=4.0, length=5.0, height=3.0)
    adj_u, adj_v = build_mesh_dual_graph(tris)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross, axis=1)
    tri_centroids = (v0 + v1 + v2) / 3.0
    tri_normals = cross / np.maximum(2.0 * tri_areas[:, None], 1e-8)
    gravity_up = np.array([0, 1, 0], dtype=np.float32)

    instances, counts = extract_instances(
        lbls, [1, 2, 3, 4], adj_u, adj_v, tri_areas, tri_centroids, tri_normals, gravity_up, min_instance_size=15
    )
    assert len(instances) >= 4

    base_colors = np.full((len(verts), 3), 128, dtype=np.uint8)
    comb_v, comb_f, comb_c, n_badges = embed_class_id_badges_on_mesh(
        vertices=verts,
        triangles=tris,
        vertex_colors_rgb=base_colors,
        instances=instances,
        tri_centroids=tri_centroids,
        tri_normals=tri_normals,
        tri_areas=tri_areas,
        gravity_up=gravity_up,
    )
    assert n_badges >= 4
    assert len(comb_v) > len(verts)
    assert len(comb_f) > len(tris)
    assert len(comb_c) == len(comb_v)




