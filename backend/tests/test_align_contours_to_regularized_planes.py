import importlib.util
from pathlib import Path
import sys

import numpy as np
import trimesh


SCRIPT = (Path(__file__).resolve().parents[1] / "07_alignment"
          / "temporary_object_contact_contours" / "align_contours_to_regularized_planes.py")
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("align_contours_to_regularized_planes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plane_mesh(x=0.0, y=0.0, z=0.0):
    return trimesh.Trimesh(
        vertices=np.array([[x, y, z], [x, y + 1, z], [x, y, z + 1]], dtype=float),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )


def test_regularized_source_prefers_pristine_mesh_before_applying_scan_transform(tmp_path):
    pristine = tmp_path / ".raw_unaligned/mesh_elaboration/complete.ply"
    current = tmp_path / "mesh_elaboration/complete.ply"
    pristine.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    pristine.touch()
    current.touch()

    assert MODULE.regularized_source(tmp_path) == pristine


def test_instance_output_path_drops_redundant_scan_prefix():
    assert MODULE.instance_relative_path("scan_1", "scan_1/class_006/instance_001.ply") == Path(
        "class_006/instance_001.ply")


def colored_object(x_values):
    vertices = []
    faces = []
    for x in x_values:
        offset = len(vertices)
        vertices.extend([[x, 0.1, 0.1], [x, 0.2, 0.1], [x, 0.1, 0.2]])
        faces.append([offset, offset + 1, offset + 2])
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(vertices), 1))
    return mesh


def test_candidates_are_axis_only_and_discard_plane_hits_over_15cm():
    scene = MODULE.axis_plane_scene(plane_mesh())
    mesh = colored_object([0.10, 0.20])

    candidates = MODULE.translation_candidates(mesh, np.array([True, True]), scene, set())

    assert candidates
    assert {item["axis"] for item in candidates} == {0}
    assert all(item["delta_m"] < 0 for item in candidates)
    assert all(item["distance_m"] <= MODULE.MAX_TRANSLATION_M for item in candidates)
    assert {item["face_index"] for item in candidates} == {0}


def test_bounds_check_considers_every_red_contour_triangle():
    mesh = colored_object([0.10, 0.30])

    assert not MODULE.red_vertices_stay_in_bounds(
        mesh, np.array([True, True]), np.array([-0.10, 0, 0]),
        np.array([[0, -1, -1], [0.15, 1, 1]], dtype=float),
    )


def test_object_gets_at_most_two_single_axis_moves_from_distinct_contours():
    scene = MODULE.axis_plane_scene(plane_mesh())
    mesh = colored_object([0.05, 0.10, 0.20])

    result = MODULE.align_one_object(
        mesh, np.array([[-1, -1, -1], [0.3, 1, 1]], dtype=float), scene)

    assert len(result["translations"]) == 2
    assert len({item["contour_face_index"] for item in result["translations"]}) == 2
    assert all(np.count_nonzero(item["translation_m"]) == 1 for item in result["translations"])
    assert all(abs(item["translation_m"]["XYZ".index(item["axis"])]) <= 0.15
               for item in result["translations"])


def test_next_plane_pair_is_tried_when_nearest_plane_fails_bounds_check():
    regularized = trimesh.util.concatenate([
        plane_mesh(0.0, 10, 10), plane_mesh(0.1), plane_mesh(0.15), plane_mesh(1.0, 10, 10),
    ])
    scene = MODULE.axis_plane_scene(regularized)
    mesh = colored_object([0.05, -0.08])

    result = MODULE.align_one_object(
        mesh, np.array([[0, -1, -1], [1, 1, 1]], dtype=float), scene)

    assert result["translations"]
    first = result["translations"][0]
    assert first["contour_face_index"] == 0
    assert first["axis"] == "X"
    np.testing.assert_allclose(first["translation_m"], [0.1, 0, 0], atol=1e-5)
