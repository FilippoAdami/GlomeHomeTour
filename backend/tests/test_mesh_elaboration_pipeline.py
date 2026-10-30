"""Small complete scenes exercise the production mesh elaboration driver."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import open3d as o3d
import pytest
import trimesh

STAGE = Path(__file__).resolve().parents[1]/'06_mesh_elaboration'
sys.path.insert(0, str(STAGE))
spec = importlib.util.spec_from_file_location('step_mesh_elaboration', STAGE/'step_mesh_elaboration.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
import align_structural_vertical as vertical_alignment


def test_wall_plane_normals_recover_tilt_without_floor_ceiling_assumptions():
    rng = np.random.default_rng(73)
    rx, rz = np.deg2rad(6.0), np.deg2rad(-4.0)
    rotation_x = np.array([[1., 0., 0.], [0., np.cos(rx), -np.sin(rx)],
                           [0., np.sin(rx), np.cos(rx)]])
    rotation_z = np.array([[np.cos(rz), -np.sin(rz), 0.],
                           [np.sin(rz), np.cos(rz), 0.], [0., 0., 1.]])
    rotation = rotation_z @ rotation_x
    source_normals = np.tile([[1., 0., 0.], [0., 0., 1.]], (400, 1))
    normals = source_normals @ rotation.T + rng.normal(0., .008, source_normals.shape)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    # Floor-like outliers must not pull the wall-derived level toward their slope.
    outliers = np.tile([0., 1., 0.], (150, 1))
    outliers += rng.normal(0., .02, outliers.shape)
    outliers /= np.linalg.norm(outliers, axis=1)[:, None]
    all_normals = np.vstack((normals, outliers))
    areas = np.r_[np.ones(len(normals)), np.full(len(outliers), .2)]

    estimate = vertical_alignment._estimate_wall_up(all_normals, areas)

    assert estimate is not None
    up, support = estimate
    expected = rotation @ np.array([0., 1., 0.])
    assert np.dot(up, expected) > np.cos(np.deg2rad(1.0))
    assert support > .75


def write_scene(directory, heights, translation):
    vertices, faces, labels = [], [], []
    def quad(points, role, upward=None):
        start = len(vertices)
        vertices.extend(points)
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        if upward is not None:
            normal = np.cross(np.array(points[1])-points[0], np.array(points[2])-points[0])
            if (normal[1] > 0) != upward:
                triangles = triangles[:, ::-1]
        faces.extend(triangles+start)
        labels.extend([role]*2)
    for y in heights:
        quad([[0,y,0],[2,y,0],[2,y,2],[0,y,2]],17,True)
        roof = y+2.9
        quad([[0,roof,0],[2,roof,0],[2,roof,2],[0,roof,2]],23,False)
    low, high = min(heights), max(heights)+2.9
    corners = [[0,0],[2,0],[2,2],[0,2]]
    for a,b in zip(corners,corners[1:]+corners[:1]):
        quad([[a[0],low,a[1]],[b[0],low,b[1]],[b[0],high,b[1]],[a[0],high,a[1]]],41)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices)+translation)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces))
    mesh.paint_uniform_color([.5,.6,.7])
    mesh_path=directory/'mesh.ply'
    assert o3d.io.write_triangle_mesh(str(mesh_path),mesh)
    np.save(directory/'labels.npy',labels)
    (directory/'roles.json').write_text(json.dumps({'classifications':{str(label):{'structural_role':role} for label,role in ((17,'floor'),(23,'ceiling'),(41,'wall'))}}))
    return mesh_path


@pytest.mark.parametrize('heights,translation', [([0.],np.array([0.,0.,0.])),([0.,3.,6.],np.array([11.,20.,-7.]))])
def test_complete_pipeline_arbitrary_levels_and_world_translation(tmp_path,monkeypatch,heights,translation):
    mesh=write_scene(tmp_path,heights,translation)
    output=tmp_path/'output'
    args=['step_mesh_elaboration','--mesh-path',str(mesh),'--labels-path',str(tmp_path/'labels.npy'),
          '--structural-json',str(tmp_path/'roles.json'),'--output-dir',str(output)]
    monkeypatch.setattr(sys,'argv',args)
    assert driver.main() == 0
    patterns=json.loads((output/'plane_patterns_manifest.json').read_text())
    assert len(patterns['slabs']) == len(heights)-1
    for slab in patterns['slabs']:
        assert trimesh.load(slab['file_path']).is_watertight
    floors=[unit for unit in patterns['units'] if unit['structural_role']=='floor']
    assert len(floors) == len(heights)
    elevations=sorted(unit['mean_elevation_y'] for unit in floors)
    np.testing.assert_allclose(elevations,np.array(heights)+translation[1],atol=1e-6)
    assert (output/'all_structural_regularized.ply').is_file()


@pytest.mark.parametrize('role', ['floor', 'wall'])
def test_partial_scan_without_other_structural_roles(tmp_path, monkeypatch, role):
    mesh = write_scene(tmp_path, [0.], np.array([8., 12., -3.]))
    roles = {'classifications': {'17': {'structural_role': role}}} if role == 'floor' else {'classifications': {'41': {'structural_role': role}}}
    (tmp_path/'roles.json').write_text(json.dumps(roles))
    output = tmp_path/'partial'
    monkeypatch.setattr(sys, 'argv', ['step_mesh_elaboration', '--mesh-path', str(mesh),
        '--labels-path', str(tmp_path/'labels.npy'), '--structural-json', str(tmp_path/'roles.json'), '--output-dir', str(output)])
    assert driver.main() == 0
    patterns = json.loads((output/'plane_patterns_manifest.json').read_text())
    if role == 'wall':
        # The combined fitter drops regions below its 10,000-triangle support threshold.
        assert not patterns['units']
        assert not (output/'wall_enclosing_shell.ply').exists()
    else:
        assert patterns['units']
    assert all(unit['structural_role'] == role for unit in patterns['units'])
    assert not patterns['slabs']


def test_unlabelled_scene_extension_affects_collapse_without_inventing_a_roof(tmp_path,monkeypatch):
    path=write_scene(tmp_path,[0.],np.zeros(3))
    mesh=o3d.io.read_triangle_mesh(str(path))
    vertices=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.triangles)
    start=len(vertices)
    # This unlabeled floor patio is absent from all extracted structural instances.
    vertices=np.vstack([vertices,[[2,0,0],[3,0,0],[3,0,2],[2,0,2]]])
    faces=np.vstack([faces,np.array([[0,2,1],[0,3,2]])+start])
    mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([.5,.6,.7])
    assert o3d.io.write_triangle_mesh(str(path),mesh)
    labels=np.load(tmp_path/'labels.npy');np.save(tmp_path/'labels.npy',np.r_[labels,[99,99]])
    output=tmp_path/'collapse'
    monkeypatch.setattr(sys,'argv',['step_mesh_elaboration','--mesh-path',str(path),
        '--labels-path',str(tmp_path/'labels.npy'),'--structural-json',str(tmp_path/'roles.json'),'--output-dir',str(output)])
    assert driver.main() == 0
    from shapely.geometry import shape,Point
    units=json.loads((output/'plane_patterns_manifest.json').read_text())['units']
    floor=next(u for u in units if u['structural_role']=='floor')
    roof=next(u for u in units if u['structural_role']=='ceiling')
    assert floor['footprint_source'] == 'full_scene_per_level_height_collapse'
    assert shape(floor['footprint_xz']).covers(Point(2.5,1.))
    assert not shape(roof['footprint_xz']).covers(Point(2.5,1.))
    assert json.loads((output/'level_footprints_manifest.json').read_text())['levels']


def test_vertical_alignment_applies_one_rigid_transform_to_every_instance_and_scene_copy(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
    from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags

    output = tmp_path / "instances"
    output.mkdir()
    angle = np.deg2rad(6.0)
    source_rotation = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                                [0., np.sin(angle), np.cos(angle)]])
    base = np.array([[0., 0., 0.], [2., 0., 0.], [2., 0., 2.], [0., 0., 2.]])
    triangles = np.array([[0, 2, 1], [0, 3, 2]])
    original_vertices = {}
    instances = []
    for role in ("floor", "ceiling", "wall", "stair", "column", "ramp"):
        vertices = base.copy()
        if role == "ceiling":
            vertices[:, 1] += 3.
            triangles_for_role = triangles[:, ::-1]
        elif role == "wall":
            vertices[:, 1] = np.array([0., 0., 3., 3.])
            triangles_for_role = np.array([[0, 1, 2], [0, 2, 3]])
        else:
            triangles_for_role = triangles
        vertices = vertices @ source_rotation.T
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles_for_role)
        mesh.paint_uniform_color([.4, .5, .6])
        filename = f"{role}_inst000.ply"
        if role == "floor":
            assert export_ply_with_face_attributes(
                output / filename, vertices, triangles_for_role,
                np.tile([.4, .5, .6], (len(vertices), 1)),
                {"is_hole_fill": ("uchar", np.array([0, 1], dtype=np.uint8))},
            )
        else:
            assert o3d.io.write_triangle_mesh(str(output / filename), mesh)
        original_vertices[filename] = vertices
        instances.append({"instance_name": f"{role}_000", "role": role,
                          "file_name": filename, "elevation": {"y_mean": 0.}})

    door_path = tmp_path / "door_instance.ply"
    door_mesh = o3d.geometry.TriangleMesh()
    door_vertices = np.array([[.05, 0., .2], [.05, 0., .8], [.05, 2., .8]]) @ source_rotation.T
    door_mesh.vertices = o3d.utility.Vector3dVector(door_vertices)
    door_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    assert o3d.io.write_triangle_mesh(str(door_path), door_mesh)
    original_door_bytes = door_path.read_bytes()
    manifest_path = output / "structural_instances_manifest.json"
    legacy_door_dir = output / "vertically_aligned_door_window_instances"
    legacy_door_dir.mkdir()
    (legacy_door_dir / "legacy.ply").write_bytes(b"stale")
    manifest_path.write_text(json.dumps({"instances": instances, "door_window_merges": [{
        "source_class_id": 19, "instance_id": 1, "text_class_id": 1,
        "source_file": str(door_path), "aligned_source_file": "legacy.ply", "target_role": "wall",
    }]}))
    scene_mesh = tmp_path / "scene.ply"
    shutil.copyfile(output / "floor_inst000.ply", scene_mesh)
    original_scene_bytes = scene_mesh.read_bytes()

    result = vertical_alignment.align_structural_instances(output, scene_mesh)
    assert result["status"] == "applied"
    assert result["tilt_deg"] == pytest.approx(6., abs=1e-4)
    rotation, translation = np.asarray(result["rotation_matrix"]), np.asarray(result["translation_m"])
    for filename, vertices in original_vertices.items():
        transformed = np.asarray(o3d.io.read_triangle_mesh(str(output / filename)).vertices)
        np.testing.assert_allclose(transformed, vertices @ rotation.T + translation, atol=2e-6)
    np.testing.assert_array_equal(read_ply_face_flags(output / "floor_inst000.ply"), [0, 1])
    assert scene_mesh.read_bytes() == original_scene_bytes
    aligned_scene = o3d.io.read_triangle_mesh(str(output / "vertically_aligned_scene_mesh.ply"))
    np.testing.assert_allclose(np.asarray(aligned_scene.vertices),
                               original_vertices["floor_inst000.ply"] @ rotation.T + translation, atol=2e-6)
    np.testing.assert_array_equal(read_ply_face_flags(output / "vertically_aligned_scene_mesh.ply"), [0, 1])
    assert door_path.read_bytes() == original_door_bytes
    assert not legacy_door_dir.exists()
    saved_manifest = json.loads(manifest_path.read_text())
    assert "aligned_source_file" not in saved_manifest["door_window_merges"][0]
    assert "aligned_door_window_instances" not in saved_manifest["vertical_alignment"]
    before_rerun = {p.name: p.read_bytes() for p in output.glob("*_inst000.ply")}
    vertical_alignment.align_structural_instances(output, scene_mesh)
    assert before_rerun == {p.name: p.read_bytes() for p in output.glob("*_inst000.ply")}

    # A manifest created by the former estimator has no algorithm marker; it
    # must be inverted and re-estimated instead of locking in its old tilt.
    saved_manifest = json.loads(manifest_path.read_text())
    saved_manifest["vertical_alignment"].pop("algorithm")
    manifest_path.write_text(json.dumps(saved_manifest))
    vertical_alignment.align_structural_instances(output, scene_mesh)
    for filename, vertices in original_vertices.items():
        transformed = np.asarray(o3d.io.read_triangle_mesh(str(output / filename)).vertices)
        np.testing.assert_allclose(transformed, vertices @ rotation.T + translation, atol=3e-6)


def test_floor_grouping_is_an_independently_runnable_step(tmp_path, monkeypatch):
    calls = []
    scene_size = tmp_path / "scene_size.txt"
    scene_size.write_text("floors: 2\n")

    def capture(path, run_name):
        calls.append((Path(path).stem, list(sys.argv)))

    monkeypatch.setattr(driver.runpy, "run_path", capture)
    assert driver.run_pipeline_for_scene(
        tmp_path / "mesh.ply", tmp_path / "labels.npy", tmp_path / "classes.json",
        tmp_path / "output", scene_size_file=scene_size,
        only_step="group_floor_instances_by_level",
    ) == 0
    assert len(calls) == 1
    assert calls[0][0] == "group_floor_instances_by_level"
    assert calls[0][1][calls[0][1].index("--scene-size-file") + 1] == str(scene_size.absolute())


def test_ceiling_association_is_an_independently_runnable_step(tmp_path, monkeypatch):
    calls = []
    scene_size = tmp_path / "scene_size.txt"
    scene_size.write_text("floors: 2\n")
    monkeypatch.setattr(driver.runpy, "run_path", lambda path, run_name: calls.append(Path(path).stem))
    assert driver.run_pipeline_for_scene(
        tmp_path / "mesh.ply", tmp_path / "labels.npy", tmp_path / "classes.json",
        tmp_path / "output", scene_size_file=scene_size,
        only_step="add_ceilings_to_floor_levels",
    ) == 0
    assert calls == ["add_ceilings_to_floor_levels"]


@pytest.mark.parametrize("step_name", ["clean_floor_level_intermediates", "fit_floor_level_planes",
                                       "aggregate_fit_ceiling_planes", "slice_scene_by_floor",
                                       "clean_floor_scene_slices", "project_floor_slices_onto_planes"])
def test_floor_cleanup_and_plane_fit_are_independently_runnable(tmp_path, monkeypatch, step_name):
    calls = []
    scene_size = tmp_path / "scene_size.txt"
    scene_size.write_text("floors: 2\n")
    monkeypatch.setattr(driver.runpy, "run_path",
                        lambda path, run_name: calls.append((Path(path).stem, list(sys.argv))))
    assert driver.run_pipeline_for_scene(
        tmp_path / "mesh.ply", tmp_path / "labels.npy", tmp_path / "classes.json",
        tmp_path / "output", scene_size_file=scene_size, only_step=step_name,
    ) == 0
    assert calls[0][0] == step_name
    if step_name == "slice_scene_by_floor":
        argv = calls[0][1]
        for flag, expected in (("--instances-dir", str((tmp_path / "output").absolute())),
                               ("--mesh-path", str((tmp_path / "mesh.ply").absolute())),
                               ("--labels-path", str((tmp_path / "labels.npy").absolute())),
                               ("--class-metadata", str((tmp_path / "classes.json").absolute()))):
            assert argv[argv.index(flag) + 1] == expected
    elif step_name in {"clean_floor_scene_slices", "project_floor_slices_onto_planes"}:
        argv = calls[0][1]
        assert argv[argv.index("--instances-dir") + 1] == str((tmp_path / "output").absolute())


@pytest.mark.parametrize("skip_texture", [False, True])
def test_floor_intermediate_steps_run_in_dependency_order(tmp_path, monkeypatch, skip_texture):
    calls = []
    scene_size = tmp_path / "scene_size.txt"
    scene_size.write_text("floors: 2\n")
    monkeypatch.setattr(driver.runpy, "run_path", lambda path, run_name: calls.append(Path(path).stem))

    assert driver.run_pipeline_for_scene(
        tmp_path / "mesh.ply", tmp_path / "labels.npy", tmp_path / "classes.json",
        tmp_path / "output", scene_size_file=scene_size, skip_texture=skip_texture,
    ) == 0

    floor_steps = ["group_floor_instances_by_level", "add_ceilings_to_floor_levels",
                   "clean_floor_level_intermediates", "fit_floor_level_planes",
                   "aggregate_fit_ceiling_planes", "slice_scene_by_floor",
                   "clean_floor_scene_slices", "project_floor_slices_onto_planes"]
    start = calls.index(floor_steps[0])
    assert calls[start:start + len(floor_steps)] == floor_steps
    assert calls == ["extract_structural_meshes", "enclose_wall_instances", *floor_steps,
                     "regularize_floor_projection_contours", "fit_floor_projection_contours_to_black",
                     "overlay_fitted_floor_wall_contours", "regularize_floor_level_patterns"]


def test_merged_wall_step_fits_already_aligned_instances(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(driver.runpy, "run_path", lambda path, run_name: calls.append(Path(path).stem))
    assert driver.run_pipeline_for_scene(
        tmp_path / "mesh.ply", tmp_path / "labels.npy", tmp_path / "classes.json",
        tmp_path / "output", only_step="fit_enclosing_walls",
    ) == 0
    assert calls == ["enclose_wall_instances"]


def test_regularizer_uses_phase04_aligned_source_mesh(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    aligned_scene = output / "vertically_aligned_scene_mesh.ply"
    aligned_scene.touch()
    (output / "structural_instances_manifest.json").write_text(json.dumps({
        "vertical_alignment": {"status": "applied"},
    }))
    observed_argv = []

    def capture_argv(path, run_name):
        observed_argv.extend(sys.argv)

    monkeypatch.setattr(driver.runpy, "run_path", capture_argv)
    assert driver.run_pipeline_for_scene(
        tmp_path / "mesh.ply", tmp_path / "labels.npy", tmp_path / "classes.json",
        output, only_step="recognize_and_regularize_patterns",
    ) == 0
    assert observed_argv[observed_argv.index("--scene-mesh") + 1] == str(tmp_path / "mesh.ply")


def test_structural_extraction_and_texture_synthesis_with_hole_fill_flags(tmp_path, monkeypatch):
    from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags

    # Create scene with 2 floor triangles: 1 genuine original (red), 1 hole-fill (blue)
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    colors = np.array([
        [255, 0, 0],
        [255, 0, 0],
        [255, 0, 0],
        [0, 0, 255],  # 4th vertex is blue, belonging to hole face
    ], dtype=np.uint8)
    flags = np.array([0, 1], dtype=np.uint8)  # Face 0 is genuine, Face 1 is hole-fill

    mesh_path = tmp_path / "mesh.ply"
    assert export_ply_with_face_attributes(
        mesh_path, verts, faces, colors, face_attributes={"is_hole_fill": ("uchar", flags)}
    )

    loaded_flags = read_ply_face_flags(mesh_path, "is_hole_fill")
    assert loaded_flags is not None
    assert np.array_equal(loaded_flags, flags)

    labels = np.array([17, 17], dtype=np.int32)
    np.save(tmp_path / "labels.npy", labels)
    roles = {"classifications": {"17": {"structural_role": "floor"}}}
    (tmp_path / "roles.json").write_text(json.dumps(roles))

    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "step_mesh_elaboration",
            "--mesh-path",
            str(mesh_path),
            "--labels-path",
            str(tmp_path / "labels.npy"),
            "--structural-json",
            str(tmp_path / "roles.json"),
            "--output-dir",
            str(output),
        ],
    )
    assert driver.main() == 0

    inst_manifest = json.loads((output / "structural_instances_manifest.json").read_text())
    assert len(inst_manifest["instances"]) == 1
    inst_info = inst_manifest["instances"][0]
    assert inst_info["num_original_triangles"] == 1
    assert inst_info["num_filled_triangles"] == 1

    extracted_ply = Path(inst_info["file_path"])
    extracted_flags = read_ply_face_flags(extracted_ply, "is_hole_fill")
    assert extracted_flags is not None
    assert np.array_equal(extracted_flags, flags)

    regularized_ply = output / "floor_inst000_regularized.ply"
    assert regularized_ply.is_file()
    reg_mesh = o3d.io.read_triangle_mesh(str(regularized_ply))
    assert len(reg_mesh.vertices) > 0
    assert reg_mesh.has_vertex_colors()


def test_pipeline_stage_order_and_object_alignment(tmp_path, monkeypatch):
    mesh = write_scene(tmp_path, [0.], np.array([0., 0., 0.]))
    output = tmp_path / "output_aligned"

    # Create dummy segmentation output directory with an object instance and structural instance
    seg_dir = tmp_path / "segmentation_output"
    masks_dir = seg_dir / "3d_masks"
    masks_dir.mkdir(parents=True)
    (seg_dir / "class_metadata.json").write_text(json.dumps({
        "classes": {
            "17": {"structural_role": "floor", "name": "floor"},
            "41": {"structural_role": "wall", "name": "wall"},
            "100": {"structural_role": "object", "name": "chair"}
        }
    }))

    # Floor mesh for alignment anchor
    floor_dir = masks_dir / "class_017_floor"
    floor_dir.mkdir()
    floor_mesh = o3d.geometry.TriangleMesh()
    floor_mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [1, 0, 1]])
    floor_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    o3d.io.write_triangle_mesh(str(floor_dir / "instance_000.ply"), floor_mesh)

    # Wall mesh for alignment anchor
    wall_dir = masks_dir / "class_041_wall"
    wall_dir.mkdir()
    wall_mesh = o3d.geometry.TriangleMesh()
    wall_mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    wall_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    o3d.io.write_triangle_mesh(str(wall_dir / "instance_000.ply"), wall_mesh)

    # Object mesh to be aligned and fitted
    obj_dir = masks_dir / "class_100_chair"
    obj_dir.mkdir()
    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.vertices = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.6, 0.6, 0.5]])
    obj_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    o3d.io.write_triangle_mesh(str(obj_dir / "instance_000.ply"), obj_mesh)

    args = [
        "step_mesh_elaboration",
        "--mesh-path", str(mesh),
        "--labels-path", str(tmp_path / "labels.npy"),
        "--structural-json", str(tmp_path / "roles.json"),
        "--output-dir", str(output),
        "--segmentation-output", str(seg_dir),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert driver.main() == 0

    timings = json.loads((output / "step06_timings.json").read_text())
    executed_stages = [item["stage"] for item in timings]
    assert executed_stages == [
        "extract_structural_meshes",
        "fit_enclosing_walls",
        "fit_floors_ceilings_planes",
        "recognize_and_regularize_patterns",
        "apply_structural_textures",
    ]
    assert [item["stage"] for item in timings[1]["substeps"]] == [
        "enclose_wall_instances",
    ]

    assert (output / "all_structural_regularized.ply").is_file()
    fitted_walls = json.loads((output / "wall_planes_manifest.json").read_text())
    assert "rounding" not in fitted_walls


def test_run_extraction_then_merged_wall_step_only(tmp_path, monkeypatch):
    mesh = write_scene(tmp_path, [0.], np.zeros(3))
    output = tmp_path / "stepwise"
    common = ["--mesh-path", str(mesh), "--labels-path", str(tmp_path / "labels.npy"),
              "--structural-json", str(tmp_path / "roles.json"), "--output-dir", str(output)]

    monkeypatch.setattr(sys, "argv", ["step_mesh_elaboration", *common,
                                       "--only-step", "extract_structural_meshes"])
    assert driver.main() == 0
    assert (output / "structural_instances_manifest.json").is_file()
    assert not (output / "wall_planes_manifest.json").exists()

    monkeypatch.setattr(sys, "argv", ["step_mesh_elaboration", *common,
                                       "--only-step", "fit_enclosing_walls"])
    assert driver.main() == 0
    wall_manifest = json.loads((output / "wall_planes_manifest.json").read_text())
    assert "rounding" not in wall_manifest
    timings = json.loads((output / "step06_timings.json").read_text())
    assert [item["stage"] for item in timings] == ["fit_enclosing_walls"]
    assert [item["stage"] for item in timings[0]["substeps"]] == [
        "enclose_wall_instances",
    ]


def test_patio_with_columns_no_walls_pipeline(tmp_path, monkeypatch):
    """Verify patio scenes with floor, ceiling, and columns but NO walls process end-to-end without errors."""
    vertices, faces, labels = [], [], []

    def quad(points, label, upward=None):
        start = len(vertices)
        vertices.extend(points)
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        if upward is not None:
            normal = np.cross(np.array(points[1]) - points[0], np.array(points[2]) - points[0])
            if (normal[1] > 0) != upward:
                triangles = triangles[:, ::-1]
        faces.extend(triangles + start)
        labels.extend([label] * 2)

    # 1. Floor at y = 0
    quad([[0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4]], 17, upward=True)
    # 2. Ceiling / roof at y = 3.0
    quad([[0, 3, 0], [4, 3, 0], [4, 3, 4], [0, 3, 4]], 23, upward=False)
    # 3. Column mesh (vertical post from floor to ceiling)
    col_corners = [[1.0, 1.0], [1.3, 1.0], [1.3, 1.3], [1.0, 1.3]]
    for a, b in zip(col_corners, col_corners[1:] + col_corners[:1]):
        quad([[a[0], 0, a[1]], [b[0], 0, b[1]], [b[0], 3, b[1]], [a[0], 3, a[1]]], 6)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([0.5, 0.6, 0.7])
    mesh_path = tmp_path / "mesh.ply"
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    np.save(tmp_path / "labels.npy", np.array(labels))

    # Structural metadata with floor, ceiling, and columns (NO WALLS)
    roles_data = {
        "classifications": {
            "17": {"structural_role": "floor", "confidence": 0.98},
            "23": {"structural_role": "ceiling", "confidence": 0.95},
            "6": {"structural_role": "columns", "confidence": 0.90},
        }
    }
    (tmp_path / "roles.json").write_text(json.dumps(roles_data))

    # Setup segmentation directory with column instance
    seg_dir = tmp_path / "segmentation"
    masks_dir = seg_dir / "3d_masks"
    masks_dir.mkdir(parents=True)
    (seg_dir / "class_metadata.json").write_text(json.dumps(roles_data))

    # Floor instance
    floor_dir = masks_dir / "class_017_floor"
    floor_dir.mkdir()
    floor_mesh = o3d.geometry.TriangleMesh()
    floor_mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [4, 0, 0], [4, 0, 4]])
    floor_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    o3d.io.write_triangle_mesh(str(floor_dir / "instance_000.ply"), floor_mesh)

    # Ceiling instance
    ceil_dir = masks_dir / "class_023_ceiling"
    ceil_dir.mkdir()
    ceil_mesh = o3d.geometry.TriangleMesh()
    ceil_mesh.vertices = o3d.utility.Vector3dVector([[0, 3, 0], [4, 3, 0], [4, 3, 4]])
    ceil_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    o3d.io.write_triangle_mesh(str(ceil_dir / "instance_000.ply"), ceil_mesh)

    # Column instance
    col_dir = masks_dir / "class_006_columns"
    col_dir.mkdir()
    col_mesh = o3d.geometry.TriangleMesh()
    col_mesh.vertices = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0], [1.3, 0.0, 1.0], [1.3, 3.0, 1.0]])
    col_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    o3d.io.write_triangle_mesh(str(col_dir / "instance_000.ply"), col_mesh)

    output = tmp_path / "patio_output"
    args = [
        "step_mesh_elaboration",
        "--mesh-path", str(mesh_path),
        "--labels-path", str(tmp_path / "labels.npy"),
        "--structural-json", str(tmp_path / "roles.json"),
        "--output-dir", str(output),
        "--segmentation-output", str(seg_dir),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert driver.main() == 0

    # Verify no-walls manifest was safely written
    wall_manifest = json.loads((output / "wall_planes_manifest.json").read_text())
    assert wall_manifest["wall_planes"] == []
    assert "dominant_orientation_deg" in wall_manifest
    assert "rounding" not in wall_manifest

    # Verify floor and ceiling regularized
    assert (output / "all_structural_regularized.ply").is_file()
    patterns = json.loads((output / "plane_patterns_manifest.json").read_text())
    unit_roles = {u["structural_role"] for u in patterns["units"]}
    assert "floor" in unit_roles
    assert "ceiling" in unit_roles
    assert "wall" not in unit_roles

    # Verify column was extracted as structural instance
    struct_manifest = json.loads((output / "structural_instances_manifest.json").read_text())
    assert "column" in struct_manifest["roles"]
    assert any(inst["role"] == "column" for inst in struct_manifest["instances"])
    assert (output / "column_inst000.ply").is_file()

    # Verify executed stages in Stage 06
    timings = json.loads((output / "step06_timings.json").read_text())
    executed_stages = [item["stage"] for item in timings]
    assert executed_stages == [
        "extract_structural_meshes",
        "fit_enclosing_walls",
        "fit_floors_ceilings_planes",
        "recognize_and_regularize_patterns",
        "apply_structural_textures",
    ]
