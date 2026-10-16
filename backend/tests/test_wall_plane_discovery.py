"""Regression coverage for scene-independent wall-plane discovery."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

MODULE = Path(__file__).parents[1] / "06_mesh_elaboration" / "fit_wall_planes.py"
SPEC = importlib.util.spec_from_file_location("fit_wall_planes", MODULE)
fit_wall_planes = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(fit_wall_planes)


def _quad(points: list[np.ndarray], vertices: list[np.ndarray], triangles: list[list[int]]) -> None:
    start = len(vertices); vertices.extend(points)
    triangles.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))


def _wall_mesh(path: Path, angle: float, translation: tuple[float, float], disconnected: bool = False) -> None:
    """A rotated room with a recess and optional separate wall, plus small noise."""
    direction = np.array([np.cos(angle), 0., np.sin(angle)])
    normal = np.array([-np.sin(angle), 0., np.cos(angle)])
    origin = np.array([translation[0], 0., translation[1]])
    vertices: list[np.ndarray] = []; triangles: list[list[int]] = []
    def point(t: float, n: float, y: float) -> np.ndarray:
        return origin + t * direction + n * normal + np.array([0., y, 0.])
    # Rectangle sides and a recessed return: all finite, no assumed coordinate range.
    for t0, t1, n0, n1 in ((0., 6., 0., 0.), (0., 6., 4., 4.), (0., 0., 0., 4.), (6., 6., 0., 4.),
                            (2., 3., 1., 1.)):
        if t0 == t1:
            _quad([point(t0, n0, 0.), point(t0, n1, 0.), point(t0, n1, 3.), point(t0, n0, 3.)], vertices, triangles)
        else:
            _quad([point(t0, n0, 0.), point(t1, n1, 0.), point(t1, n1, 3.), point(t0, n0, 3.)], vertices, triangles)
    if disconnected:
        _quad([point(10., 8., 0.), point(13., 8., 0.), point(13., 8., 2.5), point(10., 8., 2.5)], vertices, triangles)
    # Moderate, deterministic vertical diagonal scan noise must not become a plane.
    for index in range(24):
        t, n = 20. + .09 * index, 20. + .07 * (index % 5)
        _quad([point(t, n, 0.), point(t + .05, n + .08, 0.),
               point(t + .05, n + .08, .05), point(t, n, .05)], vertices, triangles)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([.3, .5, .7], (len(vertices), 1)))
    o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=True)


@pytest.mark.parametrize("angle,translation", [(0., (0., 0.)), (np.deg2rad(31.), (123.4, -57.8))])
def test_discovers_finite_planes_without_scene_coordinates(tmp_path: Path, angle: float, translation: tuple[float, float]) -> None:
    mesh_path = tmp_path / "wall_inst001.ply"
    _wall_mesh(mesh_path, angle, translation, disconnected=True)
    instances = tmp_path / "instances"; instances.mkdir()
    (instances / "structural_instances_manifest.json").write_text(json.dumps({"instances": [
        {"role": "wall", "file_path": str(mesh_path)},
    ]}))
    old_argv = sys.argv
    try:
        sys.argv = ["fit_wall_planes.py", "--instances-dir", str(instances), "--output-dir", str(instances)]
        assert fit_wall_planes.main() == 0
    finally:
        sys.argv = old_argv
    result = json.loads((instances / "wall_planes_manifest.json").read_text())
    planes = result["wall_planes"]
    assert len(planes) >= 5
    assert all(np.isfinite(plane["plane_equation"]).all() for plane in planes)
    assert all(plane["bounds_y"][1] > plane["bounds_y"][0] for plane in planes)
    assert all(max(plane.get("span_u_m", plane.get("span_w_m"))) - min(plane.get("span_u_m", plane.get("span_w_m"))) < 7 for plane in planes)
    a = np.array(result["orthogonal_basis"]["dir_A_east_west"])
    assert abs(np.dot(a[[0, 2]], [np.cos(angle), np.sin(angle)])) > .999
    assert (instances / "wall_cleaned_pieces.ply").is_file()
    assert (instances / "wall_enclosing_planes.ply").is_file()



def test_discovery_rejects_tall_wall_tabs_narrower_than_12cm():
    vertices, triangles = [], []
    _quad([np.array(p) for p in [(0,0,0),(0,0,3),(0,3,3),(0,3,0)]], vertices, triangles)
    _quad([np.array(p) for p in [(1,0,0),(1,0,.08),(1,3,.08),(1,3,0)]], vertices, triangles)
    vertices, triangles = np.array(vertices), np.array(triangles)
    corners = vertices[triangles]
    cross = np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0])
    area2 = np.linalg.norm(cross,axis=1)
    planes,_ = fit_wall_planes._discover(vertices,triangles,cross/area2[:,None],corners.mean(axis=1),
        area2/2,np.array([1.,0.,0.]),np.array([0.,0.,1.]),.10,.15)
    assert len(planes) == 1
    assert planes[0]['offset_m'] == pytest.approx(0.)


def test_tall_diagonal_fragment_under_12cm_in_wall_frame_is_removed(tmp_path, monkeypatch):
    path = tmp_path/'wall.ply'
    _wall_mesh(path,0.,(0.,0.))
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, faces = np.asarray(mesh.vertices).tolist(), np.asarray(mesh.triangles).tolist()
    # Tall enough and enough area to defeat area-only/noise filtering.
    _quad([np.array(p) for p in [(2,0,2),(2.09,0,2.11),(2.09,1,2.11),(2,1,2)]], vertices, faces)
    _quad([np.array(p) for p in [(4,0,2),(4.4,0,2.4),(4.4,1,2.4),(4,1,2)]], vertices, faces)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((len(vertices),3),.6))
    assert o3d.io.write_triangle_mesh(str(path),mesh)
    monkeypatch.setattr(sys,'argv',['fit_wall_planes','--wall-mesh',str(path),'--output-dir',str(tmp_path)])
    assert fit_wall_planes.main() == 0
    result=json.loads((tmp_path/'wall_planes_manifest.json').read_text())
    assert result['unmodeled_wall_support']['triangle_count'] == 2
    assert result['unmodeled_wall_support']['area_m2'] == pytest.approx(np.sqrt(.4**2+.4**2), abs=1e-4)
