"""Small complete scenes exercise the same five-stage driver as production."""
import importlib.util
import json
from pathlib import Path
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
    assert sys.argv == args
    levels=json.loads((output/'structural_levels_manifest.json').read_text())
    assert len(levels['levels']) == len(heights)
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
