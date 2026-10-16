import sys
from pathlib import Path
import numpy as np
import open3d as o3d

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'06c_semantic_segmentation'))
from inspect_classes import ClassInspector,COLOR_CLAY


def inspector():
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]]),
        o3d.utility.Vector3iVector([[0,1,2],[1,3,2]]))
    mesh.compute_vertex_normals()
    view=ClassInspector.__new__(ClassInspector)
    view.full_mesh=mesh;view.verts=np.asarray(mesh.vertices);view.tris=np.asarray(mesh.triangles)
    view.num_verts=4;view.num_tris=2;view.orig_v_colors=np.full((4,3),.7)
    view.active_classes=[17];view.class_idx=0;view.mode='spotlight';view.active_instance_idx=-1;view.class_instances=[]
    view.valid_class_tris=np.array([0]);view.valid_class_v_indices=np.array([0,1,2])
    view.display_mesh=o3d.geometry.TriangleMesh(mesh)
    return view


def test_spotlight_cannot_color_a_neighboring_face_through_shared_vertices():
    view=inspector();view._apply_colors(update_geometry_buffers=False)
    colors=np.asarray(view.display_mesh.vertex_colors);tri=np.asarray(view.display_mesh.triangles)
    np.testing.assert_allclose(colors[tri[1]],np.tile(COLOR_CLAY,(3,1)))
    assert not np.allclose(colors[tri[0]],COLOR_CLAY)
    np.testing.assert_array_equal(np.asarray(view.display_mesh.vertices)[tri],view.verts[view.tris])
    assert len(view.full_mesh.vertices)==4 and len(view.full_mesh.triangles)==2


def test_mode_and_class_changes_restore_the_correct_display_geometry():
    view=inspector()
    for mode in ['spotlight','solo','rainbow','spotlight']:
        view.mode=mode;view._apply_colors(update_geometry_buffers=False)
        tri=np.asarray(view.display_mesh.triangles);vertices=np.asarray(view.display_mesh.vertices)
        expected=view.verts[view.tris[:1 if mode=='solo' else 2]]
        np.testing.assert_array_equal(vertices[tri],expected)
    view.valid_class_tris=np.array([1]);view.valid_class_v_indices=np.array([1,2,3])
    view._apply_colors(update_geometry_buffers=False)
    colors=np.asarray(view.display_mesh.vertex_colors);tri=np.asarray(view.display_mesh.triangles)
    np.testing.assert_allclose(colors[tri[0]],np.tile(COLOR_CLAY,(3,1)))
