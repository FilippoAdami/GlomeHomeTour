import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'06c_semantic_segmentation'))
from wall_orientation_refinement import refine_wall_orientations


def extruded(polygon,cap_triangles,length=2.):
    points=np.array(polygon,dtype=float);n=len(points)
    vertices=np.vstack([np.c_[points,np.zeros(n)],np.c_[points,np.full(n,length)]])
    triangles=[]
    for a in range(n):
        b=(a+1)%n
        triangles.extend([[a,b,b+n],[a,b+n,a+n]])
    triangles.extend([list(reversed(t)) for t in cap_triangles])
    triangles.extend([[i+n for i in t] for t in cap_triangles])
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.array(triangles)))
    assert mesh.is_watertight()
    return mesh


def cantilever():
    return extruded([[-.2,-2],[0,-2],[0,-.03],[1,-.03],[1,0],[0,0],[-.2,0]],
        [[0,1,2],[0,2,5],[0,5,6],[2,3,4],[2,4,5]])


def run(mesh,labels=None,wall_id=7,scale=1.,up=None):
    labels=np.full(len(mesh.triangles),wall_id,dtype=int) if labels is None else labels
    ids=np.unique(labels[labels>0]);weights=(labels[:,None]==ids[None,:]).astype(float)*4
    return refine_wall_orientations(mesh,labels,csr_matrix(weights),ids,[wall_id],
        np.array([0.,1.,0.]) if up is None else up,.02*scale)


def test_all_horizontal_wall_triangles_are_reassigned_even_when_supported():
    mesh=extruded([[0,-2],[1,-2],[1,0],[0,0]],[[0,1,2],[0,2,3]])
    labels=np.full(len(mesh.triangles),7);labels[8:]=103
    mesh.compute_triangle_normals();horizontal=abs(np.asarray(mesh.triangle_normals)[:,1])>=np.sqrt(.5)
    result,report=run(mesh,labels)
    assert np.all(result[horizontal & (labels==7)]==103)
    assert np.all(result[~horizontal]==labels[~horizontal])
    assert report['unknown_faces']==0


def test_local_owner_keeps_its_faces_and_receives_neighboring_horizontal_wall():
    mesh=cantilever();labels=np.full(len(mesh.triangles),7);labels[8]=103
    result,report=run(mesh,labels)
    assert result[8]==103 and result[9]==103
    assert np.all(result>0) and report['unknown_faces']==0


def test_ids_rotation_and_units_do_not_change_assignment():
    mesh=cantilever();labels=np.full(len(mesh.triangles),7);labels[8]=103
    original,_=run(mesh,labels)
    rotation=np.linalg.qr(np.random.default_rng(33).normal(size=(3,3)))[0]
    vertices=np.asarray(mesh.vertices).copy()
    mesh.vertices=o3d.utility.Vector3dVector((vertices@rotation.T)*100+np.array([17.,-35.,8.]))
    renamed=np.where(labels==7,8051,405)
    result,_=run(mesh,renamed,wall_id=8051,scale=100.,up=rotation@np.array([0.,1.,0.]))
    np.testing.assert_array_equal(result,np.where(original==7,8051,405))


def test_no_wall_faces_is_a_noop():
    mesh=cantilever();labels=np.full(len(mesh.triangles),103)
    result,report=run(mesh,labels)
    assert np.array_equal(result,labels) and report['changed_faces']==0


def test_invalid_gravity_tolerance_and_missing_owner_are_rejected():
    import pytest
    mesh=cantilever()
    with pytest.raises(ValueError):run(mesh,up=np.zeros(3))
    with pytest.raises(ValueError):run(mesh,scale=0)
    with pytest.raises(ValueError,match='non-wall owner'):run(mesh)


def test_caller_arrays_and_mesh_are_not_modified():
    mesh=cantilever();labels=np.full(len(mesh.triangles),7);labels[8]=103
    up=np.array([0.,3.,0.]);before=labels.copy();vertices=np.asarray(mesh.vertices).copy();triangles=np.asarray(mesh.triangles).copy()
    run(mesh,labels,up=up)
    np.testing.assert_array_equal(labels,before)
    np.testing.assert_array_equal(up,[0.,3.,0.])
    np.testing.assert_array_equal(mesh.vertices,vertices)
    np.testing.assert_array_equal(mesh.triangles,triangles)


def test_frozen_priors_choose_between_existing_local_owners():
    mesh=extruded([[0,-2],[1,-2],[1,0],[0,0]],[[0,1,2],[0,2,3]])
    labels=np.full(len(mesh.triangles),7);labels[2:4]=103;labels[6:8]=211
    ids=np.array([7,103,211]);weights=np.zeros((len(labels),3));weights[:,2]=4
    result,_=refine_wall_orientations(mesh,labels,csr_matrix(weights),ids,[7],np.array([0.,1.,0.]),.02)
    assert np.all(result[[4,5]]==211)
    assert np.all(result[labels!=7]==labels[labels!=7])


def test_disconnected_patch_gets_an_existing_owner_without_unknown_faces():
    mesh=cantilever();other=cantilever();other.translate([4.,0.,0.]);n=len(mesh.triangles);mesh+=other
    labels=np.r_[np.full(n,7),np.full(n,103)]
    result,report=run(mesh,labels)
    mesh.compute_triangle_normals();horizontal=abs(np.asarray(mesh.triangle_normals)[:,1])>=np.sqrt(.5)
    assert np.all(result[horizontal & (labels==7)]==103)
    assert np.all(result>0) and report['disconnected_route_faces']>0


def test_unobserved_leftover_object_specks_cannot_attract_new_faces():
    mesh=cantilever();n=len(mesh.triangles)
    speck=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[.5,.01,0],[.6,.01,0],[.6,.01,.1]]),
                                  o3d.utility.Vector3iVector([[0,1,2]]))
    owner=cantilever();owner.translate([4.,0.,0.]);mesh+=speck;mesh+=owner
    labels=np.r_[np.full(n,7),103,np.full(n,211)]
    weights=np.zeros((len(labels),3));weights[:n,0]=4;weights[n+1:,2]=4
    result,_=refine_wall_orientations(mesh,labels,csr_matrix(weights),np.array([7,103,211]),
                                     [7],np.array([0.,1.,0.]),.02)
    assert np.all(result[result!=labels]==211)
    assert result[n]==103
