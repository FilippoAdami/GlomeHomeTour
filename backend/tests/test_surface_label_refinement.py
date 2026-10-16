import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'06c_semantic_segmentation'))
from surface_label_refinement import reclaim_compatible_surfaces
from refine_class_boundaries import compute_mesh_derivatives
from detect_structural_classes import build_mesh_dual_graph


def scene():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    triangles=np.array([[0,1,2],[1,3,2]])
    return vertices,triangles,np.array([17,103])


def run(v,t,l,candidate=None,colors=None,scale=1):
    areas,n,c,_=compute_mesh_derivatives(v,t,np.zeros_like(v))
    u,w=build_mesh_dual_graph(t)
    return reclaim_compatible_surfaces(l,v,t,n,c,np.full((len(l),3),.5) if colors is None else colors,
        areas,u,w,l==103 if candidate is None else candidate,l==17,.02*scale)


def test_weak_coplanar_surface_is_reclaimed_and_seed_is_preserved():
    v,t,l=scene();result,report=run(v,t,l)
    np.testing.assert_array_equal(result,[17,17]);np.testing.assert_array_equal(l,[17,103])
    assert report['changed_faces']==1


def test_vertical_fixture_opposite_normal_and_color_boundaries_are_preserved():
    v,t,l=scene();v[3]=[0,0,1]
    result,_=run(v,t,l);np.testing.assert_array_equal(result,l)
    v,t,l=scene();t[1]=t[1,::-1]
    result,_=run(v,t,l);np.testing.assert_array_equal(result,l)
    v,t,l=scene();result,_=run(v,t,l,colors=np.array([[.5,.5,.5],[.7,.5,.5]]))
    np.testing.assert_array_equal(result,l)


def test_off_plane_corner_is_not_hidden_by_a_nearby_centroid():
    v,t,l=scene();v[3,2]=.05
    result,_=run(v,t,l);np.testing.assert_array_equal(result,l)


def test_protected_faces_are_exactly_preserved():
    v,t,l=scene();result,_=run(v,t,l,candidate=np.zeros(2,bool))
    np.testing.assert_array_equal(result,l)


def test_ids_rotation_and_metric_units_do_not_change_the_result():
    v,t,l=scene();rotation=np.linalg.qr(np.random.default_rng(23).normal(size=(3,3)))[0]
    v=v@rotation.T*100+np.array([7.,3.,-18.]);renamed=np.where(l==17,905,701)
    areas,n,c,_=compute_mesh_derivatives(v,t,np.zeros_like(v))
    u,w=build_mesh_dual_graph(t)
    result,_=reclaim_compatible_surfaces(renamed,v,t,n,c,np.full((2,3),.5),areas,u,w,renamed==701,renamed==905,2.)
    np.testing.assert_array_equal(result,[905,905])


def test_invalid_tolerance_is_rejected():
    v,t,l=scene()
    with pytest.raises(ValueError):run(v,t,l,scale=0)


def test_noisy_small_reference_triangle_does_not_block_a_supported_flat_patch():
    v=np.array([[-.01,0,.002],[0,0,0],[0,1,0],[1,0,0]])
    t=np.array([[0,1,2],[1,3,2]]);l=np.array([17,103])
    result,_=run(v,t,l)
    np.testing.assert_array_equal(result,[17,17])


def test_a_bending_patch_is_not_treated_as_one_flat_surface():
    v=np.array([[x,y,z] for x,z in enumerate([0.,0.,.15,.45,.9]) for y in (0.,1.)])
    t=np.array([[2*x,2*x+2,2*x+3] for x in range(4)]+[[2*x,2*x+3,2*x+1] for x in range(4)])
    l=np.full(8,103,dtype=int);l[[0,4]]=17
    result,_=run(v,t,l)
    np.testing.assert_array_equal(result,l)


def test_a_color_boundary_splits_an_otherwise_coplanar_patch():
    v=np.array([[x,y,0.] for x in range(4) for y in (0.,1.)])
    t=np.array([[2*x,2*x+2,2*x+3] for x in range(3)]+[[2*x,2*x+3,2*x+1] for x in range(3)])
    l=np.array([17,103,103,17,103,103]);colors=np.full((6,3),.5);colors[[2,5]]=.8
    result,_=run(v,t,l,colors=colors)
    np.testing.assert_array_equal(result,[17,17,103,17,17,103])
