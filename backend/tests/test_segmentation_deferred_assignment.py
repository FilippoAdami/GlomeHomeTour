"""Late geometric assignment must fill interior gaps without changing accepted owners."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_semantic_segmentation'))
from deferred_assignment import assign_deferred_patches,read_infill_attributes


def run(labels, centers=None, normals=None, rgb=None, walls=()):
    labels=np.array(labels,np.int32);n=len(labels)
    centers=np.column_stack((np.arange(n)*.01,np.zeros(n),np.zeros(n))) if centers is None else np.array(centers,float)
    normals=np.tile([0.,0.,1.],(n,1)) if normals is None else np.array(normals,float)
    rgb=np.full((n,3),.5) if rgb is None else np.array(rgb,float)
    result,metrics,links = assign_deferred_patches(labels,centers,normals,np.full(n,.01),rgb,
        np.arange(n-1),np.arange(1,n),np.array([0.,1.,0.]),set(walls))
    return result,metrics


def test_interior_gray_slivers_take_the_enclosing_owner():
    result,metrics=run([17,-1,17,-1,17])
    np.testing.assert_array_equal(result,[17]*5)
    assert metrics['assigned_faces']==2 and metrics['remaining_faces']==0


def test_one_connected_patch_splits_by_plane_and_preserves_boundary_owners():
    normals=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0]]
    centers=[[0,0,0],[.01,0,0],[.02,0,0],[.02,0,.01],[.03,0,.01],[.04,0,.01]]
    result,metrics=run([17,-1,-1,-1,-1,91],centers,normals)
    np.testing.assert_array_equal(result,[17,17,17,91,91,91])
    assert metrics['pieces']==2


def test_source_rgb_separates_coplanar_pieces_and_ignores_display_palette():
    rgb=[[.1,.1,.1]]*3+[[.9,.9,.9]]*3
    result,_=run([17,-1,-1,-1,-1,91],rgb=rgb)
    np.testing.assert_array_equal(result,[17,17,17,91,91,91])


def test_parallel_plane_offset_prefers_coplanar_neighbor():
    centers=[[0,0,.2],[.01,0,0],[.02,0,0],[.03,0,0]]
    result,_=run([17,-1,-1,91],centers=centers)
    np.testing.assert_array_equal(result,[17,91,91,91])


def test_invalid_horizontal_wall_seed_is_rejected():
    with pytest.raises(RuntimeError, match='horizontal wall'):
        run([5,-1,17],normals=[[0,1,0]]*3,walls={5})


def test_horizontal_wall_destination_is_forbidden():
    result,_=run([5,-1,17],normals=[[0,0,1],[0,1,0],[0,1,0]],walls={5})
    np.testing.assert_array_equal(result,[5,17,17])


def test_patch_without_any_seed_stays_explicitly_unresolved():
    result,metrics=run([-1,-1,-1])
    np.testing.assert_array_equal(result,[-1]*3)
    assert metrics['remaining_faces']==3


def test_missing_stage04_flags_is_compatible_but_present_flags_are_read_exactly(tmp_path):
    from plyfile import PlyData,PlyElement
    data=np.array([([0,1,2],1,10,20,30),([1,2,3],0,40,50,60)],
        dtype=[('vertex_indices','i4',(3,)),('is_hole_fill','u1'),('fill_r','u1'),('fill_g','u1'),('fill_b','u1')])
    p=tmp_path/'mesh.ply';PlyData([PlyElement.describe(data,'face')],text=False).write(p)
    flags,rgb=read_infill_attributes(p,2)
    np.testing.assert_array_equal(flags,[True,False]);np.testing.assert_allclose(rgb[0],[10/255,20/255,30/255])

    plain=np.array([([0,1,2],)],dtype=[('vertex_indices','i4',(3,))])
    PlyData([PlyElement.describe(plain,'face')],text=False).write(p)
    flags,rgb=read_infill_attributes(p,1)
    np.testing.assert_array_equal(flags,[False]);assert rgb is None


def test_tiny_face_normal_magnitude_cannot_bypass_wall_angle_rule():
    result,_=run([5,-1],normals=[[0,0,.001],[0,.001,0]],walls={5})
    np.testing.assert_array_equal(result,[5,-1])


def test_disconnected_same_surface_piece_gets_explicit_instance_association():
    labels=np.array([17,17,-1],np.int32)
    centers=np.array([[0,0,0],[.01,0,0],[.02,0,0]],float)
    result,metrics,links=assign_deferred_patches(labels,centers,np.tile([0.,0.,1.],(3,1)),
        np.ones(3),np.full((3,3),.5),np.array([0]),np.array([1]),np.array([0.,1.,0.]),set())
    np.testing.assert_array_equal(result,[17,17,17])
    assert metrics['spatially_associated_pieces']==1
    assert links.shape==(1,2) and links[0,0]==2 and links[0,1] in (0,1)


def test_disconnected_piece_on_a_different_plane_is_not_associated():
    labels=np.array([17,17,-1],np.int32)
    centers=np.array([[0,0,0],[.01,0,0],[0,0,.05]],float)
    result,_,links=assign_deferred_patches(labels,centers,np.tile([0.,0.,1.],(3,1)),
        np.ones(3),np.full((3,3),.5),np.array([0]),np.array([1]),np.array([0.,1.,0.]),set())
    np.testing.assert_array_equal(result,[17,17,-1]);assert len(links)==0


@pytest.mark.parametrize('point,color', [([.2,0,0],[.5,.5,.5]),([.02,0,0],[1.,1.,1.])])
def test_spatial_association_rejects_far_or_visually_different_pieces(point,color):
    labels=np.array([17,17,-1],np.int32)
    centers=np.array([[0,0,0],[.01,0,0],point],float)
    rgb=np.array([[.5,.5,.5],[.5,.5,.5],color])
    result,_,links=assign_deferred_patches(labels,centers,np.tile([0.,0.,1.],(3,1)),
        np.ones(3),rgb,np.array([0]),np.array([1]),np.array([0.,1.,0.]),set())
    np.testing.assert_array_equal(result,[17,17,-1]);assert len(links)==0
