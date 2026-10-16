import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'06c_semantic_segmentation'))
from planar_gap_completion import complete_planar_gaps
from detect_structural_classes import build_mesh_dual_graph
from refine_class_boundaries import compute_mesh_derivatives


def strip():
    vertices=np.array([[x,y,0.] for x in range(5) for y in (0.,1.)])
    triangles=np.array([[2*x,2*x+2,2*x+3] for x in range(4)]+[[2*x,2*x+3,2*x+1] for x in range(4)])
    labels=np.full(8,-1,dtype=int);labels[[0,4]]=17
    return vertices,triangles,labels


def run(vertices,triangles,labels,seed_id=17,scale=1,candidates=None):
    _,normals,centers,_=compute_mesh_derivatives(vertices,triangles,np.zeros_like(vertices))
    u,v=build_mesh_dual_graph(triangles)
    return complete_planar_gaps(labels,vertices,triangles,normals,centers,u,v,
        labels<=0 if candidates is None else candidates,labels==seed_id,.02*scale)


def test_unobserved_planar_gap_is_completed_without_altering_inputs():
    v,t,l=strip();before=l.copy();result,report=run(v,t,l)
    assert np.all(result==17) and report['filled_faces']==6
    np.testing.assert_array_equal(l,before)


def test_known_object_and_coplanar_conflict_are_preserved():
    v,t,l=strip();l[[3,7]]=103
    result,report=run(v,t,l)
    np.testing.assert_array_equal(result[l>0],l[l>0])
    assert result[1]==17 and result[2]==-1
    assert report['competing_boundary_faces']>0


def test_vertical_turn_is_not_filled():
    v,t,l=strip();v[6:,2]=v[6:,0]-2;v[6:,0]=2
    result,_=run(v,t,l)
    assert np.all(result[[2,3,6,7]]==-1)
    assert np.all(result[l==17]==17)


def test_a_large_triangle_with_one_vertex_off_plane_is_rejected():
    v,t,l=strip();v[5,2]=.05
    result,_=run(v,t,l)
    assert result[1]==-1  # Centroid is within tolerance, but the corner is not.


def test_reference_plane_does_not_drift_along_a_gently_bending_surface():
    v,t,l=strip();v[:,2]=np.maximum(v[:,0]-1,0)*.015
    result,_=run(v,t,l)
    assert result[1]==17 and result[3]==-1


def test_ineligible_unknowns_are_not_filled():
    v,t,l=strip();candidate=l<=0;candidate[[1,5]]=False
    result,_=run(v,t,l,candidates=candidate)
    np.testing.assert_array_equal(result,l)


def test_ids_rigid_transforms_and_units_do_not_change_completion():
    v,t,l=strip();expected,_=run(v,t,l)
    rotation=np.linalg.qr(np.random.default_rng(91).normal(size=(3,3)))[0]
    result,_=run((v@rotation.T)*100+np.array([7.,-33.,8.]),t,np.where(l==17,8051,l),seed_id=8051,scale=100)
    np.testing.assert_array_equal(result,np.where(expected==17,8051,expected))


def test_bad_tolerance_is_rejected():
    v,t,l=strip()
    with pytest.raises(ValueError):run(v,t,l,scale=0)


def test_single_undirected_edges_work_in_either_order():
    v,t,l=strip();_,normals,centers,_=compute_mesh_derivatives(v,t,np.zeros_like(v))
    u,w=build_mesh_dual_graph(t);take=u<w;u,w=u[take],w[take]
    result,_=complete_planar_gaps(l,v,t,normals,centers,w,u,l<=0,l==17,.02)
    assert np.all(result==17)


def test_generated_gap_normals_may_flip_without_changing_the_supported_plane():
    v,t,l=strip();t[l<=0]=t[l<=0,::-1]
    result,_=run(v,t,l)
    assert np.all(result==17)


def capped_box():
    import open3d as o3d
    mesh=o3d.geometry.TriangleMesh.create_box();mesh.compute_triangle_normals()
    v=np.asarray(mesh.vertices).copy();t=np.asarray(mesh.triangles).copy()
    top=np.asarray(mesh.triangle_normals)[:,2]>.9
    t=t[~top];labels=np.full(len(t),17,dtype=int)
    # A folded fan still closes the same original semantic opening.
    rim=[i for i,p in enumerate(v) if p[2]==1]
    rim=sorted(rim,key=lambda i:np.arctan2(v[i,1]-.5,v[i,0]-.5))
    v=np.vstack([v,[.5,.5,1.3]])
    t=np.vstack([t,[[rim[i],rim[(i+1)%4],8] for i in range(4)]])
    labels=np.r_[labels,np.full(4,-1)]
    return v,t,labels


def bounded(v,t,l,seeds=None):
    from planar_gap_completion import complete_bounded_gaps
    u,w=build_mesh_dual_graph(t)
    return complete_bounded_gaps(l,v,t,u,w,l<=0,l<=0,l==17 if seeds is None else seeds)


def test_folded_closed_cap_inherits_its_unanimous_boundary_owner():
    v,t,l=capped_box();result,report=bounded(v,t,l)
    assert np.all(result==17) and report['filled_faces']==4
    np.testing.assert_array_equal(result[l>0],l[l>0])


def test_mixed_cap_boundary_is_not_force_filled():
    v,t,l=capped_box();side=(v[t].mean(axis=1)[:,0]==0)&(l>0);l[side]=103
    result,_=bounded(v,t,l)
    np.testing.assert_array_equal(result,l)


def test_open_extension_has_no_enclosing_boundary():
    v,t,l=strip();result,_=bounded(v,t,l)
    np.testing.assert_array_equal(result,l)


def test_unknown_boundary_and_untrusted_owner_block_completion():
    v,t,l=capped_box();result,_=bounded(v,t,l,seeds=np.zeros(len(l),bool))
    np.testing.assert_array_equal(result,l)
    side=(v[t].mean(axis=1)[:,0]==0)&(l>0);l[side]=-1
    from planar_gap_completion import complete_bounded_gaps
    u,w=build_mesh_dual_graph(t)
    result,_=complete_bounded_gaps(l,v,t,u,w,np.arange(len(l))>=len(l)-4,l<=0,l==17)
    np.testing.assert_array_equal(result,l)


def test_bounded_completion_is_invariant_to_ids_rotations_and_units():
    v,t,l=capped_box();rotation=np.linalg.qr(np.random.default_rng(42).normal(size=(3,3)))[0]
    renamed=np.where(l==17,905,l);result,_=bounded(v@rotation.T*100+5,t,renamed,seeds=renamed==905)
    assert np.all(result==905)


def test_observed_unknown_inside_a_closed_patch_is_preserved():
    from planar_gap_completion import complete_bounded_gaps
    v,t,l=capped_box();u,w=build_mesh_dual_graph(t);candidates=l<=0;candidates[-1]=False
    result,_=complete_bounded_gaps(l,v,t,u,w,l<=0,candidates,l==17)
    assert result[-1]==-1 and np.all(result[-4:-1]==17)


def test_existing_foreign_label_inside_a_patch_vetoes_completion():
    from planar_gap_completion import complete_bounded_gaps
    v,t,l=capped_box();u,w=build_mesh_dual_graph(t);patches=l<=0;l[-1]=103
    result,_=complete_bounded_gaps(l,v,t,u,w,patches,l<=0,l==17)
    np.testing.assert_array_equal(result,l)


def mixed_surface_patch(fold=0.):
    # One coplanar side and three perpendicular sides of one unknown square.
    v=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
                [.5,.5,fold],[.5,-1.,0.],[1.,.5,1.],[.5,1.,1.],[0.,.5,1.]])
    t=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4],
                [1,0,5],[2,1,6],[3,2,7],[0,3,8]])
    return v,t,np.array([-1,-1,-1,-1,17,103,103,103])


def geometric(v,t,l,candidates=None,tolerance=.02,reverse_edges=False):
    from planar_gap_completion import complete_geometric_regions
    u,w=build_mesh_dual_graph(t)
    if reverse_edges:
        take=u<w;u,w=w[take],u[take]
    return complete_geometric_regions(l,v,t,u,w,l<=0 if candidates is None else candidates,tolerance)


def test_whole_region_prefers_coplanar_minority_over_three_wall_sides():
    v,t,l=mixed_surface_patch();before=l.copy()
    result,report=geometric(v,t,l)
    assert np.all(result[:4]==17) and report['filled_regions']==1
    np.testing.assert_array_equal(result[l>0],l[l>0])
    np.testing.assert_array_equal(l,before)


def test_folded_region_is_compared_as_one_surface():
    v,t,l=mixed_surface_patch(fold=.15)
    result,_=geometric(v,t,l)
    assert np.all(result[:4]==17)


def test_geometric_region_preserves_ineligible_faces_and_positive_objects():
    v,t,l=mixed_surface_patch();candidate=l<=0;candidate[0]=False
    result,_=geometric(v,t,l,candidate)
    assert result[0]==-1 and np.all(result[1:4]==17)
    np.testing.assert_array_equal(result[l>0],l[l>0])


def test_geometry_completion_is_invariant_to_ids_rotations_units_and_edge_order():
    v,t,l=mixed_surface_patch(fold=.15);rotation=np.linalg.qr(np.random.default_rng(42).normal(size=(3,3)))[0]
    renamed=np.where(l==17,905,np.where(l==103,2,l))
    result,_=geometric(v@rotation.T*100+5,t,renamed,tolerance=2,reverse_edges=True)
    assert np.all(result[:4]==905)
    np.testing.assert_array_equal(result[l>0],renamed[l>0])


def test_geometric_tie_still_assigns_one_neighbor_to_entire_region():
    v,t,l=mixed_surface_patch();v[6:]=[[2.,.5,0.],[.5,2.,0.],[-1.,.5,0.]]
    result,_=geometric(v,t,l)
    assert len(np.unique(result[:4]))==1 and result[0] in (17,103)


def test_geometric_region_without_any_positive_neighbor_does_not_invent_an_id():
    v,t,l=mixed_surface_patch();l[:]=-1
    result,report=geometric(v,t,l)
    np.testing.assert_array_equal(result,l)
    assert report['filled_faces']==0 and report['regions_without_labeled_boundary']==1


def test_geometric_completion_rejects_invalid_tolerance():
    v,t,l=mixed_surface_patch()
    with pytest.raises(ValueError):geometric(v,t,l,tolerance=np.nan)


def test_region_plane_fit_does_not_depend_on_triangle_subdivision():
    v,t,l=mixed_surface_patch(fold=.15);expected,reference=geometric(v,t,l)
    new_vertices=v.tolist();new_triangles=[]
    for triangle in t[:4]:
        center=len(new_vertices);new_vertices.append(v[triangle].mean(axis=0))
        new_triangles.extend([[triangle[k],triangle[(k+1)%3],center] for k in range(3)])
    subdivided=np.vstack([new_triangles,t[4:]])
    result,report=geometric(np.array(new_vertices),subdivided,np.r_[np.full(12,-1),l[4:]])
    assert np.all(result[:12]==expected[0])
    assert report['regions'][0]['plane_rms']==pytest.approx(reference['regions'][0]['plane_rms'],abs=1e-12)


def test_zero_area_region_falls_back_to_shared_contact_without_inventing_labels():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[0.,1.,0.]])
    t=np.array([[0,1,2],[1,0,3]]);l=np.array([-1,17])
    result,_=geometric(v,t,l)
    np.testing.assert_array_equal(result,[17,17])
