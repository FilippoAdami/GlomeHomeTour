import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '06c_semantic_segmentation'))
from mesh_prior_refinement import correct_confident_faces, refine_graph_labels, _sum_block_unaries


def test_confident_correction_keeps_unknowns_and_uncorroborated_faces():
    labels = np.array([10, 20, 10, -1, -1])
    priors = csr_matrix([[1., 9.], [9., 1.], [0., 8.], [0., 0.], [0., 6.]])
    views = csr_matrix([[3, 5], [1, 3], [0, 2], [0, 0], [0, 3]])
    result = correct_confident_faces(labels, priors, views, np.array([10, 20]))
    assert result.tolist() == [20, 20, 10, -1, 20]
    assert labels.tolist() == [10, 20, 10, -1, -1]


def test_outside_vocabulary_evidence_does_not_become_false_certainty():
    labels = np.array([10, 20])
    priors = csr_matrix([[0., 1., 99.], [0., 0., 100.]])
    views = csr_matrix([[0, 4, 8], [0, 0, 8]])
    result = correct_confident_faces(labels, priors, views, np.array([10, 20, 30]))
    assert np.array_equal(result, labels)


def test_many_low_quality_edge_samples_are_not_full_effective_views():
    labels=np.array([10,20])
    result=correct_confident_faces(labels,csr_matrix([[0.,.25],[6.,0.]]),
        csr_matrix([[0,5],[5,0]]),np.array([10,20]))
    assert result.tolist()==[10,10]


def test_source_renaming_and_column_permutation_preserve_the_decision():
    labels = np.array([10, 20])
    priors = csr_matrix([[1., 9.], [9., 1.]])
    views = csr_matrix([[3, 5], [5, 3]])
    first = correct_confident_faces(labels, priors, views, np.array([10, 20]))
    renamed = correct_confident_faces(np.array([8051, 71]), priors[:, [1, 0]],
                                     views[:, [1, 0]], np.array([71, 8051]))
    assert first.tolist() == [20, 10]
    assert renamed.tolist() == [71, 8051]


def test_invalid_evidence_is_rejected():
    labels = np.array([10])
    with pytest.raises(ValueError, match='finite'):
        correct_confident_faces(labels, csr_matrix([[np.nan]]), csr_matrix([[3]]), np.array([10]))
    with pytest.raises(ValueError, match='dimensions'):
        correct_confident_faces(labels, csr_matrix([[1., 2.]]), csr_matrix([[3]]), np.array([10]))


def refine_chain(labels, evidence, source_ids, device='cpu'):
    return refine_graph_labels(np.array(labels),csr_matrix(evidence),np.array(source_ids),
        np.array([1.,.01,1.]),np.tile([0.,1.,0.],(3,1)),np.zeros((3,3)),
        np.array([0,1]),np.array([1,2]),np.array([.1,.1]),.025,device=device)


def test_graph_removes_unsupported_seam_but_protects_observed_foreign_boundary():
    corrected, report = refine_chain([10,20,10],[[0,0,20]]*3,[10,20,30])
    assert corrected.tolist()==[10,10,10]
    assert report['converged']
    assert np.all(np.diff(report['energies'])<0)
    preserved, report = refine_chain([10,20,10],[[20,0,0],[0,0,20],[20,0,0]],[10,20,30])
    assert preserved.tolist()==[10,20,10]


def test_graph_preserves_unknowns_and_independent_instances():
    labels = np.array([10,-1,20])
    result,report = refine_chain(labels,[[10,0],[0,0],[0,10]],[10,20])
    assert np.array_equal(result,labels)
    assert report['converged']


def test_graph_is_invariant_to_source_names_and_metric_units():
    first,_ = refine_chain([10,20,10],[[0,0,20]]*3,[10,20,30])
    renamed,_ = refine_chain([71,8051,71],[[20,0,0]]*3,[999,8051,71])
    assert first.tolist()==[10,10,10] and renamed.tolist()==[71,71,71]
    scaled,_ = refine_graph_labels(np.array([10,20,10]),csr_matrix([[0,0,20]]*3),np.array([10,20,30]),
        np.array([1.,.01,1.])*100,np.tile([0.,1.,0.],(3,1)),np.zeros((3,3)),
        np.array([0,1]),np.array([1,2]),np.array([1.,1.]),.25,device='cpu')
    assert np.array_equal(scaled,first)


def test_graph_cpu_gpu_agree_when_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    cpu,_ = refine_chain([10,20,10],[[0,0,20]]*3,[10,20,30])
    gpu,report = refine_chain([10,20,10],[[0,0,20]]*3,[10,20,30],device='cuda')
    assert np.array_equal(cpu,gpu)
    assert report['converged'] and np.all(np.diff(report['energies'])<0)


def test_graph_handles_degenerate_geometry_without_changing_zero_area_faces():
    labels=np.array([10,20,10])
    result,report=refine_graph_labels(labels,csr_matrix([[0,0,20]]*3),np.array([10,20,30]),
        np.array([1.,0.,1.]),np.tile([0.,1.,0.],(3,1)),np.zeros((3,3)),
        np.array([0,1]),np.array([1,2]),np.array([0.,.1]),.025,device='cpu')
    assert np.array_equal(result,labels)
    assert report['converged']


def test_weak_isolated_observation_does_not_overrule_existing_surface():
    labels=np.array([10,10,10,20])
    result,report=refine_graph_labels(labels,csr_matrix([[20.,0.],[0.,.25],[20.,0.],[0.,20.]]),
        np.array([10,20]),np.array([1.,.01,1.,1.]),np.tile([0.,1.,0.],(4,1)),np.zeros((4,3)),
        np.array([0,1]),np.array([1,2]),np.array([.1,.1]),.025,device='cpu')
    assert np.array_equal(result,labels)
    assert report['converged']


@pytest.mark.parametrize('unobserved_middle', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_coherent_patch_escapes_single_face_local_minimum(unobserved_middle,device):
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            pytest.skip('GPU unavailable')
    labels=np.array([10,20,20,10])
    evidence=np.array([[0.,0.,20.]]*4)
    if unobserved_middle:
        evidence[2]=0
    result,report=refine_graph_labels(labels,csr_matrix(evidence),np.array([10,20,30]),
        np.array([1.,.01,.01,1.]),np.tile([0.,1.,0.],(4,1)),np.zeros((4,3)),
        np.array([0,1,2]),np.array([1,2,3]),np.full(3,.1),.025,device=device)
    assert result.tolist()==[10,10,10,10]
    assert report['converged']
    assert 'surface_block' in report['move_types']
    assert np.all(np.diff(report['energies'])<0)


def test_surface_block_preserves_a_real_observed_object_and_unobserved_geometry():
    labels=np.array([10,20,20,10])
    for evidence in ([[20.,0.,0.],[0.,0.,20.],[0.,0.,20.],[20.,0.,0.]],np.zeros((4,3))):
        result,report=refine_graph_labels(labels,csr_matrix(evidence),np.array([10,20,30]),
            np.array([1.,.01,.01,1.]),np.tile([0.,1.,0.],(4,1)),np.zeros((4,3)),
            np.array([0,1,2]),np.array([1,2,3]),np.full(3,.1),.025,device='cpu')
        assert np.array_equal(result,labels)
        assert report['converged']


def test_surface_block_is_invariant_to_source_names_and_metric_units():
    result,report=refine_graph_labels(np.array([8051,71,71,8051]),csr_matrix([[20.,0.,0.]]*4),
        np.array([999,71,8051]),np.array([1.,.01,.01,1.])*100,
        np.tile([0.,1.,0.],(4,1)),np.zeros((4,3)),np.array([0,1,2]),
        np.array([1,2,3]),np.ones(3),.25,device='cpu')
    assert result.tolist()==[8051]*4
    assert report['converged'] and 'surface_block' in report['move_types']


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_block_sums_match_reference_with_large_unsorted_and_empty_blocks(device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    rng=np.random.default_rng(26)
    component=np.r_[np.full(1700,2),np.zeros(50,dtype=int),np.full(15,4)]
    rng.shuffle(component)
    data=rng.uniform(-1,1,size=(len(component),4)).astype(np.float32)
    reference=np.column_stack([np.bincount(component,weights=data[:,j],minlength=6) for j in range(4)])
    actual=_sum_block_unaries(torch.as_tensor(data,device=device),component,6)
    np.testing.assert_allclose(actual.cpu().numpy(),reference,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_prior_region_recovers_an_object_after_background_absorption(device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    labels=np.array([10,10,30,10,20])
    evidence=csr_matrix([[10.,0.,0.],[0.,.75,0.],[0.,.75,0.],[10.,0.,0.],[0.,10.,0.]])
    result,report=refine_graph_labels(labels,evidence,np.array([10,20,30]),
        np.array([1.,.006,.006,1.,1.]),np.tile([0.,1.,0.],(5,1)),np.zeros((5,3)),
        np.array([0,1,2]),np.array([1,2,3]),np.full(3,.1),.025,device=device)
    assert result.tolist()==[10,20,20,10,20]
    assert 'prior_region' in report['move_types']
    assert report['converged'] and np.all(np.diff(report['energies'])<0)


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_unsupported_fragment_relaxes_but_unobserved_components_without_anchors_stay_put(device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    labels=np.array([10,20,20,10,20,30])
    evidence=csr_matrix([[10.,0.,0.],[0.,0.,0.],[0.,0.,0.],[10.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    result,report=refine_graph_labels(labels,evidence,np.array([10,20,30]),np.ones(6),
        np.tile([0.,1.,0.],(6,1)),np.zeros((6,3)),np.array([0,1,2,4]),
        np.array([1,2,3,5]),np.full(4,.1),.025,device=device)
    assert result.tolist()==[10,10,10,10,20,30]
    assert report['converged'] and np.all(np.diff(report['energies'])<0)


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('support_weight', [1.5,3.])
def test_corroborated_object_retains_its_unobserved_faces(device,support_weight):
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    result,report=refine_graph_labels(np.array([10,20,20,20,10]),
        csr_matrix([[10.,0.],[.5,0.],[0.,0.],[0.,support_weight],[10.,0.]]),np.array([10,20]),
        np.array([1.,1.,1.,.01,1.]),np.tile([0.,1.,0.],(5,1)),np.zeros((5,3)),
        np.array([0,1,2,3]),np.array([1,2,3,4]),np.array([.1,.2,.1,.1]),.025,device=device)
    assert result[2:4].tolist()==[20,20]
    assert report['converged'] and np.all(np.diff(report['energies'])<0)


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('own_support', [False,True])
def test_whole_fragment_can_move_across_an_internal_crease_unless_supported(device,own_support):
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    angle=np.radians(32)
    normals=np.array([[0.,1.,0.],[0.,1.,0.],[np.sin(angle),np.cos(angle),0.],
                      [np.sin(angle),np.cos(angle),0.]])
    evidence=np.array([[0.,0.,20.]]*4)
    if own_support:
        evidence[2]=[0.,.001,0.]
    result,report=refine_graph_labels(np.array([10,20,20,10]),csr_matrix(evidence),
        np.array([10,20,30]),np.array([1.,.01,.01,1.]),normals,np.zeros((4,3)),
        np.array([0,1,2]),np.array([1,2,3]),np.array([.08,.1,.08]),.025,device=device)
    assert result.tolist()==([10,20,20,10] if own_support else [10,10,10,10])
    assert ('component_block' in report['move_types']) is not own_support
    assert report['converged'] and np.all(np.diff(report['energies'])<0)


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('local_roughness', [False,True])
def test_reference_geometry_rejects_incompatible_component_but_keeps_local_roughness(device,local_roughness):
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    labels=np.array([10,10,10,10,10,20,20,20])
    reference=labels.copy()
    if not local_roughness:
        reference[4]=20
    normals=np.array([[0.,-1.,0.]]*4+[[1.,0.,0.]]*4)
    u=np.array([0,1,2,4,5,6]+([3] if local_roughness else []))
    v=np.array([1,2,3,5,6,7]+([4] if local_roughness else []))
    unconstrained,_=refine_graph_labels(labels,csr_matrix([[100.,0.]]*5+[[0.,100.]]*3),
        np.array([10,20]),np.array([1.,1.,1.,1.,.1,1.,1.,1.]),normals,np.zeros((8,3)),
        u,v,np.full(len(u),.1),.025,device=device)
    assert unconstrained[4]==10
    result,report=refine_graph_labels(labels,csr_matrix([[100.,0.]]*5+[[0.,100.]]*3),
        np.array([10,20]),np.array([1.,1.,1.,1.,.1,1.,1.,1.]),normals,np.zeros((8,3)),
        u,v,np.full(len(u),.1),.025,device=device,geometry_reference_labels=reference)
    assert result[4]==(10 if local_roughness else 20)
    assert result[:4].tolist()==[10]*4 and result[5:].tolist()==[20]*3
    assert report['converged'] and np.all(np.diff(report['energies'])<0)


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_reference_models_keep_small_rotated_instances_and_ignore_ids_and_units(device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('GPU unavailable')
    labels=np.array([10]*9+[20]*3)
    reference=labels.copy();reference[8]=20
    normals=np.array([[0.,-1.,0.]]*4+[[1.,0.,0.]]*4+[[0.,0.,1.]]*4)
    u=np.array([0,1,2,4,5,6,8,9,10]);v=u+1
    areas=np.array([1.]*4+[.001]*4+[.1]+[1.]*3)
    evidence=csr_matrix([[100.,0.]]*9+[[0.,100.]]*3)
    rotation=np.linalg.qr(np.random.default_rng(27).normal(size=(3,3)))[0]
    for transformed in [False,True]:
        ids=np.array([8051,171]) if transformed else np.array([10,20])
        mapped=np.where(labels==10,ids[0],ids[1])
        ref=np.where(reference==10,ids[0],ids[1])
        result,report=refine_graph_labels(mapped,evidence[:,::-1] if transformed else evidence,
            ids[::-1] if transformed else ids,areas*(100 if transformed else 1),
            normals@rotation.T if transformed else normals,np.zeros((12,3)),u,v,
            np.full(len(u),1. if transformed else .1),.25 if transformed else .025,
            device=device,geometry_reference_labels=ref)
        assert result.tolist()==[ids[0]]*8+[ids[1]]*4
        assert report['converged'] and np.all(np.diff(report['energies'])<0)


def test_geometry_reference_shape_is_validated():
    with pytest.raises(ValueError,match='geometry reference'):
        refine_graph_labels(np.array([10]),csr_matrix([[1.]]),np.array([10]),np.ones(1),
            np.array([[0.,1.,0.]]),np.zeros((1,3)),np.array([],dtype=int),
            np.array([],dtype=int),np.array([]),.025,geometry_reference_labels=np.array([10,10]))
