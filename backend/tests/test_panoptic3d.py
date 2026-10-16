"""Frozen-input identity, boundary, and topology regression proofs."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '06c_semantic_segmentation'))
from panoptic3d import (assign_faces, balance_votes, boundary_distance, boundary_edges, dual_graph,
                       object_metrics, object_source_aliases, observation_components, physical_ids, resolve_tracks, safe_removal, view_groups)
import run_panoptic3d as pipeline


def test_registered_depth_extent_abstains_without_deleting_or_inventing_geometry(monkeypatch):
    from panoptic3d import cached_mask_extent, extent_conflicts
    mask = np.ones((2, 2), np.uint8)
    cached = np.array([[1., 1.], [1., 0.]])
    extent = cached_mask_extent(mask, cached, .2)
    assert extent['lower_m'] == pytest.approx(.8)
    assert extent['upper_m'] == pytest.approx(1.2)
    assert cached_mask_extent(mask, np.array([[1., 0.], [0., np.nan]]), .2) is None
    assert extent_conflicts(np.array([3., 3., 1., 3.]), np.array([2, 0, 2, 2]),
        np.array([0, 0, 0, -1]), {0: extent}).tolist() == [True, False, False, False]
    monkeypatch.setattr(pipeline, 'cast_pixels', lambda *a: (
        np.array([0]), np.array([3.]), None, None, None))
    removed = np.array([False])
    counts = pipeline.dense_completeness(None, np.array([[0]]), None, None, np.array([1]),
        removed, np.array([1]), None, {'enabled': False}, {0: extent})
    assert counts[1]['uncertain_depth_extent'] == 1
    assert counts[1].get('geometrically_explained', 0) == 0
    assert not removed.any()


def test_native_depth_points_inverts_camera_and_mesh_transforms():
    camera = dict(K=np.eye(3), viewmat=np.eye(4))
    camera['viewmat'][:3, 3] = [1., 2., 3.]
    transform = np.eye(4)
    transform[:3, 3] = [4., 5., 6.]
    points = pipeline.native_depth_points(camera, np.array([0]), np.array([0]), np.array([2.]), transform)
    assert points[0] == pytest.approx([-4., -6., -7.])
    camera['viewmat'][:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    transform[:3, :3] = [[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]
    expected = np.array([2., 3., 4.])
    raw = transform[:3, :3] @ expected + transform[:3, 3]
    local = camera['viewmat'][:3, :3] @ raw + camera['viewmat'][:3, 3]
    recovered = pipeline.native_depth_points(camera, np.array([local[1]/local[2]-.5]),
        np.array([local[0]/local[2]-.5]), np.array([local[2]]), transform)
    assert recovered[0] == pytest.approx(expected)


def test_reviewed_object_extent_keeps_disconnected_pieces_and_competing_labels():
    from panoptic3d import restrict_instance_votes
    votes = sparse.csr_matrix([[0, 0, 0], [0, 2, 0], [0, 3, 0], [0, 4, 1], [0, 5, 0]])
    points = np.array([[0, 0, 0], [.1, 0, 0], [.9, 0, 0], [9, 0, 0], [0, 9, 0]])
    revised, withheld = restrict_instance_votes(votes, points, 1, [-.2, -.2, -.2], [1.2, .2, .2])
    assert withheld.tolist() == [3, 4]
    assert np.asarray(revised.argmax(axis=1)).ravel().tolist() == [0, 1, 1, 2, 0]
    assert votes[3, 1] == 4  # The original evidence stays reversible.


def test_silhouette_conflicts_require_diverse_visible_unobscured_background(tmp_path):
    import json
    (tmp_path / 'evidence').mkdir()
    frames = [dict(stem=s, group=g, pose=dict(trusted=trusted), depth=dict(enabled=True))
              for s, g, trusted in [('a', 0, True), ('b', 0, True), ('c', 1, True), ('weak', 2, False)]]
    (tmp_path / 'frames.json').write_text(json.dumps(frames))
    observations = [dict(observation_id=i, source_id=92, frame=f['stem'], pixels=10)
                    for i, f in enumerate(frames)]
    observations.append(dict(observation_id=4, source_id=251, frame='a', pixels=1))
    for frame in frames:
        state = np.array([1, 1, -1, 1, 1, 1, 1], np.int8)
        if frame['stem'] == 'c':
            state[5] = 2
        np.savez(tmp_path / 'evidence' / f"{frame['stem']}.npz", face=np.arange(7),
            observation=np.array([4, 5, 5, -1, 5, 5, 6]), labels=np.array([251, 3, 3, 0, 3, 3, 92]),
            state=state, distance=np.array([20, 20, 20, 20, 4, 20, 20]))
    # Include the competing source in the archived source vocabulary.
    observations.append(dict(observation_id=5, source_id=3, frame='a', pixels=10))
    observations.append(dict(observation_id=6, source_id=92, frame='a', pixels=10))
    result = pipeline.silhouette_conflicts(tmp_path, np.ones(7, np.int32),
        np.array([1, 1, 1, 1, 1, 2, 2]), observations, {1})
    assert result.tolist() == [0, 2, 0, 0, 0, 1, 2]
    assert not pipeline.silhouette_conflicts(tmp_path, np.ones(7, np.int32),
        np.array([1, 1, 1, 1, 1, 2, 2]), observations, set()).any()


def test_completeness_uses_exposed_surface_depth_after_cleanup(monkeypatch):
    hits = iter([(np.array([0]), np.array([3.]), None, None, None),
                 (np.array([0]), np.array([1.]), None, None, None)])
    monkeypatch.setattr(pipeline, 'cast_pixels', lambda *a: next(hits))
    counts = pipeline.dense_completeness(None, np.array([[0]]), None, None, np.array([2, 1]),
        np.array([True, False]), np.array([1]), None, None, {0: dict(lower_m=.8, upper_m=1.2)})
    assert counts[1]['geometrically_explained'] == 1
    assert counts[1].get('uncertain_depth_extent', 0) == 0


def test_pose_support_counts_triangulated_tracks_not_unmatched_features(tmp_path):
    sparse_dir = tmp_path / 'sparse/0'
    sparse_dir.mkdir(parents=True)
    tracks = ' '.join(f'0 0 {i}' for i in range(100))
    (sparse_dir / 'images.txt').write_text('# COLMAP\n1 1 0 0 0 0 0 0 1 weak.jpg\n0 0 -1 1 1 7\n'
        f'2 1 0 0 0 0 0 0 1 strong.jpg\n{tracks}\n')
    poses = pipeline.pose_support(tmp_path)
    assert poses['weak']['registered_points'] == 1 and not poses['weak']['trusted']
    assert poses['strong']['registered_points'] == 100 and poses['strong']['trusted']


def observation(frame, source=1, embedding=(1., 0.)):
    return dict(frame=frame, source_id=source, embedding=list(embedding))


def test_merge_synonymous_tracks_on_shared_geometry_and_appearance():
    support = sparse.csr_matrix([[1, 1, 1, 0], [1, 1, 1, 1]])
    roots, audit = resolve_tracks(support, [observation('a', 7), observation('b', 93)], [])
    assert roots[0] == roots[1]
    assert audit[0]['shared_faces'] == 3


def test_repeated_similar_objects_and_transitive_cannot_link_stay_distinct():
    support = sparse.csr_matrix(np.ones((3, 5)))
    roots, audit = resolve_tracks(support, [observation('a'), observation('b'), observation('c')], [(0, 2)])
    assert roots[0] != roots[2]
    assert len(set(roots)) == 2
    assert any(d['decision'] == 'separate' for d in audit)


def test_appearance_or_proximity_alone_cannot_merge():
    roots, _ = resolve_tracks(sparse.eye(2, format='csr'), [observation('a'), observation('b')], [])
    assert len(set(roots)) == 2
    roots, _ = resolve_tracks(sparse.csr_matrix(np.ones((2, 4))),
                              [observation('a'), observation('b', embedding=(0., 1.))], [])
    assert len(set(roots)) == 2


def test_cached_local_feature_tracks_bridge_disjoint_surfaces_and_source_prototypes():
    obs = [observation('a', 123), observation('b', 140, embedding=(0., 1.))]
    features = sparse.csr_matrix([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
    roots, audit = resolve_tracks(sparse.eye(2, format='csr'), obs, [], feature_support=features)
    assert roots[0] == roots[1]
    assert audit[0]['evidence'] == 'feature_tracks'
    assert audit[0]['shared_feature_tracks'] == 3 and audit[0]['shared_faces'] == 0
    roots, _ = resolve_tracks(sparse.eye(2, format='csr'), obs, [], feature_support=features[:, :2])
    assert roots[0] != roots[1]


def test_feature_track_merges_preserve_established_transitive_and_rejected_separation():
    obs = [observation('a', 1), observation('b', 1), observation('c', 2, embedding=(0., 1.))]
    support = sparse.csr_matrix([[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]])
    features = sparse.csr_matrix([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    roots, audit = resolve_tracks(support, obs, [(0, 2)], feature_support=features)
    assert roots[0] == roots[1] and roots[1] != roots[2]
    assert audit[-1]['evidence'] == 'feature_tracks' and audit[-1]['decision'] == 'separate'
    roots, _ = resolve_tracks(sparse.eye(3, format='csr'), obs, [], feature_support=features,
                              rejected_pairs=[dict(source_ids=[1, 2])])
    assert roots[1] != roots[2]
    roots, _ = resolve_tracks(sparse.eye(2, format='csr'), [observation('same'), observation('same')], [],
                              feature_support=sparse.csr_matrix(np.ones((2, 3))))
    assert roots[0] != roots[1]


def test_reviewed_alias_still_requires_geometry_and_covisible_separation():
    obs = [observation('a', 72), observation('b', 159, embedding=(.69, np.sqrt(1-.69**2)))]
    support = sparse.csr_matrix(np.ones((2, 4)))
    review = [dict(observations=[dict(frame='a', source_id=72), dict(frame='b', source_id=159)])]
    roots, _ = resolve_tracks(support, obs, [], reviewed_pairs=review)
    assert roots[0] == roots[1]
    roots, _ = resolve_tracks(support, obs, [(0, 1)], reviewed_pairs=review)
    assert roots[0] != roots[1]
    roots, _ = resolve_tracks(sparse.eye(2, format='csr'), obs, [], reviewed_pairs=review)
    assert roots[0] != roots[1]
    # The same source IDs elsewhere do not inherit this observation's exception.
    roots, _ = resolve_tracks(sparse.csr_matrix(np.ones((4, 4))),
        obs + [observation('c', 72), observation('d', 159, embedding=obs[1]['embedding'])],
        [(0, 2), (1, 3)], reviewed_pairs=review)
    assert roots[2] != roots[3]


def test_reviewed_disconnected_observations_and_rejected_source_merges():
    obs = [observation('a', 7), observation('b', 7)]
    review = [dict(observations=[dict(frame='a'), dict(frame='b')], reason='Same rigid object, disjoint visible pieces')]
    roots, audit = resolve_tracks(sparse.eye(2, format='csr'), obs, [], disconnected_pairs=review)
    assert roots[0] == roots[1] and audit[0]['shared_faces'] == 0
    roots, _ = resolve_tracks(sparse.eye(2, format='csr'), obs, [(0, 1)], disconnected_pairs=review)
    assert roots[0] != roots[1]
    roots, _ = resolve_tracks(sparse.csr_matrix(np.ones((2, 4))),
        [observation('a', 13), observation('b', 27)], [], rejected_pairs=[dict(source_ids=[13, 27])])
    assert roots[0] != roots[1]


def test_physical_identity_retains_parts_and_rejects_cycles():
    entries = [dict(instance_id=1, parent_instance_id=None), dict(instance_id=2, parent_instance_id=1),
               dict(instance_id=3, parent_instance_id=2), dict(instance_id=4, parent_instance_id=None)]
    assert physical_ids(entries).tolist() == [0, 1, 1, 1, 4]
    entries[0]['parent_instance_id'] = 3
    with pytest.raises(ValueError):
        physical_ids(entries)


def test_alias_scoring_does_not_collapse_other_objects_using_same_source():
    entries = [dict(instance_id=1, physical_instance_id=1, source_ids=[32, 42]),
               dict(instance_id=2, physical_instance_id=2, source_ids=[42])]
    mapping = object_source_aliases(entries, 42)
    assert mapping[1, 42] == 32
    assert mapping[2, 42] == 42
    assert mapping[0, 42] == 42
    # Reference and both candidates use this same static (object, source) lookup.
    assert mapping[1, 32] == mapping[1, 42]
    assert mapping[2, 42] != mapping[1, 42]


def test_region_transfer_keeps_unmatched_piece_uncertain(monkeypatch):
    mask = np.zeros((12, 12), bool)
    mask[1:4, 1:4] = True
    mask[8:11, 8:11] = True
    def cast_pixels(scene, camera, y, x):
        faces = np.resize([1, 2, 3], len(y)) if y.min() < 5 else np.full(len(y), 9)
        return faces, None, None, y, x
    monkeypatch.setattr(pipeline, 'cast_pixels', cast_pixels)
    templates = {4: [dict(frame='anchor', mask_index=0, faces=np.array([1, 2, 3]))]}
    regions = pipeline.anchored_regions(mask, 4, 'another_frame', 0, templates, None, None)
    assert [key for key, _ in regions] == [0, 1]
    assert np.array_equal(np.logical_or.reduce([region for _, region in regions]), mask)


def test_area_balance_protects_small_supported_object_without_inventing_labels():
    votes = sparse.csr_matrix(np.vstack([np.tile([0, 1, 0], (100, 1)), [0, 4, 3]]))
    balanced, area = balance_votes(votes, np.ones(101), .5)
    assert balanced.argmax(axis=1).A[-1, 0] == 2
    assert np.array_equal(votes.nonzero()[0], balanced.nonzero()[0])
    assert np.array_equal(votes.nonzero()[1], balanced.nonzero()[1])


def test_disconnected_pieces_share_observation_identity():
    support = sparse.csr_matrix([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    roots, _ = resolve_tracks(support, [observation('a'), observation('b')], [])
    assert roots[0] == roots[1]
    votes = sparse.csr_matrix(np.tile([0, 1], (6, 1)))
    labels, _ = assign_faces(votes, sparse.csr_matrix((6, 6)), np.zeros((6, 3)), np.zeros((6, 3)))
    assert labels.tolist() == [1] * 6


def test_reviewed_distinct_objects_inside_one_mask_split_without_losing_pixels():
    mask = np.zeros((12, 12), bool)
    mask[1:4, 1:4] = True
    mask[8:11, 8:11] = True
    assert len(observation_components(mask)) == 1
    parts = observation_components(mask, split=True)
    assert len(parts) == 2
    assert np.array_equal(np.logical_or.reduce([p for _, p in parts]), mask)
    assert not np.any(parts[0][1] & parts[1][1])


def test_contested_boundary_smooths_only_supported_labels_and_preserves_small_core():
    graph = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    votes = sparse.csr_matrix([[0, 100, 0], [0, .49, .51], [0, .01, .99]])
    normals = np.tile([0., 0., 1.], (3, 1))
    rgb = np.zeros((3, 3))
    labels, confidence = assign_faces(votes, graph, normals, rgb, strength=.2)
    assert labels[0] == 1 and labels[2] == 2  # Even a one-triangle object core survives.
    assert np.all(votes[np.arange(3), labels].A.ravel() > 0)
    unsupported = sparse.csr_matrix([[0, 1, 0], [0, 0, 1], [0, 0, 1]])
    labels, _ = assign_faces(unsupported, graph, normals, rgb, strength=10)
    assert labels.tolist() == [1, 2, 2]


def test_unknown_geometry_stays_unknown():
    graph = sparse.csr_matrix([[0, 1], [1, 0]])
    labels, _ = assign_faces(sparse.csr_matrix([[0, 1], [0, 0]]), graph,
                             np.tile([0, 0, 1.], (2, 1)), np.zeros((2, 3)))
    assert labels.tolist() == [1, 0]


def test_confidence_describes_final_smoothed_label():
    graph = sparse.csr_matrix([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    labels, confidence = assign_faces(sparse.csr_matrix([[0, .49, .51], [0, 1, 0], [0, 1, 0]]),
        graph, np.tile([0., 0., 1.], (3, 1)), np.zeros((3, 3)), strength=.2)
    assert labels[0] == 1
    assert confidence[0] == pytest.approx(.49)


def test_floater_removal_needs_multiple_views_and_zero_support():
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    removed, _ = safe_removal(triangles, dual_graph(triangles),
                              np.array([True, False, False]), np.array([5, 3, 2]), np.zeros(3))
    assert removed.tolist() == [False, True, False]
    removed, _ = safe_removal(triangles, dual_graph(triangles),
                              np.zeros(3, bool), np.full(3, 3), np.array([1, 0, 0]))
    assert removed.tolist() == [False, True, True]


def test_hole_prevention_keeps_attached_unsupported_surfaces():
    triangles = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
    graph = dual_graph(triangles)
    removed, _ = safe_removal(triangles, graph, np.array([True, False, False]),
                              np.full(3, 10), np.zeros(3))
    assert removed.tolist() == [False, False, True]
    before = {tuple(x) for x in boundary_edges(triangles)}
    assert all(tuple(x) in before for x in boundary_edges(triangles[~removed]))
    # Vertex-only attachment must also be retained.
    pinched = np.array([[0, 1, 2], [2, 3, 4]])
    removed, _ = safe_removal(pinched, dual_graph(pinched), np.array([True, False]),
                              np.full(2, 10), np.zeros(2))
    assert not removed.any()
    chain = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    removed, _ = safe_removal(chain, dual_graph(chain), np.array([True, False, False]),
                              np.full(3, 10), np.zeros(3))
    assert not removed.any()


def test_missing_geometry_is_recall_failure_unknown_is_not_negative():
    truth = np.array([[1, 1, 0]])
    predicted = np.array([[1, 0, 1]])
    result = object_metrics(truth, predicted, np.array([[True, False, True]]), [1])[1]
    assert result['tp'] == 1 and result['fn'] == 1 and result['missing'] == 1
    assert result['fp'] == 0
    assert result['boundary_truth'] == 0  # Unknown neighbours cannot establish a true boundary.


def test_boundary_distance_is_symmetric_and_preserves_thin_labels():
    labels = np.zeros((7, 7), np.int32)
    labels[:, 3] = 1
    d = boundary_distance(labels)
    assert np.all(d[:, 2:5] == 0)


def test_redundant_views_and_output_write_boundary(tmp_path):
    camera = dict(viewmat=np.eye(4))
    different = dict(viewmat=np.eye(4))
    different['viewmat'][0, 3] = 1
    assert view_groups([camera, camera, different]).tolist() == [0, 0, 1]
    for name in ('../inputs', 'inputs', 'panoptic3d_x/../inputs', ''):
        with pytest.raises(ValueError):
            pipeline.output_directory(tmp_path, name)
    assert pipeline.output_directory(tmp_path, 'panoptic3d_test').parent == tmp_path


def test_ray_depth_is_camera_z_and_transform_is_applied_once():
    vertices = np.array([[-20, -20, 2], [20, -20, 2], [0, 20, 2.]])
    transform = np.eye(4)
    transform[2, 3] = 1
    scene = pipeline.ray_scene(vertices, np.array([[0, 1, 2]]), transform)
    camera = dict(width=6, height=4, K=np.array([[3, 0, 3], [0, 3, 2], [0, 0, 1.]]), viewmat=np.eye(4))
    _, depth, _, _, _ = pipeline.cast(scene, camera, 2)
    assert np.allclose(depth, 3)
    assert np.all(vertices[:, 2] == 2)


def test_depth_disagreement_calibration_keeps_occlusion_separate_from_free_space():
    mesh = np.ones((20, 20), np.float32)
    cached = mesh.copy()
    cached[0, 0] = 2
    cached[1, 1] = .5
    state, calibration = pipeline.depth_evidence(mesh, cached, np.ones(mesh.shape, bool), np.ones(mesh.shape, bool))
    assert state[0, 0] == 1 and state[1, 1] == -1 and state[2, 2] == 0
    assert calibration['enabled']


def test_3d_only_end_to_end_preserves_frozen_files_and_catalog(tmp_path, monkeypatch):
    import cv2
    import torch
    # Fail if any loaded neural model is called; no downloading or prediction is needed.
    monkeypatch.setattr(torch.nn.Module, '__call__', lambda *a, **k: pytest.fail('prohibited model call'))
    stage = tmp_path / '06c_semantic_segmentation'
    inputs = stage / 'inputs'
    (inputs / 'images').mkdir(parents=True)
    (inputs / 'sparse/0').mkdir(parents=True)
    camera_text = '1 PINHOLE 12 12 8 8 6 6\n'
    (inputs / 'sparse/0/cameras.txt').write_text(camera_text)
    points = ' '.join(f'0 0 {i}' for i in range(100))
    (inputs / 'sparse/0/images.txt').write_text(f'1 1 0 0 0 0 0 0 1 frame_00000.jpg\n{points}\n'
        '2 1 0 0 0 0 0 0 1 frame_00001.jpg\n0 0 1\n')
    cv2.imwrite(str(inputs / 'images/frame_00000.jpg'), np.full((12, 12, 3), 100, np.uint8))
    cv2.imwrite(str(inputs / 'images/frame_00001.jpg'), np.full((12, 12, 3), 100, np.uint8))
    vertices = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    pipeline.export_mesh(inputs / 'mesh.ply', vertices, triangles, np.ones(2, np.int32))
    np.save(inputs / 'T_mesh_to_raw.npy', np.eye(4))
    frozen = stage / 'full362_final_pipeline/npz_data'
    frozen.mkdir(parents=True)
    labels = np.where(np.indices((12, 12))[0] > np.indices((12, 12))[1], 2, 1).astype('int32')
    labels[0, 0] = 7
    masks = np.stack([labels == 1, labels == 2, labels == 7]).astype('uint8')
    np.savez(frozen / 'frame_00000_final.npz', labels=labels, masks=masks, ids=[1, 2, 7], embeddings=np.eye(3))
    np.savez(frozen / 'frame_00001_final.npz', labels=np.full((12, 12), 9),
        masks=np.ones((1, 12, 12), np.uint8), ids=[9], embeddings=np.array([[1., 0., 0.]]))
    baseline = stage / 'mesh_objects_step3'
    baseline.mkdir()
    pipeline.export_mesh(baseline / 'mesh_cleaned_no_floaters.ply', vertices, triangles, np.ones(2, np.int32))
    originals = {p: pipeline.digest(p) for p in stage.rglob('*') if p.is_file()}
    import json
    review = tmp_path / 'review.json'
    review.write_text(json.dumps(dict(mesh_sha256=pipeline.digest(inputs / 'mesh.ply'),
        anchor_archive_sha256={'frame_00000_final.npz': pipeline.digest(frozen / 'frame_00000_final.npz')},
        part_of=[dict(parent_source=1, child_source=2, frame='frame_00000', role='panel', reason='Synthetic attached part')],
        unresolved_masks=[dict(frame='frame_00000', mask_index=2, source_id=7, reason='Synthetic conflated group')])) )
    monkeypatch.setattr(pipeline, 'REVIEW', {'frame_00000': 'synthetic small-object and missing support'})
    out = pipeline.output_directory(stage, 'panoptic3d_test')
    out.mkdir()
    pipeline.collect(tmp_path, out, 3, review)
    collected = json.loads((out / 'observations.json').read_text())
    with pytest.raises(ValueError, match='reliable registered cached depth'):
        pipeline.reviewed_object_bounds(tmp_path, dict(object_extents=[dict(
            complete_object_coverage=True, reason='Synthetic complete view',
            observations=[dict(frame='frame_00000', mask_index=0, source_id=1)])]),
            collected, np.ones(len(collected), np.int32), np.eye(4))
    pipeline.solve(tmp_path, out, .15, review)
    pipeline.evaluate(tmp_path, out, 3)
    import json
    manifest = json.loads((out / 'instances_manifest.json').read_text())
    assert {source for e in manifest['instances'] for source in e['source_ids']} == {1, 2, 7, 9}
    weak = next(e for e in manifest['instances'] if e['source_ids'] == [9])
    assert weak['geometry_triangles'] == 0
    assert weak['completeness_status'] == 'unreliable_pose'
    assert weak['reprojection']['pixels'] == 0
    support = sparse.load_npz(out / 'observation_support.npz')
    assert support[weak['observation_ids']].nnz == 0
    frames = json.loads((out / 'frames.json').read_text())
    assert not frames[1]['pose']['trusted'] and frames[1]['depth'] is None
    metrics = json.loads((out / 'metrics.json').read_text())
    assert '9' not in metrics['per_object']['trusted_source']['revised']
    assert metrics['per_object']['all_source']['revised']['9']['pixels'] > 0
    assert any(e['geometry_triangles'] == 0 for e in manifest['instances'])
    tiny = next(e for e in manifest['instances'] if e['source_ids'] == [7])
    assert tiny['reprojection']['pixels'] == 1  # Misses the stride-3 evaluation lattice.
    assert tiny['completeness_status'] == 'missing_geometry'
    assert tiny['identity_status'] == 'unresolved_within_mask_group'
    assert all(pipeline.digest(p) == h for p, h in originals.items())
    assert (out / 'segmented_mesh.ply').is_file()
    assert (out / 'geometry_audit.npz').is_file()
    from plyfile import PlyData
    faces = PlyData.read(str(out / 'segmented_mesh.ply'))['face'].data
    assert len(set(faces['instance_id'])) == 1
    assert len(set(faces['part_id'])) == 2
    child = next(e for e in manifest['instances'] if e['source_ids'] == [2])
    assert child['entry_kind'] == 'part' and child['part_role'] == 'panel'
    configuration = json.loads((out / 'configuration.json').read_text())
    configuration['source_files']['panoptic3d.py'] = '0' * 64
    (out / 'configuration.json').write_text(json.dumps(configuration))
    with pytest.raises(RuntimeError, match='implementation changed'):
        pipeline.evaluate(tmp_path, out, 3)
