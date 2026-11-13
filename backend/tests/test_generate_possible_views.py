import importlib.util
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
spec = importlib.util.spec_from_file_location('possible_views', Path(sys.path[0]) / 'generate_possible_views.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_targets_do_not_carry_scans_to_missing_levels():
    floors = [dict(scan_id=s, floor_level_name=l, plane_id=s+l, scan_level_name='BASE', height_group_index=h)
              for s,l,h in [('scan_0','BASE',0),('scan_1','BASE',0),('scan_0','BASEa',1)]]
    result = m.targets({'floor_planes':floors, 'ceiling_planes':[]})
    assert [(x['scan_id'], x['floor_level_name']) for x in result] == [('scan_0','BASE'),('scan_0','BASEa'),('scan_1','BASE')]
    assert result[1]['floor_plane_ids'] == ['scan_0BASEa']


def test_clip_band_preserves_source_and_interpolates_color():
    v=np.array([[0,-1,0],[1,1,0],[0,3,0],[5,-2,0],[6,-1,0],[5,-1,1]],np.float32)
    f=np.array([[0,1,2],[3,4,5]],np.int32); c=np.column_stack((np.full(6,100),np.arange(6)*20,np.zeros(6),np.full(6,255))).astype(np.uint8)
    out, faces, colors, ids=m.clip_band(v,f,c,(np.array([0,1,0]),0),(np.array([0,-1,0]),2))
    assert np.all(out[:,1]>=0) and np.all(out[:,1]<=2)
    assert len(faces)>0 and set(ids)=={0}
    assert np.any((colors[:,1]>0)&(colors[:,1]<20))


def test_cameras_frame_all_bounds_and_are_deterministic():
    bounds=np.array([[0,0,0],[4,3,6]],float)
    cams=m.cameras(bounds,bounds)
    assert len(cams)==4 and cams==m.cameras(bounds,bounds)
    assert len({tuple(np.round(c['eye'],4)) for c in cams})==4
    corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    for c, (dx,dz) in zip(cams, [(-1,-1),(1,-1),(1,1),(-1,1)]):
        assert np.allclose(c['target'], bounds.mean(axis=0))
        delta=np.asarray(c['eye'])-c['target']
        assert np.sign(delta[0]) == dx and np.sign(delta[2]) == dz
        assert np.arctan2(delta[1], np.hypot(delta[0], delta[2])) == pytest.approx(np.deg2rad(35))
        assert np.max(np.abs((corners-c['target']) @ np.array(c['right']))) <= c['extent']/2
        assert np.max(np.abs((corners-c['target']) @ np.array(c['up']))) <= c['extent']/2


def test_selection_combines_and_deduplicates_coverage():
    visibility=[np.array([0,2]),np.array([0,1]),np.array([1,3])]
    chosen, coverage, attainable, reason=m.select(visibility,np.ones(4),np.array([0,0,1,1]),2,1,.001)
    assert chosen == [0,2] and list(coverage)==[1,1] and list(attainable)==[1,1]
    assert reason=='coverage_target_reached'


def test_render_maps_pixels_to_primitive():
    v=np.array([[-1,0,-1],[1,0,-1],[0,0,1]],np.float32); f=np.array([[0,1,2]],np.uint32)
    c=np.tile(np.array([255,0,0,255],np.uint8),(3,1))
    scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    cam=dict(eye=[0,2,0],target=[0,0,0],right=[1,0,0],up=[0,0,1],extent=3)
    image,ids=m.render(scene,v,f,c,cam,64)
    assert set(np.unique(ids))=={-1,0} and np.all(image[ids==0]==[255,0,0])


def test_failure_preserves_previous_output(tmp_path):
    alignment=tmp_path/'scene'/'multi_room_alignment'; alignment.mkdir(parents=True)
    output=tmp_path/'scene'/'artifacts'/'views'; output.mkdir(parents=True)
    (output/'old').write_text('complete')
    metadata=tmp_path/'scene'/'metadata.json'; metadata.write_text('{"floor_planes": [], "ceiling_planes": []}')
    with pytest.raises(ValueError): m.run(alignment,metadata,output,64)
    assert (output/'old').read_text()=='complete'


def test_macro_manifest_and_failure_after_staging(tmp_path, monkeypatch):
    import json
    alignment=tmp_path/'scene'/'multi_room_alignment'; alignment.mkdir(parents=True)
    output=tmp_path/'scene'/'artifacts'/'views'
    metadata=tmp_path/'scene'/'metadata.json'
    floor=dict(scan_id='scan_0',floor_level_name='BASE',plane_id='floor',scan_level_name='BASE',height_group_index=0,
               plane_equation=[0,1,0,0],mean_elevation_m=0,support_area_m2=1,source_mesh='floor.ply')
    metadata.write_text(json.dumps({'floor_planes':[floor], 'ceiling_planes':[]}))
    v=np.array([[-1,0,-1],[1,0,-1],[0,0,1]],np.float32); f=np.array([[0,1,2]],np.int32)
    c=np.tile(np.array([255,0,0,255],np.uint8),(3,1))
    monkeypatch.setattr(m,'_read_mesh',lambda path:(v,f,c))
    result=m.run(alignment,metadata,output,64)
    assert result['schema_version']==3 and result['status']=='complete' and len(result['candidates'])==4
    assert result['settings']['coverage_excludes_supplemental_floor_faces'] is True
    assert result['settings']['cutaway_sampling_resolution'] == 64
    assert result['settings']['geometry_archive_compression'] == 'none'
    assert result['slices'][0]['surface_area_m2'] == pytest.approx(2.0)
    assert result['slices'][0]['upper_cut']['fallback'] is True
    assert result['slices'][0]['upper_cut']['plane_id'] is None
    assert (output / result['slices'][0]['contact_sheet']).is_file()
    assert (output/result['candidates'][0]['image']).is_file()
    assert (output/'scan_levels/scan_0/BASE/slice.json').is_file()
    saved = np.load(output / result['slices'][0]['slice_mesh'])
    assert np.array_equal(saved['source_face_id'], [0, 0])
    assert np.array_equal(saved['source_mesh_index'], [0, 1])
    assert np.array_equal(saved['colors'], np.concatenate((c, c)))
    assert np.all(saved['floor_protected'])
    assert result['slices'][0]['source_meshes'][1]['path'].endswith('floor.ply')
    from zipfile import ZIP_STORED, ZipFile
    with ZipFile(output / result['slices'][0]['slice_mesh']) as archive:
        assert all(member.compress_type == ZIP_STORED for member in archive.infolist())
    before=(output/'generate_possible_views_manifest.json').read_text()
    original_rename = Path.rename
    def fail_publish(path, target):
        if path.name == 'result' and Path(target) == output:
            raise OSError('publication failed')
        return original_rename(path, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'rename', fail_publish)
        with pytest.raises(OSError, match='publication failed'):
            m.run(alignment,metadata,output,64)
    assert (output/'generate_possible_views_manifest.json').read_text()==before
    monkeypatch.setattr(m,'cutaway_faces',lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('render failed')))
    with pytest.raises(RuntimeError): m.run(alignment,metadata,output,64)
    assert (output/'generate_possible_views_manifest.json').read_text()==before


def test_refuses_to_replace_unowned_output(tmp_path):
    import json
    alignment=tmp_path/'scene'/'multi_room_alignment'; alignment.mkdir(parents=True)
    output=tmp_path/'scene'/'artifacts'/'other'; output.mkdir(parents=True)
    (output/'keep').write_text('keep')
    floor=dict(scan_id='scan_0',floor_level_name='BASE',plane_id='floor',scan_level_name='BASE',height_group_index=0)
    metadata=tmp_path/'scene'/'metadata.json'; metadata.write_text(json.dumps({'floor_planes':[floor],'ceiling_planes':[]}))
    with pytest.raises(FileExistsError): m.run(alignment,metadata,output,64)
    assert (output/'keep').read_text()=='keep'


def test_composite_pixels_map_to_actual_scan_face():
    v=np.array([[-1,0,-1],[0,0,-1],[-.5,0,1], [0,0,-1],[1,0,-1],[.5,0,1]],np.float32)
    faces=np.array([[0,1,2],[3,4,5]],np.uint32)
    colors=np.array([[255,0,0,255]]*3+[[0,0,255,255]]*3,np.uint8)
    scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(faces))
    camera=dict(eye=[0,2,0],target=[0,0,0],right=[1,0,0],up=[0,0,1],extent=3)
    image,pixels=m.render(scene,v,faces,colors,camera,64)
    owners=np.array([0,1]); source=np.array([42,17])
    assert {tuple(x) for x in image[pixels>=0]} == {(255,0,0),(0,0,255)}
    assert set(zip(owners[pixels[pixels>=0]],source[pixels[pixels>=0]])) == {(0,42),(1,17)}


def test_rejects_unsafe_slice_names():
    floor=dict(scan_id='../outside',floor_level_name='BASE',plane_id='floor',scan_level_name='BASE',height_group_index=0)
    with pytest.raises(ValueError, match='Invalid scan_id'):
        m.targets({'floor_planes':[floor], 'ceiling_planes':[]})


def test_render_across_row_batches_matches_single_raycast():
    v=np.array([[-1,0,-1],[1,0,-1],[0,0,1]],np.float32); f=np.array([[0,1,2]],np.uint32)
    colors=np.tile(np.array([100,150,200,255],np.uint8),(3,1))
    scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    camera=dict(eye=[0,2,0],target=[0,0,0],right=[1,0,0],up=[0,0,1],extent=3)
    image,pixels=m.render(scene,v,f,colors,camera,256)
    xy=((np.arange(256)+.5)/256-.5)*3
    rays=np.empty((256,256,6),np.float32)
    rays[:,:,:3]=np.array([0,2,0])+xy[None,:,None]*[1,0,0]-xy[:,None,None]*[0,0,1]
    rays[:,:,3:]=[0,-1,0]
    reference=scene.cast_rays(o3d.core.Tensor(rays))['primitive_ids'].numpy()
    assert np.array_equal(pixels==0, reference==0)
    assert np.all(image[pixels==0]==[100,150,200])


def test_cutaway_removes_only_first_backfaces_and_nearby_faces():
    # Camera looks along +Z; the first triangle's normal points along +Z.
    v=np.array([[-.5,-.5,0],[.5,-.5,0],[0,.5,0],
                [0,0,.10],[.02,0,.10],[0,.02,.10],
                [0,0,.40],[.02,0,.40],[0,.02,.40],
                [-.5,-.5,2],[.5,-.5,2],[0,.5,2]],np.float32)
    f=np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]],np.int32)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f.astype(np.uint32)))
    camera=dict(eye=[0,0,-3],target=[0,0,0],right=[1,0,0],up=[0,1,0],extent=2)
    seeds,removed=m.cutaway_faces(scene,v,f,camera,256)
    assert 0 in seeds and 3 not in seeds
    assert 0 in removed and 1 in removed and 2 not in removed and 3 not in removed
    keep=np.setdiff1d(np.arange(len(f)),removed)
    cut=o3d.t.geometry.RaycastingScene();cut.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f[keep].astype(np.uint32)))
    colors=np.full((len(v),4),200,np.uint8)
    _,pixels=m.render(cut,v,f[keep],colors,camera,256)
    assert 3 in keep[pixels[pixels>=0]]


def test_floor_bands_share_ceiling_and_preserve_mezzanine_floor(tmp_path, monkeypatch):
    import json
    alignment=tmp_path/'scene'/'multi_room_alignment'; alignment.mkdir(parents=True)
    output=tmp_path/'scene'/'artifacts'/'views'; metadata=tmp_path/'scene'/'metadata.json'
    def floor(name, height, group):
        return dict(scan_id='scan_0',floor_level_name=name,plane_id=name,scan_level_name='BASE',height_group_index=group,
                    plane_equation=[0,1,0,-height],mean_elevation_m=height,support_area_m2=1,source_mesh='floor.ply')
    ceiling=dict(scan_id='scan_0',floor_level_name='BASE',plane_id='ceiling',mean_elevation_m=1.9,
                 plane_equation=[0,1,0,-1.9],source_mesh='ceiling.ply')
    metadata.write_text(json.dumps({'floor_planes':[floor('BASE',0,0),floor('BASEa',2,1)],'ceiling_planes':[ceiling]}))
    v=np.array(
        [[x,y,z] for y in (0,1.9,2,3) for x,z in ((0,0),(1,0),(0,1))],np.float32)
    f=np.arange(12,dtype=np.int32).reshape(-1,3); c=np.tile(np.array([100,150,200,255],np.uint8),(12,1))
    monkeypatch.setattr(m,'_read_mesh',lambda path:(v,f,c))
    result=m.run(alignment,metadata,output,64)
    assert len(result['candidates']) == 4
    assert result['scans'][0]['slice_indices'] == [0, 1]
    assert all(candidate['slice_indices'] == [0, 1] for candidate in result['candidates'])
    lower,upper=result['slices']
    assert np.allclose(lower['upper_cut']['equation'],upper['lower_cut_equation'])
    assert upper['lower_cut_source']=='ceiling'
    with np.load(output/upper['slice_mesh']) as saved:
        assert 2 in saved['source_face_id']


def test_cutaway_ceiling_preserves_floor():
    v=np.array([[-1,2,-1],[1,2,-1],[0,2,1],[-1,0,-1],[0,0,1],[1,0,-1]],np.float32)
    f=np.array([[0,1,2],[3,4,5]],np.uint32)
    scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    camera=dict(eye=[0,4,0],target=[0,0,0],right=[1,0,0],up=[0,0,1],extent=3)
    seeds,removed=m.cutaway_faces(scene,v,f,camera,128)
    assert 0 in seeds and 0 in removed and 1 not in removed


def test_cutaway_catches_large_face_crossing_radius_without_near_vertices():
    v=np.array([[-.5,-.5,0],[.5,-.5,0],[0,.5,0],[-2,-2,.1],[0,2,.1],[2,-2,.1]],np.float32)
    f=np.array([[0,1,2],[3,4,5]],np.uint32)
    scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    camera=dict(eye=[0,0,-3],target=[0,0,0],right=[1,0,0],up=[0,1,0],extent=5)
    seeds,removed=m.cutaway_faces(scene,v,f,camera,256)
    assert 0 in seeds and 1 not in seeds and 1 in removed


def test_front_face_stays_and_hides_deeper_backface():
    v=np.array([[-1,-1,0],[0,1,0],[1,-1,0],[-1,-1,1],[1,-1,1],[0,1,1]],np.float32)
    f=np.array([[0,1,2],[3,4,5]],np.uint32)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    camera=dict(eye=[0,0,-2],target=[0,0,0],right=[1,0,0],up=[0,1,0],extent=3)
    seeds,removed=m.cutaway_faces(scene,v,f,camera,128)
    assert len(seeds)==0 and len(removed)==0


def test_cutaway_radius_boundary():
    v=np.array([[-.5,-.5,0],[.5,-.5,0],[0,.5,0],
                [0,0,.145],[.001,0,.145],[0,.001,.145],
                [0,0,.16],[.001,0,.16],[0,.001,.16]],np.float32)
    f=np.array([[0,1,2],[3,4,5],[6,7,8]],np.uint32)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    camera=dict(eye=[0,0,-3],target=[0,0,0],right=[1,0,0],up=[0,1,0],extent=2)
    _,removed=m.cutaway_faces(scene,v,f,camera,128)
    assert 1 in removed and 2 not in removed


def test_floor_near_removed_wall_is_protected_while_clutter_is_removed():
    v=np.array([[-.5,0,0],[.5,0,0],[0,1,0],
                [-.2,0,.05],[.2,0,.05],[0,0,.5],
                [0,.2,.1],[.01,.2,.1],[0,.21,.1]],np.float32)
    f=np.array([[0,1,2],[3,4,5],[6,7,8]],np.uint32)
    floor=dict(plane_equation=[0,1,0,0],plane_id='floor')
    protected=m.floor_faces(v,f,[floor],.1,.9)
    assert list(protected)==[False,True,False]
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    camera=dict(eye=[0,.4,-2],target=[0,.4,0],right=[1,0,0],up=[0,1,0],extent=2)
    seeds,removed=m.cutaway_faces(scene,v,f,camera,128,protected=protected)
    assert 0 in seeds and 1 not in seeds
    assert 0 in removed and 1 not in removed and 2 in removed


def test_fitted_floor_fills_gaps_but_keeps_original_floor_colors():
    v=np.array([[-.5,0,-.5],[.5,0,-.5],[0,0,.5],[-1,.05,-1],[1,.05,-1],[0,.05,1]],np.float32)
    f=np.array([[0,1,2],[3,4,5]],np.uint32)
    colors=np.array([[255,0,0,255]]*3+[[0,0,255,255]]*3,np.uint8)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    original=o3d.t.geometry.RaycastingScene();original.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f[:1]))
    fallback=(original,np.array([0]),np.array([0,1]),np.array([0,0]),np.array([True,True]))
    camera=dict(eye=[0,2,0],target=[0,0,0],right=[1,0,0],up=[0,0,1],extent=3)
    image,pixels=m.render(scene,v,f,colors,camera,128,floor_fallback=fallback)
    assert np.any(pixels==0) and np.any(pixels==1)
    assert np.all(image[pixels==0]==[255,0,0]) and np.all(image[pixels==1]==[0,0,255])
    # Another slice's floor cannot overwrite the nearer fitted surface.
    fallback=(original,np.array([0]),np.array([0,1]),np.array([0,1]),np.array([True,True]))
    _,pixels=m.render(scene,v,f,colors,camera,128,floor_fallback=fallback)
    assert not np.any(pixels==0)


def test_protected_backfacing_floor_is_never_a_cutaway_seed():
    v=np.array([[-1,0,-1],[1,0,-1],[0,0,1]],np.float32)
    f=np.array([[0,1,2]],np.uint32)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f))
    camera=dict(eye=[0,2,0],target=[0,0,0],right=[1,0,0],up=[0,0,1],extent=3)
    seeds,removed=m.cutaway_faces(scene,v,f,camera,128,protected=np.array([True]))
    assert len(seeds)==0 and len(removed)==0



def test_panel_removal_follows_majority_of_neighbor_cutaway_results():
    removed=np.array([0,2,3])
    panel_ids=np.array([0,-1,-1,-1,-1])
    assert 0 in m.apply_panel_context(removed,panel_ids,{0:np.array([1,2,3])})
    assert 0 not in m.apply_panel_context(removed,panel_ids,{0:np.array([1,3,4])})
    assert 0 not in m.apply_panel_context(removed,panel_ids,{0:np.array([],np.int64)})


def test_panel_context_uses_nearby_non_panel_faces():
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0], [0,0,.1],[1,0,.1],[0,1,.1],
                       [0,0,1],[1,0,1],[0,1,1]],np.float32)
    faces=np.array([[0,1,2],[3,4,5],[6,7,8]],np.int32)
    panel_ids=np.array([-1,0,-1])
    neighbors=m.panel_context_neighbors(vertices,faces,panel_ids,radius=.2)
    assert np.array_equal(neighbors[0],[0])

def test_macro_renders_and_selects_each_scan_independently(tmp_path, monkeypatch):
    import json
    from threading import Barrier
    alignment=tmp_path/'scene'/'multi_room_alignment'; alignment.mkdir(parents=True)
    output=tmp_path/'scene'/'artifacts'/'views'; metadata=tmp_path/'scene'/'metadata.json'
    floors=[dict(scan_id=f'scan_{i}',floor_level_name='BASE',plane_id=f'floor_{i}',scan_level_name='BASE',height_group_index=0,
                 plane_equation=[0,1,0,0],mean_elevation_m=0,support_area_m2=1,source_mesh=f'floor_{i}.ply') for i in range(2)]
    metadata.write_text(json.dumps({'floor_planes':floors, 'ceiling_planes':[]}))
    v=np.array([[-1,0,-1],[1,0,-1],[0,0,1]],np.float32); f=np.array([[0,1,2]],np.int32)
    def mesh(path):
        second = 'scan_1' in str(path) or path.name == 'floor_1.ply'
        return v, f, np.tile(np.array([0,0,255,255] if second else [255,0,0,255],np.uint8),(3,1))
    monkeypatch.setattr(m,'_read_mesh',mesh)
    cutaway=m.cutaway_faces
    scans_rendering_together=Barrier(2)
    def synchronized_cutaway(*args, **kwargs):
        scans_rendering_together.wait(timeout=2)
        return cutaway(*args, **kwargs)
    monkeypatch.setattr(m,'cutaway_faces',synchronized_cutaway)
    result=m.run(alignment,metadata,output,64)
    assert len(result['scans']) == 2 and len(result['candidates']) == 8
    assert result['settings']['scan_workers'] == 2 and result['settings']['raycast_device'] == 'CPU'
    provenance=np.load(output/result['face_provenance'])['slice_index']
    for i, scan in enumerate(result['scans']):
        assert len(scan['candidate_pose_ids']) == 4 and scan['selected_pose_ids']
        for record in (x for x in result['candidates'] if x['scan_id'] == scan['scan_id']):
            pixels=np.load(output/record['pixel_face_ids'])['face_ids']
            assert pixels.max() >= 0 and set(provenance[pixels[pixels >= 0]]) == {i}
            removed=np.load(output/record['cutaway_faces'])['removed_face_ids']
            assert set(provenance[removed]) <= {i}
            assert record['per_slice_coverage'][1-i] == 0


def test_optical_panels_reflect_or_transmit_and_keep_primary_provenance():
    # Camera sees a panel at z=0; a red triangle sits behind the camera and a blue one behind the panel.
    v=np.array([[-1,-1,0],[1,-1,0],[0,1,0], [-4,-4,4],[0,4,4],[4,-4,4], [-4,-4,-2],[4,-4,-2],[0,4,-2]],np.float32)
    f=np.arange(9,dtype=np.int32).reshape(-1,3)
    colors=np.array([[150,150,150,255]]*3 + [[255,0,0,255]]*3 + [[0,0,255,255]]*3,np.uint8)
    scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f.astype(np.uint32)))
    camera=dict(eye=[0,0,2],target=[0,0,0],right=[1,0,0],up=[0,1,0],extent=1)
    for kind, expected in [(1,[255,0,0]),(2,[48,24,217])]:
        image, ids=m.render(scene,v,f,colors,camera,64,optical_types=np.array([kind,0,0]))
        assert np.any(ids == 0) and np.all(image[32,32] == expected)
        if kind == 2:
            assert any(np.all(image[ids == 0] == [55,65,75], axis=1))
            assert any(np.all(image[ids == 0] == [215,225,230], axis=1))
        assert np.all(image[ids == 2] == [0,0,255])
    panel=o3d.t.geometry.RaycastingScene(); panel.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f[:1].astype(np.uint32)))
    image, ids=m.render(panel,v,f[:1],colors,camera,64,optical_types=np.array([2]))
    assert np.all(image[32,32] == [239,241,242])
    assert np.all(image[ids == -1] == [255,255,255])


def test_panel_matching_validates_geometry():
    vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32)
    faces=np.array([[0,1,2],[0,2,3]],np.int32)
    regions=[{'scan_id':'scan_2','class':'mirror','sealed_panels':[{'plane_corners':vertices.tolist()}]}]
    kinds,panel_ids=m.panel_faces('scan_2',vertices,faces,regions)
    assert np.array_equal(kinds,[1,1]) and np.array_equal(panel_ids,[0,0])
    with pytest.raises(ValueError, match='geometry'):
        m.panel_faces('scan_2',vertices+1,faces,regions)
