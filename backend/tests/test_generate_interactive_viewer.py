import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_artifacts_generation"))
from generate_interactive_viewer import assign_floor, generate, photo_direction, project, projection_grid, versioned_asset_url, _flip_floorplan_vertically
from generate_floorplan_measurements import _flip_measurements_vertically


def test_projection_maps_world_axes_and_each_style_has_own_grid():
    row = {"resolution_px": [100, 100], "mesh_bounds_min_xyz": [0, 0, 0],
           "mesh_bounds_max_xyz": [8, 2, 4], "pixel_size_m": .1}
    grid = projection_grid(row)
    assert project([4, 0, 2], grid) == pytest.approx([50, 50])
    assert project([5, 0, 3], grid) == pytest.approx([60, 60])
    row["pixel_size_m"] = .2
    assert project([5, 0, 3], projection_grid(row)) == pytest.approx([55, 55])


def test_floorplan_measurements_flip_vertically_with_plan_and_keep_readable_labels():
    measurement = {"label": "1.25 m", "span_px": [[10, 20], [30, 40]],
                   "dimension_span_px": [[12, 22], [32, 42]],
                   "extension_spans_px": [[[10, 20], [12, 22]]],
                   "label_center_px": [21, 31]}
    flipped = _flip_measurements_vertically([measurement], 80)[0]
    assert flipped["span_px"] == [[10, 59], [30, 39]]
    assert flipped["dimension_span_px"] == [[12, 57], [32, 37]]
    assert flipped["extension_spans_px"] == [[[10, 59], [12, 57]]]
    assert flipped["label_center_px"] == [21, 48]
    assert flipped["label"] == "1.25 m"


def test_plan_image_is_flipped_vertically():
    from PIL import Image

    image = Image.fromarray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
    flipped = np.asarray(_flip_floorplan_vertically(image))
    assert np.array_equal(flipped, np.flip(image, axis=0))


def test_photo_horizontal_cone_follows_aligned_colmap_pose_and_calibration():
    matrix = np.eye(4)
    matrix[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    yaw, edges = photo_direction({"camera_to_world": matrix.tolist(), "intrinsics": [[100, 0, 50], [0, 100, 50], [0, 0, 1]], "width": 100})
    assert yaw == pytest.approx(0)
    assert edges == pytest.approx([math.atan(.5), -math.atan(.5)])


def floor(height):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor([[0, height, 0], [1, height, 0], [0, height, 1]], dtype=o3d.core.Dtype.Float32), o3d.core.Tensor([[0, 1, 2]], dtype=o3d.core.Dtype.UInt32))
    return scene


def test_floor_assignment_uses_nearest_supported_floor_below_camera():
    candidates = [("lower", floor(0), None), ("upper", floor(2), None)]
    assert assign_floor([.2, 3, .2], candidates) == "upper"
    assert assign_floor([.2, 1, .2], candidates) == "lower"
    with pytest.raises(ValueError, match="No floor below"):
        assign_floor([2, 1, 2], candidates)


def test_camera_asset_url_changes_when_image_bytes_change(tmp_path):
    asset = tmp_path / "assets/panorama_mesh.jpg"
    asset.parent.mkdir()
    asset.write_bytes(b"old panorama")
    old_url = versioned_asset_url("assets/panorama_mesh.jpg", tmp_path)
    asset.write_bytes(b"updated panorama")
    new_url = versioned_asset_url("assets/panorama_mesh.jpg", tmp_path)
    assert old_url.startswith("assets/panorama_mesh.jpg?v=")
    assert new_url.startswith("assets/panorama_mesh.jpg?v=")
    assert new_url != old_url


@pytest.mark.parametrize("panel_source", ["changed", "added"])
def test_viewer_rejects_stale_panorama_panel_manifest(tmp_path, panel_source):
    alignment = tmp_path / "multi_room_alignment"
    stats = alignment / "stats"
    plans = tmp_path / "artifacts/Generate Floor Plans"
    photos = tmp_path / "artifacts/Extract Best Views"
    panoramas = tmp_path / "artifacts/Generate 360 Pictures"
    for directory in (stats, plans / "floor_level_meshes", photos, panoramas):
        directory.mkdir(parents=True)
    transforms = stats / "aligned_scans_transforms.json"
    panel_manifest = stats / "phase07_region_replacements.json"
    metadata = plans / "floor_level_metadata.json"
    slices = plans / "floor_level_meshes/floor_level_meshes_manifest.json"
    for path in (transforms, panel_manifest, metadata, slices):
        path.write_text("{}")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    common = {"macro_step_status": "complete", "alignment_dir": str(alignment)}
    (plans / "generate_floor_plans_manifest.json").write_text(json.dumps(common))
    (photos / "extract_best_views_manifest.json").write_text(json.dumps({**common, "source_transforms_sha256": digest(transforms)}))
    hashes = {str(path): digest(path) for path in (transforms, panel_manifest, metadata, slices)}
    if panel_source == "added":
        del hashes[str(panel_manifest)]
    (panoramas / "generate_360_pictures_manifest.json").write_text(json.dumps({**common, "schema_version": 5, "source_hashes": hashes}))
    panel_manifest.write_text('{"regions": []}')
    with pytest.raises(ValueError, match="stale panel data"):
        generate(tmp_path)
    assert not (tmp_path / "artifacts/Generate Interactive Viewer").exists()


def test_minimap_crop_and_panorama_cue():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to exercise the browser map function")
    script = r"""
const fs = require('fs'), vm = require('vm');
class Element {
  constructor() { this.style = {}; this.children = []; this.attributes = {}; }
  setAttribute(key, value) { this.attributes[key] = value; }
  append(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  addEventListener() {}
}
const elements = {};
const document = { getElementById: id => elements[id] ||= new Element(),
  createElementNS: () => new Element(), addEventListener() {} };
let frame = null, nextFrame = 0, reducedMotion = false;
const context = vm.createContext({document, devicePixelRatio:1, window: {addEventListener() {},
    matchMedia: () => ({matches: reducedMotion})},
  requestAnimationFrame: cb => { frame = cb; return ++nextFrame; },
  cancelAnimationFrame: () => { frame = null; }, setTimeout: () => 1, clearTimeout() {},
  fetch: () => new Promise(() => {})});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(`map(document.getElementById('minimap'),
  {styles:{realistic:{grid:{width:1024,height:1024},image:'plan.png'}}},
  'realistic', [], [611,1,983,463], {})`, context);
if (elements.minimap.children[0].attributes.viewBox !== '566 1 462 462') process.exit(1);
if (elements.minimap.style.aspectRatio !== '1') process.exit(2);
if (elements.minimap.children[0].children[0].children[0].children[0].attributes.width !== 372) process.exit(6);
vm.runInContext(`const level = {styles:{realistic:{grid:{x_min:-4,z_max:8,size:.2,width:100,height:100},image:'plan.png'}},
  measurements:{grid:{x_min:-2,z_max:6,size:.1,width:100,height:100},image:'measures.png'}};
  map(document.getElementById('plan'), level, 'realistic', [], null, null, true);
  map(document.getElementById('plain'), level, 'realistic', [], null, null, false);
  map(document.getElementById('mini-with-toggle'), level, 'realistic', [], [0,0,100,100], {}, true)`, context);
const overlay = elements.plan.children[0].children[1].attributes;
if (overlay.href !== 'measures.png' || overlay.x !== 10 || overlay.y !== 40 || overlay.width !== 50 || overlay.height !== 50) process.exit(7);
if (elements.plain.children[0].children.length !== 1) process.exit(8);
if (elements['mini-with-toggle'].children[0].children[1].children.length !== 1) process.exit(9);
vm.runInContext(`map(document.getElementById('cone-test'),
  {styles:{realistic:{grid:{width:100,height:100},image:'plan.png'}}}, 'realistic',
  [{pixels:{realistic:[50,50]},kind:'photo',scan:'scan_0',number:1,yaw:0,edges:[0,0]}]);`, context);
const cone = elements['cone-test'].children[0].children[1].children[0].attributes.points.split(' ')[1];
if (Number(cone.split(',')[0]) <= 50) process.exit(10);
const pano = {pixels:{realistic:[50,50]},kind:'panorama',scan:'scan_0',number:2,yaw:0,edges:[-0.5,0.5]};
context.pano=pano;
vm.runInContext(`map(document.getElementById('pano-minimap'),
  {styles:{realistic:{grid:{width:100,height:100},image:'plan.png'}}}, 'realistic', [pano], null, pano)`,
  context);
const panoMini = elements['pano-minimap'], panoCone = panoMini.children[0].children[1].children[1].children[0];
panoMini.querySelector = () => panoCone;
elements.minimap=panoMini;
elements.panorama.clientWidth=100; elements.panorama.clientHeight=100;
vm.runInContext(`gl={TRIANGLE_STRIP:0,viewport(){},uniform2f(){},uniform1f(){},drawArrays(){}};
  uniforms={resolution:0,yaw:0,pitch:0,fov:0};
  data={cameras:[pano]}; active=0; yaw=0; pitch=0; fov=0;`, context);
elements.style.value='realistic';
vm.runInContext('renderPano()', context);
const liveConePoint = panoCone.attributes.points.split(' ')[1];
if (Number(liveConePoint.split(',')[0]) <= 50) process.exit(11);
vm.runInContext('cueAllowed=true; yaw=0; renderPano=()=>{}; startCue()', context);
frame(0); frame(325);
if (Math.abs(vm.runInContext('yaw', context)) < 0.05) process.exit(3);
vm.runInContext('stopCue()', context);
if (frame !== null || !elements['panorama-hint'].hidden) process.exit(4);
reducedMotion = true;
vm.runInContext('cueAllowed=true; startCue()', context);
if (frame !== null || elements['panorama-hint'].hidden) process.exit(5);
"""
    subprocess.run([node, "-e", script, str(Path(__file__).resolve().parents[1] / "08_artifacts_generation/interactive_viewer/viewer.js")], check=True)


def test_panorama_switch_preserves_orientation_and_ignores_stale_images():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to exercise panorama switching")
    script = r"""
const fs = require('fs'), vm = require('vm');
class Element {
  constructor() { this.attributes = {}; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener() {}
}
const elements = {}, pending = [], uploads = [];
const document = {getElementById: id => elements[id] ||= new Element(), addEventListener() {}};
class Image { constructor() { pending.push(this); } set src(value) { this.url = value; } }
const context = vm.createContext({document, Image, uploads, window: {addEventListener() {}},
  cancelAnimationFrame() {}, clearTimeout() {}, fetch: () => new Promise(() => {})});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(`gl={bindTexture(){}, texImage2D(...args){uploads.push(args[5].url)}};
  renderPano=()=>{}; renderMini=()=>{}; startCue=()=>{};
  data={cameras:[{kind:'panorama',floor:'BASE',scan:'scan_0',number:1,yaw:0,image:'splat.jpg',mesh_image:'mesh.jpg'},
                {kind:'photo',floor:'BASE',scan:'scan_0',number:2,image:'photo.jpg'}]};
  showCamera(0); yaw=1.2; pitch=.3; fov=.7;`, context);
if (pending[0].url !== 'splat.jpg' || elements['pano-controls'].hidden) process.exit(1);
elements['panorama-source'].onclick();
if (pending[1].url !== 'mesh.jpg' || elements['panorama-source'].textContent !== 'Show Gaussian splat') process.exit(2);
if (vm.runInContext('yaw!==1.2 || pitch!==.3 || fov!==.7', context)) process.exit(3);
pending[0].onload(); pending[1].onload();
if (uploads.join(',') !== 'mesh.jpg') process.exit(4);
elements['panorama-source'].onclick();
if (pending[2].url !== 'splat.jpg' || elements['panorama-source'].textContent !== 'Show mesh') process.exit(5);
pending[2].onload();
if (uploads.join(',') !== 'mesh.jpg,splat.jpg') process.exit(6);
elements['panorama-source'].onclick();
vm.runInContext('showCamera(1)', context);
if (!elements['pano-controls'].hidden || elements.photo.src !== 'photo.jpg') process.exit(7);
pending[3].onload();
if (uploads.join(',') !== 'mesh.jpg,splat.jpg') process.exit(8);
vm.runInContext('showCamera(0)', context);
if (pending[4].url !== 'mesh.jpg' || elements['pano-controls'].hidden) process.exit(9);
"""
    subprocess.run([node, "-e", script, str(Path(__file__).resolve().parents[1] / "08_artifacts_generation/interactive_viewer/viewer.js")], check=True)
