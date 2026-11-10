import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_artifacts_generation"))
from generate_interactive_viewer import assign_floor, photo_direction, project, projection_grid


def test_projection_maps_world_axes_and_each_style_has_own_grid():
    row = {"resolution_px": [100, 100], "mesh_bounds_min_xyz": [0, 0, 0],
           "mesh_bounds_max_xyz": [8, 2, 4], "pixel_size_m": .1}
    grid = projection_grid(row)
    assert project([4, 0, 2], grid) == pytest.approx([50, 50])
    assert project([5, 0, 3], grid) == pytest.approx([60, 40])
    row["pixel_size_m"] = .2
    assert project([5, 0, 3], projection_grid(row)) == pytest.approx([55, 45])


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
const context = vm.createContext({document, window: {addEventListener() {},
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
if (overlay.href !== 'measures.png' || overlay.x !== 10 || overlay.y !== 10 || overlay.width !== 50 || overlay.height !== 50) process.exit(7);
if (elements.plain.children[0].children.length !== 1) process.exit(8);
if (elements['mini-with-toggle'].children[0].children[1].children.length !== 1) process.exit(9);
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
