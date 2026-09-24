#
# Self-checks for the pose pipeline pieces whose logic is easy to get silently
# wrong: zoom grouping, the pose-prior DB blobs, the SE(3) delta, and the
# plane-induced homography warp.
#
# Run with:  python3 test_pose_pipeline.py
# Needs no dataset, no GPU and no COLMAP binary.
#

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import torch

from convert_transforms_to_colmap import group_by_zoom, write_pose_priors
from utils.multiview_loss import multiview_photometric_loss, pick_neighbors, build_capture_order
from utils.pose_refine import se3_exp


def _pose(tx=0.0, ty=0.0, tz=0.0):
    m = np.eye(4)
    m[:3, 3] = [tx, ty, tz]
    return m.tolist()


def test_zoom_grouping():
    # Two zoom states, with a within-tolerance wobble inside the first run.
    transforms = {
        "fl_x": 1400.0, "fl_y": 1400.0, "cx": 540.0, "cy": 960.0, "w": 1080, "h": 1920,
        "frames": [
            {"file_path": "images/frame_00000.jpg", "transform_matrix": _pose()},
            {"file_path": "images/frame_00001.jpg", "fl_x": 1405.0, "fl_y": 1405.0,
             "transform_matrix": _pose()},
            {"file_path": "images/frame_00002.jpg", "fl_x": 2100.0, "fl_y": 2100.0,
             "transform_matrix": _pose()},
            {"file_path": "images/frame_00003.jpg", "fl_x": 2100.0, "fl_y": 2100.0,
             "transform_matrix": _pose()},
        ],
    }
    groups = group_by_zoom(transforms, rel_tol=0.01)
    assert len(groups) == 2, f"expected 2 zoom states, got {len(groups)}"
    assert len(groups[0]["frames"]) == 2 and len(groups[1]["frames"]) == 2
    # 1400 and 1405 are within 1%, so they average rather than split.
    assert abs(groups[0]["fl_x"] - 1402.5) < 1e-6, groups[0]["fl_x"]
    assert abs(groups[1]["fl_x"] - 2100.0) < 1e-6
    assert groups[0]["w"] == 1080 and groups[0]["h"] == 1920

    # A fixed-zoom capture must stay one camera (the pre-merge behaviour).
    single = dict(transforms, frames=transforms["frames"][:2])
    assert len(group_by_zoom(single, rel_tol=0.01)) == 1

    # Grouping is by *contiguous run*, so a zoom that returns makes a third group.
    returning = dict(transforms, frames=transforms["frames"] + [
        {"file_path": "images/frame_00004.jpg", "fl_x": 1400.0, "fl_y": 1400.0,
         "transform_matrix": _pose()}])
    assert len(group_by_zoom(returning, rel_tol=0.01)) == 3
    print("zoom grouping            OK")


def test_pose_prior_roundtrip():
    sigma = 0.05
    frames = [{"file_path": "images/a.jpg", "_colmap_name": "a.jpg",
               "transform_matrix": _pose(1.0, 2.0, 3.0)},
              {"file_path": "images/b.jpg", "_colmap_name": "b.jpg",
               "transform_matrix": _pose(-0.5, 0.25, 4.0)}]
    groups = [{"frames": frames}]

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE images (image_id INTEGER PRIMARY KEY, camera_id INTEGER, name TEXT);
            CREATE TABLE pose_priors (corr_data_id INTEGER, corr_sensor_id INTEGER,
                corr_sensor_type INTEGER, position BLOB, position_covariance BLOB,
                gravity BLOB, coordinate_system INTEGER);
            INSERT INTO images VALUES (1, 1, 'a.jpg'), (2, 1, 'b.jpg');
        """)
        conn.commit()
        conn.close()

        assert write_pose_priors(groups, db_path, sigma) == 2

        conn = sqlite3.connect(db_path)
        rows = list(conn.execute(
            "SELECT corr_data_id, position, position_covariance, gravity, "
            "corr_sensor_type, coordinate_system FROM pose_priors ORDER BY corr_data_id"))
        conn.close()

    assert len(rows) == 2
    expected = [np.array([1.0, 2.0, 3.0]), np.array([-0.5, 0.25, 4.0])]
    for (data_id, pos, cov, grav, stype, csys), want in zip(rows, expected):
        # Raw little-endian float64, no header - COLMAP's WriteStaticMatrixBlob.
        assert len(pos) == 24, f"position blob should be 3 doubles, got {len(pos)} bytes"
        assert len(cov) == 72, f"covariance blob should be 9 doubles, got {len(cov)} bytes"
        assert len(grav) == 24
        assert np.allclose(np.frombuffer(pos, dtype="<f8"), want), data_id
        cov_m = np.frombuffer(cov, dtype="<f8").reshape(3, 3)
        assert np.allclose(cov_m, np.diag([sigma ** 2] * 3)), cov_m
        # NaN gravity means PosePrior::HasGravity() is false; zeros would claim a
        # bogus "down" direction instead.
        assert np.all(np.isnan(np.frombuffer(grav, dtype="<f8")))
        assert stype == 0 and csys == 1
    print("pose prior round-trip    OK")


def test_se3_exp():
    # Zero delta must be exactly the identity, and differentiable there - this is
    # the initialisation every refined camera starts from.
    delta = torch.zeros(6, requires_grad=True)
    out = se3_exp(delta)
    assert torch.allclose(out, torch.eye(4), atol=1e-6), out
    out.sum().backward()
    assert torch.isfinite(delta.grad).all(), delta.grad

    # 90 deg about z.
    d = torch.tensor([0.0, 0.0, np.pi / 2, 1.0, 2.0, 3.0])
    m = se3_exp(d)
    want_R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(m[:3, :3], want_R, atol=1e-6), m[:3, :3]
    assert torch.allclose(m[:3, 3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-6)
    # Still a rotation: R R^T = I, det = +1.
    R = m[:3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
    assert abs(torch.det(R).item() - 1.0) < 1e-6
    print("se3_exp                  OK")


class _StubCam:
    """Minimal stand-in for scene.cameras.Camera for the warp check."""

    def __init__(self, name, image, wvt):
        self.image_name = name
        self.original_image = image
        self.world_view_transform = wvt
        self.image_height, self.image_width = image.shape[-2:]
        # fx = fy = W, i.e. a ~53 deg field of view; the exact value is irrelevant
        # here as long as both cameras agree.
        self.FoVx = 2 * np.arctan(self.image_width / (2.0 * self.image_width))
        self.FoVy = 2 * np.arctan(self.image_height / (2.0 * self.image_width))


def test_multiview_warp():
    torch.manual_seed(0)
    H = W = 64
    img = torch.rand(3, H, W)
    eye = torch.eye(4)

    view = _StubCam("frame_00000.jpg", img, eye)
    depth = torch.full((1, H, W), 2.0)
    normal = torch.zeros(3, H, W)
    normal[2] = 1.0                      # plane facing the camera
    alpha = torch.ones(1, H, W)

    # Identical pose + identical image => H is the identity, so the warp must be
    # a no-op and the photometric error must vanish. This is the check that
    # catches a transposed matrix or a flipped convention in the homography.
    twin = _StubCam("frame_00001.jpg", img.clone(), eye.clone())
    same = multiview_photometric_loss(view, [twin], depth, normal, alpha)
    assert same.item() < 1e-5, f"identity warp should be lossless, got {same.item()}"

    # Same pose, different content => non-zero.
    other = _StubCam("frame_00002.jpg", torch.rand(3, H, W), eye.clone())
    diff = multiview_photometric_loss(view, [other], depth, normal, alpha)
    assert diff.item() > 0.01, diff.item()

    # A sideways-shifted neighbour must disagree with the source image; if the
    # translation term of the homography were dropped, this would also read ~0.
    shifted_wvt = torch.eye(4)
    shifted_wvt[3, 0] = 0.5              # W2C transposed => translation lives in row 3
    moved = _StubCam("frame_00003.jpg", img.clone(), shifted_wvt)
    assert multiview_photometric_loss(view, [moved], depth, normal, alpha).item() > 1e-3

    # Empty alpha => no sampled pixels => exactly zero, no NaN.
    empty = multiview_photometric_loss(view, [twin], depth, normal, torch.zeros(1, H, W))
    assert empty.item() == 0.0

    # Gradients must reach the geometry the loss is meant to supervise.
    d_grad = depth.clone().requires_grad_(True)
    loss = multiview_photometric_loss(view, [other], d_grad, normal, alpha)
    loss.backward()
    assert d_grad.grad is not None and torch.isfinite(d_grad.grad).all()
    print("multiview homography     OK")


def test_neighbor_selection():
    cams = [_StubCam(f"frame_{i:05d}.jpg", torch.zeros(3, 8, 8), torch.eye(4))
            for i in range(10)]
    # Shuffled input, as Scene hands it over.
    shuffled = [cams[4], cams[0], cams[9], cams[2]] + cams[5:9] + [cams[1], cams[3]]
    order, index = build_capture_order(shuffled)
    assert [c.image_name for c in order] == [c.image_name for c in cams]

    picks = pick_neighbors(cams[5], order, index, num_neighbors=2, stride=2)
    assert [c.image_name for c in picks] == ["frame_00003.jpg", "frame_00007.jpg"]

    # At the sequence start there is nothing behind, so the window keeps walking
    # forward rather than returning a single neighbour.
    edge = pick_neighbors(cams[0], order, index, num_neighbors=2, stride=2)
    assert [c.image_name for c in edge] == ["frame_00002.jpg", "frame_00004.jpg"], \
        [c.image_name for c in edge]
    assert pick_neighbors(_StubCam("absent.jpg", torch.zeros(3, 8, 8), torch.eye(4)),
                          order, index) == []
    print("neighbor selection       OK")


def test_keyframe_export():
    from PIL import Image
    from export_keyframes import export_keyframes

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images").mkdir()
        names = [f"frame_{i:05d}.jpg" for i in range(4)]
        for name in names:
            Image.new("RGB", (64, 32)).save(root / "images" / name)

        # COLMAP registered everything except frame_00002.
        registered = [n for n in names if n != "frame_00002.jpg"]
        sparse = root / "sparse" / "0"
        sparse.mkdir(parents=True)
        lines = []
        for image_id, name in enumerate(registered, start=1):
            lines += [f"{image_id} 1 0 0 0 0 0 0 1 {name}", ""]
        (sparse / "images.txt").write_text("\n".join(lines) + "\n")

        transforms = {"w": 64, "h": 32,
                      "frames": [{"file_path": f"images/{n}"} for n in names]}
        kept = export_keyframes(str(root), str(root / "images"), transforms, str(sparse))

        assert [f["file_path"] for f in kept] == [f"images/{n}" for n in registered]
        # Nothing the capture produced is lost.
        assert sorted(p.name for p in (root / "images_all").iterdir()) == names

        for divisor, size in ((1, (64, 32)), (2, (32, 16)), (4, (16, 8)), (8, (8, 4))):
            folder = root / ("images" if divisor == 1 else f"images_{divisor}")
            assert sorted(p.name for p in folder.iterdir()) == registered, divisor
            with Image.open(folder / registered[0]) as im:
                assert im.size == size, (divisor, im.size)

        out = json.loads((root / "transforms_keyframes.json").read_text())
        assert len(out["frames"]) == len(registered)
        assert out["w"] == 64  # intrinsics stay at full resolution
    print("keyframe export          OK")


def test_sampson_filter():
    from export_keyframes import filter_by_sampson

    keep = {f"frame_{i:05d}.jpg" for i in range(10)}

    kept, refused = filter_by_sampson(keep, {})
    assert kept == keep and not refused

    # A handful of bad poses is dropped.
    kept, refused = filter_by_sampson(keep, {"frame_00003.jpg": 174.0})
    assert not refused and kept == keep - {"frame_00003.jpg"}

    # Too many to be credible: reported, not applied, so a wrong threshold or a
    # broken model cannot quietly delete most of the capture.
    rejects = {f"frame_{i:05d}.jpg": 9.0 for i in range(5)}
    kept, refused = filter_by_sampson(keep, rejects)
    assert refused and kept == keep
    print("sampson filter           OK")


def test_compass_heading_estimation_and_alignment():
    from scene_extent import estimate_north_heading_deg, rotate_horizontal

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frames = []
        for i in range(5):
            mat = np.eye(4)
            mat[:3, 3] = [float(i), 1.5, float(i)]
            frames.append({
                "file_path": f"images/f{i}.jpg",
                "timestamp_ns": 1000 + i,
                "transform_matrix": mat.tolist(),
                "compass_heading_deg": 90.0,
            })
        tf_path = tmp_path / "transforms.json"
        tf_path.write_text(json.dumps({"frames": frames}))

        north_deg = estimate_north_heading_deg(tmp_path / "sparse" / "0", transforms_path=tf_path)
        assert north_deg is not None
        assert abs(north_deg - 270.0) < 1e-4

        align_deg = (90.0 - north_deg) % 360.0
        assert abs(align_deg - 180.0) < 1e-4

        # A point along North (x=-1, y=0, z=0) rotates to +X (x=1, y=0, z=0)
        pt_north = np.array([[-1.0, 0.0, 0.0]])
        pt_aligned = rotate_horizontal(pt_north, align_deg, up_axis=1)
        assert np.allclose(pt_aligned, [[1.0, 0.0, 0.0]], atol=1e-5)
    print("compass heading alignment OK")


if __name__ == "__main__":
    test_zoom_grouping()
    test_keyframe_export()
    test_sampson_filter()
    test_compass_heading_estimation_and_alignment()
    test_pose_prior_roundtrip()
    test_se3_exp()
    test_neighbor_selection()
    test_multiview_warp()
    print("\nall checks passed")
