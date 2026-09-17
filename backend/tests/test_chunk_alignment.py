"""The defect this guards: sliding windows that disagree on metric scale.

DA3 hands back a different scale/shift per window for the same frame. Fusing those
unaligned puts the frame at a standoff no window predicted, and unprojecting it lays a
duplicate of the surface next to the real one.
"""
import numpy as np

from depth_priors import (ChunkTrackAligner, DepthPriorEstimator, cross_view_scale_outliers,
                          robust_affine)


def _fake_colmap(names, n_tracks=60, seed=0):
    """A frontal plane at z=2m, observed as verified tracks by every frame."""
    rng = np.random.RandomState(seed)
    images, points = {}, {}
    pid = 0
    for name in names:
        uv, ids = [], []
        for _ in range(n_tracks):
            u, v = rng.uniform(5, 155), rng.uniform(5, 115)
            z = 2.0 + 2.0 * rng.rand()
            points[pid] = {"xyz": np.array([0.0, 0.0, z]), "error": 0.4, "track_len": 5}
            uv.append([u, v]); ids.append(pid); pid += 1
        images[name] = {"R_w2c": np.eye(3), "t_w2c": np.zeros(3),
                        "obs_xy": np.array(uv), "p3d_ids": np.array(ids, dtype=np.int64)}
    return images, points


def _truth(images, points, name, shape=(120, 160)):
    """Depth map agreeing exactly with that frame's tracks."""
    d = np.zeros(shape, dtype=np.float32)
    img = images[name]
    for (u, v), pid in zip(img["obs_xy"], img["p3d_ids"]):
        d[int(round(v)), int(round(u))] = points[pid]["xyz"][2]
    d[d == 0] = 3.0
    return d


def test_robust_affine_recovers_scale_shift_under_outliers():
    rng = np.random.RandomState(1)
    z_gt = rng.uniform(1.0, 5.0, 300)
    d_pred = (z_gt - 0.4) / 1.7
    d_pred[:40] += rng.uniform(2.0, 6.0, 40)  # 13% gross outliers
    s, t, mad = robust_affine(d_pred, z_gt)
    assert abs(s - 1.7) < 0.02 and abs(t - 0.4) < 0.02
    assert mad < 0.01


def test_aligner_rescales_a_window_and_flags_what_it_cannot():
    """One fit for the whole window, but rejection still per frame."""
    names = [f"f{i}.jpg" for i in range(4)]
    images, points = _fake_colmap(names)
    aligner = ChunkTrackAligner(names, images, points, max_residual_m=0.08)

    truth = [_truth(images, points, n) for n in names]
    window = [
        truth[0] * 0.45,              # the failure mode: the window came back half scale
        truth[1] * 0.45,
        truth[2] * 0.45,
        np.full_like(truth[3], 9.0),  # one frame unrelated to its tracks
    ]
    out = aligner(list(range(4)), window)

    for i in range(3):
        np.testing.assert_allclose(out[i], truth[i], atol=0.02)
    assert out[3] is None, "a constant map cannot match a varying plane at any scale"
    assert aligner.unreliable_frames() == ["f3.jpg"]


def test_aligner_does_not_fit_frames_individually():
    """A frame whose tracks span almost no depth must not get its own slope.

    Two free parameters on a 2 cm band of track depths is unidentifiable and used to
    extrapolate the frame metres away across the real depth range.
    """
    names = ["wide.jpg", "narrow.jpg"]
    images, points = _fake_colmap(names[:1])
    rng = np.random.RandomState(3)
    uv, ids = [], []
    pid = max(points) + 1
    for _ in range(40):  # plenty of tracks, all at ~2 m: count alone must not earn a fit
        uv.append([rng.uniform(5, 155), rng.uniform(5, 115)])
        points[pid] = {"xyz": np.array([0.0, 0.0, 2.0 + 0.01 * rng.rand()]),
                       "error": 0.4, "track_len": 5}
        ids.append(pid); pid += 1
    images["narrow.jpg"] = {"R_w2c": np.eye(3), "t_w2c": np.zeros(3),
                            "obs_xy": np.array(uv), "p3d_ids": np.array(ids, dtype=np.int64)}

    aligner = ChunkTrackAligner(names, images, points, max_residual_m=0.08)
    truth = [_truth(images, points, n) for n in names]
    out = aligner([0, 1], [truth[0] * 0.45, truth[1] * 0.45])

    # Fitted from the wide frame's spread, applied to both -- not a per-frame slope.
    np.testing.assert_allclose(out[1], truth[1], atol=0.02)
    assert aligner.frame_scale["wide.jpg"] == aligner.frame_scale["narrow.jpg"]


def test_cross_view_scale_catches_a_frame_its_own_tracks_cannot():
    """The ghost's real source: one frame at the wrong standoff, fitting its tracks fine.

    Five cameras on a baseline, all seeing one wall. Sparse tracks are clustered and few,
    so a 25% scale error can still fit them; the hundreds of pixels the other four cameras
    share with it cannot be fooled.
    """
    H, W, f = 120, 160, 150.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    names = [f"c{i}.jpg" for i in range(5)]

    w2c = []
    for i in range(5):
        m = np.eye(4)
        m[:3, 3] = [-0.15 * i, 0.0, 0.0]  # R = I, t = -centre
        w2c.append(m)
    w2c = np.stack(w2c)

    # Wall at world z = 2 m, so every camera measures a constant 2 m.
    depths = [np.full((H, W), 2.0, dtype=np.float32) for _ in names]
    depths[2] = depths[2] * 1.25  # one frame places the wall 50 cm too far

    ratios, outliers = cross_view_scale_outliers(depths, w2c, K, names, tol=0.10)

    assert outliers == ["c2.jpg"], ratios
    assert abs(ratios["c2.jpg"] - 0.8) < 0.02       # 2.0 observed / 2.5 predicted
    for good in ("c0.jpg", "c1.jpg", "c3.jpg", "c4.jpg"):
        assert abs(ratios[good] - 1.0) < 0.02, (good, ratios[good])


def test_sliding_window_fuses_aligned_windows_and_stays_index_aligned():
    estimator = DepthPriorEstimator(model_name="mock", device="cpu")
    imgs = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(15)]

    # Reject every prediction for frame 7 -- it must still come back with a depth map,
    # or every later frame silently pairs with the wrong pose.
    def reject_frame_7(indices, depths):
        return [None if g == 7 else d for g, d in zip(indices, depths)]

    depths, _, uncs = estimator.estimate_depth_sliding_window(
        images=imgs, chunk_size=6, overlap=3, align_fn=reject_frame_7)

    assert len(depths) == 15 and len(uncs) == 15
    assert all(d.shape == (120, 160) and np.isfinite(d).all() for d in depths)


def test_sliding_window_applies_the_alignment():
    estimator = DepthPriorEstimator(model_name="mock", device="cpu")
    imgs = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(8)]

    base, _, _ = estimator.estimate_depth_sliding_window(images=imgs, chunk_size=4, overlap=2)
    scaled, _, _ = estimator.estimate_depth_sliding_window(
        images=imgs, chunk_size=4, overlap=2,
        align_fn=lambda idx, ds: [d * 1.5 for d in ds])

    # 1.5x, not 0.5x: fusion clips to [min_depth, max_depth] and a shrink would land there.
    np.testing.assert_allclose(np.stack(scaled), np.stack(base) * 1.5, rtol=1e-5)


if __name__ == "__main__":
    test_robust_affine_recovers_scale_shift_under_outliers()
    test_aligner_rescales_a_window_and_flags_what_it_cannot()
    test_aligner_does_not_fit_frames_individually()
    test_cross_view_scale_catches_a_frame_its_own_tracks_cannot()
    test_sliding_window_fuses_aligned_windows_and_stays_index_aligned()
    test_sliding_window_applies_the_alignment()
    print("ALL CHUNK ALIGNMENT TESTS PASSED!")
