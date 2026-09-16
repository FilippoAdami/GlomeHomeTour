#
# Pose diagnostics for the ARCore -> COLMAP pose pipeline.
#
# Ported from the retired COLMAP_adjustment/ sandbox: refine_poses_colmap.py
# (drift metrics, plots, Sampson epipolar validation), eval_poses.py (multi-view
# reprojection validation) and path_to_ply.py (Blender PLY export).
#
# Three checks, in increasing order of how little they trust COLMAP:
#
#   drift_report()        Refined poses vs. the ARCore initialisation. Cheap, and
#                         the only place rotation drift is caught at all - COLMAP
#                         has no rotation prior, so this is what closes that gap.
#   reprojection_report() Triangulated 3D tracks projected into every view with
#                         the ARCore-init and the refined poses. The most direct
#                         "did refinement actually help" number.
#   sampson_report()      Epipolar error computed straight from the raw feature
#                         matches in database.db. The only check that never reads
#                         COLMAP's own residuals, so it catches a BA that
#                         converged to a bad but internally self-consistent
#                         solution (which the other two would happily bless).
#

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # diagnostics run headless / over ssh
import matplotlib.pyplot as plt
import cv2

from scene.colmap_loader import qvec2rotmat

# colmap::kMaxNumImages, the multiplier used to pack an image pair into one id.
MAX_NUM_IMAGES = 2147483647


# --------------------------------------------------------------------------
# COLMAP text model parsing
# --------------------------------------------------------------------------
def parse_cameras_txt(path):
    cams = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cams[int(p[0])] = {"model": p[1], "width": int(p[2]), "height": int(p[3]),
                           "params": [float(x) for x in p[4:]]}
    return cams


def camera_K(cam):
    fx, fy, cx, cy = cam["params"][:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def camera_dist(cam):
    # PINHOLE carries no distortion; OPENCV appends k1 k2 p1 p2 after fx fy cx cy.
    params = cam["params"]
    return np.array(params[4:8]) if len(params) >= 8 else np.zeros(4)


def parse_images_txt(path):
    """name -> pose + 2D observations. COLMAP writes two lines per image, the
    second holding X Y POINT3D_ID triples (empty in a manual model)."""
    path = Path(path)
    images = {}
    if not path.exists():
        return images
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]])
            tvec = np.array([float(x) for x in parts[5:8]])
            camera_id = int(parts[8])
        except ValueError:
            continue
        name = parts[9]

        obs_parts = lines[i].split() if i < len(lines) else []
        i += 1
        if obs_parts and len(obs_parts) % 3 == 0:
            obs = np.array(obs_parts, dtype=np.float64).reshape(-1, 3)
        else:
            obs = np.zeros((0, 3))

        R_w2c = qvec2rotmat(qvec)
        R_c2w = R_w2c.T
        images[name] = {"id": image_id, "name": name, "camera_id": camera_id,
                        "qvec": qvec, "R_w2c": R_w2c, "t_w2c": tvec,
                        "R_c2w": R_c2w, "t_c2w": -R_c2w @ tvec,
                        "obs_xy": obs[:, :2],
                        "p3d_ids": obs[:, 2].astype(np.int64)}
    return images


def parse_points3D_txt(path):
    path = Path(path)
    pts = {}
    if not path.exists():
        return pts
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        pts[int(p[0])] = {"xyz": np.array([float(p[1]), float(p[2]), float(p[3])]),
                          "rgb": np.array([int(p[4]), int(p[5]), int(p[6])], dtype=np.uint8),
                          "error": float(p[7]),
                          "track_len": (len(p) - 8) // 2}
    return pts


def rotation_delta_deg(R_a, R_b):
    """Geodesic angle between two rotations, via the trace formula."""
    cos_theta = (np.trace(R_a @ R_b.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


# --------------------------------------------------------------------------
# 1. Drift vs. the ARCore initialisation
# --------------------------------------------------------------------------
def drift_report(init_images, refined_images, points3D, images_dir, out_dir,
                 rot_thresh_deg=2.0, trans_thresh_m=0.05):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = sorted(set(init_images) & set(refined_images))
    if not common:
        print("[diagnostics] no image names shared between the two models, skipping drift")
        return {}

    rows, trans_d, rot_d = [], [], []
    for name in common:
        init, ref = init_images[name], refined_images[name]
        t_delta = float(np.linalg.norm(ref["t_c2w"] - init["t_c2w"]))
        r_delta = rotation_delta_deg(ref["R_c2w"], init["R_c2w"])
        trans_d.append(t_delta)
        rot_d.append(r_delta)
        rows.append({"image": name, "translation_delta_m": t_delta,
                     "rotation_delta_deg": r_delta,
                     "init_tx": init["t_c2w"][0], "init_ty": init["t_c2w"][1],
                     "init_tz": init["t_c2w"][2],
                     "refined_tx": ref["t_c2w"][0], "refined_ty": ref["t_c2w"][1],
                     "refined_tz": ref["t_c2w"][2]})

    trans_d, rot_d = np.array(trans_d), np.array(rot_d)

    with open(out_dir / "per_frame_deltas.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errors = np.array([p["error"] for p in points3D.values()]) if points3D else np.zeros(0)
    track_lens = np.array([p["track_len"] for p in points3D.values()]) if points3D else np.zeros(0)

    # COLMAP has no rotation prior, so a frame that rotated a long way from the
    # ARCore init is the one failure mode the pose prior cannot catch itself.
    rot_outliers = [r["image"] for r in rows if r["rotation_delta_deg"] > rot_thresh_deg]
    trans_outliers = [r["image"] for r in rows if r["translation_delta_m"] > trans_thresh_m]

    summary = {
        "num_frames": len(common),
        "point_cloud": {
            "total_points": len(points3D),
            "mean_reprojection_error_px": float(np.mean(errors)) if errors.size else 0.0,
            "mean_track_length": float(np.mean(track_lens)) if track_lens.size else 0.0,
        },
        "translation_drift": {
            "mean_m": float(np.mean(trans_d)), "max_m": float(np.max(trans_d)),
            "std_m": float(np.std(trans_d)),
            "num_beyond_threshold": len(trans_outliers),
            "threshold_m": trans_thresh_m,
        },
        "rotation_drift": {
            "mean_deg": float(np.mean(rot_d)), "max_deg": float(np.max(rot_d)),
            "std_deg": float(np.std(rot_d)),
            "num_beyond_threshold": len(rot_outliers),
            "threshold_deg": rot_thresh_deg,
        },
        "rotation_outliers": rot_outliers,
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_drift(trans_d, rot_d, out_dir)
    _plot_trajectories(init_images, refined_images, common, points3D, out_dir)
    _draw_reprojection_overlay(refined_images, common, points3D, images_dir, out_dir)

    print("\n" + "=" * 58)
    print(" COLMAP POSE REFINEMENT DIAGNOSTICS")
    print("=" * 58)
    print(f"Frames compared:            {len(common)}")
    print(f"Triangulated points:        {len(points3D):,}")
    print(f"Mean track length:          {summary['point_cloud']['mean_track_length']:.2f}")
    print(f"Mean reproj. error (COLMAP):{summary['point_cloud']['mean_reprojection_error_px']:.4f} px")
    print(f"Translation drift:          mean {np.mean(trans_d)*100:.2f} cm, max {np.max(trans_d)*100:.2f} cm")
    print(f"Rotation drift:             mean {np.mean(rot_d):.3f} deg, max {np.max(rot_d):.3f} deg")
    print(f"Frames beyond {rot_thresh_deg} deg:        {len(rot_outliers)} / {len(common)}")
    print("=" * 58)
    return summary


def _plot_drift(trans_d, rot_d, out_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(trans_d * 100.0, lw=1.5)
    plt.title("Per-frame translation drift vs. ARCore init")
    plt.ylabel("Displacement (cm)")
    plt.grid(True, alpha=0.6)
    plt.subplot(2, 1, 2)
    plt.plot(rot_d, lw=1.5, color="tab:orange")
    plt.title("Per-frame rotation drift vs. ARCore init")
    plt.ylabel("Angle (deg)")
    plt.xlabel("Frame index (capture order)")
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_dir / "drift_profile_2d.png", dpi=200)
    plt.close()


def _plot_trajectories(init_images, refined_images, common, points3D, out_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    init_traj = np.array([init_images[n]["t_c2w"] for n in common])
    ref_traj = np.array([refined_images[n]["t_c2w"] for n in common])
    ax.plot(init_traj[:, 0], init_traj[:, 1], init_traj[:, 2],
            color="tab:red", alpha=0.7, label="ARCore init")
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
            color="tab:green", lw=2, label="Refined (COLMAP BA)")

    if points3D:
        xyzs = np.array([p["xyz"] for p in points3D.values()])
        # Clip to a ball around the trajectory; a few wild triangulations would
        # otherwise squash the useful part of the plot to a dot.
        center = np.mean(ref_traj, axis=0)
        radius = np.max(np.linalg.norm(ref_traj - center, axis=1)) * 2.5
        valid = xyzs[np.linalg.norm(xyzs - center, axis=1) < radius]
        if len(valid):
            sample = valid[np.random.choice(len(valid), size=min(len(valid), 3000), replace=False)]
            ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2],
                       s=1.2, c="gray", alpha=0.35, label="Triangulated points")

    margin = 0.5
    ax.set_xlim(ref_traj[:, 0].min() - margin, ref_traj[:, 0].max() + margin)
    ax.set_ylim(ref_traj[:, 1].min() - margin, ref_traj[:, 1].max() + margin)
    ax.set_zlim(ref_traj[:, 2].min() - margin, ref_traj[:, 2].max() + margin)
    ax.set_title("3D trajectory and bounded scene geometry")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_comparison_3d.png", dpi=200)
    plt.close()


def _draw_reprojection_overlay(refined_images, common, points3D, images_dir, out_dir):
    """Sanity-check overlay: project the cloud into one mid-capture frame."""
    if not points3D or images_dir is None:
        return
    name = common[len(common) // 2]
    img_path = Path(images_dir) / name
    if not img_path.exists():
        return
    img = cv2.imread(str(img_path))
    if img is None:
        return

    cams = getattr(_draw_reprojection_overlay, "_cams", None)
    if cams is None:
        return
    cam = cams.get(refined_images[name]["camera_id"])
    if cam is None:
        return

    ref = refined_images[name]
    xyzs = np.array([p["xyz"] for p in points3D.values()])
    cam_pts = (ref["R_w2c"] @ xyzs.T).T + ref["t_w2c"]
    cam_pts = cam_pts[cam_pts[:, 2] > 0.1]
    if not len(cam_pts):
        return
    uv = (camera_K(cam) @ (cam_pts[:, :3] / cam_pts[:, 2:3]).T).T
    h, w = img.shape[:2]
    for u, v, _ in uv:
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img, (int(u), int(v)), 2, (0, 255, 0), -1)
    cv2.putText(img, name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(str(Path(out_dir) / "reprojection_sample.jpg"), img)


# --------------------------------------------------------------------------
# 2. Multi-view reprojection: init poses vs. refined poses
# --------------------------------------------------------------------------
def reprojection_report(init_images, refined_images, cams_init, cams_ref,
                        points3D, out_dir=None):
    """Projects every 3D track into every view that observes it, once with the
    refined poses and once with the ARCore-init poses, and compares residuals.

    Observations always come from the refined model - a manual model has empty
    POINTS2D lines, so it has no observations of its own to offer.
    """
    refined_res, init_res = [], []
    for name, img_r in refined_images.items():
        valid = img_r["p3d_ids"] != -1
        if not np.any(valid):
            continue
        ids = img_r["p3d_ids"][valid]
        obs = img_r["obs_xy"][valid]
        keep = np.array([pid in points3D for pid in ids])
        if not np.any(keep):
            continue
        xyz = np.array([points3D[pid]["xyz"] for pid in ids[keep]])
        obs = obs[keep]

        cam_r = cams_ref.get(img_r["camera_id"])
        if cam_r is None:
            continue
        refined_res.extend(_project_residuals(xyz, img_r, cam_r, obs))

        img_i = init_images.get(name)
        if img_i is not None:
            cam_i = cams_init.get(img_i["camera_id"], cam_r)
            init_res.extend(_project_residuals(xyz, img_i, cam_i, obs))

    if not refined_res:
        print("[diagnostics] no observations to reproject, skipping")
        return {}

    refined_res = np.array(refined_res)
    summary = {"num_projections": int(refined_res.size),
               "refined_mean_px": float(np.mean(refined_res)),
               "refined_p95_px": float(np.percentile(refined_res, 95))}

    print("\n" + "=" * 58)
    print(" MULTI-VIEW REPROJECTION VALIDATION (ALL 3D TRACKS)")
    print("=" * 58)
    print(f"Evaluated 2D-3D projections:  {refined_res.size:,}")
    print(f"Refined mean error:           {summary['refined_mean_px']:.4f} px")
    print(f"Refined 95th percentile:      {summary['refined_p95_px']:.4f} px")

    if init_res:
        init_res = np.array(init_res)
        summary["init_mean_px"] = float(np.mean(init_res))
        summary["init_p95_px"] = float(np.percentile(init_res, 95))
        improvement = 100.0 * (1.0 - summary["refined_mean_px"] / summary["init_mean_px"])
        summary["improvement_pct"] = float(improvement)
        print("-" * 58)
        print(f"ARCore-init mean error:       {summary['init_mean_px']:.4f} px")
        print(f"ARCore-init 95th percentile:  {summary['init_p95_px']:.4f} px")
        print(f"Improvement from refinement:  {improvement:+.2f} %")
    print("=" * 58)

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "reprojection_report.json").write_text(json.dumps(summary, indent=2))
    return summary


def _undistort(pts, cam):
    """Map distorted pixel coords back to ideal pinhole pixel coords."""
    dist = camera_dist(cam)
    if not np.any(dist):
        return pts
    K = camera_K(cam)
    return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64),
                               K, dist, P=K).reshape(-1, 2)


def _project_residuals(xyz, img, cam, obs):
    rvec, _ = cv2.Rodrigues(img["R_w2c"])
    proj, _ = cv2.projectPoints(xyz, rvec, img["t_w2c"], camera_K(cam), camera_dist(cam))
    return np.linalg.norm(obs - proj.reshape(-1, 2), axis=1).tolist()


# --------------------------------------------------------------------------
# 3. Sampson epipolar error, straight from the raw DB matches
# --------------------------------------------------------------------------
def _sampson(F, pts1, pts2):
    Fx1 = (F @ pts1.T).T
    Ftx2 = (F.T @ pts2.T).T
    num = np.sum(pts2 * Fx1, axis=1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(denom, 1e-8))


def _sampson_pairs(db_path, cams, images, min_matches=15):
    """Yield (name1, name2, errors, control) per adjacent geometrically verified pair.

    errors scores the raw matches against the F built from the refined poses;
    control scores them against the F COLMAP's own matcher stored, or is None.
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    def keypoints(image_id):
        cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,))
        res = cursor.fetchone()
        if not res or res[0] == 0:
            return None
        rows, cols, blob = res
        # Keypoint stride varies (4 / 6 / N floats); only x, y are needed.
        return np.frombuffer(blob, dtype=np.float32).reshape(rows, cols)[:, :2].astype(np.float64)

    by_id = {img["id"]: img for img in images.values()}
    sorted_ids = sorted(by_id)

    try:
        for id1, id2 in zip(sorted_ids, sorted_ids[1:]):
            img1, img2 = by_id[id1], by_id[id2]
            cam1, cam2 = cams.get(img1["camera_id"]), cams.get(img2["camera_id"])
            if cam1 is None or cam2 is None:
                continue

            # Relative pose 1 -> 2, then F = K2^-T [t]x R K1^-1
            R_rel = img2["R_w2c"] @ img1["R_w2c"].T
            t_rel = img2["t_w2c"] - R_rel @ img1["t_w2c"]
            t_x = np.array([[0.0, -t_rel[2], t_rel[1]],
                            [t_rel[2], 0.0, -t_rel[0]],
                            [-t_rel[1], t_rel[0], 0.0]])
            E = t_x @ R_rel
            F = np.linalg.inv(camera_K(cam2)).T @ E @ np.linalg.inv(camera_K(cam1))

            pair_id = (MAX_NUM_IMAGES * min(id1, id2)) + max(id1, id2)
            # two_view_geometries, not matches: the latter is every putative SIFT match
            # including outliers, which drags the mean to tens of pixels no matter how
            # good the poses are. This table holds the geometrically verified inliers.
            cursor.execute("SELECT rows, cols, data, F FROM two_view_geometries WHERE pair_id = ?",
                           (pair_id,))
            row = cursor.fetchone()
            if not row or row[0] < min_matches or row[2] is None:
                continue

            kp1, kp2 = keypoints(id1), keypoints(id2)
            if kp1 is None or kp2 is None:
                continue

            raw = np.frombuffer(row[2], dtype=np.uint32).reshape(row[0], row[1])
            # COLMAP stores match indices in (min_id, max_id) column order.
            if id1 < id2:
                idx1, idx2 = raw[:, 0], raw[:, 1]
            else:
                idx1, idx2 = raw[:, 1], raw[:, 0]

            ok = (idx1 < len(kp1)) & (idx2 < len(kp2))
            if not np.any(ok):
                continue
            n = int(np.sum(ok))
            # F is pinhole, but keypoints are detected on the distorted image. Undistort
            # them first or a camera with non-zero k1/k2 reads as tens of pixels of
            # "pose error" that is really just lens distortion.
            pts1 = np.hstack([_undistort(kp1[idx1[ok]], cam1), np.ones((n, 1))])
            pts2 = np.hstack([_undistort(kp2[idx2[ok]], cam2), np.ones((n, 1))])

            control = None
            if row[3] is not None:
                control = _sampson(np.frombuffer(row[3], dtype=np.float64).reshape(3, 3),
                                   pts1, pts2)
            yield img1["name"], img2["name"], _sampson(F, pts1, pts2), control
    finally:
        conn.close()


def per_frame_sampson(db_path, cams, images, min_matches=15):
    """name -> median Sampson error over the adjacent pairs that frame appears in.

    Registering in the model is not the same as being well posed. A frame beside
    a tracking break gets a pose its own matches do not support, and pose drift
    against the capture's VIO will not reveal it - drift measures disagreement,
    not error. This does.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return {}

    per_frame = {}
    for name1, name2, err, _control in _sampson_pairs(db_path, cams, images, min_matches):
        median = float(np.median(err))
        for name in (name1, name2):
            per_frame.setdefault(name, []).append(median)
    return {name: float(np.median(v)) for name, v in per_frame.items()}


def sampson_report(db_path, cams, images, out_dir=None, min_matches=15):
    """Independent geometric validation: for each adjacent image pair, builds the
    fundamental matrix from the *refined poses* and measures the Sampson distance
    of the raw SIFT matches against it. Nothing here comes from COLMAP's own
    optimisation output, so a self-consistent-but-wrong BA still fails this.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"[diagnostics] {db_path} is gone, skipping Sampson check "
              "(re-run with --keep_database to keep it)")
        return {}

    # `control` scores the same correspondences with the F that COLMAP's matcher
    # stored for the pair. It is the reference: if the control is small and our
    # pose-derived error is not, the poses are wrong rather than this check.
    errors, control, pairs = [], [], 0
    for _n1, _n2, err, ctl in _sampson_pairs(db_path, cams, images, min_matches):
        errors.extend(err.tolist())
        if ctl is not None:
            control.extend(ctl.tolist())
        pairs += 1

    if not errors:
        print("[diagnostics] no usable match pairs for the Sampson check")
        return {}

    errors = np.array(errors)
    summary = {"pairs_evaluated": pairs, "correspondences": int(errors.size),
               "mean_sampson_px": float(np.mean(errors)),
               "median_sampson_px": float(np.median(errors)),
               "p95_sampson_px": float(np.percentile(errors, 95))}
    if control:
        summary["matcher_F_median_px"] = float(np.median(control))

    print("\n" + "=" * 58)
    print(" INDEPENDENT GEOMETRIC POSE VALIDATION (SAMPSON)")
    print("=" * 58)
    print(f"Adjacent pairs evaluated:     {pairs}")
    print(f"Correspondences:              {errors.size:,}")
    print(f"Mean Sampson error:           {summary['mean_sampson_px']:.4f} px")
    print(f"Median Sampson error:         {summary['median_sampson_px']:.4f} px")
    print(f"95th percentile:              {summary['p95_sampson_px']:.4f} px")
    if control:
        print(f"Control (matcher's own F):    {summary['matcher_F_median_px']:.4f} px median")
        if summary["matcher_F_median_px"] < 1.0 <= summary["median_sampson_px"]:
            print("  -> the same correspondences fit the matcher's F, so the")
            print("     correspondences are fine and the POSES are the problem.")
    # Median is the verdict driver: even verified inliers keep a few bad matches,
    # and a handful of them can hold the mean above 1 px on a sound model.
    if summary["median_sampson_px"] < 1.0:
        print("VERDICT: poses and intrinsics are consistent (< 1 px)")
    else:
        print("VERDICT: median epipolar error above 1 px - poses, intrinsics or")
        print("         distortion handling are suspect. Do not train on this.")
    print("=" * 58)

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "sampson_report.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------
# 4. Blender PLY export
# --------------------------------------------------------------------------
def write_line_ply(path, points, color_rgb):
    points = list(points)
    if len(points) < 2:
        return
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(points) - 1}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for pt in points:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{color_rgb[0]} {color_rgb[1]} {color_rgb[2]}\n")
        for i in range(len(points) - 1):
            f.write(f"{i} {i + 1}\n")


def write_pointcloud_ply(path, points, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors):
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {col[0]} {col[1]} {col[2]}\n")


def write_camera_frustums_ply(path, images, scale=0.08):
    pyramid = np.array([[0.0, 0.0, 0.0], [-0.5, -0.3, 1.0], [0.5, -0.3, 1.0],
                        [0.5, 0.3, 1.0], [-0.5, 0.3, 1.0]]) * scale
    vertices, edges = [], []
    for name in sorted(images):
        img = images[name]
        world = (img["R_c2w"] @ pyramid.T).T + img["t_c2w"]
        offset = len(vertices)
        vertices.extend(world)
        for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
            edges.append((offset + a, offset + b))

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 0 255 255\n")
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")


def export_blender_ply(init_images, refined_images, points3D, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if init_images:
        write_line_ply(out_dir / "initial_trajectory_raw.ply",
                       [init_images[n]["t_c2w"] for n in sorted(init_images)], (255, 50, 50))
    if refined_images:
        write_line_ply(out_dir / "refined_trajectory.ply",
                       [refined_images[n]["t_c2w"] for n in sorted(refined_images)], (50, 255, 50))
        write_camera_frustums_ply(out_dir / "camera_frustums.ply", refined_images)
    if points3D:
        write_pointcloud_ply(out_dir / "sparse_pointcloud.ply",
                             [p["xyz"] for p in points3D.values()],
                             [p["rgb"] for p in points3D.values()])
    print(f"\n[diagnostics] Blender PLYs written to {out_dir.resolve()}/")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_all(db_path, init_dir, refined_dir, images_dir, out_dir,
            export_ply=False, rot_thresh_deg=2.0, trans_thresh_m=0.05):
    """Runs every check and returns the merged report.

    `init_dir` is the manual (ARCore) model, `refined_dir` the post-BA model.
    Both must contain text files - run `colmap model_converter --output_type TXT`
    on the refined model first.
    """
    init_dir, refined_dir, out_dir = Path(init_dir), Path(refined_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams_ref = parse_cameras_txt(refined_dir / "cameras.txt")
    images_ref = parse_images_txt(refined_dir / "images.txt")
    points3D = parse_points3D_txt(refined_dir / "points3D.txt")

    cams_init, images_init = {}, {}
    if (init_dir / "cameras.txt").exists():
        cams_init = parse_cameras_txt(init_dir / "cameras.txt")
        images_init = parse_images_txt(init_dir / "images.txt")

    # _draw_reprojection_overlay needs the camera table; pass it without
    # threading it through every signature.
    _draw_reprojection_overlay._cams = cams_ref

    report = {"drift": drift_report(images_init, images_ref, points3D, images_dir,
                                    out_dir, rot_thresh_deg, trans_thresh_m),
              "reprojection": reprojection_report(images_init, images_ref, cams_init,
                                                  cams_ref, points3D, out_dir),
              "sampson": sampson_report(db_path, cams_ref, images_ref, out_dir)}

    if export_ply:
        export_blender_ply(images_init, images_ref, points3D,
                           out_dir / "blender_visualization")

    (out_dir / "diagnostics_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[diagnostics] reports written to {out_dir.resolve()}/")
    return report
