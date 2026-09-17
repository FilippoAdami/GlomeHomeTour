#
# Triangulates a COLMAP sparse point cloud (sparse/0/) for a scene that only
# has images/ + transforms.json (known camera poses, e.g. from an on-device
# SLAM/AR capture), by feeding those poses to COLMAP's point_triangulator
# instead of re-estimating them from scratch. See:
# https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses
#
# With --refine_poses the poses are then polished. --ba_mode pose_prior (the
# default) runs COLMAP's pose_prior_mapper with the ARCore camera centres as soft
# positional priors; --ba_mode unconstrained falls back to a plain bundle_adjuster
# in case the prior misbehaves on a given capture.
#
# --diagnostics writes the drift / reprojection / Sampson reports that used to
# live in the standalone COLMAP_adjustment sandbox (see colmap_diagnostics.py).
#

import os
import sys
import json
import shutil
import sqlite3
import argparse
import subprocess
import numpy as np
from scene.colmap_loader import qvec2rotmat, rotmat2qvec

# COLMAP enum values, read off colmap/src/colmap/util/types.h (SensorType) and
# colmap/src/colmap/geometry/pose_prior.h (PosePrior::CoordinateSystem).
SENSOR_TYPE_CAMERA = 0
COORD_SYSTEM_CARTESIAN = 1

# transforms.json stores camera-to-world with OpenGL/NeRF camera axes (X right,
# Y up, Z backward). COLMAP wants X right, Y down, Z forward. Right-multiplying
# c2w by this flips the Y and Z camera axes; it leaves the translation column
# (the camera centre, and hence the pose priors) untouched.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

# colmap::kMaxNumImages, the multiplier packing an image pair into one pair_id.
MAX_NUM_IMAGES = 2147483647


def run(cmd):
    print("+ " + " ".join(cmd))
    # ponytail: COLMAP's SiftGPU needs an offscreen GL context; under a Wayland
    # session Qt's native backend fails to create one, so force xcb. Harmless
    # (a no-op) under X11.
    env = {**os.environ, "QT_QPA_PLATFORM": "xcb"}
    subprocess.run(cmd, check=True, env=env)


# --------------------------------------------------------------------------
# Zoom-state camera grouping
# --------------------------------------------------------------------------
def group_by_zoom(transforms, rel_tol=0.01):
    """Split frames into contiguous runs that share a focal length.

    A phone's zoom moves in discrete steps, so run-length grouping on the
    per-frame focal length is enough - no clustering needed. Each group becomes
    one COLMAP camera. On a fixed-zoom capture this returns a single group and
    the pipeline behaves exactly as before.
    """
    g_fl_x, g_fl_y = transforms.get("fl_x"), transforms.get("fl_y")
    g_cx, g_cy = transforms.get("cx"), transforms.get("cy")
    g_w, g_h = transforms.get("w"), transforms.get("h")

    groups = []
    for frame in transforms["frames"]:
        fl_x, fl_y = frame.get("fl_x", g_fl_x), frame.get("fl_y", g_fl_y)
        if fl_x is None or fl_y is None:
            raise RuntimeError(f"Frame {frame.get('file_path')} has no focal length, "
                               "and transforms.json has no global fl_x/fl_y either.")
        fl_x, fl_y = float(fl_x), float(fl_y)

        if groups and _within(fl_x, groups[-1]["ref"][0], rel_tol) \
                  and _within(fl_y, groups[-1]["ref"][1], rel_tol):
            groups[-1]["frames"].append(frame)
        else:
            groups.append({"ref": (fl_x, fl_y), "frames": [frame]})

    for group in groups:
        frames = group["frames"]
        group["fl_x"] = float(np.mean([float(f.get("fl_x", g_fl_x)) for f in frames]))
        group["fl_y"] = float(np.mean([float(f.get("fl_y", g_fl_y)) for f in frames]))
        group["cx"] = float(np.mean([float(f.get("cx", g_cx)) for f in frames]))
        group["cy"] = float(np.mean([float(f.get("cy", g_cy)) for f in frames]))
        group["w"] = int(frames[0].get("w", g_w))
        group["h"] = int(frames[0].get("h", g_h))
    return groups


def _within(value, ref, rel_tol):
    return abs(value - ref) <= rel_tol * abs(ref)


def stage_zoom_groups(source_path, images_path, groups):
    """Symlink each group's frames into images_zoomgroups/cam_{i}/.

    feature_extractor --ImageReader.single_camera_per_folder then assigns one
    camera per folder, which is exactly one camera per zoom state.
    """
    staging = os.path.join(source_path, "images_zoomgroups")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    for i, group in enumerate(groups):
        folder = os.path.join(staging, f"cam_{i}")
        os.makedirs(folder)
        for frame in group["frames"]:
            base = os.path.basename(frame["file_path"])
            os.symlink(os.path.abspath(os.path.join(images_path, base)),
                       os.path.join(folder, base))
            frame["_colmap_name"] = f"cam_{i}/{base}"
    return staging


# --------------------------------------------------------------------------
# Manual model
# --------------------------------------------------------------------------
def build_manual_model(groups, distortion, db_path, manual_dir):
    os.makedirs(manual_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    name_to_ids = {name: (image_id, camera_id) for image_id, camera_id, name
                   in conn.execute("SELECT image_id, camera_id, name FROM images")}
    # An image with no geometrically verified match is missing from the
    # correspondence graph pose_prior_mapper builds, but point_triangulator keeps
    # it regardless; the mapper then aborts in FindCorrespondences ("key was not
    # found in unordered_node_map"). Leave such images out of the manual model -
    # with no matches they can carry no points anyway.
    connected = set()
    for pair_id, in conn.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0"):
        connected.add(pair_id // MAX_NUM_IMAGES)
        connected.add(pair_id % MAX_NUM_IMAGES)
    conn.close()

    if any(distortion):
        # 2DGS's dataset loader only accepts PINHOLE/SIMPLE_PINHOLE (undistorted)
        # cameras, so non-zero distortion here would silently be dropped.
        k1, k2, p1, p2 = distortion
        raise RuntimeError(
            f"transforms.json declares non-zero distortion (k1={k1}, k2={k2}, p1={p1}, "
            f"p2={p2}); undistort the images before triangulating, PINHOLE output "
            "would be wrong.")

    cameras_lines, images_lines, camera_owner, unmatched = [], [], {}, []
    for gi, group in enumerate(groups):
        for frame in group["frames"]:
            name = frame["_colmap_name"]
            if name not in name_to_ids:
                raise RuntimeError(
                    f"Image '{name}' from transforms.json was not picked up by COLMAP's "
                    f"feature_extractor. Check that it exists in the image folder passed to it.")
            image_id, camera_id = name_to_ids[name]
            if image_id not in connected:
                unmatched.append(name)
                continue

            if camera_id not in camera_owner:
                cameras_lines.append(
                    f"{camera_id} PINHOLE {group['w']} {group['h']} "
                    f"{group['fl_x']} {group['fl_y']} {group['cx']} {group['cy']}")
                camera_owner[camera_id] = gi
            elif camera_owner[camera_id] != gi:
                raise RuntimeError(
                    f"COLMAP gave camera {camera_id} to zoom groups "
                    f"{camera_owner[camera_id]} and {gi}. Expected one camera per "
                    "staged folder - was --ImageReader.single_camera_per_folder used?")

            # transform_matrix is camera-to-world in the OpenGL/NeRF axis
            # convention - "camera_model": "OPENCV" in transforms.json names the
            # *distortion* model, not the axis convention. Flip to COLMAP's axes
            # before inverting to world-to-camera. Measured against the verified
            # SIFT matches: unflipped these poses score 22.9px median Sampson
            # error, flipped 1.18px (the matcher's own F scores 0.45px).
            c2w = np.array(frame["transform_matrix"]) @ OPENGL_TO_OPENCV
            w2c = np.linalg.inv(c2w)
            qvec = rotmat2qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]

            images_lines.append(
                f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{tvec[0]} {tvec[1]} {tvec[2]} {camera_id} {name}")
            images_lines.append("")  # empty POINTS2D line, filled in by the triangulator

    with open(os.path.join(manual_dir, "cameras.txt"), "w") as f:
        f.write("\n".join(cameras_lines) + "\n")
    with open(os.path.join(manual_dir, "images.txt"), "w") as f:
        f.write("\n".join(images_lines) + "\n")
    open(os.path.join(manual_dir, "points3D.txt"), "w").close()
    print(f"Manual model: {len(groups)} camera(s), {len(images_lines) // 2} images")
    if unmatched:
        print(f"Dropped {len(unmatched)} image(s) with no verified match: "
              + ", ".join(unmatched[:8]) + (" ..." if len(unmatched) > 8 else ""))


# --------------------------------------------------------------------------
# Pose priors
# --------------------------------------------------------------------------
def write_pose_priors(groups, db_path, sigma_m):
    """Write each frame's ARCore camera centre into COLMAP's pose_priors table.

    Byte layout verified against colmap/src/colmap/scene/database_sqlite.cc
    (WriteStaticMatrixBlob writes raw little-endian float64 in Eigen storage
    order) and geometry/pose_prior.h (position 3x1, position_covariance 3x3,
    gravity 3x1). The covariance we write is isotropic, so its storage order is
    moot; gravity is written as NaN because PosePrior::HasGravity() tests
    allFinite() and zeros would advertise a bogus "down" direction.

    Two properties of this prior are worth keeping in mind when reading the
    diagnostics, because they differ from a "+/-5cm, +/-2deg constraint":
      - It is soft. pose_prior_mapper adds it to the BA cost as a Gaussian term
        weighted by the covariance. COLMAP has no hard box constraint, so 5cm is
        a 1-sigma expectation, not a limit.
      - It is position-only. COLMAP has no rotation prior at all, so rotation
        drift is caught after the fact by colmap_diagnostics.drift_report()
        and optionally corrected by --fix_rotation_drift.
    """
    covariance = np.diag([sigma_m ** 2] * 3)
    gravity = np.full(3, np.nan)

    conn = sqlite3.connect(db_path)
    name_to_ids = {name: (image_id, camera_id) for image_id, camera_id, name
                   in conn.execute("SELECT image_id, camera_id, name FROM images")}

    rows = []
    for group in groups:
        for frame in group["frames"]:
            image_id, camera_id = name_to_ids[frame["_colmap_name"]]
            position = np.array(frame["transform_matrix"])[:3, 3]
            rows.append((image_id, camera_id, SENSOR_TYPE_CAMERA,
                         position.astype("<f8").tobytes(),
                         covariance.astype("<f8").tobytes(),
                         gravity.astype("<f8").tobytes(),
                         COORD_SYSTEM_CARTESIAN))

    with conn:
        conn.execute("DELETE FROM pose_priors")
        conn.executemany(
            "INSERT INTO pose_priors (corr_data_id, corr_sensor_id, corr_sensor_type, "
            "position, position_covariance, gravity, coordinate_system) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.close()
    print(f"Wrote {len(rows)} position priors (sigma = {sigma_m * 100:.1f} cm, soft)")
    return len(rows)


# --------------------------------------------------------------------------
# Bundle adjustment
# --------------------------------------------------------------------------
def run_bundle_adjustment(mode, db_path, images_path, input_dir, output_dir, sigma_m):
    os.makedirs(output_dir, exist_ok=True)
    if mode == "pose_prior":
        # With a non-empty --input_path the mapper writes the refined model
        # straight into --output_path (no numbered sub-folder).
        run(["colmap", "pose_prior_mapper",
             "--database_path", db_path,
             "--image_path", images_path,
             "--input_path", input_dir,
             "--output_path", output_dir,
             # Trust the phone intrinsics per zoom group; only poses/points move.
             "--Mapper.ba_refine_focal_length", "0",
             "--Mapper.ba_refine_principal_point", "0",
             "--Mapper.ba_refine_extra_params", "0",
             "--Mapper.tri_ignore_two_view_tracks", "0",
             # Covariance is read from the DB; overwrite it here too so the sigma
             # in effect is visible on the command line, not just in the blobs.
             "--overwrite_priors_covariance", "1",
             "--prior_position_std_x", str(sigma_m),
             "--prior_position_std_y", str(sigma_m),
             "--prior_position_std_z", str(sigma_m),
             # ARCore occasionally resets tracking; robustify so those frames
             # cannot drag the whole trajectory.
             "--use_robust_loss_on_prior_position", "1"])
    elif mode == "unconstrained":
        run(["colmap", "bundle_adjuster",
             "--input_path", input_dir,
             "--output_path", output_dir,
             # NOTE: COLMAP 4.3 removed BundleAdjustment.refine_extrinsics; pose
             # refinement is now refine_rig_from_world.
             "--BundleAdjustment.refine_focal_length", "0",
             "--BundleAdjustment.refine_principal_point", "0",
             "--BundleAdjustment.refine_extra_params", "0",
             "--BundleAdjustment.refine_rig_from_world", "1"])
    else:
        raise ValueError(f"unknown ba_mode {mode!r}")


def to_txt(model_dir):
    run(["colmap", "model_converter",
         "--input_path", model_dir, "--output_path", model_dir, "--output_type", "TXT"])


def hold_rotations_at_init(refined_dir, manual_dir, flagged, out_dir):
    """Rewrite a manual model with the flagged frames' rotations reset to ARCore.

    The plan called for pose_prior_mapper --Mapper.fix_existing_frames here, but
    that flag holds *every* existing frame, not a subset. point_triangulator
    holds all poses fixed by construction, which is the same guarantee with a
    simpler tool: keep the refined poses we trust, restore ARCore rotation on the
    ones that drifted, then re-triangulate the points against that.
    """
    from colmap_diagnostics import parse_images_txt

    os.makedirs(out_dir, exist_ok=True)
    refined = parse_images_txt(os.path.join(refined_dir, "images.txt"))
    init = parse_images_txt(os.path.join(manual_dir, "images.txt"))

    lines = []
    for name in sorted(refined):
        img = refined[name]
        if name in flagged and name in init:
            qvec = init[name]["qvec"]
            R_w2c = qvec2rotmat(qvec)
            # Preserve refined camera center C (bounded by soft prior), reset orientation to ARCore
            C = img["t_c2w"]
            t = -R_w2c @ C
        else:
            qvec = img["qvec"]
            t = img["t_w2c"]
        lines.append(f"{img['id']} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                     f"{t[0]} {t[1]} {t[2]} {img['camera_id']} {name}")
        lines.append("")

    shutil.copy(os.path.join(refined_dir, "cameras.txt"),
                os.path.join(out_dir, "cameras.txt"))
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    open(os.path.join(out_dir, "points3D.txt"), "w").close()
    print(f"Reset {len(flagged)} drifted rotation(s) to the ARCore initialisation")


def triangulate(db_path, images_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    run(["colmap", "point_triangulator",
         "--database_path", db_path,
         "--image_path", images_path,
         "--input_path", input_dir,
         "--output_path", output_dir,
         # Default drops any point seen by only 2 images and never extended to a
         # 3rd - fine for the mapper's incremental reconstruction (which keeps
         # growing tracks), but with poses fixed nothing ever extends a track, so
         # it alone silently discards most of the cloud.
         "--Mapper.tri_ignore_two_view_tracks", "0"])


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Triangulate a COLMAP sparse point cloud from images + known poses (transforms.json)")
    parser.add_argument("-s", "--source_path", required=True, help="Folder containing transforms.json and an images/ subfolder")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential",
                        help="sequential is much faster for ordered video-like captures; exhaustive suits unordered photo sets")
    parser.add_argument("--images", default="images", help="Name of the images subfolder to use")
    parser.add_argument("--transforms", default="transforms.json", help="Name of the transforms JSON file to use")
    parser.add_argument("--database", default="colmap_database.db", help="Relative path for colmap_database.db")
    parser.add_argument("--refine_poses", action="store_true",
                        help="Refine the VIO poses after triangulation (see --ba_mode). Off by "
                             "default: point_triangulator already only accepts points consistent "
                             "with the given poses, so this is polish, not a correctness requirement.")
    parser.add_argument("--ba_mode", choices=["pose_prior", "unconstrained", "none"], default="pose_prior",
                        help="Which refinement to run under --refine_poses. pose_prior uses the ARCore "
                             "camera centres as soft priors (recommended); unconstrained is the old "
                             "plain bundle_adjuster, kept as an escape hatch since pose_prior_mapper "
                             "is a new COLMAP feature.")
    parser.add_argument("--prior_sigma_m", type=float, default=0.05,
                        help="1-sigma of the positional prior in metres (default 5cm). Soft, not a bound.")
    parser.add_argument("--zoom_tol", type=float, default=0.01,
                        help="Relative focal-length tolerance for grouping frames into zoom states")
    parser.add_argument("--diagnostics", metavar="OUT_DIR", default=None,
                        help="Write pose drift / reprojection / Sampson reports to OUT_DIR")
    parser.add_argument("--export_blender_ply", action="store_true",
                        help="With --diagnostics, also export trajectory/point-cloud/frustum PLYs")
    parser.add_argument("--rot_drift_deg", type=float, default=2.0,
                        help="Flag frames whose refined rotation drifted more than this from ARCore")
    parser.add_argument("--max_features", type=int, default=8192,
                        help="Maximum SIFT features per frame (default 8192)")
    parser.add_argument("--spatial_max_dist_m", type=float, default=2.5,
                        help="Maximum distance in metres for spatial loop closure matching (default 2.5m)")
    parser.add_argument("--no_spatial_matcher", action="store_true",
                        help="Disable spatial matching pass using pose priors")
    parser.add_argument("--fix_rotation_drift", action="store_true",
                        help="With --diagnostics, re-triangulate holding flagged frames at their "
                             "ARCore rotation")
    parser.add_argument("--keep_database", action="store_true",
                        help="Keep colmap_database.db (needed to re-run the Sampson check later)")
    parser.add_argument("--max_sampson_px", type=float, default=2.0,
                        help="Drop keyframes whose median Sampson error against the raw verified "
                             "matches exceeds this (default 2 px; 0 disables). Catches frames that "
                             "registered but got a bad pose, which rotation drift does not. Refuses "
                             "if it would drop more than 20%% of the frames.")
    parser.add_argument("--no_keyframes", dest="export_keyframes", action="store_false",
                        help="Skip the keyframe export. By default the frames COLMAP registered "
                             "are written to images/ plus images_2/4/8 and transforms_keyframes.json, "
                             "with every original frame kept in images_all/.")
    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    images_path = os.path.join(source_path, args.images)
    transforms_path = os.path.join(source_path, args.transforms)
    if not os.path.exists(transforms_path):
        sys.exit(f"No transforms.json found in {source_path}")

    with open(transforms_path) as f:
        transforms = json.load(f)

    db_path = os.path.join(source_path, args.database)
    manual_dir = os.path.join(source_path, "sparse_manual", "0")
    tri_dir = os.path.join(source_path, "sparse_triangulated", "0")
    sparse_dir = os.path.join(source_path, "sparse", "0")

    if os.path.exists(db_path):
        os.remove(db_path)  # feature_extractor refuses to reuse a stale database
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # 1. Zoom grouping decides whether COLMAP sees one camera or several.
    groups = group_by_zoom(transforms, args.zoom_tol)
    print(f"Detected {len(groups)} zoom state(s): "
          + ", ".join(f"{len(g['frames'])} frames @ fl_x={g['fl_x']:.1f}" for g in groups))

    if len(groups) == 1:
        for frame in groups[0]["frames"]:
            frame["_colmap_name"] = os.path.basename(frame["file_path"])
        colmap_images_path = images_path
        # Hand COLMAP the real intrinsics instead of letting it guess
        # focal = 1.2 * max_dim; the guess is what two-view verification uses.
        g = groups[0]
        camera_flag = ["--ImageReader.single_camera", "1",
                       "--ImageReader.camera_params",
                       f"{g['fl_x']},{g['fl_y']},{g['cx']},{g['cy']}"]
    else:
        colmap_images_path = stage_zoom_groups(source_path, images_path, groups)
        camera_flag = ["--ImageReader.single_camera_per_folder", "1"]
        print(f"Staged zoom groups under {colmap_images_path}/ - train with "
              f"--images {os.path.basename(colmap_images_path)}")

    # 2. Features + matches.
    run(["colmap", "feature_extractor",
         "--database_path", db_path,
         "--image_path", colmap_images_path,
         # Must match the model build_manual_model() writes into cameras.txt:
         # Reconstruction::Load() aborts on a model_id mismatch against the DB.
         # PINHOLE is the honest choice - build_manual_model() already refuses
         # non-zero distortion, and the 2DGS loader only accepts PINHOLE.
         "--ImageReader.camera_model", "PINHOLE",
         *camera_flag,
         "--FeatureExtraction.use_gpu", "1",   # SiftGPU via OpenGL, no CUDA needed on AMD
         "--FeatureExtraction.max_image_size", "1600",
         "--SiftExtraction.max_num_features", str(args.max_features)])

    ba_mode = args.ba_mode if args.refine_poses else "none"

    # 3. Write pose priors for spatial matching and bundle adjustment
    if ba_mode == "pose_prior":
        write_pose_priors(groups, db_path, args.prior_sigma_m)

    matcher = "sequential_matcher" if args.matcher == "sequential" else "exhaustive_matcher"
    matcher_cmd = ["colmap", matcher,
                   "--database_path", db_path,
                   "--FeatureMatching.use_gpu", "1",
                   # re-matches using epipolar geometry, recovers matches the SIFT ratio-test drops
                   "--FeatureMatching.guided_matching", "1"]
    if args.matcher == "sequential":
        matcher_cmd.extend(["--SequentialMatching.overlap", "5"])
    run(matcher_cmd)

    # Spatial matcher: uses ARCore pose priors to find loop closures across the room
    if args.matcher == "sequential" and ba_mode == "pose_prior" and not args.no_spatial_matcher:
        print(f"Running spatial matcher (max distance {args.spatial_max_dist_m}m) for loop closures...")
        run(["colmap", "spatial_matcher",
             "--database_path", db_path,
             "--FeatureMatching.use_gpu", "1",
             "--FeatureMatching.guided_matching", "1",
             "--SpatialMatching.ignore_z", "1",
             "--SpatialMatching.max_distance", str(args.spatial_max_dist_m),
             "--SpatialMatching.max_num_neighbors", "30"])

    # 4. Manual model from the ARCore poses.
    distortion = (transforms.get("k1", 0.0), transforms.get("k2", 0.0),
                  transforms.get("p1", 0.0), transforms.get("p2", 0.0))
    build_manual_model(groups, distortion, db_path, manual_dir)

    # 5. Triangulate, then optionally refine.
    if ba_mode == "none":
        triangulate(db_path, colmap_images_path, manual_dir, sparse_dir)
    else:
        triangulate(db_path, colmap_images_path, manual_dir, tri_dir)
        run_bundle_adjustment(ba_mode, db_path, colmap_images_path,
                              tri_dir, sparse_dir, args.prior_sigma_m)
    to_txt(sparse_dir)

    # 6. Diagnostics, and the rotation-drift second pass they enable.
    if args.diagnostics:
        from colmap_diagnostics import run_all
        report = run_all(db_path=db_path, init_dir=manual_dir, refined_dir=sparse_dir,
                         images_dir=colmap_images_path, out_dir=args.diagnostics,
                         export_ply=args.export_blender_ply,
                         rot_thresh_deg=args.rot_drift_deg)

        flagged = report.get("drift", {}).get("rotation_outliers", [])
        if flagged and args.fix_rotation_drift:
            total = report["drift"]["num_frames"]
            print(f"\nHolding {len(flagged)}/{total} rotation outliers (> {args.rot_drift_deg} deg) at ARCore orientation and re-triangulating...")
            held_dir = os.path.join(source_path, "sparse_held", "0")
            hold_rotations_at_init(sparse_dir, manual_dir, set(flagged), held_dir)
            triangulate(db_path, colmap_images_path, held_dir, sparse_dir)
            to_txt(sparse_dir)
            run_all(db_path=db_path, init_dir=manual_dir, refined_dir=sparse_dir,
                    images_dir=colmap_images_path,
                    out_dir=os.path.join(args.diagnostics, "after_rotation_fix"),
                    rot_thresh_deg=args.rot_drift_deg)
            shutil.rmtree(os.path.dirname(held_dir), ignore_errors=True)

    # 7. Keyframes, at the four resolutions the splatting tooling expects.
    if args.export_keyframes:
        from export_keyframes import export_keyframes
        export_keyframes(source_path, images_path, transforms, sparse_dir,
                         max_sampson_px=args.max_sampson_px, db_path=db_path)

    # 7. Clean up intermediates, keep only the triangulated sparse/0/. The
    # database is part of the deliverable once keyframes are exported - it is
    # what a later pass re-runs the Sampson check against.
    if not (args.keep_database or args.export_keyframes):
        os.remove(db_path)
    shutil.rmtree(os.path.dirname(manual_dir), ignore_errors=True)
    shutil.rmtree(os.path.dirname(tri_dir), ignore_errors=True)

    print(f"\nDone. Sparse point cloud written to {sparse_dir}")
    train_images = "" if len(groups) == 1 else f" --images {os.path.basename(colmap_images_path)}"
    print(f"You can now train directly with: python3 train.py -s {source_path}{train_images}")


if __name__ == "__main__":
    main()
