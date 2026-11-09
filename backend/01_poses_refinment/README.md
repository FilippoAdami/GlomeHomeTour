# Pose Refinement and Scene Geometry

`01_poses_refinment` is backend pipeline step **1** (the project calls it `colmap`; the directory keeps its historical spelling). It processes each scan independently. For ARCore captures, it runs image-only COLMAP, first excluding images with fewer than 200 extracted SIFT keypoints from matching and mapping. Direct-COLMAP captures keep their supplied poses. Both routes then remove unregistered images and frames with fewer than 30 triangulated 3D observations. When ARCore metric calibration is reliable, the stage applies the metres-per-COLMAP-unit factor directly to sparse points and camera translations; the resulting sparse model, PLY and transforms are in metres. Without a reliable calibration—including direct-COLMAP input—the model remains in COLMAP units. The cloud is then cleaned and axis-aligned with minimum Y at zero, and the same rigid alignment is applied to the COLMAP poses. `metric_scale.json` records the calibration evidence; it is not a scale that later stages should apply again.

The production entry point is `step_colmap.py`. A single scan may have its result manifest at the workspace root; multi-scan ingestion writes one under each `scan_N/` folder. Other files in this directory are standalone tools and are not invoked by the production step.

## Pipeline flow

```text
each ingestion_results.json + its relative image/pose/model paths
                    │
                    ├── source_format: arcore / arcore_app_certified
                    │     SIFT → remove <200-keypoint images → match → image-only mapper
                    │
                    └── source_format: colmap
                          load supplied model → feature-count filter
                    │
                    ▼
          remove unregistered images and those with <30 triangulated observations
          for ARCore: estimate scale from consistent early ARCore/COLMAP steps
          if reliable, scale sparse points and camera translations into metres
          clean cloud → align walls to X/Z and minimum cloud Y to zero
          apply rigid alignment to COLMAP poses; regenerate transforms.json
```

### Inputs and outputs

Each scan is selected from its `ingestion_results.json`; paths are relative to that scan. For ARCore captures, only the images are used for feature extraction, matching, mapping, and bundle adjustment. ARCore poses and position priors are not passed to COLMAP. The existing `transforms.json` supplies frame names, timestamps, and optional compass readings, then is rewritten with the final COLMAP poses and intrinsics. The continuous `trajectory.csv` is retained as source capture metadata and is not transformed or used as a pose source. For direct COLMAP input, the supplied sparse model is filtered without remapping. `--matcher` selects spatial candidate matching (default), exhaustive matching, or sequential matching for ARCore captures.

#### Spatial Candidate Pairing (Default Matcher)

Instead of testing all $O(N^2)$ exhaustive image pairs across the entire capture (~110,000 pairs for 470 frames, where >90% yield zero overlap across distant rooms), **Spatial Candidate Pairing** combines a sequential temporal window ($\pm 20$ frames) along the capture timeline with coarse ARCore camera positions as a 3D distance filter ($R \le 3.5\text{ m}$, up to 80 neighbors). This evaluates only ~11,000 candidate pairs ($10\times$ fewer pairs to evaluate). Because indoor line-of-sight is physically bounded by walls and room dimensions, cameras separated by $>3.5\text{ m}$ cannot share visual features anyway; thus, 100% of legitimate cross-pass loop closures are checked, while eliminating accidental false-positive matches between repetitive textures in distant rooms. Crucially, before bundle adjustment begins, all pose priors are completely purged from the COLMAP database: camera orientations, translations, and 3D points are solved 100% from visual feature correspondences without any ARCore drift contamination.

To ensure processing comfortably finishes **sub-5 minutes** on standard captures (~4.5 min), Global Bundle Adjustment triggering ratios are tuned via `--Mapper.ba_global_frames_ratio 1.25` and `--Mapper.ba_global_points_ratio 1.25`. This reduces the frequency of expensive full-scene Ceres Solver optimizations while the model grows, achieving virtually identical coverage and point counts to exhaustive matching (445 vs. 446 frames, 105k vs. 108k points) at roughly half the runtime (~4.6 min vs. 8.2 min).

The step writes or updates:

| Path | Contents |
| --- | --- |
| `<scan>/images/` | Kept images. ARCore images with fewer than 200 extracted SIFT keypoints and either-route images with fewer than 30 triangulated 3D observations are removed. |
| `<scan>/sparse/0/{cameras,images,points3D}.{txt,bin}` | Filtered COLMAP model, with points and camera poses transformed into the final aligned world frame. |
| `<scan>/sparse/0/points3D.ply` | Voxel-deduplicated and statistical-outlier-filtered initialization cloud, transformed with the sparse model. |
| `<scan>/metric_scale.json` | Early ARCore/COLMAP step-length calibration, rejected-transition evidence, scale-application status, and output units. This is an audit report; the factor is already applied to the sparse model and poses when calibration is available. |
| `<scan>/transforms.json` | For ARCore-origin captures, frame timestamps plus camera-to-world poses and intrinsics generated from the final COLMAP model. |
| `<scan>/trajectory.csv` | Original ARCore capture trajectory, retained unchanged as metadata; it does not align with the image-only COLMAP world. |
| `<scan>/ingestion_results.json` | Updated to `poses_format: colmap` and point `colmap_model_path` to `sparse/0` after this stage succeeds. `source_format` still records how ingestion received the capture. |

After editing text poses, the stage removes old binary and rig/frame records before
rebuilding the binary model. COLMAP infers fresh per-image frames from the final
image poses; stale `frames.txt` would otherwise override `images.txt`.

COLMAP databases, reports, and intermediate models stay in scratch storage. Visualizations, stage-specific folders, and discard folders are not produced. On ARCore-origin scans, images with fewer than 200 extracted SIFT keypoints are removed from the feature database before matching, saving matching and mapping work; this does not save feature-extraction time. The prefilter is not applied to direct-COLMAP scans, which skip SIFT extraction. Both routes then remove frames with fewer than 30 triangulated 3D observations. The stage repeats this second check after pruning tracks, since deleting one image can lower another image's observation count. Monocular COLMAP starts with arbitrary scale; wall alignment and moving minimum Y to zero only set orientation and origin. A validated ARCore calibration converts the output to metres; direct-COLMAP inputs remain in arbitrary units. When it replaces a model, the stage clears downstream `depth`, `train`, and `mesh` completion records so the pipeline recomputes them, while leaving their existing artifacts on disk.

## Coordinate and processing notes

For ARCore-origin captures, ingestion transforms are not read as poses or intrinsics by COLMAP; only frame names and timestamps are carried into the output manifest. COLMAP estimates a shared `SIMPLE_PINHOLE` camera and all camera poses from the ordered images. Direct-COLMAP input skips feature extraction, matching, and pose refinement; it only filters the supplied model by tracked 3D observations.

After image filtering, ARCore-origin scans estimate metres per COLMAP unit from corresponding early ARCore and COLMAP camera-center motion. Large adjacent-step disagreements are rejected and split the trajectory; the estimate then uses within-segment baselines to avoid carrying a detected local drift into scale calibration. The report records the accepted and rejected steps, robust estimate and consistency statistics. When the estimate is available, the factor is applied uniformly to every sparse 3D point and camera translation before cloud cleanup and axis alignment; the regenerated binary/text model, PLY and `transforms.json` therefore contain metric coordinates. The depth phase consumes those coordinates as-is and must not apply the recorded factor again. Direct-COLMAP scans have no ARCore scale reference and remain in arbitrary COLMAP units. This metric estimate is grounded in ARCore motion, not an independent surveyed room dimension.

The cleaned cloud uses 0.01-coordinate-unit voxel deduplication (one centimetre when metric calibration is available) and statistical outlier removal. The final rigid transform rotates the cloud so its dominant horizontal edges follow X/Z, then translates it so the cleaned cloud's minimum Y is zero. It is applied to every COLMAP point and camera pose. For ARCore-origin input, `transforms.json` is regenerated from those aligned poses using the mapper's shared `SIMPLE_PINHOLE` camera calibration.

Both routes estimate up from the consensus camera-up direction and fail if camera orientations are too inconsistent. This step does not emit scene dimensions, floor charts, or visual diagnostics. Multi-scan iteration is implemented here; later pipeline steps still expect a single scene at the workspace root.

## Module and function reference

### Production path and COLMAP conversion

#### `step_colmap.py`

| Function | Inputs → outputs and behavior |
| --- | --- |
| `discover_scenes(workspace)` | Workspace → ordered `(scene_dir, ingestion_results.json)` pairs; supports root-single-scan and `scan_N/` layouts. |
| `resolve_result_path(scene_dir, value, field, required=False)` | Scan dir and result-relative path → validated path contained within the scan, or `None`. |
| `load_ingestion_result(result_path)` | Result manifest → validates the supported format and resolves its image, transform, or COLMAP-model paths. |
| `count_registered_features(sparse_dir)` | Text COLMAP model → `{image_basename: tracked_3d_observation_count}`. This counts triangulated tracks, not untriangulated SIFT detections. |
| `scale_colmap_model(sparse_dir, scale)` | Text COLMAP model and metres-per-unit factor → multiplies point coordinates and camera translations uniformly, leaving rotations unchanged. |
| `_run_colmap(command)` | COLMAP argument list → runs the executable with the backend subprocess environment and raises on failure. |
| `discard_low_sift_images(database)` | COLMAP feature database → removes images and their extracted features when fewer than 200 keypoints were detected; returns rejected basenames before matching. |
| `run_image_only_colmap(images_dir, files, work_dir, matcher)` / `write_model_text(model_path, output_path)` | RGB images, ordered image list, scratch path and matcher / source model and destination → extracts features, applies the SIFT prefilter, matches and maps without pose priors, chooses the largest component and returns prefilter rejections, or converts a model to text. |
| `prune_model(sparse_dir, names)` | Sparse model and dropped basenames → removes image observations/tracks with COLMAP `image_deleter`. |
| `prune_low_feature_frames(sparse_dir, keep)` | Sparse model and mutable kept-image set → repeatedly deletes frames that fall below 30 registered 3D observations as tracks are pruned; returns removed basenames. |
| `clean_point_cloud(sparse_dir)` | Filtered model → writes and returns voxel-deduplicated, statistical-outlier-filtered XYZ/RGB arrays. |
| `_rotation_to_up(up)`, `_alignment_transform(xyz, sparse_dir)`, `transform_colmap_model(...)`, `write_colmap_transforms(...)` | Up vector / cleaned cloud and model / model and frame timestamp metadata → builds the rigid transform, applies it to sparse points and camera poses, then writes an aligned transforms manifest from COLMAP. |
| `_image_files(images_dir)`, `_stage_images(...)`, `_commit_scene(...)` | Image folder and kept basenames → stage the retained image set and atomically publish it with the sparse model and pose files. |
| `_remove_legacy_outputs(scene_dir)` | Scan folder → removes prior stage/debug outputs left by the older layout. |
| `process_scene(scene_dir, result_path, scratch_root, matcher, ctx)` | One scan and its ingestion result → runs image-only COLMAP for ARCore-origin images or filters an existing COLMAP model, then aligns and publishes outputs. |
| `_invalidate_downstream_stats(workspace, scene_dirs)` | Workspace and changed scans → removes stale depth/training/mesh completion records without deleting their files. |
| `main(argv=None)` | `--workspace`, `--matcher`, `--force` → processes all scans in order and records aggregate pipeline stats. `--force` reruns image-only mapping from the images currently present; deleted images are not recoverable by this step. |

#### `convert_transforms_to_colmap.py`

| Function | Inputs → outputs and behavior |
| --- | --- |
| `run(cmd)` | Argument-list command → runs a COLMAP subprocess and fails on nonzero exit. |
| `group_by_zoom(transforms, rel_tol=0.01)` / `_within(value, ref, rel_tol)` | Transform manifest → contiguous frame groups with approximately equal focal lengths, using relative tolerance; helper tests focal similarity. |
| `stage_zoom_groups(source_path, images_path, groups)` | Source, image root and groups → stages per-camera image folders (links where possible) for `single_camera_per_folder`. |
| `build_manual_model(groups, distortion, db_path, manual_dir)` | Zoom groups, distortion parameters, COLMAP DB and target dir → writes the initial sparse model from known ARCore poses and camera intrinsics. |
| `write_pose_priors(groups, db_path, sigma_m)` | Groups, COLMAP DB and positional sigma → writes camera-center priors into COLMAP’s pose-prior tables. |
| `run_bundle_adjustment(mode, db_path, images_path, input_dir, output_dir, sigma_m)` | Mode (`pose_prior`, `unconstrained`, `none`), DB, image/model paths and sigma → selects the configured BA route and writes the resulting model. |
| `to_txt(model_dir)` | Binary model directory → produces/updates its COLMAP text representation. |
| `hold_rotations_at_init(refined_dir, manual_dir, flagged, out_dir)` | Refined and initial models, flagged image names and output dir → restores selected initial rotations while retaining other refined poses. |
| `triangulate(db_path, images_path, input_dir, output_dir)` | Database, images and input/output model dirs → runs fixed-pose point triangulation. |
| `main()` | CLI options include source, matcher, BA mode/prior, feature and spatial matching limits, diagnostics, keyframe export and database retention → orchestrates the older ARCore-seeded conversion utility. It is not called by the production step. |

### Scene extent and floor diagnostics

These diagnostics now live in `02_depth_estimation/` and read that stage's
`depth/points3D_depth.ply` and `depth/poses_da3.npz`. See its README for the
commands and outputs. The wall-orientation primitive remains shared in
`Utilities/scene_alignment.py` because this stage uses it to axis-align the
COLMAP model.

### Keyframe selection and export utilities

#### `select_keyframes_arcore.py`

| Function | Inputs → outputs and behavior |
| --- | --- |
| `rotation_angle_deg(r_a, r_b)` | Two 3×3 rotations → relative angle in degrees. |
| `estimate_room_extent_from_arcore(frames, standoff_m=0.8, up_axis=1, floor_height_threshold_m=2.5)` | Transform frames → estimates trajectory floor bands, dilated horizontal footprint, per-floor dimensions and total area. |
| `compute_keyframe_budget(extent, total_frames)` | Estimated room extent and available count → low/high/target selection budget. |
| `select_keyframes_arcore(frames, target_count=None, min_translation_m=0.08, min_rotation_deg=4.0, standoff_m=0.8, sharpness_scores=None)` | Frames and optional budget/sharpness scores → selects geometrically spread frames, with local sharpness swaps; returns selected frames and metadata. |

#### `select_keyframes.py`

| Function | Inputs → outputs and behavior |
| --- | --- |
| `rotation_angle_deg(R_a, R_b)` | Batch of 3×3 rotations and one rotation → relative angles in degrees. |
| `select_keyframes(frames, min_translation, min_angle_deg)` | Ordered transform frames and separation limits → greedy subset; a candidate is dropped only when both its position and rotation are close to a previously kept frame. |
| `main()` | Source path and optional separation limits → copies selected images to `images_keyframes/` and writes `transforms_keyframes.json`. |

#### `export_keyframes.py`

| Function | Inputs → outputs and behavior |
| --- | --- |
| `registered_names(sparse_dir)` | COLMAP model directory → set of registered image basenames. |
| `sampson_rejects(db_path, sparse_dir, max_px)` | Match database/model/threshold → `{basename: median_pair_error_px}` for unsupported registered frames. |
| `filter_by_sampson(keep, rejects, max_reject_frac=...)` | Candidate names, rejects and cap → `(filtered_names, refused)`; refuses a suspiciously large rejection. |
| `_resize_one(args)` / `_link_or_copy(src, dst)` | Resize job tuple / source and destination → writes a resized image / hard-links with copy fallback. |
| `prune_model(sparse_dir, names)` | Model directory and image names → uses COLMAP `image_deleter` and converts the result back to text, removing observations/tracks too. |
| `export_keyframes(source_path, images_dir, transforms, sparse_dir, quality=95, max_sampson_px=None, db_path=None)` | Full image folder, transforms, model and optional Sampson settings → preserves originals as `images_all/`, exports registered keyframes to `images/`, `images_2/4/8/`, writes `transforms_keyframes.json`, returns frames. |

### Diagnostics, dense cloud and visualization

#### `colmap_diagnostics.py`

| Function | Inputs → outputs and behavior |
| --- | --- |
| `parse_cameras_txt(path)`, `parse_images_txt(path)`, `parse_points3D_txt(path)` | COLMAP text file → camera, image/observation, or 3D point mappings. |
| `camera_K(cam)`, `camera_dist(cam)` | Parsed camera → intrinsic matrix or distortion parameters. |
| `rotation_delta_deg(R_a, R_b)` | Two rotations → geodesic angular difference in degrees. |
| `drift_report(init_images, refined_images, points3D, images_dir, out_dir, ...)` | Initial/refined models, points and paths/thresholds → pose drift statistics and diagnostic plots/files. |
| `_plot_drift(...)`, `_plot_trajectories(...)`, `_draw_reprojection_overlay(...)` | Report data, poses, points and paths → drift, trajectory or projected-cloud diagnostic images. |
| `reprojection_report(init_images, refined_images, cams_init, cams_ref, points3D, out_dir=None)` | Initial/refined poses/intrinsics and tracks → compares observed-point reprojection residuals under both models. |
| `_undistort(pts, cam)`, `_project_residuals(xyz, img, cam, obs)` | Pixel observations/camera or 3D track/image/camera/observations → ideal pinhole pixels or projected residuals. |
| `_sampson(F, pts1, pts2)` | Fundamental matrix and corresponding points → Sampson errors. |
| `_sampson_pairs(db_path, cams, images, min_matches=15)` | Database and model maps → yields verified adjacent-pair errors under refined poses and optional matcher control. |
| `per_frame_sampson(db_path, cams, images, min_matches=15)` | Database/model → median pair error per participating frame. |
| `sampson_report(db_path, cams, images, out_dir=None, min_matches=15)` | Database/model and optional output dir → independent epipolar consistency report and plots. |
| `write_line_ply(...)`, `write_pointcloud_ply(...)`, `write_camera_frustums_ply(...)` | Path plus geometry/colors (and frustum scale) → diagnostic PLY geometry. |
| `export_blender_ply(init_images, refined_images, points3D, out_dir)` | Models, points and output dir → exports colored paths/cloud/frustums for Blender inspection. |
| `run_all(db_path, init_dir, refined_dir, images_dir, out_dir, ...)` | Match DB and initial/refined model/image/report paths → runs available diagnostics and returns merged report. |

#### `densify_pointcloud.py`

Standalone PyTorch plane-sweep densifier; **not called by `step_colmap.py`**. It can overwrite the derived `sparse/0/points3D.ply`; keep a backup if the current cleaned initialization cloud must be retained.

| Function | Inputs → outputs and behavior |
| --- | --- |
| `load_cameras(sparse_dir, images_dir)` | Binary COLMAP model and images path → sorted camera records with intrinsics, poses, centers and image paths. |
| `select_neighbors(cams, i, num_neighbors, search_window, max_angle_deg=20)` | Camera list/index and selection limits → viewing-compatible neighbors spread over the available baselines. |
| `load_gray(cam, max_side=640)` | Camera record and optional size limit → grayscale tensor and resize scale. |
| `scaled_K(K, scale)` | Intrinsic matrix and image scale → scaled matrix. |
| `apply_mat(pts, M)` | Batched (...,3) tensor and 3×3 matrix → matrix-transformed tensor using elementwise operations. |
| `plane_sweep(ref, neighbors, depths)` | Reference and neighbor image/camera records plus candidate depths → winner-take-all depth and matching-cost maps. |
| `backproject(cam, depth_map, color_img)` | Camera, depth map and color tensor → flattened world-space points, estimated world-space normals and flattened supplied color values. |
| `estimate_scene_depth(sparse_dir, cams)` | Binary sparse model and cameras → `(5th_percentile_depth, 95th_percentile_depth)` from sampled positive camera-space point depths. |
| `voxel_dedup(xyz, rgb, nrm, voxel_size)` | Point attributes and voxel size → one representative per voxel. |
| `remove_outliers(xyz, rgb, nrm, nb_neighbors, std_ratio)` | Point attributes and statistical filter parameters → filtered point attributes. |
| `main()` | Source path and plane, neighbor, resolution, threshold, stride, voxel, outlier and depth options → estimates and writes a denser `points3D.ply`. |

#### `visualize_camera_path.py`

| Function | Inputs → outputs and behavior |
| --- | --- |
| `write_obj(path, point_clouds, paths)` | Output path, colored clouds and named colored trajectories → writes a viewable OBJ with geometry/material data. |
| `load_transforms_path(transforms_path)` | Transform JSON → camera-center path. |
| `load_colmap_path(sparse_dir)` | COLMAP text model → registered camera-center path. |
| `load_colmap_points(sparse_dir)` | COLMAP text model → sparse XYZ/RGB points. |
| `main()` | Source path, optional transform file and output path → exports combined capture path, reconstructed path and sparse cloud OBJ. |

#### `inspect_pose_alignment.py`

Interactive desktop inspector using the existing Open3D dependency. It reads a processed scan without changing files or creating pipeline outputs. `transforms.json` supplies the ARCore poses; `sparse/0/` supplies COLMAP poses and the tracked points from `points3D.txt`. Both pose sets must already share the world frame, as they do after this stage.

| Function / class | Inputs → outputs and behavior |
| --- | --- |
| `Pose`, `Frame` | Validated OpenCV camera-to-world pose, calibration and image dimensions; frame records pair sources by unique image basename and retain COLMAP point track IDs. |
| `camera_calibration(...)` | Perspective COLMAP model parameters → camera pose and OpenCV calibration, preserving each source's intrinsics and distortion. |
| `select_scene(workspace, scan=None)` | Session or scan path → selected processed scan; selects the first numbered scan unless `--scan` is supplied. |
| `load_frames(scene)` | Scan images, transforms and COLMAP text/binary model → naturally ordered paired frames. Direct COLMAP input can omit ARCore data. |
| `load_cloud(scene)` | COLMAP `points3D.txt` → finite colored Open3D point cloud and point-ID-to-row index. |
| `frustum_mask(xyz, pose)` | Standalone projection helper for world points and a camera, including lens distortion; it does not drive the viewer's incremental point display. |
| `camera_wireframe(pose, depth)` | Camera pose and display depth → Blender-style camera outline with a pyramid, image rectangle and triangular up marker. |
| `pose_delta(frame)` | Paired frame → camera-center distance in metres and rotation difference in degrees. |
| `line_set(points, lines, color)` | Vertices and edges → colored Open3D lines for camera outlines, trajectories and pose-pair connectors. |
| `PoseAlignmentViewer` | Frames, cloud and point index → side-by-side RGB and interactive 3D window; navigation accumulates the points assigned to each frame. |
| `main(argv=None)` | Workspace and optional scan name → opens the desktop viewer. |

## Usage

Run the production stage from the backend directory:

```bash
cd backend
.venv/bin/python 01_poses_refinment/step_colmap.py --workspace current_scene
.venv/bin/python 01_poses_refinment/step_colmap.py --workspace current_scene --force
```

The stage uses the `colmap` executable found through the inherited `PATH`, matching the binary selected by the calling shell.

The main pipeline also runs it as:

```bash
.venv/bin/python run_pipeline.py --only-step colmap
```

Standalone inspection utilities (not run by the production step):

```bash
.venv/bin/python 02_depth_estimation/scene_extent.py --workspace current_scene/scan_0
.venv/bin/python 02_depth_estimation/floor_count.py --workspace current_scene/scan_0
.venv/bin/python 02_depth_estimation/topdown_view.py --workspace current_scene/scan_0
.venv/bin/python 02_depth_estimation/collapse_y_slice.py --workspace current_scene/scan_0
```

Open the interactive pose comparison from `backend/` on a desktop with a display:

```bash
.venv/bin/python 01_poses_refinment/inspect_pose_alignment.py --workspace current_scene --scan scan_0
# An individual scan folder also works:
.venv/bin/python 01_poses_refinment/inspect_pose_alignment.py --workspace current_scene/scan_0
```

The left pane shows the actual RGB image; the right pane shows both camera trajectories. ARCore camera outlines and paths are **blue**, COLMAP outlines and paths are **purple**. The selected camera is brighter and thicker; a yellow segment connects the selected ARCore/COLMAP pair. The point cloud starts empty. As you advance through naturally ordered images, the viewer adds and retains only points whose COLMAP track has an observation in the current image and an earlier image. COLMAP's final model does not store the frame when a point was first triangulated, so its second chronological observation is used as an incremental proxy. This makes early alignment changes visible while avoiding the misleading effect of showing every point in a camera frustum. Going back removes later point batches.

| Control | Action |
| --- | --- |
| Left / right arrows | Previous / next image, keeping the selected pose source. |
| Up / down arrows | Select COLMAP / ARCore for the same image. |
| `O` / Overview | Show the accumulated cloud, camera outlines and paths. |
| `V` / Camera view | Look through the selected pose with its own intrinsics and image aspect ratio; further navigation follows the selected camera. |
| Mouse | Orbit, pan and zoom in the 3D pane. |

The status line shows frame name, source, ARCore/COLMAP position and rotation differences, points added by the current frame, and the accumulated point count. A missing source is reported explicitly. Perspective PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV and FULL_OPENCV camera models are supported. Camera view uses a pinhole renderer: nonzero lens distortion is not applied to the 3D rendering, so raw RGB edges may differ for distorted inputs.

Inspector geometry and pose-convention checks: `.venv/bin/pytest tests/test_pose_alignment_viewer.py -q`.

The module self-checks are in `test_pose_pipeline.py` and `test_densify_pointcloud.py`; the latter is a standalone synthetic plane-sweep check rather than a pytest test. Run tests from `backend/` with `.venv/bin/pytest 01_poses_refinment/`.

`test_pose_pipeline.py` contains `test_zoom_grouping()`, `test_pose_prior_roundtrip()`, `test_se3_exp()`, `test_multiview_warp()`, `test_neighbor_selection()`, `test_keyframe_export()` and `test_sampson_filter()`. Its `_StubCam` supplies a minimal test camera record. `test_densify_pointcloud.py` uses `synth_view(K, R, t, H, W)` to make synthetic pinhole images and checks recovered depth and backprojection in its `main()` self-check.

## Optimization and maintenance observations

- Scene-extent and floor diagnostics run after depth reconstruction and are documented under `02_depth_estimation/`; this COLMAP stage removes any stale copies of their old outputs.
- `select_keyframes.py` and `select_keyframes_arcore.py` are alternate selection policies, neither used here. The 500-SIFT-keypoint prefilter removes very low-feature ARCore images before matching, but SIFT extraction itself still runs on every input image; assess retained-image counts and registration coverage when changing this cutoff.
- `export_keyframes.py` is a separate post-registration export utility. It preserves all source images in `images_all/` and keeps the COLMAP model consistent with exported image folders. It is not run by the production step.
- `densify_pointcloud.py` is an optional, expensive plane-sweep path, while the production step only cleans COLMAP’s sparse cloud. Its local plane-sweep path uses PyTorch and can increase initialization density, but its output has different provenance and it overwrites `points3D.ply`; compare downstream reconstruction quality before making it automatic.
- `colmap_diagnostics.py` contains report-specific plotting and parsing helpers; its Sampson functions remain available to `export_keyframes.py` and standalone diagnostics, but the production step no longer uses them.
- Direct-COLMAP inputs carry no gravity direction, so up is estimated from the consensus camera-up direction. A scan with inconsistent camera roll may fail this estimate rather than produce a questionable alignment.
- Certified ingestion manifests contain hashes of the pre-refinement images and transforms. Since this step intentionally edits those files, those hashes continue to describe the ingested source, not the post-COLMAP scene.
- `project_history.md` is historical implementation context, not an input to the pipeline. `__init__.py` is empty and exposes no package API; numbered stage folders are loaded through the backend path bootstrap instead of normal dotted imports.

Optimization work should first profile the COLMAP subprocess, which is explicitly the long-running stage, and measure frame count, matching time, registration coverage and downstream cloud quality together. Reducing input frames or enabling dense stereo can change geometry and should be treated as a quality/throughput trade-off, not a free speed-up.
