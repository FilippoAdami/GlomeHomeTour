# Depth Estimation and Surfel Initialization

This phase processes each scan under the workspace independently: it reads that scan’s images, transforms, and refined COLMAP model, predicts depth and surface normals, then builds an oriented surfel cloud. It writes depth maps, normal maps, validated poses, alignment state, and `points3D_depth.ply` into a `depth/` folder beside that scan’s `images/` and `sparse/` folders. Timing and completion metrics go to `pipeline_stats.json`; this phase does not write separate depth log or stats files. Scans are not fused together, and the standard pipeline skips visual/debug exports so per-scan outputs stay deployment-ready.

## Processing flow

```text
current_scene/
├── certificate.json
├── pipeline_stats.json  ← aggregate stage status
├── scan_0/
│   ├── images/
│   ├── sparse/0/
│   ├── depth/  → maps + poses + surfel PLY
│   └── pipeline_stats.json
├── scan_1/  (same layout)
└── ...
```

For every scan, the stage validates ordered COLMAP/OpenCV poses, runs DA3 on overlapping image windows, aligns window predictions against that scan’s COLMAP tracks, checks cross-view scale consistency, estimates normals, cleans depth, then initializes and regularizes surfels. After the point cloud is written, it derives scene dimensions and, when metric calibration is available, estimates floor count from the cloud and COLMAP camera path. One scan’s data is never used to align or fuse another scan.

This phase does not estimate or apply a global metric scale. It assumes the
COLMAP model and camera translations already use the intended world units, and
fits DA3's overlapping depth windows to the sparse COLMAP tracks so the depth
maps and surfels share those units. When the COLMAP phase has calibrated its
model, these outputs are in metres; otherwise they remain in the COLMAP model's
units. The calibration process and evidence are documented in the
[pose-refinement README](../01_poses_refinment/README.md). This phase does not
apply `metric_scale.json`'s factor to current COLMAP outputs a second time.

## Inputs and outputs

### Input layout

The workspace passed to `--workspace` is the parent containing `scan_N/` directories. Each scan must contain:

| Input | Use |
| --- | --- |
| `scan_N/images/` | RGB frames matching registered COLMAP image basenames. |
| `scan_N/transforms.json` | Frame metadata, COLMAP-aligned camera-to-world poses, and intrinsics for validation and depth priors. |
| `scan_N/sparse/0/images.txt`, `cameras.txt`, `points3D.txt` | Refined COLMAP poses, intrinsics, and sparse tracks used for depth alignment. |
| `scan_N/sparse/0/images.bin`, `cameras.bin` | Required only when explicitly generating camera preview diagnostics. |
| `scan_N/ingestion_results.json` | Identifies each scan in the current multi-scan ingestion layout. |

The phase also accepts a single-scene workspace for compatibility. For older scenes, the transforms manifest may be in `00_ingestion/transforms.json` while images and `sparse/0/` remain at the workspace root. Passing one `scan_N/` directly as `--workspace` processes only that scan.

### Production outputs

Each scan writes production depth artifacts to `scan_N/depth/` and operational files to `scan_N/`:

| Path | Purpose |
| --- | --- |
| `depth/depth_maps/<frame>.npy` | Cleaned DA3 depth aligned to the scan's COLMAP world scale, float16; metres when COLMAP has calibrated the model, otherwise the COLMAP model's units. |
| `depth/normal_maps/<frame>.npy` | Camera-frame unit normals, float16. |
| `depth/poses_da3.npz` | Ordered COLMAP `w2c`, intrinsics `K`, and image names. |
| `depth/depth_alignment.json` | Window alignment and cross-view outlier state; read by surfel initialization. |
| `depth/points3D_depth.ply` | Final oriented surfel cloud in the same world frame and units as the COLMAP model, consumed by the next 3DGS stage. |
| `depth/scene_extent.json`, `depth/scene_size.txt` | Scene dimensions, footprint and calibration status; includes the estimated floor count when metric scale is available. Derived from the depth point cloud and COLMAP camera poses; these reports do not modify the geometry. |
| `pipeline_stats.json` | Per-scan timing and completion metrics. The workspace root also keeps the aggregate pipeline status. |

The DA3 window cache is temporary: it remains available after a failure for resume, then is removed when the requested depth stage finishes successfully. The scan's COLMAP sparse model under `scan_N/sparse/` is read-only; the derived surfel cloud is written separately to `scan_N/depth/points3D_depth.ply`. The workspace root keeps the aggregate `pipeline_stats.json`; each scan has its own pipeline metrics file. This phase writes no `depth_log.txt` or `depth_stats.json`. The default run does not write depth/normal preview images, camera renders, summary Markdown, colored plane PLYs, plane legends, or roof reports. A normal run also removes those known debug artifacts if they were left by an earlier run. `--debug-output` enables them for inspection; `--no-plane-diagnostics` can skip the most expensive plane-enclosure reports while debug output is enabled.

## Running

Run all scans from `backend/`:

```bash
.venv/bin/python 02_depth_estimation/step_depth.py --workspace current_scene
```

Useful variants:

```bash
# Only one scan
.venv/bin/python 02_depth_estimation/step_depth.py --workspace current_scene/scan_0

# Only run the depth/normal pass for every scan
.venv/bin/python 02_depth_estimation/step_depth.py --substep depth --workspace current_scene

# Build surfels for every scan from existing maps and poses
.venv/bin/python 02_depth_estimation/step_depth.py --substep surfels --workspace current_scene

# Reprocess all scans; write debug previews and reports
.venv/bin/python 02_depth_estimation/step_depth.py --workspace current_scene --force --debug-output
```

`--substep all` is the default. `--force` reprocesses each scan instead of resuming completed output. If a scan fails, the stage logs the failure, continues processing other scans, then returns an error so downstream stages do not start with an incomplete set. Important inference controls are `--chunk-size`, `--overlap`, and `--process-res`; `--help` lists the remaining filters. The optional TSDF substep replaces the default surfel initializer and writes the same per-scan `points3D_depth.ply` output.

## Processing details

### 1. Pose preparation and depth scale alignment

`run_depth_estimation_substep()` loads the scene manifest and COLMAP text
model, orders camera poses to match image names, and validates rotation
orthonormality and camera centres. COLMAP poses are already OpenCV world-to-
camera transforms, which DA3 expects; they must not be converted a second
time. Camera-to-world OpenGL transforms are separately derived for geometric
unprojection and cross-view checks.

`DepthPriorEstimator.estimate_depth_sliding_window()` predicts depth in
overlapping windows. `ChunkTrackAligner` fits each window's scale and offset to
COLMAP tracks before overlapping predictions are blended. This brings windows
into the reconstruction's existing scale; it does not estimate the global
metric scale. A second test
compares overlapping frames' depth scales and marks outliers in
`depth_alignment.json`; unreliable frames are excluded from surfel
initialization, except upward-looking frames retained for ceiling coverage.
Optional post-processing includes RGB-guided filtering, sparse-landmark
anchoring, and global depth-graph optimization; these options are disabled by
default.

### 2. Normals and depth cleanup

Normals come from StableNormal Turbo by default. `--normal-model
depth_gradient` derives normals geometrically from the depth map and RGB-guided
polishing. `clean_all_depth_maps()` then masks saturation bloom, excessive
depth gradients, discontinuity ramps, implausible depth ceilings, grazing
views, and multi-view free-space inconsistencies; it also aligns compatible
cross-view depths. Cleaned maps are saved to disk; colorized side-by-side
images are emitted only with `--debug-output`.

StableNormal keeps the same Turbo weights, FP16 inference, 768-pixel processing
resolution, full-size output and unit-normal post-processing. On ROCm, its
inference temporarily disables exhaustive MIOpen convolution tuning and skips
the unused square-image warm-up. Text conditioning is cached under inference
mode so it does not retain the text encoder's autograd graph after unloading.

The depth entry point enables fused ROCm attention before DA3's first attention
call, because PyTorch caches `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` on first
use. This setting applies to both models in that process; unsupported attention
shapes still use PyTorch's fallback. An explicit environment value of `0`
preserves the slower attention path. No model, resolution, or frame-count
reduction is used.

On the RX 9060 XT, a controlled 320-frame benchmark at 768 processing resolution
fell from 237.46 s to 116.63 s including model loading, saving maps, and unloading
(the baseline already disabled expensive convolution tuning). Across all 320
saved maps, mean angular difference was 0.0204 degrees; each frame's 99th
percentile stayed below 0.352 degrees. These compare numerical agreement with
the original predictor, rather than accuracy against ground truth.

The fused attention setting also affects DA3. On a real four-view window at
756-pixel processing resolution, its depth differed from the math-attention
reference by 0.0445% mean relative error, 0.1423% at p99, and 0.6173% maximum;
all values were finite. This checks numerical agreement, not depth accuracy or
an isolated DA3 speedup. A recent 24-frame DA3 run with the fused setting took
6.37 s for model loading and calibration warm-up, then 11.92 s for seven
inference windows including alignment, fusion, and cache writes (1.49 s median
per steady-state window). These figures are workload context, not a before/after
speed comparison.

To measure loading, inference plus saved FP16 maps, unloading, and process time
including imports, use an empty output directory outside production outputs:

```bash
.venv/bin/python 02_depth_estimation/benchmark_normal_full_pass.py \
  --images-dir current_scene/scan_0/images \
  --output-dir ../agents_files/artifacts/stable_normal_optimization/new_run
```

The benchmark rejects nonempty output directories, propagates background write
failures, and writes timings and peak/final allocated GPU memory to
`benchmark.json`. `--limit N` allows a shorter smoke run.

### 3. Surfel initialization (default)

`SurfelCloudInitializer.initialize_from_keyframes()` unprojects valid depth
pixels using the image intrinsics and camera poses. Hybrid image/depth sampling
uses denser samples around detail and coarser samples on flat regions. It
constructs oriented tangent frames from normals, then applies the configured
geometric checks, normal consensus, optional free-space carving, optional
normal-tube collapse and multi-scale reduction, and statistical outlier
removal. The output is voxel-downsampled and capped at the configured surfel
budget before writing a PLY.

By default, `regularize_planes.regularize()` then detects major planes,
classifies horizontal/vertical/sloped surfaces, merges nearby plane
fragments, snaps near-Manhattan walls and floors, and fills supported planar
regions on an evenly spaced lattice. This is a heuristic geometric cleanup;
small furniture and clutter are not treated as structural planes.

### 4. Optional TSDF path

`TSDFVolume` integrates each frame into a bounded voxel grid, accumulates
truncated signed distances and color, then extracts weighted zero-crossing
surfels and estimates normals from TSDF gradients. This is an alternative to
the default multi-view surfel initializer, not an extra stage in the standard
pipeline. Its dense memory use depends on the full scene bounding-box volume.

## Module and function map

The public processing functions below describe the main inputs, outputs, and
role of each module. Helpers are included where they define a reusable
operation; CLI-only scripts are grouped by purpose.

## Metric scale reports and floor diagnostics

After the surfel or TSDF point cloud is written, `step_depth.py` automatically
uses `depth/points3D_depth.ply` and the COLMAP camera poses in
`depth/poses_da3.npz` to write `depth/scene_extent.json` and
`depth/scene_size.txt`. It reports robust cloud dimensions and footprint. For
floor count, it takes the cloud's estimated floor height as the vertical
reference, shifts the COLMAP camera centers to that reference, and counts
distinct camera-height bands. A floor count is reported only when the COLMAP
calibration metadata confirms metric units; otherwise dimensions are labelled
in COLMAP units and the floor count is unavailable. These reports do not change
the point cloud. This runs after the default `all` pipeline and after the
`surfels` or `tsdf` substeps; `--substep depth` only generates maps and has no
point cloud to measure. The standalone tools below can regenerate the reports
or create optional visualizations.

```bash
.venv/bin/python 02_depth_estimation/scene_extent.py --workspace current_scene/scan_0
.venv/bin/python 02_depth_estimation/floor_count.py --workspace current_scene/scan_0
.venv/bin/python 02_depth_estimation/topdown_view.py --workspace current_scene/scan_0
.venv/bin/python 02_depth_estimation/collapse_y_slice.py --workspace current_scene/scan_0
```

The COLMAP phase writes `scan_N/metric_scale.json` with its calibration result
and whether the factor has already been applied to the sparse model and camera
translations. This phase consumes the resulting COLMAP coordinates as-is; it
does not infer another scale or alter the source model. Extent and floor
reports use the calibration metadata to label dimensions as metres or COLMAP
units, and can convert legacy, not-yet-scaled geometry in memory for reporting
only. `keyframe_budget.read_scene_size()` accepts only reports marked in
metres. The top-down tools rotate around the gravity axis for wall alignment
but do not correct a tilt in the cloud's vertical axis. See the
[pose-refinement README](../01_poses_refinment/README.md) for calibration
method and limitations.

`scene_extent.py` writes the extent report and size summary;
`floor_count.py` writes a side-view diagnostic and updates floor count when a
metric scale is available. `topdown_view.py` and `collapse_y_slice.py` write
optional plan and floor-slice images using metres when calibrated, otherwise
COLMAP units.

| Module | Main functions / classes | Inputs → outputs and processing |
| --- | --- | --- |
| [`step_depth.py`](step_depth.py) | `main(argv)`, `run_depth_estimation_substep(workspace, ctx, ...)`, `run_surfels_substep(workspace, ctx, ...)`, `run_tsdf_substep(workspace, ctx, ...)` | CLI discovers scans and runs each independently. Depth substep consumes one scan's manifest, images, and COLMAP; returns in-memory cleaned maps, names, poses, and intrinsics while writing maps and normal arrays. Surfel substep loads those artifacts and writes a filtered/regularized PLY. TSDF substep loads the same scan's poses/maps and writes the alternative PLY. |
| [`../Utilities/metric_scale.py`](../Utilities/metric_scale.py), [`scene_extent.py`](scene_extent.py), [`floor_count.py`](floor_count.py), [`floor_bands.py`](floor_bands.py) | `estimate_metric_scale`, `load_depth_scene`, `load_metric_depth_scene`, `scene_extent`, `write_scene_extent_outputs`, `write_scene_size_txt`, `estimate_floor_count`, `count_floor_bands` | COLMAP estimates and applies scale before depth estimation when calibration is available. This phase does not re-estimate or reapply it; extent helpers read `metric_scale.json`, convert legacy unscaled geometry in memory for reporting, and write unit-labelled extent/floor metadata. |
| [`topdown_view.py`](topdown_view.py), [`collapse_y_slice.py`](collapse_y_slice.py) | `render_topdown`, `render_y_slice`, `render_floor_slices`, `find_floor_plane_y`, `get_camera_height_clusters` | Render the aligned cloud and camera trajectory from above or as floor height slices; images are saved under the scan's `depth/` folder. |
| [`colmap_poses_to_da3.py`](colmap_poses_to_da3.py) | `build_da3_poses(sparse_dir, names, scale, ...)`, `validate_poses(w2c, expected_centres, tol_m)` | Reads COLMAP text cameras/images, preserves COLMAP world-to-camera poses, builds scaled intrinsic matrices, and returns ordered `(w2c, K, names)`. Validation checks rigid rotations and optional camera-centre agreement. |
| [`depth_priors.py`](depth_priors.py) | `DepthPriorEstimator`, `ChunkTrackAligner`, `MetricDepthAligner`, `GlobalDepthGraphOptimizer`; `guided_filter_depth`, `robust_affine`, `cross_view_scale_outliers`, `anchor_depths_to_sparse_points`, `compute_surface_normals`, `extract_and_polish_normals`, `colorize_normals` | DA3 wrapper accepts images, poses, intrinsics, and window settings; returns depth, confidence, and uncertainty maps. Aligners fit prediction depths to sparse metric tracks or to each other. Outlier and anchoring functions return revised maps plus diagnostics. Normal functions map metric depth/RGB/intrinsics to camera-frame normals; colorization maps normals to RGB for inspection. `DepthEstimationError` distinguishes unusable model output from recoverable fallback conditions. |
| [`clean_depth_maps.py`](clean_depth_maps.py) | `clean_single_depth_gradient`, `clean_saturation_mask`, `clean_depth_ceiling`, `clean_depth_discontinuities`, `clean_grazing_angles`, `clean_all_depth_maps`, `gpu_multiview_depth_refine_and_carve`, `clean_depth_maps_in_place` | Individual filters accept a depth map and relevant image/normal/camera data, returning a masked map. `clean_all_depth_maps` accepts a sequence of maps, RGB frames, camera transforms, intrinsics, and optional normal maps; returns cleaned maps and per-filter counts. The in-place wrapper reads/writes workspace maps. The GPU refinement aligns depths across views and carves points that contradict observed free space. |
| [`initialization.py`](initialization.py) | `SurfelCloud`, `SurfelCloudInitializer.initialize_from_keyframes()`, `build_orthonormal_tangent_frame`, `compute_overexposed_mask`, `compute_hybrid_sampling_coords`, `filter_multiview_consistency`, `global_cross_view_freespace_carving`, `regularize_surface_normals_multiview`, `statistical_outlier_removal` | Cloud stores positions, colors, normals, tangent frames, and 2D scales; it supports PLY I/O, voxel downsampling, and optional reductions. Initializer consumes keyframes, depth/normal maps, intrinsics, and thresholds and returns the filtered oriented cloud. Helpers create tangent bases, sample pixels, classify cross-view support, cull free-space conflicts, refine normals, and remove statistical outliers. |
| [`regularize_planes.py`](regularize_planes.py) | `regularize(src, dst, args, write_diagnostics=True)`, `extract_planes`, `classify`, `merge_groups`, `fit_groups`, `snap_planes`, `fill_planes`, `align_storeys`, `diagnose_planes_quads`, `extract_and_save_roof_data` | Reads an oriented PLY and settings; estimates gravity, detects/merges planes, snaps supported structural orientations and optionally aligns storeys, then optionally resamples planar regions. Writes the requested PLY and report dictionary; optional diagnostics add colored PLY, Markdown legend, and roof metadata. |
| [`tsdf_fusion.py`](tsdf_fusion.py) | `TSDFVolume(bbox_min, bbox_max, voxel_size, ...)`, `integrate_frame(depth_map, rgb_img, c2w, intrinsics, ...)`, `extract_multiscale_surfels(min_weight, max_surfels, ...)` | Allocates dense TSDF/weight/color arrays for a world-space bounding box, integrates a depth/RGB frame from OpenGL camera-to-world coordinates, and converts sufficiently weighted zero crossings into an oriented `SurfelCloud`. |
| [`stable_normal.py`](stable_normal.py) | `StableNormalEstimator.load()`, `unload()`, `estimate_normal(image)` | Loads and releases the StableNormal Turbo predictor on the selected device. Maps an RGB NumPy/PIL image to float32 `(H,W,3)` unit normals in camera coordinates. Context-manager use releases model VRAM after inference. |
| [`keyframe_budget.py`](keyframe_budget.py) | `read_scene_size`, `frame_budget`, `voxel_coverage`, `coverage_topup`, `coverage_prune` | Reads scene dimensions; computes an empirical frame-count band; maps COLMAP tracks to occupied voxels; greedily adds/removes frames by coverage while preserving chain/covisibility constraints. This helper is not called by `step_depth.py`'s current default flow. |
| [`preview_cloud.py`](preview_cloud.py) | `intrinsics(cam)`, `render(xyz, rgb, R, t, fx, fy, cx, cy, W, H, down)` | Converts supported COLMAP camera parameters, then projects/colors cloud points into a simple camera-view preview. Used for visual QA, not reconstruction. |
| [`export_depth_plys.py`](export_depth_plys.py) | `build_camera_frustum_geom`, `save_ply_with_camera`, `main(argv)` | Reads saved maps and poses, optionally subsamples/cleans them, unprojects points, and exports PLYs with camera frusta for external inspection. |
| [`label_planes.py`](label_planes.py) | `label(src, args)`, `angle_matrix(planes)`, `candidate_merges(planes, args)`, `main()` | Diagnostic plane extraction and coloring; reports plane-to-plane angles and candidate merges. Does not feed a pipeline artifact. |
| [`run_sliding_window_reconstruction.py`](run_sliding_window_reconstruction.py) | `extract_depth_adaptive_keyframes(package, target_count)`, `main()` | Separate end-to-end runner that reads a capture package, quality-gates and motion-samples frames, runs DA3 sliding windows, aligns/ensembles predictions, initializes surfels, and stages a `GS_input` directory; flags can skip depth or launch training. It is not called by `step_depth.py` or the repository pipeline. |
| [`run_inspection_chunks.py`](run_inspection_chunks.py), [`diagnose_single_frame.py`](diagnose_single_frame.py), [`generate_robust_portrait_inspection.py`](generate_robust_portrait_inspection.py) | Script `main()` functions; local projection, plane-fit, and rendering helpers | Standalone tools: render selected depth chunks in orthographic views; infer one frame and report plane residuals/depth semantics; or select portrait frames and write side-by-side RGB/depth PNGs. They create diagnostics outside the normal pipeline. |
| [`benchmark_normal_full_pass.py`](benchmark_normal_full_pass.py), [`test_stable_normal.py`](test_stable_normal.py) | Script `main()` functions | Manual StableNormal timing/memory and smoke-check scripts. The benchmark requires `--images-dir` and an empty `--output-dir`; it never writes to production paths implicitly. These scripts are not invoked by the pipeline. |

`regularize_planes.py` also contains low-level geometry helpers for rotations,
plane bases, convex enclosures, occupied-area estimates, 2D ICP, and synthetic
self-check scenes. They support the functions listed above and are not
separate pipeline stages. `clean_depth_maps.gpu_free_space_carve()` is a
compatibility wrapper around the newer multiview refinement function;
`fit_minimum_enclosing_quadrilateral()` and `is_point_inside_quad()` are
compatibility aliases for polygon implementations.

### Supporting function inventory

These helpers complete the module map. Inputs and outputs are given in their
natural unit/type; helpers generally return a new array/report rather than
writing workspace artifacts unless noted.

| Module | Helpers and behavior |
| --- | --- |
| `depth_priors.py` | `DepthPriorEstimator.estimate_depth_streaming()` / `estimate_depth_sequence()` process image sequences without window blending; `estimate_depth_sliding_window()` batches overlapping images, blends predictions, and optionally calls an alignment callback. `estimate_depth()` predicts one image; `_generate_fallback_depth()` provides a fallback estimate, while `DepthEstimationError` signals unusable model output. `robust_affine()` fits prediction-to-metric scale/shift and returns fit plus residual statistic. `MetricDepthAligner.align_tracks()` / `align()` fit sparse track depth constraints; `ChunkTrackAligner.__call__()` applies this to one inference window and exposes frame status via `unreliable_frames()` / `stats()`. `anchor_depths_to_sparse_points()` returns maps aligned against COLMAP landmarks and diagnostics. `cross_view_scale_outliers()` compares overlapping map predictions and returns per-frame ratios and outlier names. `GlobalDepthGraphOptimizer.optimize()` builds overlap constraints, solves joint per-frame scale/shift, and returns aligned maps, residuals, and graph counts in `GlobalDepthGraphResult`. `guided_filter_depth()` returns an RGB-edge-guided depth map; normal helpers compute depth-gradient normals, polish them with RGB guidance, and colorize valid normals for images. `arcore_c2w_to_da3_w2c()` is a coordinate-conversion helper retained for callers that start with ARCore poses; the main pipeline uses refined COLMAP poses instead. |
| `initialization.py` | `SurfelCloud.from_random()` builds a seeded cloud; `voxel_downsample()` aggregates points by spatial cell; `normal_tube_collapse()` projects compatible nearby points onto a shared normal tube; `multiscale_pyramid_decimate()` reduces planar density while preserving detail; `to_ply()` / `from_ply()` serialize surfel fields. `estimate_adaptive_depth_ceiling()` derives a scene depth cutoff from confidence samples. `filter_multiview_consistency()` projects candidates into neighboring views and returns a boolean support mask; `global_cross_view_freespace_carving()` returns globally validated points/colors/normals; `regularize_surface_normals_multiview()` returns normals blended from agreeing views. `statistical_outlier_removal()` returns a geometric inlier mask. `SurfelCloudInitializer` configures these filters; `initialize_from_keyframes()` unprojects sampled valid pixels and returns the assembled cloud; its private scale/voxel/outlier helpers prepare final surfel attributes. |
| `regularize_planes.py` | `_rotate()` and `rotation_between()` transform vectors; `detect_up()` estimates gravity from normal directions; `plane_from_points()` / `plane_basis()` fit a plane and its coordinates. Polygon helpers `polygon_area()`, `intersect_lines()`, `compute_quad_metrics()`, `fit_minimum_enclosing_polygon()`, and `fit_2d_oriented_bbox()` map 2D plane samples to polygon/bounds and quality metrics. `occupied_area()` estimates supported area from occupied cells. `extract_planes()` returns RANSAC plane groups; `classify()` labels their orientation; `merge_groups()` clusters coplanar fragments; `azimuth_of()` / `wrap90()` normalize wall direction; `fit_groups()` computes merged plane stats; `filter_furniture_planes()` marks likely furniture; `snap_planes()` updates supported point/normal arrays. `is_point_inside_polygon()` returns a containment mask; `fill_planes()` appends lattice points within supported plane footprints while respecting holes and existing points. `wall_yaw()`, `find_storey_split()`, `icp_2d()`, and `align_storeys()` estimate and apply multi-storey yaw/translation alignment. `extract_and_save_roof_data()` writes roof metadata. `build_legend_markdown()` formats the report; `default_args()` supplies pipeline defaults; `diagnose_planes_quads()` writes non-Manhattan enclosure diagnostics. `selftest*()` functions build synthetic clouds for plane snapping, storey alignment, merging, fill, and sloped-roof checks; `build_parser()` / `main()` expose the standalone CLI. |
| `clean_depth_maps.py` | `clean_single_depth_gradient()` masks steep depth changes; `clean_saturation_mask()` removes bright low-chroma bloom; `clean_depth_ceiling()` masks samples beyond a scene limit; `clean_depth_discontinuities()` detects and removes edge-bridging ramps from RGB/depth gradients; `clean_grazing_angles()` masks rays too oblique to the supplied normals. `gpu_multiview_depth_refine_and_carve()` returns adjusted maps and counts after reprojection-based multi-view alignment/carving. `clean_depth_maps_in_place()` loads and replaces workspace maps; `main()` exposes those options as a standalone command. |
| `keyframe_budget.py` | `read_scene_size()` parses a text size report; `frame_budget()` returns a min/max count from floor area; `voxel_coverage()` converts track IDs to occupied voxel sets per frame; `coverage_topup()` greedily adds views to under-covered cells; `coverage_prune()` removes frames while respecting coverage/connectivity constraints. `_demo()` prints a small example. |
| `colmap_poses_to_da3.py` | `main()` loads a scene, matches manifest frame names to COLMAP, validates camera centres, and writes `poses_da3.npz`; `build_da3_poses()` performs the array construction and `validate_poses()` raises on invalid rigid transforms or excessive pose-centre drift. |
| `tsdf_fusion.py` | `TSDFVolume.__init__()` allocates the padded world-space voxel arrays; `integrate_frame()` projects voxel slices into an RGB/depth frame and updates truncated distance, weight, and color; `extract_multiscale_surfels()` finds weighted zero crossings, estimates normals/scales, and returns a `SurfelCloud`. |
| Diagnostic scripts | `diagnose_single_frame.py` rotates intrinsics for portrait imagery, unprojects one depth map, fits least-squares/RANSAC planes, and writes a point PLY. `run_inspection_chunks.py` writes point PLYs and top/side/front renders for selected chunks. `generate_robust_portrait_inspection.py` writes portrait-pose inspection renders. `export_depth_plys.py` exports selected/all depth frames with frusta. `preview_cloud.py` renders point samples to camera images. `label_planes.py` creates colored plane/angle diagnostics. `benchmark_normal_full_pass.py` and `test_stable_normal.py` time/check normal inference and report VRAM. Each script's `main()` parses local CLI options and writes to its requested output location. |

## Optimization and cleanup review

These are code-inspection findings; performance gains have not been benchmarked
for this review.

1. **Prioritize peak-memory measurement and streaming.** `step_depth.py` retains
   depth maps, confidence/uncertainty results and normals for the full capture
   while running post-processing; the lazy image reader only bounds decoded
   image caching. Large captures can therefore need substantial host and GPU
   memory. A chunked save/cleanup design could reduce peak use, but must retain
   enough neighboring views for cross-view alignment and normal estimation.
2. **Profile the optional dense TSDF before promoting it.** `TSDFVolume` stores
   three dense 3D volumes (TSDF, weights, color) sized from the scene bounds and
   projects every volume slice per input frame. This can become expensive as
   the bounding-box volume grows; sparse/block grids or a bounded ROI would be
   the next design to measure. The current pipeline leaves this path disabled.
3. **Check repeated pixel filtering.** The depth substep applies saturation
   masking and free-space checks before saving maps. The surfel initializer can
   apply saturation and free-space checks again. They operate at different
   stages (pixel depths versus candidate surfels), so compare per-filter
   rejection counts and output quality before disabling or consolidating either.
4. **Retire dormant options only after caller and artifact checks.** Sparse
   anchoring and global depth-graph alignment are implemented but disabled by
   default; keyframe budgeting is not wired into `step_depth.py`. Confirm no
   standalone workflow depends on their outputs before removing code or flags.
5. **Simplify compatibility and CLI surface when safe.** The two quadrilateral
   helpers and GPU free-space wrapper are aliases; `--only-depth` and
   `--only-surfels` duplicate `--substep`. They are small, but maintaining
   multiple names adds documentation and validation cost. Search external
   scripts before removal.
6. **Keep diagnostic work optional for throughput runs.** Plane enclosure
   reports, colored PLYs, previews, and paired JPEGs improve reviewability but
   add CPU and disk I/O. The plane diagnostic has a switch; consider a
   diagnostics profile only if stage timing shows these writes are material.

## Data and coordinate conventions

- DA3 receives COLMAP/OpenCV world-to-camera matrices (`w2c`). Do not invert or
  axis-flip them before inference.
- Unprojection and TSDF integration use OpenGL/ARCore camera-to-world (`c2w`),
  derived as `inverse(w2c) @ diag(1, -1, -1, 1)`.
- Intrinsics are kept in the same image order as pose names; matching by the
  image basename is required for maps, poses, and keyframes to stay aligned.
- Saved depth maps and `points3D_depth.ply` use the COLMAP model's world frame
  and units. They are in metres when the COLMAP phase applied a valid metric
  calibration, and otherwise remain in COLMAP units. This phase does not
  estimate or apply a global scale. Normal maps are unit vectors in camera
  coordinates.
- `points3D_depth.ply` is a surfel-style PLY with orientation/scale fields, not
  a triangle mesh. It is a training initialization for the following 3DGS stage.
