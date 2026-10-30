# Stage 04 — Gaussian splat to mesh reconstruction

Stage 04 renders expected depth from each stage 03 Gaussian model, fuses it with
registered photographs, removes floaters, seals holes, adds Gaussian infill and
refines mesh RGB. One `mesh` invocation runs every pass in this fixed order.

Production retains **one final mesh**. Raw, cleaned, sealed and color-refined
versions replace one another during processing rather than remaining as copies.
The original Gaussian checkpoint/PLY remain unchanged; stage 04 adds one sealed
checkpoint/PLY pair beside them in the same directory. Cleanup and color metrics
share `mesh.json`; standard step logs/stats follow the backend convention.

## Stage files

| Component | Responsibility | Runtime status |
| --- | --- | --- |
| [`step_mesh.py`](step_mesh.py) | Scan discovery, expected-depth rendering, TSDF fusion, complete linear orchestration, publication, resume and removal of obsolete generated copies. | Single production entry point. |
| [`mesh_cleanup.py`](mesh_cleanup.py) | Connected-component classification, vertex SOR, mesh sealing, face metadata and required Gaussian infill. | Called by `mesh`; explicit-input CLI for diagnostic cleanup. |
| [`fill_mesh_holes.py`](fill_mesh_holes.py) | Boundary tracing, triangulation, coplanar color transfer, PLY properties and inspection CLI. | Production geometry helpers; diagnostic outputs only when explicitly requested. |
| [`infill_gsplat.py`](infill_gsplat.py) | Sample planar Gaussians on cap/coverage-deficit faces; export sealed model siblings. | Required in every reconstruction. |
| [`regularize_mesh_rgb.py`](regularize_mesh_rgb.py) | Geometry-aware speckle cleanup, sharpening and luminance contrast; return color metrics. | Required final processing pass; no standalone CLI. |
| [`restoration_benchmark/`](restoration_benchmark/README.md) | Independent OSEDiff/SeeSR downloads, comparisons, timing and quality reports. | Experiment tooling; outside the production reconstruction sequence. |
| `vendor/SeeSR/`, `models/`, `env/`, `env_seesr/` | Downloaded restoration source, weights and isolated runtimes. | Optional restoration assets; not required for meshing. |

The reference below inventories first-party callables. Installed packages,
weights, caches and third-party SeeSR internals are dependencies with their own
documentation/license, rather than additional pipeline stages.

## Pipeline flow

```text
Refined COLMAP cameras/photos + original stage 03 Gaussian model
                              │
                              ▼
expected camera-Z depth → validity mask → RGB-D TSDF fusion
                              │
                              ▼
component filtering + SOR → hole filling + face labels
                              │
                              ▼
Gaussian infill → RGB refinement → publish final mesh and sealed model
                              │
                              ▼
mesh.ply + mesh.json + standard step log/stats
```

The source photographs provide mesh color; Gaussian SH colors are not rendered
for TSDF RGB. Stage 02 priors influence the trained geometry but are not fused
directly here. Scans remain independent; stage 04 does not register/merge them.

## Input contract

The CLI accepts a workspace containing `scan_N/`, one scan directory, or a
standalone refined COLMAP scene. It reuses stage 01 discovery: numeric scan
ordering, one ingestion manifest per scan, and rejection of mixed root/scan
manifests or incomplete scan manifests. Standalone scenes need no manifest.

The original Gaussian checkpoint/PLY remain unchanged. Phase 04 aligns the
registered COLMAP poses and optional Stage 02 poses in place so they match the
final mesh and sealed Gaussian model. The transform is recorded in `mesh.json`;
Stage 03 and forced Phase 04 rebuilds use it to recover source-frame camera
inputs. Re-running Stage 01 regenerates its source model and starts a new frame.

```text
scan_N/
├── ingestion_results.json
├── images/<registered image name>
├── sparse/0/{cameras.txt,images.txt,points3D.txt} # poses/points aligned in place by Phase 04
├── gaussian_splat/
│   ├── checkpoint.pt                 # original, unchanged
│   ├── gaussian_splat.ply            # original, unchanged
│   ├── checkpoint_sealed.pt          # stage 04 addition
│   └── gaussian_splat_sealed.ply     # stage 04 addition
├── mesh.ply                         # only retained mesh
├── mesh.json                        # fusion, cleanup, infill and RGB report
├── mesh_log.txt
├── mesh_stats.json
└── pipeline_stats.json
```

| Input relative to one scan | Convention | Processing |
| --- | --- | --- |
| `gaussian_splat/checkpoint.pt` | Torch checkpoint with `format="gsplat-stage03-v1"`, `resolution`, active `sh_degree` and `splats`. | Load on CPU with `weights_only=True`, check format, move render parameters to CUDA/ROCm. Other checkpoint formats fail explicitly. |
| `gaussian_splat/gaussian_splat.ply` | Original Gaussian PLY, not a mesh or RGB surfel cloud. | Check presence and record provenance; rendering reads checkpoint tensors. |
| `sparse/0/cameras.txt` | Refined pinhole intrinsics and source image dimensions. | Reuse stage 03 camera loading; scale intrinsics and image dimensions together. |
| `sparse/0/images.txt`, `points3D.txt` | COLMAP poses and sparse points. Before Phase 04 they are in the Stage 01 frame; after Phase 04 they are aligned in place. | Preserve file order/names/tracks while applying the rigid mesh transform. Forced rebuilds invert the recorded prior transform before processing. |
| `images/<name>` | Photographs referenced by registered cameras. | Decode RGB and resize to rendered depth dimensions; missing images fail explicitly. |
| `transforms.json`, depth/normal NPY maps, initialization clouds, metric reports | Other upstream artifacts. | Not alternative inputs here; never rewrite, regenerate or rescale them for meshing. `depth/poses_da3.npz`, when present, is aligned in place with the mesh. |

For `N` Gaussians, `splats` contains `means: (N,3)` in world coordinates,
`scales: (N,3)` in log space, scalar-first `quats: (N,4)`, opacity logits
`opacities: (N,)`, `sh0: (N,1,3)` and allocated `shN: (N,15,3)`. Rendering uses
`exp(scales)`, normalized quaternions and sigmoid opacity. Infill matches the
allocated SH slots even when short training uses fewer active bands.

### Coordinates and scale

- Cameras use COLMAP/OpenCV X right, Y down, Z forward. gsplat and Open3D receive
  the same world-to-camera transforms. No ARCore/OpenGL flips, floor offsets or
  further pose fitting are applied.
- Shared loading accepts `SIMPLE_PINHOLE`, `PINHOLE` and zero-distortion
  `SIMPLE_RADIAL`, `RADIAL`, `OPENCV` or `FULL_OPENCV`. Nonzero distortion and
  unsupported models are rejected; locked images/models are not undistorted here.
- Depth, TSDF distances, cleanup lengths/areas and infill sizes inherit upstream
  scene units. They are metres/metres squared only for calibrated scenes.
  Historical `_m` names do not certify metric scale.
- `--model-path` must select the original model matching the chosen scan's camera
  coordinate system. The checkpoint does not embed camera identities for proving
  that correspondence automatically.

## Processing

1. **Load and select views.** Use checkpoint resolution unless overridden by a
   positive integer `--resolution`; select `cameras[::frame_stride]`.
2. **Render/mask depth.** gsplat `render_mode="ED"`, `packed=False` returns
   opacity-normalized camera-Z depth and alpha under `torch.no_grad()`. Keep
   finite values with `0 < depth < depth_trunc` and `alpha >= min_alpha`; zero
   rejected pixels. No valid depth or no extracted triangles is an error.
3. **Fuse RGB-D.** Integrate source RGB with rendered depth into an Open3D CPU
   `ScalableTSDFVolume`, using `depth_scale=1.0` and RGB8 color. Write a temporary
   working mesh, leaving any accepted final mesh untouched during processing.
4. **Remove floaters.** Keep the largest connected triangle component. Other
   components need at least 4,000 triangles, area 0.15, PCA thickness 0.04,
   hull/OBB volume ratio 0.36 and boundary openness at most 0.065. Required SOR
   uses 30 neighbors/std-ratio 2.0; retain a triangle with at least two inlier
   vertices. Filtered geometry stays in memory instead of being exported again.
5. **Seal holes.** Trace directed rims, project to fitted planes, use constrained
   Delaunay caps, split conflicting internal edges and use fan fallbacks where
   needed. Zero-length rim edges can move slightly; cap centers recess by at most
   0.015 scene units. Original faces precede appended fill faces. Sample new colors
   from nearby source faces with progressively relaxed plane/normal gates. Require
   all loops filled and every final edge used twice; this does not establish
   absence of self-intersections. Replace the working mesh with its sealed result.
6. **Add Gaussian infill.** Target appended cap faces and original TSDF face
   centroids farther than 0.06 from existing Gaussian centers. Deterministic
   area-weighted sampling (seed 42) creates overlapping planar disks with thin
   normal scales, SH DC colors and zero new higher-order coefficients. Append
   them to the original tensors. Even a no-target result exports sealing metadata
   and an unchanged parameter set. Stage these files temporarily for publication.
7. **Refine RGB.** Process the sealed working mesh in place: normal-weighted
   neighborhood statistics (normal power 12; 500,000-face chunks), isolated
   speckle blending, strength-1.4 unsharp masking above detail threshold 0.004,
   then 6% luminance contrast around median luminance. Clamp RGB and preserve
   supported face properties. Read the export back and verify identical positions,
   triangle indices and cap flags before replacement.
8. **Align the final world frame.** Estimate up from major wall-normal families,
   oriented toward COLMAP camera up; align the dominant wall direction to X/Z and
   translate the supported floor reference plane to `Y=0`. Apply that rigid
   transform to the sealed Gaussian checkpoint/PLY, COLMAP poses/sparse points,
   and optional Stage 02 camera poses. The pose files are updated in their
   existing folders; no alignment output folder is created. The transform in
   `mesh.json` lets Stage 03 and forced Phase 04 rebuilds recover source-frame
   cameras. Stage 05 and Stage 06 consume the aligned frame directly.
9. **Publish and clean.** Replace the sealed Gaussian siblings and final `mesh.ply`,
   then publish the aligned poses and consolidated report. Remove the former
   `phase04_alignment/` sidecar after success. Failed processing preserves the
   accepted mesh, report and Gaussian pair. Individual replacements are atomic;
   the complete output set is not one filesystem transaction.

## Outputs and downstream interface

Phase 04 saves four files directly in each `current_scene/scan_N/` folder: `mesh.ply`,
`mesh.json`, `mesh_log.txt` and `mesh_stats.json`. The two sealed Gaussian files
are added to the existing `gaussian_splat/` directory, alongside its two original
files. No `04_3DGS_to_mesh/` output folder or additional Gaussian-model directory
is created. An explicit single-scan `--output` overrides the mesh/report/log/stats
directory; the default is the scan root.

| Retained artifact | Contents |
| --- | --- |
| `mesh.ply` | Final sealed, RGB-refined triangle mesh; raw/clean/pre-color/red copies are not retained. |
| `mesh.json` | Rendering/valid-depth metrics, raw/final counts, source/final paths, `cleanup` component decisions/timings, `gsplat_sealing` counts/paths and `rgb_regularization` metrics. |
| `mesh_log.txt`, `mesh_stats.json` | Standard phase log and metrics, matching other backend steps. Helper stdout remains on the console. |
| `gaussian_splat/checkpoint_sealed.pt` | Complete original-plus-infill tensor model, retaining checkpoint format and adding sealing metadata. |
| `gaussian_splat/gaussian_splat_sealed.ply` | Matching original-plus-infill Gaussian PLY, beside the unchanged original PLY. |
| `sparse/0/images.txt`, `points3D.txt` and optional `depth/poses_da3.npz` | Camera poses and sparse points transformed in place with the mesh; no Phase 04 alignment folder is produced. |
| Scan/parent `pipeline_stats.json` | Per-scan and aggregate completion; parent processing creates no duplicate mesh directory. |

### Retention policy

- **Mesh:** sealing and RGB refinement replace the extracted mesh. Only the final
  `mesh.ply` remains; raw, cleaned, pre-color and red diagnostic copies are not
  produced as retained production outputs.
- **Gaussians:** keep `checkpoint.pt` and `gaussian_splat.ply` unchanged. Add
  `checkpoint_sealed.pt` and `gaussian_splat_sealed.ply` in that same folder.
  These are complete original-plus-infill models using the existing formats,
  rather than delta-only files. Reruns replace only the sealed pair and never
  create numbered versions or backups of the original model.
- **Reports:** keep cleanup, infill and color metrics together in `mesh.json`,
  with the standard phase log/stats. Separate cleanup/RGB report files are not
  retained.
- **Working files:** use temporary mesh/model files while processing, then remove
  them after success or a handled failure. Rebuild from the original inputs for
  retries instead of retaining intermediate meshes.

### Metadata and consumers

PLY faces retain `is_hole_fill: uchar` (`0` original, `1` appended), plus
pre-refinement `fill_r/g/b` when available. Positions are float32, colors uint8
and indices int32. RGB refinement changes vertex colors and retains those face
properties; normals are computed as needed rather than stored by the custom writer.

The manifest's `gsplat_sealing.checkpoint_path` and `gsplat_sealing.ply_path`
select the sealed model for stage 06 rendering and web export. The original
`source_model_path` remains provenance. Explicit stage 06 checkpoint overrides
retain priority; manifests without sealing metadata retain the legacy source
fallback. Mesh consumers read `<scan>/mesh.ply`; sealed-model metadata is read from
`<scan>/mesh.json`.

Completion requires the mesh/report, both sealed siblings and cleanup/infill/RGB
summaries. Scans resume independently; failures withhold aggregate success while
other scans are attempted. Resume does not fingerprint settings or input contents:
use `--force` after changing those. A retry rebuilds from the original model/photos,
so RGB sharpening and Gaussian additions do not accumulate. There is no separate
RGB-only production step or saved pre-color mesh.

### Existing output directories

After a successful rebuild, known obsolete meshes/reports from older stage 04
versions are removed from the former `<scan>/04_3DGS_to_mesh/` directory
**only when using the default scan-root output**. This includes the former final
mesh, report, log and stats after their replacements succeed. The old directory
is removed if empty. Unknown files, symlinked stage/model directories, explicit
source models and custom-output directory contents are preserved. Existing
production directories are not cleaned merely by updating the code; cleanup follows a successful reconstruction.
The old stage-local Gaussian copy is removed when it is not the selected source;
its sealed replacement is now beside the original model. Older versions that
modified the original checkpoint in place cannot be reversed from that checkpoint.

## Running

From the repository root, use the existing backend environment and working
CUDA/ROCm Torch/gsplat installation from stage 03. Open3D, NumPy, SciPy, Pillow,
plyfile and Shapely are existing dependencies. Shapely/GEOS must expose
`constrained_delaunay_triangles`. Restoration weights/environments are unnecessary.

```bash
# Complete required sequence on every scan, or select one scan directory.
.venv/bin/python 04_3DGS_to_mesh/step_mesh.py --workspace current_scene

# Same complete phase through the pipeline orchestrator.
backend/.venv/bin/python backend/run_pipeline.py \
  --workspace backend/current_scene --only-step mesh

# Rebuild from upstream inputs after changing settings; replaces previous outputs.
backend/.venv/bin/python backend/04_3DGS_to_mesh/step_mesh.py \
  --workspace backend/current_scene/scan_0 --force

# Inspect the final mesh without creating another production mesh copy.
backend/.venv/bin/python backend/04_3DGS_to_mesh/fill_mesh_holes.py \
  --input backend/current_scene/scan_0/mesh.ply \
  --report agents_files/artifacts/mesh_boundaries.json
```

`--model-path` selects the original source directory; `--output` relocates mesh,
report and logs for one scan. Sealed siblings still go beside that selected
source model, and each successful run replaces them. Use an isolated model/scan
fixture for comparisons that must preserve an accepted sealed pair or production
completion state. Overrides cannot be shared across a multi-scan invocation.

| Extraction setting | Default | Meaning |
| --- | --- | --- |
| `--voxel-size` | `0.01` | TSDF cell width in scene units. |
| `--sdf-trunc` | `0.05` | Signed-distance truncation band. |
| `--depth-trunc` | `10.0` | Maximum accepted camera-Z depth, exclusive. |
| `--min-alpha` | `0.5` | Minimum accumulated alpha, inclusive. |
| `--frame-stride` | `1` | Use every Nth registered view. |
| `--resolution` | checkpoint value | Positive integer image downscale. |
| `--gsplat-max-ratio` | `0.10` | Addition budget `max(1, round(ratio * original_count))`; rounding/minimum matter for tiny models. |
| `--gsplat-target-disk-radius` | `0.035` | Area-based sampling density; final tangent radius also depends on target area/budget. |
| `--gsplat-overlap-factor` | `1.4` | Tangent-scale overlap multiplier. |
| `--gsplat-sample-density` | unset | Explicit spacing overriding disk-based density. |
| `--gsplat-normal-thickness` | `0.002` | Thin normal scale. |
| `--gsplat-coverage-deficit-threshold` | `0.06` | Centroid-to-nearest-Gaussian-center deficit distance. |

Distances must be positive/finite, alpha in `[0,1]`, and ratio in `(0,1]`.
There are no production infill/SOR bypass switches. Full rebuilds include RGB.
Standalone `mesh_cleanup.py` requires an explicit `--input` raw mesh and source
model for diagnostic threshold tuning; it invalidates full completion and writes
its report to `mesh.json`. It does not render or refine RGB. Its historical infill
spacing/thickness defaults are 0.015/0.001. Use the full production CLI to finish.
The RGB-only wrapper was removed because safe retries now rebuild instead of
keeping a duplicate pre-color mesh. `fill_mesh_holes.py` can still explicitly
produce selected/automatic/all-hole and red diagnostic meshes outside production.

## Function reference

### Extraction and color entry points

| Function | Inputs → outputs; processing |
| --- | --- |
| `step_mesh.camera_intrinsic(camera, o3d)` | Camera dictionary → Open3D pinhole intrinsic; transfers image size and `fx/fy/cx/cy` from `K`. |
| `valid_depth(depth, alpha, min_alpha, depth_trunc)` | Render arrays/thresholds → float32 depth with rejected pixels zeroed and valid-pixel count. |
| `extract_mesh(workspace, model_path, output, voxel_size, sdf_trunc, depth_trunc, min_alpha, frame_stride, resolution, ctx)` | One scan/model and fusion settings → temporary raw mesh plus summary dictionary; activates tensors, renders/masks depth, integrates RGB-D views and extracts TSDF triangles. Requires Open3D and a GPU. |
| `remove_obsolete_outputs(workspace, output, model_path)` | Completed default scan-root outputs → removes known generated copies from the former stage directory; preserves custom outputs, unknown files and the selected source model. |
| `mesh_scenes(workspace)` | Parent/scan path → ordered scan paths via the existing upstream discovery contract; standalone scenes need no manifest. |
| `mesh_scan(workspace, args)` | One scan and parsed settings → `0` or exception; validates checkpoint presence, resumes, runs fusion/cleanup/infill/RGB in order using temporary outputs; publishes one mesh, one report and sealed sibling models only after every pass succeeds. |
| `step_mesh.main(argv)` | CLI → `0` or exception/parser error; validates settings and override scope, processes scans independently, withholds aggregate success on failure. |
| `regularize_mesh_rgb.regularize_mesh_rgb(mesh_path, strength, detail_threshold)` | Colored mesh → checked atomic replacement plus report dictionary, without an extra report file; computes normal-weighted speckle cleanup, thresholded sharpening and luminance contrast, preserving topology and supported face metadata. |

### Cleanup, boundary geometry and texture sampling

| Function | Inputs → outputs; processing |
| --- | --- |
| `mesh_cleanup.profile_component_geometry(component_tris, all_vertices, tri_areas, comp_tri_indices)` | One component's faces/area indices → counts, area, PCA thickness, hull/OBB solidity, open-edge fraction, centroid and extents; handles insufficient/degenerate points. |
| `clean_mesh_floaters(input_mesh_path, output_dir, gsplat_model_path, thresholds, ..., output_mesh_path, gsplat_output_dir)` | Raw triangle PLY and cleanup/infill settings → one sealed mesh and a report dictionary; filters components/SOR, caps boundaries, validates edge counts, exports the required Gaussian derivative beside the original model, or in the caller's temporary model directory. The source checkpoint is required; missing input fails explicitly. |
| `mesh_cleanup.main(argv)` | CLI → diagnostic cleanup and `0`; requires an explicit raw input, resolves the source checkpoint, writes mesh.json and invalidates full completion. Does not discover a parent of multiple scans. |
| `fill_mesh_holes.export_ply_with_face_attributes(file_path, vertices, triangles, vertex_colors, face_attributes)` | Arrays plus named `uchar`/`int` face properties → binary little-endian triangular PLY and `True`; converts normalized RGB to uint8. I/O/unsupported-property errors propagate. |
| `read_ply_face_flags(ply_path, property_name)` | ASCII/binary triangle PLY → copied face-property array; returns `None` for absent file/face/property, rejects malformed PLY through the existing plyfile reader. Fixed three-index faces permit memory mapping for binary meshes. |
| `read_ply_face_property` | Alias of `read_ply_face_flags`; used for `fill_r/g/b` as well as flags. |
| `boundary_loops(triangles, vertices)` | Indexed faces and optional positions → simple boundary vertex loops; finds edges used once, matches directed continuations at junctions, splits repeated-vertex cycles. |
| `loop_geometry(points)` | Rim positions → planar 2D projection, maximum plane deviation, SVD axes. |
| `patch_loop(loop, vertices)` | Boundary vertex IDs/positions → consistently oriented cap faces, or `None`; constrains Delaunay triangles to the projected polygon and verifies count/vertex correspondence. |
| `edge_keys(triangles)` | Faces → packed uint64 undirected edge IDs for uniqueness, incidence and conflict checks. |
| `CoplanarTextureSampler.__init__(vertices, triangles, source_colors, normal_threshold, plane_tolerance)` | Source geometry/RGB → cached nondegenerate face centroids, normals, mean colors and KD-tree; disables sampling without valid colors/faces. |
| `CoplanarTextureSampler.query(positions, normals, k)` | Target points/normals → interpolated RGB array, or `None` if disabled; inverse-distance samples nearest source faces, relaxing geometric gates before nearest-face fallback. |
| `derive_infilled_face_textures(mesh, source_triangle_count, normal_threshold, plane_tolerance)` | Colored sealed mesh and original-face prefix length → per-added-face RGB, empty array with no added faces, or `None` without colors. |
| `fill_mesh(mesh, regions, auto, seal_all, normal_threshold, plane_tolerance)` | Mesh, optional region boxes and one fill policy → mesh plus per-loop decision records; triangulates/splits/recesses caps and colors new vertices. No accepted patches returns the input mesh. |
| `fill_mesh.add_patch(patch, loop, geometry)` | Candidate cap and rim → method name or `None`; stages local vertices/colors, resolves edge conflicts, rejects degenerate triangles, appends accepted patch data. |
| `add_patch.point(index)` | Existing/staged vertex ID → its position during cap construction. |
| `add_patch.color(index)` | Existing/staged vertex ID → color; currently unused. |
| `red_patches(mesh, source_triangle_count)` | Sealed mesh and original-face prefix → copied mesh with vertices incident to appended faces painted red. |
| `fill_mesh_holes.main(argv)` | Inspection/selective-fill CLI → boundary JSON and optional colored/red PLYs; validates mutually exclusive modes, region limits and nonoverlapping file paths. |

### Gaussian infill

| Function | Inputs → outputs; processing |
| --- | --- |
| `infill_gsplat.normal_to_quaternion(normals)` | One/many normals → scalar-first quaternions aligning local +Z; builds a right-handed tangent basis and defaults zero-length normals to +Z. |
| `sample_squeezed_gaussians_from_triangles(vertices, triangles, triangle_colors, max_gaussians, ..., sh_degree, sample_density_m)` | Colored faces and density/shape budget → CPU float32 splat tensor dictionary; rejects degenerate faces, stratifies area sampling, uses barycentric points, thin oriented disks, opacity logits and DC SH colors. Empty geometry returns correctly shaped empty tensors. |
| `export_splats_ply(path, splats)` | Torch/NumPy splat arrays → atomically replaced binary Gaussian PLY; channel-major SH layout, zero placeholder normals, 45 higher-order fields padded when needed. |
| `atomic_save_checkpoint(path, payload)` | Torch payload → temporary serialization and atomic replacement; cleans its temporary file on failure. |
| `seal_gsplat_model(gsplat_model_path, sealed_mesh, clean_num_triangles, ..., output_dir)` | Read-only trained checkpoint, mesh/original-face prefix and infill settings → derived model files and statistics. Matches allocated SH slots, handles no-target/already-sealed inputs, adds distinct `checkpoint_sealed.pt`/`gaussian_splat_sealed.ply` files in the source directory by default; returns explicit file paths and rejects aliases of original files. A caller can supply a staging directory. |

### Standalone restoration benchmark

This tooling consumes prepared **2D images**, not training checkpoints or TSDF
depth. Default input folders are under the historical flat
`backend/current_scene/04_3DGS_to_mesh/`; they are not produced by the scan-local
mesh CLI. `run.py --inputs`, `--originals`, and `--artifacts` select other roots.
See the [benchmark README](restoration_benchmark/README.md) for full setup,
recorded experiments, model settings and result interpretation.

| Function | Inputs → outputs; processing |
| --- | --- |
| `run.dump(path, data)` | JSON-compatible data → temporary JSON then atomic replacement; rejects NaN serialization. |
| `write_csv(path, rows)` | Record dictionaries → CSV using the union of encountered fields; empty records write an empty file. |
| `sha(path)` | File → SHA256 digest. |
| `select(inputs, originals)` | Blur-level image folders and optional originals → ten evenly spaced lexicographically ordered filenames, common dimensions, absolute paths and hashes; requires matching levels 3/5/7/13. |
| `Trace.__init__(path)` | Trace CSV path → resource sampler state, stop event and daemon thread. |
| `Trace.sample()` | Process and Linux DRM counters → timestamped phase/RSS/CPU/device-VRAM/busy rows every 200 ms until stopped. |
| `Trace.__enter__()` | Trace instance → starts sampling and returns itself. |
| `Trace.__exit__(*exc)` | Context completion → stops/joins sampler and writes resource CSV. |
| `amortized(rows, load, unload)` | Successful image timings and lifecycle overhead → 40-image-session amortized timing and projected ten-image timing. |
| `worker(args)` | One model/selection manifest → restored PNGs, per-image metrics CSV, resource trace, benchmark JSON and status; loads once, warms each resolution, times all four levels, stops on allocation failure, unloads. |
| `run.main()` | CLI → selection/batch manifests and isolated subprocess runs, or selection-only validation; returns failure unless all requested runs complete. |
| `model_adapters.load_model(name, artifacts_root, package_root)` | Model name and two asset roots → restoration callable, with SeeSR runtime metadata; loads isolated upstream inference source and configures FP16/tiling/tagger placement. Changes process import paths and working directory. |
| `load_model.init_tokenizer()` | Cached BERT assets → tokenizer with RAM special tokens; substitutes the OSEDiff developer-specific path in memory. |
| `load_model.restore(image, seed)` — OSEDiff | PIL image/seed → same-sized restored PIL image; RAM tagging, normalized model inference and AdaIN color correction. |
| `load_model.restore(image, seed, num_inference_steps)` — SeeSR | PIL image/seed/optional steps → same-sized restored image; CPU/GPU tagging, diffusion pipeline, tiled inference and AdaIN correction. Turbo's callable default is three steps; named benchmark workers explicitly request two/four. |
| `model_adapters.prepare(image)` | PIL image → Lanczos image with minimum side 512 and dimensions divisible by eight; restoration callables resize final output back. |
| `report.metrics(image, reference, perceptual)` | Result plus optional original/LPIPS model → sharpness and reference-based PSNR/SSIM, mean-color error, tolerant Canny edge precision/recall/F1 and optional LPIPS. |
| `panel(path, entries, crop)` | Named image paths → comparison PNG, optionally using 256-pixel center crops. |
| `report.report(batch_path)` | Exact batch manifest/saved runs → per-image quality metrics, per-level CSV/Markdown summaries, comparison panels and plots after restoration models unload. |
| `plots(directory, summary, baseline)` | Evaluated summaries/baseline → timing, memory, sharpness, perceptual/edge/color plots. |
| `setup.prepare_seesr_env()` | Existing shared environment/backend Torch → local SeeSR environment with its older Diffusers version layered over shared dependencies. |
| `prepare_blur()` | Original images and existing blurred images → verifies the Pillow blur/JPEG75 recipe and creates missing blur-13 files; rejects conflicting existing files. |
| `download()` | Asset definitions → local diffusion/RAM/SeeSR weights and verified hardlinks for shared text/VAE assets; uses bounded parallel downloads. |
| `download.fetch(job)` | Repository/pattern job → selected assets, canonical filename hardlinks; sanitizes download URLs in failures. |
| `setup.main()` | CLI → source checkouts, project-local environments, blur preparation and model downloads, or the selected setup subset. Not part of normal reconstruction. |
| `download_asset.transfer(url, destination, size, expected_sha, workers, chunk_size)` | Asset specification → verified file; exclusive lock, resumable ranges, retries, final size/SHA validation and atomic publication. |
| `transfer.length(index)` | Chunk index → expected byte count, including the shorter last chunk. |
| `transfer.fetch(index)` | Chunk index → verified-size local part and byte count; reuses completed ranges, refreshes signed URLs, validates HTTP Content-Range and retries failures. |
| `download_asset.download_asset(name)` | Named RAM/SeeSR asset → resolves repository revision/LFS metadata and invokes validated transfer. |
| `download_asset.main()` | Asset CLI → download one allowed asset. |
| `test_benchmark.BenchmarkTests.fixture(root, count)` | Temporary directory/image count → deterministic local original and blur-level images for tests. |

The benchmark's `test_benchmark.py` contains focused checks for asset-root
separation, deterministic selection, dimension mismatch, amortization, reference
metrics, one-load/four-level execution, named Turbo step overrides, the callable
default, allocation-failure stopping, report preservation, legacy timing
correction, and resumable transfer. Its test functions assert those contracts;
they do not run the actual restoration models.

## Performance and quality considerations

### Opportunities to measure

| Area | Candidate change | Expected tradeoff / required evidence |
| --- | --- | --- |
| TSDF throughput | Profile render, device-to-host copies, RGB decoding, integration, extraction and export separately. | Current fusion has progress notes, not a phase breakdown. Establish the bottleneck before changing the implementation. |
| View cost | Increase `--resolution` downscale or `--frame-stride`; consider coverage-aware view selection. | Less rendering/fusion work can miss thin geometry or texture detail. Compare coverage, boundaries, geometry and color on the same scene. |
| Volume size | Increase voxel width and choose a corresponding truncation band. | Lower spatial detail and memory demand; inspect thin structures, room corners and reconstruction bias. |
| Component classification | Group face indices by component once and precompute component areas. | The current per-component `clusters == cid` mask scans all faces repeatedly, including tiny rejected components. A grouped implementation can remove that repeated work; verify identical retained faces/decisions before timing. |
| Coplanar sampling | Batch/vectorize the per-query fallback selection; reuse source-face/KD-tree work where lifetimes allow. | Preserve strict-to-relaxed fallback ordering and compare sampled colors. Avoid replacing it with unrestricted nearest-color transfer. |
| RGB pass | Reuse or recompute cached edge/normal weights after measuring the memory tradeoff. | Two passes intentionally use different color fields; blindly merging them changes the algorithm. Reusing adjacency costs memory on large meshes. |
| Gaussian derivative | Profile/batch centroid queries, sampling and checkpoint/PLY export while retaining every pass. | Verify identical targets, counts and output tensors; required infill cannot be skipped to save time. |
| Restoration | Tune tile sizes/tagger placement only in the separate benchmark. | Image sharpness does not prove reconstruction accuracy; generated detail needs multi-view consistency and geometry checks before integration into training/fusion. |

These are investigation targets, not measured speedup claims. The input-contract
changes do not alter the TSDF/rendering or RGB algorithms. Synthetic regressions
and an isolated reduced-resolution scene run check integration; they do not
establish full-resolution quality or throughput improvements.

### Quality limits and maintenance observations

- **Scale-sensitive thresholds:** fixed scene-unit voxel, cleanup, cap and
  Gaussian sizes can behave differently for uncalibrated COLMAP scenes. Choose
  settings using known scene scale; do not apply another calibration to locked
  upstream data inside this phase.
- **Aggressive sealing:** production closes every boundary, including large
  openings. Doors, windows and genuinely unobserved regions can acquire
  invented surfaces. The standalone selective modes support inspection, but
  view-based evidence would be needed for safer production hole classification.
- **Detached-object pruning:** size/thickness/solidity/open-edge gates can remove
  real thin or partially observed objects. Inspect the final mesh and component manifest in `mesh.json` before tuning thresholds.
- **Coverage is approximate:** centroid distance to Gaussian **centers** ignores
  Gaussian extent, alpha and visibility. It can classify broad existing splats
  as deficient, while a global area budget gives larger regions more samples.
  The rounded, minimum-one budget is not a strict fractional ceiling for tiny
  models. Tightening this needs explicit sampling/coverage regression checks.
- **Repeated color work:** `fill_mesh()` computes an `infill_face_colors` local
  result that is not returned; cleanup subsequently rebuilds a sampler to derive
  face colors. Its nested `color()` helper is unused. These are cleanup targets;
  retain vertex-color generation and face-order guarantees when simplifying.
- **Shared exporters:** Gaussian PLY/checkpoint export logic overlaps stage 03.
  Keep the boundary stable; consider a shared exporter only if both stages are
  changed together in a separately authorized task. The stage 04 exporter pads
  lower-order SH fields, while the trainer allocates all degree-three slots.
- **Artifact checks:** source Gaussian PLY presence is currently required even
  though rendering needs only the checkpoint. This verifies the stage 03 pair;
  it is redundant for the renderer itself.
- **No stored intermediate meshes:** raw/clean/pre-color meshes and automatic red
  copies are deliberately absent. Inspect face flags/metrics or reconstruct in an
  isolated fixture when diagnosing a change. Full retries prevent compounded RGB.
- **Optional restoration storage:** two local environments, vendor source and
  multi-gigabyte weights are outside the production mesh path. Keep generated
  assets out of source control. Consolidating the environments is unsafe without
  resolving their different upstream Diffusers requirements. The SeeSR adapter
  is also imported by stage 06 texture enhancement, so it is not dead code.
- **Restoration isolation:** adapters mutate cwd/import paths; retain subprocess
  isolation for conflicting upstream packages. OSEDiff's RAM tagger always moves
  to GPU, unlike SeeSR's headroom check. Real-model memory behavior needs separate
  validation; reporting also requires any manifest-listed originals to remain
  available.
- **Publication/resume:** working mesh, Gaussian pair and report are staged until
  processing succeeds; prior accepted outputs survive pre-publication failures.
  Individual replacements are atomic, not a joint multi-file transaction. A hard
  process kill can leave temporary files; do not infer completion from file
  existence alone. Custom-output runs preserve pre-existing unrelated files.

## Verification

From the repository root:

```bash
backend/.venv/bin/pytest \
  backend/tests/test_mesh_stage_inputs.py \
  backend/tests/test_fastgs_mesh.py \
  backend/tests/test_gsplat_sealing.py \
  backend/tests/test_mesh_floater_removal.py \
  backend/tests/test_fill_mesh_holes.py \
  backend/tests/test_extract_non_structural_frames_gs.py \
  backend/tests/test_pipeline_steps.py -q

backend/04_3DGS_to_mesh/env/bin/python -m pytest \
  backend/04_3DGS_to_mesh/restoration_benchmark/test_benchmark.py -q
```

The focused tests cover upstream-byte preservation, allocated SH bands,
no-target exports/counts, source/output collision rejection, missing models,
numeric settings, scan ordering/resume/partial failure, PLY metadata,
boundary sealing, component filtering, and color retry/topology preservation.
Stage 03/04 shared checks also verify camera scaling, intrinsic transfer, depth
masking and the trainer's checkpoint/PLY contract. Most checks run on CPU;
mocked fusion does not establish native rasterizer/TSDF integration.

Compact-output verification is kept under
[`agents_files/artifacts/stage04_compact/`](../../agents_files/artifacts/stage04_compact/).
Tests cover exact retained file sets, same-folder sealed additions, unchanged
original bytes, metadata/geometry preservation, safe forced retries, missing
outputs, default-folder migration and preservation of unrelated custom outputs.
The GPU check compares the compact result against the previous coarse result,
including positions, topology and colors, and records original-file SHA256 hashes.
The final focused suite passed **86 tests**. The 10-view GPU comparison preserved
positions, indices and colors exactly while reducing retained mesh files from
five to one: 5,987,022 → 865,614 bytes (85.54% less mesh storage for this fixture).
Full-resolution reconstruction was not rerun.
The restoration benchmark is unchanged; its tests require the isolated runtime
because production `backend/.venv` does not include `lpips`.

## Licensing

Core extraction uses the backend's existing gsplat/Open3D stack. The optional
SeeSR checkout includes its own Apache-2.0 license. OSEDiff, diffusion bases,
RAM and downloaded checkpoints retain their respective upstream terms; they
are not required for production meshing.
