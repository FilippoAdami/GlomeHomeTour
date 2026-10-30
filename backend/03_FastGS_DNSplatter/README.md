# Stage 03 — Gaussian splat training

Stage 03 converts each refined scan into a renderable 3D Gaussian model. It
combines COLMAP cameras and RGB images from stage 01 with the initialization
cloud, cleaned depth maps, and normal maps from stage 02. Training uses Apache
AMD gsplat, a local FastGS-style refinement strategy, and DN-Splatter-style
geometric supervision. Stage 04 consumes the resulting checkpoint for mesh
reconstruction.

The backend source directory keeps the name `03_FastGS_DNSplatter` for import
and pipeline compatibility; scan outputs go to `scan_N/gaussian_splat/`, not
to a matching stage-named output folder. The production trainer is
`gsplat_train.py`; the original Inria/FastGS trainer, renderer, and checkpoint
format are not used.

## Stage files

| Component | Responsibility | Runtime status |
| --- | --- | --- |
| [`step_train.py`](step_train.py) | Pipeline CLI: discovers scans, launches one training process per scan, records status, resumes completed scans. | Production entry point. |
| [`gsplat_train.py`](gsplat_train.py) | Input loading, Gaussian initialization, losses, rendering, optimization, final cleanup, checkpoint/PLY export. | Production trainer. |
| [`fastgs_strategy.py`](fastgs_strategy.py) | Clone/split decisions, stochastic pruning, opacity resets, and final pruning using gsplat topology operations. | Production refinement controller. |
| [`scene/colmap_loader.py`](scene/colmap_loader.py) | COLMAP text/binary readers and quaternion conversions. | Shared with stages 01/02 and tests. |
| [`scene/dataset_readers.py`](scene/dataset_readers.py) | PLY I/O and older COLMAP/Blender scene adapters. | PLY helpers remain shared; scene adapters are outside the current training path. |
| [`native_ssim/`](native_ssim/README.md) | MIT-licensed differentiable local SSIM extension, Python autograd wrapper, and numerical oracle. | Required for training, including `--eager-loss`. |
| [`gsplat_gfx1200.patch`](gsplat_gfx1200.patch) | ROCm/gfx1200 rasterizer changes, checkpointed backward, compact footprints, and contribution counting. | Applied to the pinned external gsplat source at installation. |
| `install_gsplat_gfx1200.sh`, `install_fused_ssim.sh` | Build the pinned gsplat release and local SSIM extension into the selected Python environment. | Setup tools; not invoked by training. |
| `check_*_gpu.py`, `check_fastgs_metrics.py` | Independent compositing, gradient, footprint, and contribution-count checks. | Developer verification. |
| [`ablation_sor_floaters.py`](ablation_sor_floaters.py) | Evaluate a grid of final cluster-removal settings on an existing checkpoint. | Optional analysis; does not retrain or replace the checkpoint. |
| `experiments/` | Historical experiment scripts, results, and checkpoints retained for reference. | Not invoked by the production pipeline; some scripts target an older trainer/checkpoint format. |
| `amd-port-progress.md`, `ROCm_perfromance.txt` | Porting notes and historical performance observations. | Reference material; not runtime inputs. |
| `environment.yml` | Supplemental Python dependency list. | Setup aid; does not install ROCm/PyTorch or the native extensions. |

## Pipeline flow

```text
Stage 01: refined COLMAP cameras, registered images, sparse point cloud
Stage 02: depth/normal maps and oriented initialization cloud
                         │
                         ▼
step_train.py: discover scans, validate completion, launch one trainer per scan
                         │
                         ▼
gsplat_train.py: validate inputs → initialize Gaussians → optimize → clean → export
                         │
                         ▼
scan_N/gaussian_splat/{checkpoint.pt,gaussian_splat.ply}
                         │
                         ▼
Stage 04: consume one scan's Gaussian model for mesh reconstruction
```

Each scan is trained independently. Upstream images, COLMAP files, depth maps,
normal maps, and initialization clouds are read-only inputs to this stage.
After training, Phase 04 aligns COLMAP and optional depth poses in place. If Stage
03 is rerun afterward, it restores source-frame camera views from the transform
recorded in `mesh.json` before training against the original initialization cloud.

## Input contract

Stage 03 does not replace the sparse PLY, change camera poses, rename images, or
rescale geometry. The later Phase 04 pose update is reversible for Stage 03
retraining because its transform is recorded in `mesh.json`.

The wrapper accepts either a workspace containing `scan_N/` directories, one
scan directory, or a standalone COLMAP scene. For multi-scan workspaces it
reuses stage 01's `discover_scenes()` contract: scans are sorted numerically,
each needs `ingestion_results.json`, and mixed root/scan manifests or missing
scan manifests are rejected. Each scan is trained independently; scans are
never combined into a shared coordinate system here.

```text
current_scene/
├── certificate.json
├── pipeline_stats.json              # aggregate pipeline status
├── scan_0/
│   ├── ingestion_results.json
│   ├── images/<registered image name>
│   ├── sparse/0/
│   │   ├── cameras.txt
│   │   ├── images.txt
│   │   └── points3D.ply
│   ├── depth/
│   │   ├── points3D_depth.ply
│   │   ├── depth_maps/<image stem>.npy
│   │   ├── normal_maps/<image stem>.npy
│   │   └── poses_da3.npz
│   ├── gaussian_splat/
│   │   ├── checkpoint.pt           # final stage 03 checkpoint
│   │   └── gaussian_splat.ply      # final Gaussian model
│   ├── train_log.txt
│   ├── train_stats.json
│   └── pipeline_stats.json
└── scan_1/                         # same independent layout
```

| Input relative to one scan | Type / convention | Processing |
| --- | --- | --- |
| `sparse/0/cameras.txt` | COLMAP `SIMPLE_PINHOLE`/`PINHOLE`, or zero-distortion `SIMPLE_RADIAL`/`RADIAL`/`OPENCV`/`FULL_OPENCV`. | Builds a float32 `K` for each camera. Nonzero distortion is explicitly rejected; it is never silently ignored. |
| `sparse/0/images.txt` | COLMAP scalar-first quaternion and translation, world-to-camera. | Defines the registered training views and their order. Observation lines are skipped. Image names containing spaces are preserved. |
| `images/<name>` | RGB files named by the COLMAP model. | Decode to RGB, resize to training dimensions, cache CPU uint8 pixels, convert to GPU float32 `[0,1]`. |
| `depth/points3D_depth.ply` | Stage 02 oriented surfels in the COLMAP world frame/units. | Default seed: reads xyz and RGB, optionally samples at most 300,000 points. Other surfel fields are currently ignored. At least four points are required. |
| `sparse/0/points3D.ply` | Stage 01 sparse xyz/RGB cloud. | Used only with `--no-depth-cloud`. It is never overwritten or backed up by this phase. |
| `depth/depth_maps/<stem>.npy` | Floating `(H,W)` camera-Z depth; normally float16. | Load when depth supervision is enabled, bilinearly resize if needed, widen to float32 on device, mask nonfinite/nonpositive targets in the loss. |
| `depth/normal_maps/<stem>.npy` | Floating `(H,W,3)` camera normals; normally float16. | Load for normal supervision; resize if needed and normalize valid vectors inside the loss. Default StableNormal values are preserved. |
| `depth/poses_da3.npz` | Stage 02 ordered `w2c`, `K`, and image names. | Upstream provenance; not a second source of training cameras. Stage 03 reads the authoritative refined COLMAP model. |
| `transforms.json`, `metric_scale.json`, `depth/depth_alignment.json`, confidence/extent reports | Other upstream metadata. | Not required by this trainer. Reliability metadata is not currently used to mask supervised frames. |

Map lookup uses the image **stem**, not list position. Supervised registered views must
have unique stems, including across different extensions/subdirectories.
Every requested map must exist; a missing map raises an error before GPU
training instead of silently disabling that view's supervision. Loaded maps
must have the documented floating-point shape. Missing depth initialization
also fails explicitly; choose the sparse cloud deliberately with
`--no-depth-cloud`. The obsolete `02_depth_estimation/depth/` location is not
used.

### Coordinates, scale, and normal models

- Camera transforms remain COLMAP/OpenCV world-to-camera: rightward X,
  downward Y, forward Z. The trainer does not apply ARCore/OpenGL pose flips.
- Direct-COLMAP inputs can retain radial/OpenCV model names. Those are
  accepted when their distortion coefficients are exactly zero. Nonzero
  distortion remains unsupported: the current trainer and downstream mesh
  renderer use pinhole projection. Supporting it needs a separate stage-local
  image/prior remapping design; locked upstream files are not rewritten.
- Depths and point positions inherit the upstream world units. They are
  metres only when stage 01 has applied a valid metric calibration. No scale
  factor is estimated or applied again here.
- `--resolution R` uses integer dimensions `W//R, H//R`; focal lengths and
  principal points scale by the actual width/height ratios, including when
  dimensions are not divisible by `R`.
- The rendered-depth normal formula has a positive-Z orientation. Stage 02's
  default StableNormal maps use the established compatible target convention.
  For stage 02's optional `--normal-model depth_gradient`, pass the same option
  to this phase. Its OpenGL, camera-facing normals convert in memory as
  `(-nx, ny, nz)` to the trainer's oriented convention. Source arrays remain
  unchanged. The producer is not encoded in the NPY files and cannot be
  reliably inferred automatically.
- Cleanup distance `0.10` means **0.10 world units**. It is 10 cm only for a
  metric model; arbitrary-scale inputs need a deliberately chosen threshold.

## Processing

1. **Resolve and validate inputs.** Discover scan-local artifacts, read the
   registered cameras, require the selected seed and requested supervision
   maps, and reject ambiguous map names.
2. **Initialize Gaussians.** Read seed xyz/RGB. Initial scales are the RMS
   distance to the three nearest other seed points; initial rotations are
   identity quaternions, opacity is 0.1, and RGB becomes the SH DC term.
   Higher SH coefficients start at zero. Surfel normals, tangent frames, and
   stored scales are not used for initialization.
3. **Train on shuffled camera cycles.** Every registered view is visited once
   before refilling the camera pool. Production training uses all views;
   there is no automatic held-out evaluation set. SH degree grows by one
   every 1,000 steps up to degree 3.
4. **Render and supervise.** The renderer returns RGB and, when geometric
   supervision is active, accumulated depth and alpha. Expected camera-Z
   depth is accumulated depth divided by alpha. The loss is:

   ```text
   0.8 × RGB L1 + 0.2 × (1 − SSIM)
     + 0.15 × depth loss              # when enabled
     + 0.075 × normal cosine loss     # when enabled
   ```

   Depth defaults to Pearson correlation loss; alternatives are `l1`,
   `log_l1`, and fitted `scale_shift`. Fewer than 16 valid depth pixels give
   zero depth loss. Geometry requires finite positive rendered depth and
   alpha greater than 0.05. Normal supervision additionally requires valid
   neighboring depths and finite target vectors of norm greater than 0.5.
   Geometry supervision lasts through the final iteration unless
   `--geometry-until` overrides it. The geometry loss is compiled by default;
   `--eager-loss` disables this compilation.
5. **Refine topology.** From steps 600 through 14,900, every 100 steps samples
   up to ten views to measure flagged-pixel support and photometric pruning
   scores. Supported small Gaussians clone on ordinary projected gradients;
   larger ones split on absolute gradients. Weighted stochastic pruning,
   opacity caps, and scheduled resets regulate the model. Final pruning
   occurs at 18k, 21k, 24k, and 27k if the run reaches those steps. Short runs
   do not execute milestones they never reach.
6. **Update parameters.** Adam uses separate parameter groups. Positions use
   a camera-extent-scaled exponential learning-rate schedule. Through 15k,
   non-SH-rest groups step every iteration and SH-rest every 16 steps; after
   15k all groups step every 32 iterations, then every 64 after 20k.
   Gradients accumulate between those optimizer updates.
7. **Remove detached clusters and export.** By default, Open3D DBSCAN groups
   final centers at radius 0.10 with ten-point core support. Clusters of at
   least 500 Gaussians are protected; if none qualify, the largest cluster is
   protected. Candidates at least 0.10 away from protected points are removed
   using a nearest-neighbor query. All parameters use the same keep mask.
   The historical “SOR” option names refer to this DBSCAN/distance operation,
   not conventional mean-neighbor-distance SOR. If no DBSCAN cluster exists,
   the model is retained unchanged.

The renderer uses dense, one-camera rasterization, 16-pixel tiles and compact
footprints of 0.5 by default. Compact bounds intentionally omit peripheral
contributions; `--footprint-multiplier 1` uses full support. The gfx1200
backward stores forward checkpoints to reduce repeated compositing work;
storage is bounded to 1 GiB and one third of available VRAM, with a standard
kernel fallback. These mechanisms and the numerical loss definitions are
unchanged by the input-contract correction.

## Outputs and downstream interface

Each scan writes:

The `gaussian_splat/` directory contains exactly the two final model artifacts
below; scan logs, step stats, and pipeline stats remain beside that directory.

| Output | Contents |
| --- | --- |
| `gaussian_splat/checkpoint.pt` | Atomic tensor checkpoint containing the final CPU splat tensors, SH degree, and training resolution. |
| `gaussian_splat/gaussian_splat.ply` | Final binary Gaussian PLY with positions, zero placeholder normals, SH coefficients, opacity logits, log-scales, and scalar-first rotations. This is not an RGB surfel cloud or mesh. |
| `train_log.txt`, `train_stats.json` | Per-scan wrapper commands, timers, output paths/sizes, and failures. Iteration stdout goes to the console rather than being captured into the wrapper log. |
| `scan_N/pipeline_stats.json` and workspace `pipeline_stats.json` | Per-scan metrics and aggregate pipeline completion state, respectively. The workspace root has no separate train log or stats file. |

For `N` Gaussians, checkpoint `splats` contains `means: (N,3)`,
`scales: (N,3)` in log space, `quats: (N,4)`, `opacities: (N,)` in logit space,
`sh0: (N,1,3)`, and `shN: (N,15,3)`. Optimizer state is not saved; these are
render/export checkpoints, not training-resume checkpoints. PLY is written
before the atomic checkpoint; the two files are not a joint transaction.

The wrapper requires the final checkpoint and Gaussian PLY. Training iteration
count controls optimization only; output paths never include an iteration number.
Completed scans are skipped unless `--force` is used. After one scan fails,
other scans are attempted, successful scan status is retained, and aggregate
failure prevents reporting the whole workspace complete. Retry resumes only
unfinished scans. Resume checks use completion state and output existence,
not an input/settings fingerprint; force a retrain after changing inputs or
hyperparameters.

Stage 04 still accepts **one scan workspace per invocation**. Pass
`current_scene/scan_N`, rather than the multi-scan parent, to its existing CLI.
This review changes stage 03 only; it does not add multi-scan orchestration to
later stages. Original Inria/FastGS checkpoints are incompatible.

## Running

Use the existing ROCm-enabled backend environment. The recorded target is
ROCm 7.1/gfx1200 with a Radeon RX 9060 XT and PyTorch ROCm. The installer pins
AMD gsplat tag `1.5.3b2` to commit
`b01acd43e3c7fa942f95fda0974e9125e4de7395`, verifies the commit, applies the
local patch, then builds gsplat and fused SSIM. It needs network access, a
working ROCm compiler, and the matching PyTorch environment. `environment.yml`
is a supplemental dependency list, not a complete GPU installation recipe.

From the repository root:

```bash
# Install native dependencies into the existing backend environment
backend/03_FastGS_DNSplatter/install_gsplat_gfx1200.sh backend/.venv/bin/python

# Train all scans, or pass .../scan_0 to train only that scan
.venv/bin/python 03_FastGS_DNSplatter/step_train.py --workspace current_scene

# Direct trainer: one scan; wrapper depth/normal defaults are explicit here
backend/.venv/bin/python backend/03_FastGS_DNSplatter/gsplat_train.py \
  -s backend/current_scene/scan_0 -m /tmp/glome-stage03-model \
  --iterations 22000 --depth-supervision --normal-supervision

# Validate cameras, seed loading, and requested map existence without a GPU run
backend/.venv/bin/python backend/03_FastGS_DNSplatter/gsplat_train.py \
  -s backend/current_scene/scan_0 -m /tmp/glome-stage03-check \
  --self-check --depth-supervision --normal-supervision

# Mesh the scan-local model with the existing downstream interface
backend/.venv/bin/python backend/04_3DGS_to_mesh/step_mesh.py \
  --workspace backend/current_scene/scan_0
```

The self-check loads seed xyz/RGB and checks map presence; shape/type checks
occur when maps are loaded. It does not certify the accuracy of the priors.

| Option | Wrapper default / meaning |
| --- | --- |
| `--iterations` | 22,000 optimization steps; controls training duration only. |
| `--no-depth-cloud` | Opt out of the depth seed and use the untouched sparse PLY. |
| `--no-depth-supervision`, `--no-normal-supervision` | Disable the corresponding loss and map requirement. The direct trainer instead enables them with positive flags. |
| `--lambda-depth`, `--lambda-normal`, `--depth-loss` | 0.15, 0.075, `pearson`. |
| `--normal-model` | `stablenormal`; use `depth_gradient` to match the alternate upstream producer. |
| `--geometry-until` | Final iteration. |
| `--resolution`, `--max-points` | Full resolution (`1`), at most 300,000 seed points (`0` means all). |
| `--target-cache-mb` | Shared CPU depth/normal LRU budget, 8,192 MiB; `0` disables caching. RGB caching is separate and unbounded. |
| `--footprint-multiplier` | 0.5, or 1 for full support. |
| `--no-sor-floater-removal` | Disable the final cluster/distance cleanup. |
| `--sor-min-cluster-size`, `--sor-min-distance` | 500 points and 0.10 world units. |
| `--eager-loss`, `--seed` | Disable loss compilation for debugging; random seed defaults to 42. |
| `--force` | Wrapper only: rerun completed scans. It does not delete the output directory first; a successful retrain replaces the final model. |

Trainer options not parsed by the wrapper are forwarded to each scan's
training process. `-s/--source/--source-path` and
`-m/--model/--model-path` select one direct-training input/output pair. Use the
wrapper's `--workspace` for multi-scan runs.

## Function reference

The following inventory covers the maintained stage source. Helpers are
listed explicitly so the data flow can be followed without treating every
helper as a separate pipeline stage.

### Pipeline and trainer

| Function | Inputs → outputs; processing |
| --- | --- |
| `step_train.train(workspace, ctx, ...)` | One scan and training settings → validates subprocess exit and exact checkpoint/PLY existence; records output paths/sizes. Passes cloud choice without copying upstream data. |
| `step_train.main(argv)` | CLI → integer success code or exception; discovers scans, skips completed outputs, runs each independently and records aggregate failure/success. |
| `gsplat_train._rotation(q)` | Scalar-first quaternion → normalized float32 world-to-camera rotation. |
| `load_cameras(workspace, resolution)` | COLMAP text + image paths → camera dictionaries containing `image_path`, `width`, `height`, `K: (3,3)`, `viewmat: (4,4)`; checks supported models, file presence and resolution. |
| `_training_inputs(workspace, cameras, use_depth_cloud, depth_supervision, normal_supervision)` | Scan and requested modes → selected seed PLY path; requires that seed, unambiguous supervised image stems, and all requested maps without creating or modifying them. |
| `_load_points(path, max_points, seed)` | PLY → float32 xyz/RGB; defaults absent colors to gray, optionally samples deterministically, rejects absent/empty positions. |
| `_scene_scale(cameras)` | Camera transforms → 1.1 times maximum distance of camera centers from their mean, floored at 0.001. This sets optimizer/refinement scale, not metric calibration. |
| `_make_splats(xyz, rgb, device)` | Seed arrays → trainable `ParameterDict`; nearest-neighbor scales, identity quaternions, opacity logits and SH color encoding. |
| `_image_tensor(camera, device)` | Camera/image → float32 RGB tensor; lazily decodes/resizes and caches CPU uint8 data, pins it for GPU transfer when possible. |
| `_pin_for_transfer(value)` | CPU tensor → pinned tensor, or original tensor if pinning raises `RuntimeError`. |
| `_camera_tensor(camera, name, device)` | Camera array name and device → cached device tensor with a leading camera dimension. |
| `_map_tensor(root, camera, channels, device, cache, normal_model)` | NPY map → resized float32 device tensor or `None` if absent; validates shape/type, converts alternate normal convention, maintains byte-limited shared CPU LRU. Unresized float16 maps stay float16 in the CPU cache. Production preflight requires enabled maps. |
| `_depth_loss(pred, target, mode, render_valid)` | Rendered/target depth and optional mask → scalar Pearson/L1/log-L1/scale-shift loss over valid pixels, or zero with insufficient support. |
| `_normals_from_depth(depth, K)` | Perspective camera-Z depth and intrinsics → normalized interior `(H−2,W−2,3)` normals from central differences, with positive-Z orientation. |
| `_training_loss(colors, alphas, target, K, depth_target, normal_target, weights, depth_loss, ssim_score)` | RGB/depth rendering, priors and weights → scalar photometric + masked depth/normal objective. Missing optional targets omit their term. |
| `remove_cluster_floaters(splats, min_cluster_size, min_distance, min_samples)` | Trained parameters → filtered aligned `ParameterDict` and count/timing report; CPU DBSCAN plus distance to protected clusters, with all-noise/largest-cluster safeguards. |
| `_export_ply(path, splats)` | Gaussian parameters → binary PLY; SH channels are flattened in the standard channel-major layout. |
| `_atomic_checkpoint(path, payload)` | Checkpoint dictionary → temporary torch serialization followed by atomic replacement; cleans temporary file on failure. |
| `_render(splats, camera, degree, geometry, absgrad, footprint_multiplier)` | Parameters/camera → RGB or RGB+accumulated-depth, alpha, and gsplat metadata; activates scales, opacity and quaternions. |
| `_fastgs_metrics(splats, cameras, degree, footprint_multiplier)` | Model + up to ten randomly sampled views → per-Gaussian mean flagged contribution counts and normalized photometric pruning scores, without gradients. |
| `_next_camera(cameras, pending)` | Full list and mutable remaining pool → one randomly chosen camera, refilling only after every view was visited. |
| `_optimizer_due(step, name)` | Iteration/parameter group → boolean optimizer cadence decision. |
| `train(args)` | Parsed single-scan settings → final PLY path; validates inputs, constructs model/optimizers/strategy, trains, optionally cleans clusters, exports. |
| `_write_model(model_dir, splats, degree, resolution)` | Final parameters → `checkpoint.pt` and `gaussian_splat.ply`; returns the PLY path. |
| `_self_check(args)` | Single-scan CLI settings → assertions on cameras/seed and a printed input summary; checks all requested map paths without GPU training. |
| `main(argv)` | Trainer CLI → input self-check or training/export, then integer success code. |

### Refinement strategy

Every method below belongs to `FastGSStrategy` in `fastgs_strategy.py`.
Topology operations update parameter, optimizer, and strategy arrays together.

| Method | Inputs → outputs; processing |
| --- | --- |
| `initialize_state(scene_scale)` | Scene extent → state dictionary with deferred gradient/count/radius/score/origin buffers. |
| `check_sanity(params, optimizers)` | Parameters and optimizers → assertions on matching groups and required fields. |
| `step_pre_backward(..., step, info)` | Rasterizer metadata → retained projected-mean gradient before refinement stops. |
| `step_post_backward(..., step, info, packed)` | Gradients/metrics → state and topology updates at scheduled refinement, reset, or final-prune events; rejects packed inputs. |
| `_update_state(params, state, info, require_metrics)` | One camera's gradient, absolute gradient, visibility and radius → accumulated state with resolution-scaled projected gradients. |
| `_update_metrics(params, state, info, required_keys)` | Optional/fresh contribution and pruning scores → validated state arrays, one finite score per Gaussian. |
| `_growth_masks(params, state)` | Support, scale, gradient averages → clone and split boolean masks. |
| `_zero_new_scores(state, start)` | Start of newly created rows → zero importance/pruning and origin `−1` for those rows. |
| `_refine(params, optimizers, state, step)` | Current model/state → duplicate/split, reset radius history, stochastic-prune, cap opacity and clear accumulated gradients/counts. |
| `_stochastic_mask(marked, pruning_score, origin)` | Candidate mask/scores/original IDs → sampled removal mask; samples original IDs before intersecting marked candidates, protecting new children. |
| `_stochastic_prune(params, optimizers, state, step)` | Opacity/size gates + scores → removes a weighted sampled subset of marked original Gaussians. |
| `_final_prune(params, optimizers, state)` | Current model/pruning scores → removes opacity below 0.1 or pruning score above 0.9. |

### Shared COLMAP and scene helpers

These files are used by earlier stages; their behavior remains unchanged.

| Function / record | Inputs → outputs; processing |
| --- | --- |
| `colmap_loader.qvec2rotmat`, `rotmat2qvec`, `Image.qvec2rotmat` | Quaternion ↔ rotation matrix; validates shapes/nonzero quaternion and orthogonalizes input matrices by SVD. The record method converts its stored quaternion. |
| `read_next_bytes`, `_read_c_string` | Binary stream → checked unpacked values or null-terminated image name; truncated data raises an error. |
| `_data_lines` | Text path → iterator of nonempty, noncomment stripped lines. |
| `read_intrinsics_text`, `read_intrinsics_binary` | COLMAP model file → camera-ID dictionary with model, dimensions, parameters. |
| `read_extrinsics_text`, `read_extrinsics_binary` | COLMAP image file → image-ID dictionary with qvec/tvec, camera ID, name, observations and point IDs. |
| `read_points3D_text`, `read_points3D_binary` | COLMAP point file → xyz, RGB and reprojection-error arrays; binary tracks are consumed but not returned. |
| `read_colmap_bin_array` | COLMAP dense-map file → reshaped NumPy array from its width/height/channel header and Fortran-ordered data. |
| `colmap_loader._self_check` | Sample model directory → quaternion round-trip and binary-reader assertions. |
| `BasicPointCloud`, `CameraInfo`, `SceneInfo` | Named records for xyz/colors/normals, camera/FOV/image metadata, and train/test scene partitions. |
| `dataset_readers.focal2fov`, `fov2focal` | Focal length and pixel size ↔ field of view. |
| `_world_to_camera` | Stored transposed rotation and translation → 4×4 transform. |
| `_sh0_to_rgb` | SH DC array → RGB using the DC normalization constant. |
| `getNerfppNorm` | Camera records → negative center translation and 1.1× camera radius; empty input returns a default normalization. |
| `readColmapCameras` | Extrinsic/intrinsic dictionaries + image directory → opened image/camera records for pinhole models. |
| `fetchPly` | PLY with xyz/RGB/normals → `BasicPointCloud`; requires all nine fields and normalizes colors. |
| `storePly` | xyz and RGB `[0,255]` arrays → PLY with zero normals, validating `(N,3)` shapes. |
| `_read_colmap_model` | Sparse directory → binary camera/image dictionaries, falling back to text on read errors. |
| `readColmapSceneInfo` | Scene path and holdout settings → `SceneInfo`; optional every-eighth split, and creates a PLY if absent. This legacy adapter can write upstream data and is not called by the production trainer. |
| `readCamerasFromTransforms` | Blender/NeRF JSON, RGBA images and background choice → composited RGB cameras with OpenGL-to-COLMAP axis conversion. |
| `readNerfSyntheticInfo` | Synthetic scene/settings → train/test `SceneInfo`; optionally generates a random seed PLY. |
| `sceneLoadTypeCallbacks` | Static mapping from `Colmap`/`Blender` to those scene adapters. |

### Native extensions, installers, and checks

| Function / component | Inputs → outputs; processing |
| --- | --- |
| `native_ssim.fused_ssim` | Same-shaped contiguous float32 GPU NCHW images → mean local 11×11 Gaussian SSIM, zero-padded (`same`) or border-cropped (`valid`); target gradients are unsupported. |
| `_FusedSSIMMap.forward`, `.backward` | Autograd tensors → SSIM map and saved derivatives; backward restores cropped borders and returns only prediction-image gradients. |
| `ext.cpp: check_image`, `check_pair` | Tensor arguments → device/type/layout/shape/fixed-target checks. |
| `fusedssim_checked`, `fusedssim_backward_checked` | Validated tensors → native calls under a device guard; exported through Python bindings. |
| `ssim.cu: get_pix_value`, `load_into_shared`, `multiply_shared_mem`, `do_sq`, `flush_conv_scratch` | Image/tile values → zero-padded reads, shared tile loads/products, scalar squares and cleared convolution scratch. |
| `do_separable_conv_x`, `do_separable_conv_y` | Shared tiles → separable 11-tap Gaussian-filtered moments/derivatives. |
| `fusedssimCUDA`, `fusedssim_backwardCUDA` | Device image/derivative arrays → local SSIM/derivative maps or image gradients. |
| `fusedssim`, `fusedssim_backward` in native code | Torch tensors → allocated outputs and kernel launches on the current GPU stream. `ssim.hip` and HIP compatibility headers, when generated by the build, are build products. |
| `native_ssim/setup.py` | Installed Torch/ROCm build configuration → `fused_ssim_cuda` extension. |
| `native_ssim/check.py: reference` | Prediction/target tensors → grouped-convolution SSIM oracle; module assertions check values, gradients, borders and nondefault streams. |
| `install_gsplat_gfx1200.sh`, `install_fused_ssim.sh` | Python executable → installed native packages, using a temporary pinned-source checkout and project-local SSIM sources. |
| `gsplat_gfx1200.patch` | Pinned upstream source → changes `_wrapper.py`, `rendering.py`, native forward/backward/2DGS kernels, bindings, compatibility/operation headers and `setup.py`; adds the checkpoint kernel source. Checkpoint forward saves prefix/transmittance states; backward buckets accumulate Gaussian gradients; VCP counting reports flagged-pixel contributions. |
| `check_gsplat_gpu.reference`, `main` | Small synthetic compositing cases → differentiable CPU oracle and assertions against GPU RGB/alpha/gradients across packed/dense layouts, channels and tiles. |
| `check_checkpoint_gpu.check_occluded_long_list`, `main` | Occluded/synthetic intersection lists → checkpoint-compaction, forward/gradient and boundary-case assertions. The large case allocates eight million IDs. |
| `check_compact_gpu.main` | Synthetic Gaussian footprints → tile-bound/count and invalid-multiplier assertions at full/compact support. |
| `check_fastgs_metrics.oracle`, `tiled_oracle` | Scalar/tiled means, conics, opacity, pixel flags, sorted intersections → independent contribution counts. |
| `cases`, `cpu_check`, `gpu_check`, `threshold_checks`, `saved_checks` | Synthetic/captured contribution-count cases → assertions for flags, empty lists, borders and alpha/transmittance thresholds. `saved_checks` requires hardcoded `/tmp/gsplat-perf-next/` captures; the full script is not a portable CI test. |
| `ablation_sor_floaters.run_ablation`, `main` | Existing checkpoint and cluster-size/distance grids → cleanup statistics, optional JSON/Markdown report; clones tensors per trial. Supply an explicit checkpoint: its default is a historical scene path. |

### Historical experiment inventory

`experiments/` is ignored historical material, retained for provenance. Many
scripts import removed `arguments`, `gaussian_renderer`, `scene.Scene`,
`scene.gaussian_model` or `utils` APIs and load legacy `chkpnt*.pth` files.
They need a deliberate port before use with current checkpoints.

| Scripts / functions | Inputs → outputs and intent |
| --- | --- |
| Repeated `load_img` | Image path → RGB float32 `[0,1]` arrays. |
| Repeated `calc_psnr`, `calc_ssim_cpu` | Image pair → MSE PSNR or global grayscale moment-based SSIM; the latter differs from training's local SSIM. `diag6`'s PSNR helper instead accepts MSE. |
| `create_parameter_swaps.main`, `eval_parameter_swaps` | Legacy model variants → parameter-swapped exports and image comparisons. |
| `eval_stage2a`, `eval_trajectory_cpu` | Saved renders/targets → CPU appearance metric reports. |
| `decompose_displacement_normals.main`, nested `stats` | Legacy models/normals → displacement decomposition and summary statistics. |
| `stage1_observability_analysis.main`, `diagnostic_part_ab.main` | Fixed legacy experimental cases → observability/appearance diagnostic reports. |
| `render_hybrid_difficult.render_model_subset` | Selected legacy model/cameras → difficult-view renders. |
| `diagnostics/diag1..diag6: run_diagnostic_1`, `run_diagnostic_2`, nested `make_submodel`, `run_diagnostic_3`, `run_diagnostic_sh`, `run_diagnostic_full_coverage`, `run_diagnostic_oracle` | Legacy scene/models → coverage decomposition, floater/tidiness, thickness, SH-order, full test-view coverage and SH-oracle analyses. |
| `gldn/analyze_stage1.main`, `analyze_stage2.main` | Saved experimental outputs → stage comparison reports. |
| `gldn/evaluate_comparison: colorize_depth`, `normal_to_rgb`, `make_side_by_side`, `render_difficult_suite` | Depth/normal/render arrays and cameras → colorized/comparison panels and difficult-view exports. |
| `iteration_sweep/eval_iteration_sweep: evaluate_checkpoint`, `main` | Legacy checkpoints at fixed milestones → RGB/LPIPS/depth/normal quality comparisons. |
| `cutoff_15k/eval_cutoff_ab: load_model_from_dir`, `evaluate_model_on_cameras`, `compute_pairwise_geometric_drift`, `compute_splat_drift`, `main` | Legacy checkpoints/cameras → appearance metrics and geometry-drift comparisons for supervision windows. |
| `cutoff_15k/profile_iteration_components: benchmark_condition`, `main` | Legacy renderer/loss conditions → warmed component timings. |

## Performance and quality considerations

The experiments and benchmark notes retained in this directory describe
earlier runs. Their results are specific to those scenes, checkpoints, and
software versions; they are not a performance guarantee for the current
pipeline. The opportunities below come from the current source and need a
controlled before/after comparison before claiming an improvement.

### Opportunities to measure

These are source-review candidates, not newly demonstrated speedups.

| Priority | Opportunity | Required comparison / risk |
| --- | --- | --- |
| High | Profile the ten-view refinement metrics separately from ordinary training iterations. They add multiple renders, SSIM and native contribution counts every refinement event. | Keep the same view coverage, support thresholds and final quality when changing cadence or metric collection. |
| High | Bound/prefetch RGB and target transfers for larger scans. RGB remains an unbounded CPU cache; depth/normals share an 8 GiB LRU. Target maps are not pinned like RGB. | Measure peak host RAM, cache misses and transfer/compute overlap before adding pinned staging or a loader thread. A cache smaller than a shuffled epoch's working set can repeatedly reread maps. |
| High, quality | Use stage 02 alignment reliability information when deciding which priors to supervise. Pixel validity alone does not identify an entire incorrectly aligned frame. | Compare geometry and coverage on held-out views; do not silently discard ceiling views or assume prior agreement is ground-truth accuracy. |
| Medium, quality | Compare surfel-normal/scale-aware initialization with current isotropic nearest-neighbor initialization. | Preserve the upstream cloud and validate normal conventions, small details and convergence; additional fields are currently discarded. |
| Medium | Profile final CPU DBSCAN and nearest-neighbor cleanup at the full trained Gaussian count. It synchronizes the device and copies all surviving parameters. | Separate cleanup cost from training; preserve detached real objects. The per-run cleanup report is currently printed, not saved in checkpoint metadata. |
| Medium | Profile tile enumeration, intersections/sorting and repeated activation/SH concatenation in `_render`. | Require an end-to-end benefit before adding kernel complexity. |
| Quality tradeoff | Lower resolution, fewer initial points, compact footprint, shorter geometry supervision, or different loss weights. | These change the training problem or coverage. Report both runtime and held-out image/geometry effects; they are not lossless optimizations. |

### Maintenance notes

- **Shared helpers must remain:** `scene/` has older adapter code and
  duplicates some trainer parsing, but `storePly`, `fetchPly`, and COLMAP
  readers have active upstream callers. Removing the directory would break
  earlier stages. Consolidating only the unused adapters needs a caller audit
  and a separate change.
- **Experiments are reference material:** `experiments/` is retained, but the
  production pipeline does not read it. Many scripts target the former trainer
  APIs and checkpoint format; treat their results as historical unless a
  script is ported and validated against the current model.
- **Setup is split by design:** `environment.yml` lists basic Python packages;
  `install_gsplat_gfx1200.sh` builds the pinned AMD gsplat and local SSIM
  extension. The YAML alone is not a reproducible ROCm environment.
- **No automatic checkpoint resume or cache fingerprint:** completion checks
  do not prove inputs/settings are unchanged. A future fingerprint could
  prevent stale reuse; current operation requires `--force` after changes.
- **Keep numerical checks separate from legacy analyses:** the old global
  grayscale SSIM helpers and hardcoded temporary capture paths are unsuitable
  as quality gates for the current local-SSIM trainer.

## Verification

Focused CPU contract, objective, strategy, serialization and cleanup tests:

```bash
backend/.venv/bin/pytest -q \
  backend/tests/test_fastgs_inputs.py \
  backend/tests/test_fastgs_mesh.py \
  backend/tests/test_fastgs_strategy.py \
  backend/tests/test_sor_floater_removal.py
```

The input tests cover scan-local paths, upstream byte preservation, explicit
sparse initialization, missing/malformed priors, unique map names, zero-distortion camera aliases, odd-size
intrinsic scaling, independent plane-normal conversion, and multi-scan
failure/resume behavior. Existing tests retain objective/gradient, cache,
optimizer/topology, final-output and stage 04 interface coverage.

GPU kernel or dependency changes require the independent checks in
`check_gsplat_gpu.py`, `check_checkpoint_gpu.py`, `check_compact_gpu.py`,
`check_fastgs_metrics.py`, and `native_ssim/check.py`; run GPU checks
sequentially. `check_fastgs_metrics.py` also has saved-capture checks tied to
specific `/tmp` capture paths, so the full script is not a portable CI test.
An input-only smoke run does not replace GPU checks or establish
full-resolution reconstruction quality.

## Licensing

The installed [AMD gsplat source](https://github.com/AMD-Ecosystem/gsplat) is
Apache-2.0; its license is included in `GSPLAT_APACHE_LICENSE.txt`.
`native_ssim/` has its own MIT license. Do not reintroduce the restricted
original Inria implementation or native extensions. The repository root
`LICENSE` and `backend/pyproject.toml` declare different licenses
(AGPL-3.0 and Apache-2.0); resolve the intended local-code license before
distribution. This review does not alter licensing declarations.
