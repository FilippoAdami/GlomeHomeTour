# 2DGS on ARCore captures (AMD / ROCm)

2D Gaussian Splatting reconstruction from phone captures, on a Radeon RX 9060 XT
(16 GB, ROCm 7.0). Feed it a folder of video frames plus the `transforms.json`
that the ARCore capture app wrote, and it produces a COLMAP sparse model, a
trained 2DGS scene and a mesh.

The pose pipeline that used to live in a separate `COLMAP_adjustment/` sandbox is
now part of `2d-gaussian-splatting/`. There is one pose pipeline, not two.

```
2dgs_project/
  2d-gaussian-splatting/     the code: pipeline, training, diagnostics
  room/  room_test_a|b|c/    captures (images/ + transforms.json)
  .venv/                     the one virtualenv everything runs in
  requirements.txt
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # torch comes from the ROCm index, see below
pip install 2d-gaussian-splatting/submodules/diff-surfel-rasterization
pip install 2d-gaussian-splatting/submodules/simple-knn
```

`requirements.txt` pins `torch==2.10.0+rocm7.0` / `torchvision==0.25.0+rocm7.0`
against `https://download.pytorch.org/whl/rocm7.0`. Installing torch from plain
PyPI gets you the CUDA build, which will not run here. COLMAP itself is the
system binary (4.3.0.dev0, HIP build) and is not a pip dependency.

## Pipeline

### 1. Poses and sparse model

```bash
cd 2d-gaussian-splatting
python3 convert_transforms_to_colmap.py -s ../room --refine_poses \
    --diagnostics ../room/diagnostics --export_blender_ply
```

The ARCore poses are already metric and gravity-aligned, so COLMAP is used to
*triangulate against* them rather than to re-estimate them from scratch:
`feature_extractor` → `sequential_matcher` → `point_triangulator` with the poses
held fixed, then optional refinement.

Useful flags:

| Flag | Default | What it does |
|---|---|---|
| `--refine_poses` | off | Refine poses after triangulation. Off is safe: `point_triangulator` only keeps points consistent with the given poses. |
| `--ba_mode pose_prior` | `pose_prior` | `pose_prior_mapper` with the ARCore camera centres as soft priors. `unconstrained` is the old plain `bundle_adjuster`; `none` skips it. |
| `--prior_sigma_m 0.05` | 5 cm | 1σ of the positional prior. |
| `--zoom_tol 0.01` | 1 % | Focal-length tolerance for grouping frames into zoom states. |
| `--diagnostics DIR` | off | Write the reports described below. |
| `--fix_rotation_drift` | off | Re-triangulate with badly drifted frames held at their ARCore rotation. |
| `--keep_database` | off | Keep `colmap_database.db` so the Sampson check can be re-run. |
| `--no_keyframes` | on | Skip the keyframe export described below. |
| `--max_sampson_px 2.0` | 2 px | Also drop keyframes whose pose the raw matches do not support. `0` disables. |

**Keyframes and resolutions.** A capture goes in as one `images/` folder plus
one `transforms.json`, and comes out laid out the way the splatting tooling
expects. The keyframes are the frames COLMAP actually registered in `sparse/0/` -
the ones that survived matching and triangulation, and the only ones a later
calibration pass would use. Frames it dropped carry no tracks, so they add
training time and no geometry. Afterwards:

```
images/                 keyframes, full resolution (hardlinks, so free)
images_2/ _4/ _8/       the same keyframes at 1/2, 1/4, 1/8
images_all/             every original frame, untouched
sparse/0/               triangulated model
colmap_database.db      kept, so the Sampson check can be re-run
diagnostics/            reports
transforms_keyframes.json
```

`cameras.txt` keeps its full-resolution intrinsics. The loader derives FovX/FovY
from focal/width - angles, so resolution-independent - and sizes each camera from
the image it actually loads, which is why `--images images_4` needs no change to
the sparse model. Nothing is deleted: the export renames `images/` to
`images_all/` and rebuilds `images/` from it, so a re-run starts from the full
set. Pass `--no_keyframes` to leave the capture alone.

Two different filters run here, and the distinction matters. Registering in
`sparse/0/` only means COLMAP *could* use the frame; a frame next to a tracking
break registers happily with a pose its own matches do not support.
`--max_sampson_px` catches those by scoring each frame's raw verified matches
against the F built from its refined pose - nothing from BA's own output, so a
self-consistent-but-wrong solution still fails it. Pose drift against the capture's
VIO does **not** substitute: drift measures disagreement, not error, and on the
bedroom capture the high-drift frames scored no worse than the stable ones
(0.446 px vs 0.476 px median). If the filter wants more than 20 % of the frames it
refuses and reports instead, on the grounds that the model, not the frames, is
then the thing at fault.

**Zoom states.** A capture where the user pinch-zoomed has more than one focal
length, and one shared camera would be wrong for all of them. Contiguous runs of
frames with the same focal length (±1 %) each become one COLMAP camera; the
frames are staged into `images_zoomgroups/cam_N/` and extracted with
`--ImageReader.single_camera_per_folder`. A fixed-zoom capture produces exactly
one group and behaves as before. If groups were staged, train with
`--images images_zoomgroups`.

**Pose priors are soft and position-only.** `pose_prior_mapper` adds the ARCore
camera centre to the BA cost as a Gaussian term; 5 cm is a 1σ expectation, not a
bound. COLMAP has no rotation prior at all, so rotation is not constrained during
BA — it is checked afterwards by the diagnostics and optionally corrected with
`--fix_rotation_drift`. Intrinsics are frozen (`ba_refine_focal_length 0`,
`ba_refine_principal_point 0`): the phone's are trustworthy, and letting BA move
them is how the old pipeline invented distortion out of nothing.

### 2. Train

```bash
python3 train.py -s ../room -m output/room -r 2
python3 render.py -m output/room --skip_train      # images + mesh
python3 metrics.py -m output/room                  # PSNR / SSIM / LPIPS
```

Two additions, **both off by default** so the baseline is unchanged:

```bash
# PGSR-style multi-view planar consistency
python3 train.py -s ../room -m output/room --lambda_multiview 0.05 --mv_from_iter 7000

# TrackGS-style pose refinement during training (highest-risk piece)
python3 train.py -s ../room -m output/room --refine_poses_during_training \
    --pose_lr 1e-4 --lambda_track 0.1
```

`lambda_multiview` treats each pixel's rendered depth+normal as a local plane and
compares the current view against 2 neighbouring frames through the
plane-induced homography `H = K_n (R − t nᵀ/d) K_c⁻¹`. It samples pixels and
gathers from the neighbour's *ground-truth* image, so it needs no second render
pass. Neighbours are picked by capture order (`sorted` by image name) — not by
list position, because `Scene` shuffles the camera list, and not by `uid`, which
is the shared intrinsics id on the COLMAP path.

`--refine_poses_during_training` gives each camera a learnable zero-initialised
SE(3) delta on top of its COLMAP pose, at a much lower LR than the Gaussians,
regularised by reprojection of COLMAP's fixed 3D tracks. Without that track term
the poses drift to whatever flatters the photometric loss.

### 3. Diagnostics

`--diagnostics DIR` writes `metrics_summary.json`, `per_frame_deltas.csv`,
`drift_profile_2d.png`, `trajectory_comparison_3d.png`, `reprojection_sample.jpg`
and the JSON reports. Three checks, in increasing order of how little they trust
COLMAP:

1. **Drift** — refined poses vs. the ARCore initialisation. The only place
   rotation drift is caught, since nothing constrains it during BA.
2. **Multi-view reprojection** — triangulated tracks projected into every
   observing view, under init and refined poses. The direct "did refinement
   help" number.
3. **Sampson epipolar error** — computed from the raw verified matches in
   `database.db`, using nothing from COLMAP's own optimisation output. This is
   the only check that catches a bundle adjustment which converged to a wrong but
   internally self-consistent solution.

Check 3 prints a **control**: the same correspondences scored with the
fundamental matrix COLMAP's matcher stored for that pair. If the control is
below 1 px and the pose-derived error is not, the correspondences are fine and
the poses are the problem — that distinction is worth having before you go
hunting for a bug in the check.

**Read the mean track length before trusting any reprojection number.** On the
archived pre-merge baseline (`2d-gaussian-splatting/docs/baseline_diagnostics/`)
the old unconstrained BA reported a beautiful 0.29 px reprojection error while
simultaneously failing the epipolar check at 16.7 px median (control: 0.91 px).
Both were true: mean track length was 2.01, and a 2-view track is exactly
determined, so it has ~zero residual under *any* pose pair. That model had also
drifted 4.04° mean / 44.7° max in rotation and 0.50 m mean / 7.62 m max in
translation from the ARCore trajectory, and had let its focal length wander
1406→1477 px. This is what the pose-prior mode, the frozen intrinsics and check 3
exist to prevent.

## Tests

```bash
python3 test_pose_pipeline.py
```

Covers zoom grouping (including a zoom that returns, and the single-camera case),
the pose-prior database blob layout, `se3_exp` at and away from zero, neighbour
selection from a shuffled camera list, and the homography warp (an identical
neighbour must produce zero loss — that assertion is what caught a real shape bug
during development).

## Limits

- Phase 2 and Phase 3 are wired, unit-tested and default-off, but **not yet
  benchmarked on a full training run**; no PSNR/SSIM/LPIPS comparison has been
  made. Turn them on expecting to tune `lambda_multiview` and `pose_lr`.
- The multi-view loss compares single pixels, not patches. If it proves too noisy
  to converge, patch/census windows are the upgrade path (noted in the code).
- `--fix_rotation_drift` deliberately refuses to act when many frames are
  flagged: that means the ARCore trajectory itself is bad over that segment, and
  pinning rotations to it would hide the problem rather than fix it.
- 2DGS only accepts PINHOLE/SIMPLE_PINHOLE cameras, so the converter errors out
  on a `transforms.json` that declares non-zero distortion instead of silently
  dropping it. Undistort first.

SAM2-based semantic/segmentation tooling and the 4-stage VRAM-optimised training
schedule are documented separately in `2d-gaussian-splatting/README.md`.
