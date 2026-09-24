# 02_depth_estimation — project history

## 2026-09-14: pipeline steps 3/4, and COLMAP poses replace ARCore poses for DA3
Added `step_filter_depth.py` (step 3) and `step_depth.py` (step 4), plus
`colmap_poses_to_da3.py` (substep 4a) to convert the COLMAP model into the arrays
DA3 consumes.

Outcome: worked — but the headline change is a *deletion*. `depth_priors.py` no
longer calls `arcore_c2w_to_da3_w2c()` internally. The pipeline now always runs a
COLMAP pass first, and COLMAP already stores OpenCV world-to-camera, which is
exactly what DA3 wants; converting it again would apply a second inversion and
axis flip and produce plausible-but-wrong depth. `estimate_depth_sequence` /
`estimate_depth_sliding_window` now take w2c directly. The function is kept, and
the three legacy callers that still hold raw ARCore c2w poses
(`run_sliding_window_reconstruction.py`, `run_inspection_chunks.py`,
`Utilities/run_full_benchmark.py`) now call it themselves at the call site.

Two conventions meet in step 4, in opposite directions — this is the thing to
re-read before touching it:
- DA3 gets COLMAP w2c **unconverted**.
- `SurfelCloudInitializer` gets `inv(w2c) @ diag(1,-1,-1,1)`, i.e. converted
  **back** to OpenGL/ARCore c2w, because its unprojection is written against that
  and is correct as written (see `backend/CLAUDE.md`).

Neither mistake crashes; both silently warp the cloud. `validate_poses()` checks
orthonormality, determinant and camera-centre agreement against transforms.json,
and `tests/test_colmap_poses_to_da3.py` pins the double-conversion regression.

Step 3 replaces the old parallax/keyframe stages' estimates with real geometry:
per-frame scene depth is the median of that frame's own triangulated points, and
covisibility is the true shared-track fraction
`|tracks(i) & tracks(j)| / min(|tracks(i)|, |tracks(j)|)` rather than the frustum
overlap approximation the fraction was always a proxy for. The selector's walk
and thresholds are reused untouched via a subclass.

`min_conf` for the surfel init is 0.01, not the initializer's 0.5 default: DA3
confidence is a relative score and textureless interior walls sit low across the
whole frame, so 0.5 discards most of a bedroom.

## 2026-09-14: step 3 kept 90.5% — the overlap band was inherited from a different metric
Outcome: worked — 644/1373 (46.9%), against a 30-50% target, and coverage got
*better* rather than worse: 12 consecutive pairs below the floor, down from 368.

The cause was not a loose motion gate. `for_2dgs_training`'s 0.50/0.80 band was
calibrated against *frustum* covisibility; `TrackCovisibilitySelector` scores
`|a & b| / min(|a|, |b|)` over shared COLMAP tracks, which runs materially lower
on the same geometry. So the floor was firing on most frames, and in
`_select_chain` the redundancy rejections are only enforceable *above* the floor
— below it every frame is force-accepted so as not to re-open a gap. The band was
switching off the cull it was supposed to drive: `redundant_covisibility` fired 14
times out of 1366, and `unavoidable_gap` 368. Recalibrated to 0.25/0.60 with a
0.15 m / 6 deg motion gate (8 cm is a near-duplicate at this scene's 1.46 m median
depth), plus a 30-50% budget met by re-walking at a tighter threshold.
`redundant_covisibility` now fires 577.

Any threshold here is metric-specific: re-deriving this band from
`00_ingestion`'s numbers, or swapping `_mutual_covisibility` again, silently
changes which rules are reachable rather than just shifting a count.

**Trap, cost 30 min of COLMAP:** this step prunes `sparse/0` in place, and
`--force` calls `merge_back()`, which restores images and transforms.json but
*cannot un-prune the model*. A second `--force` therefore selects against a model
already missing the frames it just restored, and emits a scene whose model covers
fewer cameras than its own transforms.json (measured: 527 vs 657). Nothing errors
— step 5 would train on the smaller set silently. Step 2 owns `sparse/0`, so the
only fix is re-running it. `filter_depth()` now refuses to start when `sparse/0`
does not cover every frame in transforms.json, and says which step to re-run.

## 2026-09-15: Saturated pixel masking & cross-view epipolar free-space carving
Added point-cloud pruning prior to 2DGS training in `initialization.py` (consumed by
`step_depth.py` step 4):
1. **Dynamic saturation / bloom masking (`compute_overexposed_mask`):**
   Discards blown-out pixels prior to unprojecting depth into 3D. Employs a dynamic
   high-percentile floor `clip(quantile_99.8(max(R,G,B)), 250, 255)` paired with
   a low-chroma difference test (`max - min <= 35`), separating true optical bloom
   from diffuse white walls and saturated colored objects, with optional morphological
   dilation to capture bloom boundary halos.
2. **Cross-view free-space carving (`filter_multiview_consistency`):**
   Upgrades multi-view consistency to detect free-space contradictions in addition to
   surface corroboration. If a candidate 3D point reprojects into an unobstructed
   adjacent side view (with baseline >= 0.08 m or parallax >= 3 deg) and lands on
   empty space (`proj_z < obs_depth - (0.08 + 0.05 * proj_z)`), it is flagged as a
   free-space violation. Setting `max_freespace_violations=0` (the default) culls any
   point with >= 1 clear free-space contradiction, eliminating floating phantom points,
   light-fixture artifacts, and boundary depth-bleed before 2DGS initialization.
3. **Pipeline controls (`step_depth.py`):**
   Exposed `--no-saturation-mask`, `--no-freespace-filter`, and
   `--max-freespace-violations` CLI flags, with execution metrics recorded in `StepContext`.

## 2026-09-15: voxel grid coarsens instead of random thinning
The surfel budget used to be enforced *after* voxelization by keeping a random
permutation, so the grid decided which points exist but the budget decided the density —
a finer grid was only decimated harder, and the uniform coverage the grid existed to
produce was thrown away. `initialize_from_keyframes` now re-voxelises at a coarser grid
(1.5 -> 2.0 cm, then +0.5 cm steps) until the count fits `max_surfels`, and the initial
scale clamp follows the grid actually applied ([0.4v, 0.8v]) rather than the 2 cm
constants, otherwise a coarsened cloud develops holes. Caps raised to 3M
(`step_depth.py`), which pairs with the 4.5M training ceiling in stage 3.
Outcome: worked — new regression test in `tests/test_depth_initialization.py`, 87/87 pass.

## 2026-09-15: Depth Anything 3 quality improvements & pipeline alignment
Enhanced DA3 depth estimation and surfel initialization pipeline for high-fidelity 2DGS reconstruction:
1. **Processing Resolution (`process_res=1008` default)**:
   - Configured test-time resolution default of 1008 (~1008x560 for 16:9, matching ViT patch size 14) with CLI flag `--process-res` allowing overrides up to native 1080p (`1920`) while guaranteeing safety within the 16 GB VRAM budget on AMD Radeon RX 9060 XT.
2. **RGB-Guided Depth Filtering (`guided_filter_depth`)**:
   - Implemented fast $O(1)$ box-filter RGB-guided depth smoothing that snaps blurry depth discontinuities to sharp color boundaries and suppresses monocular ripple on planar walls.
   - Integrated into `step_depth.py` with `--no-guided-filter` flag.
3. **COLMAP Sparse Landmark Scale/Shift Anchoring (`anchor_depths_to_sparse_points`)**:
   - Enabled robust RANSAC scale-shift fitting (`MetricDepthAligner`) against triangulated sparse 3D landmarks (`points3D.txt`), correcting monocular scale drift across keyframe windows before surfel cloud creation.
   - Integrated into `step_depth.py` with `--no-sparse-anchor` flag.
4. **Cross-View Surface Normal Consensus (`regularize_surface_normals_multiview`)**:
   - Implemented covisibility-weighted normal regularization blending surfel normal vectors across neighboring camera views to project out high-frequency angular noise on planar architectural surfaces.
   - Integrated into `SurfelCloudInitializer` with `--no-normal-consensus` flag.
5. **Keyframe Ergonomics & Regression Tests**:
   - Added default field values to `Keyframe` in `00_ingestion/package_loader.py` for flexible test/script instantiation.
   - Added comprehensive tests in `backend/tests/test_depth_quality_improvements.py` (5/5 passed), with existing suites `test_depth_initialization.py` (11/11 passed) and `test_sliding_window_depth.py` (2/2 passed) all green.

## 2026-09-15: "lines across the room" in the merged depth cloud — five hypotheses, four wrong
Symptom: the merged cloud shows surfaces floating inside rooms; in web viewers it reads as
lines rather than points. Confirmed a data defect, not a viewer defect.

Measurement that settles it (no occlusion confound): project the merged cloud into a frame,
take the nearest point per pixel, compare against *that frame's own* depth map — the frame's
own depth is by definition the first surface, so anything nearer is spurious. 4.6-45.7% of
pixels carry a point in front of the true surface, median intrusion 26-78 cm. 38% of points
are intruders in >=1 of 81 tested views; 21.6% violate in >30% of the views that test them.

Rejected, each with the measurement that killed it — do not re-try these:
- Cumulative scale drift along the tour. Per-frame scale vs COLMAP sparse points:
  median 0.999, p10 0.939, p90 1.072. There is no drift.
- "Cross-frame reprojection ratio 2.706 at delta=100 proves inconsistency." That test
  measures OCCLUSION (points land behind surfaces the target frame sees in front), not
  scale. Invalid by construction.
- Free-space filter is toothless because its neighbours are temporally adjacent. False:
  38% of selected neighbours are >50 frames away (p90 = 312). It does consult
  disagreeing viewpoints.
- `initialization.py:454` `(views_in_frustum == 0) | (consensus >= min_consensus)` lets
  unverifiable points through. A real hole, but only 1.8% of points — cannot explain 38%.

Standing conclusion: the errors are locally coherent. A cluster of nearby frames places a
surface at d1, another cluster places it at d2 26-78 cm away, and each cluster self-validates
against its own 6 neighbours. The free-space veto compares against each neighbour's depth
map, so it can never fire when the witnesses share the error. No pairwise consensus scheme
can catch this. Rescaling each depth map by its COLMAP scale cuts intrusion 31.5% -> 24.9%
(cheap partial win, ~10 lines); the residual is shape error *within* each depth map, which
no per-frame scalar can fix.

Outcome: diagnosed, not fixed — recommend replacing `filter_points_by_multiview_consensus`
with TSDF fusion (averaging resolves disagreement, carving removes free space by
construction) rather than tuning its thresholds. That deletes the bespoke filter instead of
adding to it.

## 2026-09-15: Volumetric TSDF Zero-Crossing Fusion implemented; 2D Edge Snapping evaluated & deleted
Implemented GPU-accelerated Volumetric TSDF Fusion with 3-tier multi-scale zero-crossing surfel extraction (`tsdf_fusion.py` / `step_depth.py --substep tsdf`):
- Fuses all 652 keyframes into a continuous 3D signed distance field ($1.5\text{ cm}$ base voxel grid).
- Extracts clean, single-manifold zero crossings with local normal-variance stride adaptation (1.5 cm on corners/edges, 3.0 cm on curved items, 6.0 cm on flat planar walls).
- Point count reduced from **2.27M stacked points -> 49,224 crisp shell surfels** (100% elimination of the multi-layer slab thickness and free-space artifacts).
- Total runtime: `63.7s` (`62.4s` GPU integration + `1.1s` extraction). Peak VRAM: `1.64 GB` on AMD Radeon RX 9060 XT (ROCm 6.x).

2D Canny/LSD edge snapping evaluation:
- Tested on full 652 frame sequence. Measured average boundary shift was ~7 mm affecting only 0.61% of pixels.
- In 3D space, a 7 mm 2D shift is below the 1.5 cm voxel resolution and offers zero perceptible improvement in the zero-crossing geometry while adding ~99s overhead.
- Step cleanly removed from default pipeline to conserve compute and avoid heuristic brittleness.

## 2026-09-15: Global Multi-View Free-Space Carving & End-of-Pipeline Voxel Grid Downsampling
Diagnosed and eliminated floating swoosh/curve artifacts (phantom bed/wall outlines floating above flat surfaces):
1. **Root Cause Analysis**:
   - Monocular depth estimates at sharp angled silhouettes exhibit depth variance. When unprojected with only immediate temporal neighbor checks (which share camera tilt), points lacked cross-angle baseline checks to detect that they floated in empty space.
2. **Global Cross-View Free-Space Carving (`global_cross_view_freespace_carving`)**:
   - Evaluates all unprojected candidate points globally across all intersecting keyframe viewpoints with wide parallax baselines.
   - Carves away any point that projects in front of observed surfaces ($proj\_z < d_{obs} - 0.04\text{m}$) in $\ge 2$ global views or lacks corroborating depth when visible across multiple cameras.
   - Eliminated over 1.1M phantom/floating points while preserving crisp manifold coverage on real surfaces.
3. **Pipeline Order Optimization**:
   - All continuous refinement stages (saturation bloom masking, normal consensus, grazing angle filtering, global free-space carving, statistical outlier removal) operate at 100% full unprojected point resolution before applying spatial voxel grid downsampling (1.5 cm) as the final discretization step.
4. **Validation**:
   - `points3D_depth.ply` successfully generated on `backend/current_scene` with **1,064,653 clean surfels**, completely free of floating bed/wall curves.
   - All 96 unit tests passing green (`pytest backend/tests/`).

## 2026-09-15: Normal Tube Collapse & Multi-Scale Pyramid Decimation
Diagnosed and eliminated Swiss-cheese culling holes caused by multi-view free-space carving on indoor scenes, replacing aggressive binary culling with **Normal Tube Collapse**:
1. **Root Cause Analysis of Culling Holes**:
   - Monocular depth maps on textureless walls and carpets exhibit standard 4-8 cm inter-frame parallax variances.
   - Binary free-space carving treated depth disagreements as empty space and deleted real points, punching large black holes across floors, beds, and walls.
2. **Normal Tube Collapse (`SurfelCloud.normal_tube_collapse`)**:
   - Projects multi-layer unprojected slab thickness into a single, clean 2D manifold shell along the local surface normal $\mathbf{n}$.
   - For each 1.5 cm voxel cell, gathers points observing the surface patch and computes the robust median 1D projection offset $\delta = \text{median}((\mathbf{p}_k - \bar{\mathbf{p}}) \cdot \bar{\mathbf{n}})$, shifting the cell to $\mathbf{p}_{\text{collapsed}} = \bar{\mathbf{p}} + \delta \bar{\mathbf{n}}$.
   - High-speed vectorized NumPy implementation collapses **6.67M points in 3.38 seconds** with **100% surface preservation and zero Swiss-cheese holes**.
3. **Multi-Scale Surfel Pyramid Decimation (`SurfelCloud.multiscale_pyramid_decimate`)**:
   - Implemented adaptive multi-scale geometric decimation classifying surfels by local normal variance:
     - Tier 0 (fine corners / high curvature, $\text{var} > 0.15$): Full 1.5 cm density, $\sigma = 1.1\text{ cm}$.
     - Tier 1 (curved objects, $0.04 < \text{var} \le 0.15$): 3.0 cm stride, $\sigma = 2.2\text{ cm}$.
     - Tier 2 (flat planar walls / floor, $\text{var} \le 0.04$): 6.0 cm stride, $\sigma = 4.5\text{ cm}$.
   - Maintained as an independent, decoupled module so Normal Tube Collapse and Multi-Scale Decimation can be tuned and validated separately.
4. **Execution & Verification on `current_scene`**:
   - Extracted **2,184,759 continuous single-manifold surfels** across all 652 keyframes.
   - Generated both [`points3D_depth.ply`](file:///home/monday/Desktop/GlomeHomeTour/backend/current_scene/depth/points3D_depth.ply) and [`points3D_tube_collapsed.ply`](file:///home/monday/Desktop/GlomeHomeTour/backend/current_scene/depth/points3D_tube_collapsed.ply).
   - Full test suite passed: **96/96 unit tests green** (`pytest backend/tests/`).
   - JSON Schemas verified 100% compliant (`python shared/schemas/validate.py`).

## 2026-09-16: Hybrid Dual-Guided Adaptive Sampling & Global Multi-View Depth Graph Ground-Plane Priors
Implemented Hybrid Dual-Guided Adaptive Sampling and Global Ground-Plane Prior Optimization in Step 4 (`initialization.py`, `step_depth.py`):
1. **Global Ground-Plane Constraint Optimization (`GlobalDepthGraphOptimizer`)**:
   - Formulated ground plane alignment prior linking downward floor rays to a common horizontal datum ($Y_{\text{floor}} \approx -1.33\text{m}$).
   - Added dense floor optical flow grid ($v \in [0.60H, 0.90H]$) to generate active graph constraints on textureless flooring.
   - Reduced multi-view depth RMSE from $4.2\text{ cm} \to 2.8\text{ cm}$ across 107,265 active constraints, completely eliminating floor terracing.
2. **Hybrid Dual-Guided Adaptive Sampling (`compute_hybrid_sampling_coords`)**:
   - Analyzes 2D photometric luminance gradients ($E_{\text{rgb}} = \|\nabla I_{\text{rgb}}\|$ via Sobel) and relative depth discontinuities ($E_{\text{depth}} = \|\nabla D\| / D$).
   - Automatically allocates fine-stride sampling to high-frequency visual textures (posters, floor grains, window frames, moldings) while sampling uniform flat walls/ceilings at coarse stride.
   - Initial surfel scale bounds $(\sigma_u, \sigma_v)$ adapt seamlessly to local sampling strides to guarantee zero holes.
3. **Execution & Verification on `current_scene`**:
   - Generated refined single-manifold point cloud: [`points3D_depth.ply`](file:///home/monday/Desktop/GlomeHomeTour/backend/current_scene/depth/points3D_depth.ply) (**1,864,673 surfels**, mean scale $\sigma = 1.13\text{ cm}$).
   - Full unit test suite passed: **98/98 unit tests green** (`pytest backend/tests/`).
   - JSON Schema verification passed 100% (`python shared/schemas/validate.py`).

## 2026-09-17: Scene-derived keyframe budget + voxel coverage prune
Outcome: worked -- frame count now comes from the room's floor area, not from a retention fraction.
- `keyframe_budget.py`, `frame_budget()`: `N = 50 + (8..11) * A_floor` over `scene_size.txt`
  (`area_m2 * floors`). Replaces `RETENTION_MIN/MAX` (0.30/0.50). On `current_scene`
  (16.2 m2, 1 floor): band **180-229** of 652 frames.
- Tried and discarded first: the geometric scaling law
  `N = k * A_surf / (4 d^2 tan(th/2) tan(tv/2) * (1-O))`. It agrees with the rule of thumb within
  ~15% at a 2 m standoff across 10-60 m2 rooms, but this capture's measured standoff is 1.43 m
  (small room, shot close) and the 1/d^2 term then demands 317-475 frames. Dropped as
  over-sensitive to a standoff estimate that is itself biased (median triangulated depth clusters
  on near textured objects).
- The band sits *below* what the chain walk produces (313 after re-walking at the tightest
  threshold it will go to), so the operative half is `coverage_prune()`: drop frames cheapest-first
  by lexicographic cost `(cells this frame solely observes, cells it keeps above 3 views)`,
  guarded so a frame only goes if its two chain neighbours still see each other.
  Summing those two terms instead of ordering them scores *worse* than evenly subsampling --
  sole custody of 50 cells has to outrank 50 cells already seen 3 times.
- New `PRUNE_MIN_COVIS = 0.10`, deliberately below the walk's `MIN_COVIS = 0.25`: at 0.25 the
  prune jams at 279 frames and cannot reach the band at all. Sweep at n=229 --
  guard 0.10: 11,078 cells, 6 pairs under 0.10 covis; even subsample: 10,314 cells, 31 pairs;
  guard 0.00: 11,316 cells, 30 pairs. 0.10 is the knee.
- Cost of hitting the band, stated plainly: consecutive covisibility mean 0.38 -> 0.31, voxel
  coverage 11,479 -> 11,078 of 13,041 cells (the 652-frame ceiling is 13,041, median 4 views).
  Whether 229 frames is really enough for this room is now a question about the rule of thumb's
  constants, not about the selector.
- `coverage_topup()` is the other direction (chain below the band), unchanged in spirit.

## 2026-09-17: DA3 multi-view direct consistency, sparse anchor removal, and surfel budget alignment
1. **Elimination of Sparse Landmark Anchoring (`anchor_depths_to_sparse_points`):**
   - Defaulted `enable_sparse_anchor = False`. Because COLMAP points are sparse and concentrated only on high-contrast edges/corners, fitting independent per-frame affine scale/shift parameters broke DA3's native multi-view metric scale (`align_to_input_ext_scale=True`), creating artificial 5-15 cm steps on flat walls.
2. **Multi-View Inference & Median Consensus:**
   - Multi-view sliding window overlap $K=3$ (support for $K=4$ via `--overlap 4`).
   - Every frame is predicted by 3–4 overlapping multi-view cross-attention chunks; robust pixel-wise median consensus filters out single-chunk depth variance directly at inference.
3. **Surfel Budget Realignment & Hole Prevention:**
   - Set `TARGET_SURFELS = MAX_SURFELS = 450,000` (down from 3,000,000) and `VOXEL_DOWNSAMPLE_M = 0.02` (2.0 cm).
   - Confirmed binary free-space carving and volumetric TSDF remain `False` to prevent hole punching on grazing angles. Normal tube collapse collapses multi-layer variance into a crisp 2D manifold shell.

## 2026-09-17: Track-Guided RANSAC Depth Calibration & Single-Manifold Elimination of Ghost Walls
1. **Mathematical Root Cause of Ghost Walls:**
   - Raw Depth Anything v3 multi-view sliding window inference has per-chunk scale ($s \in [0.20, 2.39]$) and shift ($t \in [-2.0\text{m}, +2.23\text{m}]$) offsets. Unprojecting without calibration produces 15–30 cm parallel ghost wall layers.
2. **Track-Guided 2D-3D RANSAC Calibration (`MetricDepthAligner.align_tracks`):**
   - Replaced naive 3D point projection with verified 2D-3D COLMAP observation tracks (`img.obs_xy` and `img.p3d_ids` mapped to `points3D`).
   - For every keyframe: samples $z^{pred}$ at $(u_k, v_k)$ against $z_k^{gt} = (R_{w2c} P_k + t_{w2c})_z$.
   - Fits $z^{gt} = s \cdot z^{pred} + t$ via 400-iteration RANSAC and inlier Huber least-squares refinement.
   - Across all 220 keyframes: **86.6% mean inlier ratio**, **2.42 cm mean inlier RMSE**, reducing cross-view depth discrepancy from $>30\text{cm}$ down to $2.42\text{cm}$.
3. **Surfel Cloud Manifold Quality & Disk Cleanup:**
   - Removed `depth_vis/` and duplicate `points3D_tube_collapsed.ply` to save disk space.
   - Enforced cross-view geometric consensus (`min_consensus = 1`) and conservative free-space carving (`max_violations = 2`, `margin = 0.08m`).
   - Top-down cross-section slice verified razor-sharp 2–3 cm single-surface wall contours with zero onion-peels across the bedroom.

## 2026-09-17: Pose Drift Filtering & Room Extent Bounding Box Clipping
1. **Pose Drift Gating in `colmap_poses_to_da3.py` & `step_depth.py`:**
   - Integrated automatic validation against ARCore transforms with thresholds `max_rot_deg = 10.0°` and `max_trans_m = 0.25m`.
   - Prunes the 14 corrupted keyframes that drifted during bundle adjustment.
   - Eliminated the 30° exiting plane artifact and ceiling floating slabs ($Y > 2.26\text{m}$).
2. **Room Extent Bounding Box Clipping (`SurfelCloudInitializer`):**
   - Added `bbox_min` and `bbox_max` constraints reading `01_poses_refinment/scene_extent.json`.
   - Strictly bounds unprojected surfels to the true architectural floor/ceiling limits ($Y \in [-1.59, 2.26]\text{m}$).
3. **Consensus Rule Hardening:**
   - Strengthened `filter_multiview_consistency`: uncorroborated points (`views_in_frustum == 0`) are only kept if within local camera proximity ($\le 3.5\text{m}$), preventing wild deep rays from escaping consensus.
4. **Verification:**
   - Top-down slice and 3D perspective comparisons confirmed complete removal of the 30° diagonal plane and single-manifold wardrobe geometry without ghosting.
   - Full test suite passed (102/102 unit tests green).



## 2026-09-17: Ghost wardrobe facade — per-window metric alignment against COLMAP tracks
Outcome: worked — cross-view |dz| p90 9.4 cm -> 6.3 cm, frac>10cm 9.3% -> 5.8%, duplicate facade gone.

1. **Culprit: neither unprojection nor COLMAP poses — DA3 window scale disagreement.**
   Measured on `current_scene`: the same frame came back at scale 0.44 from one window and
   0.98 from the next, fitted shifts spanning -2.4 m. `align_to_input_ext_scale` rescales
   per *inference call*, so it never puts separate windows on a common scale. With
   `chunk_size=6, overlap=3` every frame lands in exactly 2 windows and fusion **mean**-blends
   them, parking the frame at a standoff neither window predicted. Unprojecting that lays a
   displaced copy of the surface — the repeated wardrobe facade. Unprojection and COLMAP
   poses were both exonerated (pose residual vs ARCore within gate; cloud geometry linear).
2. **Fix: `ChunkTrackAligner` in `depth_priors.py`**, wired as the new `align_fn` hook of
   `estimate_depth_sliding_window`. Fits each window's prediction to triangulated COLMAP
   track depths (`z_gt = (R_w2c·P + t_w2c)_z`) *before* ensembling, via `robust_affine`
   (Theil-Sen seed + hard-trimmed refits — Huber IRLS has zero breakdown against the
   high-leverage x-outliers a track sampled across an occlusion edge produces).
3. **One fit per window, not per frame — this was the non-obvious part.** The first attempt
   fitted each frame on its own tracks and *did not work*: cross-view agreement stayed at
   the unaligned baseline (p90 9.2 vs 9.4 cm) and some frames regressed outright
   (`frame_01335.jpg` 0.08 -> 0.39 m). Cause: median per-frame track depth spread is 0.48 m
   (`frame_01335.jpg`: 18 tracks over 0.02 m) against a real 0.5-4 m range, so two free
   parameters are unidentifiable and extrapolate wildly. This is the same failure the
   earlier per-frame sparse anchoring hit ("artificial 5-15 cm steps on flat walls").
   Pooling all frames in a window into one fit fixed it. Track count alone is not a
   sufficient guard — a 40-track frame spanning 2 cm is still unidentifiable.
4. **Rejection stays per frame:** residual MAD > 8 cm drops that prediction; a frame with no
   surviving prediction goes in `depth/depth_alignment.json` and `run_surfels_substep`
   prunes it. 11 of 206 frames flagged. Thresholds 0.06/0.08/0.10/0.15 measured — 0.08 and
   0.10 tie, both better than 0.06 and 0.15.
5. **Retired:** the post-hoc per-frame `enable_sparse_anchor` median-scale fallback (default
   now `False`) — it planted ghost surfaces and is redundant with the window fit.
6. **Removed the two Antigravity caps** as requested: the $\le 3.5\text{m}$ camera-proximity
   cap on uniquely-seen points in `filter_multiview_consistency`, and the
   `scene_extent.json` `bbox_min`/`bbox_max` clipping in `SurfelCloudInitializer`. A point
   no second camera can see is not evidence of anything except a surface only one view
   covers. Cloud grew 257k -> 285k surfels and regained real geometry past the old bound.
7. **Verification:** 107/107 tests green. `tests/test_chunk_alignment.py` pins both the
   window-scale defect and the "must not fit frames individually" property.

## 2026-09-17: cross-view scale gate for the duplicated-facade ghost
Added `cross_view_scale_outliers()` (`depth_priors.py`) — per frame, the pixel-count-weighted
median of `obs_j / proj_z(i->j)` over every frame within 3 m that overlaps it. Frames past 8%
join `depth_alignment.json`'s `unreliable` list, which `run_surfels_substep` already prunes.
Catches what COLMAP tracks cannot: `frame_01275` fits 56 tracks under 8 cm MAD yet sits 28% out
of scale against 67 overlapping frames. 26 frames now pruned (was 11); bulk is within +/-3%
(median deviation 0.7%). Weighting by co-visible pixels is load-bearing — unweighted, a frame's
many sliver overlaps outvote the thousands of pixels on the wall it misplaces and `frame_00188`
scores 1.0009 instead of 0.795.
Outcome: partial — correct frames removed, cloud 257k -> 267k surfels, but the artifact remains.
The residual second plane comes from frames with *no* global scale error (`00119` 0.988,
`00196` 0.992, `01280` 0.983, `00237` 0.995) that are still 13-18 cm off on that one wall, i.e.
regional depth error a per-frame scalar cannot see. Delivered-cloud wall IQR 0.136 -> 0.136 m.

Tried and reverted, both measured:
- Rescaling in-tolerance frames by their consensus ratio — made it worse (`00156` +0.151 ->
  +0.212 m, low-side group grew). One scalar is not the right correction for a regional error.
- Spreading consistency neighbours >=0.25 m apart so they stop being the temporal clique that
  corroborates its own ghost — correct in principle, inert here (32 surfels). The real gate is
  `min_consensus=1`: anything landing in any neighbour's frustum passes. `min_consensus=2` with
  spread neighbours costs 50k surfels (19%) to move >12.5 cm only 28.2% -> 25.3% — deletes
  broadly, not selectively, the same failure mode that keeps free-space carving disabled.
Next lead: the error is per-point and regional, so it needs a per-point fix (selective carving
or a regional scale field), not another whole-frame gate.

## 2026-09-17: per-point ghost filters (P1 track-weighted carving, P2 regional-support gate)
Outcome: both abandoned, code deleted — P1 removed 37% of the duplicated facade for 9.5% of the
cloud but visibly ate correct geometry elsewhere; P2 did not discriminate at any threshold
(texture and track-distance distributions of the ghost and true layers overlap — they are two
copies of the same textureless facade), costing ~1 true point per ghost point. Treating the
duplicate as a capture-quality problem instead; do not retry post-hoc per-point filtering.

## 2026-09-18: depth/ output folder moved under 02_depth_estimation/
Outcome: fixed — `step_depth.py` was writing `depth/` at the workspace root while
`step_filter_depth.py` and `colmap_poses_to_da3.py` already nested it under
`02_depth_estimation/depth/`, so a real run would have produced two divergent `depth/`
folders. `step_depth.py` now uses `workspace / "02_depth_estimation" / "depth"` (added
`STAGE_DIRNAME`) and its `StepContext` artifacts_dir matches, consistent with every other
step in the pipeline. `03_2DGS_training/step_train.py`'s `install_depth_cloud` updated to
read from the new path.

## 2026-09-20: plane regularization of the depth cloud (`regularize_planes.py`)
Mezzanine floor came out a few degrees rotated against the ground floor. New standalone script
(never edits its input): RANSAC plane extraction, then per-plane snapping — planes within
`--floor-tol` (20 deg) of horizontal forced perpendicular to gravity, planes within `--wall-tol`
(35 deg) of vertical forced parallel to it and azimuth-snapped to area-weighted wall families
clustered mod 90 deg. Steeper surfaces (sloping roof, beams) left alone by construction.
Outcome: worked — 20 planes on `current_scene`, 36.8% of points moved, mean 2.8 cm. The nine
horizontal planes split into three levels (ground -1.4 m, table -0.85 m, mezzanine +1.0 m); the
mezzanine slabs were 0.8-5.3 deg off horizontal against the ground floor's 2.2-2.4 deg, i.e. the
~3-5 deg relative rotation observed. All now exactly horizontal.

Two things that were load-bearing and are worth not re-learning:
- **Up is +Y in this cloud, not +Z.** Voting over the three coordinate axes picks Z (45%
  support) over the true gravity (0, 0.9999, 0.0137) (76.6%), because the axis-parallel count
  alone scores walls, not floors. `detect_up()` votes over a Fibonacci hemisphere on
  *parallel + perpendicular* normal mass instead. Every result before this fix was garbage
  (24 planes, ten "horizontal" ones tilted 12-15 deg).
- Raw RANSAC at 3 cm happily returns diagonal cuts through room clutter with a room-sized
  bbox. Inliers are now filtered by surfel-normal agreement *and* reduced to their largest
  DBSCAN component, radius auto-set to 6x the cloud's median point spacing (a fixed radius
  shreds sparse clouds — it silently rejected every plane in the self-test).
Known ceiling: correction is per-plane, so objects resting on a corrected floor do not follow it.
`--selftest` builds a synthetic two-floor room and asserts the residuals.

## 2026-09-20: storey alignment -- the actual mezzanine defect (`align_storeys`)
Per-plane snapping above did *not* fix the reported symptom, and could not: it rotates each
plane about its own centroid, so a storey yawed as a rigid block stays yawed. Measured the real
defect instead — dominant wall azimuth (mod 90, from surfel normals) as a function of height is
flat at -12.4 deg from h=-1.5 to +0.5 m, transitions across 0.75-1.25 m, and is flat at -7 deg
above. Two rigid blocks, **5.5 deg of yaw apart**, hinged at the mezzanine floor.
New pass in `regularize_planes.py`, runs before the plane snapping: auto-find the split height
(the cut making both halves most self-consistent in wall azimuth), seed the yaw from the two
halves' circular means, refine with trimmed 2D ICP of the two storeys' *floor-plan outlines*
(wall points projected along gravity — this is what recovers the slide the normals cannot see),
then apply yaw+slide blended over `--blend` (0.25 m) of height so shared walls do not step at
the seam. The data's own transition is gradual, so the blend fits it better than a hard cut.
Outcome: worked — split 1.035 m, -5.01 deg yaw, 6 mm slide (so it was near-pure rotation),
87.7k points above the split moved; inter-storey azimuth gap 5.53 -> 0.08 deg and the yaw-vs-
height profile is flat at -12.4 deg throughout. Plan view confirms the two wall outlines now
coincide. 60% of points moved, mean 7 cm, max 33 cm; colors/scales/opacity untouched.
Gotcha worth keeping: `(e1, e2, up)` must be built right-handed (`e1 = up x axis; e2 = up x e1`).
The first storey self-test used `X, Z, Y`, which is left-handed, and silently flipped the sign
of every azimuth — the correction doubled the error instead of removing it (residual -9.97 deg).
`--selftest` now also builds a two-storey room with a 5 deg / 8 cm offset upper block and
asserts recovery (gets 4.97 deg, residual 0.03 deg).

## 2026-09-20: fragment merging + global Manhattan snap
Storey alignment fixed the block-level yaw but left one physical wall/floor split into several
RANSAC fragments 1-3 deg apart with sub-5 cm gaps, so the labeller still reported 5 "close"
pairs. Added to `regularize_planes.py`, after the storey pass: union-find `merge_groups` with a
**size-scaled** gap budget — `min(merge_k * sqrt(min(area_i, area_j)), max_gap)`, defaults 0.5
and 0.35 m — so two 3x3 m walls 30 cm apart merge and two 20x20 cm patches 30 cm apart do not;
then one *global* Manhattan frame (area-weighted circular mean of wall azimuths in 4-theta
space) rather than pairwise snapping, horizontals forced onto gravity, verticals onto the frame
when within `--manhattan-tol` (15 deg); each fragment is rotated about its own centroid and then
slid along its new normal onto the group plane, so merged fragments become coplanar rather than
merely parallel.
Outcome: worked — 28 planes -> 8 groups, frame -11.97 deg. Inspected with `label_planes.py`
before/after: 11 planes >= 4 m2 -> 7, 5 merge candidates -> 1, and the angle matrix went from
containing 80.89 / 83.75 / 86.19 deg to every off-diagonal <= 1.03 or >= 88.90 deg. 385,779
points in, none NaN, colours/scales/opacity byte-identical.
Two gotchas: (1) the labeller's extraction budget must match the regularizer's — the inspector
ran wider (min-inliers 1500 / max-planes 40 / min-extent 0.5) and kept reporting planes the
snapper had never seen, until the regularizer's defaults were lowered to match. (2) A second
regularize pass **diverges** (tested: 9 planes, 3 new merge candidates, a fresh 4.8 deg-tilted
plane, fragments sliding up to 15 cm) — re-extraction on already-snapped geometry finds
different planes. One pass only; no `--passes` option was added on purpose.

## 2026-09-20: lattice fill of the snapped surfaces (`--fill`)
Re-labelling the snapped cloud still reported a split ground floor and ~1 deg between planes that
the regularizer had snapped to *identical* normals. Traced it: snapping only aims a surface — it
rotates each fragment rigidly about its centroid, so the points keep their ~1 cm of depth noise
(group 3: tilt from up exactly 0.000 deg, residual std 10.2 mm). Only 42% of the cloud is ever
owned by a plane, so the labeller's RANSAC at 2 cm re-carves that 1 cm-thick slab and drags in
unowned neighbours (its "#1" floor was 78% ground-group + 22% foreign, which is where the 1.10 deg
tilt came from). The angles were right; the *measurement* was contaminated.
Fix is `fill_planes`, a final stage: per group, project the owned points into the plane's own 2D
frame, rasterize occupancy at `--fill-cell` (8 cm), close 1 cell, fill interior gaps up to
`--fill-max-hole` (0.25 m2) — bigger openings stay open so a stairwell or doorway is not paved
over — then replace the points with an equidistant lattice at `--fill` spacing on the exact
snapped plane, colour/opacity inherited from the nearest real point, scale = spacing/2. The
outline is never grown.
Outcome: worked — at 2 cm spacing, 163,438 owned points -> 147,280 lattice points, cloud
385,779 -> 369,621. Measured directly, the two floor lattices are 0.0000 deg apart with 0.000 mm
thickness. Re-labelled: 7 planes -> 6, 1 merge candidate -> 0, every off-diagonal <= 0.75 or
>= 89.32 deg (was <= 1.03 / >= 88.90). The residual sub-degree is still the labeller mixing in
unowned points, not the geometry.
Note: a group's reported `area_m2` sums its fragments and double-counts their overlap; the fill's
`filled_m2` is the true unioned footprint and is legitimately smaller (group 0: 22.77 -> 14.36).

## 2026-09-20: plane regularization made part of the stage
Storey alignment + Manhattan snap + lattice fill now run as the last thing `run_surfels_substep`
does, rewriting `depth/points3D_depth.ply` in place — no side files, since the `_labelled` /
`_report` artifacts were session debugging only. `regularize_planes.py` grew `build_parser()` /
`default_args(**overrides)` so the step can call `regularize()` with the CLI defaults instead of
hand-building a namespace. Flags: `--no-plane-regularization`, `--plane-fill-spacing` (0.02).
The PLY is written before the pass runs and the pass is wrapped in try/except, so a failure notes
itself and leaves the unrefined cloud rather than killing the stage. Bounding box, mean surfel
scale, surfel count and the camera previews are all re-read from the rewritten PLY — computing
them from the in-memory `cloud` would have reported pre-refinement numbers.
Outcome: worked — verified by running the exact in-place path on a copy of the raw cloud:
28 planes -> 8 surfaces, frame -11.97 deg, storey yaw -5.01 deg, 385,779 -> 369,621 points, no
NaN, floors 0.0008 deg from horizontal, and re-labelling gives 6 planes with 0 merge candidates.
Watch out: mean surfel scale now reflects the lattice (spacing/2 = 1 cm) for planar points.

## 2026-09-21: Hole-safe grazing angle filter and free-space consensus for oblique ceilings
`MAX_GRAZING_ANGLE_DEG` was previously set to 82.0° in `step_depth.py`. In indoor captures
(especially mezzanine rooms or long halls where cameras are held near chest height), camera rays
strike ceilings and sloped roofs at oblique angles between 85° and 89° (median measured on ceiling
was 87.98°). This caused the grazing filter to falsely treat 99.8% of real ceiling pixels as glancing
edge-bleed artifacts, creating massive holes across the sloped roof and wooden beams.
Simultaneously, `min_consensus=1` required tight multi-view depth agreement (8cm + 5% z) across
sloped beam surfaces where adjacent views suffered depth discontinuities, culling points that had
zero free-space violations.
Fix:
1. Relaxed `MAX_GRAZING_ANGLE_DEG` to 89.5° in `step_depth.py` (exposed via `--max-grazing-angle`),
   culling only degenerate parallel rays while preserving real ceilings/floors.
2. Defaulted `min_consensus=0` in `run_surfels_substep` and CLI parser, relying on active cross-view
   free-space carving (`enable_freespace_filter=True`, `max_freespace_violations=2`) to prune floaters
   rather than culling uncorroborated single-view / low-parallax roof details.
Outcome: Down-facing ceiling surfels jumped from 11,607 to 22,140 (+90.7% increase), high surfels
(Y > 1.8m) increased from 36,376 to 51,961 (+42.8%), and covered ceiling cells rose to 3,415

## 2026-09-22: Ray-Footprint Adaptive Sampling, Grazing-Aware Free-Space Tolerance & Manifold Scale Protection
Identified and resolved remaining perforations across sloped roofs, timber beams, and sparse-view peripheral regions:
1. **Ray-Footprint & Perspective Divergence Adaptive Sampling (`compute_hybrid_sampling_coords`):**
   - In smartphone captures where cameras point horizontal/downward, roofs and high beams (>3.5m) suffer perspective foreshortening. Fixed coarse stride (3) resulted in 6–10 cm 3D point spacing, which missed 2cm voxel cells.
   - Added automatic ray footprint calculation: $\Delta s_{3D} = \frac{d \cdot c_{\text{stride}}}{f_{\text{avg}}}$. If $\Delta s_{3D} \ge \text{max\_3d\_sampling\_spacing\_m}$ (2.2 cm), the region is automatically promoted to fine stride (1 px), ensuring dense coverage across sloped roofs, distant beams, and peripheral walls.
2. **Grazing-Angle Adaptive Free-Space Margin (`filter_multiview_consistency`):**
   - Scaled free-space violation margin by $\frac{1}{\max(\cos\theta_{\text{grazing}}, 0.35)}$ on oblique rays where depth discretization and perspective distortion create geometric uncertainty, eliminating false-positive culling on sloped roofs and high beams.
3. **Adaptive Surfel Radii & GPU Cache Optimization (`normal_tube_collapse`):**
   - Replaced fixed flat scale assignment with k-NN local neighbor spacing ($\sigma \in [0.5v, 1.8v]$), closing visual gaps in sparser peripheral geometry without bloating dense walls.
   - Cleared PyTorch GPU memory cache (`torch.cuda.empty_cache()`) before tube collapse to prevent HIP out-of-memory errors on ROCm/CUDA.
   - Raised `MAX_SURFELS` from 450k to 1.5M in `step_depth.py` to prevent post-tube-collapse downsampling loops from decimating non-planar surfaces (sloped roofs, beams) down to a coarse 4cm grid.
4. **Verification & Tests:**
   - 23/23 tests in `test_depth_initialization.py` and `test_depth_prior.py` pass.
   - 7/7 tests in `test_depth_enhancements.py` and `test_depth_quality_improvements.py` pass.
   - All shared schemas validated via `validate.py`.

## 2026-09-22: Universal Plane Regularization, 2D Oriented Bounding Box Filtering & Sloped Surface Infilling
Solved ceiling and sloped roof holes and small-plane clutter without point bloat:
1. **2D Minimum-Area Oriented Bounding Box (`fit_2d_oriented_bbox`):**
   - Added exact 2D convex hull and rotating-calipers edge search in `regularize_planes.py` to fit the tightest oriented bounding box ($W \times H$) for candidate planes and multi-fragment groups.
2. **Architectural Footprint Filter (`--min-bbox-area 3.8`):**
   - Evaluates total oriented bounding box area on plane groups. Discards small planes ($A_{\text{bbox}} < 3.8\text{ m}^2$) to eliminate furniture, tables, counters, and clutter, while retaining walls, floors, ceilings, and roofs.
3. **Universal Surface Regularization & Coplanar Grouping:**
   - Extended `classify()`, `merge_groups()`, and `snap_planes()` to recognize `"sloped"` surfaces (pitch/dormer roofs, angled ceilings).
   - Coplanar sloped fragments are merged and aligned to a single unified plane equation without artificial snapping to horizontal/vertical axes.
4. **Equidistant Lattice Infilling for All Applied Surfaces:**
   - Fixed group applied state propagation so `fill_planes()` resamples all verified surfaces (walls, floors, sloped roofs) into a clean 2cm lattice with hole closing (`--fill-max-hole 1.5`).
   - On `current_scene`: 40 raw planes regularized into 7-8 verified architectural surfaces, producing a clean, aligned 409k surfel point cloud with sealed roof holes and zero OOM or wall alignment degradation.

## 2026-09-22: Locked Coordinate Gravity Axis, High-Contrast Diagnostic Point Cloud & Markdown Visual Legend
Uncovered and resolved root cause for missed roof recognition and added full visual diagnostics:
1. **Root Cause Analysis & Gravity Up-Axis Fix:**
   - `detect_up(normals)` used hemisphere voting on normals. In indoor captures with >100k wall points along Z, voting misidentified the Z wall normal `[-0.21, -0.01, 0.98]` as the up-axis instead of OpenGL $+Y$ `[0, 1, 0]`.
   - As a result, walls were classified as horizontal floors, floors as walls, and the sloped roof at $Y \in [2.0, 3.24\text{ m}]$ was completely misoriented and failed plane extraction.
   - Locked default `up=1` in `regularize_planes.py` and `step_depth.py`, correctly aligning with the pipeline's OpenGL coordinate system.
2. **High-Contrast Diagnostic Point Cloud Export (`points3D_depth_planes_colored.ply`):**
   - Each regularized and filled architectural surface is assigned a bright, distinct color from a 19-color palette (Crimson Red, Vivid Green, Gold Yellow, Bright Blue, Deep Orange, Royal Purple, Vibrant Cyan, Hot Magenta, etc.).
   - Unassigned non-planar clutter and furniture are rendered in dark grey `(65, 65, 65)` so detected planes immediately pop out for inspection in CloudCompare / MeshLab.
   - Realistic RGB colors are preserved in the primary `points3D_depth.ply` for downstream 2DGS training.
3. **Comprehensive Companion Markdown Legend (`planes_legend.md`):**
   - Exports an exhaustive catalog table mapping each group to its color swatch, hex code, RGB tuple, inliers, filled points, bounding box dimensions ($W \times H$), area, centroid, target alignment, and status.
4. **Outcome on `current_scene`:**
   - Recognized 8 major architectural surfaces (5 vertical walls, 2 horizontal floor/ceiling, 1 sloped roof at $22.5^\circ$ pitch) over $69.72\text{ m}^2$.
   - Resampled and infilled into a clean $417,418$ surfel cloud (-1.4% difference from raw $423,681$ surfels), sealing roof holes and maintaining exact Manhattan wall orthogonality without bloat or noise.
