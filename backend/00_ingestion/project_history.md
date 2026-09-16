# Project History: Backend Ingestion

## Milestone: Ingestion, Quality Gating & Dynamic Keyframe Selection

- **Package Loader:** Handles ZIP archives and directories; validates against `transforms.json`, `coverage_summary.json`, and `trajectory.csv` schemas.
- **Quality Gate:** Filters motion-blurred, over/underexposed, and redundant frames.
- **Pose Synchronization:** Quaternion SLERP and spline translation matching keyframes to 60 Hz VIO trajectory.
- **Dynamic Keyframe Selection:**
  - Implemented `DynamicKeyframeSelector` in `keyframe_selector.py`.
  - Enforces minimum spatial baseline (>= 0.35m) and angular parallax (>= 18 deg).
  - Evaluates 3D frustum co-visibility overlap (0.35 <= covis <= 0.78) against all historical anchor poses to prune loop-revisit redundancies and prevent double-wall smearing from cumulative VIO drift.
  - Verified on 8.8-min scan: condensed 941 quality-gate keyframes down to 103 optimal anchor views.

## 2026-09-13: Rebuilt the quality gate after it discarded ~50% of the Bedroom2 scan
Outcome: worked — Bedroom2 (hard sun through the windows, very uneven lighting) now
stages 97.8% -> 96.6% -> 4.5% across the three stages, and the 31 blur rejections were
visually confirmed to be genuinely smeared frames.

Root cause: sharpness was judged on absolute Laplacian variance against a fixed floor
(`min_blur_floor=30`, which sat at the scan's own median). Laplacian variance scales with
scene brightness and contrast, so dim wood ceilings scored identically to real motion blur.
Now scored on two axes, absolute detail and detail normalised by local intensity variance,
relative to the scene median — a frame must fail *both* to count as blurred. The two
metrics fail on opposite material (normalised punishes high-contrast sharp frames, absolute
punishes dark flat ones), so neither works alone.

Also: exposure now judges clipped-pixel fractions instead of mean luminance, and the
`max_illumination_jump` rule was deleted outright — in an unevenly-lit room, panning from a
window to a wall is a legitimate 60+ level jump between consecutive good frames.

## 2026-09-13: Split filtering into three staged folders (`run_staged_filtering.py`)
Outcome: worked — `01_quality/`, `02_parallax/`, `03_keyframes/`, each with `images/` and a
`transforms.json` covering exactly those images. Redundancy moved out of `QualityGate` into
`prune_redundant()` so stage 1 is purely per-image quality; stage 3 is the existing
`DynamicKeyframeSelector.for_2dgs_training`. Supersedes `stage_filtered_images_only.py`
(left in the tree, it has uncommitted local edits).

## 2026-09-13: Loosened stage-3 anchor selection (61 frames was too sparse)
Outcome: worked — 17.6% of stage 2 instead of 4.5%. Swept the selector on Bedroom2 poses:
`max_covisibility` is the only parameter that matters (0.65 -> 0.95 moves the count 5% -> 26%
of input, while 2-3x swings in `min_translation_m`/`min_rotation_deg` move it <1%, because
covisibility rejects almost everything first). `for_2dgs_training` now defaults to 0.88
overlap. Since the mapping from overlap to count is scene-dependent, `run_staged_filtering.py`
also passes `min_keyframes`/`max_keyframes` at 10%/20% of the stage-2 count, so the band holds
on captures with different room sizes or walking speeds.

## 2026-09-14: Fixed stage-3 keyframes with no overlap between neighbours
Outcome: worked — 1st-percentile overlap between consecutive anchors went 0.00 -> 0.61 on
Bedroom2, 491/1353 kept (36.3%, was 249/17.6%). Four separate causes, none of them the
20%/50% cap (which never fired — the uncapped walk returned 238, under the 270 ceiling):

1. Redundancy was tested against *every* previously selected pose within 4 m, not the last
   one. A frame that was the sole bridge between two anchors got dropped as a "loop revisit",
   and the next accepted frame could be anywhere. This killed 1115 of 1353 frames.
2. `coverage_gap_prevention` accepted the frame that had *already* fallen below the overlap
   floor, one frame too late. Now it anchors on the previous frame, which was still above it.
3. Overlap was measured in one direction only (`_compute_frustum_covisibility` samples A's
   frustum and asks what B sees). Approaching a wall scores high one way, low the other, so
   disjoint pairs passed. Now `_mutual_covisibility` takes the worse direction.
4. Selection ran on landscape poses with portrait intrinsics — `run_staged_filtering.py`
   applied `R_ROLL` at write time, after stage 3. The roll moved to just after `PoseAligner`,
   so every stage shares the upright frame (stages 1-2 output is byte-identical).

The `max_keyframes` cap used to be enforced by subsampling every Nth anchor out of a chain
built for overlap, which is a direct way to manufacture the reported bug. Caps are now met by
re-walking at a tighter/looser `max_covisibility`, and are explicitly best-effort — the walk
never drops below `min_covisibility` to fit a budget. `for_2dgs_training` floor raised
0.45 -> 0.60, sample grid 6x6 -> 10x10 (36 samples was too coarse to trust a threshold on).

Residual: 2 output pairs still overlap <0.4, both inherited from 4 adjacent-frame VIO
relocalisation jumps present in stage 2 itself — unbridgeable by any subset. They are counted
as `unavoidable_gap` in the result reasons rather than hidden.

## 2026-09-14: Keyframe overlap was measured against a plane hardcoded at 2 m
Outcome: worked — 580/1353 kept (42.9%), consecutive-overlap p1 0.00 -> 0.31, median 0.72, and
the only 6 remaining disjoint pairs are discontinuities that exist in stage 2 itself.

The earlier fix the same day was verified with the very metric that was broken, which is why it
looked fine and wasn't. `_compute_frustum_covisibility` samples a grid at `reference_depth_m`
and asks what the other camera sees. That constant was 2.0 m; Bedroom2's real subject distance
is ~0.83 m median (p25 0.55). Where the scene is nearer than the assumed plane the sampled
points sit *behind* the real surfaces, so a sidestep that sweeps the entire view still scores as
a near-duplicate. Measured against the 2 m plane the shipped selection looked clean at p1=0.61;
measured against actual depth it had 137 disjoint pairs. Visually confirmed on frames reported
by the user: 164/165 scored 0.85 and 461/462 scored 0.90 while sharing essentially nothing.

Fix: `estimate_scene_depths()` measures per-frame depth directly — corners tracked into a later
frame (baseline grown 4/8/16/32 until triangulation is conditioned) and triangulated against the
known poses. No depth network, ~40 ms/frame, ~60 s for a scene. Overlap is then sampled at the
*nearer* of the two frames' depths, since near geometry is what leaves the frame first. Same
poses, same pairs: 164/165, 454/455 and 461/462 all drop to 0.00 while healthy pairs hold 0.84.

Two things that were not obvious:
- Raw triangulated depth is noisy enough to flip borderline pairs (same frame measuring 0.40 m
  on one pass and 0.81 m on the next, 428 jumps >0.2 m between adjacent frames). A 5-frame
  rolling median cuts that to 187 without shifting the level (0.81 -> 0.82). Without it, a
  re-run disagrees with itself about which pairs are acceptable.
- A fixed *band* of depths (worst overlap over 0.5-4 m) does not work as a substitute: catching
  the near-field cases costs 80% of stage 2. The depth has to be per-frame, not a wider guess.

Thresholds are calibrated against measured depth and are not comparable to the old ones:
`for_2dgs_training` is now 0.50/0.80 (was 0.60/0.88). ~14% of frames pin to the 0.4 m clamp
floor, which biases toward assuming things are close, i.e. toward keeping more frames — the
safe direction, and `depth_range_m`/`near_percentile` are the knobs if it needs tuning.

`inspect_scan.py` funnels through the same selector and was passing no depths, so it silently
kept the 2 m assumption; it now measures depth too.

Still not fixable here: 6 pairs of *adjacent* stage-2 frames are themselves disjoint (covis
0.00-0.29), including the user-reported 454/455 (stage2 1257->1258). No subset of the input can
bridge those — they are capture-side jumps, possibly widened by stage 1/2 dropping a blurred
frame that was the only link. Reported as `unavoidable_gap` rather than hidden.
