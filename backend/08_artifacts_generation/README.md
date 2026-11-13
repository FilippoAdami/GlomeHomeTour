# Stage 08 — views and artifact generation

Stage 08 generates presentation and inspection products after mesh elaboration and Phase 07 scan alignment. It owns top-down floor views, elevation sections, cutaway views, floorplan symbols, and image comparisons. Phase 06 may create temporary 2-D masks and contours when they are needed to build structural geometry; those internal mesh-building inputs remain in Phase 06.

## Stage files

| Component | Responsibility | Runtime status |
| --- | --- | --- |
| [`generate_floor_plans.py`](generate_floor_plans.py) | Runs the complete first Phase 08 macro step, from floor-level metadata through final per-instance categorization. | Canonical entry point: **Generate floor plans**; writes to `artifacts/Generate Floor Plans/`. |
| [`extract_best_views.py`](extract_best_views.py) | Selects compact acquired-image galleries and preserves their Phase 07 aligned COLMAP calibration and poses. | Completed second macro step: **extract_best_views**; writes to `artifacts/Extract Best Views/`. |
| [`generate_360_pictures.py`](generate_360_pictures.py) | Selects locations using triangle coverage and adaptive height near acquired poses, then renders paired sealed Gaussian and mesh panoramas. | Third macro step: **Generate 360 Pictures**; writes to `artifacts/Generate 360 Pictures/`. |
| [`generate_interactive_viewer.py`](generate_interactive_viewer.py) | Combines plans, acquired photos, and panoramas into a local tour and static comparison images. | Fourth macro step: **Generate Interactive Viewer**; writes to `artifacts/Generate Interactive Viewer/`. |
| [`generate_cinematic_flythrough.py`](generate_cinematic_flythrough.py) | Plans and validates the viewer's numbered tour, then renders synchronized mesh and Gaussian portrait videos with a Standard minimap. | **Generate Cinematic Flythrough**; preview must be inspected before the final render. |
| [`generate_possible_views.py`](generate_possible_views.py) | Generates and selects per-scan orthographic cutaway candidates with per-slice provenance and coverage. | Standalone **Generate Possible Views** macro; writes to `artifacts/Generate Possible Views/`. |
| [`assign_floor_level_metadata.py`](assign_floor_level_metadata.py) | Assigns scan-level and floor-level names to aligned floor/ceiling planes without rendering images. | Metadata-only classifier for Phase 07 aligned regularized elements. |
| [`generate_floor_level_meshes.py`](generate_floor_level_meshes.py) | Writes one regularized and one sealed cut mesh per overall floor level, merged across scans. | Uses aligned per-scan meshes and `floor_level_metadata.json`. |
| [`generate_floor_level_projections.py`](generate_floor_level_projections.py) | Renders one top-down PNG for each regularized and sealed per-floor mesh. | Reads the generated floor-mesh manifest. |
| [`generate_floorplan_instance_overlays.py`](generate_floorplan_instance_overlays.py) | Overlays final categorized instances on separate regularized top-down views. | Runs after final Phase 08 categorization. |
| [`generate_floorplan_measurements.py`](generate_floorplan_measurements.py) | Writes one transparent PNG of approximate wall-segment dimensions per floor. | Final artifact step in **Generate floor plans**. |
| [`generate_classified_object_floor_slices.py`](generate_classified_object_floor_slices.py) | Creates separate regularized floor-mesh copies with height-fitting classified objects, axis-aligned boxes, and top-face labels. | Experimental overlay from Phase 07 aligned objects and fixed-class associations. |
| [`extract_cross_section_views.py`](extract_cross_section_views.py) | Renders unobstructed top-down floor views and four orthographic elevation sections from the Stage 04 mesh, using Stage 07 alignment metadata when supplied. | Main section-view generator. |
| [`generate_scene_views.py`](generate_scene_views.py) | Chooses two wall-aligned vertical cuts and writes matching cutaway views for the sealed mesh and Gaussian Splat model. | Existing Stage 08 view generator; requires the configured GPU renderer. |
| [`generate_floorplan_symbols.py`](generate_floorplan_symbols.py) | Produces dimension-constrained SVG symbols for the fixed 28-category vocabulary. | Standalone SVG utility. |
| [`project_valid_instances_by_floor.py`](project_valid_instances_by_floor.py) | Projects classified object instances onto fitted floor outlines and writes annotated PNG/PDF plans. | Optional utility; currently requires legacy floor-level and refined-shell inputs. |
| [`extract_floor_ceiling_residual.py`](extract_floor_ceiling_residual.py) | Builds a floor/ceiling residual evaluation mesh and projection reports from Stage 04/05 inputs. | Optional geometry diagnostic, not the Phase 06 delivery. |
| [`reproject_enhanced_cross_sections.py`](reproject_enhanced_cross_sections.py) | Reprojects enhanced section-image colors to a separate mesh copy using cross-section projection maps. | Optional post-processing. |
| [`match_real_captured_frames.py`](match_real_captured_frames.py) | Matches Stage 05 synthetic instance views to registered camera frames and writes comparisons. | Optional inspection artifact. |

Phase 08 outputs are collected under `backend/current_scene/artifacts/`. The macro steps
keep their products under `Generate Floor Plans/`, `Extract Best Views/`, `Generate 360 Pictures/`, `Generate Interactive Viewer/`, `Generate Cinematic Flythrough/`, and `Generate Possible Views/`;
other standalone utilities retain their documented per-product output paths. Existing outputs are
not moved automatically; rerun the relevant macro step to create its products under the new path.

## Macro step 1: Generate floor plans

Phase 08 is still in progress. Its first macro step is **Generate floor plans**; run it after Phase 07
alignment:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_floor_plans.py
```

This single command runs the complete **Generate floor plans** macro step directly after Phase 07.
It groups aligned floor planes into overall floor levels and assigns per-scan level names, then
writes regularized and sealed mesh cuts for each overall floor. Every cut retains all aligned scans;
a scan without a floor at that level carries forward its previous cut. It projects the sealed cuts
as direct-color top-down views and the regularized cuts as height-grayscale views.

The macro also writes a separate classified-object slice artifact, checks each categorized Phase 05
instance against its assigned regularized floor, and applies the room/category consistency rules to
produce final per-instance associations and Phase 08 reports. It uses Phase 05's saved per-instance
scores and dimensions; it does not rerun semantic segmentation or edit Phase 05 files. Floor-height
rejections and the WC-anchored room rule can move an instance to its next eligible prediction; the
single-class-per-scan constraint applies only to the selected fixture categories. The final
associations drive furnished regularized floorplans, including eligible Phase 07 door/window
panels. Last, it creates a reusable measurement overlay for each floor.

Within `current_scene/artifacts/Generate Floor Plans/`, the macro produces these floorplan variants:

| Variant | Artifact |
| --- | --- |
| Realistic, without measures | Sealed top-down PNG in `floor_level_projections/` |
| Regularized, without furniture or measures | Top-down PNG in `floor_level_projections/` |
| Regularized, with furniture, without measures | SVG in `floorplan_instance_overlays/` |
| Measurements for each floor | Transparent PNG in `floorplan_measurements/` |

Each `floor_<level>_measurements.png` has a transparent RGBA background, orange dimension spans,
and rasterized numeric labels. Its manifest records the reference projection grid and the
approximation notice; the viewer shows that notice in a hover- and keyboard-accessible tooltip
beside the measurements toggle.

It also saves the floor/scan plane metadata, both types of per-floor cut meshes, projection and
categorization reports, final instance associations, unmeasured furnished SVGs, and per-step manifests
under `current_scene/artifacts/Generate Floor Plans/`. `generate_floor_plans_manifest.json` in that
folder links the outputs and summarizes the run. Its macro status is complete on success;
`phase_status` remains `in_progress` because
Phase 08 has further macro steps to come. Measurement placement requires Chrome or Chromium on
`PATH`. `--alignment-dir` selects another Phase 07 output, and `--resolution` sets the square
projection resolution (default 1024 px).

The final per-instance data for the next Phase 08 step is in
`current_scene/artifacts/Generate Floor Plans/room_category_consistency/associations_after_room_check.json`,
with decision details in `room_category_consistency_manifest.json`. Each final instance row preserves
its Phase 05 association, per-category scores and report reference, plus the Phase 08 floor
measurement, robust bounds, floor assignment, and final category decision. Floor measurements and
the input floor assignments are preserved in `Generate Floor Plans/instance_floor_distance_check/instance_floor_distance_manifest.json`
and `Generate Floor Plans/classified_object_slice_experiment/classified_object_floor_slices_manifest.json`. Final
visuals are under `Generate Floor Plans/room_category_consistency/instance_reports/`; the combined macro-step manifest
at `current_scene/artifacts/Generate Floor Plans/generate_floor_plans_manifest.json` records all output manifests,
counts, and paths. The command logs the start and elapsed time of each substep; the manifest stores
each substep duration and the total macro duration in seconds.

## Macro step 2: extract_best_views

**Status: complete.** The current-scene run completed successfully for all three scans;
Phase 08 as a whole remains in progress while later macro steps are pending.

Extract a compact gallery of acquired photographs for each Phase 07 aligned scan:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/extract_best_views.py
```

A scan can include several connected rooms, corridors, or stairs. There is **no
maximum photo count**. The selector chooses an overview, adds images that reveal
useful new content, and removes redundant supplements. It stops at 80% attainable
weighted utility or when the best additional image contributes less than 10% of
the best single-image utility. This relative threshold scales with scan complexity;
low relevance areas may remain uncovered. Initial utility is 75% scene layout and
25% accepted semantic features, with category budgets shared among repeated
instances. Door, window, stairs, and rug categories receive one quarter of the
weight assigned to other predefined categories. Missing semantic evidence shifts
its utility budget to geometry. Layout regions saturate after one adequate view,
so unseen back sides and small extra surface fragments do not demand more photos.

Candidates are rejected for severe softness (contrast-normalized Laplacian energy
below 80% of that scan's median or raw thumbnail variance below 4), roll beyond
35°, or pitch outside −45° to +35°. When metric scale and nearby segmented floor
support are available, camera heights outside 1.20–2.25 m are also rejected;
otherwise the absolute height gate is skipped. Level, eye-height views remain
preferred. Floor-dominated (>90%), ceiling-dominated (>85%), mesh-projected nearby
object/unknown foreground (>40%), and known single-surface (>75%) framings are
excluded. These gates reduce unusable captures; they cannot turn the existing
photographs into landscape or retouched listing images.

When accepted scan categories include anything beyond doors and windows, every
selected image must adequately show at least one such categorized object (feature
adequacy ≥0.5). Door/window-only scans keep the existing image eligibility rule.
If no acquired image can meet the object requirement, that scan fails explicitly
and its previous gallery is preserved. Supplements are also skipped when at least
50% of the smaller image's visible acquisition-mesh samples are visible in any
already selected image. The report records qualifying categories, per-image
object visibility, overlap rejections, and the maximum selected-pair overlap.

The macro uses aligned COLMAP models in `multi_room_alignment/scans/<scan>/sparse/0/`
and verifies them against `.raw_unaligned/sparse/0/` and the saved Phase 07 scan
transform. Visibility is measured against `.raw_unaligned/mesh.ply` transformed into
the aligned world; later object-placement edits and replacement panels do not
describe the original photographs. Original acquisition geometry and source poses
are required. Supported calibration is pinhole or supported radial/OpenCV models
with zero distortion; unsupported/distorted inputs fail explicitly.
When sparse observations are available, a sampled reprojection check also rejects
internally inconsistent COLMAP models (median error above 1% of image diagonal or
more than 5% of observed points behind their cameras). Pose lineage alone cannot
detect a model whose source and aligned copies contain the same corrupted poses.

GPU mode projects and scores batches of sixteen cameras on ROCm/CUDA; Open3D performs
exact first-hit visibility queries for the in-frame surface samples using a shared
CPU BVH. The installed local AMD nvdiffrast build failed real dense-mesh depth
checks, so it is not used for production visibility. `--device cpu` runs projection
and scoring on the CPU as well; `--batch-size` controls batch memory use. `--alignment-dir`
selects another Phase 07 output, and `--output-dir` overrides the default
`current_scene/artifacts/Extract Best Views/`. Selection is deterministic for
fixed inputs and settings. No new view, crop, rotation, or image enhancement is
generated for the published photo subset.

Outputs live under `current_scene/artifacts/Extract Best Views/`. The aggregate
`extract_best_views_manifest.json` links each scan's byte-identical `images/` subset,
`selected_images.json`, `selection_report.json`, contact sheet, and camera plot.
Each image keeps its COLMAP IDs, calibration, aligned quaternion/translation and
both pose matrix directions in `phase07_aligned_world`, using Phase 07 scene units.
Reports include image quality, useful coverage, semantic priorities, stopping
reason, backend checks, and timings. A scan failure preserves its previous gallery
and is recorded in the macro manifest; the command exits unsuccessfully if any
scan fails. The command logs overall and per-scan start/completion, selected counts,
and elapsed seconds. The aggregate manifest stores total elapsed time, and each
scan report stores its stage timings and elapsed time. These records are a photo/pose
product rather than a filtered sparse reconstruction with observation tracks.

## Macro step 3: Generate 360 Pictures

Run after **Generate floor plans** and Phase 07 alignment:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_360_pictures.py
```

The macro processes actual scan-floor pairs from the first macro's mesh-slice manifest;
carried-forward scan contributions do not produce additional cameras. Attributed floor
element triangles determine candidate footprints, including holes and disconnected components.
Calibrated cameras start at 1.5 m vertically above their own floor and can lower in
20 cm steps through 0.7 m when coverage or obstruction requires it. Each trial uses one
height, and the best trial supplies the floor's panorama set. All heights must remain within 1 m in
3D of a registered same-floor acquisition camera, with a clear segment to that pose.
Uncalibrated scans derive one nominal height H for all their floors: half the smallest of
the levels' maximum floor-to-ceiling heights. Trial heights are H, 13H/15, ... 7H/15. Their
pose radius stays 0.66H regardless of lowering; auxiliary distances scale by H/1.5.
Ceiling source geometry determines the upper extent, with sealed level geometry as fallback.
Calibration status never prevents this geometry-derived policy.

Candidates sample the permitted region and source-camera horizontal locations. Registered
cameras are assigned to the highest attributed floor below them. Candidates cannot cross
the local ceiling or intersect a surface. Horizontal rays measure near obstructions and
their angular extent. Rays 5–55 degrees above the horizon measure nearby upper obstructions,
allowing lowered cameras under sloped roofs without penalizing proximity at the zenith or
nadir. Final candidates are rejected if their nearest horizontal hit is under 10 cm, their
horizontal 10th-percentile clearance is under 25 cm, or more than 50% of cosine-weighted
rays in the -15 to +55 degree band hit geometry within 75 cm. These thresholds scale for
uncalibrated units; they prevent coverage gains from admitting slab-dominated views.
Unobstructed locations retain the nominal height. Acquired-image visibility and
source/novel viewing-angle agreement provide a second quality score.

Coverage is the **number of visible triangles / number of current-level slice triangles**,
with equal weight per triangle, a **70% target**, and no floor-area metric. The macro-one
merge order and per-scan face counts recover exact scan ownership. A target triangle's
centroid must be on or above that scan's own attributed floor plane, allowing only a
0.1 mm numerical tolerance (scaled in uncalibrated units). This excludes ceiling/slab
triangles from the level below, even when macro one's 15 cm grayscale band retained them.
Carried-forward scans are excluded. The published coverage of selected panoramas uses an
exact first-hit raycast for every target triangle centroid against the complete sealed
per-scan mesh, including ceilings removed from floor slices. Hidden triangles remain in
the denominator. This centroid rule does not measure the visible fraction of a large
triangle and counts small triangles equally with large ones. To avoid full-mesh raycasts
for every candidate at every height, the height search uses one deterministic uniform
sample of up to 32,768 target triangles to rank candidates and estimate attainable coverage.
It exact-checks leading candidates one at a time and expands that set until the selected
panorama set reaches 70%. When the sample estimates that even the full candidate pool is
below 69%, the height trial checks only its first candidate before lowering. If no trial
reaches 70%, it exact-checks the best trial's full candidate pool before publication. The
report labels sampled candidate coverage and preserves exact per-candidate values for
candidates that were raycast in full; selected-set coverage stays exact.

At each height, a single eligible camera is preferred if it reaches 70%. Otherwise greedy
selection adds visible triangles, discounted for obstruction and poor source support;
additions below 0.5% of the total are omitted unless they reach the target. The height
search retries 20 cm lower when coverage misses 70%, when at least three selected cameras
lie within a 3 m neighborhood and each individually misses 70%, or when a selected view
still has more than 10% nearby obstruction in the measured view band. Neighborhood and
height distances scale for uncalibrated units; the original 3D acquisition-pose radius
never increases. Lower-only candidates still require evidence of physical obstruction
at nominal height, preventing lowering solely to expand pose support.

Trials are ranked by reaching the target, then fewer images (coverage first when all
trials miss the target), then less near-view obstruction, source quality and coverage.
Search stops on a clear target-reaching set or at the minimum height, preserving the
best earlier trial if lower views get worse. `height_search` records each trial, its
coverage, density, obstruction and lowering reason. Attainable coverage estimates the
union across all candidates using the same triangle sample unless that pool was fully
raycast; it is never used to renormalize the denominator. Tour links follow mesh-visible
paths through the permitted candidate region, check floor support, and report disconnected
components.

Each selected location produces matching 4096 x 2048 equirectangular JPEGs from the sealed
Gaussian model and vertex-colored mesh. `--width` selects another width divisible by four;
height is half the width. +Y is up, longitude zero faces +X, and longitude increases toward
+Z. The renderer loads Phase 07's filtered per-scan sealed Gaussian PLY, restores its
geometry and cameras to the source frame for SH evaluation, and preserves all filtering.
Six overlapping pinhole faces provide Gaussian RGB/alpha; chunked exact raycasts render
the mesh. The macro validates appended mirror and window triangles against the Phase 07
panel manifest and traces secondary rays from each visible panel pixel. Rays use all aligned
sealed scan meshes with every classified panel face removed, so windows can show geometry
captured in an adjacent scan. Mirrors reflect the room; windows blend 90% transmission, 5%
reflection, and 5% gray tint. If a transmission ray misses captured geometry, it uses the
configured missing-geometry environment color (white by default), rather than turning the
window into a second mirror. Panel classifications come from each scan's Phase 07 manifest;
the renderer has no scene-name or fixed-layout rules. This is an idealized single-bounce
effect: the GLB has vertex colors but no authored optical materials or lighting model.
No new renderer dependency is required.

Outputs under `Generate 360 Pictures/<scan>/<floor>/` include paired panoramas,
`selected_points.json`, `coverage.png`, and a contact sheet. Each point records height,
nearest supporting acquired pose, actual 3D distance, obstruction/source scores, and render
support. The aggregate `generate_360_pictures_manifest.json` records input hashes,
per-floor triangle coverage, counts and runtimes. Schema version 5 also records the panel
manifest hash and per-panorama optical pixel and secondary-hit counts. Each point records
its actual selected height. A completed floor below 70% is published as
`coverage_limited`; the aggregate status is `partial` and the CLI exits nonzero.
Execution failures are reported per scan and preserve already published floor outputs.
`--alignment-dir` and `--output-dir` follow the other Phase 08 macro conventions.

The command streams timestamped progress logs to the terminal and writes the same run to
`Generate 360 Pictures/generate_360_pictures.log`. The aggregate manifest records the total
elapsed time plus setup, scan-processing, and summary timings. It also records timings for
each scan and floor, including floor-output publication. Per-floor `selected_points.json`
contains timings for floor sampling, target filtering, height search and visibility, tour
diagnostics, and panorama rendering; the final report write and publication time is recorded
in the aggregate manifest. Failed floor stages are timed and logged as well.

A pre-optics measured run of the current scene (2026-11-09) completed in **66.38 s** across three
scans and four scan-floor pairs, producing four Gaussian/mesh panorama pairs at 4096 x 2048.
Exact triangle coverage was 78.43% (scan_0/BASE), 74.82% (scan_0/BASEa), 81.69% (scan_1/BASE),
and 85.11% (scan_2/BASE). This is 23% faster than the previous recorded 86.07 s run on this
scene, which preceded the 20 cm height schedule and the coverage-map and Gaussian-rendering
optimizations. The aggregate manifest and log remain the source for timings from each run;
total runtime varies with the scene and hardware.

## Macro step 4: Generate Interactive Viewer

After completing the first three macro steps, generate the tour and static comparisons:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_interactive_viewer.py
backend/.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory 'backend/current_scene/artifacts/Generate Interactive Viewer'
```

Open `http://127.0.0.1:8765/` on the same machine. The preview server binds only to loopback;
stop it with Ctrl-C when finished to release port 8765.
`--scene-dir` selects a different scene. The viewer copies its images and uses relative paths;
it does not require the original artifacts at runtime or access to the internet. It requires WebGL
for panoramas. The generator validates completed upstream manifests, the photo transform hash,
projection geometry, and floor support; a missing floor under a photo is an error.

Camera order is numeric scan ID. For each later scan, if Phase 07 has exactly one aligned door
panel contributed by that scan and its predecessor, the photo or 360° viewpoint nearest the panel
center by absolute 3D distance is first. The remaining cameras on all floors follow absolute 3D
distance from the scan's earliest timestamped capture pose. If there is no unique valid paired
door, the scan uses capture-start distance for all its cameras. Photos and 360° viewpoints count
equally, and numbering starts at 1 and continues across scans. The tour, static image labels,
static image paths, and camera markers use that same sequence. Interactive and exported plan and
minimap markers display only camera numbers; scan and floor identifiers come from the input
manifests. Plans use each projection's own
saved grid; photo directions use aligned COLMAP poses and horizontal calibration. Panorama
longitude zero is +X and positive longitude is +Z. The minimap is cropped to the current
scan's attributed floor geometry. The floor plan's **Show measurements** control starts off and
toggles the selected floor's transparent PNG over all three styles; the viewer maps it into each
style's projection grid. Measurements never appear on interactive or static minimaps.
For display, the viewer vertically flips copied plan images (top to bottom) and applies the same
pixel transform to camera positions and measurement geometry; measurement labels stay upright. This
puts world +Z downward on screen, so static photo cones and the live 360° minimap cone negate yaw
when drawing their screen-space direction. The source floor-plan artifacts are unchanged.
`static/plans/<level>/` contains each of the three camera-marked plans both without measures
(`<style>.png`) and with measures (`<style>_with_measures.png`). Measurement overlays are aligned
to each style's saved projection grid. `static/<scan>/<number>_<kind>/` contains one static image,
with the Standard floor plan in its minimap (`standard.jpg`).

The manifest links each copied camera image to its source, the three upstream macro manifests,
and the measurement manifest. `tour.json` is fetched without cache, and camera-image and viewer
script URLs include content hashes so regenerated assets cannot reuse stale browser-cache entries. Execution
logs are written to
`generate_interactive_viewer.log`. Click the minimap to return to the full floor plan; choose the
plan style there or right-click a plan or minimap. Side arrows navigate between cameras. The
controls support keyboard, mouse, and touch; panoramas open at the widest supported view and use
drag, arrow keys, or zoom buttons. **Show mesh** switches between the paired Gaussian and mesh
panoramas without changing the viewing direction or zoom; the selection carries across panoramas.
Mesh panoramas simulate reflections and transmission for classified Phase 07 panels. Static camera
images use the Gaussian panorama. A brief movement and label
identify a newly opened panorama; the movement is skipped when the browser requests reduced motion
and stops on interaction.

## Generate Cinematic Flythrough

Run after the viewer and a Phase 07 run that retained its trajectory-pairing report:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_cinematic_flythrough.py --mode preview
backend/.venv/bin/python backend/08_artifacts_generation/generate_cinematic_flythrough.py --mode final

# Or run both passes continuously with in-memory model reuse:
backend/.venv/bin/python backend/08_artifacts_generation/generate_cinematic_flythrough.py --mode both
```

Inspect the paired videos and `preview/representative_frames/` between commands. The final command
refuses changed inputs or trajectory data. The output directory is
`artifacts/Generate Cinematic Flythrough/`; preview refuses to replace an existing directory.
Use `--scene-dir` or `--output-dir` for another scene or a separate run.

Older Phase 07 outputs may lack the deleted pairing report. Reconstruct the identities from current
inputs without modifying segmentation, then pass the resulting provenance file to the preview:

```bash
backend/.venv/bin/python backend/07_alignment/recover_trajectory_pairings.py \
  --workspace backend/current_scene --output /path/to/new/reconstructed_pairings.json
backend/.venv/bin/python backend/08_artifacts_generation/generate_cinematic_flythrough.py \
  --pairings /path/to/new/reconstructed_pairings.json --mode preview
```

Reconstruction compares saved relative rotations and aligned-panel source identities, but cannot
prove that inputs still match the original Phase 07 run. A missing or ambiguous connection fails
with the affected checkpoint numbers rather than reordering or cutting the tour.

The generator follows the viewer's published numbers, passes through each saved photo pose without
stopping, and passes within 31 cm of each panorama. It uses a fixed 70° horizontal field of view at
30 fps. Between photo poses, it chooses viewing directions using mesh visibility and a preference
for the direction of travel, while fully preserving cinematic sideways or backward viewing when
open-space geometry warrants it.
Each panorama viewpoint receives at least a 240° turn during the moving flight; the viewing direction
is chosen using scene visibility. When a panorama lies on a different floor from an adjacent
checkpoint in the same scan (such as mezzanine level `BASEa`), localized velocity pacing ($0.22\text{ m/s}$)
is applied across its level footprint so that the turn is concentrated near that viewpoint and validated
to cover at least 180° continuously while the camera remains on that floor. Doorway passages are
centered comfortably through door openings with fallback safety offsets if obstructions are present.
When nearby surfaces dominate a panorama, a bounded yaw adjustment can reduce a three-ray
near-surface penalty without increasing it over the full adjustment window, moving photo poses, or
adding significant rotation. The complete trajectory is revalidated after the adjustment; the
original view is retained if validation fails.
This geometry heuristic does not identify objects or establish semantic coverage.

### Trajectory, velocity scheduling, and motion

All participating scans must have a Phase 01 `metric_scale.json` recording that metric scale was
available and applied to the geometry. Calibrate upstream and regenerate dependent artifacts if it
is missing or uncalibrated; this step does not infer or apply a scale.

The planner tries bounded alternatives when a paired door or scan route fails. If its coarse grid
cannot connect a leg, it retries with 6 cm grids around that connection at margins of 0.6, 1.2,
and 2.4 m, then tries a scan-wide fine grid if its cell limit permits. It compares at most three feasible scan routes using a heuristic
for length, clearance, turning, and door crossings; the selected route is not necessarily globally
optimal. If smoothing, retiming, ascent planning or sampled validation fails, it tries combinations
of retained alternatives, ordered by their spatial scores, up to 16 full trajectory attempts.
Retiming uses an F1-inspired curvature-aware velocity scheduler:
- Peak cruise velocity: **1.25 m/s** (configurable via `speed`).
- Apex velocity: $v_{\text{apex}} = \min(v_{\text{max}}, \sqrt{a_{\text{lat}} / \max(\kappa, 10^{-8})})$, holding a steady, fast pace through turns.
- Asymmetric braking and acceleration: $a_{\text{brake}} = 1.10\text{ m/s}^2$ creates a crisp braking zone before corners via backward DP pass; $a_{\text{accel}} = 0.95\text{ m/s}^2$ accelerates out of turns via forward DP pass.
- Door transit speed: paced up to $\min(v_{\text{max}}, 1.25\text{ m/s})$ for snappy, centered door passages.
- Panorama pacing: paced down to $\le 0.35\text{ m/s}$ ($0.22\text{ m/s}$ on floor transitions) during the $\ge 240^\circ$ continuous panorama sweeps.

Sampled interior acceleration and turning/cruise angular speed are checked against 1.21 m/s²,
65.1°/s and 37.6°/s respectively; the overall angular speed limit is 120°/s. The manifest reports
measured speed, jerk, acceleration, angular speed, clearance, panorama turns, and each door's speed
and duration. Photo poses are checked to 1e-6. A turn of $\ge 240^\circ$ is validated at each panorama;
where a same-scan neighboring checkpoint is on a different floor, the turn is also checked on the
panorama's floor. Camera framing uses geometry and does not establish semantic scene coverage.

The trajectory routes inside supported floor geometry, validates the sampled path against the sealed mesh,
and allows panel-shaped intersections only at designated paired doors when the hit also matches a source door-class face.
After a Phase 07 split, source face labels may identify the original instance rather than its derived child.
Long clear stretches curve where clearance permits. The last panorama is followed by an ascent through validated
horizontal ceiling surfaces to a scene-wide overhead view; terminal panoramas seamlessly continue their sweep into
the ascent ($+60^\circ$) before tilting downward. Doors and ceiling are rendered from the actual moving camera.
Sealed surfaces can temporarily fill the view during a crossing. The Standard minimap uses the
viewer’s flipped plan images and world mapping.
It shows the nearest supporting floor below the camera; inside an unsupported door gap or during
the overhead ascent it retains the previous floor. Photo and panorama markers are revealed when their
checkpoints are passed; route and revealed-marker state persist across floor changes.

### Performance and measured benchmarks

Key optimizations accelerate the pipeline:
1. **Native GPU Hardware MSAA ($4\times$)**: Configured Open3D Filament's offscreen view with native $4\times$ multisample antialiasing and antialiasing enabled, completely eliminating CPU-side 2× supersampling and `cv2.resize` bilinear downsamping. Native frame rendering runs at $3.59\text{ ms/frame}$ (a **3.05× speedup** over CPU downscaling).
2. **Asynchronous Non-Blocking Video Encoding (`AsyncVideoWriter`)**: Video pipe writes (`os.write` to ffmpeg stdin) are offloaded to a dedicated background worker thread with a bounded queue (`maxsize=16`), completely decoupling OpenGL/CUDA rendering from video encoding pipe bottlenecks ($0.12\text{ s}$ preview / $0.58\text{ s}$ final IO overhead, down from ~11 s).
3. **In-Memory Model Caching Across Passes (`--mode both`)**: Preloaded GLB mesh geometry arrays and the resolution-independent PyTorch `GaussianRenderer` are held in memory across preview and final passes, reducing duplicate disk re-loading from $12.36\text{ s}$ down to $0.62\text{ s}$ (**19.9× faster**).
4. **Fast SE(3) Matrix Inversion & CUDA Tensor Caching**: Replaced general $O(N^3)$ LU matrix inversions (`np.linalg.inv`) with closed-form rigid Euclidean camera pose transpositions ($R^T$ and $-R^T t$), and cached GPU intrinsic and background tensors on device.
5. **Direct NumPy Minimap Composition**: In-place array slicing and assignment in `compose_minimap()` without unnecessary full-frame copies, keeping HUD overlay time to $< 0.8\text{ s}$ for 1,911 frames.
6. **Floor-First Clearance Pruning & Raycast Early Exit**: Downward floor raycasts are evaluated first in `clearance_and_support` to immediately discard out-of-bounds cells, pruning 60%–85% of full-mesh distance queries during grid initialization.

A measured end-to-end run on `current_scene` (1,911 trajectory frames, 63.7s video, 3 scans, 4 floors, 4 panoramas) achieved:
- **Preview Pass (270×480)**: **52.89 s** (Input validation & setup: 0.9s, Planning & retiming: 9.3s, Model loading: 11.9s, Mesh: 19.87s, Gaussian: 9.35s, Minimap: 0.38s, Async Encoding IO: 0.12s).
- **Final Pass (540×960)**: **47.49 s** (Verification: 0.7s, In-memory model reuse: 0.62s, Mesh: 34.59s, Gaussian: 9.51s, Minimap: 0.81s, Async Encoding IO: 0.58s).
- **Combined Active Wall Time**: **100.38 s** (~1 min 40 s) for 3,822 frames (7,644 total video frames across mesh and Gaussian channels), a **33.82 s (25%) speedup** over the 134.20 s baseline.

Preview and final hash all input assets, calibration, code, and encoders; inputs are verified before and after each render.
The manifest records structured atomic timings, validation metrics, and trajectory hashes.
The videos are silent, 270×480 previews and 540×960 finals.

## Top-down and elevation views

After Stages 06 and 07, render views from the scan mesh and the Stage 07 alignment report:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/extract_cross_section_views.py \
  --mesh-path backend/current_scene/scan_0/mesh.ply \
  --instances-dir backend/current_scene/scan_0/mesh_elaboration_steps \
  --output-dir backend/current_scene/artifacts/cross_sections/scan_0
```

This writes `cross_sections_manifest.json`, one `topdown_floor_*.png` per floor manifest it finds, and four elevation images. Pass `mesh_elaboration_steps/` when its floor and ceiling manifests are available. The current `run_phase06.py` removes its temporary manifests after publishing `complete.ply` and `elements/`; without a retained level manifest, the view extractor falls back to one floor at the mesh's lower bound. It can also write per-pixel triangle/barycentric projection maps with `--projection-map-dir` for the optional enhanced-image reprojection pass.

## Floor and scan level metadata

Run the metadata-only classifier after Phase 07:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/assign_floor_level_metadata.py \
  --alignment-dir backend/current_scene/multi_room_alignment
```

When invoked standalone, it writes `current_scene/artifacts/floor_level_metadata.json`; the macro writes it under `current_scene/artifacts/Generate Floor Plans/floor_level_metadata.json`. It produces no rendered output. It reads the aligned per-scan Phase 06 `elements/` meshes so each plane retains its scan ID; the merged `aligned_regularized_meshes.ply` has no per-face scan ownership. Horizontal support in floor elements and the upper side of slabs are floors; slab undersides and ceiling groups are ceilings. A new height group starts when a plane is more than 0.5 m above the group's lowest plane. Each scan's lowest floor inherits its scan-level name (`BASE`, `+1`, ...); additional floors in that scan add letters (`BASEa`, `+1a`, ...). A ceiling inherits the name of the nearest floor plane below it in its scan.

## Per-floor cut meshes

After assigning floor metadata, generate the aligned cut meshes:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_floor_level_meshes.py \
  --alignment-dir backend/current_scene/multi_room_alignment
```

Standalone output is written to `current_scene/artifacts/floor_level_meshes/`; the macro writes under
`current_scene/artifacts/Generate Floor Plans/floor_level_meshes/`: one `floor_<name>_regularized.ply` and one
`floor_<name>_sealed.ply` for each overall floor name. Both outputs contain every aligned scan. A scan without
that floor carries forward its previous regularized and sealed geometry. If it has no floor at the current
level, its most recent lower-floor cut is carried forward and the entire carried scan component is grayscale.

The hard ceiling cut is offset 20 cm along the fitted plane normal, so tilted planes keep the same
clearance; crossing unowned triangles are clipped. The regularized ceiling element is removed (for
a slab, its underside plane triangles), and corresponding sealed ceiling surfaces are selected from
the fitted plane's finite support region. Sealed PLYs do not contain face-level object IDs, but Phase
05 retains separate object and structural-wall instance PLYs. Phase 8 maps those source triangles
back onto each aligned sealed mesh using its saved scan transform. Within 70 cm below the fitted
ceiling, a triangle is included when any vertex falls within the band: object and wall triangles are
kept, including triangles inside the 20 cm hard-cut zone; unassigned triangles outside the wall
border are removed. This avoids ray-based deletion of
furniture surfaces while clearing unclassified ceiling remnants. Faces whose centers lie more than
15 cm below the overall floor-level reference plane are converted to grayscale in both generated
and carried-forward scan components. Elongated-triangle removal applies only to sealed meshes: the
longest-edge-to-altitude ratio limit is 20 generally and 5 within 1 m below the fitted ceiling. The
manifest records ownership matches, removed and protected triangle counts, generated versus
carried-forward scans, selected planes, and output paths.

The script reads per-scan aligned `mesh_elaboration/complete.ply` and `mesh.ply` files to retain
scan-specific plane ownership while producing merged per-level outputs. If a scan has no ceiling
above a floor, its newly generated contribution is left without an upper cut and the manifest
records a null ceiling plane.

Render top-down projections from those floor meshes:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_floor_level_projections.py
```

The script reads every level and mesh path from `floor_level_meshes_manifest.json` and writes
two images per overall floor to `current_scene/artifacts/floor_level_projections/` when run standalone,
or `current_scene/artifacts/Generate Floor Plans/floor_level_projections/` as part of the macro. Sealed
projections show the topmost visible mesh color; regularized projections encode top-surface
height in grayscale, with surfaces closer to the projection plane darker and lower surfaces
lighter. Hit pixels use intensities 32–240 against a white empty background; each PLI normalizes
depth across its own vertical mesh range. Two-by-two supersampling reduces small ray-sampling
gaps. Each projection derives each scan's contour from that same per-floor PLI mesh. In sealed
projections those mesh contours also define the fill boundary; ray misses and near-white pixels
(RGB ≥250) inside those boundaries are filled with Telea interpolation from nearby rendered
colors, using a 9-pixel radius. A 41-pixel closing is applied to each scan footprint separately
before combining them, so empty gaps between scans remain white. A 5-pixel black outline is drawn
afterward. Regularized projections keep their height shading; enclosed raster holes smaller than
0.005 m² are interpolated only inside an individual scan, while open gaps between scans stay empty.
The combined footprint gets a 5-pixel outer outline. Scan interfaces, larger enclosed holes, and
adjacent occupied pixels with a grayscale difference of at least 16 get 2-pixel outlines.
Sealed gap filling only changes the projection image and does not add
geometry to the meshes. The square orthographic view is centered on each mesh's
horizontal bounds. Resolution defaults to 1024 pixels and can be changed with `--resolution`.

## Classified-object slice experiment

Add aligned objects with valid fixed-class associations to separate copies of the regularized
floor meshes:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_classified_object_floor_slices.py
```

Phase 05 object-role classes and fixed-class wall instances copied into the object export are
considered. Each aligned instance is assigned to exactly one generated floor slice: the slice with
the greatest vertical overlap at the object's X/Z bounding-box center. The full object is retained
in that slice, even when it extends beyond the cut, and its manifest records overlap scores for all
candidate levels. Each object keeps its aligned mesh colors and receives a magenta 3D axis-aligned
box with its recognized category on a small top-face plaque. The script writes augmented PLYs,
top-down previews, and an assignment manifest under
`current_scene/artifacts/classified_object_slice_experiment/` when run standalone, or under
`current_scene/artifacts/Generate Floor Plans/classified_object_slice_experiment/` as part of the macro; source floor meshes and projections
are not modified. Use `--mesh-dir`, `--output-dir`, or `--resolution` to point the experiment at
alternate inputs or output locations.

## Instance floor-distance check

Check accepted Phase 05 per-instance category predictions against the floor level already
assigned to each aligned instance by Phase 08:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/check_instance_floor_distance.py
```

Each report reuses the Phase 05 instance-view collage and shows that instance's category scores
on the right. The per-instance prediction is used, including `instance_text_class_id` when it
differs from a class-propagated label; class-level labels are not used. Door/window instances and
instances without a valid per-instance category are skipped. Green and red mark the selected
category's floor-distance result; other score candidates stay neutral. The Phase 08 slice
manifest supplies the existing floor assignment, and the associated floor plane is used to
measure the robust bounding box's bottom height at its horizontal center. The box allows up to
10% of triangle surface area outside it to reduce mesh outliers. Only the category's `base_z_m`
range is checked; all categories allow a base height down to −0.20 m, while chair allows up to
0.40 m, table 0.90 m, rug 0.20 m, sink 1.15 m, and refrigerator 0.20 m. The visuals also show
the robust box dimensions beside their category dimension ranges for review; dimensions do not
affect acceptance. No class-level aggregation is performed. Instances without a usable Phase 08
floor assignment still get a report marked not evaluated. Reports and the manifest are written
under `current_scene/artifacts/instance_floor_distance_check/` when run standalone, or under
`current_scene/artifacts/Generate Floor Plans/instance_floor_distance_check/` as part of the macro; the script replaces only its own
`instance_reports/` output folder. `--scene-dir` and `--output-dir`
can select another scene or artifact directory.

## Room category consistency

Apply room-context rules to the per-instance associations retained by the floor-distance check:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/check_instance_room_consistency.py
```

Each scan is treated as one room. A room with at least one floor-retained WC is a bathroom, where
sofa, bed, cooktop, refrigerator, oven, and dishwasher predictions are rejected. Without a
floor-retained WC, WC, bidet, and shower predictions are rejected. A rejected prediction moves to
its highest-scoring positive Phase 05 alternative that is dimension-valid, room-compatible, and
within its base-height range; if no unambiguous alternative exists, it becomes unassigned. The
check then enforces one source class per WC, washing machine, refrigerator, oven, dishwasher,
shower, and bidet within each room: the source class with the highest individual instance score
owns that category, and multiple instances from that winning class may retain it. Other categories
are unrestricted, and instances of the same source class may have different category labels.
Conflicting instances try their next valid alternative, subject to ownership. Phase 05 already
filters candidate scores by instance dimensions; final per-instance reports show the instance
AABB, selected category's dimension limits, and floor-height result. The script writes
`associations_after_room_check.json`, a manifest, and reports every floor-checked prediction,
including floor-distance rejections, under
`current_scene/artifacts/room_category_consistency/` when run standalone, or under
`current_scene/artifacts/Generate Floor Plans/room_category_consistency/` as part of the macro; it does not edit Phase 05 or the
floor-distance outputs.

## Categorized instance floorplan overlays

The canonical `generate_floor_plans.py` run creates one additional SVG view per overall floor
after the final room/category check. Each SVG rebuilds the regularized projection from vector
grayscale regions and contour paths derived from that floor's regularized PLI; white projection
pixels remain empty. Category symbols are placed in the same 2D coordinate system at the final
instances' robust X/Z bounding-box locations and dimensions. Final Phase 08 category decisions
are used as-is; SVG generation does not repeat category or dimension validation. The original
projection PNGs and all earlier mesh, projection, and association outputs are left unchanged by
this overlay stage; the aggregate macro manifest is extended with the new output path and counts.

Instances assigned to the displayed floor are opaque. Lower-floor instances from the same scan
level are shown whole at 35% opacity only when aligned floor-plane coverage occupies no more than
10% of their footprint; otherwise the whole symbol is hidden. Lower-floor doors use their full swing
envelope for the floor-coverage check and are hidden when it is over 10% covered or overlaps any
current-floor symbol. The aligned Phase 07 doors and windows use floorplan SVG symbols snapped to
nearby parallel segments from the regularized PLI contours (up to 25 cm). Segment matching considers endpoint fit
and width; if a panel is wider than its matched segment, it is shortened to fit. Panels are not moved
by the furniture layout pass. If no matching segment is found, the Phase 07 panel position is
retained. The scan associated with the most door panels is treated as the central room; its doors
swing away from that room into the adjacent space.

For an assigned-floor door sourced from two scans whose regularized footprints remain disconnected,
the overlay fills the gap between their nearest footprint edges with a passage exactly as wide as
the fitted door segment. It only bridges where each scan occupies the opposite side of that door;
already touching footprints are left intact. The passage extent is limited to 1 m from the door
plane and its grayscale is interpolated from the adjacent projection. Per-level passage geometry
and source scans are recorded in the overlay manifest. Both bridge-end lines are overpainted with
interpolated floor grayscale after contour rendering; the door symbol is drawn afterward to supply
the single closed-door line across the opening.

Symbols are kept inside their own scan's projected contour, with contour fitting limited to 25 cm.
Each instance's position is resolved on its assigned-floor view and reused unchanged in higher-floor
views. Lower-floor symbols therefore stay at the same world X/Z location; they act as fixed overlap
blockers there, so overlapping current-floor symbols are moved or suppressed instead of shifting the
lower-floor symbol. The manifest records each displayed symbol's final world-space X/Z center.
Same-category instances in the same scan and floor are merged when their projected boxes touch,
overlap, or have at most a 5 cm gap; the merged SVG uses their union footprint and the manifest
retains all source instance IDs. A non-panel symbol within 25 cm of its scan contour is moved
toward that contour as far as its scan footprint allows. If an upper object covers at least 85% of
a lower object's footprint and their vertical bounds are separated, the lower object wins and the
upper one is suppressed. Remaining non-rug footprint overlaps are resolved by moving candidates
up to 40 cm while keeping them inside their contour; fixed fixtures are placed before movable
furniture. Instances that cannot be placed without overlap are omitted and listed in the manifest.
Rugs remain beneath other symbols, with transparent space around each symbol's actual SVG geometry.
WC tank/back edges use the shorter footprint dimension; bed, sink, bidet, and shower face the
nearest scan contour, chair backs use the longer footprint dimension, and stair arrows follow the
triangle-density-weighted height gradient. SVG strokes use a thicker minimum line weight for
legibility at floorplan scale.
The aggregate symbol count includes both categorized instances and Phase 07 panels.
The `floorplan_instance_overlays_manifest.json` is written under
`current_scene/artifacts/Generate Floor Plans/floorplan_instance_overlays/` by the macro; a standalone
run uses `current_scene/artifacts/floorplan_instance_overlays/`. It records displayed and suppressed
instances, including objects fully covered by the current floor, and each reason; its furnished
SVG output paths point to the unmeasured views.

The macro retains the furnished SVG in `floorplan_instance_overlays/` and writes one transparent
measurement PNG per floor in `floorplan_measurements/`. It renders the furnished view using headless
Chrome or Chromium to choose label positions that avoid existing marks.
Orange dimension spans and numeric labels measure the straight segments of that floor's regularized
PLI wall contours; furniture and door/window symbols are not measured. Labels use metres to two
decimal places, and segments shorter than 10 cm are omitted as contour noise. The viewer displays
the approximation notice as manifest metadata and shows it in a question-mark tooltip beside the
measurement toggle. The notice is not rendered in the measurement PNG. Annotation
placement checks nearby plan marks and other dimensions, choosing the clearer inside or outside side
of each wall segment. A segment facing another scan within 70 cm is always placed inward, so its
annotations cannot spill into the gap or neighboring scan. The measurement manifest records each
span, side, and displayed length.

## Other view artifacts

`generate_scene_views.py` takes a current-scene root or one scan directory. It requires both `mesh.ply` and `gaussian_splat/checkpoint_sealed.pt`, plus a working CUDA/ROCm-compatible renderer, and writes its cut-plane report and four corresponding views under `current_scene/artifacts/scene_views/<scan>/`:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_scene_views.py \
  --workspace backend/current_scene/scan_0
```

The real-camera comparison utility reads the Phase 05 mesh-frame manifest and writes only inspection artifacts under `current_scene/artifacts/real_frame_matches/<scan>/`:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/match_real_captured_frames.py \
  --workspace backend/current_scene/scan_0
```

Generate a single fixed-class SVG symbol (class IDs match `run_non_structural_category_similarity.py`):

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_floorplan_symbols.py 18 1.8 2.0 \
  --output backend/current_scene/artifacts/floorplan_symbols/bed.svg
```

## Floorplan data dependencies

`project_valid_instances_by_floor.py` consumes the Stage 05 associations at `segmentation/non_structural_similarity/valid_class_associations.json`, but it also requires `structural_levels_manifest.json` and a refined structural mesh from the older standalone floor-level workflow. The current `run_phase06.py` publishes `mesh_elaboration/complete.ply` and `elements/` only, so those older plan inputs must be supplied explicitly; this utility is not run automatically by Stage 08. Its default output is `current_scene/artifacts/floorplan/instance_projections/`.

`extract_floor_ceiling_residual.py` requires explicit source mesh, labels, structural metadata, fitted-wall inputs, and a new output PLY path; place it under `current_scene/artifacts/floor_ceiling_residual/`. It reads the same Stage 05 association artifact when available. Its result is an auxiliary evaluation mesh, not a replacement for `mesh_elaboration/complete.ply`. `reproject_enhanced_cross_sections.py` also takes an explicit `--output-dir`; use `current_scene/artifacts/reprojected_cross_sections/`.

## Generate Possible Views

Run `backend/.venv/bin/python backend/08_artifacts_generation/generate_possible_views.py`
after Generate Floor Plans and Phase 07. The macro clips each actual per-scan
floor to its floor and associated ceiling. Adjacent levels of the same scan meet
at the lower level's ceiling plane, retaining the mezzanine floor and slab rather
than leaving a clearance gap. Scan faces within 10 cm of their floor plane and
whose normals are within about 26° of the plane normal (either winding) are
protected; aligned Phase 7
floor elements fill holes in the scanned floor. Where both are
present, the scanned floor colors take precedence. The first floor allows 5 cm
below its floor plane; a missing ceiling falls back to the mesh upper bound.

Scan slices are assembled once, then scans render concurrently on CPU (one worker
per represented scan); the GPU is not used because this environment's Open3D
raycasting runtime has no GPU device. The full-size view render stays at the
requested resolution, while the cutaway seed pass is capped at 3072 px. Compared
with full-resolution cutaway sampling, this changed at most 0.08% of pixels per image;
all 12 selected view IDs and each slice's coverage-target outcome were preserved.
Geometry and provenance NPZ archives are written without ZIP compression to avoid
their measured compression bottleneck; view pixel-ID and cutaway archives remain
compressed. On 2026-11-10, the 4096-pixel macro completed in 64.9 s wall time
(63.2 s in the manifest), 17.4% faster than the prior 78.6 s wall run, with
identical output to the measured optimized prototype. The artifact directory grew
from 172 MB to 392 MB. The manifest records the cutaway sampling resolution,
archive compression, summed per-scan render time, and concurrent `scan_work_wall`.

Each scan is rendered independently, combining its actual floor slices and framing
four orthographic cameras at that scan’s bounding-box corners. View selection runs separately for each
scan, while coverage remains reported per floor slice. For each camera, the renderer identifies first-hit triangles whose
normals point along the viewing direction, removes that visible back-facing layer,
and removes nearby triangles within 15 cm in 3D. This applies to walls, ceilings,
and other first-hit back-facing geometry, except the protected floors. Triangle
proximity is conservative:
subdivision bounds uncertainty to 5 mm before an entire intersecting triangle is
removed, so long triangles can extend farther than the radius. The rule depends
on the source mesh's triangle winding; it can also remove objects that are close
to a removed wall. Images and coverage are computed from the same camera-specific
cutaway. No source mesh is modified.

Phase 07 replacement-panel records identify mirrors and windows in the scan mesh.
The macro validates their faces against the recorded corners, then casts one reflected
ray for a mirror or one transmitted ray for a window, with transmitted light, a partial reflection, a light gray tint, and a
contrasting edge. For each view, a panel is removed only when more than half of its non-panel
triangle neighbors within 30 cm were removed by that view’s cutaway; otherwise it
is retained. Only retained geometry in the same scan can appear in these effects; missing
geometry is white. The current
GLB contains vertex colors but no optical material values, so these are idealized
surfaces rather than measured material properties. Pixel provenance identifies the
panel itself; secondary hits do not add surface coverage. The manifest records panel
classes and visible optical pixel counts per candidate.

Outputs are staged and published under `artifacts/Generate Possible Views/`.
The manifest links candidate images, per-pixel composite face IDs, removed and
first-layer face IDs, per-slice reports, and selected views. Each `scan_levels/`
slice contains uncompressed NPZ arrays for vertices, faces, vertex colors, source face IDs,
source-mesh indices, and the floor-protection mask. The indices select the
slice's `source_meshes` list, which contains the aligned per-scan `mesh.ply`
and its aligned floor elements; split faces retain their source IDs. Contact sheets include all four candidates and the selected set for each scan,
plus a combined sheet of selected views. The default 4096-pixel render counts a face only if at least two
pixels hit it. Selection uses area, a 70% per-slice target, and stops when
marginal coverage is at most 10% of the best first view. `--resolution`,
`--coverage-threshold`, and `--minimum-gain` change these settings;
`--alignment-dir`, `--metadata`, and `--output-dir` select inputs and output.

Per-slice coverage uses actual visible scan faces and excludes the supplemental
fitted floors from its area totals to avoid counting overlapping floor area twice.
Greedy selection uses the visible face areas within each scan. The manifest
reports attainable coverage and an individual stopping reason for each slice.
