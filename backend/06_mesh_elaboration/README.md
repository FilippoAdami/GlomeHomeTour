# Structural mesh elaboration

Run all five stages from the repository root with explicit inputs and an isolated output:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/step_mesh_elaboration.py \
  --mesh-path backend/current_scene/06_mesh_elaboration/original_mesh.ply \
  --labels-path backend/current_scene/06_mesh_elaboration/generic_graph_regions/triangle_class_labels.npy \
  --structural-json backend/current_scene/06_mesh_elaboration/generic_graph_regions/structural_classes.json \
  --output-dir backend/current_scene/06_mesh_elaboration/per_level_collapse_candidate
```

The extractor assigns unique role-based IDs (`floor_inst000`, `ceiling_inst000`,
`wall_inst000`); downstream stages follow manifests rather than scene names or fixed
coordinates. Finite wall planes and their Manhattan frame come from measured support.
Footprints come from the full source scene collapsed within each level's height band,
using projected triangle area rather than semantic floor/ceiling contours. Semantic planes
supply local height limits and identify connected level components. The cleaned contour is
sampled every 1 cm; a constrained squared-distance fit jointly optimizes orthogonal line
offsets with segment lengths >=10cm. Simpler valid topologies are tested for lower error.
Wall snapping and small-hole/overhang cleanup follow fitting, before slab synthesis.
Paired ceilings inherit the fitted upper-floor contour; ceiling contours are not unioned
back into slabs. For unpaired roofs, full-scene ceiling-facing support bounds coverage,
so a floor patio does not fabricate a roof. The numerical optimizer uses local refinement
and topology removal; it does not certify a global minimum over every possible polygon. Level reporting accepts arbitrary floor counts. Floor-only and wall-only partial scans are supported. Unsupported multi-slope
surfaces and significant non-Manhattan wall support retain source geometry.

`plane_patterns_manifest.json` lists individual surfaces, planes, footprints, patterns,
and closed `slab_instNNN.ply` artifacts. `all_structural_regularized.ply` uses slabs in
place of duplicate overlapping surfaces, retains non-overlap remnants and stairs.
`structural_levels_manifest.json` reports measured elevations and overlapping ceilings.
`level_footprints_manifest.json` records source mesh, selected triangle counts, cleaned
contours, fitted outlines, pre/post-fit distance scores and final wall-constrained scores.
`floor_instNNN_rough_collapsed.ply` exports the full-scene per-level collapse for inspection.
`--scene-mesh` supplies the source to the regularizer; otherwise it follows the extractor's
`source_mesh`. Missing source evidence fails explicitly rather than reverting to semantic
contour fitting. `step06_timings.json` records all five stages. The driver shares imports in one process;
individual stage CLIs remain available.

The current per-level-collapse candidate is separate from the accepted
`extracted_structural_instances/` checkpoint. Its estimated roof angle and beam dimensions
have changed; it has not been certified visually equivalent to that checkpoint. Compare
`agents_files/artifacts/per_level_collapse/comparison.png` and `validation.json`.
The pipeline assumes metric, Y-up input and uses Manhattan regularization where supported;
it cannot guarantee equal reconstruction quality for arbitrary unseen scenes.

Structural junction rules:

- Floor/ceiling contour segments are at least 10 cm after wall snapping and paired
  footprint merging, before triangulation or slab creation. Short orthogonal jogs are
  collapsed, preferring trimming protrusions over filling wall recesses.
- Holes with both local wall-frame dimensions below 40 cm are filled. Openings with
  either dimension at least 40 cm survive. Beam-induced triangulation subdivisions
  are not contour deformations.
- Wall planes narrower/shorter than 12 cm, shallow source flaps within 12 cm of supported
  wall planes, and wall-end overshoots below 12 cm are removed. Larger supported wall
  extensions and actual roof-relief contact remain.

- Triangulation keeps original perimeter indices, including clockwise concave outlines.
- Perimeter coordinates within 15 cm of a wall snap to it from either side.
- Before floor triangulation and ceiling synthesis, thin overhangs are cut against the
  full wall footprint, including recessed column corners. Both slab surfaces inherit
  that trimmed footprint before slab closure.
- Larger horizontal extensions survive when supported by a connected planar scan
  patch wider than 15 cm. Upstream observed wall spans longer than this are preserved.
- Ceilings use the wall perimeter, including column recesses, and their own supported
  extensions. A floor patio does not imply a roof above it.
- Beam direction follows the nearest lateral ceiling edge, obtained by lifting the wall
  direction vertically onto the roof plane. Alignment preserves measured spacing, width,
  and depth and anchors rotation at the scan's area-weighted centre. Beam axes are
  orthonormal in the fitted plane, relief faces stay parallel to it, and
  every strip is clipped to the exact footprint. Vertical relief sides keep all beam
  levels inside that horizontal perimeter.
- A slab floor is translated upward only when necessary to keep its lowest vertex above
  the highest ceiling vertex (10 micrometres of clearance for a nondegenerate solid).
  The effective floor equation and translation are recorded in the pattern manifest.
- Slab skirts follow actual relief edges, including beam corners. Walls terminate at the
  generated roof profile and lowest floor plane, closing local gaps and trimming vertical
  excess without flattening roof beams.

The existing extraction, plane fitting, and pattern classification stages still determine
which observations belong to each surface. These constraints do not reclassify the scan
or guarantee that the complete structural envelope is a single watertight solid.

## New-step correspondence analysis (Milestone A)

Run from the repository root:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/analyze_new_step.py
backend/.venv/bin/python backend/06_mesh_elaboration/plot_new_step.py
backend/.venv/bin/python backend/06_mesh_elaboration/view_new_step.py
```

If Open3D fails to initialize under Wayland, the viewer automatically retries through
XWayland when `DISPLAY` is available. To force that path manually:

```sh
env -u WAYLAND_DISPLAY backend/.venv/bin/python backend/06_mesh_elaboration/view_new_step.py
```

The read-only analysis matches source structural face centroids to role-compatible
regularized-shell surfaces, rejects distant, normal-incompatible, and centroid-screened
ambiguous matches, and writes face-indexed displacements plus a colored line-set preview
under `backend/current_scene/06_mesh_elaboration/new_step/outputs/`. The distance limit
is configurable with `--max-distance` (metres).
`outputs/contact_analysis.json` records the segmentation artifacts checked and why
instance-level contact constraints are unavailable when physical instance IDs cannot
be traced to the source faces.
Each preview line runs from a sampled source triangle centroid to its shell match;
green indicates higher confidence, red lower confidence, gray a rejected match, and
magenta an ambiguous match. The static overview includes a legend; source colors indicate
wall, floor, or ceiling role.
The viewer opens the shell and overlays the source samples, target samples, and lines in
one orbitable Open3D window; source colors distinguish wall, floor, and ceiling samples.
The static `outputs/structural_correspondence_overview.png` samples every 1,000th line for
readability. If Open3D cannot initialize its graphics context, the viewer prints this
image path and exits cleanly; use an X11 session or XWayland for the interactive view.

The existing `05_semantic_segmentation/step2_mesh_projection` labels do not align with
the complete new-step source mesh, but its projected face list is an exact prefix of the
source face list. The analyzer verifies this and writes
`outputs/source_face_global_class_labels.npy`, using `-1` for the 183,471 source faces
outside that prefix. These are global class IDs, not physical instance IDs. The 362-frame
2D segmentation has 93 global classes, and no validated source-aligned physical-instance
labels are available, so contact analysis is reported as unavailable rather than
guessing instance boundaries. The ambiguity screen compares nearby shell triangle
centroids and normals; it is a diagnostic heuristic, not a global correspondence proof.

To build provisional 3D instance tracks from the frozen DINO/SAM observations on the
current source mesh, run:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/build_instance_candidates.py
```

This reuses the existing Panoptic3D correspondence/identity solver without model
inference and stages its workspace under `new_step/outputs/`. The result remains a
candidate for contact review; the previous full-scene candidate did not pass its
physical-instance boundary gate. No geometry is deformed by this command.

With the candidate available, generate provisional contact diagnostics:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/analyze_instance_contacts.py
backend/.venv/bin/python backend/06_mesh_elaboration/plot_new_step.py --contacts-only
backend/.venv/bin/python backend/06_mesh_elaboration/view_new_step.py --contacts-only
```

The report requires nearby, normal-compatible faces with support from at least two
independent camera groups and a connected patch. In the viewer, candidate object surfaces
and their contact vectors share a color; target roles and vector ranges are listed in
`outputs/instance_contact_candidates.json`. The mesh preview shows the 31 objects with
geometric contact patches, keeping the review view compact. The candidate IDs still
require review; the JSON includes each candidate track's geometry-component count and
completeness status to expose fragmentation and the lack of reprojection validation.
`outputs/instance_contact_overview.png` provides static plan and elevation
views; x marks are shell endpoints, and the legend maps object colors to candidate IDs and
target roles.

### Rigid baseline (Milestone B)

Build and inspect the global rigid baseline:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/build_rigid_baseline.py
# Optional: permit a weighted global rotation as well as translation
backend/.venv/bin/python backend/06_mesh_elaboration/build_rigid_baseline.py --allow-rotation
backend/.venv/bin/python \
  backend/06_mesh_elaboration/view_new_step.py --rigid-baseline
```

By default the baseline fits translation only from all accepted, unambiguous wall, floor,
and ceiling correspondences. Use `--allow-rotation` to fit a weighted rigid rotation too.
Source triangle area and correspondence confidence set each anchor's weight. It applies
that same transform to every retained non-structural source
triangle, preserving their relative positions. Source structural faces and nearby
unlabeled ceiling fragments are replaced by the regularized shell. No per-instance
corrections or boundary-based triangle removals are applied. The report contains the
global transform and per-role fit residuals; provenance maps each output face to its
source face and candidate instance ID.
`rigid_baseline_shell.ply` is a byte-for-byte copy of the authoritative shell;
`rigid_baseline_scene.ply` combines that shell and globally aligned object mesh for
inspection. The build also transforms the source's individual instance PLYs into
`outputs/rigid_baseline_instances/`; instances sharing a source triangle edge are listed
in `outputs/rigid_baseline_clusters.json`. The visualizer loads those individual files
directly (plus a PLY for untracked faces), all under the same global transform. It
highlights region class 189 in orange and uses XWayland when
available in a Wayland session. If Open3D cannot start a graphics context,
`view_new_step.py --rigid-baseline` writes `outputs/rigid_baseline_overview.png`.
Local shell penetration and object-object collisions are not checked.

### Minimum per-instance translation candidate

Run this after the global baseline:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/translate_instances_inside.py
backend/.venv/bin/python backend/06_mesh_elaboration/view_new_step.py --instance-translated
```

Each tracked instance receives the minimum horizontal translation found to put its
projected triangles inside a 1 cm inset of the shell footprint, followed by a minimum
vertical adjustment where local floor and ceiling hits define a feasible interval.
The candidate preserves triangle counts, instance IDs, and face provenance; it does
not rotate or deform instances. The report marks cases that cannot be certified by
these shell bounds, including parts above openings or beyond the available floor/ceiling
ray hits. Unknown IDs are processed as connected semantic
pieces when they contain at least 20 faces; smaller unknown fragments remain at the
global alignment and are counted in the report. Geometrically infeasible groups stay
in the output and are marked unresolved. `outputs/instance_translation_report.json`
contains each group's transform and boundary counts. The shell is read-only and copied
unchanged into the combined scene.
The viewer also writes `outputs/instance_translation_delta.png`, which shows moved
group centroids before (blue) and after (red) with displacement arrows.
