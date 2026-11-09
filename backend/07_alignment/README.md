# Phase 07 — Unified Alignment (Objects & Multi-Room Scans)

Phase 07 registers the scans, applies those transforms to each scan's objects and assets, then fits object contact contours against the aligned regularized shells.

The pipeline runs in one command:
1. **Scan alignment**: Uses ARCore tracked paths, trajectory-guided door matching, portal clearance, and coplanar regularized planes to align scans.
2. **Asset synchronization**: Applies each scan's final rigid transform to its source object instances, sealed and regularized meshes, Gaussian splats, and scan assets.
3. **Object contour fitting**: Detects filtered sealed-mesh contact contours, translates touching instance groups into their regularized shells, then pulls them toward nearby planes and applies per-scan wall median regularization. The plane pull allows up to 5% of unique red-contour vertex positions outside, forbids any object vertices below the lowest floor, and has a 15 cm cap per axis. The median pull has a separate 15 cm per-plane cap.
4. **Region selection and replacement**: A triangle is a candidate when any vertex is more than 1 mm beyond its scan's exterior regularized shell. Edge-connected patches pass the first cutoff only when area is greater than 0.5 m² and maximum depth is greater than 5 cm. Adjacent patches within 1 mm are grouped. A group is kept when its seam score—attachment boundary length divided by the square root of group area—is below 3.0, or its rectangle similarity—projected union area divided by the minimum rotated rectangle area on the dominant nearest regularized plane—is above 0.7. Each kept group's projected contour produces a size-regularized axis-aligned box retaining at least 95% of its area and boundary. It is a window when Gemma 2 associated its instance with window or the box bottom is over 1.5 m above its closest same-scan floor; otherwise it is a mirror. The retained faces are removed from individual object PLYs and the exact corresponding Phase 05 source faces are removed from that scan's sealed mesh. Object-frame panels remain with `aligned_objects.ply`; sealed-frame panels reverse each contributing instance's post-scan placement translation while preserving global scan alignment, then move only as needed to fit within the sealed mesh bounds and stay above the selected same-scan floor plane. If those constraints cannot both be met, replacement fails instead of publishing a misplaced panel. For a region with one placement group, the sealed panel reuses the region's already-optimized box; multiple placement groups get their own boxes. Panels are added to each scan's sealed `mesh.ply`, `aligned_sealed_meshes.ply`, and GLB. Mirror panels sit 1 cm inside the support plane. Object-frame panels are saved under `objects/<scan>/panel_instances/`; sealed-frame panels are saved under `scans/<scan>/panel_instances/`. Regularized meshes are not modified.

Door candidates come from accepted per-instance Phase 05 associations. When the oversized-door rule finds multiple door-facing patches, Phase 07 overwrites that parent in the non-structural object copy with matching child PLYs, adds child door matches to `valid_instance_associations.json`, and refreshes `segmentation.json`'s `door_instances` list. The structural wall PLY stays intact. Aligned object manifests mark those derived instances as category `door` with text class ID 1.

At the end of Phase 07, the separate door-panel verification artifact merges duplicate parallel panels when their face-to-face gap is at most 25 cm and they overlap at least 15% of the smaller panel's projected width-by-height area. Face gap subtracts half the 10 cm depth of each panel from the center-plane distance. Remaining panels whose center planes are within 15 cm are snapped to their mean plane offset and share the largest whole-panel dimensions in that group. Width and height are then snapped first to 5 cm and then to 20 cm. A final overlap pass finds parallel panels whose projections overlap at least 60% of the smaller panel, with no limit on plane separation. It takes the width and height modes from the entire panel set, applies them to matching overlap groups, and moves each group's projected centers to their mean so the panels fully overlap while preserving every panel's normal offset and inter-plane gaps. Panel bottoms initially use the aligned Phase 06 floor; this last in-plane alignment may move them vertically. Panel depth stays 10 cm. The resulting PLY is not included in production object, sealed-mesh, or Gaussian PLYs.

`stats/aligned_door_panels.json` records final panel dimensions, placements, and source instances. `stats/door_panel_regularization_report.json` records duplicate merge checks and groups, plane and floor adjustments, before/after panel dimensions and positions, and global dimension clusters. Door segment bounds use `.raw_unaligned/mesh.ply` when available, applying the saved scan transform once. The segment fitter still protects its own door segments from being shortened below 60 cm.

The Windows panel pipeline runs automatically during Phase 07 Step 5, immediately after door-panel generation. Its `window_panel_regularization.py` implementation reads accepted Phase 05 window associations (text class ID 2) and the aligned object manifest. It rejects a window when its aligned 3D axis-aligned bounds intersect any aligned door bounds, then validates width 0.30–4.50 m, depth 0.03–0.45 m, and height 0.30–3.00 m. After 15 cm center-plane snapping, each coplanar group gets one sill height: the median source sill is rounded to the nearest 15 cm, while each panel retains its own same-scan Phase 06 floor elevation. The independent stages save semantic candidates, eligible panels, duplicate merges (25 cm face gap and 15% projected overlap), plane snapping, floor/sill alignment, 5 cm then 20 cm size snapping, and final 60% overlap alignment. The final overlap step gets width/height modes from all eligible panels and preserves panel floor-relative sill heights and center-plane offsets. These are separate verification outputs; this step does not modify the door panels, aligned scene objects, sealed meshes, or Gaussian splats. To rerun only this stage from existing aligned objects, use:

```bash
backend/.venv/bin/python backend/07_alignment/window_panel_regularization.py --workspace backend/current_scene --alignment-dir backend/current_scene/multi_room_alignment
```

Intermediate PLY snapshots and stage reports are under `window_panel_pipeline/` (`00_semantic_window_instances`, `01_eligible_window_panels`, `02_duplicate_merged_panels`, `03_planes_snapped`, `04_floor_sill_aligned_panels`, `05_dimensions_regularized`, and `06_overlap_regularized_panels`). `05_overlap_regularized_panels.ply` remains as an alias for the prior final-snapshot path. `00_candidate_report.json` records door overlaps and exclusions. `aligned_window_panels.ply` is the final alias; `stats/aligned_window_panels.json` associates each panel with its source scan/class/instance and floor/sill placement, and `stats/window_panel_regularization_report.json` contains all stage decisions and checks.

Each of the five macro steps logs its elapsed wall time, and the runner records named substep timings. After a successful run, total, macro-step, and substep timings are written to `stats/phase07_timing_report.json`.

Perimeter wall direction is inferred with 2 cm probes against the regularized floor footprint; patch grouping estimates surface gaps from vertices, edge midpoints, and face centers.

Contour colors, the red-marked object PLY, classification images, and earlier contour/PBR experiment folders are temporary working data. The runner removes them after the final replacement files are published. The replacement manifest at `stats/phase07_region_replacements.json` records region decisions, exact source-face removals, and sealed-panel placement transforms. Regularized meshes remain exactly as synchronized before this step.

---

## Directory Structure

```text
backend/07_alignment/
├── step_alignment.py                 # Unified Production Orchestrator & CLI
├── step_objects_alignment.py         # Standalone legacy per-scan alignment utility
├── temporary_alternative_pipeline/  # Production contour detection and fitting stages
├── step_scan_alignment.py           # Multi-room scan alignment runner
├── scan_loader.py                   # Dynamic dataset ingestion for arbitrary scan sets
├── arcore_trajectory_bridge.py      # ARCore transit tracking & Manhattan rotation estimation
├── portal_detector.py               # Trajectory ray-piercing & composite door decomposition
├── door_alignment_engine.py         # Door pairing & coaxial portal registration
├── coplanar_planes_engine.py        # All-plane coplanar optimization (floors & facades)
├── multi_asset_synchronizer.py      # Step 3: Global SE(3) transformation & deliverables exporter
├── object_outlier_regions.py        # Final red outlier detection, cutoffs, and face marker
├── step_replace_red_regions_with_panels.py # Final classification and mesh replacement
└── README.md
```

---

## Running the Unified Pipeline

From the repository root:

```bash
backend/.venv/bin/python backend/07_alignment/step_alignment.py --workspace backend/current_scene
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--workspace` | `backend/current_scene` | Path to workspace containing `scan_*` directories (or single scan). |
| `--output-dir` | `<workspace>/multi_room_alignment` | Output directory for synchronized property models and deliverables. |
| `--skip-scans` | `False` | Skip multi-room scan alignment and reuse existing transforms. |

---

## Deliverables & Outputs

All 3D assets, scan models, and statistical manifests are exported inside `<workspace>/multi_room_alignment/`:

| Asset Path | Description |
|---|---|
| `aligned_objects.ply` | Final objects with selected outlier faces removed and object-frame silver panels included. |
| `objects/<scan_name>/<class_name>/instance_*.ply` | Final object instances with selected outlier faces removed. |
| `objects/<scan_name>/panel_instances/*.ply` | Object-frame silver panels included in `aligned_objects.ply`. |
| `aligned_regularized_meshes.ply` | Merged Phase 06 regularized meshes in global space. |
| `aligned_sealed_meshes.ply` | Merged Phase 04 sealed meshes with the exact selected source faces removed and sealed-frame replacement panels. |
| `aligned_sealed_meshes.glb` | GLB version of the final aligned sealed meshes, including sealed-frame replacement panels. |
| `aligned_sealed_gaussians.ply` | Merged Phase 04 Gaussian splats; centers inside another scan's aligned sealed mesh are removed. |
| `aligned_door_panels.ply` | Separate verification PLY containing only the aligned regularized door panels. |
| `stats/aligned_door_panels.json` | Final panel dimensions, placement, and source instance IDs. |
| `stats/door_panel_regularization_report.json` | Pairwise center-plane and face-gap checks, merge and plane groups, final overlap groups and all-panel dimension modes, floor/plane adjustments, and dimension clusters. |
| `aligned_window_panels.ply` | Separate verification PLY containing the final regularized window panels. |
| `stats/aligned_window_panels.json` | Window panel dimensions, floor-relative sill placement, and source instance IDs. |
| `stats/window_panel_regularization_report.json` and `window_panel_pipeline/` | Window decisions and all intermediate stage PLY/JSON artifacts. |
| `stats/phase07_region_replacements.json` | Per-region classification, exact source-face removal counts, and sealed-panel placement transforms. |
| `stats/phase07_timing_report.json` | Phase 07 macro-step and substep wall times from the most recent successful run. |
| `scans/<scan_name>/mesh.ply` | That scan's aligned sealed mesh with corresponding source faces removed and sealed-frame replacement panels. |
| `scans/<scan_name>/panel_instances/*.ply` | Sealed-frame panels with each contributing object's placement translation reversed. |
| `scans/<scan_name>/mesh_elaboration/complete.ply` and `elements/*.ply` | That scan's aligned regularized mesh and elements. |
| `scans/<scan_name>/gaussian_splat/*.ply` | That scan's aligned Gaussian splats. |
| `scans/<scan_name>/trajectory.csv`, `transforms.json`, `sparse/0/*` | Aligned ARCore trajectory, NeRF/3DGS poses, and COLMAP reconstruction/poses. |
| `stats/aligned_scans_transforms.json` | Solved $4 \times 4$ $\mathrm{SE}(3)$ transform for each scan; retained so `--skip-scans` can reuse the alignment. |

Multi-scan registration uses the exact union of Phase 06 floor triangles to reject any positive-area overlap between room footprints. Candidate wall planes are snapped even along doorway tangent axes when their aligned surface spans remain disjoint and the paired door openings still overlap. Alignment reports and visual diagnostics are temporary and removed after successful publication.

The merged and per-scan Gaussian clouds omit centers enclosed by another scan's aligned Phase 04 sealed mesh. Source scan assets are backed up in each scan's `.raw_unaligned/` before in-place alignment. The runner does not produce `aligned_scene_with_objects.ply`; the sealed meshes and objects remain available as separate outputs.
