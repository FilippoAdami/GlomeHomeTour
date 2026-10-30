# Stage 06 — structural mesh elaboration

Stage 06 consumes the Phase 04 Y-up, X/Z-aligned, floor-zero mesh and refined semantic labels from Stage 05. It extracts major structural surfaces (walls, floors, ceilings, stairs, columns, and ramps), removes degenerate and elongated inferred fill faces from every role before region grouping, and filters floor/ceiling instances by their expected face orientation. The wall enclosure uses wall support bounds; ceilings do not set its height. Floor projection slices use the next floor or wall enclosure top as their upper bound and omit ceiling-role faces. Stage 06 also synthesizes regularized geometry and watertight floor/ceiling slabs with a 4.0° tilt snapping tolerance. It preserves the Phase 04 world frame; it does not level or globally rotate the scene.

Object alignment, contact clustering, containment optimization, and unobstructed interior cross-sections have transitioned downstream to [Stage 07 (`07_objecs_alignment`)](../07_objecs_alignment/README.md). Earlier experimental passes (SEESR super-resolution Step 13, structural level grouping Step 14, and 2D wall perimeter infill Step 15) have been retired.

The stage treats Stage 04/05 geometry and labels as read-only. The standalone `step_mesh_elaboration.py` command writes its inspectable atomic outputs to `<scan>/mesh_elaboration_steps/`. The full `run_phase06.py` command rebuilds all Phase 06 steps in a temporary workspace and publishes only `<scan>/mesh_elaboration/complete.ply` and `<scan>/mesh_elaboration/elements/`.

Step 1 removes statistical outliers from no role. Inferred fill triangles are removed when their longest edge is at least 30 cm and their edge-to-altitude ratio is at least 20; original triangles are exempt. During component assignment, a small disconnected blob within the merge distance is discarded only when fewer than 10% of its faces have a nearby major-instance face plane within 3 cm and 25 degrees. Face-center lookup allows 15 cm beyond the vertex merge distance to account for triangle-centroid spacing. Cleanup preserves retained coordinates, colors, and face fill labels, and records the policy and counts in `structural_instances_manifest.json`.

## Stage files

| Component | Responsibility | Runtime status |
| --- | --- | --- |
| [`step_mesh_elaboration.py`](step_mesh_elaboration.py) | Runs Steps 1–3: structural extraction, wall enclosure, and floor/ceiling regularization. | Standalone atomic-output entry point. |
| [`run_phase06.py`](run_phase06.py) | Rebuilds Steps 1–3 in a temporary workspace, finalizes walls and ceilings, assembles the final mesh, and publishes the complete PLY plus finalized element PLYs. | One-command full Phase 06 entry point. |
| [`mesh_elaboration_inputs.py`](mesh_elaboration_inputs.py) | Scene discovery, input resolution, and validation of Stage 01–05 artifacts. | Production input boundary. |
| [`extract_structural_meshes.py`](extract_structural_meshes.py) | Decomposes mesh faces by semantic role (`wall`, `floor`, `ceiling`, `stair`, `column`, `ramp`), filters floaters, merges small patches into major instances, and associates recognized doors/windows. | Production extraction pass (Step 1). |
| [`enclose_wall_instances.py`](enclose_wall_instances.py) | Runs region growth, robust plane fitting/snapping, shell assembly, and support-aware plane texture regularization. Publishes the closed side shell and, when requested by the pipeline, its footprint metadata. | Production wall-fitting step (Step 2). |
| [`group_floor_instances_by_level.py`](group_floor_instances_by_level.py) | Groups extracted floor instances by height into the declared floor count and appends ramp instances sharing an exact geometric edge with a floor instance. | First atomic floor pass; outputs `intermediate_floor_levels/floor_level_NNN_intermediate.ply`. |
| [`add_ceilings_to_floor_levels.py`](add_ceilings_to_floor_levels.py) | Adds ceiling instances whose triangle-centroid median is at most 0.50 m below each floor and marks face roles. | Second atomic floor pass; updates matched intermediate PLYs. |
| [`clean_floor_level_intermediates.py`](clean_floor_level_intermediates.py) | Uses the shared dominant-plane filter to remove outliers while retaining vertical faces within 20 cm of their role's fitted plane. | Third floor pass; rewrites the intermediate PLYs. |
| [`planar_outlier_filter.py`](planar_outlier_filter.py) | Shared area-weighted plane fit and 20 cm face-centroid outlier filter used by floor cleanup and ceiling aggregation. | Stage 06 utility. |
| [`fit_floor_level_planes.py`](fit_floor_level_planes.py) | Splits floor faces into edge-connected areas when ramps are present, robustly fits from floor-role triangles only, snaps tilts below 5° to horizontal, and appends bounding-box-sized plane previews. | Fourth floor pass; writes one `floor_level_NNN_fitted.ply` per level. |
| [`aggregate_fit_ceiling_planes.py`](aggregate_fit_ceiling_planes.py) | Cleans raw ceiling instances with the shared dominant-plane filter, groups them by joint plane support, fits each aggregate, and matches associated dimensions to the fitted floor. | Fifth floor pass; writes `ceiling_plane_NNN_fitted.ply` groups and a manifest. |
| [`slice_scene_by_floor.py`](slice_scene_by_floor.py) | Crops the original sealed mesh between fitted floor levels (or the wall enclosure top for the top level) and excludes ceiling-role faces. | Sixth floor pass; writes one scene slice per level with floor-only plane previews. |
| [`clean_floor_scene_slices.py`](clean_floor_scene_slices.py) | Removes only elongated, long hole-fill faces from the floor scene slices. | Seventh floor pass; updates slice PLYs and their manifest. |
| [`project_floor_slices_onto_planes.py`](project_floor_slices_onto_planes.py) | Vertically collapses each full floor slice onto every fitted floor-area plane, applies the locked morphology first refinement, and optionally fits rectangles into wall-enclosed white gaps. | Eighth floor pass; writes separate raw, legacy-refined, morphology-refined, and hole-overlay PNGs with manifests. |
| [`regularize_floor_projection_contours.py`](regularize_floor_projection_contours.py) | Clears black pixels inside fitted holes and matches parallel nearby wall-contour segments across levels, then re-enforces perpendicular neighboring edges and the 10 cm minimum after alignment. | Optional pre-fill floor pass; writes separate regularized contour and blue-hole previews. |
| [`fit_floor_projection_contours_to_black.py`](fit_floor_projection_contours_to_black.py) | Snaps fitted hole-rectangle edges to parallel wall-contour segments when the gap is under 10 cm, removes black pixels outside the contour, fills its interior except for holes, and annotates segment lengths. | Optional post-projection floor pass; writes dimensioned contour-fitted PNGs and a manifest. |
| [`overlay_fitted_floor_wall_contours.py`](overlay_fitted_floor_wall_contours.py) | Exports each final floor polygon and regularized wall contour in world X/Z, then draws the dimensioned top-down comparison. Shared wall/floor edges receive one measurement. | Final part of the unified `floors` step; writes one JSON per level and one combined PNG. |
| [`regularize_floor_level_patterns.py`](regularize_floor_level_patterns.py) | Runs the shared planar pattern/appearance utility on observed floor-role triangles aligned within 30° of the fitted floor plane, excluding ramps, ceilings, fill faces, and fitted previews. Texture tiles are applied only when the shared utility finds coherent spatial texture evidence. | Final floor pass; writes one regularized floor-only PLY per level and a recognition manifest. |
| [`planar_pattern_regularization.py`](planar_pattern_regularization.py) | Applies an optional caller-supplied neutral color reference, removes compact color outliers, requires coherent spatial evidence before synthesizing texture, and preserves a color split only when strong support follows one straight in-plane boundary. | Shared by wall assembly and final floor regularization. |
| [`fit_wall_planes.py`](fit_wall_planes.py) | Earlier Manhattan subdivision fitter retained for comparison and diagnostic use. | Legacy; not run in production. |
| [`grow_wall_regions.py`](grow_wall_regions.py) | Region-growth implementation used by the combined shell builder; can also export a colored diagnostic. | Reused implementation. |
| [`fit_wall_region_planes.py`](fit_wall_region_planes.py) | Robustly fits and snaps planes for each non-black grown region; can also export plane previews and a detailed diagnostic manifest. | Reused implementation. |
| [`assemble_wall_region_planes.py`](assemble_wall_region_planes.py) | Preserves local support extents, carves overlapping panels, adds seam connectors, then repairs floor-level gaps using supported panels and audits closed horizontal sections. | Reused implementation. |
| [`fit_floors_ceilings_planes.py`](fit_floors_ceilings_planes.py) | Fits horizontal (within 4.0° tilt tolerance) and inclined floor/ceiling support planes using area-weighted covariance and residual scoring. | Production per-instance plane fitting pass after floor-level previews. |
| [`recognize_and_regularize_patterns.py`](recognize_and_regularize_patterns.py) | Earlier combined floor/ceiling regularizer and slab builder. | Legacy; Phase 06 Steps 3–5 now use the explicit floor, ceiling, wall, and assembly artifacts below. |
| [`structural_geometry.py`](structural_geometry.py) | Shared geometric constraints, pattern frame coordinate transforms, perimeter snapping, and overhang trimming. | Geometry engine helper. |
| [`surface_footprints.py`](surface_footprints.py) | Horizontal polygon rasterization, contour cleaning, and wall-constrained snapping. | Footprint helper. |
| [`level_footprints.py`](level_footprints.py) | Constrained orthogonal distance ring fitting (SLSQP) for per-level collapse outlines. | Level footprint helper. |
| [`axis_align_structural_mesh.py`](axis_align_structural_mesh.py) | Legacy global yaw-rotation utility. | Not run in production; Phase 04 owns world-frame alignment. |
| [`apply_structural_textures.py`](apply_structural_textures.py) | Re-paints structural surfaces with authentic, seamless parquet tiles, periodic beam textures, and plaster lighting regularization. | Legacy standalone slab workflow. |
| [`reproject_enhanced_cross_sections.py`](reproject_enhanced_cross_sections.py) | Reprojects high-resolution enhanced cross-section views back onto 3D mesh vertices. | Optional texture pass. |
| [`extract_floor_ceiling_residual.py`](extract_floor_ceiling_residual.py) | Slices scene by level, retains two-hop floor-connected components, and projects residual geometry onto 2D raster grids. | 2D floor plan generation pass. |
| [`fit_floor_projection_contours.py`](fit_floor_projection_contours.py) | Searches floor projections for axis-aligned bounding contours with 15 cm minimum edge lengths. | Floor contour fitting helper. |
| [`fit_wall_planes_to_floor_perimeters.py`](fit_wall_planes_to_floor_perimeters.py) | Adjusts 3D wall planes to match the 2D perimeter outlines aligned across multiple floors. | Wall-to-perimeter alignment helper. |
| [`fit_enclosing_walls_to_regularized_contours.py`](fit_enclosing_walls_to_regularized_contours.py) | Fits one enclosing wall plane per regularized Step 3 contour edge in each source-shell height band, rebuilds horizontal transition ledges between adjacent contours, and retains panel-local regularized appearance. | Phase 06 Step 4.1; writes `wall_enclosing_planes_fitted_to_regularized_contours.ply`. |
| [`regularize_ceiling_planes.py`](regularize_ceiling_planes.py) | Establishes ceiling planes, uses the associated final floor polygon or a wall slice 30 cm below unassociated ceilings as support, and constrains upper-ceiling fits against a tilted plane fitted to the upper boundary of the extracted Step 1 wall-instance PLYs; lower paired ceilings remain at their floor-supported elevation. | Phase 06 Step 4.2; writes `ceiling_planes_fitted_to_support_contours_regularized.ply` and a manifest. |
| [`finalize_single_elements.py`](finalize_single_elements.py) | Runs Step 4.1 wall contour fitting followed by Step 4.2 ceiling finalization, passing the newly fitted wall PLY into ceiling support discovery. | Phase 06 Step 4 command-line entry point. |
| [`assemble_final_regularized_mesh.py`](assemble_final_regularized_mesh.py) | Clips walls to the floor and fitted roof, extends gaps along the upper wall perimeter, clips floors and paired slabs to wall support, and writes the combined PLY plus final per-element PLYs. | Phase 06 Step 5 command-line entry point. |
| [`extract_non_structural_frames.py`](extract_non_structural_frames.py) | Renders delivered non-structural instances for fixed-category matching. | Reused by Stage 05 finalization. |
| [`extract_non_structural_frames_gs.py`](extract_non_structural_frames_gs.py) | Offscreen CUDA/ROCm Gaussian Splat rasterization wrapper for instance framing. | Optional renderer used by Stage 05 finalization. |
| [`run_non_structural_category_similarity.py`](run_non_structural_category_similarity.py) | Ranks rendered frames against 28 indoor categories using EmbeddingGemma. | Invoked at the end of Stage 05; results are delivered under `scan_N/segmentation/non_structural_similarity/`. |
| [`step_semantic_similarity.py`](step_semantic_similarity.py) | Standalone embedding extraction, caching, and cosine matrix ranking pipeline. | Similarity engine. |
| [`project_valid_instances_by_floor.py`](project_valid_instances_by_floor.py) | Projects classified non-structural instances onto 2D floor plans with bounding dimension annotations. | Floorplan annotation pass. |
| [`generate_floorplan_symbols.py`](generate_floorplan_symbols.py) | Procedural dimension-constrained SVG symbol generator for 28 indoor furniture and architectural classes. | Floor plan symbol generator. |
| [`match_real_captured_frames.py`](match_real_captured_frames.py) | Matches synthetic instance viewpoints to actual registered camera frames using Phase 04-aligned COLMAP poses when available. | Viewpoint verification helper. |
| [`scene_inputs.py`](scene_inputs.py) | Compatibility alias of `mesh_elaboration_inputs.py`. | Legacy compatibility boundary. |

## Pipeline flow

```text
Stage 04: aligned sealed mesh.ply (with is_hole_fill) + mesh.json
Stage 05: triangle_class_labels.npy + class_metadata.json + Instances PLYs/
                               │
                               ▼
Step 1: extract_structural_meshes
        (Extract Walls, Floors, Ceilings, Stairs, Columns, Ramps; merge door/window instances)
                               │
                               ▼
Step 2: fit_enclosing_walls
        (Use the Phase 04 Y-up, axis-aligned, floor-zero frame;
         grow regions from triangle normals and orthogonal support distance;
         robustly fit and snap one plane per non-black region;
         preserve local support height where overlapping regions compete;
         connect panels, correct shared color cast and local shadows, preserve abrupt material changes, and require a closed side shell at every height;
         leave the floor and ceiling uncapped)
                               │
                               ▼
Step 3: floors
         (Group extracted floor instances by height and attach touching ramps;
          associate ceilings, clean and fit floor/ceiling planes, slice the source mesh,
          project and refine the floor masks, regularize wall contours, fit final floor
          polygons, and export dimensioned plan view plus one polygon JSON per floor)
                               │
                               ▼
Step 4: finalize_single_elements
        4.1 Fit enclosing walls to Step 3 regularized floor contours;
            retain detected geometric relief over each panel, facing inward
        4.2 Establish and group ceiling planes; choose floor or wall-slice support;
            enforce floor clearance; fit final contours; regularize ceiling surfaces
                               │
                               ▼
Step 5: assemble_final_regularized_mesh
        Pair associated floor and ceiling areas into closed slabs;
        retain detached surfaces; color slab sides from both source regions;
        cut walls at the floor and roof, extend upper perimeter gaps to the roof;
        constrain floors/slabs from wall support; export one final PLY
                               │
                               ▼
Downstream: Stage 07 (07_objecs_alignment) & 2D Floor Plans
        (Rigid object baseline, contact clustering, containment optimization, cross-sections)
```

## Input contract

All inputs below are generated by previous pipeline stages and locked. Stage 06 never modifies camera poses, images, depth maps, `mesh.ply`, or `mesh_segmented.ply`.

```text
current_scene/
└── scan_N/                                    # Target scan root
    ├── sparse/0/                              # Phase 04-aligned COLMAP poses (written in place)
    │   ├── cameras.txt
    │   └── images.txt
    ├── depth/                                 # Optional Stage 02 calibrated depth & scale
    │   └── scene_size.txt
    ├── gaussian_splat/                        # Stage 03/04 trained Gaussian Splatting model
    │   ├── checkpoint.pt
    │   ├── checkpoint_sealed.pt               # Preferred sealed model
    │   └── gaussian_splat_sealed.ply
    ├── mesh.ply                               # Stage 04 authoritative sealed mesh
    ├── mesh.json                              # Stage 04 reconstruction report
    ├── segmentation/                          # Stage 05 delivery package
    │   ├── triangle_class_labels.npy          # Face-order aligned class IDs
    │   ├── class_metadata.json                # Class ID to structural role & confidence
    │   ├── segmentation.json                  # Top-level instance & role manifest
    │   ├── mesh_segmented.ply                 # Face properties class_id, instance_id
    │   ├── Instances PLYs/                    # Exported submeshes by role and class
    │   └── non_structural_similarity/          # Fixed-category associations and reports
    │   ├── non_structural_category_frames/     # Rendered views used for matching
    │   └── 3d_masks/                           # Compatibility symlinks
    ├── mesh_elaboration_steps/                 # Atomic Steps 1–3 and their working files
    └── mesh_elaboration/                      # Final delivery: complete.ply and elements/
```

| Input | Required | Source | Validation and use |
| --- | --- | --- | --- |
| `mesh.ply` | Yes | Stage 04 | Authoritative watertight sealed mesh. Triangle order defines the coordinate frame and face-attribute indexing. Reads `is_hole_fill` face attribute to preserve genuine scan data during texture synthesis. |
| `triangle_class_labels.npy` | Yes | Stage 05 | 1D NumPy integer array of class IDs. Length must exactly match the number of triangles in `mesh.ply`. |
| `class_metadata.json` | Yes | Stage 05 | JSON mapping class IDs to structural roles (`"wall"`, `"floor"`, `"ceiling"`, `"stairs"`, `"column"`, `"ramp"`, `"object"`). Also accepts `segmentation.json` or legacy `structural_classes.json`. |
| `segmentation/` | Optional | Stage 05 | Directory containing `3d_masks/` or `Instances PLYs/`; Stage 05 category associations allow recognized door/window object meshes to be merged into wall instances before plane fitting. |
| `segmentation/non_structural_similarity/valid_class_associations.json` | Optional for legacy runs | Stage 05 | Maps opaque source class IDs to fixed category IDs; recognized door and window instances are merged into wall meshes during extraction. Plane fitting uses only the resulting wall geometry. |
| `gaussian_splat_sealed.ply` | Optional | Stage 04 | Sealed 3D Gaussian Splat model used for dual-view photorealistic non-structural frame rendering. |

### Coordinate and geometric invariants

- **World frame**: Phase 04 supplies Y-up, X/Z-aligned geometry with its supported floor reference at $Y=0$. Stage 06 fits and regularizes in this frame without a scene-wide rotation or translation.
- **Units**: Coordinates and geometric thresholds use the mesh's scene units. They are metres only when upstream scale is calibrated; image-only COLMAP reconstructions may have arbitrary scale.
- **Manhattan walls**: Wall planes use the Phase 04-aligned $+X$ and $+Z$ directions.
- **Watertight slab closure**: Floor/ceiling slabs must form closed, 2-manifold, zero-boundary, watertight solids with positive volume.
- **Supported wall panels**: A final panel must be strictly larger than $0.20$ and $1.0$ scene units in its shorter and longer dimensions. Same-facing overlapping support is consolidated into one outward envelope; the fitter greedily evaluates cuts at support deciles and subdivides while the area-weighted mean closest-panel distance (capped at 10 cm per triangle) plus 0.5 mm per panel decreases. Each candidate must preserve the closed side shell and 2% outside cap; corner spans are reconciled after offset changes. A resulting offset step gets an inferred connector only when that connector also meets both dimension limits. Every vertically filtered triangle whose projected footprint intersects supported panels is assigned to only its closest panel, and outside status follows that owner; each fit-source triangle is also audited for duplicate ownership. Missing mesh regions do not add distance samples. Panels use nearby floor/ceiling support when available; junction closure extends spans and vertical bounds. Every horizontal band between floor and ceiling must form one closed connected loop; the shell is uncapped at floor and ceiling.
- **4.0° horizontal tilt snapping tolerance**: In `fit_floors_ceilings_planes.py`, planes with tilt $\le 4.0^\circ$ relative to horizontal ($\arccos(|n_y|) \le 4.0^\circ$) are strictly snapped to horizontal ($n_y = \pm 1.0$).
- **Slender structural elements**: Slender column and ramp instances down to $0.20\text{ m}^2$ are preserved as major structural entities (vs $1.0\text{ m}^2$ for expansive walls and floors).

## Locked production policy

Step 2 repairs the floor perimeter after overlap cuts and junction connectors exist. Repairs extend supported panels vertically, preserve overlap masks, and are recorded in `inferred_floor_closure_planes`; when multiple extensions are required, redundant ones are pruned. Rectangle subtraction stays inside its source panel and is safe to reapply. Coincident panel/connector edges count as one geometric edge, so duplicate segments cannot masquerade as a closed room. Perpendicular corner matching includes the existing 1 cm topology tolerance around its nominal 20 cm junction limit.

| Policy | Enforcement |
| --- | --- |
| Wall enclosure | Supported panels require unique triangle ownership and dimensions strictly over 0.20 m and 1.0 m. Same-facing overlaps collapse into one envelope. Each triangle projecting onto supported panels gets only its closest panel owner; at most 2% of vertical-filtered wall triangles may fall outside the finite enclosure. Local floor/ceiling bounds must include observed wall support. Dimension-valid split connectors and perpendicular junction closure preserve one closed connected side loop at every interior height; if the floor-level section is open, supported panel extensions are retained only after all interior height bands pass topology checks. The shell remains uncapped at floor and ceiling. |
| Horizontal tilt tolerance | Any floor or ceiling support plane within **4.0° of horizontal** ($\arccos(|n_y|) \le 4.0^\circ$) is snapped to strict horizontal ($n_y = \pm 1.0$). |
| Slender element threshold | Columns and ramps are retained as major structural instances at $\ge 0.20$ scene units squared (standard walls/floors require $\ge 1.0$). |
| Watertight slab synthesis | All floor/ceiling slabs are synthesized as closed, watertight 2-manifold solids with zero open boundary edges and positive volume. |
| Frame preservation | Phase 04 supplies the Y-up, X/Z-aligned, floor-zero frame; Stage 06 applies no scene-wide transform. |
| Authentic textures | Procedural texture synthesis (parquet tiles, ceiling beam strips, plaster lighting regularization) derives colors exclusively from genuine scan vertices (`is_hole_fill == 0`). |
| Stage boundaries | Object alignment, multi-body contact clustering, and cross-section slicing are delegated to Stage 07; Stage 06 never modifies non-structural meshes. |

## Processing

1. **Structural instance extraction (`extract_structural_meshes.py`)**: Decomposes mesh faces by structural role (`wall`, `floor`, `ceiling`, `stair`, `column`, `ramp`) based on `class_metadata.json`. Connected components $\ge 1.0\text{ m}^2$ (or $\ge 0.20\text{ m}^2$ for columns and ramps) are retained as major instances; smaller components within $0.25\text{ m}$ are merged, and isolated crumbs are discarded. Recognized door and window object instances are appended to their nearest wall instance. Outputs `structural_instances_manifest.json` and instance PLYs (`wall_instNNN.ply`, `floor_instNNN.ply`, `ceiling_instNNN.ply`, `stair_instNNN.ply`, `column_instNNN.ply`, `ramp_instNNN.ply`).
2. **Wall fitting (`fit_wall_planes.py`)**: Filters degenerate faces and faces within $30^\circ$ of horizontal, then assigns each support triangle to its best-supported Phase 04 X/Z direction (dot product at least 0.85), so one triangle cannot seed parallel or orthogonal support planes. Connected patches need at least 0.02 area units squared, 15 cm in each discovery dimension, and 10% projected coverage; nearby patches merge within 20 cm when offsets differ by at most 5 cm. Same-facing overlapping supports consolidate into one outward envelope. The offset is the nearest outward position covering at least 99% of support triangles, selected from each triangle's farthest-out vertex, with no extra margin. Projected triangle geometry is intersected with each finite panel rectangle; only observed triangles contribute to distance, so holes have no infinite or synthetic penalty. Each triangle intersecting one or more supported panel projections is assigned to its closest panel, while inferred connectors do not claim observed support. A run may split once when two local offsets reduce area-weighted closest-triangle distance, each resulting panel is strictly over 0.20 and 1.0 scene units in its two dimensions, their offset step can be bridged by a dimension-valid connector, and the whole side shell remains closed. Local floor and ceiling support is sampled within 35 cm, must include observed wall support, and otherwise falls back to robust scene bounds. At nearby perpendicular junctions, panel spans and vertical bounds are reconciled. The fit fails if more than 2% of vertically filtered wall triangles have no finite projected overlap with an enclosing panel or lie beyond any intersecting panel's outward face; source-face uniqueness is audited separately to prevent onion-layer overlap. A strict topology audit requires one closed connected horizontal loop at every interior height; the wall shell remains uncapped at floor and ceiling. The manifest reports closest-triangle distance residuals for both fitted source support and uniquely assigned projected support, normal deviation, support coverage, inferred area, per-panel owner counts, and outside triangle/area fractions. Diagonal vertical support remains in the cleaned wall mesh and counts toward the 2% geometric outside allowance; if it exceeds the cap, fitting fails. Semantic door/window labels do not create or split planes.
3. **Floor polygon generation (`floors` orchestration step)**: Runs the established floor pipeline in order: height-group floor instances to the `scene_size.txt` count and append touching ramps; associate eligible ceilings, clean role-specific outliers, and fit floor planes; aggregate and fit ceilings; slice the original sealed mesh using the fitted floor and ceiling clearances; remove elongated sealing fills; project each slice, apply the locked morphology refinement and fit hole rectangles; clear holes and regularize wall contours across levels, then re-clean after line alignment to enforce perpendicular neighboring edges and the 10 cm minimum; fit the contours to retained black projection pixels and fill the final floor polygons; then infer geometric and visual patterns from observed floor-role scan triangles, extract and match clean lighting-corrected texture tiles in memory, correct their seams, and repeat accepted tiles across their detected floor regions. Source floor triangles provide appearance evidence only; the exported geometry is the final polygon rings on their fitted planes, with no ramp, ceiling, hole-fill, or fitted-preview triangles. Every final floor, wall, and ceiling footprint uses perpendicular neighboring edges and edges at least 10 cm long. A texture region must cover at least 0.5 m² and every edge of its simplified boundary must be at least 0.5 m; rejected regions retain the standard planar appearance. The texture utility returns samples and seam-corrected tiles to the floor writer without saving image files or manifests. Only final textured floor PLYs are written by this pass. The polygon export writes `floor_level_NNN_polygons.json` per level (regularized wall contours, final polygon rings, and unique dimensions) and `floor_level_all_polygons_topdown.png`. Shared wall-contour/floor-polygon edges are measured once; raster fragments below 5 cm are omitted from dimensions. All substeps overwrite their own named outputs on rerun.

Run Phase 06 Steps 1–3 independently for one scan. Step 3 runs the complete floor pipeline and overwrites its outputs on rerun:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/step_mesh_elaboration.py \
  --workspace backend/current_scene/scan_0 --only-step extract_structural_meshes

backend/.venv/bin/python backend/06_mesh_elaboration/step_mesh_elaboration.py \
  --workspace backend/current_scene/scan_0 --only-step fit_enclosing_walls

backend/.venv/bin/python backend/06_mesh_elaboration/step_mesh_elaboration.py \
  --workspace backend/current_scene/scan_0 --only-step floors

# Rerun only the final floor pattern/appearance regularization pass:
backend/.venv/bin/python backend/06_mesh_elaboration/step_mesh_elaboration.py \
  --workspace backend/current_scene/scan_0 --only-step regularize_floor_level_patterns
```

That final pass now rebuilds texture recognition, lighting-corrected matched tiles, seam correction, and the final textured floor PLYs in memory. It writes only the final PLYs.

All floor artifacts are under `mesh_elaboration_steps/intermediate_floor_levels/`. Step 3 rebuilds floors, ceilings, slices, projections, regularized contours, and final polygons in dependency order, so rerun it after changing any upstream extraction or wall-fit result.

To report a simple texture-present yes/no for each floor from observed floor-role scan triangles, run this detection-only pass after Step 3:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/detect_floor_texture_presence.py \
  --instances-dir backend/current_scene/scan_0/mesh_elaboration_steps
```

It writes `intermediate_floor_levels/floor_texture_presence_manifest.json` and does not modify floor geometry or appearance.

Texture extraction, matching, region validation, and seam correction now run in memory inside the final floor appearance pass. Regions that fail the 0.5 m² area or 0.5 m minimum boundary-segment test keep the standard planar appearance; no texture sample PNGs or 3×3 previews are written.

After Steps 1–3, run Phase 06 Step 4. Its atomic order is fixed: fit the enclosing wall panels to the final Step 3 contours and reapply any detected geometric relief toward each contour's interior; then finalize associated and unassociated ceilings using the selected support plane. Step 4.2 derives the top-cap plane from a 95th-quantile fit to the spatial upper envelope of significant boundary loops in the original extracted `wall_inst*.ply` files listed in `structural_instances_manifest.json`. Sampling all significant loops avoids extrapolating from one narrow wall strip; lower openings do not pull the cap down, and a 2 cm clearance keeps it above the sampled contour. This plane may be tilted; the fitted wall shell remains the source for wall-slice footprint support only. Only ceiling patches that reach within 30 cm of the extracted-wall cap plane are constrained to stay above it, preserving lower paired ceilings that form floor slabs. The source shell and prior ceiling source remain unchanged; separate output PLYs are written beside them:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/finalize_single_elements.py \
  --workspace backend/current_scene/scan_0
```

This writes `wall_enclosing_planes_fitted_to_regularized_contours.ply` and `ceiling_planes_fitted_to_support_contours_regularized.ply`, plus the ceiling manifest, under `mesh_elaboration_steps/`. Geometric relief is emitted as low-poly parallelepiped ribs only inside the observed support band, with a 5 cm cap and stricter repetition evidence; no smooth or triangulated relief profile is applied across the whole plane. Wall color synthesis stays panel-local, and horizontal transition ledges are rebuilt from the symmetric difference of adjacent contour footprints.

After Step 4, run Step 5 to assemble the final structural PLY without replacing any source artifact:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/assemble_final_regularized_mesh.py \
  --workspace backend/current_scene/scan_0
```

This writes `all_single_elements_finalized.ply` and its manifest in the atomic workspace. Associated floor and ceiling surfaces are consumed into watertight slabs; unpaired floors and ceilings remain standalone. Walls are clipped to the global lowest finalized floor and the highest upper-ceiling plane constrained against the extracted wall cap; the roof cut follows plane slope, with interpolated vertex colors. Any upper wall-perimeter segments that still sit below the roof are extended vertically to the fitted plane, closing gaps while preserving vertical-plane door and window openings. Every standalone floor is clipped to the fitted wall contour at its own height, and the floor level nearest the source wall base is aligned to that base; clipped vertex colors are interpolated. Each paired slab is clipped to the intersection of wall contours across its full vertical span before its perimeter is rebuilt. Orphan ceilings retain their Step 4 geometry, including elevation, contour, and relief. No floor-wall connectors or separate roof caps are generated. The `elements/` PLYs contain the same finalized faces as the combined PLY. Separate element boundaries may remain open; non-manifold edges fail the final runner, and each paired slab must be watertight.

Run the complete Phase 06 pipeline from the Stage 04/05 inputs with one command:

```sh
backend/.venv/bin/python backend/06_mesh_elaboration/run_phase06.py \
  --workspace backend/current_scene/scan_0
```

The command rebuilds Steps 1–3 from the selected scan's Stage 04 mesh and Stage 05 labels, runs Step 4 wall and ceiling finalization, then assembles Step 5. Intermediate files stay in a temporary workspace and are removed after the run; any existing standalone `mesh_elaboration_steps/` workspace is left untouched and is not read by this command. The final delivery contains only `complete.ply` and `elements/`; the element PLYs contain the finalized faces used to build the complete mesh. Open boundaries between separate finalized elements are allowed; non-manifold edges fail publication. A previous delivery with this expected layout is replaced only after the new build passes validation; an unexpected directory at `mesh_elaboration/` is preserved and causes the command to stop. With a scene-size file, Steps 1–3 run only the explicit extraction, enclosing-wall, and floor-level passes. The older combined plane/regularization/texture workflow remains available for diagnostic runs without a floor count and explicit API calls, and is omitted from the complete runner.

## Legacy standalone plane and slab workflow

The following utilities are separate from the current Phase 06 Step 4 wall-and-ceiling finalization command.

4. **Floor and ceiling plane fitting (`fit_floors_ceilings_planes.py`)**: Fits horizontal and inclined support planes for each floor and ceiling instance using area-weighted principal components and mode binning, scoring perpendicular residuals. If the plane tilt relative to horizontal is $\le 4.0^\circ$, it is snapped to strict horizontal ($n_y = \pm 1.0$). Outputs `floors_ceilings_planes_manifest.json`.
5. **Pattern recognition and slab synthesis (`recognize_and_regularize_patterns.py`)**: Regularizes 2D footprints, detects architectural patterns (e.g., periodic ceiling beams), clips wall tops to roof profiles, and synthesizes closed, watertight 3D slab solids (`slab_instNNN.ply`). If a partial scan contains only floors or only ceilings without walls, applies robust architectural default bounds ($h \pm 2.8\text{ m}$). Combines surfaces into `all_structural_regularized.ply`.
6. **Authentic texture synthesis (`apply_structural_textures.py`)**: Synthesizes seamless parquet tiles, ceiling beam strips, and wall plaster lighting regularization from scan colors, then updates the regularized wall meshes and re-exports `wall_regularized.ply` and `all_structural_regularized.ply`. All outputs remain in the Phase 04 world frame.

### Auxiliary submodules

- **2D Floor plan projection (`extract_floor_ceiling_residual.py`, `fit_floor_projection_contours.py`, `fit_wall_planes_to_floor_perimeters.py`)**: Slices the scene by level, projects residual geometry onto 2D raster grids, fits orthogonal contours, and aligns 3D wall boundaries with 2D plan outlines.
- **Non-structural framing & dual rendering (`extract_non_structural_frames.py`, `extract_non_structural_frames_gs.py`)**: Calculates canonical human-height camera viewpoints with $\ge 0.8\text{ m}$ wall clearance and dual-renders perspective frames of non-structural objects from both the mesh and 3D Gaussian Splats.
- **Indoor category classification** runs at the end of Stage 05. Stage 06 consumes its `segmentation/non_structural_similarity/valid_class_associations.json` output.
- **Parametric SVG symbols (`generate_floorplan_symbols.py`, `project_valid_instances_by_floor.py`)**: Synthesizes dimension-constrained parametric SVG architectural symbols for 28 indoor categories and plots them onto the floor plan projections.

## Outputs and downstream interface

Production atomic outputs are written to `<scan>/mesh_elaboration_steps/` (or an explicit `--output-dir`); the published Phase 06 delivery is `<scan>/mesh_elaboration/`. Rerunning wall fitting clears and regenerates only its known output files and panel directories; extracted structural instances and their manifest are preserved.

| Output | Producer | Contents and consumer contract |
| --- | --- | --- |
| `structural_instances_manifest.json` | Extraction | Inventory of structural instance PLYs, their areas/elevations, and door/window wall merges. |
| `wall_enclosing_shell.ply` | Wall fitting | The final assembled side shell with support-derived vertex colors; in Blender Solid view, choose Color → Attribute to display `Col`. Side edges are closed while floor and ceiling remain open. |
| `wall_planes_manifest.json` | Wall fitting | Panel bounds and Manhattan basis for floor/ceiling footprint regularization, plus per-panel pattern, color-split, and cross-plane regularization diagnostics. |
| `wall_enclosing_planes_fitted_to_regularized_contours.ply` | Phase 06 Step 4.1 | Separate contour-fitted wall panels, with detected periodic relief reapplied toward the contour interior. |
| `ceiling_planes_fitted_to_support_contours_regularized.ply` | Phase 06 Step 4.2 | Separate final ceiling planes fitted to associated-floor or wall-slice support contours; the accompanying manifest records groups, clearances, and detected patterns. |
| `all_single_elements_finalized.ply` | Phase 06 Step 5 | Combined walls, unpaired floors/ceilings, and closed paired slabs, with `element_role` and `source_floor_level` face properties. |
| `all_single_elements_finalized_manifest.json` | Phase 06 Step 5 | Paired slab count, preserved ceiling elevations, floor alignment, and welded topology counts. |
| `intermediate_floor_levels/floor_level_NNN_fitted.ply` | Floor-level fitting | Aggregate floor geometry with fitted floor plane previews. |
| `intermediate_floor_levels/ceiling_plane_NNN_fitted.ply` | Ceiling aggregation | Co-planar aggregate ceiling geometry with its fitted plane preview and face-role metadata. |
| `intermediate_floor_levels/ceiling_plane_groups_manifest.json` | Ceiling aggregation | Co-planar source grouping, associated floor level, fitted plane equations, snap state, and preview dimensions. |
| `intermediate_floor_levels/floor_level_NNN_scene_slice.ply` | Slicing and cleanup | Original sealed scene triangles clipped between fitted plane clearances, plus fitted plane previews and source class IDs; elongated sealing fills are removed. |
| `intermediate_floor_levels/floor_level_scene_slices_manifest.json` | Slicing and cleanup | Source mesh, selected ceiling groups and plane equations, per-level triangle counts, and sealing-fill cleanup thresholds/counts. |
| `intermediate_floor_levels/floor_level_NNN_area_NNN_collapsed.png` | Floor slice projection | Top view of non-vertical source triangles in a floor slice, collapsed to black onto the fitted floor-area plane, refined with seven sigma-13 unsharp-mask turns by default and black-mask strip/outlier cleanup, then overlaid with the red fitted-wall contour at 30 cm above the floor. The view bounds include the entire projected slice. |
| `intermediate_floor_levels/floor_level_NNN_area_NNN_collapsed_original.png` | Floor slice projection | Raw rendered projection before blur, mask refinement, strip/outlier cleanup, or wall-contour overlay. |
| `intermediate_floor_levels/floor_level_NNN_area_NNN_collapsed_first_refinement.png` | Floor slice projection | Locked morphology first refinement generated by `--first-refinement` or as part of `--detect-holes`. |
| `intermediate_floor_levels/floor_level_NNN_area_NNN_collapsed_first_refinement_with_holes.png` | Floor slice projection | First refinement with translucent-blue rectangles aligned to the dominant wall-contour direction and fit to qualifying wall-enclosed white regions. |
| `intermediate_floor_levels/floor_level_NNN_area_NNN_collapsed_first_refinement_contour_regularized.png` | Contour regularization | First-refinement projection with the cross-floor regularized red wall contour. |
| `intermediate_floor_levels/floor_level_NNN_area_NNN_collapsed_first_refinement_contour_regularized_with_holes.png` | Contour regularization | Regularized contour with the original blue hole rectangles retained, even if they become small. |
| `intermediate_floor_levels/floor_level_NNN_area_NNN_collapsed_first_refinement_contour_fitted.png` | Projection contour fitting | Dimensioned final image with fitted contour (red), filled-polygon boundary (orange), and hole areas left unfilled without blue overlays. |
| `intermediate_floor_levels/floor_level_collapsed_projections_manifest.json` | Floor slice projection | Per-level and per-area raw/refined image names, plane equations, source triangle counts and vertical ranges, bounds expansion, refinement settings, and wall-contour segment counts. |
| `intermediate_floor_levels/floor_level_first_refinement_manifest.json` | Floor slice projection | First-refinement method parameters, per-level image names, component counts and pixel-removal metrics. |
| `intermediate_floor_levels/floor_level_collapsed_first_refinement_holes_manifest.json` | Floor slice projection and contour regularization | Original fitted hole rectangles, pixels cleared inside them, regularized contour coordinates, matched edges, and output image names. |
| `intermediate_floor_levels/floor_level_contour_fitted_manifest.json` | Projection contour fitting | Per-area cleared and filled pixel counts, simplified contour moves, rectangle-area checks, and length in meters for every fitted-contour and black-part segment. |
| `intermediate_floor_levels/floor_level_NNN_polygons.json` | Final floor export | Final floor polygon boundary rings and regularized wall contour in world X/Z metres, plus edge lengths with shared wall/floor edges measured once. |
| `intermediate_floor_levels/floor_level_all_polygons_topdown.png` | Final floor export | Combined world X/Z view of the filled floor polygons and dashed regularized wall contours, with unique segment measurements. |
| `intermediate_floor_levels/floor_level_NNN_patterns_regularized.ply` | Final floor pattern pass | Final polygon rings on the fitted floor plane with regularized appearance and repeated, seam-corrected texture colors over detected textured regions; excludes ramps, ceilings, hole-fill triangles, and fitted plane previews. |
| `intermediate_floor_levels/floor_texture_presence_manifest.json` | Texture presence check | Per-floor yes/no result and fine-scale color-structure diagnostics from observed floor triangles; detection only, no texture is synthesized or applied. |
| `floors_ceilings_planes_manifest.json` | Floor/ceiling fitting | Fitted plane equations ($ax + by + cz + d = 0$), tilts, and support areas with 4.0° horizontal snap. |
| `plane_patterns_manifest.json` | Regularization | Comprehensive manifest of regularized units, detected patterns (beams, plain), and closed slab paths. |
| `slab_instNNN.ply` | Regularization | Watertight 3D solid meshes representing floors, ceilings, and mezzanine slabs. Guaranteed zero open edges. |
| `all_structural_regularized.ply` | Regularization and texture synthesis | Unified regularized architectural shell combining walls, floors, ceilings, and slabs; texture synthesis updates its surface colors in place. |
| `step06_timings.json` | Orchestration | Per-step timings and exit codes. |

## Running

Run from `backend/` in the project Python virtual environment:

```bash
# 1. Full Stage 06 execution via workspace auto-discovery:
backend/.venv/bin/python 06_mesh_elaboration/step_mesh_elaboration.py \
  --workspace current_scene/scan_0

# Run only Step 1 (structural extraction):
backend/.venv/bin/python 06_mesh_elaboration/step_mesh_elaboration.py \
  --workspace current_scene/scan_0 --only-step extract_structural_meshes

# Run only Step 2 (after Step 1 has created structural instances):
backend/.venv/bin/python 06_mesh_elaboration/step_mesh_elaboration.py \
  --workspace current_scene/scan_0 --only-step fit_enclosing_walls

# Run only Step 3 (after Steps 1 and 2):
backend/.venv/bin/python 06_mesh_elaboration/step_mesh_elaboration.py \
  --workspace current_scene/scan_0 --only-step floors

# Run the combined wall-shell builder directly after extraction:
.venv/bin/python 06_mesh_elaboration/enclose_wall_instances.py \
  --instances-dir current_scene/scan_0/mesh_elaboration \
  --manifest-out current_scene/scan_0/mesh_elaboration/wall_planes_manifest.json

# Inspect original wall instances by region (all triangle orientations are included):
.venv/bin/python 06_mesh_elaboration/grow_wall_regions.py \
  --instances-dir current_scene/scan_0/mesh_elaboration \
  --output-dir current_scene/scan_0/mesh_elaboration/wall_regions

# Fit one plane per non-black region and write the combined PLY/GLB previews:
.venv/bin/python 06_mesh_elaboration/fit_wall_region_planes.py \
  --regions-dir current_scene/scan_0/mesh_elaboration/wall_regions

# Assemble the fitted planes into a floor-to-ceiling closed side shell:
.venv/bin/python 06_mesh_elaboration/assemble_wall_region_planes.py \
  --regions-dir current_scene/scan_0/mesh_elaboration/wall_regions

# 2. Execution with explicit input overrides:
backend/.venv/bin/python 06_mesh_elaboration/step_mesh_elaboration.py \
  --mesh-path current_scene/scan_0/mesh.ply \
  --labels-path current_scene/scan_0/segmentation/triangle_class_labels.npy \
  --structural-json current_scene/scan_0/segmentation/class_metadata.json \
  --segmentation-output current_scene/scan_0/segmentation \
  --output-dir current_scene/scan_0/mesh_elaboration

# 3. Non-structural dual-render frame extraction (Mesh & 3DGS):
backend/.venv/bin/python 06_mesh_elaboration/extract_non_structural_frames.py \
  --mesh-path current_scene/scan_0/mesh.ply \
  --labels-path current_scene/scan_0/segmentation/triangle_class_labels.npy \
  --structural-json current_scene/scan_0/segmentation/class_metadata.json \
  --workspace current_scene/scan_0 \
  --output-dir current_scene/scan_0/mesh_elaboration/non_structural_frames \
  --gs-output-dir current_scene/scan_0/mesh_elaboration/non_structural_frames_gs

# 4. Generate a parametric dimensioned floor plan symbol (e.g. Bed, 1.8m wide x 2.0m long):
backend/.venv/bin/python 06_mesh_elaboration/generate_floorplan_symbols.py 18 1.8 2.0 -o /tmp/bed.svg

# 5. Finalize ceiling plane contours after Step 3 and wall contour fitting:
backend/.venv/bin/python 06_mesh_elaboration/regularize_ceiling_planes.py \
  --workspace current_scene/scan_0
```

## First-party callable reference

### Input and orchestration

| Callable | Inputs → processing → outputs |
| --- | --- |
| `mesh_elaboration_inputs._is_scene` | Path → checks for `mesh.ply`, `sparse/0`, or `segmentation/` → boolean. |
| `discover_scans`, `resolve_scene` | Workspace → discovers `scan_N` directories or validates single scene → scene paths. |
| `resolve_mesh`, `resolve_labels` | Scene and optional override → resolves Stage 04 `mesh.ply` or Stage 05 `triangle_class_labels.npy` → validated path. |
| `resolve_structural_metadata` | Scene and optional override → resolves `class_metadata.json`, `segmentation.json`, or fallback → validated path. |
| `resolve_segmentation_output`, `resolve_output_dir` | Scene and optional path → resolves Stage 05 package or creates Stage 06 output directory → path. |
| `step_mesh_elaboration.run_pipeline_for_scene`, `main` | Workspace or CLI flags → sequences elaboration stages, logs timings, updates pipeline stats → exit code. |
| `align_structural_vertical.align_structural_instances`, `axis_align_structural_mesh.axis_align` | Legacy frame-alignment utilities; not run in production because Phase 04 owns the world-frame transform. |

### Structural extraction and plane fitting

`enclose_wall_instances.py` is the production entry point for the full wall enclosure. It reads wall instance PLYs from `structural_instances_manifest.json`, runs the following operations in a temporary directory, validates shell topology, and publishes `wall_enclosing_shell.ply`. If the fitted walls do not enclose the 30 cm floor-projection slice, it extends the smallest already-supported wall panel whose extension closes that slice; the inferred panel and height range are recorded in `inferred_floor_closure_planes`. If no single supported panel closes the contour, the layout remains an error. Before publication, `planar_pattern_regularization.py` uses the assembler's final one-owner triangle assignment to find periodic relief only when its spacing is consistent across a contiguous run of support bands. It replaces the noisy average profile with one repeated rectangular, cylindrical, or triangular rib profile using shared spacing, width, and height across the fitted plane, then projects the relief toward the room-facing side; displacement fades at the plane edge and every mesh boundary vertex stays fixed so panel joins remain closed. The Phase 06 Step 4 wall fit applies the same regularizer to the final contour-sized wall panels. This is geometric repetition detection, not semantic material classification. For appearance, it applies softened scene color adaptation and smooth lighting correction, removes thin spatial color outliers, then classifies each surface as a plain color, a distinct connected chromatic region, or a repeating/spatially coherent texture. Periodic visual textures are synthesized from a seam-corrected tile only inside the longest contiguous region with repeat evidence; outside it, the cleaned support colors are retained. A narrow edge correction (at most eight raster cells) makes opposite tile borders meet without broad blending. Neutral regions with similar chromaticity and up to 2.5× brightness difference are treated as lighting variation; stronger neutral differences and chromatic boundaries remain separate. A color split is retained only when supported colors on both sides agree with one straight in-plane cut at least 90% by area and the boundary is observed across most supported rows or columns; missing support is filled according to that cut only after the split passes. Scattered patches and unsupported regions collapse to the plane's base color. Cross-plane color matching cannot create a new split on an otherwise uniform plane. Highlight handling preserves color ratios instead of clipping channels independently. Unstructured color noise becomes a solid surface. Unsupported samples are filled from the nearest supported edge rather than wrapped across a gap; a supportless connector selects one nearest supported panel and extends that panel's regularized appearance across the connector, avoiding face-by-face switching between different materials. It does not transfer raw support colors. Compatible touching uniform panels share a color estimate, while panels with coherent color divisions remain independent. Blender's PLY importer ignores face colors, so the closed-shell audit runs on welded geometry first, then vertices are duplicated only along color seams so each face gets a flat vertex color with no interpolation halo; duplicated seam vertices remain at identical coordinates. In Blender's Solid viewport, select **Color → Attribute** to display its `Col` attribute. Per-panel pattern and color diagnostics are included in the optional `wall_planes_manifest.json`. The builder overwrites the shell on rerun and writes no colored-region, preview, owner-map, or intermediate mesh artifacts. The optional `--manifest-out` is used by the Stage 06 runner to write panel bounds for the later floor/ceiling footprint logic; direct use omits that metadata by default. If no wall support survives region filtering, the prior shell is removed and the pipeline receives an empty wall manifest.

`grow_wall_regions.py` remains useful on its own for inspecting the region labels. The grower starts up to 128 deterministic random seeds per batch, sampled proportional to triangle area, then grows fronts across shared mesh edges. A triangle joins when its normal is within 35° of the current area-weighted cluster normal and all its vertices lie within 15 cm of the cluster's continuously updated centroid plane. When fronts meet, clusters merge under those same limits. After growth, a region with fewer than 10,000 triangles is absorbed by its edge-neighbor with the closest average normal, provided the closest orthogonal plane offset is at most 15 cm; if none qualifies, it remains separate and is painted solid black. The final pass only merges clusters sharing triangle edges and meeting the 35° normal and 15 cm orthogonal plane limits. Disconnected mesh components remain separate regardless of how well their infinite planes align, preventing distant patches from joining through a transitive cluster chain. The colored copy, face-aligned `wall_regions_triangle_labels.npy` (which preserves IDs for black regions), and `wall_regions_manifest.json` are produced only when this diagnostic is run directly.

`fit_wall_region_planes.py` consumes those outputs and fits one robust plane per non-black region. Regions marked `painted_black` are excluded from fitting and receive no plane; their counts are reported separately. For the remaining regions, it minimizes area-weighted absolute orthogonal vertex distance using iteratively reweighted least squares, trimming at most 1% of the complete input wall PLY triangle count while keeping at least one triangle in each fitted region. This trim budget is global, not 1% per cluster. The fitted normals determine two shared horizontal directions using an area-weighted fourfold angular average, plus the Y axis for horizontal support. Before snapping, any region whose normal is more than 30° from its nearest shared axis is excluded and counted as unassigned. The remaining planes snap to their nearest direction, and their offsets are refit by weighted medians of their support vertices. Same-direction planes are merged only when the group's full offset range is strictly under 0.15 scene units (15 cm for metric scans). A merge is also rejected if its proposed weighted-median offset would come within 15 cm of another plane when at least one merge member was more than 15 cm from it before the merge; rejected members remain separate. This prevents a merge from pulling a previously distant plane into the same close-offset band. Each final plane is then translated along its snapped normal to one of two one-sided offsets that leave at most 1% of that plane's source triangles outside. A triangle is outside if any vertex crosses the selected exterior side. The side is chosen by which candidate leaves more area of the complete input wall mesh on its interior side; area-weighted absolute vertex-distance sum breaks ties. The aggregate outside count is checked against 1% of the original input PLY triangle count. Black and angle-rejected regions stay unassigned. A merged plane takes the color of its largest supporting region. The manifest retains the original fits in `fitted_planes_before_snap`, accepted snapped planes in `snapped_planes_before_merge`, and final merged planes in `planes`, with excluded regions, rejected merges, source IDs, enclosure offsets, interior-mesh fractions, and outside counts. It exports `wall_region_planes.glb` and `wall_regions_with_planes.ply`; both show the final snapped/merged/enclosing planes in region colors at 50% opacity. The combined PLY contains the original colored region mesh plus appended plane previews and can be opened as one mesh in Blender or another PLY viewer. Preview rectangles show enclosed-support bounds with a small border; manifest equations describe the infinite planes.

`fit_wall_region_planes.py` and `assemble_wall_region_planes.py` are also callable separately when inspecting intermediate fits. The assembler keeps each winning local-support patch at its observed height, trims the competing panel only over that footprint, restores its remaining coverage, and adds vertical and horizontal transition faces. The union of panels and transitions is checked as one side shell over every height band; individual patch panels are not each required to span floor-to-ceiling. The production combined builder uses this same assembly implementation and fails if any final height band is open or the shell has non-manifold edges. Phase 06 derives these bounds from wall vertices rather than ceiling vertices.

| Callable | Inputs → processing → outputs |
| --- | --- |
| `extract_structural_meshes.component_face_indices` | Triangle cluster labels, count → splits indices preserving face order → list of index arrays. |
| `closest_major_instance` | Points, k-d trees → queries nearest major component tree → index and distance. |
| `load_door_window_associations` | Path to valid class associations JSON → filters classes mapped to door (1) or window (2) → association dictionary. |
| `load_associated_object_instances` | Associations, segmentation directory → loads instance PLYs from `3d_masks` or `Instances PLYs` → mesh data list. |
| `choose_structural_target` | Recognized door/window vertices and wall instances → finds the nearest wall by vertex distance → target and distance. |
| `merge_associated_instances` | Staging directory, manifest, associated instances → appends door/window triangles to structural meshes → updated manifest. |
| `extract_structural_meshes.main` | Source mesh, labels, metadata JSON → exports isolated structural PLYs (`wall`, `floor`, `ceiling`, `stair`, `column`, `ramp`) with hole flags and manifest → manifest and files. |
| `fit_wall_planes._remove_horizontal_faces` | Vertices, triangles, tolerance → removes degenerate and horizontal faces ($<30^\circ$) → filtered triangles and normals. |
| `fit_wall_planes.main` | Wall instance meshes → assigns unique support, fits and optionally subdivides planes, enforces the 2% outside cap, reconciles corners, and requires strict side closure → wall plane manifest and PLYs. |
| `fit_floors_ceilings_planes._weighted_plane` | Points, area weights → covariance eigensystem solution → plane normal and offset. |
| `fit_floors_ceilings_planes.fit_planes` | Instance meshes, tolerance, max tilt → fits horizontal and ramp support planes (with 4.0° tilt horizontal snap) → `floors_ceilings_planes_manifest.json`. |

### Pattern regularization, geometry, and texture synthesis

| Callable | Inputs → processing → outputs |
| --- | --- |
| `recognize_and_regularize_patterns.generate_surface` | Unit metadata, optional footprint → evaluates plane equations and meshes 2D polygons → vertices and faces. |
| `recognize_and_regularize_patterns.regularize_scanned_mesh_to_pattern` | Vertices, faces, plane parameters, pattern parameters → regularizes periodic ceiling beams → regularized mesh. |
| `recognize_and_regularize_patterns.surface_vertical_bounds` | Surface units, wall planes → determines floor/ceiling elevation limits with fallback for partial scans → min and max height. |
| `recognize_and_regularize_patterns.build_watertight_slab` | Floor unit, ceiling unit, overlap polygon → sweeps perimeter skirt and stitches top/bottom → watertight closed solid. |
| `structural_geometry.pattern_frame` | Normal, offset, inward vector, transverse → computes orthonormal 3D frame for beam patterns → basis vectors. |
| `structural_geometry.perimeter_pattern_axes` | Normal, longitudinal vector, horizontal axes → lifts nearest wall axis into plane → transverse and longitudinal axes. |
| `structural_geometry.snap_small_excursions` | Points, wall offsets, tolerance → snaps near-boundary coordinates to walls within 15 cm → snapped coordinates. |
| `surface_footprints.orthogonal_ring` | 2D points, tolerance → fits alternating orthogonal straight line segments → corners array. |
| `level_footprints.fit_ring` | Initial ring, point samples, min edge → SLSQP optimization of orthogonal line offsets → fitted polygon. |
| `apply_structural_textures.synthesize_floor_unit_tile` | Floor mesh, chromatic adaptation → generates seamless parquet texture → UV-mapped tile. |
| `apply_structural_textures.synthesize_ceiling_unit_tile` | Ceiling mesh, beam parameters → generates periodic beam strip texture → textured mesh. |
| `apply_structural_textures.main` | Regularized shell, texture options → applies parquet/beam/plaster colors and rewrites `wall_regularized.ply` and `all_structural_regularized.ply`. |

### Non-structural framing, similarity, and symbols

| Callable | Inputs → processing → outputs |
| --- | --- |
| `extract_non_structural_frames.compute_canonical_camera` | Instance vertices, face normals, clearance constraints → calculates optimal camera position and orientation → camera extrinsics. |
| `extract_non_structural_frames.render_instance_mesh` | Mesh, camera, offscreen renderer → renders high-resolution frame of isolated instance → RGB image and purity mask. |
| `extract_non_structural_frames_gs.render_gaussian_splats` | Splats, camera, resolution → CUDA/ROCm Gaussian Splat rasterization → rendered GS image. |
| `run_non_structural_category_similarity.main` | Rendered instance frames, 28 indoor categories → computes EmbeddingGemma similarity and specificity → `valid_class_associations.json`. |
| `project_valid_instances_by_floor.main` | Valid associations, floor outlines, object meshes → plots annotated bounding boxes on floor plans → PNG and PDF floor plans. |
| `generate_floorplan_symbols.sized_svg` | Class ID (1–28), width (Y), length (X) → generates procedural dimension-constrained architectural SVG → SVG string. |

## Quality and speed opportunities

| Opportunity | Evidence in code | Safe direction |
| --- | --- | --- |
| Offscreen rendering throughput | `extract_non_structural_frames.py` initializes Filament/EGL contexts and performs CPU PNG encoding. | Retain a single persistent `OffscreenRenderer` across instances, batch camera query tensors, and use multi-threaded non-blocking PNG compression (`ThreadPoolExecutor`). |
| Wall plane RANSAC speed | `fit_wall_planes.py` samples triangle candidates iteratively over thousands of faces. | Subsample dense coplanar support patches using normal-binned seeds before plane equation refinement; reuse vectorized NumPy broadcasting for residual testing. |
| Authentic texture synthesis | `apply_structural_textures.py` re-meshes and subdivides surfaces down to 8 cm edges (`subdivide_to_size`). | Restrict remeshing strictly to non-planar or relief areas (beams); keep flat walls and floors coarse with standard UV coordinate mapping. |
| DAE/CAD vector export | Floor plan symbols currently export to SVG. | Add lightweight DXF/IFC export using the aligned wall planes and 2D floor contours for direct BIM integration. |

## Redundant and legacy paths

- **Steps 13, 14, and 15 (Retired)**:
  - `enhance_structural_textures.py` (Step 13): SEESR Turbo super-resolution was experimental, high latency, and introduced synthetic hallucinations on architectural textures.
  - `group_floors_ceilings_levels.py` (Step 14): Elevation clustering was superseded by level decomposition within 2D residual floor planning and pattern regularization.
  - `fit_wall_perimeter_infill.py` (Step 15): 5 mm morphological Gaussian raster infill was replaced by direct vector boundary regularization.
- **Stage 07 Delegation**:
  - `build_rigid_baseline.py`, `group_touching_instances.py`, `fit_contact_groups_inside.py`, and `extract_cross_section_views.py` (formerly Steps 8–11) have moved to [`backend/07_objecs_alignment/`](../07_objecs_alignment/README.md).
  - Diagnostic tools (`analyze_new_step.py`, `plot_new_step.py`, `view_new_step.py`, `render_side_by_side_all.py`, `translate_instances_inside.py`, `build_instance_candidates.py`, `analyze_instance_contacts.py`) have also moved to Stage 07.
- **Compatibility Aliases**:
  - `scene_inputs.py`: An exact alias of `mesh_elaboration_inputs.py` preserved for backward-compatibility with external scripts.

## Verification

Run the comprehensive Stage 06 test suite:

```bash
# Run all end-to-end and unit tests for Stage 06
backend/.venv/bin/pytest \
  backend/tests/test_mesh_elaboration_pipeline.py \
  backend/tests/test_geometry_pattern_recognition.py \
  backend/tests/test_extract_non_structural_frames.py \
  backend/tests/test_floorplan_symbols.py -v
```

### Recorded test results

Focused Stage 06 verification is run against the current repository state; historical suite counts are not a current result. It covers:
- Full execution of the production stages, with supported wall-plane fitting run directly on the Phase 04-aligned geometry.
- Column and ramp extraction into distinct `column_instNNN.ply` and `ramp_instNNN.ply` components.
- Wall-free patio scenes with columns, floor, and ceiling correctly recovering dominant orientation without walls.
- Strict 4.0° horizontal tilt snap tolerance in `fit_floors_ceilings_planes.py`.
- Closed, zero-boundary watertight slab synthesis in `recognize_and_regularize_patterns.py`.
- Dimension-constrained parametric SVG symbol validity across all 28 indoor classes.

### Visual verification checklist
- Inspect `all_structural_regularized.ply` in MeshLab or Blender to confirm that walls terminate cleanly at ceiling/roof profiles without floating slivers or gaps.
- Inspect `floors_ceilings_planes_manifest.json` to verify that floor and ceiling planes with tilt $\le 4.0^\circ$ were snapped to strict horizontal.
- Inspect `all_structural_regularized.ply` after the texture pass to verify parquet and plaster color synthesis.
