# Stage 05 — semantic segmentation

Stage 05 lifts class-agnostic 2-D regions from registered photographs onto the sealed Stage 04 mesh. It produces persisted face-label evidence, a geometry-aware refinement result, and downstream delivery files. Face-label IDs remain opaque; a separate final EmbeddingGemma pass evaluates every rendered source class, including structural classes, against the fixed 28-category vocabulary. It writes one class-level category association when the existing automatic confidence filters accept it; the structural role remains unchanged.

The stage keeps geometry and upstream products read-only. It may assign a face an ID, leave it unknown, associate compatible labelled pieces into an instance, and derive a structural role as a separate, fallible analysis. A colour in `mesh_segmented.ply` only visualizes a face ID.

## Stage files

| Component | Responsibility | Runtime status |
| --- | --- | --- |
| [`step_segmentation.py`](step_segmentation.py) | Discovers scans, resolves locked inputs, coordinates 2-D proposals, projection, regularization, delivery, and final object-category matching. | Production entry point. |
| [`scene_inputs.py`](scene_inputs.py) | Shared scene discovery, input resolution and validation. | Production input boundary. |
| [`fast_sam2_generator.py`](fast_sam2_generator.py) | SAM 2 inference, letterboxing and batched point prompts. | Production model adapter. |
| [`run_sam2_proposals.py`](run_sam2_proposals.py) | Standalone SAM proposal experiments, archives and overlays. | Optional diagnostics; not called by the production entry point. |
| [`run_full_scene_pipeline.py`](run_full_scene_pipeline.py) | All-frame SAM/DINO inference, appearance-based memory-bank propagation and consolidation. | Production 2-D producer; frozen archives are reused for 3-D-only reruns. |
| [`step2_project_masks_to_3d.py`](step2_project_masks_to_3d.py) | Raycasts labelled pixels to mesh faces and writes persisted multi-view vote evidence. | Production projection worker. |
| [`mesh_prior_refinement.py`](mesh_prior_refinement.py) | Evidence-preserving graph label refinement. | Production regularizer shared by the projection/orchestration path. |
| [`geometric_regularization.py`](geometric_regularization.py) | Evidence refinement and compact-part association; also retains older cleanup helpers. | Production adapter; the old normal/ID-based passes are outside its production sequence. |
| [`detect_structural_classes.py`](detect_structural_classes.py) | Uses the Phase 04 fixed +Y up-axis for gravity/height analysis, role inference and connected-instance extraction; also retains older cleanup helpers. | Production analysis/extraction; standalone cleanup is a separate legacy path. |
| [`deferred_assignment.py`](deferred_assignment.py) | Reads locked infill flags/colors, fits deferred surface planes, matches neighboring RGB/geometry and records same-surface associations. | Final assignment after observed-region cleanup. |
| [`export_delivery.py`](export_delivery.py) | Writes the segmented PLY, extracted meshes and JSON manifests. | Production delivery writer. |
| [`extract_non_structural_frames.py`](extract_non_structural_frames.py), [`extract_non_structural_frames_gs.py`](extract_non_structural_frames_gs.py) | Render delivered object instances from the sealed mesh and, when available, the sealed Gaussian Splat model. | Called by the final Stage 05 pass. |
| [`run_non_structural_category_similarity.py`](run_non_structural_category_similarity.py), [`step_semantic_similarity.py`](step_semantic_similarity.py) | Embed rendered instance images and the fixed 28-category vocabulary with EmbeddingGemma 2; write per-class matches and reports. | Called by the final Stage 05 pass. |
| `panoptic3d*.py`, `planar_gap_completion.py`, `surface_label_refinement.py`, `wall_orientation_refinement.py` | Frozen-prior, evaluation and optional geometry-refinement helpers. | Available only when explicitly called. |
| Remaining `run_*`, `clean_*`, `refine_*`, `inspect_*`, `test_*` files | Earlier experiments, diagnostics, visualizers and tests. | Retained for investigation; not production prerequisites. |

## Pipeline flow

```text
Stage 00–01: registered RGB images + refined COLMAP cameras
Stage 02: optional depth maps and alignment metadata
Stage 04: sealed mesh.ply (+ mesh.json)
                         │
                         ▼
2-D proposals / opaque source IDs → per-frame full-resolution label maps
                         │
                         ▼
raycast visibility + bounded per-view face votes → persisted evidence
                         │
                         ▼
evidence-constrained mesh graph refinement → face IDs or unknown
                         │
                         ▼
optional role analysis + physical-part association → delivery package
                         │
                         ▼
render delivered source classes, including structural classes → EmbeddingGemma fixed-category matching
                         │
                         ▼
scan_N/segmentation/ → Stage 06 and inspection tools
```

The graph cannot manufacture a semantic category from geometry. Ceiling/wall orientation is evidence for a correction only where compatible persisted observations support it; unsupported faces remain unknown. This avoids normal-only reassignment of beam faces and broad infill that gave every hole an arbitrary neighbouring ID.

## Input contract

The production entry writes only below each scan’s `segmentation/` directory. It never edits camera poses, images, depth maps or `mesh.ply`; Phase 04 has already aligned the COLMAP and optional depth poses in their existing files.

```text
current_scene/
└── scan_N/                         # one scan, or pass a single-scene root
    ├── images/<COLMAP image name>  # Stage 00/01 registered RGB
    ├── sparse/0/
    │   ├── cameras.txt             # Stage 01 intrinsics
    │   ├── images.txt              # poses/tracks, aligned in place by Phase 04
    │   └── points3D.txt             # sparse points, aligned in place by Phase 04
    ├── depth/                      # optional Stage 02 support
    │   ├── poses_da3.npz            # camera poses aligned in place by Phase 04
    │   ├── depth_maps/<stem>.npy
    │   └── depth_alignment.json
    ├── mesh.ply                    # Stage 04 sealed authoritative mesh
    ├── mesh.json                   # Stage 04 report, if available
    └── segmentation/               # this stage's output only
```

| Input | Required | Validation and use |
| --- | --- | --- |
| `mesh.ply` | Yes | Stage 04 mesh, read as indexed triangles and RGB. Face ordering is the identity of every evidence row and label array. |
| `images/` | Yes | Registered RGB images. Paths, dimensions and unique stems must agree with COLMAP. |
| `sparse/0/cameras.txt`, `images.txt` | Yes | Stage 01 camera intrinsics and registered poses/tracks. Phase 04 transforms poses in place to match `mesh.ply`; registered-track counts remain unchanged. |
| 2-D label NPZ files | Produced or supplied | Map is integer, nonnegative and exactly `(camera.height, camera.width)`; `0` means uncovered/unknown. Source IDs are opaque. |
| `depth/` | No | Optional calibrated support for visibility/conflict checks. Absence must not be replaced with invented depth. |
| mesh-to-camera transform | No | Identity for the Stage 04 contract, or an explicit finite proper rigid `4×4` transform. |

Legacy fallback paths under `05_semantic_segmentation/inputs/` are read-only compatibility inputs. They are not an upstream contract and must not be regenerated or preferred over a valid scan-local Stage 04 mesh.

### Coordinate and label invariants

- Cameras use COLMAP world-to-camera extrinsics. Phase 04 transforms camera poses alongside the mesh and sealed Gaussian model; Stage 05 reads aligned poses directly from `sparse/0/images.txt`. The transform in `mesh.json` lets Stage 03 recover source-frame camera views for retraining.
- Face labels, `face_class_votes.npz`, `face_class_source_ids.npy`, face counts and vertex/triangle ordering must describe the same mesh revision.
- Votes are accumulated per camera/view group, with unit ray directions and bounded per-face/per-frame contribution. A close camera must not win merely because its direction vector is longer or it samples more pixels.
- Raw projection output is a local evidence argmax. It is not globally flipped to help an underrepresented ID. No-vote faces carry the unknown sentinel.
- A source ID may be split by visibility and a physical object may contain several source IDs. Physical association requires mesh contact/seams/support evidence; numeric adjacency is insufficient.

## Processing

1. **Resolve and validate.** `scene_inputs` accepts one scene or discovers numbered `scan_N` directories. A helper that needs one scene rejects ambiguity.
2. **Make 2-D observations.** SAM proposes class-agnostic regions. DINO and the historical memory bank can propagate opaque IDs across views; neither produces a fixed vocabulary such as wall, ceiling or horse.
3. **Project observations.** The worker raycasts pixels to faces, rejects invalid labels/weak views as configured, weights interior pixels and incidence, reduces each face's evidence within a view, then persists source-ID order and face-by-ID votes.
4. **Refine from recorded evidence.** `refine_graph_labels` minimizes a graph objective over candidate IDs using multi-view unary evidence, shared edges, crease/colour/orientation terms and optional trusted references. It uses no hard-coded class IDs and does not force coverage.
5. **Associate and describe.** Contact-compatible foreground pieces may be associated as one physical instance. Structural roles are inferred separately and remain hypotheses, particularly on noisy mesh caps, beam faces and thin mounted objects.
6. **Enforce final constraints.** Wall owners cannot retain faces within 20° of horizontal, upward or downward: `abs(unit_normal · gravity_up) >= cos(20°)`. Edge-connected upward and downward wall patches are measured separately; patches with at least **2 m²** are assigned floor and ceiling classes, respectively. Smaller forbidden wall patches take a touching non-wall owner where possible, then the smallest edge-connected regions merge iteratively into eligible neighbors, preferring coherent shared boundaries. Component sizes and contacts update after each merge; touching regions of the same owner coalesce immediately. Horizontal regions cannot merge back into walls. The regularization report records promoted patch areas, triangle counts, class IDs and whether each retained its role after the minimum-size pass. Every exported instance still must contain at least **2000 triangles**, including any explicitly associated same-surface pieces, with no evidence exception; a promoted patch below that count may merge into a neighboring non-wall class. This production constant is locked; there is no threshold override. A fragment with no eligible touching owner is deferred to the final geometry/RGB pass. This is a count policy on the existing tessellation, not a physical-size guarantee.
7. **Assign inferred surfaces last.** Stage 04 `is_hole_fill` faces are excluded from observed refinement and role inference, even if raycasting happens to hit them. Their votes are not semantic observations. After the minimum-size cleanup, the final pass partitions deferred surfaces by creases, source-RGB edges and wall-angle eligibility. It fits area-weighted regional planes and compares boundary owners using plane residual, normal agreement and source RGB (`fill_r/g/b` on infill), then grows competing owners through the piece graph. One connected infill patch can therefore receive multiple owners on different surfaces. Accepted observed owners stay fixed. Remaining pieces can associate with a nearby accepted surface only within six median face-neighbor spacings, two spacings of its plane, 25° normal agreement and RGB distance ≤0.25. Explicit same-surface associations join the existing instance for counting; they never alter source mesh geometry. Truly incompatible/unseeded pieces remain unresolved rather than taking a distant arbitrary label.
8. **Deliver.** The writer preserves sealed mesh geometry, stores face `class_id`/`instance_id`, emits instance submeshes and manifests, and can expose compatibility links for consumers expecting `3d_masks/`.
9. **Name source classes.** Render each delivered source class from the sealed mesh and, when available, the Gaussian Splat model. Structural classes are included. EmbeddingGemma compares the views with all 28 categories, aggregates the evidence to one association per source class, and omits classes that fail the existing automatic ambiguity or specificity filters. These category labels are independent of structural roles; the face-label IDs remain opaque.

## Outputs and downstream interface

All produced files are under `scan_N/segmentation/`.

| Output | Producer | Contents and consumer contract |
| --- | --- | --- |
| `panoptic_data/npz_data/*.npz` | 2-D pipeline | Per-view masks/labels and source observations. IDs are evidence, not named categories. |
| `face_class_votes.npz` | Projection | Face-by-source-ID vote evidence plus source-ID ordering required to decode columns. |
| `face_class_source_ids.npy` | Projection | Column-to-opaque-source-ID mapping for votes and local labels. |
| `triangle_class_labels_raw.npy` | Projection | Face-order-aligned local evidence argmax; unknown persists where no valid vote exists. |
| `projection_stats.json` | Projection | Input/evidence counts, settings and `projection_version`; use this to reject stale caches. |
| `triangle_class_labels_observed.npy` | Final constraints | Accepted observed-region labels before any late patch assignment; all Stage 04 infill is unknown here. |
| `deferred_instance_links.npy` | Late assignment | Source-face pairs recording bounded same-surface semantic instance associations across mesh seams. They are not source mesh edges. |
| `triangle_class_labels.npy`, `regularization.json` | Regularization | Face-order-aligned refined IDs and solver/change/unknown-face metrics; unknown may remain. |
| `mesh_segmented.ply` | Delivery | Same face positions and face order as the sealed mesh, with display vertices split at class boundaries so colours never blend across a face label. Vertex count/indices can therefore differ. Exact labels are face `class_id`/`instance_id` properties. |
| `Instances PLYs/` | Delivery | Per-instance submeshes with original mesh RGB. Role-folder filenames are unique across all classes, preventing accidental overwrite. These are exports, not an alternative source of segmentation truth. |
| `class_metadata.json`, `segmentation.json` | Delivery | Roles, confidence/provenance where available, bounds/counts/areas and relative paths. `segmentation.json` records `algorithm_version: evidence-geometry-v5`, locked input hashes, unknown-face count and `regularization.json`. |
| `non_structural_similarity/valid_class_associations.json` | Category matching | Accepted fixed-vocabulary association per opaque source class, including structural classes; uncertain classes are omitted automatically. Structural roles are not changed. Consumed by Stage 06 door/window handling and Stage 08 floorplan annotation. |
| `non_structural_similarity/` | Category matching | Category list, embedding caches, per-view top matches, specificity baseline and comparison images. |
| `non_structural_category_frames/` | Category matching | Rendered views for all eligible source classes, including structural classes, plus optional Gaussian Splat views used as classifier inputs. |
| `3d_masks/` | Delivery | Optional compatibility links to delivered instance meshes. Its manifest uses the same class-local, one-based `instance_index` as the face metadata. |

Do not infer a role from palette colour, ID value, output directory or filename. Downstream code must treat unknowns and missing confidence as meaningful data rather than filling them by nearest neighbour.

## Running

The production policy is fixed in `step_segmentation.py`: **2000 triangles per exported instance**, wall exclusion within **20° of horizontal**, and inferred-surface assignment after observed-region cleanup. Each discovered `scan_N` writes one package to `scan_N/segmentation/`. `MIN_INSTANCE_TRIANGLES` is the single production threshold constant.

Run from `backend/` using the existing environment:

```bash
# Process every scan_N using the locked policy and standard output path
.venv/bin/python 05_semantic_segmentation/step_segmentation.py --workspace current_scene

# Rebuild the 3-D result for every scan, reusing valid 2-D/projection evidence
.venv/bin/python 05_semantic_segmentation/step_segmentation.py \
  --workspace current_scene --force

# Rebuild one scan
.venv/bin/python 05_semantic_segmentation/step_segmentation.py \
  --workspace current_scene/scan_0 --force

# Explicitly regenerate model observations and projection when necessary
.venv/bin/python 05_semantic_segmentation/step_segmentation.py \
  --workspace current_scene/scan_0 --recompute-2d --recompute-projection
```

Completed compatible scans resume automatically. `--force` reruns refinement, constraints, late assignment and delivery; it reuses valid evidence unless the corresponding recompute flag is supplied. Cache validation includes algorithm version, input hashes and the locked threshold. `--max-frames` is a diagnostic subset, not a complete scene.

The production entry no longer accepts `--min-instance-size` or `--output-dir`. It does not create threshold-comparison folders. Existing `segmentation_min_*` packages from earlier manual comparisons are historical artifacts and are left untouched. Standalone research workers may retain their own diagnostic CLI options; they are not the production workflow.

## Locked production policy

| Policy | Enforcement |
| --- | --- |
| Minimum instance size | Fixed **2000 triangles**, iterative contact-based merging; no evidence bypass. |
| Wall orientation | Neither upward nor downward faces within **20° of horizontal** can remain wall-owned. |
| Stage 04 infill | `is_hole_fill` faces and their vote rows are excluded from observed refinement and role inference. |
| Final patch assignment | Split deferred surfaces at geometry/RGB boundaries, fit regional planes, score neighboring owners using plane residual, normals and source RGB. |
| Disconnected same-surface pieces | Only bounded coplanar/color-compatible associations; recorded separately from source mesh topology. |
| Final acceptance | Verify all positive faces belong to exported instances meeting the minimum and the wall-angle rule still holds after late assignment. |
| Unresolved geometry | Preserve explicitly unknown when no compatible owner is available; never invent distant ownership or remove source geometry. |
| Output | Exactly `segmentation/` under each scan; no threshold-named production packages. |

## First-party callable reference

“Returns” includes files written by CLI/orchestration functions. Underscore-prefixed functions are internal helpers documented because they affect results.

### Input and orchestration

| Callable | Inputs → processing → outputs |
| --- | --- |
| `scene_inputs._is_scene` | Path → checks image/COLMAP layout → boolean. |
| `discover_scans`, `resolve_scene` | Workspace → validate/sort scans or resolve exactly one → scene paths. |
| `stage_directory`, `resolve_stage_output` | Workspace and optional child → resolve/create output path → path. |
| `resolve_mesh`, `resolve_mesh_report`, `resolve_sparse_dir`, `resolve_images_dir`, `resolve_depth_dir` | Scene and optional override → validate locked Stage 01/02/04 locations → paths or optional value. |
| `_camera_source_sizes`, `load_scene_cameras`, `registered_images` | COLMAP text/images/resolution → validate dimensions/stems → camera dictionaries or sorted paths. |
| `colmap_point_counts` | COLMAP `images.txt` → parses tracks → image-to-triangulated-count map. |
| `load_mesh_transform`, `resolve_mesh_transform` | Optional NPY/workspace → validate rigid transform → `4×4` matrix or identity. |
| `validate_labels`, `dino_grid` | Label map/camera or image geometry → validate map / calculate ViT patch grid → map or `(rows, columns)`. |
| `step_segmentation.run_segmentation_pipeline_for_scan`, `main` | Scan/CLI settings/context → coordinate phases and completion checks → delivery manifest/exit status. |

### 2-D proposals and propagation

| Callable | Inputs → processing → outputs |
| --- | --- |
| `FastSAM2Generator` | Stateful SAM 2 adapter. Its constructor loads/configures the model; the methods below prepare prompts and return proposals without assigning category names. |
| `FastSAM2Generator.__init__`, `_prepare_grid`, `_transform_coords` | Model settings/image geometry → construct grid prompts/model coordinates → generator/points. |
| `FastSAM2Generator._decode`, `_infer`, `generate`, `generate_complete`, `warmup` | Image/features/prompts → SAM decoding, optional gap prompts, filtering and compilation/cache warmup → masks/metadata/logits. |
| `run_sam2_proposals.main` | Image workspace/settings → proposals, optional filtering/overlap resolution → NPZ/overlay/JSON/summary. |
| `run_sam2_proposals.guided_filter_rgb`, `run_pipeline_bench_first50.guided_filter_rgb` | RGB guide and scalar/boolean mask → edge-aware local smoothing → filtered mask; benchmark copy is historical. |
| `clean_masks.shape_reasons`, `clean_frame`, `main` | Proposal masks/labels/thresholds → identify scattered, thin or low-solidity shapes → cleaned archives/overlays/summary. |
| `run_dinov2_patches.group_neighbors`, `adjacent_similarities`, `grow_cluster_means`, `grow_and_merge_grid_seeds`, `merge_grid_rounds`, `merge_small_regions`, `main` | Patch embeddings/threshold CLI → spatial region growth/cleanup → label grids and visual/NPY outputs. |
| `run_dinov2_merge.main` | Historical one-frame input → DINO merge experiment → experimental outputs. |
| `get_adaptive_gamma` | Mask area fraction → memory-bank decay → float. |
| `get_class_color` | Opaque positive ID → deterministic display BGR/RGB → colour pair. |
| `find_best_label_anchor`, `create_mask_overlay` | Masks/labels/image → interior label position and overlay → point/image. |
| `GlobalMemoryBank` | Stateful historical prototype registry used by both full-scene and propagation experiments; it owns the entries updated by its methods below. |
| `GlobalMemoryBank.__init__`, `register_initial_frame`, `get_bank_matrix`, `get_dormancy_threshold`, `update_matches`, `register_new_classes` | Embeddings/areas/frame → maintain prototype registry and claims → updated registry/matches/diagnostics. |
| `render_and_save_frame`, `create_final_triangular_matrix`, `create_filtered_matrix_plot`, `create_end_of_pass_triangular_matrix`, `create_pre_merge_highlighted_matrix`, `create_consolidated_triangular_matrix`, `create_cross_matrix_plot` | Render/class-similarity data → overlay/archive or matrix image/PDF. |
| `run_full_scene_pipeline.main` | Scan photographs/COLMAP support and CLI → SAM proposals, overlap cleanup, DINO pooling, within-frame merges, global-memory propagation, post-hoc merges and rare-ID cleanup → `panoptic_data` archives/overlays/summary/cache. |
| `run_global_memory_id_propagation.main`, `run_id_propagation.main`, `run_post_hoc_memory_consolidation.main`, `run_cleanup_and_final_matrix.main`, `run_cross_frame_similarity_bench.main` | Historical CLI input → named ID/diagnostic experiment → run-specific archives/figures. |
| `run_pipeline_staged_experiments.box_filter_gpu`, `guided_filter_gpu`, `run_experiment`, `main`; `run_pipeline_bench_first50.main`; `run_streamlined_grid12_bench.main` | Images/tensors/settings → GPU filtering or timed ablation → experiment metrics/artifacts. |
| `run_full_scene_pipeline.clean_carved_mask`, `run_streamlined_grid12_bench.clean_carved_mask` | Binary mask and area/thickness thresholds → removes small components and ribbons → cleaned mask. |

### Projection, evidence and frozen-prior reconstruction

| Callable | Inputs → processing → outputs |
| --- | --- |
| `step2_project_masks_to_3d.load_colmap_triangulated_points` | `images.txt` → strict shared parser → registered-point counts. |
| `step2_project_masks_to_3d._sha256_file` | File path → chunked SHA-256 calculation → hexadecimal input fingerprint. |
| `normalized_incidence_cosines` | Hit normals/ray directions → normalizes both and clamps finite cosine → weights. |
| `reduce_view_face_votes` | Pixel face IDs/source columns/quality → sums by face/source, divides by all face hits in the view → sparse indices/bounded values. |
| `local_evidence_labels` | Face-by-source vote matrix → local maximum where evidence exists → source-column labels or `-1`. |
| `compute_mask_distance_weights` | Integer 2-D map → boundary distance transform, zero unknown → pixel weights. |
| `build_dual_mesh_graph` | Triangles → shared-edge face adjacency → directed edges. |
| `absorb_isolated_islands` | Labels/graph/minimum size → legacy neighbouring-component cleanup → labels/counts; not enough evidence for production ownership alone. |
| `export_colored_ply`, `render_vector_pdf_matrix` | Mesh/labels or contention → diagnostic PLY/PDF. |
| `project_masks_to_3d`, `main` | Workspace/mesh/NPZ/settings → raycast and persist multi-view evidence/raw labels/statistics → metrics/files/exit status. |
| `run_mesh_quality_vote_step1.render_contention_matrix_image`, `render_top_contended_subset` | Vote/contention arrays and output path → full or selected contention diagnostic image → PNG. |
| `mesh_prior_refinement._sum_block_unaries`, `_component_orientation_penalties` | Unary terms/components/geometry → aggregate energy terms → tensors. |
| `corroborated_faces` | Final owners, sparse priors and source IDs → require matching evidence winner, ≥3 effective views and ≥75% confidence → evidence mask for optional area/span retention; it cannot bypass the hard triangle minimum. |
| `correct_confident_faces` | Labels/priors/view counts/source IDs/thresholds → copies only corroborated source IDs → labels. |
| `refine_graph_labels` | Labels/evidence/graph/face geometry/colour/optional reference → graph-energy updates → labels and solver diagnostics. |
| `panoptic3d.boundary_distance`, `observation_components`, `cached_mask_extent`, `extent_conflicts` | Labels/masks/depth state → boundary, observation and conflict data → arrays/records. |
| `view_groups`, `resolve_tracks`, `physical_ids`, `object_source_aliases` | Cameras/track support/instances → constrained correspondence/identity mapping → IDs and decisions. |
| `dual_graph`, `assign_faces`, `balance_votes`, `restrict_instance_votes`, `safe_removal`, `boundary_edges` | Mesh/vote/geometry evidence → graph, smoothed/rebalanced/restricted labels or safe removal → arrays/records. |
| `object_metrics`, `summarize_metrics` | Reference/prediction/visibility counts → precision, recall, IoU, leakage and boundary metrics → records. |
| `run_panoptic3d.write_json`, `digest`, `verify_inputs`, `output_directory` | Paths/data → serialize, fingerprint and validate inputs → files/digests/path. |
| `ray_scene`, `cast`, `cast_pixels`, `native_depth_points` | Mesh/camera/pixels/depth/transform → raycasting scene/hits/native 3-D points. |
| `load_depth_context`, `depth_evidence`, `pose_support`, `read_scene_review`, `region_templates`, `anchored_regions` | Stage inputs/review/masks → depth/reliability/review-region structures → context/records. |
| `collect`, `palette`, `export_mesh`, `reviewed_object_bounds`, `silhouette_conflicts`, `component_depth_evidence` | Workspace/evidence/review/mesh → persisted evidence, visual mesh and object support → files/arrays. |
| `solve`, `dense_completeness`, `evaluate`, `run_panoptic3d.main` | Evidence run/settings → solve/evaluate frozen-prior reconstruction → labels/meshes/reports/exit status. |

### Geometry, instances and delivery

| Callable | Inputs → processing → outputs |
| --- | --- |
| `compute_mesh_derivatives`, `build_dual_graph` | Mesh arrays → face derivatives and shared-edge CSR graph → arrays. |
| `reclaim_ceiling_bounding_slabs`, `reclaim_wall_encroached_furniture`, `reclaim_thin_wall_plates`, `align_structural_planes` | Labels/geometry/thresholds → legacy structural boundary repair → labels/counts. |
| `absorb_enclosed_subparts`, `absorb_flush_wall_instances`, `propagate_planar_faces`, `propagate_planar_boundaries`, `propagate_cabinet_faces`, `absorb_embedded_furniture_subparts`, `refine_all_boundaries`, `refine_class_boundaries.main` | Mesh labels/candidates/settings → legacy propagation/refinement → labels/reports. |
| `complete_planar_gaps`, `complete_geometric_regions`, `complete_bounded_gaps` | Labels/mesh graph/eligible patches/seeds → optional unknown-region completion → labels/fill report. |
| `reclaim_compatible_surfaces` | Labels/derivatives/trusted references → geometry/colour-compatible repair → labels/support report. |
| `wall_orientation_refinement._components`, `_nearest_nonwall`, `refine_wall_orientations` | Wall mask/graph/priors/up axis → components or cautious reassignment → labels/audit. |
| `run_complete_geometric_regularization` | Mesh, exact raw labels, sparse votes and ordered source IDs → measured-edge-scale graph refinement to convergence, then compact-part association → revised labels and solver/association metrics. Unknown faces stay unknown. |
| `regularize_foreground_connected_components` | Mesh derivatives/graph and structural exclusions → cut broad fitted support planes (including tilted surfaces), their lower lips and concave contacts; require compact extent and substantial contact; complete a compact source part only when ≥75% of its area joined one owner → component-local part merges/count. No proximity-only merge. |
| `regularize_disconnected_manifold_components`, `absorb_wall_mounted_coplanar_patches`, `suppress_micro_floor_instances` | Mesh/labels/graph/rules → legacy cleanup/metrics → labels/metrics. These rules are not called by the production regularizer. |
| `read_infill_attributes` | Sealed PLY and face count → read Stage 04 provenance and coplanar infill RGB in source order → flags and optional RGB. |
| `assign_deferred_patches` | Accepted labels, face geometry/RGB, adjacency, gravity and wall IDs → planar/curved pieces, competing geometric/color owner growth and bounded seam associations → final labels, metrics and instance links. |
| `enforce_instance_constraints` | Face labels, adjacency, normals, areas, gravity, wall IDs and minimum → promote connected ≥2 m² upward/downward wall patches to floor/ceiling, reject other near-horizontal wall ownership, then merge undersized contacting components or assign unknown → final labels and checked counts. |
| `detect_primary_structural_cids` | Mesh derivatives and observed labels → area fractions and relative height bands identify supported floor/wall/ceiling owners → primary IDs (zero when absent) and structural exclusions. Source-ID magnitude does not determine a role. |
| `suppress_micro_ceiling_instances`, `eliminate_horizontal_wall_triangles`, `absorb_flat_coplanar_wall_patches` | Legacy labels/mesh/graph and thresholds → reassign small/flat/horizontal structural patches → labels/counts. These are outside the production sequence. |
| `detect_structural_classes.get_class_color`, `export_colored_ply` | IDs/mesh → palette or coloured mesh → colours/file. |
| `create_cylinder_mesh`, `build_instance_badge_mesh` | Geometry/text/style → diagnostic primitives → vertices/faces/colours. |
| `read_scene_sizes`, `load_canonical_dino_embeddings`, `estimate_gravity_and_scene_bands`, `build_mesh_dual_graph` | Reports/embeddings/mesh → metadata, gravity/elevation bands, adjacency → records/arrays. |
| `absorb_structural_floaters`, `fill_unassigned_triangles`, `absorb_outlier_floaters`, `absorb_enclosed_patches`, `prune_teleported_scattered_crumbs`, `merge_proximate_components` | Labels/graph/geometry → legacy reassignment/pruning/merge → labels/counts. `fill_unassigned_triangles` conflicts with preserving unknowns. |
| `extract_instances`, `converge_class_instances_iteratively`, `absorb_class_outliers`, `decompose_class_into_connected_pieces` | Labels/adjacency/geometry and optional `supported_faces` → component/instance records; small components survive ordinary size gates when at least 75% of their faces have independent three-view support. `absorb_class_outliers` is an alias. |
| `fit_ransac_plane`, `analyze_class_geometry`, `classify_structural_classes` | Points or class geometry/embeddings → plane/features/role hypotheses → records. |
| `embed_class_id_badges_on_mesh`, `export_structural_manifest`, `detect_structural_classes.main` | Mesh/role/instance data → badges and structural JSON/CSV/Markdown → files. |
| `export_segmented_mesh_ply`, `export_single_instance_ply`, `deliver_stage05_outputs` | Sealed mesh/labels/instances/roles → PLYs, manifests and compatibility links → delivery manifest/files. |
| `export_per_class_meshes.export_per_class_meshes`, `main` | Segmented mesh/manifests → visual meshes and Blender helper → files. |
| `ClassInspector`, `run_inspector`, `_resolve_path`, `inspect_classes.main` | Mesh/manifests/viewer paths → interactive inspection → session. |
| `run_3d_object_extraction_step3.get_class_color`, `create_cylinder_mesh`, `build_3d_text_mesh`, `compute_robust_obb`, `build_bounding_box_geometry`, `export_binary_rgba_ply`, `main` | Legacy mesh/classes → OBB/badge geometry/extraction package → experimental files. |
| `clean_and_update_pipeline.absorb_disconnected_wall_crumbs`, `main`; `step1_remove_mesh_floaters.main` | Old projected mesh/compatibility CLI → historical cleaning / Stage 04 cleanup forwarding → legacy output/exit status. |
| `panoptic3d_baseline.class_color`, `legacy_face_sources`, `_triangle_keys`, `recover_baseline_labels`; `panoptic3d_features._records`, `_int`, `_float`, `load_colmap_tracks` | Legacy mesh/manifests or JSON/COLMAP → normalized baseline labels/feature support → arrays/records. |

### Tests and executable checks

| Callable | Purpose |
| --- | --- |
| `test_clean_masks.test_shape_filter` | Deterministic acceptance/rejection of the 2-D shape filter. |
| `test_dinov2_patches.test_only_neighboring_similar_patches_share_labels`, `test_running_cluster_mean_rejects_neighbor_that_only_matches_last_patch`, `test_small_region_merges_with_semantically_closer_touching_region`, `test_grid_rounds_wait_to_update_region_means`, `test_grow_and_merge_grid_seeds_mean_vs_boundary` | DINO patch adjacency, mean growth and merging rules. |
| `test_letterbox.test_letterbox_coordinates` | Image/model coordinate round trip. |
| `test_regularized_163.regularize_carved_mask`, `test_regularized_163` | One-off component/morphology regularization check. |
| `test_fast_refinement`, `test_streamlined_refine` | Executable timing/visual experiments, not stable unit tests or production inputs. |

## Quality and speed opportunities

These are candidates, not measured claims for every scene.

| Opportunity | Evidence in code | Safe direction |
| --- | --- | --- |
| Projection throughput | Raycasting is per camera and vote reductions allocate large arrays. | Reuse one Open3D scene/BVH, batch ray rows, retain fixed-dtype tensors and persist only necessary evidence. Benchmark before changing stride. |
| Vote quality | Pixel-distance and incidence weighting already exist. | Keep weights bounded per face/view, count independent view groups and compare raw/refined labels on held-out reviewed views. |
| Beam/ceiling correctness | Normal-only reassignment confuses vertical beam sides with walls. | Use persisted priors plus crease/contact support; preserve unknown when evidence is mixed. |
| Small-object integrity | A higher triangle minimum can merge a real small prop into its support. | Compare 1000 and 3000 on the same frozen evidence; choose the threshold using exported instances and representative views. |
| 2-D memory and I/O | Full-scene experiments retain masks/embeddings and emit many overlays. | Stream NPZ evidence, make diagnostics opt-in and test float16 embeddings against a fixed review set. |
| Structural analysis | `detect_structural_classes` has scene-scale and ID-oriented legacy heuristics. | Keep it downstream of evidence, cache one graph/derivative set, record RANSAC seeds and report roles as hypotheses. |

## Redundant and legacy paths

- `run_mesh_quality_vote_step1.py` and `run_mesh_dispute_resolution_step2.py` overlap projection/regularization. The supported worker persists source-ID-aligned evidence.
- `run_*memory*`, `run_id_propagation.py`, `run_*bench*`, one-frame DINO scripts and executable refinement tests duplicate palettes, overlays and memory-bank logic. Keep them as research artefacts until contracts are consolidated.
- Old cleanup functions that fill unknowns, use fixed IDs or delete small components cannot provide final semantic truth. They explain the former blobs, ceiling/wall bleed and fragmented-object failures.
- Full-mesh diagnostics are costly. The delivery package needs one labelled mesh; retain evidence arrays and solver reports instead of duplicate coloured geometry.

## Verification

The last accepted numerical validation used **421 frozen 2-D frames** and **2796890 source faces**. It recorded **123 passing tests**, covering projection evidence, hard minimums, wall-angle limits, tilted supports, small-fragment merging, late plane/RGB assignment, bounded seam associations and delivery indexing.

| Recorded result | Value |
| --- | ---: |
| Exported instances / classes | 66 / 41 |
| Smallest exported instance | 2014 triangles |
| Instances below the locked minimum | 0 |
| Forbidden near-horizontal wall faces | 0 |
| Stage 04 infill faces deferred | 140622 |
| Infill ray hits ignored as semantic evidence | 40353 |
| Deferred faces assigned at the final pass | 388140 of 398891 |
| Bounded same-surface associations | 282 |
| Residual unknown faces | 10751 (0.38% of the mesh) |
| Full cached-evidence run / late assignment | 68.07s / 12.287s |

These are recorded measurements, not a speedup or perfect-accuracy claim. Independent checks confirmed unchanged source hashes and geometry/face order, unchanged accepted observed owners during late assignment, all infill absent from observed labels, exact palette colors, valid instance files, matching face/manifest indices and bounded association links. Residual incompatible or unseeded faces remain explicit unknowns.

Local evidence: [horse and shelf](../../agents_files/artifacts/segmentation_core_fix/deferred_horse_and_shelf.png), [wall patches](../../agents_files/artifacts/segmentation_core_fix/deferred_wall_patches.png), [ceiling](../../agents_files/artifacts/segmentation_core_fix/deferred_ceiling.png), [independent acceptance](../../agents_files/artifacts/segmentation_core_fix/deferred_acceptance.json), and [residual diagnostics](../../agents_files/artifacts/segmentation_core_fix/unresolved_diagnostic.json). Images compare the earlier 1000-triangle result with the accepted 2000-triangle result from identical cameras. Earlier comparison packages and backups are historical evidence; production writes only `scan_N/segmentation/`.
