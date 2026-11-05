# Stage 08 — views and artifact generation

Stage 08 generates presentation and inspection products after mesh elaboration and Phase 07 scan alignment. It owns top-down floor views, elevation sections, cutaway views, floorplan symbols, and image comparisons. Phase 06 may create temporary 2-D masks and contours when they are needed to build structural geometry; those internal mesh-building inputs remain in Phase 06.

## Stage files

| Component | Responsibility | Runtime status |
| --- | --- | --- |
| [`extract_cross_section_views.py`](extract_cross_section_views.py) | Renders unobstructed top-down floor views and four orthographic elevation sections from the Stage 04 mesh, using Stage 07 alignment metadata when supplied. | Main section-view generator. |
| [`generate_scene_views.py`](generate_scene_views.py) | Chooses two wall-aligned vertical cuts and writes matching cutaway views for the sealed mesh and Gaussian Splat model. | Existing Stage 08 view generator; requires the configured GPU renderer. |
| [`generate_floorplan_symbols.py`](generate_floorplan_symbols.py) | Produces dimension-constrained SVG symbols for the fixed 28-category vocabulary. | Standalone SVG utility. |
| [`project_valid_instances_by_floor.py`](project_valid_instances_by_floor.py) | Projects classified object instances onto fitted floor outlines and writes annotated PNG/PDF plans. | Optional utility; currently requires legacy floor-level and refined-shell inputs. |
| [`extract_floor_ceiling_residual.py`](extract_floor_ceiling_residual.py) | Builds a floor/ceiling residual evaluation mesh and projection reports from Stage 04/05 inputs. | Optional geometry diagnostic, not the Phase 06 delivery. |
| [`reproject_enhanced_cross_sections.py`](reproject_enhanced_cross_sections.py) | Reprojects enhanced section-image colors to a separate mesh copy using cross-section projection maps. | Optional post-processing. |
| [`match_real_captured_frames.py`](match_real_captured_frames.py) | Matches Stage 05 synthetic instance views to registered camera frames and writes comparisons. | Optional inspection artifact. |

## Top-down and elevation views

After Stages 06 and 07, render views from the scan mesh and the Stage 07 alignment report:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/extract_cross_section_views.py \
  --mesh-path backend/current_scene/scan_0/mesh.ply \
  --instances-dir backend/current_scene/scan_0/mesh_elaboration_steps \
  --output-dir backend/current_scene/scan_0/artifacts/cross_sections
```

This writes `cross_sections_manifest.json`, one `topdown_floor_*.png` per floor manifest it finds, and four elevation images. Pass `mesh_elaboration_steps/` when its floor and ceiling manifests are available. The current `run_phase06.py` removes its temporary manifests after publishing `complete.ply` and `elements/`; without a retained level manifest, the view extractor falls back to one floor at the mesh's lower bound. It can also write per-pixel triangle/barycentric projection maps with `--projection-map-dir` for the optional enhanced-image reprojection pass.

## Other view artifacts

`generate_scene_views.py` takes a current-scene root or one scan directory. It requires both `mesh.ply` and `gaussian_splat/checkpoint_sealed.pt`, plus a working CUDA/ROCm-compatible renderer, and writes its cut-plane report and four corresponding views under `<scan>/artifacts/`:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_scene_views.py \
  --workspace backend/current_scene/scan_0
```

The real-camera comparison utility reads the Phase 05 mesh-frame manifest and writes only inspection artifacts:

```bash
backend/.venv/bin/python backend/08_artifacts_generation/match_real_captured_frames.py \
  --workspace backend/current_scene/scan_0
```

Generate a single fixed-class SVG symbol (class IDs match `run_non_structural_category_similarity.py`):

```bash
backend/.venv/bin/python backend/08_artifacts_generation/generate_floorplan_symbols.py 18 1.8 2.0 \
  --output /tmp/bed.svg
```

## Floorplan data dependencies

`project_valid_instances_by_floor.py` consumes the Stage 05 associations at `segmentation/non_structural_similarity/valid_class_associations.json`, but it also requires `structural_levels_manifest.json` and a refined structural mesh from the older standalone floor-level workflow. The current `run_phase06.py` publishes `mesh_elaboration/complete.ply` and `elements/` only, so those older plan inputs must be supplied explicitly; this utility is not run automatically by Stage 08.

`extract_floor_ceiling_residual.py` requires explicit source mesh, labels, structural metadata, fitted-wall inputs, and a new output PLY path. It reads the same Stage 05 association artifact when available. Its result is an auxiliary evaluation mesh, not a replacement for `mesh_elaboration/complete.ply`.
