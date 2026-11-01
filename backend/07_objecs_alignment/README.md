# Stage 07 — object alignment

Stage 07 aligns Stage 05 `structural_role: object` instances to Stage 06, groups nearby instances, and snaps supported groups to floor or wall surfaces when the move does not increase surface intersections. It writes final per-instance PLYs to `<scan>/objects_alignment/aligned_instances/`. Top-down plans and elevation sections are generated in [Stage 08](../08_artifacts_generation/README.md).

Inputs from previous stages are read-only. Stage 07 uses accepted Stage 05 category associations as weak floor/wall support preferences. It prefers role-labeled PLYs under Stage 06 `elements/`; alternate shells without those files use geometric surface classification.

## Stage files

| Component | Responsibility | Runtime status |
| --- | --- | --- |
| [`step_objects_alignment.py`](step_objects_alignment.py) | Runs rigid baseline registration followed by contact grouping and support fitting. | Production entry point. |
| [`build_rigid_baseline.py`](build_rigid_baseline.py) | Estimates and confidence-gates a global rigid transform from structural anchors. | Production alignment pass. |
| [`structural_surfaces.py`](structural_surfaces.py) | Resolves Stage 06 floor, ceiling, and wall meshes from element PLYs or shell geometry. | Shared geometry helper. |
| [`group_touching_instances.py`](group_touching_instances.py) | Computes contact graphs and groups nearby aligned instances into multi-body blocks. | Contact helper. |
| [`fit_contact_groups_inside.py`](fit_contact_groups_inside.py) | Fits nearby floor/wall support candidates, preferring fewer mesh-edge intersections, stronger category-compatible support, then less movement. | Production placement pass. |
| [`translate_instances_inside.py`](translate_instances_inside.py) | Area-stratified triangle sampling and raycast envelope probe helper. | Containment helper. |
| `analyze_new_step.py`, `plot_new_step.py`, `view_new_step.py`, `render_side_by_side_all.py` | Correspondence analysis, diagnostic plots, interactive inspection, and comparison renders. | Optional diagnostics. |

## Inputs and outputs

| Path | Producer | Use |
| --- | --- | --- |
| `<scan>/mesh_elaboration/complete.ply` | Stage 06 | Completed structural surfaces used for registration and containment. |
| `<scan>/mesh_elaboration/elements/` | Stage 06 | Role-labeled floor, slab, ceiling, and wall surfaces used for registration and support detection. |
| `<scan>/segmentation/` | Stage 05 | Instance meshes, class metadata, and category-association artifact. |
| `<scan>/objects_alignment/aligned_instances/` | Stage 07 | Final object-only PLYs plus `manifest.json`; Stage 05 and Stage 06 files remain unchanged. |
| `<scan>/objects_alignment/structural_fit/` | Stage 07 | Support-fit report, fitted contact groups, and combined diagnostic scene. |

The current full Phase 06 runner publishes `complete.ply`; Stage 07 defaults to that file. Older standalone Phase 06 workspaces can pass an alternate shell with `--shell`.

## Processing

### 1. Rigid baseline registration (`build_rigid_baseline.py`)

- Delivers only Stage 05 instances labeled `structural_role: object`; wall, floor, and ceiling instances provide alignment anchors, while other structural roles are excluded.
- Finds closest-point correspondences between anchor faces and matching Stage 06 role surfaces, then estimates an area-weighted rigid correction.
- With `--allow-rotation`, estimates rotation using deterministic RANSAC over three-point samples.
- Applies the correction only when the full-anchor inlier fraction is at least 10%, translation is at most 5 cm, and rotation is at most 3°. Otherwise it records the proposal and applies identity; `transform` remains the applied transform for Stage 08 compatibility.
- Fit residual metrics describe the proposed transform, including when the confidence gate rejects it.
- Writes transformed instances to `rigid_baseline_instances/`, a combined object mesh, a combined scene mesh, and `rigid_baseline_report.json`.

### 2. Contact grouping (`group_touching_instances.py`)

- Builds KD-trees over vertices and sampled face centers of aligned instances.
- Groups instances connected by pairwise proximity at the configured contact distance.
- Writes the member lists and contact decisions to `contact_groups.json`.

### 3. Support fitting (`fit_contact_groups_inside.py`)

- Detects horizontal floor and parallel wall support within 5 cm, targets a 5 mm gap, and tests floor, wall, and combined translations.
- Ranks candidates by transverse mesh-edge intersections, then category-weighted support and movement. Supported placements are eligible only when at least 95% of each member instance's area-weighted surface samples remain inside the room envelope; the unchanged baseline remains available when no supported move meets that limit.
- Reports unresolved intersection edge hits and per-instance room-envelope containment. The edge test is a mesh-surface intersection check, not a signed penetration-depth measurement; unsupported objects are left in place.
- Publishes one PLY per explicitly non-structural instance under `aligned_instances/`, with source mapping, global registration transform, and local support translations in its manifest.

## Run

From the repository root:

```bash
backend/.venv/bin/python backend/07_objecs_alignment/step_objects_alignment.py \
  --workspace backend/current_scene/scan_1
```

For explicit inputs:

```bash
backend/.venv/bin/python backend/07_objecs_alignment/step_objects_alignment.py \
  --shell backend/current_scene/scan_1/mesh_elaboration/complete.ply \
  --segmentation-output backend/current_scene/scan_1/segmentation \
  --output-dir backend/current_scene/scan_1/objects_alignment
```

## Function reference

| Callable | Behavior |
| --- | --- |
| `step_objects_alignment.run_pipeline_for_scene` | Runs rigid registration and containment, writing per-pass timings. |
| `build_rigid_baseline.robust_fit` | Computes area-weighted translation and optional RANSAC rotation. |
| `group_touching_instances.build_groups` | Forms connected contact groups from nearby aligned instances. |
| `fit_contact_groups_inside.build` | Fits floor/wall support candidates, reports intersection checks, and publishes final object instances. |

Stage 08 uses `objects_alignment/rigid_baseline_report.json` together with the Stage 04 mesh to generate top-down and elevation views; see [`08_artifacts_generation/README.md`](../08_artifacts_generation/README.md).
