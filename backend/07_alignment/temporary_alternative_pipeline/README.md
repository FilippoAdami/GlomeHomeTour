# Phase 07 object contour fitting

These contour and placement stages run from the production Phase 07 entrypoint after
scan transforms have been applied to the source object instances. Run the complete
pipeline from the repository root:

```bash
backend/.venv/bin/python backend/07_alignment/step_alignment.py \
  --workspace backend/current_scene
```

During the one-command production run, reports and red diagnostic meshes are written
under `backend/current_scene/multi_room_alignment/`, consumed by the next fitting
stage, then removed after final objects are published. A failed run leaves them for
diagnosis. The ordered object path is:

1. `outside_contact_contours/`: mark sealed-mesh contact faces red, then remove
   degenerate/elongated faces, oversized faces, edge-isolated faces, and per-instance
   red groups of 1–4 triangles that are mutually within 10 cm and have no other red
   triangle within 10 cm. Discover connected groups of aligned objects whose
   surfaces touch within 1 mm.
2. `contour_inside_fit/`: find one shared axis-only translation per contact group
   that fits all its filtered red contours inside the shell. There is no distance cap;
   non-contour geometry does not constrain this correction.
3. `nearest_plane_fit/`: search up to two shared axis-only translations per contact
   group, each on a separate axis and capped at 15 cm, to reduce its mean red-contour
   distance to the nearest shell plane.
   Before that search, move the whole group upward just enough to clear its lowest
   floor. Each member independently allows at most 5% of the unique vertex positions
   belonging to its red contour triangles outside. No unique vertex of any member may
   fall below its aligned lowest floor element; floor recovery leaves a 0.1 mm
   clearance. The floor is read from Phase 06's `floor_level_000_area_*.ply` elements.
4. For each scan and detected X/Z wall plane, extend only single-scan contact groups
   already pulled toward that plane that moved less than the group's median amount.
   Cross-scan contact groups keep their shared nearest-plane translation. The median
   pull has its own additional 15 cm per-plane cap and must preserve the 5% contour
   and zero-percent floor limits.

The contour colors are inspection-only. Phase 07 rebuilds canonical aligned object PLYs
from the original-color synchronized meshes using the combined translations. The
regularized meshes and Stage 05 source objects are never edited. Open3D 0.19 reports
CUDA unavailable in this environment, so closest-surface queries run in bounded CPU
batches. `run_alternative_pipeline.py` remains a standalone trial harness for isolated
experiments.
