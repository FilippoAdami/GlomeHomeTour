# Temporary sealed-mesh object contours

This isolated Phase 07 step finds each instance's exact Stage 05 faces in the sealed mesh, then marks an instance face red when it shares a mesh edge with a face owned by a different class or instance. It uses Stage 05 `class_id` and `instance_id` face properties and verifies their face geometry/order against the pristine sealed mesh before using adjacency. It then checks the globally aligned instances together and restores original RGB on red faces whose vertices, edge midpoints, or centroid come within 1 mm of a different instance surface. All triangles remain in the PLY.

The aligned instance meshes are transformed into the common Phase 07 scene frame. The step writes per-instance PLYs, one combined `aligned_objects_contact_contours.ply`, and `contact_contours.json`; it does not modify Phase 07 inputs or production behavior.

Run after Phase 07:

```bash
backend/.venv/bin/python backend/07_alignment/temporary_object_contact_contours/discover_contact_contours.py \
  --workspace backend/current_scene
```

Then test axis-aligned contact placement:

```bash
backend/.venv/bin/python backend/07_alignment/temporary_object_contact_contours/align_contours_to_regularized_planes.py \
  --workspace backend/current_scene
```

Before the existing plane snap, this step now checks only the red contour triangles (their vertices and centroids) against the nearest axis-aligned face of that scan's regularized mesh. It applies the smallest feasible translation that improves contour containment, one axis at a time, repeating until the contours are inside or no improving shift is possible. There is no distance limit on this pre-fit. Non-contour triangles do not constrain it. The existing plane snap then runs as before: rays from red contour vertices and centroids find the nearest valid plane translation, up to 15 cm per move and two moves per object. The room PLY is only a reference and is never exported or modified. Since the regularized meshes are open, the outside direction is estimated from the midpoint between area-weighted 2nd and 98th percentiles of triangle centers; the later plane-snap safety check also keeps red contour vertices inside the room mesh's axis-aligned bounds. The output is isolated under `temporary_object_contact_contours/axis_plane_alignment/`, with translated-object-only PLYs and a JSON report.

Then detect connected object-surface patches outside each scan's parent regularized mesh:

```bash
backend/.venv/bin/python backend/07_alignment/temporary_object_contact_contours/detect_large_outside_surface_patches.py \
  --workspace backend/current_scene
```

This reads the translated object-only PLYs and the matching per-scan regularized meshes, highlights edge-connected outside patches with at least 0.2 m² of object triangle surface area in bright red, and writes an object-only combined PLY under `temporary_object_contact_contours/outside_surface_detection/`. The regularized meshes are reference geometry and are never moved or exported. Since they are open surfaces, the detector estimates the outside direction from the midpoint between area-weighted 2nd and 98th percentiles of parent-mesh triangle centers, and ignores offsets within 1 mm.
