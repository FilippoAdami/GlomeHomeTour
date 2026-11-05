# Temporary aligned-mesh blob detector

This experiment is isolated from the Phase 07 pipeline. It compares the globally aligned per-instance meshes in `objects/scan_*/` with their corresponding regularized room meshes, then writes a red-highlighted copy of `aligned_regularized_meshes.ply` with all aligned objects appended for visual context under `temporary_blob_detection/`.

An aligned object triangle counts as outside when its centroid is more than 15 cm beyond the nearest vertical regularized face, in the outward direction estimated from that room mesh's bounds center. Its nearest regularized faces are grouped by mesh adjacency; only connected projected sections whose regularized surface area is greater than 0.5 m² are colored bright red. The source objects and aligned regularized PLY are never modified; the temporary PLY contains both the colored structural meshes and the appended object meshes.

Run from the repository root:

```bash
backend/.venv/bin/python backend/07_alignment/temporary_blob_detection/detect_outside_blobs.py \
  --alignment-dir backend/current_scene/multi_room_alignment
```

This is an exploratory geometric proxy. It relies on aligned object meshes, nearest-face centroids, and a radial bounds-center estimate for wall outward direction; concave room layouts or large regularization offsets can produce missed regions or false positives.
