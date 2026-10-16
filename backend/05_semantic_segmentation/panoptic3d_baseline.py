"""Read-only recovery of labels from the legacy step-3 mesh artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def class_color(class_id: int) -> np.ndarray:
    """The RGB palette used by the legacy step-3 writer."""
    k = int(class_id) - 1
    return np.array(((197 * k + 31) % 255 / 255.0,
                     (109 * k + 71) % 255 / 255.0,
                     (37 * k + 93) % 255 / 255.0))


def legacy_face_sources(vertex_colors, triangles, persistent_classes) -> np.ndarray:
    """Reproduce step 3's per-channel palette decoder; unmatched faces stay zero."""
    colors = np.round(np.asarray(vertex_colors)[np.asarray(triangles)[:, 0]], 3)
    sources = np.zeros(len(colors), np.int32)
    for class_id in sorted(set(map(int, persistent_classes))):
        palette = np.round(class_color(class_id), 3)
        sources[(np.abs(colors - palette) < 0.05).all(axis=1)] = class_id
    return sources


def _triangle_keys(vertices, triangles):
    """Canonical exact triangle bytes, independent of vertex and winding order."""
    triangle_vertices = np.asarray(vertices, dtype='<f8')[np.asarray(triangles, np.int64)]
    order = np.lexsort((triangle_vertices[:, :, 2], triangle_vertices[:, :, 1], triangle_vertices[:, :, 0]))
    canonical = np.take_along_axis(triangle_vertices, order[:, :, None], axis=1)
    return np.ascontiguousarray(canonical.reshape(len(canonical), 9)).view(np.dtype((np.void, 72))).ravel()


def recover_baseline_labels(mesh_path, manifest_path, instances_dir, persistent_classes):
    """Return legacy source labels, saved instance IDs, and recovery provenance.

    Exported object meshes are authoritative when their triangles exactly survive in
    the final cleanup mesh.  The legacy palette decoder fills all remaining faces.
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    sources = legacy_face_sources(np.asarray(mesh.vertex_colors), triangles, persistent_classes)
    instance_ids = np.zeros(len(triangles), np.int32)
    manifest = json.loads(Path(manifest_path).read_text()) if Path(manifest_path).exists() else []
    mesh_keys = _triangle_keys(vertices, triangles)
    order = np.argsort(mesh_keys)
    sorted_keys = mesh_keys[order]
    matched = np.zeros(len(triangles), bool)
    missing_exports = []
    unmatched_export_triangles = 0

    for entry in manifest:
        instance_id, class_id = int(entry['instance_id']), int(entry['class_id'])
        exact = Path(instances_dir) / f"inst_{instance_id:03d}_{entry.get('instance_name', '')}.ply"
        paths = [exact] if exact.exists() else list(Path(instances_dir).glob(f'inst_{instance_id:03d}_*.ply'))
        if len(paths) != 1:
            missing_exports.append(instance_id)
            continue
        exported = o3d.io.read_triangle_mesh(str(paths[0]))
        keys = _triangle_keys(np.asarray(exported.vertices), np.asarray(exported.triangles))
        positions = np.searchsorted(sorted_keys, keys)
        valid = positions < len(sorted_keys)
        valid[valid] &= sorted_keys[positions[valid]] == keys[valid]
        # Repeated geometric triangles cannot establish a unique saved identity.
        unique = valid.copy()
        at_left = valid & (positions > 0)
        unique[at_left] &= sorted_keys[positions[at_left] - 1] != keys[at_left]
        at_right = valid & (positions + 1 < len(sorted_keys))
        unique[at_right] &= sorted_keys[positions[at_right] + 1] != keys[at_right]
        _, inverse, counts = np.unique(positions[valid], return_inverse=True, return_counts=True)
        unique[valid] &= counts[inverse] == 1
        faces = order[positions[unique]]
        if np.any(matched[faces] & (instance_ids[faces] != instance_id)):
            raise ValueError('legacy instance exports assign conflicting IDs to one final mesh face')
        sources[faces] = class_id
        instance_ids[faces] = instance_id
        matched[faces] = True
        unmatched_export_triangles += int((~unique).sum())

    provenance = {
        'total_faces': int(len(triangles)),
        'export_matched_faces': int(matched.sum()),
        'palette_fallback_faces': int((~matched).sum()),
        'palette_unknown_faces': int(np.count_nonzero((~matched) & (sources == 0))),
        'unmatched_export_triangles': int(unmatched_export_triangles),
        'missing_export_instance_ids': missing_exports,
        'decoder': 'legacy per-channel rounded RGB tolerance < 0.05; no nearest-color assignment',
    }
    return {'sources': sources, 'instance_ids': instance_ids, 'provenance': provenance}
