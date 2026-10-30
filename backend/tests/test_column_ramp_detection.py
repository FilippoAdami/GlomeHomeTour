import pytest
import numpy as np
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "05_semantic_segmentation"))

from detect_structural_classes import (
    estimate_gravity_and_scene_bands,
    analyze_class_geometry,
    classify_structural_classes,
)


def test_detect_column_from_geometry():
    """
    Test that a narrow vertical pillar extending from floor (y=0) to ceiling (y=3.0)
    with multi-directional vertical normals is classified as 'columns'.
    """
    vertices = []
    triangles = []
    labels = []

    def add_quad(p0, p1, p2, p3, label):
        idx = len(vertices)
        vertices.extend([p0, p1, p2, p3])
        triangles.append([idx, idx + 1, idx + 2])
        triangles.append([idx, idx + 2, idx + 3])
        labels.extend([label, label])

    # Floor (Class 1)
    for x in np.arange(-3.0, 3.0, 1.0):
        for z in np.arange(-3.0, 3.0, 1.0):
            add_quad([x, 0, z], [x, 0, z + 1.0], [x + 1.0, 0, z + 1.0], [x + 1.0, 0, z], 1)

    # Ceiling (Class 2)
    for x in np.arange(-3.0, 3.0, 1.0):
        for z in np.arange(-3.0, 3.0, 1.0):
            add_quad([x, 3.0, z], [x + 1.0, 3.0, z], [x + 1.0, 3.0, z + 1.0], [x, 3.0, z + 1.0], 2)

    # Big Wall (Class 3)
    for y in np.arange(0, 3.0, 0.5):
        for z in np.arange(-3.0, 3.0, 0.5):
            add_quad([-3.0, y, z], [-3.0, y + 0.5, z], [-3.0, y + 0.5, z + 0.5], [-3.0, y, z + 0.5], 3)

    # Narrow Pillar / Column (Class 4): 0.4m x 0.4m footprint, y from 0 to 3.0m
    cx, cz, cw = 0.5, 0.5, 0.4
    # 4 vertical sides
    for y in np.arange(0, 3.0, 0.5):
        # side 1: normal +x
        add_quad([cx + cw, y, cz], [cx + cw, y + 0.5, cz], [cx + cw, y + 0.5, cz + cw], [cx + cw, y, cz + cw], 4)
        # side 2: normal -x
        add_quad([cx, y, cz + cw], [cx, y + 0.5, cz + cw], [cx, y + 0.5, cz], [cx, y, cz], 4)
        # side 3: normal +z
        add_quad([cx, y, cz + cw], [cx + cw, y, cz + cw], [cx + cw, y + 0.5, cz + cw], [cx, y + 0.5, cz + cw], 4)
        # side 4: normal -z
        add_quad([cx + cw, y, cz], [cx, y, cz], [cx, y + 0.5, cz], [cx + cw, y + 0.5, cz], 4)

    verts = np.array(vertices, dtype=np.float32)
    tris = np.array(triangles, dtype=np.int32)
    lbls = np.array(labels, dtype=np.int32)

    gravity_up, scene_bands = estimate_gravity_and_scene_bands(verts, tris)
    features = analyze_class_geometry(verts, tris, lbls, [1, 2, 3, 4], gravity_up, scene_bands)
    classifications = classify_structural_classes(features, scene_bands)

    assert classifications[4]["structural_role"] == "columns", (
        f"Expected class 4 to be 'columns', got {classifications[4]['structural_role']}: {classifications[4]['details']}"
    )


def test_detect_ramp_from_geometry():
    """
    Test that a straight linear tilted floor plane connecting two floor elevations
    with uniform tilt (e.g. 10 deg) is classified as 'ramps'.
    """
    vertices = []
    triangles = []
    labels = []

    def add_quad(p0, p1, p2, p3, label):
        idx = len(vertices)
        vertices.extend([p0, p1, p2, p3])
        triangles.append([idx, idx + 1, idx + 2])
        triangles.append([idx, idx + 2, idx + 3])
        labels.extend([label, label])

    # Floor Level 0 (Class 1): y=0, z in [-3, 0]
    for x in np.arange(-2.0, 2.0, 0.5):
        for z in np.arange(-3.0, 0.0, 0.5):
            add_quad([x, 0, z], [x, 0, z + 0.5], [x + 0.5, 0, z + 0.5], [x + 0.5, 0, z], 1)

    # Ceiling (Class 2): y=3.0
    for x in np.arange(-2.0, 2.0, 0.5):
        for z in np.arange(-3.0, 3.0, 0.5):
            add_quad([x, 3.0, z], [x + 0.5, 3.0, z], [x + 0.5, 3.0, z + 0.5], [x, 3.0, z + 0.5], 2)

    # Big Wall (Class 3): x = -2.0
    for y in np.arange(0, 3.0, 0.5):
        for z in np.arange(-3.0, 3.0, 0.5):
            add_quad([-2.0, y, z], [-2.0, y + 0.5, z], [-2.0, y + 0.5, z + 0.5], [-2.0, y, z + 0.5], 3)

    # Ramp (Class 4): z in [0, 2.0], y slopes uniformly from 0 to 0.35m (~10 degrees tilt)
    slope = 0.35 / 2.0
    for x in np.arange(-1.0, 1.0, 0.25):
        for z in np.arange(0.0, 2.0, 0.25):
            y0 = slope * z
            y1 = slope * (z + 0.25)
            add_quad(
                [x, y0, z], [x, y1, z + 0.25],
                [x + 0.25, y1, z + 0.25], [x + 0.25, y0, z],
                4
            )

    verts = np.array(vertices, dtype=np.float32)
    tris = np.array(triangles, dtype=np.int32)
    lbls = np.array(labels, dtype=np.int32)

    gravity_up, scene_bands = estimate_gravity_and_scene_bands(verts, tris)
    features = analyze_class_geometry(verts, tris, lbls, [1, 2, 3, 4], gravity_up, scene_bands)
    classifications = classify_structural_classes(features, scene_bands)

    assert classifications[4]["structural_role"] == "ramps", (
        f"Expected class 4 to be 'ramps', got {classifications[4]['structural_role']}: {classifications[4]['details']}"
    )
