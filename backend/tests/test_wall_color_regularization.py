import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_mesh_elaboration"))
from recognize_and_regularize_patterns import (
    regularize_wall_lighting_and_outliers,
    detect_wall_color_splits,
    unify_adjacent_wall_colors,
)


def test_regularize_wall_lighting_and_outliers():
    """Verifies that dark cables, specular glare, switch plates, and foreign objects
    are rejected, and that 2D illumination variance is reduced.
    """
    np.random.seed(42)
    n_pts = 1000

    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 1.0, 0.0])

    u = np.random.uniform(-2.0, 2.0, n_pts)
    v = np.random.uniform(-1.0, 2.0, n_pts)
    pts = np.column_stack([u, v, np.zeros(n_pts)])

    # True base albedo: warm beige plaster [0.36, 0.28, 0.22]
    albedo = np.array([0.36, 0.28, 0.22])
    # Smooth illumination field: brighter at +u, darker at -v
    illum = 0.8 + 0.15 * u - 0.10 * v
    cols = albedo[None, :] * illum[:, None] + np.random.normal(0, 0.01, (n_pts, 3))
    cols = np.clip(cols, 0.0, 1.0)

    # Inject foreign objects:
    # 1. Dark cable points
    pts[10:20] = np.array([0.5, 0.5, 0.0])
    cols[10:20] = np.array([0.08, 0.08, 0.09])

    # 2. Specular glare / white switch plate
    pts[30:40] = np.array([-0.5, 0.5, 0.0])
    cols[30:40] = np.array([0.95, 0.95, 0.92])

    # 3. Cold blue/green curtain remnant
    pts[50:60] = np.array([1.0, -0.5, 0.0])
    cols[50:60] = np.array([0.20, 0.35, 0.55])

    clean_pts, reg_cols, inl_u, inl_v, diag = regularize_wall_lighting_and_outliers(
        pts, cols, u_axis, v_axis
    )

    # Inliers should filter out the injected outliers
    assert diag['inlier_ratio'] > 0.60
    assert diag['reg_variance'] < diag['raw_variance']

    # None of the cable or glare points should be among inliers
    for c in reg_cols:
        assert not (c[0] < 0.15 and c[1] < 0.15)  # cable rejected
        assert not np.all(c > 0.80)                # glare rejected
        assert c[0] > c[2]                         # warm plaster preserved


def test_detect_wall_color_splits_negative():
    """Verifies that a uniform plaster wall is correctly classified as single-color (no split)."""
    np.random.seed(42)
    N = 1000
    v = np.random.uniform(-1.0, 2.0, N)
    u = np.random.uniform(-2.0, 2.0, N)
    cols = np.random.normal(loc=[0.36, 0.28, 0.22], scale=0.015, size=(N, 3))
    cols = np.clip(cols, 0.0, 1.0)

    res = detect_wall_color_splits(v, u, cols)
    assert not res['is_split']
    assert res['split_type'] == 'none'
    assert res['delta_E'] < 0.08
    np.testing.assert_allclose(res['color_1'], [0.36, 0.28, 0.22], atol=0.03)


def test_detect_wall_color_splits_positive_horizontal():
    """Verifies positive detection of an artificial two-tone wall with horizontal wainscoting."""
    np.random.seed(42)
    N = 1200
    v = np.random.uniform(-1.0, 2.0, N)
    u = np.random.uniform(-2.0, 2.0, N)
    cols = np.random.normal(loc=[0.36, 0.28, 0.22], scale=0.015, size=(N, 3))

    # Wainscoting: dark slate blue below 0.5m
    wainscot_split = 0.50
    wainscot_mask = v < wainscot_split
    cols[wainscot_mask] = np.random.normal(loc=[0.14, 0.18, 0.25], scale=0.015, size=(np.count_nonzero(wainscot_mask), 3))
    cols = np.clip(cols, 0.0, 1.0)

    res = detect_wall_color_splits(v, u, cols, split_threshold=0.12)
    assert res['is_split']
    assert res['split_type'] == 'horizontal'
    assert abs(res['split_coord'] - wainscot_split) < 0.10
    assert res['delta_E'] > 0.15
    np.testing.assert_allclose(res['color_1'], [0.14, 0.18, 0.25], atol=0.04)
    np.testing.assert_allclose(res['color_2'], [0.36, 0.28, 0.22], atol=0.04)


def test_detect_wall_color_splits_positive_vertical():
    """Verifies positive detection of an artificial two-tone wall with a vertical accent section."""
    np.random.seed(42)
    N = 1200
    v = np.random.uniform(-1.0, 2.0, N)
    u = np.random.uniform(-2.0, 2.0, N)
    cols = np.random.normal(loc=[0.36, 0.28, 0.22], scale=0.015, size=(N, 3))

    # Accent wall: terracotta on left half (u < 0.0)
    vert_split = 0.0
    accent_mask = u < vert_split
    cols[accent_mask] = np.random.normal(loc=[0.55, 0.25, 0.18], scale=0.015, size=(np.count_nonzero(accent_mask), 3))
    cols = np.clip(cols, 0.0, 1.0)

    res = detect_wall_color_splits(v, u, cols, split_threshold=0.12)
    assert res['is_split']
    assert res['split_type'] == 'vertical'
    assert abs(res['split_coord'] - vert_split) < 0.15
    assert res['delta_E'] > 0.15


def test_unify_adjacent_wall_colors():
    """Verifies that adjacent walls touching at corners with small border color differences
    are merged into a single cluster and assigned a unified color.
    """
    wall_data = []
    base_color = np.array([0.35, 0.27, 0.21])

    planes_info = [
        ('w0', [1.0, 0.0, 0.0, 2.0], (-2.0, None)),
        ('w1', [-1.0, 0.0, 0.0, 2.0], (2.0, None)),
        ('w2', [0.0, 0.0, 1.0, 2.0], (None, -2.0)),
        ('w3', [0.0, 0.0, -1.0, 2.0], (None, 2.0)),
    ]

    for name, eq, fixed in planes_info:
        N = 200
        if fixed[0] is not None:
            x = np.full(N, fixed[0])
            z = np.random.uniform(-2.0, 2.0, N)
        else:
            x = np.random.uniform(-2.0, 2.0, N)
            z = np.full(N, fixed[1])
        y = np.random.uniform(0.0, 3.0, N)
        pts = np.column_stack([x, y, z])
        wall_c = base_color + np.random.normal(0, 0.01, (N, 3))
        wall_data.append({
            'name': name,
            'plane_equation': np.array(eq),
            'clean_pts': pts,
            'reg_cols': wall_c,
            'split_info': {'is_split': False},
        })

    unified = unify_adjacent_wall_colors(wall_data, border_dist_m=0.5, match_threshold=0.10)
    u_colors = [w['unified_color'] for w in unified]
    assert all(c is not None for c in u_colors)
    for c in u_colors[1:]:
        np.testing.assert_allclose(u_colors[0], c, atol=1e-6)
