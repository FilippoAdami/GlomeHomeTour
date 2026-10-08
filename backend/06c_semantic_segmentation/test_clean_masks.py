"""Small deterministic shape checks for the proposal filter."""

import cv2
import numpy as np

from clean_masks import shape_reasons


def test_shape_filter() -> None:
    def canvas():
        return np.zeros((400, 400), np.uint8)

    solid = canvas()
    cv2.rectangle(solid, (100, 100), (300, 300), 1, -1)
    assert shape_reasons(solid)[0] == []

    thin = canvas()
    cv2.rectangle(thin, (100, 50), (108, 350), 1, -1)
    assert "thin" in shape_reasons(thin)[0]

    scattered = canvas()
    for x, y in ((20, 20), (250, 20), (100, 250)):
        cv2.rectangle(scattered, (x, y), (x + 60, y + 60), 1, -1)
    assert "scattered" in shape_reasons(scattered)[0]

    two_parts = canvas()
    for x in (20, 250):
        cv2.rectangle(two_parts, (x, 80), (x + 90, 280), 1, -1)
    assert shape_reasons(two_parts)[0] == []  # an occluder may split one object

    branch = canvas()
    cv2.line(branch, (200, 300), (200, 100), 1, 36)
    cv2.line(branch, (200, 150), (50, 50), 1, 36)
    cv2.line(branch, (200, 150), (350, 50), 1, 36)
    assert "branched" in shape_reasons(branch)[0]


if __name__ == "__main__":
    test_shape_filter()
