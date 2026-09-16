import pytest
import numpy as np
from pathlib import Path
from depth_priors import DepthPriorEstimator

def test_sliding_window_mock_k2():
    # Use mock model to avoid loading weights
    estimator = DepthPriorEstimator(model_name="mock", device="cpu")
    
    # 15 dummy images
    dummy_imgs = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(15)]
    
    depths, confs, uncs = estimator.estimate_depth_sliding_window(
        images=dummy_imgs,
        chunk_size=6,
        overlap=2,
    )
    
    assert len(depths) == 15
    assert uncs is not None
    assert len(uncs) == 15
    for d in depths:
        assert d.shape == (120, 160)
        assert np.isfinite(d).all()

def test_sliding_window_mock_k3_median():
    estimator = DepthPriorEstimator(model_name="mock", device="cpu")
    
    dummy_imgs = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(15)]
    
    depths, confs, uncs = estimator.estimate_depth_sliding_window(
        images=dummy_imgs,
        chunk_size=6,
        overlap=3,
    )
    
    assert len(depths) == 15
    assert uncs is not None
    assert len(uncs) == 15
    for d in depths:
        assert d.shape == (120, 160)
        assert np.isfinite(d).all()

if __name__ == "__main__":
    test_sliding_window_mock_k2()
    test_sliding_window_mock_k3_median()
    print("ALL MOCK SLIDING WINDOW TESTS PASSED!")
