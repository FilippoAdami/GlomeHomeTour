"""Sparse Adam (Taming 3DGS) -- HIP fused optimiser step.

Two things must hold: with every splat visible it is plain Adam, and with a mask it moves
the visible splats exactly as Adam would while leaving the rest untouched.

Run: backend/.venv/bin/python -m pytest backend/tests/test_sparse_adam.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

TRAIN_DIR = Path(__file__).resolve().parents[1] / "03_2DGS_training"
sys.path.insert(0, str(TRAIN_DIR))

pytest.importorskip("diff_surfel_rasterization")
from utils.sparse_adam import SparseGaussianAdam  # noqa: E402

N, M, STEPS = 512, 3, 5
LR, EPS = 1e-3, 1e-15


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")


def _run(opt_cls, visible, seed=0):
    """Same params and same gradient sequence under both optimisers."""
    torch.manual_seed(seed)
    p = torch.randn(N, M, device="cuda")
    grads = [torch.randn(N, M, device="cuda") for _ in range(STEPS)]
    p = torch.nn.Parameter(p)
    opt = opt_cls([{"params": [p], "lr": LR}], lr=0.0, eps=EPS)
    for g in grads:
        p.grad = g.clone()
        opt.step(visible, N) if visible is not None else opt.step()
    return p.detach()


def test_matches_adam_when_everything_visible():
    _cuda_or_skip()
    vis = torch.ones(N, dtype=torch.bool, device="cuda")
    got = _run(SparseGaussianAdam, vis)
    want = _run(torch.optim.Adam, None)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max().item()


def test_invisible_splats_are_untouched():
    _cuda_or_skip()
    torch.manual_seed(0)
    vis = torch.rand(N, device="cuda") > 0.5
    start = _run(SparseGaussianAdam, torch.zeros(N, dtype=torch.bool, device="cuda"))
    got = _run(SparseGaussianAdam, vis)
    dense = _run(torch.optim.Adam, None)
    assert torch.equal(got[~vis], start[~vis]), "an invisible splat moved"
    assert torch.allclose(got[vis], dense[vis], atol=1e-6), "a visible splat did not match Adam"
    assert vis.any() and (~vis).any(), "mask is degenerate"


def test_state_layout_survives_pruning():
    """GaussianModel's prune/cat surgery indexes exp_avg/exp_avg_sq directly."""
    _cuda_or_skip()
    p = torch.nn.Parameter(torch.randn(N, M, device="cuda"))
    opt = SparseGaussianAdam([{"params": [p], "lr": LR}], lr=0.0, eps=EPS)
    p.grad = torch.randn(N, M, device="cuda")
    opt.step(torch.ones(N, dtype=torch.bool, device="cuda"), N)
    state = opt.state[p]
    assert set(("step", "exp_avg", "exp_avg_sq")) <= set(state)
    assert state["exp_avg"].shape == (N, M) and state["exp_avg_sq"].shape == (N, M)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
