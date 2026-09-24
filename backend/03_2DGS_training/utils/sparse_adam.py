"""Sparse Adam (Taming 3DGS) -- update only the splats visible in the current view.

A splat that no tile touched has a zero gradient, so dense Adam only decays its momentum.
Skipping those turns the step into a pass over the visible subset. Subclasses torch's Adam
so `state[p]["exp_avg"]` / `["exp_avg_sq"]` stay where GaussianModel's prune/cat/replace
surgery expects them.

Bias correction is folded into lr and eps host-side, so a step with everything visible
matches torch.optim.Adam. The only intended difference is the sparsity.
"""
import torch
from diff_surfel_rasterization import _C


class SparseGaussianAdam(torch.optim.Adam):
    @torch.no_grad()
    def step(self, visible=None, N=None):
        """`visible`: bool tensor, one entry per splat. None falls back to dense Adam."""
        if visible is None:
            return super().step()

        visible = visible.contiguous()
        for group in self.param_groups:
            assert len(group["params"]) == 1, "expected one tensor per param group"
            param = group["params"][0]
            if param.grad is None:
                continue

            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1

            b1, b2 = group["betas"]
            t = state["step"].item()
            bc1 = 1.0 - b1 ** t
            bc2_sqrt = (1.0 - b2 ** t) ** 0.5
            # The kernel indexes flat row-major, so a strided grad would read wrong data.
            # Grads are contiguous in practice; this is a no-op guard because the params
            # here needed the same treatment (see create_from_pcd). param and the moments
            # are written in place, so those must be genuinely contiguous, not repacked.
            _C.adamUpdate(
                param, param.grad.contiguous(), state["exp_avg"], state["exp_avg_sq"], visible,
                group["lr"] * bc2_sqrt / bc1, b1, b2, group["eps"] * bc2_sqrt,
                N, param.numel() // N,
            )
