"""Bounded FastGS-style densification for the Apache gsplat trainer.

This strategy accepts per-Gaussian ``importance_score`` and ``pruning_score``
in rasterizer ``info``.  Computing those ten-view residual metrics remains the
trainer's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import duplicate, remove, reset_opa, split


@dataclass
class FastGSStrategy(Strategy):
    """Small FastGS control layer using Apache gsplat topology operations."""

    clone_grad: float = 0.0002
    split_absgrad: float = 0.0012
    min_support: float = 5.0
    prune_opa: float = 0.005
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    reset_every: int = 3_000
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        return {"scene_scale": scene_scale, "grad2d": None, "absgrad2d": None,
                "count": None, "radii": None, "importance_score": None,
                "pruning_score": None, "origin": None}

    def check_sanity(self, params, optimizers) -> None:
        super().check_sanity(params, optimizers)
        for key in ("means", "scales", "quats", "opacities"):
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(self, params, optimizers, state, step: int, info: Dict[str, Any]) -> None:
        if step >= self.refine_stop:
            return
        assert "means2d" in info, "means2d is required for FastGSStrategy."
        info["means2d"].retain_grad()

    def step_post_backward(
        self, params, optimizers, state, step: int, info: Dict[str, Any], packed: bool = False
    ) -> None:
        if packed:
            raise ValueError("FastGSStrategy currently supports dense one-camera rasterization only.")
        final = step in (18_000, 21_000, 24_000, 27_000)
        refine = self.refine_start < step < self.refine_stop and step % self.refine_every == 0
        if step >= self.refine_stop:
            if final:
                self._update_metrics(params, state, info, required_keys=("pruning_score",))
                self._final_prune(params, optimizers, state)
            return
        self._update_state(params, state, info, require_metrics=refine)
        if refine:
            self._refine(params, optimizers, state, step)
        if step > 0 and step % self.reset_every == 0:
            reset_opa(params, optimizers, state, 0.01)

    def _update_state(self, params, state: Dict[str, Any], info: Dict[str, Any], require_metrics: bool) -> None:
        for key in ("means2d", "radii", "width", "height"):
            assert key in info, f"{key} is required in info."
        means2d = info["means2d"]
        assert means2d.grad is not None and hasattr(means2d, "absgrad"), (
            "means2d.grad and means2d.absgrad are required; rasterize with absgrad=True."
        )
        if means2d.ndim != 3 or means2d.shape[0] != 1:
            raise ValueError("FastGSStrategy currently supports one dense camera per step.")
        n, device = len(params["means"]), means2d.device
        if state["grad2d"] is None:
            for key in ("grad2d", "absgrad2d", "count", "radii", "importance_score", "pruning_score"):
                state[key] = torch.zeros(n, device=device)
            state["origin"] = torch.arange(n, device=device)
        grads = means2d.grad[0].clone()
        absgrads = means2d.absgrad[0].clone()
        for value in (grads, absgrads):
            value[:, 0] *= info["width"] / 2.0
            value[:, 1] *= info["height"] / 2.0
        visible = (info["radii"][0] > 0).all(dim=-1)
        state["grad2d"] += grads.norm(dim=-1) * visible
        state["absgrad2d"] += absgrads.norm(dim=-1) * visible
        state["count"] += visible
        state["radii"] = torch.maximum(state["radii"], info["radii"][0].max(dim=-1).values)
        self._update_metrics(
            params, state, info,
            required_keys=("importance_score", "pruning_score") if require_metrics else (),
        )

    @staticmethod
    def _update_metrics(params, state: Dict[str, Any], info: Dict[str, Any], required_keys=()) -> None:
        n, device = len(params["means"]), params["means"].device
        for key in ("importance_score", "pruning_score"):
            if key not in info:
                if key in required_keys:
                    raise ValueError(f"fresh {key} is required at FastGS refinement/final-prune events.")
                continue
            value = torch.as_tensor(info[key], device=device, dtype=torch.float32).reshape(-1)
            if len(value) != n or not torch.isfinite(value).all():
                raise ValueError(f"{key} must be finite with one value per Gaussian.")
            if key == "importance_score" and (value < 0).any():
                raise ValueError("importance_score must be nonnegative.")
            if key == "pruning_score" and ((value < 0).any() or (value > 1).any()):
                raise ValueError("pruning_score must be normalized to [0, 1].")
            state[key] = value

    def _growth_masks(self, params, state: Dict[str, Any]):
        count = state["count"].clamp_min(1)
        support = state["importance_score"] > self.min_support
        small = torch.exp(params["scales"]).max(dim=-1).values <= 0.001 * state["scene_scale"]
        clone = support & small & (state["grad2d"] / count >= self.clone_grad)
        split_mask = support & ~small & (state["absgrad2d"] / count >= self.split_absgrad)
        return clone, split_mask

    @staticmethod
    def _zero_new_scores(state: Dict[str, Any], start: int) -> None:
        for key in ("importance_score", "pruning_score"):
            state[key][start:] = 0
        state["origin"][start:] = -1

    @torch.no_grad()
    def _refine(self, params, optimizers, state: Dict[str, Any], step: int = 600) -> None:
        state["origin"] = torch.arange(len(params["means"]), device=params["means"].device)
        clone, split_mask = self._growth_masks(params, state)
        n_before, n_clone, n_split = len(clone), int(clone.sum()), int(split_mask.sum())
        if n_clone:
            duplicate(params, optimizers, state, clone)
            self._zero_new_scores(state, n_before)
        if n_split:
            split_mask = torch.cat((split_mask, torch.zeros(n_clone, dtype=torch.bool, device=clone.device)))
            split(params, optimizers, state, split_mask)
            self._zero_new_scores(state, len(params["means"]) - 2 * n_split)
        # Original FastGS resets screen-radius history before event pruning.
        # Retaining it makes the >20px screen gate prune trained coverage.
        state["radii"].zero_()
        self._stochastic_prune(params, optimizers, state, step)
        reset_opa(params, optimizers, state, 0.8)
        for key in ("grad2d", "absgrad2d", "count", "radii"):
            state[key].zero_()
        if self.verbose:
            print(f"FastGS: cloned={n_clone}, split={n_split}, splats={len(params['means'])}", flush=True)

    @staticmethod
    def _stochastic_mask(marked: torch.Tensor, pruning_score: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        """Sample original IDs, then intersect them with post-growth marked splats."""
        original = origin >= 0
        budget = min(int(marked.sum().item() * 0.5), int(original.sum().item()))
        selected = torch.zeros_like(marked)
        if budget:
            weights = torch.zeros_like(pruning_score)
            weights[original] = 1 / (1e-6 + 1 - pruning_score[original])
            ids = torch.multinomial(weights, budget, replacement=False)
            selected[ids] = True
        return selected & marked & original

    @torch.no_grad()
    def _stochastic_prune(self, params, optimizers, state: Dict[str, Any], step: int) -> None:
        marked = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            marked |= ((state["radii"] > 20)
                       | (torch.exp(params["scales"]).max(dim=-1).values > 0.1 * state["scene_scale"]))
        mask = self._stochastic_mask(marked, state["pruning_score"], state["origin"])
        if mask.any():
            remove(params, optimizers, state, mask)

    @torch.no_grad()
    def _final_prune(self, params, optimizers, state: Dict[str, Any]) -> None:
        mask = ((torch.sigmoid(params["opacities"].flatten()) < 0.1)
                | (state["pruning_score"] > 0.9))
        if mask.any():
            remove(params, optimizers, state, mask)
