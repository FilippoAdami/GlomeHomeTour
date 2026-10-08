import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE = Path(__file__).parents[1] / "03_FastGS_DNSplatter" / "fastgs_strategy.py"
spec = importlib.util.spec_from_file_location("fastgs_strategy", MODULE)
fastgs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fastgs
spec.loader.exec_module(fastgs)
FastGSStrategy = fastgs.FastGSStrategy


def _model(n=4):
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.zeros(n, 3)),
        "scales": torch.nn.Parameter(torch.full((n, 3), -7.0)),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0, 0, 0.]] * n)),
        "opacities": torch.nn.Parameter(torch.full((n,), 2.0)),
    })
    optimizers = {key: torch.optim.Adam([value]) for key, value in params.items()}
    for optimizer in optimizers.values():
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.ones_like(parameter)
        optimizer.step(); optimizer.zero_grad()
    return params, optimizers


def _info(n, importance=None, pruning=None):
    means2d = torch.zeros(1, n, 2, requires_grad=True)
    means2d.grad = torch.tensor([[[.001, 0.]] * n])
    means2d.absgrad = torch.tensor([[[.002, 0.]] * n])
    return {"means2d": means2d, "radii": torch.ones(1, n, 2), "width": 100, "height": 100,
            "importance_score": torch.full((n,), 6.) if importance is None else importance,
            "pruning_score": torch.zeros(n) if pruning is None else pruning}


def test_scores_are_finite_and_support_gates_growth():
    params, optimizers = _model()
    strategy, state = FastGSStrategy(), FastGSStrategy().initialize_state(1.0)
    info = _info(4, torch.tensor([6., 5., 0., 7.]))
    strategy._update_state(params, state, info, require_metrics=True)
    clone, split = strategy._growth_masks(params, state)
    assert clone.tolist() == [True, False, False, True]
    assert not split.any()
    assert torch.allclose(state["grad2d"], torch.full((4,), .05))
    info["pruning_score"] = torch.ones(4)
    strategy._update_metrics(params, state, info, required_keys=("importance_score", "pruning_score"))
    assert state["pruning_score"].tolist() == [1.] * 4
    params["scales"].data[1] = -2.0
    state["grad2d"].zero_(); state["absgrad2d"].zero_(); state["count"].fill_(1)
    state["importance_score"][1] = 6
    state["grad2d"][0] = strategy.clone_grad
    state["absgrad2d"][1] = strategy.split_absgrad
    clone, split = strategy._growth_masks(params, state)
    assert clone[0] and split[1]


def test_refine_replaces_optimizer_state_and_resets_stats():
    params, optimizers = _model(3)
    strategy, state = FastGSStrategy(), FastGSStrategy().initialize_state(1.0)
    strategy._update_state(params, state, _info(3), require_metrics=True)
    params["scales"].data[1:] = -2.0  # large splats use absolute-gradient splitting
    strategy._refine(params, optimizers, state)
    assert len(params["means"]) == 6  # one clone plus two children for each split
    assert all(len(opt.param_groups[0]["params"][0]) == 6 for opt in optimizers.values())
    for optimizer in optimizers.values():
        parameter = optimizer.param_groups[0]["params"][0]
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
        assert optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape
    assert not state["grad2d"].any() and torch.all(torch.sigmoid(params["opacities"]) <= .8)
    state["count"].fill_(1)
    state["importance_score"].zero_()
    state["pruning_score"].zero_()
    strategy._refine(params, optimizers, state)
    assert torch.equal(state["origin"], torch.arange(6))


def test_stochastic_and_final_prune_masks_are_bounded_and_deterministic():
    marked = torch.tensor([True, True, True, True, False])
    scores = torch.tensor([0., .2, .9, 1., 0.])
    origin = torch.tensor([0, 1, 2, 3, -1])
    torch.manual_seed(7)
    mask = FastGSStrategy._stochastic_mask(marked, scores, origin)
    assert mask.tolist() == [False, False, True, True, False]
    assert mask.sum() == 2  # unique draws; replacement would draw ID 3 twice for this seed
    params, optimizers = _model(3)
    state = FastGSStrategy().initialize_state()
    state.update({"pruning_score": torch.tensor([0., .95, 0.]), "grad2d": torch.zeros(3),
                  "absgrad2d": torch.zeros(3), "count": torch.zeros(3), "radii": torch.zeros(3),
                  "importance_score": torch.zeros(3), "origin": torch.arange(3)})
    params["opacities"].data[0] = torch.logit(torch.tensor(.05))
    FastGSStrategy()._final_prune(params, optimizers, state)
    assert len(params["means"]) == 1


def test_events_require_fresh_metrics_and_do_not_reset_opacity_late():
    params, optimizers = _model(2)
    strategy, state = FastGSStrategy(), FastGSStrategy().initialize_state()
    missing_importance = {key: value for key, value in _info(2).items() if key != "importance_score"}
    with pytest.raises(ValueError, match="fresh importance_score"):
        strategy.step_post_backward(params, optimizers, state, 600, missing_importance)
    params["opacities"].data.fill_(torch.logit(torch.tensor(.5)))
    strategy.step_post_backward(params, optimizers, state, 18_000, {
        "pruning_score": torch.zeros(2),
    })
    assert torch.allclose(torch.sigmoid(params["opacities"]), torch.full((2,), .5))


def test_refine_resets_screen_radius_before_pruning_like_original():
    params, optimizers = _model(10)
    strategy, state = FastGSStrategy(), FastGSStrategy().initialize_state(1.0)
    strategy._update_state(params, state, _info(10, torch.zeros(10)), require_metrics=True)
    state["radii"].fill_(100)
    strategy._refine(params, optimizers, state, step=3100)
    assert len(params["means"]) == 10
    assert not state["radii"].any()
