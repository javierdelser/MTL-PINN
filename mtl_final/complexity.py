"""Runtime-free cost of a trained model.

Query cost is the linear-layer FLOPs of one forward at one collocation point.
Training cost multiplies that forward by the collocation points, tasks, and
epochs, and adds a backward equal to twice the forward of the layers that
receive gradients. Frozen layers stay in the forward and drop out of the
backward. Wall-clock time is not used.

FLOPs follow the Linear hook in ``metrics.count_linear_flops``:
``2 * batch * in_features * out_features``. Softmax is omitted: it is a few
operations per head and does not change the count at the reported precision,
so softmax and no-softmax share a row's cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mtl_final.config import RunSpec, matched_two_phase_split
from mtl_final.problems import get_problem
from mtl_final.train import build_baseline, build_model
from mtl_final.utils import count_parameters

EPOCHS = 20_000
# Manuscript cost of adding a numerical head by retraining only the voter.
VOTER_ONLY_EPOCHS = 500


def collocation_points(problem) -> int:
    """Network evaluations inside one task's physics loss."""
    spec = problem.spec
    name = spec.name
    if name == "poisson":
        return spec.n_res + 2 * spec.n_bc
    if name in {"heat", "burgers"}:
        return spec.n_res + spec.n_ic + 2 * spec.n_bc
    if name == "fisher_kpp":
        return spec.n_res + spec.n_ic + spec.n_bc
    if name == "lamb_chaplygin":
        # interior n_res, exterior 200, interface 100, initial, boundary
        return spec.n_res + 200 + 100 + spec.n_ic + spec.n_bc
    raise ValueError(name)


def _linear_flops(root: nn.Module, call) -> int:
    flops = 0
    handles = []

    def hook(mod, inputs, _output):
        nonlocal flops
        batch = inputs[0].shape[0]
        flops += 2 * batch * mod.in_features * mod.out_features

    for module in root.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        call()
    for handle in handles:
        handle.remove()
    return int(flops)


def _parts(problem, hybrid: bool) -> dict[str, int]:
    """Forward FLOPs of one collocation point, split by submodule.

    A hybrid still has four network heads. The extra atom is the numerical
    field, so only the voter (one more task pair) and the interpolant grow.
    """
    spec = RunSpec(pde=problem.spec.name, mode="one-phase", softmax=True)
    model = build_model(problem, spec)
    coords, params, t = problem.example_inputs(torch.device("cpu"), n=1)

    def full():
        model(coords, params, t=t)

    def isolated():
        model(coords, params, t=t, head_idx=0)

    backbone = _linear_flops(model.shared, full)
    heads = _linear_flops(model.heads, full)
    voter = _linear_flops(model.attn_net, full)
    one_head = _linear_flops(model.heads[0], isolated)
    trunk_once = _linear_flops(model.shared, isolated)
    interp = 0
    if hybrid:
        model.expand_voter_for_extra_task(problem.mu_for_split("interpolation"))
        voter = _linear_flops(model.attn_net, lambda: model.voter_logits(params, t))
        interp = 2 * problem.spec.n_fields * (1 << problem.spec.n_coord)
    return {
        "full": backbone + heads + voter + interp,
        "isolated": trunk_once + one_head,
        "backward_full": backbone + heads + voter,
        "backward_voter": voter,
        "backward_rest": heads + voter,
        "backward_isolated": trunk_once + one_head,
        "interp": interp,
    }


def _pinn_forward(problem) -> tuple[int, int]:
    spec = RunSpec(pde=problem.spec.name, mode="baseline")
    model = build_baseline(problem, spec)
    coords, params, _t = problem.example_inputs(torch.device("cpu"), n=1)
    flops = _linear_flops(model, lambda: model(coords, params))
    return flops, count_parameters(model)


def _train_flops(kind: str, parts: dict[str, int], points: int, n_tasks: int, snapshot_points: int) -> int:
    """kind is pinn, one-phase, two-phase, only-voter, or backbone-frozen."""
    heads, voter = matched_two_phase_split(EPOCHS)

    def step(forward: int, backward: int, n_points: int, epochs: int) -> int:
        return n_points * (forward + 2 * backward) * epochs

    if kind == "pinn":
        return step(parts["full"], parts["full"], points, EPOCHS) * n_tasks
    if kind == "one-phase":
        cost = step(parts["full"], parts["backward_full"], points, EPOCHS) * n_tasks
        if snapshot_points:
            cost += step(parts["full"], parts["backward_full"], snapshot_points, EPOCHS)
        return cost
    if kind == "two-phase":
        cost = step(parts["isolated"], parts["backward_isolated"], points, heads) * n_tasks
        cost += step(parts["full"], parts["backward_voter"], points, voter) * n_tasks
        if snapshot_points:
            cost += step(parts["full"], parts["backward_voter"], snapshot_points, voter)
        return cost
    if kind == "only-voter":
        cost = step(parts["full"], parts["backward_voter"], points, VOTER_ONLY_EPOCHS) * n_tasks
        cost += step(parts["full"], parts["backward_voter"], snapshot_points, VOTER_ONLY_EPOCHS)
        return cost
    if kind == "backbone-frozen":
        cost = step(parts["full"], parts["backward_rest"], points, EPOCHS) * n_tasks
        cost += step(parts["full"], parts["backward_rest"], snapshot_points, EPOCHS)
        return cost
    raise ValueError(kind)


def variant_costs(pde: str) -> list[dict]:
    problem = get_problem(pde)
    points = collocation_points(problem)
    n_tasks = len(problem.spec.task_params)
    mtl = _parts(problem, hybrid=False)
    hybrid = _parts(problem, hybrid=True)
    pinn_flops, pinn_params = _pinn_forward(problem)
    mtl_model = build_model(problem, RunSpec(pde=pde, mode="one-phase", softmax=True))
    mtl_params = count_parameters(mtl_model)
    hybrid_model = build_model(problem, RunSpec(pde=pde, mode="one-phase", softmax=True, hybrid=True))
    hybrid_model.expand_voter_for_extra_task(problem.mu_for_split("interpolation"))
    # The numerical field is not a parameter. The voter buffer grows by one task.
    hybrid_params = count_parameters(hybrid_model)

    rows = [
        {
            "pde": pde,
            "variant": "Plain PINN",
            "group": "comparison",
            "n_parameters": pinn_params,
            "query_flops": pinn_flops,
            "train_flops": _train_flops("pinn", {"full": pinn_flops}, points, n_tasks, 0),
        },
        {
            "pde": pde,
            "variant": "One-phase",
            "group": "comparison",
            "n_parameters": mtl_params,
            "query_flops": mtl["full"],
            "train_flops": _train_flops("one-phase", mtl, points, n_tasks, 0),
        },
        {
            "pde": pde,
            "variant": "Two-phase",
            "group": "comparison",
            "n_parameters": mtl_params,
            "query_flops": mtl["full"],
            "train_flops": _train_flops("two-phase", mtl, points, n_tasks, 0),
        },
    ]
    labels = {
        "one-phase": "Hybrid one-phase",
        "two-phase": "Hybrid two-phase",
        "only-voter": "Hybrid voter only",
        "backbone-frozen": "Hybrid backbone frozen",
    }
    kinds = {
        "one-phase": "one-phase",
        "two-phase": "two-phase",
        "only-voter": "only-voter",
        "backbone-frozen": "backbone-frozen",
    }
    for key, label in labels.items():
        rows.append(
            {
                "pde": pde,
                "variant": label,
                "group": "hybrid",
                "n_parameters": hybrid_params,
                "query_flops": hybrid["full"],
                "train_flops": _train_flops(kinds[key], hybrid, points, n_tasks, problem.spec.n_res),
            }
        )
    return rows


def all_costs() -> list[dict]:
    rows = []
    for name in ("poisson", "heat", "burgers", "fisher_kpp", "lamb_chaplygin"):
        rows.extend(variant_costs(name))
    return rows
