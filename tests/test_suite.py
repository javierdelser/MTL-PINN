from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mtl_final.config import PDE_NAMES, RunSpec
from mtl_final.evaluate import evaluate_run
from mtl_final.problems import get_problem
from mtl_final.train import build_model, run_training, wrap_hybrid


def test_all_problems_instantiate():
    for name in PDE_NAMES:
        problem = get_problem(name)
        assert len(problem.spec.task_params) == 4
        mu = problem.spec.task_params[0]
        grid = problem.eval_grid(mu, torch.device("cpu"), nx=8, nt=4, ny=8, snapshot_time=0.1)
        true = problem.reference(grid.coords, mu)
        for field in problem.spec.field_names:
            assert field in true
            assert np.isfinite(np.asarray(true[field])).all()


def test_complexity_separates_schedules():
    from mtl_final.complexity import variant_costs

    rows = {row["variant"]: row for row in variant_costs("poisson")}
    assert rows["One-phase"]["query_flops"] == rows["Two-phase"]["query_flops"]
    assert rows["One-phase"]["train_flops"] > rows["Two-phase"]["train_flops"]
    assert rows["Hybrid voter only"]["train_flops"] < rows["Hybrid one-phase"]["train_flops"]
    assert rows["Hybrid one-phase"]["query_flops"] > rows["One-phase"]["query_flops"]


def test_softmax_weight_shape():
    problem = get_problem("heat")
    spec = RunSpec(pde="heat", mode="one-phase", softmax=False)
    model = build_model(problem, spec)
    x = torch.rand(5, 2)
    p = torch.full((5, 1), 0.1)
    t = x[:, 1:2]
    w = model.get_attention_weights(p, t)
    assert w.shape == (5, 4)
    u = model(x, p, t)
    assert u.shape == (5, 1)


def test_hybrid_adds_solver_head():
    problem = get_problem("poisson")
    spec = RunSpec(pde="poisson", mode="only-voter", softmax=True, hybrid=True, snapshot_split="extrapolation")
    model = wrap_hybrid(build_model(problem, spec), problem, problem.mu_for_split("extrapolation"))
    assert model.mtl.task_params.shape[1] == 5
    assert abs(float(model.extra_param[0]) - 8.0) < 1e-8
    assert "extrap" in spec.tag()
    x = torch.linspace(-1, 1, 7).view(-1, 1)
    p = torch.full((7, 1), 2.0)
    u = model(x, p, solver_coords=(x,))
    assert u.shape == (7, 1)


def test_poisson_smoke_writes_csv(tmp_path: Path):
    spec = RunSpec(
        pde="poisson",
        mode="one-phase",
        softmax=True,
        seed=0,
        head_epochs=2,
        device="cpu",
        output_root=str(tmp_path),
    )
    bundle = run_training(spec)
    ev = evaluate_run(
        bundle["problem"],
        bundle["model"],
        bundle["device"],
        bundle["out"],
        {"pde": "poisson", "mode": "one-phase", "softmax": True, "time_conditioned_voter": False, "hybrid": False, "seed": 0},
        {"nx": 16},
    )
    assert (bundle["out"] / "train_metrics.csv").exists()
    assert (bundle["out"] / "evaluation.csv").exists()
    assert (bundle["out"] / "voter_weights.csv").exists()
    assert list(Path(bundle["out"] / "fields").glob("*.csv"))
    assert ev["rows"]


def test_two_phase_and_nosoftmax(tmp_path: Path):
    spec = RunSpec(
        pde="heat",
        mode="two-phase",
        softmax=False,
        seed=1,
        head_epochs=1,
        voter_epochs=1,
        device="cpu",
        output_root=str(tmp_path),
    )
    bundle = run_training(spec)
    assert (bundle["out"] / "losses.csv").exists()
    assert bundle["metrics"]["n_parameters"] > 0
    assert bundle["metrics"]["flops_per_forward"] > 0
