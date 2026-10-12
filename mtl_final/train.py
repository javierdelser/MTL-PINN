"""Training loops for one-phase, two-phase, only-voter, and backbone-frozen modes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from mtl_final.config import RunSpec, matched_two_phase_split
from mtl_final.csv_io import write_csv
from mtl_final.hybrid import HybridMTL
from mtl_final.metrics import TrainTimer, model_size_metrics, peak_memory_mb, reset_peak_memory
from mtl_final.models import MTLPINN, ParametricPINN, hidden_dim_for_budget
from mtl_final.solvers import load_snapshot, save_snapshot
from mtl_final.utils import count_parameters, package_root
from mtl_final.problems import get_problem
from mtl_final.utils import ensure_dir, make_scheduler, set_seed, write_json

# Weight of the numerical-head data term. The term is added only when the voter
# is trained, so the two-phase head phase does not see it.
SNAPSHOT_DATA_WEIGHT = 10.0


def _widths(problem, spec: RunSpec):
    """Backbone and voter widths. Unset fields use the per-PDE specification."""
    s = problem.spec
    hidden = s.hidden_dim if spec.hidden_dim is None else spec.hidden_dim
    layers = s.layers if spec.layers is None else spec.layers
    attn = s.attn_hidden if spec.attn_hidden is None else spec.attn_hidden
    return hidden, layers, attn


def build_baseline(problem, spec: RunSpec) -> ParametricPINN:
    """Plain PINN whose parameter count matches the MTL model for this PDE."""
    s = problem.spec
    _, layers, _ = _widths(problem, spec)
    reference = build_model(
        problem,
        RunSpec(
            pde=spec.pde,
            mode="one-phase",
            softmax=True,
            time_conditioned_voter=False,
            hidden_dim=spec.hidden_dim,
            layers=spec.layers,
            attn_hidden=spec.attn_hidden,
        ),
    )
    target = count_parameters(reference)
    hidden = hidden_dim_for_budget(s.n_coord + s.n_param, layers, s.n_fields, target)
    extra = s.extra
    return ParametricPINN(
        n_coord=s.n_coord,
        n_param=s.n_param,
        n_fields=s.n_fields,
        hidden_dim=hidden,
        layers=layers,
        coord_scale=extra.get("coord_scale"),
        param_offset=extra.get("param_offset"),
        param_scale=extra.get("param_scale"),
    )


def build_model(problem, spec: RunSpec) -> MTLPINN:
    s = problem.spec
    parametric = spec.parametric_trunk if spec.parametric_trunk is not None else s.parametric_trunk
    extra = s.extra
    hidden, layers, attn = _widths(problem, spec)
    return MTLPINN(
        n_coord=s.n_coord,
        n_param=s.n_param,
        n_fields=s.n_fields,
        task_params=s.task_params,
        hidden_dim=hidden,
        layers=layers,
        attn_hidden=attn,
        softmax=spec.softmax,
        time_conditioned_voter=spec.time_conditioned_voter and s.has_time,
        parametric_trunk=parametric,
        t_max=s.t_max,
        coord_scale=extra.get("coord_scale"),
        param_offset=extra.get("param_offset"),
        param_scale=extra.get("param_scale"),
    )


def snapshot_parameter(problem, spec: RunSpec) -> tuple[float, ...]:
    if spec.snapshot_split:
        return problem.mu_for_split(spec.snapshot_split)
    return tuple(float(v) for v in problem.spec.snapshot_param)


def cached_numerical_snapshot(problem, mu: tuple[float, ...]):
    root = package_root() / "outputs" / "hybrid" / "_solver_cache"
    key = problem.spec.name + "_" + "_".join(f"{float(v):.6g}" for v in mu).replace(".", "p").replace("-", "m")
    path = root / key
    loaded = load_snapshot(path)
    if loaded is not None:
        print(f"loaded solver snapshot {key}")
        return loaded
    print(f"computing solver snapshot {key}")
    snap = problem.numerical_snapshot(mu)
    save_snapshot(path, snap)
    return snap


def wrap_hybrid(model: MTLPINN, problem, mu: tuple[float, ...] | None = None) -> HybridMTL:
    chosen = tuple(float(v) for v in (mu if mu is not None else problem.spec.snapshot_param))
    snap = cached_numerical_snapshot(problem, chosen)
    return HybridMTL(model, snap, chosen, problem.spec.field_names)


def _snapshot_points(problem, model: HybridMTL, device, n: int) -> torch.Tensor:
    """Collocation points at the snapshot parameter.

    Lamb–Chaplygin samples inside and outside the moving dipole, so a generic
    residual box is not defined for that problem.
    """
    if hasattr(problem, "sample_interior") and hasattr(problem, "sample_exterior"):
        a, speed = (float(v) for v in model.extra_param)
        n_in = n // 2
        interior = problem.sample_interior(n_in, a, speed, device)
        exterior = problem.sample_exterior(n - n_in, a, speed, device)
        return torch.cat([interior, exterior], dim=0)
    return problem.sample_batch(n, device, "residual")


def _snapshot_match_loss(problem, model: HybridMTL, device) -> torch.Tensor:
    """Pull the mixture at the snapshot parameter onto the numerical field."""
    n = problem.spec.n_res
    packed = _snapshot_points(problem, model, device, n)
    cols = tuple(packed[:, i : i + 1] for i in range(problem.spec.n_coord))
    params = problem.pack_params(n, model.extra_param, device)
    t = problem.voter_time(packed)
    pred = model(packed, params, t=t, solver_coords=cols)
    target = model.solver_fields(*cols).detach()
    return torch.mean((pred - target) ** 2)


def _set_requires(module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def configure_trainable(model, mode: str, phase: str) -> None:
    """phase is 'joint', 'heads', 'voter', or 'rest'."""
    if mode == "baseline" or not hasattr(model, "attn_net"):
        for param in model.parameters():
            param.requires_grad = True
        return
    _set_requires(model.shared, False)
    for head in model.heads:
        _set_requires(head, False)
    _set_requires(model.attn_net, False)
    if mode == "one-phase" or phase == "joint":
        _set_requires(model.shared, True)
        for head in model.heads:
            _set_requires(head, True)
        _set_requires(model.attn_net, True)
    elif phase == "heads":
        _set_requires(model.shared, True)
        for head in model.heads:
            _set_requires(head, True)
    elif phase == "voter" or mode == "only-voter":
        _set_requires(model.attn_net, True)
    elif phase == "rest" or mode == "backbone-frozen-retrain-rest":
        for head in model.heads:
            _set_requires(head, True)
        _set_requires(model.attn_net, True)


def _active_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _task_loss(problem, model, mu, device, head_idx, attn_reg, task_index):
    hybrid = model if isinstance(model, HybridMTL) else None
    core = model.mtl if isinstance(model, HybridMTL) else model
    target = model if hybrid is not None else core
    return problem.physics_loss(
        target,
        mu,
        device,
        head_idx=head_idx,
        attn_reg=attn_reg,
        task_index=task_index,
        hybrid=hybrid,
    )


def train_epochs(
    problem,
    model,
    optimizer,
    epochs: int,
    device,
    *,
    isolated_heads: bool,
    attn_reg: float,
    lr_schedule: str,
    log_every: int = 100,
):
    scheduler = make_scheduler(optimizer, epochs, lr_schedule)
    timer = TrainTimer()
    timer.start()
    history = []
    tasks = problem.spec.task_params
    for epoch in range(epochs):
        with timer.epoch():
            optimizer.zero_grad()
            if isolated_heads:
                n_tasks = len(tasks)
                _set_requires(model.attn_net, False)
                _set_requires(model.shared, True)
                for head in model.heads:
                    _set_requires(head, True)
                task_losses = [
                    _task_loss(problem, model, mu, device, i, 0.0, None)[0]
                    for i, mu in enumerate(tasks)
                ]
                for head in model.heads:
                    _set_requires(head, False)
                backbone_loss = torch.stack(task_losses).mean()
                backbone_loss.backward(retain_graph=True)
                total = backbone_loss
                _set_requires(model.shared, False)
                for i, _ in enumerate(tasks):
                    for j, head in enumerate(model.heads):
                        _set_requires(head, j == i)
                    task_losses[i].backward(retain_graph=(i < n_tasks - 1))
                optimizer.step()
                for head in model.heads:
                    _set_requires(head, True)
                _set_requires(model.shared, True)
            else:
                total = 0.0
                for i, mu in enumerate(tasks):
                    loss, _ = _task_loss(problem, model, mu, device, None, attn_reg, i)
                    total = total + loss
                if isinstance(model, HybridMTL):
                    total = total + SNAPSHOT_DATA_WEIGHT * _snapshot_match_loss(problem, model, device)
                total.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        history.append(float(total.detach()))
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:6d} | loss {history[-1]:.3e}")
    return history, timer.summary()


def save_checkpoint(path: Path, model, spec: RunSpec, extra: dict | None = None) -> None:
    core = model.mtl if isinstance(model, HybridMTL) else model
    payload = {
        "state_dict": core.state_dict(),
        "spec": {
            "pde": spec.pde,
            "softmax": spec.softmax,
            "time_conditioned_voter": spec.time_conditioned_voter,
            "parametric_trunk": spec.parametric_trunk,
            "hidden_dim": spec.hidden_dim,
            "layers": spec.layers,
            "attn_hidden": spec.attn_hidden,
            "hybrid": isinstance(model, HybridMTL),
            "snapshot_param": getattr(model, "extra_param", None),
        },
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, problem, spec: RunSpec, device, hybrid: bool | None = None):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(problem, spec).to(device)
    saved = payload["spec"]
    want_hybrid = hybrid if hybrid is not None else saved.get("hybrid", False)
    if want_hybrid:
        mu = snapshot_parameter(problem, spec)
        # Expand voter before loading if the checkpoint already has the extra task.
        n_tasks = payload["state_dict"]["task_params"].shape[1]
        if n_tasks == model.n_heads + 1:
            model.expand_voter_for_extra_task(mu)
        model.load_state_dict(payload["state_dict"], strict=False)
        model = wrap_hybrid(model, problem, mu).to(device)
        # wrap_hybrid expands again only if heads == tasks; after load, tasks already extra.
        if model.mtl.task_params.shape[1] == len(model.mtl.heads) + 1:
            pass
        return model, payload
    model.load_state_dict(payload["state_dict"], strict=False)
    return model, payload


def run_training(spec: RunSpec) -> dict:
    set_seed(spec.seed)
    device = torch.device("cpu")
    from mtl_final.utils import resolve_device

    device = resolve_device(spec.device)
    problem = get_problem(spec.pde)
    budgets = spec.resolved_budgets()
    if spec.mode == "two-phase" and spec.head_epochs is None and spec.voter_epochs is None:
        heads, voter = matched_two_phase_split(budgets.head_epochs)
        budgets = replace(budgets, head_epochs=heads, voter_epochs=voter)
    out = ensure_dir(Path(spec.output_root) / spec.tag())
    reset_peak_memory(device)

    if spec.mode == "baseline":
        model = build_baseline(problem, spec).to(device)
    elif spec.init_checkpoint:
        model, _ = load_checkpoint(
            Path(spec.init_checkpoint), problem, spec, device, hybrid=spec.hybrid
        )
    else:
        model = build_model(problem, spec).to(device)
        if spec.hybrid:
            model = wrap_hybrid(model, problem, snapshot_parameter(problem, spec)).to(device)

    all_losses = []
    train_summary = {"n_epochs": 0, "time_per_epoch_s": 0.0, "total_train_time_s": 0.0}

    def _fit(phase: str, epochs: int, isolated: bool, attn_reg: float):
        nonlocal train_summary
        if epochs <= 0:
            return
        configure_trainable(model, spec.mode, phase)
        params = _active_params(model)
        if not params:
            raise RuntimeError(f"no trainable parameters for mode={spec.mode} phase={phase}")
        opt = torch.optim.Adam(params, lr=budgets.lr)
        print(f"[{spec.tag()}] phase={phase} epochs={epochs}")
        hist, summary = train_epochs(
            problem,
            model,
            opt,
            epochs,
            device,
            isolated_heads=isolated,
            attn_reg=attn_reg,
            lr_schedule=budgets.lr_schedule,
            log_every=max(1, epochs // 10),
        )
        all_losses.extend({"epoch": len(all_losses) + i, "phase": phase, "loss": v} for i, v in enumerate(hist))
        train_summary = {
            "n_epochs": train_summary["n_epochs"] + summary["n_epochs"],
            "total_train_time_s": train_summary["total_train_time_s"] + summary["total_train_time_s"],
        }
        n_ep = train_summary["n_epochs"]
        train_summary["time_per_epoch_s"] = (
            train_summary["total_train_time_s"] / n_ep if n_ep else float("nan")
        )

    mode = spec.mode
    if mode == "baseline":
        _fit("joint", budgets.head_epochs, False, 0.0)
    elif mode == "one-phase":
        _fit("joint", budgets.head_epochs, False, budgets.attn_reg)
    elif mode == "two-phase":
        _fit("heads", budgets.head_epochs, True, 0.0)
        _fit("voter", budgets.voter_epochs, False, 0.0)
    elif mode == "only-voter":
        _fit("voter", budgets.retrain_epochs, False, 0.0)
    elif mode == "backbone-frozen-retrain-rest":
        _fit("rest", budgets.retrain_epochs, False, budgets.attn_reg)
    else:
        raise ValueError(mode)

    mem = peak_memory_mb(device)
    save_checkpoint(out / "model.pth", model, spec)
    write_csv(out / "losses.csv", all_losses, ["epoch", "phase", "loss"])
    example = problem.example_inputs(device)
    size_metrics = model_size_metrics(model, example[0], example[1], example[2])
    metrics = {
        "pde": spec.pde,
        "mode": spec.mode,
        "softmax": spec.softmax,
        "time_conditioned_voter": spec.time_conditioned_voter,
        "hybrid": spec.hybrid,
        "snapshot_split": spec.snapshot_split or "",
        "snapshot_mu": ",".join(f"{float(v):g}" for v in getattr(model, "extra_param", ()) or ()),
        "seed": spec.seed,
        "peak_memory_mb": mem,
        **train_summary,
        **size_metrics,
        "output_dir": str(out),
    }
    write_json(out / "train_metrics.json", metrics)
    write_csv(out / "train_metrics.csv", [metrics], list(metrics.keys()))
    return {"model": model, "problem": problem, "device": device, "out": out, "metrics": metrics}
