"""CLI for the clean MTL-PINN experiment suite."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

from mtl_final.config import PDE_NAMES, RUN_MODES, RunSpec, matched_two_phase_split, parse_seeds
from mtl_final.problems import get_problem
from mtl_final.evaluate import SUMMARY_FIELDS, append_master_summary, evaluate_run
from mtl_final.train import run_training
from mtl_final.utils import package_root


def _bool_flag(parser, name, dest, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: None})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MTL-PINN suite: multi-seed training, voter ablations, hybrid solver heads.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one configuration (one seed).")
    _add_common(train)
    train.add_argument("--pde", required=True, choices=PDE_NAMES)
    train.add_argument("--mode", default="two-phase", choices=RUN_MODES)
    train.add_argument("--seed", type=int, default=50)
    train.add_argument("--softmax", dest="softmax", action="store_true")
    train.add_argument("--nosoftmax", dest="softmax", action="store_false")
    train.set_defaults(softmax=True)
    train.add_argument("--hybrid", action="store_true")
    train.add_argument("--init-checkpoint", type=Path, default=None)
    train.add_argument("--skip-eval", action="store_true")

    run = sub.add_parser("run", help="Sweep PDEs, modes, softmax, and seeds.")
    _add_common(run)
    run.add_argument("--pdes", nargs="+", default=list(PDE_NAMES), choices=PDE_NAMES)
    run.add_argument("--modes", nargs="+", default=["one-phase", "two-phase"], choices=RUN_MODES)
    run.add_argument("--softmax-modes", nargs="+", default=["softmax", "nosoftmax"], choices=["softmax", "nosoftmax"])
    run.add_argument("--hybrid-modes", nargs="+", default=["off"], choices=["off", "on", "both"] + list(RUN_MODES))
    run.add_argument("--seeds", nargs="+", type=int, default=None)
    run.add_argument("--n-seeds", type=int, default=1)
    run.add_argument("--seed-start", type=int, default=50)
    run.add_argument("--base-checkpoint-root", type=Path, default=None,
                     help="If set, hybrid/retrain modes load PINN checkpoints from this tree.")

    smoke = sub.add_parser("smoke", help="Tiny CPU run that writes CSVs for every PDE.")
    smoke.add_argument("--output-root", type=Path, default=None)
    smoke.add_argument("--device", default="cpu")

    compare = sub.add_parser(
        "compare",
        help="Matched-budget comparison: one-phase, two-phase, softmax, plain PINN.",
    )
    _add_common(compare)
    compare.add_argument("--pdes", nargs="+", default=list(PDE_NAMES), choices=PDE_NAMES)
    compare.add_argument("--epochs", type=int, default=20_000, help="Shared epoch budget for every model.")
    compare.add_argument("--seed", type=int, default=50)

    hybrid = sub.add_parser(
        "hybrid",
        help="Add one numerical solver head at an interpolation or extrapolation parameter.",
    )
    _add_common(hybrid)
    hybrid.add_argument("--pdes", nargs="+", default=list(PDE_NAMES), choices=PDE_NAMES)
    hybrid.add_argument(
        "--modes",
        nargs="+",
        default=["one-phase", "two-phase", "only-voter", "backbone-frozen-retrain-rest"],
        choices=["one-phase", "two-phase", "only-voter", "backbone-frozen-retrain-rest"],
    )
    hybrid.add_argument(
        "--snapshot-splits",
        nargs="+",
        default=["interpolation", "extrapolation"],
        choices=["interpolation", "extrapolation"],
    )
    hybrid.add_argument("--epochs", type=int, default=20000, help="Budget for one-phase and two-phase.")
    hybrid.add_argument("--seed", type=int, default=50)
    hybrid.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Comparison tree whose two-phase softmax checkpoints initialize the dictionary.",
    )
    return p


def _add_common(p):
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--head-epochs", type=int, default=None)
    p.add_argument("--voter-epochs", type=int, default=None)
    p.add_argument("--retrain-epochs", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None, help="Backbone width. Default: the width of that PDE.")
    p.add_argument("--layers", type=int, default=None, help="Backbone depth. Default: the depth of that PDE.")
    p.add_argument("--attn-hidden", type=int, default=None, help="Voter width. Default: the width of that PDE.")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-schedule", default=None, choices=["cosine", "constant"])
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--nt", type=int, default=None)


def _output_root(args) -> Path:
    if args.output_root is not None:
        return Path(args.output_root)
    return package_root() / "outputs"


def _spec_from_train(args) -> RunSpec:
    return RunSpec(
        pde=args.pde,
        mode=args.mode,
        softmax=args.softmax,
        time_conditioned_voter=False,
        hybrid=bool(args.hybrid),
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        attn_hidden=args.attn_hidden,
        head_epochs=args.head_epochs,
        voter_epochs=args.voter_epochs,
        retrain_epochs=args.retrain_epochs,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        device=args.device,
        output_root=str(_output_root(args)),
        init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None,
    )


def _grid_kw(args) -> dict:
    kw = {}
    if getattr(args, "nx", None):
        kw["nx"] = args.nx
        kw["ny"] = args.nx
    if getattr(args, "nt", None):
        kw["nt"] = args.nt
    return kw


def _execute(spec: RunSpec, grid_kw: dict, skip_eval: bool = False) -> dict:
    bundle = run_training(spec)
    meta = {
        "pde": spec.pde,
        "mode": spec.mode,
        "softmax": spec.softmax,
        "time_conditioned_voter": spec.time_conditioned_voter,
        "hybrid": spec.hybrid,
        "snapshot_split": spec.snapshot_split or "",
        "snapshot_mu": "",
        "seed": spec.seed,
    }
    if spec.hybrid:
        problem = get_problem(spec.pde)
        mu = problem.mu_for_split(spec.snapshot_split or "interpolation")
        meta["snapshot_mu"] = ",".join(f"{float(v):g}" for v in mu)
    if not skip_eval:
        ev = evaluate_run(bundle["problem"], bundle["model"], bundle["device"], bundle["out"], meta, grid_kw)
        bundle["metrics"]["mean_test_time_per_point_s"] = ev["mean_test_time_per_point_s"]
    else:
        bundle["metrics"]["mean_test_time_per_point_s"] = ""
    append_master_summary(Path(spec.output_root), {k: bundle["metrics"].get(k, "") for k in SUMMARY_FIELDS})
    print(f"Wrote {bundle['out']}")
    return bundle


def run_comparison(args) -> None:
    """One-phase vs two-phase at the same epoch count, softmax on or off, and a plain PINN."""
    root = Path(args.output_root) if args.output_root is not None else package_root() / "outputs" / "comparison"
    heads, voter = matched_two_phase_split(args.epochs)
    grid = _grid_kw(args)
    if "nx" not in grid:
        grid.update({"nx": 41, "ny": 41, "nt": 21})
    for pde in args.pdes:
        run_grid = dict(grid)
        base = RunSpec(
            pde=pde,
            mode="baseline",
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            attn_hidden=args.attn_hidden,
            head_epochs=args.epochs,
            lr=args.lr,
            lr_schedule=args.lr_schedule,
            device=args.device,
            output_root=str(root),
        )
        _execute(base, run_grid)
        for softmax in (True, False):
            one = RunSpec(
                pde=pde,
                mode="one-phase",
                softmax=softmax,
                time_conditioned_voter=False,
                seed=args.seed,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                attn_hidden=args.attn_hidden,
                head_epochs=args.epochs,
                voter_epochs=0,
                lr=args.lr,
                lr_schedule=args.lr_schedule,
                device=args.device,
                output_root=str(root),
            )
            two = RunSpec(
                pde=pde,
                mode="two-phase",
                softmax=softmax,
                time_conditioned_voter=False,
                seed=args.seed,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                attn_hidden=args.attn_hidden,
                head_epochs=heads,
                voter_epochs=voter,
                lr=args.lr,
                lr_schedule=args.lr_schedule,
                device=args.device,
                output_root=str(root),
            )
            _execute(one, run_grid)
            _execute(two, run_grid)
    from mtl_final.report import write_comparison_report

    path = write_comparison_report(root)
    print(f"Comparison report: {path}")


def run_hybrid(args) -> None:
    """Retrain every hybrid mode with a solver atom at the interp or extrap query.

    The dictionary is the two-phase softmax model from the comparison.
    One-phase, two-phase, voter-only, and backbone-frozen retraining use
    ``--epochs`` unless ``--retrain-epochs`` is set for the last two modes.
    Two-phase splits that budget 5:1. The numerical-head data term is used
    whenever the voter is updated, and not during the isolated head phase.
    """
    root = _output_root(args) if args.output_root is not None else package_root() / "outputs" / "hybrid"
    ckpt_root = Path(args.checkpoint_root) if args.checkpoint_root else package_root() / "outputs" / "comparison"
    if not ckpt_root.is_absolute():
        ckpt_root = package_root() / ckpt_root
    heads, voter = matched_two_phase_split(args.epochs)
    retrain = args.retrain_epochs if args.retrain_epochs is not None else args.epochs
    grid = _grid_kw(args)
    if "nx" not in grid:
        grid.update({"nx": 41, "ny": 41, "nt": 21})
    for pde in args.pdes:
        run_grid = dict(grid)
        ckpt = ckpt_root / pde / "mtl" / "two-phase" / "softmax" / "muvoter" / f"seed{args.seed}" / "model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing dictionary checkpoint: {ckpt}")
        for split in args.snapshot_splits:
            for mode in args.modes:
                spec = RunSpec(
                    pde=pde,
                    mode=mode,
                    softmax=True,
                    time_conditioned_voter=False,
                    hybrid=True,
                    snapshot_split=split,
                    seed=args.seed,
                    hidden_dim=args.hidden_dim,
                    layers=args.layers,
                    attn_hidden=args.attn_hidden,
                    head_epochs=args.epochs if mode == "one-phase" else (heads if mode == "two-phase" else None),
                    voter_epochs=voter if mode == "two-phase" else None,
                    retrain_epochs=retrain if mode in {"only-voter", "backbone-frozen-retrain-rest"} else None,
                    lr=args.lr,
                    lr_schedule=args.lr_schedule,
                    device=args.device,
                    output_root=str(root),
                    init_checkpoint=str(ckpt),
                )
                done = Path(root) / spec.tag() / "model.pth"
                if done.exists() and (done.parent / "evaluation.csv").exists():
                    print(f"skip existing {spec.tag()}")
                    continue
                _execute(spec, run_grid)
    from mtl_final.report import write_comparison_report

    path = write_comparison_report(root)
    print(f"Hybrid report: {path}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        run_comparison(args)
        return
    if args.command == "hybrid":
        run_hybrid(args)
        return
    if args.command == "train":
        _execute(_spec_from_train(args), _grid_kw(args), skip_eval=args.skip_eval)
        return
    if args.command == "smoke":
        root = args.output_root or (package_root() / "outputs" / "smoke")
        for pde in PDE_NAMES:
            spec = RunSpec(
                pde=pde,
                mode="one-phase",
                softmax=True,
                time_conditioned_voter=False,
                hybrid=False,
                seed=0,
                head_epochs=2,
                voter_epochs=1,
                device=args.device,
                output_root=str(root),
            )
            _execute(spec, {"nx": 16, "nt": 8, "ny": 16, "snapshot_time": 0.2})
        print(f"Smoke outputs in {root}")
        return

    seeds = parse_seeds(args.seeds, args.n_seeds, args.seed_start)
    hy_request = args.hybrid_modes
    hybrid_flags = []
    hybrid_retrain = []
    if "both" in hy_request:
        hybrid_flags = [False, True]
        hybrid_retrain = [m for m in RUN_MODES if m != "baseline"]
    elif "on" in hy_request:
        hybrid_flags = [True]
        hybrid_retrain = [m for m in hy_request if m in RUN_MODES] or list(args.modes)
    elif any(m in RUN_MODES for m in hy_request):
        hybrid_flags = [True]
        hybrid_retrain = [m for m in hy_request if m in RUN_MODES]
    else:
        hybrid_flags = [False]
        hybrid_retrain = list(args.modes)

    root = _output_root(args)
    for pde, softmax_name, seed in product(args.pdes, args.softmax_modes, seeds):
        softmax = softmax_name == "softmax"
        if False in hybrid_flags:
            for mode in args.modes:
                spec = RunSpec(
                    pde=pde,
                    mode=mode,
                    softmax=softmax,
                    time_conditioned_voter=False,
                    hybrid=False,
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    layers=args.layers,
                    attn_hidden=args.attn_hidden,
                    head_epochs=args.head_epochs,
                    voter_epochs=args.voter_epochs,
                    retrain_epochs=args.retrain_epochs,
                    lr=args.lr,
                    lr_schedule=args.lr_schedule,
                    device=args.device,
                    output_root=str(root),
                )
                bundle = _execute(spec, _grid_kw(args))
                if True in hybrid_flags:
                    ckpt = bundle["out"] / "model.pth"
                    for hmode in hybrid_retrain:
                        hspec = RunSpec(
                            pde=pde,
                            mode=hmode,
                            softmax=softmax,
                            time_conditioned_voter=False,
                            hybrid=True,
                            seed=seed,
                            hidden_dim=args.hidden_dim,
                            layers=args.layers,
                            attn_hidden=args.attn_hidden,
                            head_epochs=args.head_epochs,
                            voter_epochs=args.voter_epochs,
                            retrain_epochs=args.retrain_epochs,
                            lr=args.lr,
                            lr_schedule=args.lr_schedule,
                            device=args.device,
                            output_root=str(root),
                            init_checkpoint=str(ckpt),
                        )
                        _execute(hspec, _grid_kw(args))
        elif True in hybrid_flags:
            for hmode in hybrid_retrain:
                spec = RunSpec(
                    pde=pde,
                    mode=hmode,
                    softmax=softmax,
                    time_conditioned_voter=False,
                    hybrid=True,
                    seed=seed,
                    hidden_dim=args.hidden_dim,
                    layers=args.layers,
                    attn_hidden=args.attn_hidden,
                    head_epochs=args.head_epochs,
                    voter_epochs=args.voter_epochs,
                    retrain_epochs=args.retrain_epochs,
                    lr=args.lr,
                    lr_schedule=args.lr_schedule,
                    device=args.device,
                    output_root=str(root),
                    init_checkpoint=str(args.base_checkpoint_root) if args.base_checkpoint_root else None,
                )
                _execute(spec, _grid_kw(args))


if __name__ == "__main__":
    main()
