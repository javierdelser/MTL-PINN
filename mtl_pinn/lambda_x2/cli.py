"""Command-line interface for the λx² benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from mtl_pinn.lambda_x2 import core
from mtl_pinn.utils import configure_matplotlib, repo_root, resolve_device, set_seed


def _default_output_dir() -> Path:
    return repo_root() / "outputs" / "lambda_x2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multitask PINN for -u_xx + 2λ = 0 with u(x;λ) = λx².",
    )
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument(
        "--strategy",
        choices=["one-phase", "two-phase"],
        default="two-phase",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--head-epochs", type=int, default=30_000)
    parser.add_argument("--attn-epochs", type=int, default=5_000)
    parser.add_argument("--params", type=float, nargs="+", default=None)
    parser.add_argument("--nx", type=int, default=30)
    parser.add_argument("--save-figure", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_matplotlib(interactive=args.show)
    set_seed(args.seed)
    core.device = resolve_device(args.device)
    output_dir = args.output_dir or _default_output_dir()

    if args.command == "train":
        path = core.train(
            strategy=args.strategy,
            output_dir=output_dir,
            head_epochs=args.head_epochs,
            attn_epochs=args.attn_epochs,
            seed=args.seed,
        )
        print(f"Saved model to {path}")
    elif args.command == "predict":
        core.predict(
            output_dir=output_dir,
            param_values=args.params,
            nx=args.nx,
            show=args.show,
            save_path=args.save_figure,
        )


if __name__ == "__main__":
    main()
