"""Experiment constants and default budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

PDE_NAMES = (
    "poisson",
    "heat",
    "burgers",
    "fisher_kpp",
    "lamb_chaplygin",
)

RUN_MODES = (
    "one-phase",
    "two-phase",
    "baseline",
    "only-voter",
    "backbone-frozen-retrain-rest",
)

SOFTMAX_CHOICES = ("softmax", "nosoftmax")


# Shared budget of the manuscript. Two-phase keeps a 5:1 split of this total
# (16,667 epochs on the backbone and heads, 3,333 on the voter).
PAPER_EPOCHS = 20_000


@dataclass
class TrainBudgets:
    head_epochs: int = PAPER_EPOCHS
    voter_epochs: int = 3_333
    retrain_epochs: int = PAPER_EPOCHS
    lr: float = 1e-3
    attn_reg: float = 0.1
    lr_schedule: str = "cosine"


DEFAULT_BUDGETS: dict[str, TrainBudgets] = {name: TrainBudgets() for name in PDE_NAMES}


@dataclass
class RunSpec:
    pde: str
    mode: str
    softmax: bool = True
    time_conditioned_voter: bool = False
    hybrid: bool = False
    seed: int = 50
    hidden_dim: int | None = None
    layers: int | None = None
    attn_hidden: int | None = None
    parametric_trunk: bool | None = None
    head_epochs: int | None = None
    voter_epochs: int | None = None
    retrain_epochs: int | None = None
    lr: float | None = None
    attn_reg: float | None = None
    lr_schedule: str | None = None
    device: str = "auto"
    output_root: str = "outputs"
    init_checkpoint: str | None = None
    snapshot_split: str | None = None
    extra: dict = field(default_factory=dict)

    def resolved_budgets(self) -> TrainBudgets:
        base = DEFAULT_BUDGETS[self.pde]
        return TrainBudgets(
            head_epochs=self.head_epochs if self.head_epochs is not None else base.head_epochs,
            voter_epochs=self.voter_epochs if self.voter_epochs is not None else base.voter_epochs,
            retrain_epochs=self.retrain_epochs if self.retrain_epochs is not None else base.retrain_epochs,
            lr=self.lr if self.lr is not None else base.lr,
            attn_reg=self.attn_reg if self.attn_reg is not None else base.attn_reg,
            lr_schedule=self.lr_schedule if self.lr_schedule is not None else base.lr_schedule,
        )

    def tag(self) -> str:
        if self.mode == "baseline":
            return f"{self.pde}/baseline/pinn/seed{self.seed}"
        sm = "softmax" if self.softmax else "nosoftmax"
        tv = "timevoter" if self.time_conditioned_voter else "muvoter"
        hy = "hybrid" if self.hybrid else "mtl"
        if self.hybrid:
            split = "interp" if (self.snapshot_split or "interpolation").startswith("interp") else "extrap"
            return f"{self.pde}/{hy}/{split}/{self.mode}/{sm}/{tv}/seed{self.seed}"
        return f"{self.pde}/{hy}/{self.mode}/{sm}/{tv}/seed{self.seed}"


def matched_two_phase_split(total_epochs: int, head_fraction: float = 5.0 / 6.0) -> tuple[int, int]:
    """Split a shared epoch budget into head specialization and voter training.

    One-phase and the plain PINN use ``total_epochs`` directly. Two-phase uses
    the same total, with the paper's 5:1 head-to-voter ratio.
    """
    total = max(int(total_epochs), 2)
    heads = int(round(total * head_fraction))
    heads = min(max(heads, 1), total - 1)
    return heads, total - heads


def parse_seeds(values: Sequence[int] | None, n_seeds: int | None, seed_start: int) -> list[int]:
    if values:
        return [int(v) for v in values]
    n = int(n_seeds or 1)
    return [int(seed_start) + i for i in range(n)]
