"""Runtime accounting: parameters, FLOPs, wall time, peak memory."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mtl_final.utils import count_parameters


def _rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    return _rss_mb()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def count_linear_flops(model: nn.Module, *example_inputs: torch.Tensor) -> int:
    """MAC-style FLOPs (2 * in * out * batch) for Linear layers in one forward."""
    flops = 0
    handles = []

    def hook(mod, inputs, _output):
        nonlocal flops
        x = inputs[0]
        batch = x.shape[0]
        flops += 2 * batch * mod.in_features * mod.out_features

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(*example_inputs)
    for handle in handles:
        handle.remove()
    return int(flops)


@dataclass
class TrainTimer:
    epoch_times: list[float] = field(default_factory=list)
    t0: float = 0.0

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.epoch_times.clear()

    @contextmanager
    def epoch(self):
        start = time.perf_counter()
        yield
        self.epoch_times.append(time.perf_counter() - start)

    def summary(self) -> dict:
        n = len(self.epoch_times)
        total = float(sum(self.epoch_times))
        return {
            "n_epochs": n,
            "time_per_epoch_s": (total / n) if n else float("nan"),
            "total_train_time_s": total,
            "wall_time_s": time.perf_counter() - self.t0 if self.t0 else total,
        }


def model_size_metrics(model: nn.Module, *example_inputs: torch.Tensor) -> dict:
    return {
        "n_parameters": count_parameters(model),
        "n_trainable_parameters": count_parameters(model, trainable_only=True),
        "flops_per_forward": count_linear_flops(model, *example_inputs),
    }
