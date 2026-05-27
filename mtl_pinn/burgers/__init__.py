"""Viscous Burgers equation experiments."""

from mtl_pinn.burgers.core import BatchedMultiHeadPINN_Burgers_Attention, predict, train

__all__ = ["BatchedMultiHeadPINN_Burgers_Attention", "train", "predict"]
