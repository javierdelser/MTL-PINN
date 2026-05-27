"""λx² elliptic benchmark experiments."""

from mtl_pinn.lambda_x2.core import BatchedMultiHeadPINN_Attention, predict, train

__all__ = ["BatchedMultiHeadPINN_Attention", "train", "predict"]
