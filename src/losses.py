"""Loss factory. Ignore_index=6 drops Unknown pixels from all losses."""
from __future__ import annotations

from typing import Callable, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .utils import IGNORE_INDEX, NUM_CLASSES


class CombinedLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, ce_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
        self.w = ce_weight

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.w * self.ce(pred, tgt) + (1.0 - self.w) * self.dice(pred, tgt)


def get_loss(
    name: str,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    class_weights: Optional[torch.Tensor] = None,
) -> Callable:
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    if name == "dice":
        return smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
    if name == "focal":
        return smp.losses.FocalLoss(
            mode="multiclass", alpha=0.25, gamma=2.0, ignore_index=ignore_index
        )
    if name == "combined":
        return CombinedLoss(ignore_index=ignore_index)
    raise ValueError(f"unknown loss: {name}")
