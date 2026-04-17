"""Confusion-matrix based mIoU / Dice / pixel accuracy."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .utils import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


class ConfusionMatrix:
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.n = num_classes
        self.ig = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.mat.fill(0)

    def update(self, pred: np.ndarray, tgt: np.ndarray) -> None:
        if pred.shape != tgt.shape:
            raise ValueError(f"shape mismatch {pred.shape} vs {tgt.shape}")
        mask = tgt != self.ig
        p = pred[mask].astype(np.int64)
        t = tgt[mask].astype(np.int64)
        valid = (p >= 0) & (p < self.n) & (t >= 0) & (t < self.n)
        p, t = p[valid], t[valid]
        idx = self.n * t + p
        self.mat += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def _valid_classes(self) -> List[int]:
        return [i for i in range(self.n) if i != self.ig]

    def iou(self) -> Tuple[np.ndarray, float]:
        tp = np.diag(self.mat)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        denom = np.maximum(tp + fp + fn, 1)
        iou = tp / denom
        valid = self._valid_classes()
        return iou, float(np.mean(iou[valid]))

    def dice(self) -> Tuple[np.ndarray, float]:
        tp = np.diag(self.mat)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        denom = np.maximum(2 * tp + fp + fn, 1)
        d = 2 * tp / denom
        valid = self._valid_classes()
        return d, float(np.mean(d[valid]))

    def pixel_acc(self) -> float:
        total = self.mat.sum()
        return float(np.diag(self.mat).sum() / max(total, 1))

    def report(self) -> dict:
        iou, miou = self.iou()
        dice, mdice = self.dice()
        return {
            "mIoU": miou,
            "mDice": mdice,
            "pixel_acc": self.pixel_acc(),
            "per_class_IoU": {CLASS_NAMES[i]: float(iou[i]) for i in range(self.n)},
            "per_class_Dice": {CLASS_NAMES[i]: float(dice[i]) for i in range(self.n)},
        }
