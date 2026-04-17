"""Albumentations pipelines."""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import IMAGENET_MEAN, IMAGENET_STD


def train_tf(tile_size: int = 816) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5,
                border_mode=0,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def val_tf() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
