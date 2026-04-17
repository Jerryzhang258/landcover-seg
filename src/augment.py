"""Albumentations pipelines.

smp encoders with 5 downsample stages need H and W divisible by 32. Our tiles
are 816x816 and 816 % 32 == 16, so we reflect-pad to 832x832 at the top of
each pipeline. Padded mask regions are set to IGNORE_INDEX so they don't
contribute to the loss.
"""
from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD


def _pad_to_multiple_of_32(x: int) -> int:
    return ((x + 31) // 32) * 32


def train_tf(tile_size: int = 816) -> A.Compose:
    pad_to = _pad_to_multiple_of_32(tile_size)
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=pad_to, min_width=pad_to,
                border_mode=cv2.BORDER_REFLECT_101,
                mask_value=IGNORE_INDEX,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5,
                border_mode=cv2.BORDER_REFLECT_101,
                mask_value=IGNORE_INDEX,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def val_tf(tile_size: int = 816) -> A.Compose:
    pad_to = _pad_to_multiple_of_32(tile_size)
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=pad_to, min_width=pad_to,
                border_mode=cv2.BORDER_REFLECT_101,
                mask_value=IGNORE_INDEX,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
