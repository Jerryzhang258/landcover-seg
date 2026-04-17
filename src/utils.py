"""DeepGlobe land-cover constants and RGB<->label conversion."""
from __future__ import annotations

import os
import random
from typing import Dict, Tuple

import numpy as np

CLASS_COLORS: Dict[Tuple[int, int, int], int] = {
    (0, 255, 255): 0,    # Urban
    (255, 255, 0): 1,    # Agriculture
    (255, 0, 255): 2,    # Rangeland
    (0, 255, 0): 3,      # Forest
    (0, 0, 255): 4,      # Water
    (255, 255, 255): 5,  # Barren
    (0, 0, 0): 6,        # Unknown
}

NUM_CLASSES = 7
IGNORE_INDEX = 6
CLASS_NAMES = [
    "Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rgb_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 RGB mask to an HxW uint8 label map."""
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB mask, got shape {mask_rgb.shape}")
    h, w, _ = mask_rgb.shape
    label = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
    for rgb, idx in CLASS_COLORS.items():
        label[np.all(mask_rgb == np.array(rgb, dtype=mask_rgb.dtype), axis=-1)] = idx
    return label


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    """Invert rgb_to_label for visualization."""
    inv = {v: k for k, v in CLASS_COLORS.items()}
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in inv.items():
        out[label == idx] = rgb
    return out


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
