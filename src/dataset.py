"""PyTorch datasets for pre-tiled DeepGlobe imagery."""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TileDataset(Dataset):
    """Reads HxW uint8 label masks and HxWx3 BGR images written by prepare_tiles."""

    def __init__(self, root: str, split: str, transform, ids: Optional[List[str]] = None):
        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        if ids is not None:
            self.files = sorted(ids)
        else:
            self.files = sorted(
                f for f in os.listdir(self.img_dir) if f.endswith(".png")
            )
        self.tf = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        fn = self.files[i]
        img = cv2.imread(os.path.join(self.img_dir, fn), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(os.path.join(self.img_dir, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, fn), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(os.path.join(self.mask_dir, fn))
        aug = self.tf(image=img, mask=mask)
        x = aug["image"]
        y = aug["mask"].long() if isinstance(aug["mask"], torch.Tensor) else torch.from_numpy(aug["mask"]).long()
        return x, y


class FullImageDataset(Dataset):
    """Loads full-resolution 2448x2448 images + masks; used only at test time."""

    def __init__(self, raw_dir: str, image_ids: List[str]):
        self.raw_dir = raw_dir
        self.ids = list(image_ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        iid = self.ids[i]
        img_path = os.path.join(self.raw_dir, f"{iid}_sat.jpg")
        mask_path = os.path.join(self.raw_dir, f"{iid}_mask.png")
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(cv2.imread(mask_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return iid, img, mask_rgb
