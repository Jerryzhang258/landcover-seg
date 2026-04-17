"""Tile and stitch DeepGlobe 2448x2448 imagery."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def tile_image(img: np.ndarray, grid: int = 3, tile_size: int = 816) -> List[np.ndarray]:
    """Split an image into grid*grid non-overlapping tiles (row-major)."""
    h, w = img.shape[:2]
    need = grid * tile_size
    if (h, w) != (need, need):
        raise ValueError(f"expected {need}x{need}, got {h}x{w}")
    tiles: List[np.ndarray] = []
    for i in range(grid):
        for j in range(grid):
            y, x = i * tile_size, j * tile_size
            tiles.append(img[y : y + tile_size, x : x + tile_size].copy())
    return tiles


def stitch_tiles(tiles: List[np.ndarray], grid: int = 3, tile_size: int = 816) -> np.ndarray:
    """Inverse of tile_image for non-overlapping tiles."""
    if len(tiles) != grid * grid:
        raise ValueError(f"expected {grid*grid} tiles, got {len(tiles)}")
    rows = []
    for i in range(grid):
        row = np.concatenate(tiles[i * grid : (i + 1) * grid], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def sliding_window_positions(
    h: int, w: int, tile: int, overlap: int
) -> List[Tuple[int, int]]:
    """Top-left (y, x) coords for a regular grid with edge-snap on boundary."""
    if overlap >= tile:
        raise ValueError("overlap must be < tile")
    stride = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if not ys or ys[-1] != h - tile:
        ys.append(max(h - tile, 0))
    if not xs or xs[-1] != w - tile:
        xs.append(max(w - tile, 0))
    return [(y, x) for y in ys for x in xs]


def stitch_with_overlap(
    logits_tiles: List[np.ndarray],
    positions: List[Tuple[int, int]],
    full_shape: Tuple[int, int],
    num_classes: int,
) -> np.ndarray:
    """Average-blend softmax logits in overlapping regions and argmax."""
    H, W = full_shape
    acc = np.zeros((num_classes, H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    for logits, (y, x) in zip(logits_tiles, positions):
        _, h, w = logits.shape
        acc[:, y : y + h, x : x + w] += logits
        cnt[y : y + h, x : x + w] += 1.0
    acc /= np.maximum(cnt, 1e-6)[None]
    return acc.argmax(axis=0).astype(np.uint8)
