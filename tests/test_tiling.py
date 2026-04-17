"""Round-trip and sliding-window sanity tests."""
import numpy as np
import pytest

from src.tiling import (sliding_window_positions, stitch_tiles,
                        stitch_with_overlap, tile_image)


def test_3x3_round_trip_image():
    img = np.random.randint(0, 255, (2448, 2448, 3), dtype=np.uint8)
    tiles = tile_image(img, grid=3, tile_size=816)
    assert len(tiles) == 9
    for t in tiles:
        assert t.shape == (816, 816, 3)
    recon = stitch_tiles(tiles, grid=3, tile_size=816)
    assert np.array_equal(img, recon)


def test_3x3_round_trip_mask():
    mask = np.random.randint(0, 7, (2448, 2448), dtype=np.uint8)
    tiles = tile_image(mask, grid=3, tile_size=816)
    recon = stitch_tiles(tiles, grid=3, tile_size=816)
    assert np.array_equal(mask, recon)


def test_4x4_round_trip():
    img = np.random.randint(0, 255, (2448, 2448, 3), dtype=np.uint8)
    tiles = tile_image(img, grid=4, tile_size=612)
    assert len(tiles) == 16
    recon = stitch_tiles(tiles, grid=4, tile_size=612)
    assert np.array_equal(img, recon)


def test_tile_image_wrong_size_raises():
    bad = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        tile_image(bad, grid=3, tile_size=816)


def test_sliding_window_covers_image():
    positions = sliding_window_positions(2448, 2448, tile=816, overlap=64)
    # every pixel must be covered by at least one window
    cov = np.zeros((2448, 2448), dtype=bool)
    for y, x in positions:
        cov[y:y + 816, x:x + 816] = True
    assert cov.all()


def test_sliding_window_edge_snap():
    # when (H - tile) % stride != 0, an extra edge-snap window must be added
    positions = sliding_window_positions(2448, 2448, tile=816, overlap=100)
    # final window must end exactly at H
    ys = sorted({y for y, _ in positions})
    assert ys[-1] + 816 == 2448


def test_stitch_with_overlap_matches_argmax():
    # construct 2 tiles with overlap so we can check averaging works
    H, W, C, tile = 16, 32, 3, 16
    logits = np.zeros((C, H, W), dtype=np.float32)
    logits[0, :, :16] = 1.0  # left half class 0
    logits[1, :, 16:] = 1.0  # right half class 1
    t_left = logits[:, :, :tile]
    t_right = logits[:, :, 16:]
    stitched = stitch_with_overlap(
        [t_left, t_right], [(0, 0), (0, 16)], (H, W), num_classes=C
    )
    assert stitched[:, :16].max() == 0
    assert stitched[:, 16:].min() == 1
