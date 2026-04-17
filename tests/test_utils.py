"""RGB <-> label round trip."""
import numpy as np

from src.utils import CLASS_COLORS, label_to_rgb, rgb_to_label


def test_rgb_label_round_trip():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    colors = list(CLASS_COLORS.keys())
    # place each class color in a different region
    for i, c in enumerate(colors[:4]):
        r, g, b = c
        rgb[i, :] = (r, g, b)
    label = rgb_to_label(rgb)
    rgb2 = label_to_rgb(label)
    assert np.array_equal(rgb, rgb2)


def test_unknown_default():
    # An unmapped color should fall back to the Unknown class (index 6)
    rgb = np.full((3, 3, 3), 42, dtype=np.uint8)
    label = rgb_to_label(rgb)
    assert (label == 6).all()
