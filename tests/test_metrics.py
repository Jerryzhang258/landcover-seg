import numpy as np

from src.metrics import ConfusionMatrix


def test_perfect_prediction():
    cm = ConfusionMatrix(num_classes=7, ignore_index=6)
    y = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    cm.update(y.copy(), y.copy())
    _, miou = cm.iou()
    assert miou == 1.0
    assert cm.pixel_acc() == 1.0


def test_ignore_index_excluded():
    cm = ConfusionMatrix(num_classes=7, ignore_index=6)
    pred = np.array([[0, 0, 6]], dtype=np.uint8)
    tgt = np.array([[0, 0, 6]], dtype=np.uint8)
    cm.update(pred, tgt)
    # only two pixels counted (the two class-0 ones); class 6 excluded
    assert cm.mat.sum() == 2


def test_half_wrong():
    cm = ConfusionMatrix(num_classes=7, ignore_index=6)
    pred = np.array([0, 0, 0, 0], dtype=np.uint8)
    tgt = np.array([0, 0, 1, 1], dtype=np.uint8)
    cm.update(pred, tgt)
    iou, _ = cm.iou()
    # class 0: TP=2, FP=2, FN=0 -> IoU = 0.5
    assert abs(iou[0] - 0.5) < 1e-6
    # class 1: TP=0, FP=0, FN=2 -> IoU = 0.0
    assert iou[1] == 0.0
