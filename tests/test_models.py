"""Forward-pass shape check on CPU with tiny input so it runs in CI."""
import torch

from src.models import build_model


def test_all_models_forward():
    names = ["vanilla_unet", "unet_scratch", "unet_resnet34",
             "deeplab_r50", "attn_unet"]
    x = torch.randn(1, 3, 128, 128)
    for n in names:
        m = build_model(n, num_classes=7).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape[0] == 1 and y.shape[1] == 7
        assert y.shape[-2:] == x.shape[-2:]
