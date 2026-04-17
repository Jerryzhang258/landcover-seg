"""Model factory. Keep API stable so train.py only needs a name string."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .utils import NUM_CLASSES


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class VanillaUNet(nn.Module):
    """Ronneberger-style U-Net (5 levels) with BN, random init.

    Use this as a true "from-scratch" baseline when you want to avoid the
    ResNet34 encoder's inductive biases.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, base: int = 64):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.u4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))
        u4 = self.u4(torch.cat([self.up4(b), c4], dim=1))
        u3 = self.u3(torch.cat([self.up3(u4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return self.head(u1)


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    name = name.lower()
    if name == "vanilla_unet":
        return VanillaUNet(num_classes=num_classes)
    if name == "unet_scratch":
        return smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=num_classes,
        )
    if name == "unet_resnet34":
        return smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=num_classes,
        )
    if name == "deeplab_r50":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights="imagenet",
            in_channels=3, classes=num_classes,
        )
    if name == "deeplab_r101":
        return smp.DeepLabV3Plus(
            encoder_name="resnet101", encoder_weights="imagenet",
            in_channels=3, classes=num_classes,
        )
    if name == "attn_unet":
        # SCSE attention in the decoder; closest drop-in to Attention U-Net in smp.
        return smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            decoder_attention_type="scse",
            in_channels=3, classes=num_classes,
        )
    raise ValueError(f"unknown model name: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
