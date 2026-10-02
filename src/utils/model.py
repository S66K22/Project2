import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
    ):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)
        return x


class SeparableConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv(x)

        x = x + identity

        x = self.relu(x)

        return x


class SmallXception1(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(
                3,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            # Block 1
            XceptionBlock(
                32,
                64,
                stride=1,
            ),
            # Block 2
            XceptionBlock(
                64,
                128,
                stride=2,
            ),
            # Block 3
            XceptionBlock(
                128,
                256,
                stride=2,
            ),
            # Block 4
            XceptionBlock(
                256,
                512,
                stride=2,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512, num_classes)
        )

    def forward(self, x, return_features=False):
        x = self.features(x)

        x = self.pool(x)

        feat = torch.flatten(x, 1)

        logits = self.classifier(feat)

        if return_features:
            return logits, feat
        return logits


class SmallXception2(nn.Module):
    def __init__(self, num_classes, dropout=0.3):

        super().__init__()

        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Block 1
            XceptionBlock(16, 32, stride=1),
            # Block 2
            XceptionBlock(32, 64, stride=2),
            # Block 3
            XceptionBlock(64, 128, stride=2),
            # Block 4
            XceptionBlock(128, 256, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.pool(x)
        feat = torch.flatten(x, 1)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape

        # Squeeze: [B, C, H, W] -> [B, C]
        y = self.squeeze(x).view(b, c)

        # Excitation: [B, C] -> [B, C]
        y = self.excitation(y)

        # Reshape: [B, C] -> [B, C, 1, 1]
        y = y.view(b, c, 1, 1)

        # Channel-wise recalibration
        return x * y


class SmallXception3(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(
                3,
                32,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            # Block 1
            XceptionBlock(
                32,
                64,
                stride=1,
            ),
            SEBlock(64, reduction=16),
            # Block 2
            XceptionBlock(
                64,
                128,
                stride=2,
            ),
            SEBlock(128, reduction=16),
            # Block 3
            XceptionBlock(
                128,
                256,
                stride=2,
            ),
            SEBlock(256, reduction=16),
            # Block 4
            XceptionBlock(
                256,
                512,
                stride=2,
            ),
            SEBlock(512, reduction=16),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, return_features=False):

        x = self.features(x)

        x = self.pool(x)

        feat = torch.flatten(x, 1)

        logits = self.classifier(feat)

        if return_features:
            return logits, feat
        return logits


class VGGOpenSet(nn.Module):
    """
    Wraps torchvision VGG16. forward(x, return_features=True) returns both
    the classification logits and the 4096-d embedding feeding the final
    Linear layer — that embedding is what the distance-based detector uses.
    """

    def __init__(self, num_classes):
        super().__init__()
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier_trunk = backbone.classifier[:-1]  # everything but last Linear
        in_features = backbone.classifier[-1].in_features
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.classifier_trunk(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits


def create_model(model_name, num_classes, dropout=0.2):
    if model_name == "small-xception1":
        return SmallXception1(num_classes, dropout)
    if model_name == "small-xception2":
        return SmallXception2(num_classes, dropout)
    if model_name == "small-xception3":
        return SmallXception2(num_classes, dropout)
    if model_name == "vgg":
        return VGGOpenSet(num_classes)


def log_number_of_params(model):
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
