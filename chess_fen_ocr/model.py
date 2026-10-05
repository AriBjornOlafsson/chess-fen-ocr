"""Small CNN for classifying a single chessboard square (64x64 grayscale+edge
-> 13 classes). See preprocess.py for the channel definitions; color is
deliberately not fed in, so classification is invariant to board theme/hue."""
from __future__ import annotations

import torch
from torch import nn

from .labels import NUM_CLASSES

IN_CHANNELS = 2


class SquareNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
