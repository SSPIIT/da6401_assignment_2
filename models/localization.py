"""Localization modules."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based object localizer.

    Outputs [cx, cy, w, h] in pixel space (0-224).
    Final ReLU ensures non-negative outputs.
    Trained with normalised MSE + IoULoss.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            CustomDropout(p=dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.encoder(x))
