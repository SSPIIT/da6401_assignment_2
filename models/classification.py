"""Classification components."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3,
                 dropout_p: float = 0.5, pretrained: bool = False):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            CustomDropout(p=dropout_p),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
