"""Classification components."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead.

    Head design follows the original VGG paper:
      AdaptiveAvgPool → Flatten → FC(25088→4096) → BN → CustomDropout → ReLU
                                → FC(4096→4096)  → BN → CustomDropout → ReLU
                                → FC(4096→num_classes)

    BatchNorm is placed BEFORE Dropout in the dense head because:
    - BN normalises the pre-activation distribution, preventing Dropout from
      inadvertently skewing the scale of surviving neurons.
    - Empirically this order (BN → Dropout → ReLU) shows faster convergence
      and better final accuracy than placing BN after activation.
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # After block5 + AdaptiveAvgPool(7,7): 512 * 7 * 7 = 25088
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
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x)
        return self.classifier(features)