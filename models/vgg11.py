"""VGG11 encoder — implemented from scratch per https://arxiv.org/abs/1409.1556.

Architecture (conv layers only, BN added after each conv):
  Block 1: Conv(3→64)   + BN + ReLU → MaxPool  → 112×112
  Block 2: Conv(64→128) + BN + ReLU → MaxPool  →  56×56
  Block 3: Conv(128→256)+ BN + ReLU
           Conv(256→256)+ BN + ReLU → MaxPool  →  28×28
  Block 4: Conv(256→512)+ BN + ReLU
           Conv(512→512)+ BN + ReLU → MaxPool  →  14×14
  Block 5: Conv(512→512)+ BN + ReLU
           Conv(512→512)+ BN + ReLU → MaxPool  →   7×7
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


def _conv_bn_relu(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.

    BatchNorm is inserted after every Conv layer (before ReLU) — this stabilises
    training and allows higher learning rates without sacrificing accuracy.
    No Dropout in the convolutional backbone; Dropout belongs in the dense head
    where it acts as a regulariser on the high-dimensional feature vector.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # Block 1: 224→112
        self.block1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2: 112→56
        self.block2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3: 56→28
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 4: 28→14
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 5: 14→7
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, 224, 224].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor [B, 512, 7, 7].
            - if return_features=True: (bottleneck, feature_dict) where
              feature_dict keys are 'block1'..'block5' before each MaxPool.
        """
        features: Dict[str, torch.Tensor] = {}

        if return_features:
            # Run each sub-layer separately to capture pre-pool feature maps
            # Block 1
            x1 = list(self.block1.children())[0](x)   # conv+bn+relu
            features["block1"] = x1
            x = list(self.block1.children())[1](x1)   # maxpool

            # Block 2
            x2 = list(self.block2.children())[0](x)
            features["block2"] = x2
            x = list(self.block2.children())[1](x2)

            # Block 3
            x3 = self.block3[0](x)
            x3 = self.block3[1](x3)
            features["block3"] = x3
            x = self.block3[2](x3)

            # Block 4
            x4 = self.block4[0](x)
            x4 = self.block4[1](x4)
            features["block4"] = x4
            x = self.block4[2](x4)

            # Block 5
            x5 = self.block5[0](x)
            x5 = self.block5[1](x5)
            features["block5"] = x5
            x = self.block5[2](x5)

            return x, features
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            return x


# Alias expected by autograder: `from models.vgg11 import VGG11`
VGG11 = VGG11Encoder