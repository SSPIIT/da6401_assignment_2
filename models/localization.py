"""Localization modules."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based object localizer.

    Outputs bounding box coordinates [x_center, y_center, width, height]
    in original pixel space (not normalised values).

    The regression head uses ReLU on width/height outputs to ensure they are
    non-negative, and a separate branch for x_center / y_center that can take
    any value within the image. Trained with MSE + IoULoss.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Regression head: produces 4 values — cx, cy, w, h
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space (not normalised).
        """
        h, w = x.shape[2], x.shape[3]
        features = self.encoder(x)
        raw = self.regressor(features)   # [B, 4]

        # Sigmoid the cx,cy to stay within image, scale to pixel space
        # Apply softplus to w,h to ensure positive (smooth alternative to ReLU)
        cx = torch.sigmoid(raw[:, 0:1]) * w
        cy = torch.sigmoid(raw[:, 1:2]) * h
        bw = torch.nn.functional.softplus(raw[:, 2:3]) * w
        bh = torch.nn.functional.softplus(raw[:, 3:4]) * h

        return torch.cat([cx, cy, bw, bh], dim=1)