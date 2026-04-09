"""Segmentation model — VGG11 encoder + symmetric U-Net decoder."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _dec_block(in_c: int, out_c: int) -> nn.Sequential:
    """Conv → BN → ReLU block used in decoder."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style segmentation network built on VGG11 encoder.

    Decoder mirrors the encoder: each stage uses a ConvTranspose2d to double
    spatial resolution, then concatenates the corresponding encoder skip
    connection, then applies conv blocks.

    Upsampling strategy: ConvTranspose2d with stride=2 (learnable parameters).
    Standard bilinear/nearest interpolation is NOT used as per requirements.

    Loss: CrossEntropyLoss is appropriate for 3-class pixel-wise classification
    (foreground / background / uncertain). It handles class imbalance well when
    combined with class weights derived from the dataset pixel distribution.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.dropout = CustomDropout(p=dropout_p)

        # ── Decoder ──────────────────────────────────────────────────────────
        # After each ConvTranspose2d the skip connection is concatenated,
        # so in_channels for the following conv = upsample_out + skip_channels.
        #
        # Encoder feature sizes (channels, spatial):
        #   block5: 512,  7×7   (bottleneck from encoder.forward)
        #   block4: 512, 14×14  skip
        #   block3: 256, 28×28  skip
        #   block2: 128, 56×56  skip
        #   block1:  64,112×112 skip

        # Dec5: 512→512, upsample 7→14
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)  # concat with block4 skip (512)

        # Dec4: 512→256, upsample 14→28
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _dec_block(256 + 256, 256)  # concat with block3 skip (256)

        # Dec3: 256→128, upsample 28→56
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _dec_block(128 + 128, 128)  # concat with block2 skip (128)

        # Dec2: 128→64, upsample 56→112
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _dec_block(64 + 64, 64)     # concat with block1 skip (64)

        # Dec1: 64→64, upsample 112→224
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = _dec_block(64, 64)

        # Final 1×1 conv → num_classes
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder — collect skip connections
        bottleneck, skips = self.encoder(x, return_features=True)
        # skips: block1(64,112), block2(128,56), block3(256,28), block4(512,14), block5(512,7)

        # Apply dropout to bottleneck
        d = self.dropout(bottleneck)

        # Decode with skip connections
        d = self.up5(d)
        d = torch.cat([d, skips["block4"]], dim=1)
        d = self.dec5(d)

        d = self.up4(d)
        d = torch.cat([d, skips["block3"]], dim=1)
        d = self.dec4(d)

        d = self.up3(d)
        d = torch.cat([d, skips["block2"]], dim=1)
        d = self.dec3(d)

        d = self.up2(d)
        d = torch.cat([d, skips["block1"]], dim=1)
        d = self.dec2(d)

        d = self.up1(d)
        d = self.dec1(d)

        return self.head(d)