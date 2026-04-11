"""Segmentation model — VGG11 encoder + symmetric U-Net decoder."""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _dec_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=3, dropout_p=0.5):
        super().__init__()

        self.enc = VGG11Encoder(in_channels=in_channels)
        self.dropout = CustomDropout(p=dropout_p)

        self.up5  = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec5 = _dec_block(1024, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = _dec_block(512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = _dec_block(256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = _dec_block(128, 64)

        self.up1  = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = _dec_block(64, 64)

        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        f1 = self.enc.block1[0](x)
        x  = self.enc.block1[1](f1)
        skip1 = x

        f2 = self.enc.block2[0](x)
        x  = self.enc.block2[1](f2)
        skip2 = x

        x  = self.enc.block3[0](x)
        f3 = self.enc.block3[1](x)
        x  = self.enc.block3[2](f3)
        skip3 = x

        x  = self.enc.block4[0](x)
        f4 = self.enc.block4[1](x)
        x  = self.enc.block4[2](f4)
        skip4 = x

        x  = self.enc.block5[0](x)
        x  = self.enc.block5[1](x)
        x  = self.enc.block5[2](x)

        x = self.dropout(x)

        # Decoder
        x = self.up5(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        return self.head(x)