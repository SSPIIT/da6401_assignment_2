"""Segmentation model — VGG11 encoder + symmetric U-Net decoder."""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _dec_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style segmentation network built on VGG11 encoder.

    Skip connections are taken from POST-pool outputs so spatial sizes
    match correctly after each ConvTranspose2d upsampling step:

        bottleneck : 512 @  7x7
        skip4      : 512 @ 14x14  (output of block4 after maxpool)
        skip3      : 256 @ 28x28  (output of block3 after maxpool)
        skip2      : 128 @ 56x56  (output of block2 after maxpool)
        skip1      :  64 @112x112  (output of block1 after maxpool)

    Decoder:
        up5: 7->14   cat skip4(512) -> dec5 -> 512
        up4: 14->28  cat skip3(256) -> dec4 -> 256
        up3: 28->56  cat skip2(128) -> dec3 -> 128
        up2: 56->112 cat skip1(64)  -> dec2 -> 64
        up1: 112->224               -> dec1 -> 64 -> head
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.enc     = VGG11Encoder(in_channels=in_channels)
        self.dropout = CustomDropout(p=dropout_p)

        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _dec_block(256 + 256, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _dec_block(128 + 128, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _dec_block(64 + 64, 64)

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = _dec_block(64, 64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder — collect POST-pool skip connections ──────────────────
        # block1[0]=conv_bn_relu, block1[1]=maxpool
        # input 224x224 -> conv -> 224x224 -> pool -> 112x112
        f1   = self.enc.block1[0](x)        # [B, 64,224,224]
        skip1 = self.enc.block1[1](f1)      # [B, 64,112,112]  post-pool

        f2   = self.enc.block2[0](skip1)    # [B,128,112,112]
        skip2 = self.enc.block2[1](f2)      # [B,128, 56, 56]  post-pool

        t3   = self.enc.block3[0](skip2)    # [B,256, 56, 56]
        f3   = self.enc.block3[1](t3)       # [B,256, 56, 56]
        skip3 = self.enc.block3[2](f3)      # [B,256, 28, 28]  post-pool

        t4   = self.enc.block4[0](skip3)    # [B,512, 28, 28]
        f4   = self.enc.block4[1](t4)       # [B,512, 28, 28]
        skip4 = self.enc.block4[2](f4)      # [B,512, 14, 14]  post-pool

        t5   = self.enc.block5[0](skip4)    # [B,512, 14, 14]
        f5   = self.enc.block5[1](t5)       # [B,512, 14, 14]
        neck  = self.enc.block5[2](f5)      # [B,512,  7,  7]  bottleneck

        # ── Dropout on bottleneck ─────────────────────────────────────────
        d = self.dropout(neck)

        # ── Decoder ──────────────────────────────────────────────────────
        d = self.up5(d)                          # [B,512,14,14]
        d = torch.cat([d, skip4], dim=1)         # [B,1024,14,14]
        d = self.dec5(d)                         # [B,512,14,14]

        d = self.up4(d)                          # [B,256,28,28]
        d = torch.cat([d, skip3], dim=1)         # [B,512,28,28]
        d = self.dec4(d)                         # [B,256,28,28]

        d = self.up3(d)                          # [B,128,56,56]
        d = torch.cat([d, skip2], dim=1)         # [B,256,56,56]
        d = self.dec3(d)                         # [B,128,56,56]

        d = self.up2(d)                          # [B,64,112,112]
        d = torch.cat([d, skip1], dim=1)         # [B,128,112,112]
        d = self.dec2(d)                         # [B,64,112,112]

        d = self.up1(d)                          # [B,64,224,224]
        d = self.dec1(d)                         # [B,64,224,224]

        return self.head(d)                      # [B,num_classes,224,224]
