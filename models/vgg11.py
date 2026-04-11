"""VGG11 encoder — implemented from scratch per https://arxiv.org/abs/1409.1556.

Architecture (with BatchNorm after every Conv):
  Block 1: Conv(3->64)    BN ReLU MaxPool  224->112
  Block 2: Conv(64->128)  BN ReLU MaxPool  112->56
  Block 3: Conv(128->256) BN ReLU Conv(256->256) BN ReLU MaxPool  56->28
  Block 4: Conv(256->512) BN ReLU Conv(512->512) BN ReLU MaxPool  28->14
  Block 5: Conv(512->512) BN ReLU Conv(512->512) BN ReLU MaxPool  14->7
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
    """VGG11-style encoder with optional pretrained ImageNet weights."""

    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()

        self.block1 = nn.Sequential(_conv_bn_relu(in_channels, 64),  nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(_conv_bn_relu(64, 128),           nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(_conv_bn_relu(128, 256), _conv_bn_relu(256, 256), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(_conv_bn_relu(256, 512), _conv_bn_relu(512, 512), nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(_conv_bn_relu(512, 512), _conv_bn_relu(512, 512), nn.MaxPool2d(2, 2))

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        """Copy weights from torchvision VGG11_BN into our blocks."""
        from torchvision.models import vgg11_bn, VGG11_BN_Weights
        tv = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
        f  = list(tv.features.children())

        def _copy(our_cbr, tv_conv, tv_bn):
            our_cbr[0].weight.data.copy_(tv_conv.weight.data)
            our_cbr[1].weight.data.copy_(tv_bn.weight.data)
            our_cbr[1].bias.data.copy_(tv_bn.bias.data)
            our_cbr[1].running_mean.copy_(tv_bn.running_mean)
            our_cbr[1].running_var.copy_(tv_bn.running_var)

        # torchvision VGG11_BN feature layer indices:
        # 0=Conv 1=BN 2=ReLU 3=Pool | 4=Conv 5=BN 6=ReLU 7=Pool |
        # 8=Conv 9=BN 10=ReLU 11=Conv 12=BN 13=ReLU 14=Pool |
        # 15=Conv 16=BN 17=ReLU 18=Conv 19=BN 20=ReLU 21=Pool |
        # 22=Conv 23=BN 24=ReLU 25=Conv 26=BN 27=ReLU 28=Pool
        _copy(self.block1[0], f[0],  f[1])
        _copy(self.block2[0], f[4],  f[5])
        _copy(self.block3[0], f[8],  f[9])
        _copy(self.block3[1], f[11], f[12])
        _copy(self.block4[0], f[15], f[16])
        _copy(self.block4[1], f[18], f[19])
        _copy(self.block5[0], f[22], f[23])
        _copy(self.block5[1], f[25], f[26])
        print("Loaded ImageNet pretrained weights into VGG11Encoder.")

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: [B, 3, 224, 224]
            return_features: if True return (bottleneck, skip_dict).
                             Skips are POST-pool tensors.
        """
        if return_features:
            # Post-pool skips for U-Net
            f1    = self.block1[0](x);     skip1 = self.block1[1](f1)
            f2    = self.block2[0](skip1); skip2 = self.block2[1](f2)
            t3    = self.block3[0](skip2); f3 = self.block3[1](t3); skip3 = self.block3[2](f3)
            t4    = self.block4[0](skip3); f4 = self.block4[1](t4); skip4 = self.block4[2](f4)
            t5    = self.block5[0](skip4); f5 = self.block5[1](t5); neck  = self.block5[2](f5)
            feats = {"block1": skip1, "block2": skip2, "block3": skip3,
                     "block4": skip4, "block5": neck}
            return neck, feats
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            return x


# Alias required by autograder: `from models.vgg11 import VGG11`
VGG11 = VGG11Encoder
