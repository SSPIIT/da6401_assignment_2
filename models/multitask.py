"""Unified multi-task model."""

import os
import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Loads pre-trained weights from classifier.pth, localizer.pth, unet.pth and
    assembles a single model with one shared VGG11 backbone and three task heads:
      - ClassificationHead  → breed logits      [B, num_breeds]
      - LocalizationHead    → bbox coordinates  [B, 4]
      - SegmentationDecoder → pixel mask logits [B, seg_classes, H, W]
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        import gdown

        # 🔽 Download classifier weights
        gdown.download(
            id="1DMNiCz-bn6_DSGcSeyESmMuIS0cWZWSS",
            output=classifier_path,
            quiet=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        Initialize the shared backbone/heads using trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Relative path to trained classifier weights.
            localizer_path: Relative path to trained localizer weights.
            unet_path: Relative path to trained unet weights.
        """
        # super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load full task models to extract weights ─────────────────────────
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        def _load(model, path):
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=device)
                sd = ckpt.get("state_dict", ckpt)
                model.load_state_dict(sd, strict=False)
            return model

        classifier = _load(classifier, classifier_path)
        localizer  = _load(localizer,  localizer_path)
        unet       = _load(unet,       unet_path)

        # ── Shared backbone — initialise from classifier checkpoint ──────────
        self.backbone = VGG11Encoder(in_channels=in_channels)
        self.backbone.load_state_dict(classifier.encoder.state_dict())

        # ── Classification head ───────────────────────────────────────────────
        self.cls_head = classifier.classifier

        # ── Localisation head ─────────────────────────────────────────────────
        self.loc_head = localizer.regressor

        # ── Segmentation decoder ──────────────────────────────────────────────
        # self.seg_up5  = unet.up5
        # self.seg_dec5 = unet.dec5
        # self.seg_up4  = unet.up4
        # self.seg_dec4 = unet.dec4
        # self.seg_up3  = unet.up3
        # self.seg_dec3 = unet.dec3
        # self.seg_up2  = unet.up2
        # self.seg_dec2 = unet.dec2
        # self.seg_up1  = unet.up1
        # self.seg_dec1 = unet.dec1
        # self.seg_head = unet.head
        # self.seg_drop = unet.dropout
        self.segmenter = unet
        # Store input size for output scaling in localizer
        self._in_channels = in_channels

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization':   [B, 4] bounding box tensor.
            - 'segmentation':   [B, seg_classes, H, W] segmentation logits tensor.
        """
        h, w = x.shape[2], x.shape[3]

        # Single shared backbone pass with skip connections
        bottleneck, skips = self.backbone(x, return_features=True)

        # ── Classification ────────────────────────────────────────────────────
        cls_out = self.cls_head(bottleneck)

        # ── Localisation ──────────────────────────────────────────────────────
        raw = self.loc_head(bottleneck)
        cx = torch.sigmoid(raw[:, 0:1]) * w
        cy = torch.sigmoid(raw[:, 1:2]) * h
        bw = torch.nn.functional.softplus(raw[:, 2:3]) * w
        bh = torch.nn.functional.softplus(raw[:, 3:4]) * h
        loc_out = torch.cat([cx, cy, bw, bh], dim=1)

        # ── Segmentation ──────────────────────────────────────────────────────
        # d = self.seg_drop(bottleneck)
        # d = self.seg_up5(d);  d = torch.cat([d, skips["block4"]], dim=1); d = self.seg_dec5(d)
        # d = self.seg_up4(d);  d = torch.cat([d, skips["block3"]], dim=1); d = self.seg_dec4(d)
        # d = self.seg_up3(d);  d = torch.cat([d, skips["block2"]], dim=1); d = self.seg_dec3(d)
        # d = self.seg_up2(d);  d = torch.cat([d, skips["block1"]], dim=1); d = self.seg_dec2(d)
        # d = self.seg_up1(d);  d = self.seg_dec1(d)
        # seg_out = self.seg_head(d)
        seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }