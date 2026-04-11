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

    Loads classifier.pth, localizer.pth, unet.pth and assembles a single
    model with one shared VGG11 backbone and three task heads.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate all three task models
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer  = VGG11Localizer(in_channels=in_channels)
        unet       = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        def _load(model, path):
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=device)
                sd = ckpt.get("state_dict", ckpt)
                model.load_state_dict(sd, strict=False)
                print(f"Loaded: {path}")
            else:
                print(f"WARNING: checkpoint not found: {path}")
            return model

        classifier = _load(classifier, classifier_path)
        localizer  = _load(localizer,  localizer_path)
        unet       = _load(unet,       unet_path)

        # Shared backbone — from classifier
        self.backbone = classifier.encoder

        # Classification head
        self.cls_head = classifier.classifier

        # Localisation head
        self.loc_head = localizer.regressor

        # Segmentation — full UNet, share backbone
        self.segmenter     = unet
        self.segmenter.enc = self.backbone   # weight sharing

    def forward(self, x: torch.Tensor):
        """Single forward pass → classification + localisation + segmentation.

        Args:
            x: [B, 3, 224, 224] normalised input tensor.
        Returns:
            dict with keys:
                'classification': [B, num_breeds]
                'localization':   [B, 4]  (cx, cy, w, h) in pixel coords
                'segmentation':   [B, seg_classes, H, W]
        """
        # Shared backbone (no skip connections needed for cls/loc)
        bottleneck = self.backbone(x)           # [B, 512, 7, 7]

        # Classification
        cls_out = self.cls_head(bottleneck)     # [B, num_breeds]

        # Localisation — raw ReLU output in pixel coords
        loc_out = self.loc_head(bottleneck)     # [B, 4]

        # Segmentation — UNet handles its own encoder pass with skips
        seg_out = self.segmenter(x)             # [B, seg_classes, H, W]

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }
