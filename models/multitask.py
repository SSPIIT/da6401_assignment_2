"""Unified multi-task model."""

import os
import torch
import torch.nn as nn

from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds=37, seg_classes=3, in_channels=3,
                 classifier_path="classifier.pth",
                 localizer_path="localizer.pth",
                 unet_path="unet.pth"):
        super().__init__()

        import gdown

        # 🔽 DOWNLOAD WEIGHTS (add IDs later for all 3)
        gdown.download(id="1DMNiCz-bn6_DSGcSeyESmMuIS0cWZWSS", output=classifier_path, quiet=False)
        # gdown.download(id="LOCALIZER_ID", output=localizer_path)
        # gdown.download(id="UNET_ID", output=unet_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        localizer  = VGG11Localizer(in_channels=in_channels)
        unet       = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        def _load(model, path):
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=device)
                sd = ckpt.get("state_dict", ckpt)
                model.load_state_dict(sd, strict=False)
            return model

        classifier = _load(classifier, classifier_path)
        localizer  = _load(localizer,  localizer_path)
        unet       = _load(unet,       unet_path)

        # Shared backbone
        self.backbone = classifier.encoder

        # Heads
        self.cls_head = classifier.classifier
        self.loc_head = localizer.regressor

        # Segmentation (full UNet)
        self.segmenter = unet

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # Backbone forward
        bottleneck = self.backbone(x)

        # Classification
        cls_out = self.cls_head(bottleneck)

        # Localization
        raw = self.loc_head(bottleneck)
        cx = torch.sigmoid(raw[:, 0:1]) * w
        cy = torch.sigmoid(raw[:, 1:2]) * h
        bw = torch.nn.functional.softplus(raw[:, 2:3]) * w
        bh = torch.nn.functional.softplus(raw[:, 3:4]) * h
        loc_out = torch.cat([cx, cy, bw, bh], dim=1)

        # Segmentation
        seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }