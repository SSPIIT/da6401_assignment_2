"""Unified multi-task model."""

import os
import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet


def _download_if_missing(path: str, file_id: str):
    """Download from Google Drive if checkpoint not found locally."""
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"Downloading checkpoint to {path} ...")
    try:
        import subprocess
        subprocess.run(["pip", "install", "gdown", "-q"], check=True)
        import gdown
        gdown.download(id=file_id, output=path, quiet=False)
    except Exception as e:
        print(f"gdown failed: {e}. Trying wget fallback...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        os.system(f"wget -q --no-check-certificate '{url}' -O '{path}'")


# Google Drive file IDs — must be shared as "Anyone with the link"
_GDRIVE_IDS = {
    "classifier.pth": "1Uoa_OhEWbWpThKlmtkQ0Jat7WyeGaoXo",
    "localizer.pth":  "1crPXHeDFytXCdch4MptP-WxW3nItYzo9",
    "unet.pth":       "1aadFoUhUvNmUpNsFMMDm49ByycF8wVH_",
}


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

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

        # Auto-download any missing checkpoints
        for path, key in [
            (classifier_path, "classifier.pth"),
            (localizer_path,  "localizer.pth"),
            (unet_path,       "unet.pth"),
        ]:
            _download_if_missing(path, _GDRIVE_IDS[key])

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

        # Shared backbone from classifier
        self.backbone = classifier.encoder

        # Task heads
        self.cls_head = classifier.classifier
        self.loc_head = localizer.regressor

        # Segmentation UNet with shared backbone
        self.segmenter     = unet
        self.segmenter.enc = self.backbone

    def forward(self, x: torch.Tensor):
        """Single forward pass.
        Args:
            x: [B, 3, 224, 224] normalised input.
        Returns:
            dict with keys: classification, localization, segmentation.
        """
        bottleneck = self.backbone(x)           # [B, 512, 7, 7]

        cls_out = self.cls_head(bottleneck)     # [B, num_breeds]
        loc_out = self.loc_head(bottleneck)     # [B, 4]
        seg_out = self.segmenter(x)             # [B, seg_classes, H, W]

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }