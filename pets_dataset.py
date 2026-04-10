"""Dataset loader for Oxford-IIIT Pet dataset."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalisation stats (used because VGG11 was originally trained on ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# VGG11 fixed input size
IMAGE_SIZE = 224


def get_transforms(split: str = "train") -> A.Compose:
    """Return albumentations transform pipeline for a given split.

    Args:
        split: 'train' or 'val'/'test'.
    Returns:
        Albumentations Compose with bbox_params for bounding-box handling.
    """
    bbox_params = A.BboxParams(
        format="coco",          # [x_min, y_min, w, h] — converted to cx,cy,w,h after
        label_fields=["labels"],
        min_visibility=0.1,
    )
    if split == "train":
        return A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )
    else:
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Directory layout expected:
        root/
          images/      *.jpg
          annotations/
            xmls/      *.xml      (bounding boxes)
            trimaps/   *.png      (segmentation masks: 1=fg, 2=bg, 3=uncertain)
            list.txt             (split list)
            trainval.txt
            test.txt

    Each __getitem__ returns a dict with:
        image   : FloatTensor [3, 224, 224]  (normalised)
        label   : int                        (breed index 0-36)
        bbox    : FloatTensor [4]            (cx, cy, w, h in pixel coords)
        mask    : LongTensor  [224, 224]     (0=bg, 1=fg, 2=uncertain)
    """

    # 37 breed class names in alphabetical order (matches list.txt CLASS_ID 1-37)
    # Replace the BREEDS list and the class_id line in __init__

# DELETE the old BREEDS list entirely and replace with this:
    BREEDS = [
        "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
        "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer",
        "British_Shorthair", "chihuahua", "Egyptian_Mau", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
        "japanese_chin", "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher",
        "newfoundland", "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
        "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu", "Siamese",
        "Sphynx", "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier",
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
    ):
        """
        Args:
            root:      Path to dataset root (contains images/ and annotations/).
            split:     'train' or 'test'.
            transform: Optional albumentations Compose. If None, default
                       transforms for the split are applied.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform if transform is not None else get_transforms(split)

        # ── Parse split file ──────────────────────────────────────────────────
        split_file = "trainval.txt" if split == "train" else "test.txt"
        list_path = self.root / "annotations" / split_file

        self.samples = []   # list of (image_stem, class_idx)
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                stem      = parts[0]                # e.g. "Abyssinian_1"
                class_id  = int(parts[1]) - 1       # 1-indexed → 0-indexed
                self.samples.append((stem, class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_bbox(self, stem: str, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Load bounding box from XML annotation.
        Returns (x_min, y_min, width, height) in original pixel coords.
        Falls back to whole-image box if XML not found.
        """
        xml_path = self.root / "annotations" / "xmls" / f"{stem}.xml"
        if not xml_path.exists():
            return (0.0, 0.0, float(img_w), float(img_h))
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find(".//bndbox")
            xmin = float(obj.find("xmin").text)
            ymin = float(obj.find("ymin").text)
            xmax = float(obj.find("xmax").text)
            ymax = float(obj.find("ymax").text)
            return (xmin, ymin, xmax - xmin, ymax - ymin)
        except Exception:
            return (0.0, 0.0, float(img_w), float(img_h))

    def _load_mask(self, stem: str) -> Optional[np.ndarray]:
        """Load trimap segmentation mask.
        Returns HxW uint8 array with values 1(fg)/2(bg)/3(uncertain), or None.
        """
        mask_path = self.root / "annotations" / "trimaps" / f"{stem}.png"
        if not mask_path.exists():
            return None
        return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict:
        stem, class_idx = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img_path = self.root / "images" / f"{stem}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        img_h, img_w = image.shape[:2]

        # ── Load bbox (x_min, y_min, w, h) ───────────────────────────────────
        xmin, ymin, bw, bh = self._load_bbox(stem, img_w, img_h)
        # clamp to image bounds
        xmin = max(0.0, min(xmin, img_w - 1))
        ymin = max(0.0, min(ymin, img_h - 1))
        bw   = max(1.0, min(bw,   img_w - xmin))
        bh   = max(1.0, min(bh,   img_h - ymin))

        # ── Load mask ─────────────────────────────────────────────────────────
        raw_mask = self._load_mask(stem)
        if raw_mask is None:
            raw_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # ── Apply transforms ──────────────────────────────────────────────────
        transformed = self.transform(
            image=image,
            mask=raw_mask,
            bboxes=[[xmin, ymin, bw, bh]],
            labels=[class_idx],
        )

        image_t = transformed["image"].float()          # [3, 224, 224]
        mask_t  = transformed["mask"].long()            # [224, 224]

        # Remap mask values: trimap uses 1=fg, 2=bg, 3=uncertain → 0-based
        # 1→1(fg), 2→0(bg), 3→2(uncertain)  so class 0 = background (majority)
        mask_t = mask_t.clone()
        remap = {1: 1, 2: 0, 3: 2}
        remapped = torch.zeros_like(mask_t)
        for src, dst in remap.items():
            remapped[mask_t == src] = dst
        mask_t = remapped

        # ── Convert bbox back to cx,cy,w,h in post-resize pixel coords ────────
        if len(transformed["bboxes"]) > 0:
            bx, by, bw2, bh2 = transformed["bboxes"][0]
            cx = bx + bw2 / 2
            cy = by + bh2 / 2
            bbox_t = torch.tensor([cx, cy, bw2, bh2], dtype=torch.float32)
        else:
            # Fallback: full image box
            bbox_t = torch.tensor(
                [IMAGE_SIZE / 2, IMAGE_SIZE / 2, float(IMAGE_SIZE), float(IMAGE_SIZE)],
                dtype=torch.float32,
            )

        return {
            "image": image_t,
            "label": class_idx,
            "bbox":  bbox_t,
            "mask":  mask_t,
        }