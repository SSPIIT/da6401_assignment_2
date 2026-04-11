"""Dataset loader for Oxford-IIIT Pet dataset."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224


def get_transforms(split: str = "train") -> A.Compose:
    bbox_params = A.BboxParams(
        format="coco",
        label_fields=["labels"],
        min_visibility=0.1,
    )
    if split == "train":
        return A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-15, 15), p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
                A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32),
                                hole_width_range=(16, 32), p=0.3),
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
    """Oxford-IIIT Pet multi-task dataset.

    Returns per sample:
        image : FloatTensor [3, 224, 224]  (ImageNet normalised)
        label : int                        (breed index 0-36)
        bbox  : FloatTensor [4]            (cx, cy, w, h) in pixel coords
        mask  : LongTensor  [224, 224]     (0=bg, 1=fg, 2=uncertain)
    """

    # Breed names in dataset order (class_id 1-37 from list.txt -> index 0-36)
    BREEDS = [
        "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
        "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer",
        "British_Shorthair", "chihuahua", "Egyptian_Mau",
        "english_cocker_spaniel", "english_setter", "german_shorthaired",
        "great_pyrenees", "havanese", "japanese_chin", "keeshond",
        "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
        "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
        "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
        "Siamese", "Sphynx", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier",
    ]

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[A.Compose] = None):
        self.root  = Path(root)
        self.split = split
        self.transform = transform if transform is not None else get_transforms(split)

        # Auto-detect annotation root (handles flat and nested Kaggle layouts)
        ann_root = self.root / "annotations"
        if not (ann_root / "trainval.txt").exists():
            candidate = ann_root / "annotations"
            if candidate.exists():
                ann_root = candidate
        self.ann_root = ann_root

        split_file = "trainval.txt" if split == "train" else "test.txt"
        list_path  = self.ann_root / split_file

        self.samples = []
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts    = line.split()
                stem     = parts[0]
                class_id = int(parts[1]) - 1   # 1-indexed -> 0-indexed
                self.samples.append((stem, class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_bbox(self, stem: str, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        xml_path = self.ann_root / "xmls" / f"{stem}.xml"
        if not xml_path.exists():
            return (0.0, 0.0, float(img_w), float(img_h))
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj  = root.find(".//bndbox")
            xmin = float(obj.find("xmin").text)
            ymin = float(obj.find("ymin").text)
            xmax = float(obj.find("xmax").text)
            ymax = float(obj.find("ymax").text)
            return (xmin, ymin, xmax - xmin, ymax - ymin)
        except Exception:
            return (0.0, 0.0, float(img_w), float(img_h))

    def _load_mask(self, stem: str) -> Optional[np.ndarray]:
        mask_path = self.ann_root / "trimaps" / f"{stem}.png"
        if not mask_path.exists():
            return None
        return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict:
        stem, class_idx = self.samples[idx]

        # Load image
        img_path = self.root / "images" / f"{stem}.jpg"
        image    = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        img_h, img_w = image.shape[:2]

        # Load bbox (xmin, ymin, w, h) clamped to image bounds
        xmin, ymin, bw, bh = self._load_bbox(stem, img_w, img_h)
        xmin = max(0.0, min(xmin, img_w - 1))
        ymin = max(0.0, min(ymin, img_h - 1))
        bw   = max(1.0, min(bw,   img_w - xmin))
        bh   = max(1.0, min(bh,   img_h - ymin))

        # Load mask
        raw_mask = self._load_mask(stem)
        if raw_mask is None:
            raw_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # Apply transforms
        transformed = self.transform(
            image=image,
            mask=raw_mask,
            bboxes=[[xmin, ymin, bw, bh]],
            labels=[class_idx],
        )

        image_t = transformed["image"].float()
        mask_t  = transformed["mask"].long()

        # Remap trimap: 1=fg->1, 2=bg->0, 3=uncertain->2
        remapped = torch.zeros_like(mask_t)
        remapped[mask_t == 1] = 1
        remapped[mask_t == 2] = 0
        remapped[mask_t == 3] = 2
        mask_t = remapped

        # Bbox -> cx, cy, w, h in post-resize pixel coords
        if len(transformed["bboxes"]) > 0:
            bx, by, bw2, bh2 = transformed["bboxes"][0]
            bbox_t = torch.tensor([bx + bw2/2, by + bh2/2, bw2, bh2], dtype=torch.float32)
        else:
            s = float(IMAGE_SIZE)
            bbox_t = torch.tensor([s/2, s/2, s, s], dtype=torch.float32)

        return {"image": image_t, "label": class_idx, "bbox": bbox_t, "mask": mask_t}
