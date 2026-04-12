"""Inference and evaluation utilities."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from multitask import MultiTaskPerceptionModel


# ImageNet normalisation — must match training
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224

BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
    "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]


def preprocess_image(img_path: str) -> torch.Tensor:
    """Load and preprocess a single image for inference."""
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return transform(image=image)["image"].unsqueeze(0).float()  # [1, 3, 224, 224]


def run_multitask_inference(img_path: str, device: torch.device,
                            classifier_path: str = "checkpoints/classifier.pth",
                            localizer_path:  str = "checkpoints/localizer.pth",
                            unet_path:       str = "checkpoints/unet.pth") -> dict:
    """Run full pipeline on a single image."""
    model = MultiTaskPerceptionModel(
        classifier_path=classifier_path,
        localizer_path=localizer_path,
        unet_path=unet_path,
    ).to(device).eval()

    img_tensor = preprocess_image(img_path).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    cls_logits = outputs["classification"][0]          # [37]
    bbox       = outputs["localization"][0].cpu()      # [4]
    seg_logits = outputs["segmentation"][0]            # [3, H, W]

    probs      = torch.softmax(cls_logits, dim=0).cpu()
    breed_idx  = probs.argmax().item()
    confidence = probs[breed_idx].item()
    seg_mask   = seg_logits.argmax(dim=0).cpu().numpy()  # [H, W]

    return {
        "breed":      BREEDS[breed_idx] if breed_idx < len(BREEDS) else str(breed_idx),
        "confidence": confidence,
        "bbox":       bbox.numpy(),   # [cx, cy, w, h]
        "seg_mask":   seg_mask,
    }


def visualize_prediction(img_path: str, result: dict, out_path: str = "prediction.png"):
    """Save a 3-panel visualisation: original + bbox | segmentation mask."""
    image = np.array(Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    bbox  = result["bbox"]
    cx, cy, bw, bh = bbox
    x1, y1 = cx - bw / 2, cy - bh / 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel 1: image + bounding box
    axes[0].imshow(image)
    rect = patches.Rectangle(
        (x1, y1), bw, bh,
        linewidth=2, edgecolor="red", facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].set_title(
        f"{result['breed']} ({result['confidence']:.2%})",
        fontsize=11,
    )
    axes[0].axis("off")

    # Panel 2: segmentation mask
    cmap = matplotlib.colors.ListedColormap(["#4B7BE5", "#E56B4B", "#F0C060"])
    axes[1].imshow(result["seg_mask"], cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title("Segmentation (bg / fg / uncertain)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved visualisation → {out_path}")


def evaluate_classification(model_path: str, data_root: str, device: torch.device):
    """Evaluate classification model on test split and print macro F1."""
    from data.pets_dataset import OxfordIIITPetDataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    model = VGG11Classifier(num_classes=37).to(device)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    ds = OxfordIIITPetDataset(data_root, split="test")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"]
            preds  = model(imgs).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Classification macro F1: {macro_f1:.4f}")
    return macro_f1


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 A2 inference")
    p.add_argument("--image",  help="Path to input image for single-image inference")
    p.add_argument("--eval_cls", action="store_true",
                   help="Run classification evaluation on test split")
    p.add_argument("--data",   default="./data", help="Dataset root")
    p.add_argument("--cls_ckpt",  default="checkpoints/classifier.pth")
    p.add_argument("--loc_ckpt",  default="checkpoints/localizer.pth")
    p.add_argument("--unet_ckpt", default="checkpoints/unet.pth")
    p.add_argument("--out",    default="prediction.png")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.image:
        result = run_multitask_inference(
            args.image, device,
            classifier_path=args.cls_ckpt,
            localizer_path=args.loc_ckpt,
            unet_path=args.unet_ckpt,
        )
        print(f"Breed: {result['breed']}  confidence: {result['confidence']:.2%}")
        print(f"BBox (cx,cy,w,h): {result['bbox']}")
        visualize_prediction(args.image, result, out_path=args.out)

    if args.eval_cls:
        evaluate_classification(args.cls_ckpt, args.data, device)
