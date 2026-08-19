#!/usr/bin/env python3
"""Predict ACRISS Category/Type for an image and optionally copy it into folders."""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
import torchvision
from PIL import Image

from acriss_codes import CATEGORIES, TYPES, folder_name
from resnet import get_pretrained_resnet


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    codes = ckpt["codes"]
    model_name = ckpt.get("model", "ResNet18")
    freeze = ckpt.get("freeze_weights", True)
    model = get_pretrained_resnet(freeze, len(codes), model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, codes, ckpt.get("target", "category")


def predict_image(model, image_path: str, device: torch.device) -> int:
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    if len(list(image.getbands())) < 3:
        image = image.convert("RGB")
    tensor = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ACRISS label for car images")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt from training")
    parser.add_argument("--images", nargs="+", required=True, help="Image file(s) or folder(s)")
    parser.add_argument(
        "--output_root",
        default=None,
        help="If set, copy each image into predicted label folder under this path",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, codes, target = load_model(args.checkpoint, device)

    paths: list[str] = []
    for item in args.images:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    paths.append(os.path.join(item, name))
        else:
            paths.append(item)

    for path in paths:
        idx = predict_image(model, path, device)
        code = codes[idx]
        if target == "category":
            label = f"{code} {CATEGORIES.get(code, code)}"
            folder = folder_name(code, CATEGORIES.get(code, code))
        elif target == "type":
            label = f"{code} {TYPES.get(code, code)}"
            folder = folder_name(code, TYPES.get(code, code))
        else:
            cat, typ = code[0], code[1]
            label = f"{code} ({CATEGORIES.get(cat, cat)} / {TYPES.get(typ, typ)})"
            folder = code

        print(f"{path} -> {label}")
        if args.output_root:
            dest_dir = os.path.join(args.output_root, folder)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(path))
            shutil.copy2(path, dest)
            print(f"  copied to {dest}")


if __name__ == "__main__":
    main()
