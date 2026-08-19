#!/usr/bin/env python3
"""Train a ResNet to predict ACRISS Category and/or Type from car images."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter

import torch
import torchvision
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from acriss_codes import CATEGORIES, TYPES, get_acriss
from resnet import get_pretrained_resnet

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("acriss_train")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AcrissCarsDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transforms):
        self.samples = samples
        self.transforms = torchvision.transforms.Compose(transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path)
        if len(list(image.getbands())) < 3:
            image = image.convert("RGB")
        return {"image": self.transforms(image), "label": label}


def load_annotations(mat_path: str) -> list[tuple[str, int]]:
    annotations = loadmat(mat_path)["annotations"][0]
    return [(str(row["fname"][0]), int(row["class"][0][0])) for row in annotations]


def build_label_space(target: str) -> tuple[list[str], dict[str, int]]:
    if target == "category":
        codes = sorted(CATEGORIES.keys())
    elif target == "type":
        codes = sorted(TYPES.keys())
    elif target == "acriss":
        # Only codes that appear in the Stanford mapping
        from acriss_codes import STANFORD_TO_ACRISS

        codes = sorted({f"{c}{t}" for c, t in STANFORD_TO_ACRISS.values()})
    else:
        raise ValueError(target)
    return codes, {code: i for i, code in enumerate(codes)}


def stanford_to_target_label(class_id: int, target: str, code_to_idx: dict[str, int]) -> int:
    cat, typ = get_acriss(class_id)
    if target == "category":
        code = cat
    elif target == "type":
        code = typ
    else:
        code = f"{cat}{typ}"
    return code_to_idx[code]


def collect_samples(
    image_dir: str,
    anno_path: str,
    target: str,
    code_to_idx: dict[str, int],
) -> list[tuple[str, int]]:
    samples = []
    for fname, class_id in load_annotations(anno_path):
        path = os.path.join(image_dir, fname)
        label = stanford_to_target_label(class_id, target, code_to_idx)
        samples.append((path, label))
    return samples


def train_one_epoch(model, loader, optimizer, criterion) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true, all_pred = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    acc = correct / max(total, 1)
    f1 = f1_score(all_true, all_pred, average="weighted")
    return total_loss / max(total, 1), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true, all_pred = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
    acc = correct / max(total, 1)
    f1 = f1_score(all_true, all_pred, average="weighted")
    return total_loss / max(total, 1), acc, f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACRISS Category/Type classifier")
    parser.add_argument("--dataset_root", default="./stanford_cars")
    parser.add_argument(
        "--target",
        choices=["category", "type", "acriss"],
        default="category",
        help="Predict Category, Type, or Category+Type code",
    )
    parser.add_argument("--model", choices=["ResNet18", "ResNet50"], default="ResNet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_weights", action="store_true", default=True)
    parser.add_argument("--no-freeze_weights", dest="freeze_weights", action="store_false")
    parser.add_argument("--output_dir", default="./acriss_models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    codes, code_to_idx = build_label_space(args.target)
    num_classes = len(codes)
    logger.info("target=%s num_classes=%d device=%s", args.target, num_classes, device)
    logger.info("label codes: %s", codes)

    train_samples = collect_samples(
        os.path.join(args.dataset_root, "cars_train"),
        os.path.join(args.dataset_root, "devkit", "cars_train_annos.mat"),
        args.target,
        code_to_idx,
    )
    val_samples = collect_samples(
        os.path.join(args.dataset_root, "cars_test"),
        os.path.join(args.dataset_root, "cars_test_annos_withlabels.mat"),
        args.target,
        code_to_idx,
    )

    logger.info("train=%d val=%d", len(train_samples), len(val_samples))
    logger.info("train label counts: %s", Counter(label for _, label in train_samples))

    transforms = [
        torchvision.transforms.Resize(224),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    val_transforms = [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    train_loader = DataLoader(
        AcrissCarsDataset(train_samples, transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        AcrissCarsDataset(val_samples, val_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = get_pretrained_resnet(args.freeze_weights, num_classes, args.model)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=3, verbose=True
    )

    run_dir = os.path.join(args.output_dir, f"{args.target}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"target": args.target, "codes": codes}, f, indent=2)

    best_acc = 0.0
    best_path = os.path.join(run_dir, "best.pt")
    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        logger.info(
            "epoch %d train_loss=%.4f train_acc=%.4f train_f1=%.4f | val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_acc,
            val_f1,
        )
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "target": args.target,
                    "codes": codes,
                    "model": args.model,
                    "freeze_weights": args.freeze_weights,
                },
                best_path,
            )
            logger.info("saved best checkpoint to %s (acc=%.4f)", best_path, best_acc)

    logger.info("Finished. Best val accuracy: %.4f", best_acc)
    logger.info("Model: %s", best_path)


if __name__ == "__main__":
    main()
