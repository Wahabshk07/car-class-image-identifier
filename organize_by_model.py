#!/usr/bin/env python3
"""Put each Stanford Cars image into a folder named after its car model.

Creates:
  cars_by_model/
    <model_name>/train_00001.jpg ...
  cars_by_category_model/
    <Category>/
      <model_name>/...
"""

from __future__ import annotations

import argparse
import os
import re
import shutil

from scipy.io import loadmat

from acriss_codes import CATEGORIES, folder_name, get_acriss


def safe_name(text: str) -> str:
    text = text.strip().replace("/", "-")
    text = re.sub(r"[^\w.\- ]+", "", text)
    text = re.sub(r"\s+", "_", text)
    return text


def load_class_names(meta_path: str) -> list[str]:
    meta = loadmat(meta_path)
    return [str(x[0]) for x in meta["class_names"][0]]


def load_annotations(mat_path: str) -> list[tuple[str, int]]:
    annotations = loadmat(mat_path)["annotations"][0]
    return [(str(row["fname"][0]), int(row["class"][0][0])) for row in annotations]


def link_or_copy(src: str, dst: str, mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def organize(dataset_root: str, output_root: str, mode: str) -> None:
    class_names = load_class_names(os.path.join(dataset_root, "devkit", "cars_meta.mat"))
    splits = {
        "train": (
            os.path.join(dataset_root, "cars_train"),
            os.path.join(dataset_root, "devkit", "cars_train_annos.mat"),
        ),
        "test": (
            os.path.join(dataset_root, "cars_test"),
            os.path.join(dataset_root, "cars_test_annos_withlabels.mat"),
        ),
    }

    by_model = os.path.join(output_root, "by_model")
    by_category_model = os.path.join(output_root, "by_category_model")
    os.makedirs(by_model, exist_ok=True)
    os.makedirs(by_category_model, exist_ok=True)

    placed = 0
    for split, (image_dir, anno_path) in splits.items():
        for fname, class_id in load_annotations(anno_path):
            model_name = safe_name(class_names[class_id - 1])
            cat, _typ = get_acriss(class_id)
            cat_folder = folder_name(cat, CATEGORIES[cat])
            src = os.path.join(image_dir, fname)
            out_name = f"{split}_{fname}"

            link_or_copy(src, os.path.join(by_model, model_name, out_name), mode)
            link_or_copy(
                src,
                os.path.join(by_category_model, cat_folder, model_name, out_name),
                mode,
            )
            placed += 1

    print(f"Done. Placed {placed} images under {output_root}")
    print(f"  models only:     {by_model}")
    print(f"  category+model:  {by_category_model}")
    print(f"  model folders:   {len(os.listdir(by_model))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort images into car model folders")
    parser.add_argument("--dataset_root", default="./stanford_cars")
    parser.add_argument("--output_root", default="./cars_by_model")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    organize(args.dataset_root, args.output_root, args.mode)
