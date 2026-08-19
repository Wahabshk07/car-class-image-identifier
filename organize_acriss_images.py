#!/usr/bin/env python3
"""Organize Stanford Cars images into ACRISS Category and Type folders.

Creates:
  acriss_images/
    by_category/<code>_<name>/...
    by_type/<code>_<name>/...
    by_acriss/<category><type>_<catname>__<typename>/...
    mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict

from scipy.io import loadmat

from acriss_codes import (
    CATEGORIES,
    TYPES,
    STANFORD_TO_ACRISS,
    folder_name,
    get_acriss,
)


def load_class_names(meta_path: str) -> list[str]:
    meta = loadmat(meta_path)
    return [str(x[0]) for x in meta["class_names"][0]]


def load_split_annotations(mat_path: str) -> list[tuple[str, int]]:
    """Return list of (filename, class_id_1based)."""
    annotations = loadmat(mat_path)["annotations"][0]
    rows = []
    for row in annotations:
        fname = str(row["fname"][0])
        class_id = int(row["class"][0][0])
        rows.append((fname, class_id))
    return rows


def ensure_link_or_copy(src: str, dst: str, mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def organize(
    dataset_root: str,
    output_root: str,
    mode: str,
    splits: list[str],
) -> None:
    meta_path = os.path.join(dataset_root, "devkit", "cars_meta.mat")
    class_names = load_class_names(meta_path)

    split_files = {
        "train": (
            os.path.join(dataset_root, "cars_train"),
            os.path.join(dataset_root, "devkit", "cars_train_annos.mat"),
        ),
        "test": (
            os.path.join(dataset_root, "cars_test"),
            os.path.join(dataset_root, "cars_test_annos_withlabels.mat"),
        ),
    }

    by_category = os.path.join(output_root, "by_category")
    by_type = os.path.join(output_root, "by_type")
    by_acriss = os.path.join(output_root, "by_acriss")
    os.makedirs(by_category, exist_ok=True)
    os.makedirs(by_type, exist_ok=True)
    os.makedirs(by_acriss, exist_ok=True)

    counts_category: dict[str, int] = defaultdict(int)
    counts_type: dict[str, int] = defaultdict(int)
    counts_acriss: dict[str, int] = defaultdict(int)

    mapping_rows = []
    for class_id, (cat, typ) in sorted(STANFORD_TO_ACRISS.items()):
        mapping_rows.append(
            {
                "stanford_class_id": class_id,
                "stanford_class_name": class_names[class_id - 1],
                "category_code": cat,
                "category_name": CATEGORIES[cat],
                "type_code": typ,
                "type_name": TYPES[typ],
                "acriss_ct": f"{cat}{typ}",
            }
        )

    mapping_csv = os.path.join(output_root, "mapping.csv")
    with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mapping_rows)

    placed = 0
    for split in splits:
        image_dir, anno_path = split_files[split]
        for fname, class_id in load_split_annotations(anno_path):
            cat, typ = get_acriss(class_id)
            src = os.path.join(image_dir, fname)
            if not os.path.isfile(src):
                raise FileNotFoundError(src)

            cat_folder = folder_name(cat, CATEGORIES[cat])
            type_folder = folder_name(typ, TYPES[typ])
            acriss_code = f"{cat}{typ}"
            acriss_folder = f"{acriss_code}_{CATEGORIES[cat]}__{folder_name(typ, TYPES[typ]).split('_', 1)[1]}"

            # Keep unique names across train/test
            out_name = f"{split}_{fname}"

            ensure_link_or_copy(
                src,
                os.path.join(by_category, cat_folder, out_name),
                mode,
            )
            ensure_link_or_copy(
                src,
                os.path.join(by_type, type_folder, out_name),
                mode,
            )
            ensure_link_or_copy(
                src,
                os.path.join(by_acriss, acriss_folder, out_name),
                mode,
            )

            counts_category[cat] += 1
            counts_type[typ] += 1
            counts_acriss[acriss_code] += 1
            placed += 1

    summary_path = os.path.join(output_root, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"images_placed={placed}\nmode={mode}\nsplits={','.join(splits)}\n\n")
        f.write("CATEGORY COUNTS\n")
        for code, count in sorted(counts_category.items(), key=lambda x: -x[1]):
            f.write(f"  {code} {CATEGORIES[code]}: {count}\n")
        f.write("\nTYPE COUNTS\n")
        for code, count in sorted(counts_type.items(), key=lambda x: -x[1]):
            f.write(f"  {code} {TYPES[code]}: {count}\n")
        f.write("\nACRISS CATEGORY+TYPE COUNTS\n")
        for code, count in sorted(counts_acriss.items(), key=lambda x: -x[1]):
            f.write(f"  {code}: {count}\n")

    print(f"Done. Placed {placed} images under {output_root}")
    print(f"Mapping: {mapping_csv}")
    print(f"Summary: {summary_path}")
    print("Folders:")
    print(f"  {by_category}")
    print(f"  {by_type}")
    print(f"  {by_acriss}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort car images into ACRISS folders")
    parser.add_argument(
        "--dataset_root",
        default="./stanford_cars",
        help="Folder with cars_train/, cars_test/, devkit/",
    )
    parser.add_argument(
        "--output_root",
        default="./acriss_images",
        help="Where to create category/type folders",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="symlink saves disk space; copy duplicates files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Which splits to organize",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    organize(args.dataset_root, args.output_root, args.mode, args.splits)
