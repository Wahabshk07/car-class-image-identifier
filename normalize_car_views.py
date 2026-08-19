#!/usr/bin/env python3
"""Keep side-view cars, face them right, letterbox to NxN. Reject non-side views."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def car_mask(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3] > 10
    gray = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # car usually darker/foreground; pick larger blob polarity via edge density
    if m.mean() > 127:
        m = 255 - m
    return m > 0


def bbox(mask: np.ndarray, pad: float = 0.04) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h, w = mask.shape
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px + 1), min(h, y1 + py + 1)


def faces_left(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Front hood tends to leave more empty space in the upper-front corner."""
    x0, y0, x1, y1 = box
    mid_x = (x0 + x1) // 2
    top = y0 + max(1, int((y1 - y0) * 0.4))
    left_empty = (~mask[y0:top, x0:mid_x]).mean()
    right_empty = (~mask[y0:top, mid_x:x1]).mean()
    return right_empty > left_empty  # more empty on right => nose points left


def letterbox(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")
    scale = size / max(img.size)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    bg = (0, 0, 0, 0) if resized.mode == "RGBA" else (255, 255, 255)
    canvas = Image.new(resized.mode, (size, size), bg)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def process_one(path: Path, size: int, min_aspect: float) -> tuple[str, Image.Image | None]:
    img = Image.open(path)
    mask = car_mask(img)
    box = bbox(mask)
    if box is None:
        return "reject", None
    x0, y0, x1, y1 = box
    aspect = (x1 - x0) / max(1, y1 - y0)
    if aspect < min_aspect:
        return "reject", img.crop(box).convert("RGBA")
    if faces_left(mask, box):
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = np.fliplr(mask)
        box = bbox(mask) or box
        x0, y0, x1, y1 = box
    return "keep", letterbox(img.crop((x0, y0, x1, y1)), size)


def iter_images(root: Path):
    seen: set[Path] = set()
    for p in root.rglob("*"):
        if p.suffix.lower() not in EXTS or not (p.is_file() or p.is_symlink()):
            continue
        real = Path(os.path.realpath(p))
        if not real.is_file() or real in seen:
            continue
        seen.add(real)
        yield p, real


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize side-view cars to square canvases")
    ap.add_argument("--input", default="./cars_by_model_nobg/by_model")
    ap.add_argument("--output", default="./cars_side_800")
    ap.add_argument("--rejected", default="./cars_rejected_not_side")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--min_aspect", type=float, default=1.35)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip-existing", dest="skip", action="store_false")
    ap.set_defaults(skip=True)
    args = ap.parse_args()

    root = Path(args.input).resolve()
    out_root, rej_root = Path(args.output).resolve(), Path(args.rejected).resolve()
    if not root.is_dir():
        raise SystemExit(f"missing input: {root}")

    jobs = list(iter_images(root))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    kept = rejected = skipped = 0
    for link, real in tqdm(jobs, desc="normalize"):
        rel = link.relative_to(root)
        dest_keep = out_root / rel.with_suffix(".png")
        dest_rej = rej_root / rel.with_suffix(".png")
        if args.skip and (dest_keep.exists() or dest_rej.exists()):
            skipped += 1
            continue
        verdict, result = process_one(real, args.size, args.min_aspect)
        if result is None:
            rejected += 1
            continue
        dest = dest_keep if verdict == "keep" else dest_rej
        dest.parent.mkdir(parents=True, exist_ok=True)
        if verdict == "keep":
            result.save(dest, format="PNG")
            kept += 1
        else:
            letterbox(result, args.size).save(dest, format="PNG")
            rejected += 1

    print(f"done. kept={kept} rejected={rejected} skipped={skipped} out={out_root}")


if __name__ == "__main__":
    main()
