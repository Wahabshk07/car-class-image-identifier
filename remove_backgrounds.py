#!/usr/bin/env python3
"""Remove backgrounds from images under acriss_images/ and cars_by_model/ only.

Symlink targets are deduplicated so each unique photo is processed once.
Outputs go to mirrored folders with transparent PNGs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from bg_remove import BgSession

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_jobs(roots: list[Path]) -> list[tuple[Path, Path, Path]]:
    """Return list of (link_or_file, real_source, root)."""
    jobs: list[tuple[Path, Path, Path]] = []
    for root in roots:
        if not root.is_dir():
            print(f"skip missing folder: {root}", file=sys.stderr)
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            if not (path.is_file() or path.is_symlink()):
                continue
            try:
                real = Path(os.path.realpath(path))
            except OSError:
                continue
            if not real.is_file():
                continue
            jobs.append((path, real, root))
    return jobs


def process(
    roots: list[Path],
    output_map: dict[Path, Path],
    model_name: str,
    limit: int | None,
    skip_existing: bool,
) -> None:
    session = BgSession(model_name)
    jobs = collect_jobs(roots)
    print(f"found {len(jobs)} image links/files under {', '.join(str(r) for r in roots)}")

    unique_sources: dict[Path, list[tuple[Path, Path]]] = {}
    for link_path, real, root in jobs:
        dest = output_map[root] / link_path.relative_to(root).with_suffix(".png")
        unique_sources.setdefault(real, []).append((link_path, dest))

    items = list(unique_sources.items())
    if limit is not None:
        items = items[:limit]
    print(f"unique images to process: {len(items)}")

    processed = 0
    written = 0
    skipped = 0

    for real, destinations in tqdm(items, desc="remove-bg"):
        pending = [dest for _link, dest in destinations if not (skip_existing and dest.exists())]
        if not pending:
            skipped += len(destinations)
            continue

        with Image.open(real) as img:
            result = session.remove(img)

        processed += 1
        saved_for: set[Path] = set()
        for _link, dest in destinations:
            if skip_existing and dest.exists():
                skipped += 1
                continue
            if dest in saved_for:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            result.save(dest, format="PNG")
            saved_for.add(dest)
            written += 1

        del result

    print(f"done. unique_processed={processed} files_written={written} skipped={skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove backgrounds for images in acriss_images and cars_by_model only"
    )
    parser.add_argument("--acriss_root", default="./acriss_images")
    parser.add_argument("--models_root", default="./cars_by_model")
    parser.add_argument("--acriss_out", default="./acriss_images_nobg")
    parser.add_argument("--models_out", default="./cars_by_model_nobg")
    parser.add_argument(
        "--model",
        default="u2netp",
        choices=["u2netp", "u2net"],
        help="u2netp = faster, u2net = better quality",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N unique images")
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing outputs",
    )
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [
        (Path(args.acriss_root).resolve(), Path(args.acriss_out).resolve()),
        (Path(args.models_root).resolve(), Path(args.models_out).resolve()),
    ]
    existing = [(src, dst) for src, dst in candidates if src.is_dir()]
    if not existing:
        raise SystemExit("Neither acriss_images nor cars_by_model exists.")

    roots = [src for src, _ in existing]
    output_map = {src: dst for src, dst in existing}
    process(
        roots=roots,
        output_map=output_map,
        model_name=args.model,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
