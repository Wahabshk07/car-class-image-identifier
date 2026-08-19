#!/usr/bin/env python3
"""Minimal background removal with onnxruntime + u2netp (no rembg/pymatting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pooch
from PIL import Image

MODEL_URLS = {
    "u2netp": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "u2netp.onnx",
        "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
    ),
    "u2net": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "u2net.onnx",
        None,
    ),
}


class BgSession:
    def __init__(self, model_name: str = "u2netp") -> None:
        if model_name not in MODEL_URLS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(MODEL_URLS)}")
        url, filename, known_hash = MODEL_URLS[model_name]
        model_dir = Path.home() / ".u2net"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = pooch.retrieve(
            url=url,
            known_hash=known_hash,
            fname=filename,
            path=model_dir,
            progressbar=True,
        )
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 320

    def remove(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        original_size = image.size
        arr = np.array(image.resize((self.input_size, self.input_size), Image.BILINEAR)).astype(
            np.float32
        )
        # normalize like rembg/u2net
        arr = (arr / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = arr.transpose(2, 0, 1)[None, ...]
        ort_outs = self.session.run(None, {self.input_name: tensor})
        mask = ort_outs[0][0][0]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
            original_size, Image.BILINEAR
        )
        rgba = image.convert("RGBA")
        rgba.putalpha(mask_img)
        return rgba
