from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def iter_images(input_path: str) -> Iterable[Path]:
    p = Path(input_path)
    if p.is_file():
        yield p
        return
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        for f in sorted(p.glob(ext)):
            yield f


def read_rgb_uint8(path: str | os.PathLike) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def resize_rgb_uint8(rgb_uint8: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(rgb_uint8, mode="RGB")
    img = img.resize((size, size), resample=Image.BICUBIC)
    return np.array(img, dtype=np.uint8)


def save_rgb_uint8(rgb_uint8: np.ndarray, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8, mode="RGB").save(p)

