from __future__ import annotations

import numpy as np


def _jet_channel(x: np.ndarray, shift: float) -> np.ndarray:
    return np.clip(1.5 - np.abs(4.0 * x - shift), 0.0, 1.0)


def jet_colormap(gray01: np.ndarray) -> np.ndarray:
    x = np.clip(gray01.astype(np.float32), 0.0, 1.0)
    r = _jet_channel(x, 3.0)
    g = _jet_channel(x, 2.0)
    b = _jet_channel(x, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)

