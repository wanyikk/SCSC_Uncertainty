from __future__ import annotations

import numpy as np


_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    a = 0.055
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4).astype(np.float32)


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    a = 0.055
    lin = np.clip(lin, 0.0, 1.0).astype(np.float32)
    return np.where(lin <= 0.0031308, lin * 12.92, (1 + a) * (lin ** (1 / 2.4)) - a).astype(np.float32)


def rgb01_to_xyz(rgb01: np.ndarray) -> np.ndarray:
    rgb_lin = _srgb_to_linear(rgb01)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    return rgb_lin @ m.T


def xyz_to_rgb01(xyz: np.ndarray) -> np.ndarray:
    m = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float32,
    )
    rgb_lin = xyz @ m.T
    return _linear_to_srgb(rgb_lin)


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz_n = xyz / _D65

    delta = 6 / 29
    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29).astype(np.float32)

    fx, fy, fz = f(xyz_n[..., 0]), f(xyz_n[..., 1]), f(xyz_n[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    delta = 6 / 29
    def finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29)).astype(np.float32)

    x = finv(fx)
    y = finv(fy)
    z = finv(fz)
    xyz_n = np.stack([x, y, z], axis=-1)
    return xyz_n * _D65


def rgb01_to_lab(rgb01: np.ndarray) -> np.ndarray:
    xyz = rgb01_to_xyz(rgb01)
    return xyz_to_lab(xyz)


def lab_to_rgb01(lab: np.ndarray) -> np.ndarray:
    xyz = lab_to_xyz(lab)
    return xyz_to_rgb01(xyz)

