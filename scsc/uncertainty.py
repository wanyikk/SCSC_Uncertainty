from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class UncertaintyStats:
    mean: float
    std: float
    max: float
    p50: float
    p90: float
    p95: float


def summarize_map(u_map: np.ndarray) -> UncertaintyStats:
    flat = u_map.astype(np.float32).reshape(-1)
    return UncertaintyStats(
        mean=float(np.mean(flat)),
        std=float(np.std(flat)),
        max=float(np.max(flat)),
        p50=float(np.percentile(flat, 50)),
        p90=float(np.percentile(flat, 90)),
        p95=float(np.percentile(flat, 95)),
    )


def normalize_percentile(
    u_map: np.ndarray,
    p_low: float = 5.0,
    p_high: float = 95.0,
    gamma: Optional[float] = 2.2,
    eps: float = 1e-6,
) -> np.ndarray:
    u = u_map.astype(np.float32)
    lo = float(np.percentile(u, p_low))
    hi = float(np.percentile(u, p_high))
    if hi - lo < eps:
        out = np.zeros_like(u, dtype=np.float32)
    else:
        out = np.clip((u - lo) / (hi - lo + eps), 0.0, 1.0)
    if gamma is not None:
        out = np.power(out, gamma).astype(np.float32)
    return out


def assess_relative(mean_norm: float) -> str:
    if mean_norm < 0.35:
        return "相对不确定性较低（相对高置信）"
    if mean_norm < 0.55:
        return "相对不确定性中等（相对中等置信）"
    if mean_norm < 0.75:
        return "相对不确定性较高（相对低置信）"
    return "相对不确定性很高（相对很低置信）"

