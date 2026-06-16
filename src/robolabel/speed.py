"""Deterministic episode `speed` metadata (no VLM) — one of pi0.7's two metadata signals.

Per episode: a scalar pace = mean per-step action velocity (L2 of consecutive action deltas),
which is length-robust and reads straight from the action stream. Then, across the dataset, bin
the scalars into ``fast`` / ``medium`` / ``slow`` by tercile. On easy datasets where quality is
near-degenerate (all-5s), speed is usually the more informative metadata signal — see README.
"""
from __future__ import annotations

import numpy as np


def episode_speed_norm(actions: np.ndarray | None) -> float:
    """Mean per-step action velocity over the episode (0.0 if no usable action stream)."""
    if actions is None or len(actions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(actions, dtype="float64"), axis=0), axis=1).mean())


def bin_speeds(speed_by_ep: dict[str, float], cuts: tuple[float, float] = (1 / 3, 2 / 3)) -> dict[str, str]:
    """Bin per-episode speed scalars into fast/medium/slow by dataset tercile (higher = faster)."""
    vals = [v for v in speed_by_ep.values()]
    if len(speed_by_ep) < 3 or len(set(vals)) < 2:
        return {e: "medium" for e in speed_by_ep}     # not enough spread to bin honestly
    lo, hi = float(np.quantile(vals, cuts[0])), float(np.quantile(vals, cuts[1]))
    out = {}
    for e, v in speed_by_ep.items():
        out[e] = "slow" if v <= lo else "fast" if v >= hi else "medium"
    return out
