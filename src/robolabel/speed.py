"""Deterministic episode `speed` from the action stream (no VLM) — pi0.7's other metadata.

The primary descriptor is **continuous, phase-agnostic, and generalizable**: an *active window*
defined purely by motion, not by named phases. From the per-step action velocity we find motion
**onset** (first velocity above a relative threshold) and **offset** (last), and report
``active_frames`` / ``active_seconds`` and ``active_fraction = active/total`` — the same for any
task. A categorical fast/medium/slow tier is emitted ONLY when a large/heterogeneous enough
population exists to bin against (corpus-relative; see ``corpus.py``), else it is left null.
Thresholds: ``rubric.yaml -> speed``.
"""
from __future__ import annotations

import numpy as np


def episode_speed_norm(actions: np.ndarray | None) -> float:
    """Mean per-step action velocity over the episode (raw continuous; 0.0 if no stream)."""
    if actions is None or len(actions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.asarray(actions, dtype="float64"), axis=0), axis=1).mean())


def _motion_speed(actions: np.ndarray, smooth: int = 5) -> np.ndarray:
    v = np.linalg.norm(np.diff(np.asarray(actions, dtype="float64"), axis=0), axis=1)
    if smooth > 1 and len(v) >= smooth:
        v = np.convolve(v, np.ones(smooth) / smooth, mode="same")
    return v


def active_window(actions: np.ndarray | None, *, rel_threshold: float = 0.12, smooth: int = 5) -> dict:
    """Motion-defined active window: onset/offset + active_frames + active_fraction.

    Phase-agnostic — the window is "from when the arm starts moving to when it settles", defined
    by action velocity crossing ``rel_threshold`` * the episode's 90th-percentile velocity.
    """
    n = 0 if actions is None else len(actions)
    if actions is None or n < 3:
        return {"onset": 0, "offset": max(0, n - 1), "active_frames": 0, "active_fraction": 0.0}
    v = _motion_speed(actions, smooth)
    ref = float(np.quantile(v, 0.9)) or float(v.max())
    above = np.where(v > rel_threshold * ref)[0]
    if len(above) == 0:
        return {"onset": 0, "offset": 0, "active_frames": 0, "active_fraction": 0.0}
    onset, offset = int(above[0]), int(above[-1]) + 1     # +1: v is diff-based (len n-1)
    af = offset - onset
    return {"onset": onset, "offset": offset, "active_frames": int(af),
            "active_fraction": round(af / max(1, n), 4)}


def bin_speeds(speed_by_ep: dict[str, float], cuts: tuple[float, float] = (1 / 3, 2 / 3),
               min_population: int = 15) -> dict[str, str | None]:
    """Bin per-episode speed scalars into fast/medium/slow by tercile (higher = faster).

    Corpus-relative + guarded: returns all-``None`` ("insufficient population to tier") unless
    the population is at least ``min_population`` and has real spread. Never fabricates tiers on
    a tiny same-y set."""
    vals = list(speed_by_ep.values())
    if len(speed_by_ep) < min_population or len(set(vals)) < 3:
        return {e: None for e in speed_by_ep}
    lo, hi = float(np.quantile(vals, cuts[0])), float(np.quantile(vals, cuts[1]))
    if hi - lo < 1e-9:
        return {e: None for e in speed_by_ep}
    return {e: ("slow" if v <= lo else "fast" if v >= hi else "medium") for e, v in speed_by_ep.items()}
