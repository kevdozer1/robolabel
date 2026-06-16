"""Proprioception-fused grasp/release boundary refinement (deterministic, no VLM).

The grounded VLM segmentation is good at *which* phases happen but imprecise at the exact frame
of a contact event. The gripper, however, marks grasp (open→close) and release (close→open)
exactly. So: detect the gripper transitions (reusing the S_grip logic) and **snap** — never
resegment — grasp-onset / release-onset boundaries to the nearest in-window transition. Every
other boundary is left untouched, and the snap **no-ops cleanly** when no transition exists
in-window (e.g. pour/fold, where the gripper holds throughout and never marks the boundary).
"""
from __future__ import annotations

import numpy as np

from .labelers.gripper_baseline import _gripper_transitions

_GRASP = ("grasp", "grip", "pick", "pinch", "clamp")
_RELEASE = ("release", "place", "drop", "deposit", "let go", "set down", "put down")


def _kind(phase: str | None) -> str:
    p = (phase or "").lower()
    if any(t in p for t in _GRASP):
        return "grasp"
    if any(t in p for t in _RELEASE):
        return "release"
    return "other"


def _directed_transitions(grip: np.ndarray, threshold: float, min_spacing: int) -> list[tuple[int, str]]:
    """Gripper transition frames tagged 'close' (open→closed) or 'open' (closed→open)."""
    grip = np.asarray(grip, dtype="float64").reshape(-1)
    rng = float(grip.max() - grip.min())
    if rng <= 1e-9:
        return []
    closed = ((grip - grip.min()) / rng < threshold).astype(int)
    return [(int(f), "close" if (0 <= f < len(closed) and closed[f]) else "open")
            for f in _gripper_transitions(grip, threshold, min_spacing)]


def snap_contact_boundaries(segments, gripper_series, *, window: int = 8,
                            threshold: float = 0.5, min_spacing: int = 8):
    """Snap grasp-onset / release-onset boundaries to the nearest in-window gripper transition.

    ``gripper_series``: the 1-D gripper position over the episode. Mutates + returns
    ``segments`` and the number of boundaries snapped. No-op (0 snaps) when there is no gripper
    motion or no transition lands within ``window`` of the boundary — so it cannot invent a
    contact event where the gripper never moves.
    """
    if gripper_series is None or not segments or np.asarray(gripper_series).size < 3:
        return segments, 0
    trans = _directed_transitions(np.asarray(gripper_series), threshold, min_spacing)
    if not trans:
        return segments, 0
    closes = [f for f, d in trans if d == "close"]
    opens = [f for f, d in trans if d == "open"]
    snapped = 0
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        ka, kb = _kind(a.phase), _kind(b.phase)
        if kb == "grasp" and ka != "grasp":
            cands = closes            # the boundary that begins the grasp -> gripper closes
        elif kb == "release" and ka != "release":
            cands = opens             # the boundary that begins the release -> gripper opens
        else:
            continue
        if not cands:
            continue
        bd = a.end_frame
        near = min(cands, key=lambda t: abs(t - bd))
        if abs(near - bd) <= window and a.start_frame < near < b.end_frame:
            a.end_frame = int(near)
            b.start_frame = int(near) + 1
            snapped += 1
    return segments, snapped


def gripper_dim(state_names: list[str] | None, n_dims: int) -> int:
    """Index of the gripper dim in a state/action vector (by name, else last)."""
    if state_names:
        for i, n in enumerate(state_names):
            if "gripper" in str(n).lower():
                return i
    return max(0, n_dims - 1)
