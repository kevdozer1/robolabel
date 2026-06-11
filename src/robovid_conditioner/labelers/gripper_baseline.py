"""S_grip: a proprioceptive, zero-API subtask segmentation baseline.

It segments a pick-and-place episode from the robot's own signals — no VLM, no
frames — using two cues:

* **gripper open/close transitions** (``gripper.pos`` crossing a normalized
  threshold): the first major transition is the grasp, the last is the release;
* **end-effector-speed pauses** (low-speed minima of the arm-joint velocity just
  *before* a gripper event): where the arm arrives and settles before grasping or
  placing.

Boundaries are ``approach | grasp | transport | release-place | retract`` and the
phase labels are assigned from the closed vocabulary **by event order**. This is the
"free baseline" — if it matches or beats the VLM on a given dataset, that is worth
saying out loud. On messy signals it degrades to fewer segments (caught by the gate's
degenerate / uniform-split detectors), never silently.

Thresholds live in ``rubric.yaml`` under ``gripper_baseline``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schema import SubtaskSegment

# Default phase order (truncated to the number of detected segments).
_PHASE_ORDER = ["approach", "grasp", "transport", "release-place", "retract"]


def segment_from_state(state: np.ndarray, cfg: dict[str, Any] | None = None,
                       phase_vocabulary: list[str] | None = None) -> list[SubtaskSegment]:
    """Segment one episode from its ``(num_frames, dof)`` proprioceptive state array.

    The last column is the gripper position; the rest are arm joints. Returns
    contiguous, full-coverage :class:`SubtaskSegment`s with phase labels.
    """
    cfg = cfg or {}
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    if n <= 1:
        return [SubtaskSegment(0, 0, max(0, n - 1), "complete the task", phase="other")]
    last = n - 1
    grip = state[:, -1]
    arm = state[:, :-1] if state.shape[1] > 1 else state

    transitions = _gripper_transitions(
        grip,
        threshold=float(cfg.get("gripper_norm_threshold", 0.5)),
        min_spacing=int(cfg.get("min_transition_frames", 8)),
    )
    speed = _ee_speed(arm, smooth=int(cfg.get("ee_smooth_window", 5)))
    pause_window = int(cfg.get("ee_pause_window", 40))
    min_seg = int(cfg.get("min_segment_frames", 5))

    boundaries: list[int] = []
    if transitions:
        b_grasp = transitions[0]
        b_release = transitions[-1]
        b_approach = _pause_before(speed, b_grasp, pause_window, floor=0)
        boundaries += [b for b in (b_approach, b_grasp) if b is not None]
        if b_release > b_grasp:
            b_place = _pause_before(speed, b_release, pause_window, floor=b_grasp + 1)
            boundaries += [b for b in (b_place, b_release) if b is not None]
    else:
        # No usable gripper signal: fall back to arm-speed pauses alone.
        boundaries = _speed_pause_boundaries(speed, last)

    boundaries = _dedup_monotonic(boundaries, last, min_seg)
    return _build_segments(boundaries, last, phase_vocabulary or _PHASE_ORDER)


def _gripper_transitions(grip: np.ndarray, threshold: float, min_spacing: int) -> list[int]:
    rng = float(grip.max() - grip.min())
    if rng <= 1e-9:
        return []
    g = (grip - grip.min()) / rng
    closed = (g < threshold).astype(int)
    raw = [int(i) + 1 for i in np.where(np.diff(closed) != 0)[0]]
    # Debounce: keep the first of any cluster within min_spacing.
    kept: list[int] = []
    for f in raw:
        if not kept or f - kept[-1] >= min_spacing:
            kept.append(f)
    return kept


def _ee_speed(arm: np.ndarray, smooth: int) -> np.ndarray:
    v = np.linalg.norm(np.diff(arm, axis=0), axis=1)
    v = np.concatenate([v, v[-1:]]) if len(v) else np.zeros(arm.shape[0])
    if smooth > 1 and len(v) >= smooth:
        kernel = np.ones(smooth) / smooth
        v = np.convolve(v, kernel, mode="same")
    return v


def _pause_before(speed: np.ndarray, event: int, window: int, floor: int) -> int | None:
    lo = max(floor, event - window)
    if event - lo < 2:
        return None
    seg = speed[lo:event]
    return int(lo + int(np.argmin(seg)))


def _speed_pause_boundaries(speed: np.ndarray, last: int) -> list[int]:
    if len(speed) < 4:
        return []
    thr = float(np.percentile(speed, 25))
    low = speed < thr
    # Boundary at the centre of each contiguous low-speed run (a settle point).
    bounds: list[int] = []
    i = 1
    while i < len(low) - 1:
        if low[i] and not low[i - 1]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            bounds.append((i + j) // 2)
            i = j
        else:
            i += 1
    return bounds


def _dedup_monotonic(boundaries: list[int], last: int, min_seg: int) -> list[int]:
    out: list[int] = []
    for b in sorted(set(int(x) for x in boundaries)):
        if b <= 0 or b >= last:
            continue
        if out and b - out[-1] < min_seg:
            continue
        out.append(b)
    return out


def _build_segments(boundaries: list[int], last: int, phase_vocab: list[str]) -> list[SubtaskSegment]:
    edges = [0] + [b + 1 for b in boundaries]
    segments: list[SubtaskSegment] = []
    for i, start in enumerate(edges):
        end = (boundaries[i] if i < len(boundaries) else last)
        end = min(max(start, end), last)
        phase = phase_vocab[i] if i < len(phase_vocab) else "other"
        segments.append(SubtaskSegment(
            segment_idx=i, start_frame=start, end_frame=end,
            subtask_text=f"{phase} (proprioceptive)", phase=phase,
            evidence="gripper/end-effector event",
        ))
    if not segments:
        segments = [SubtaskSegment(0, 0, last, "complete the task", phase="other")]
    segments[-1].end_frame = last
    return segments
