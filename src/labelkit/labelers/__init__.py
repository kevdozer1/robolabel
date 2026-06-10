"""Labelers: turn an :class:`~labelkit.episode.Episode` into annotations.

Each labeler runs the provider's two-stage observe→label flow and returns typed
results plus the provider calls (for cost/receipt accounting). Shared helpers for
keyframe selection and segment validation live here.
"""

from __future__ import annotations

import numpy as np

from ..episode import Episode
from ..schema import SubtaskSegment


def select_keyframes(episode: Episode, n_keyframes: int) -> list[int]:
    """Evenly-spaced keyframe indices, biased toward gripper transitions if known.

    If the episode carries an ``actions`` array whose last channel behaves like a
    gripper, frames around large gripper changes are added — these tend to be the
    semantically meaningful boundaries. This is a best-effort hint, never required.
    """
    n = episode.num_frames
    if n <= 1:
        return [0]
    wanted = {int(round(x)) for x in np.linspace(0, n - 1, min(n_keyframes, n))}
    for boundary in _gripper_transitions(episode):
        wanted.add(min(max(boundary, 0), n - 1))
    return sorted(wanted)


def _gripper_transitions(episode: Episode, threshold: float = 0.25) -> list[int]:
    actions = episode.actions
    if actions is None or getattr(actions, "ndim", 0) != 2 or actions.shape[1] < 1 or actions.shape[0] < 4:
        return []
    grip = np.asarray(actions[:, -1], dtype=float)
    rng = float(grip.max() - grip.min())
    if rng <= 1e-6:
        return []
    grip = (grip - grip.min()) / rng
    return [int(i) + 1 for i in np.where(np.abs(np.diff(grip)) > threshold)[0]]


def validate_segments(raw: object, num_frames: int, min_seg: int, max_seg: int) -> list[SubtaskSegment]:
    """Coerce a model's segment list into contiguous, in-range subtasks.

    Guarantees: 1..``max_seg`` segments, sorted, non-overlapping and contiguous,
    first starts at 0, last ends at ``num_frames - 1``, non-empty subtask text.
    """
    items = raw.get("segments", raw.get("subtasks", [])) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        items = []
    last = max(0, num_frames - 1)
    clean: list[SubtaskSegment] = []
    cursor = 0
    for item in items[:max_seg]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("subtask_text") or item.get("text") or item.get("description") or "").strip()
        if not text:
            continue
        end = item.get("end_step", item.get("end_frame"))
        try:
            end_i = int(end)
        except (TypeError, ValueError):
            end_i = cursor
        end_i = min(max(cursor, end_i), last)
        clean.append(SubtaskSegment(segment_idx=len(clean), start_frame=cursor, end_frame=end_i, subtask_text=text[:160]))
        cursor = end_i + 1
        if cursor > last:
            break
    if not clean:
        clean = [SubtaskSegment(0, 0, last, "complete the task")]
    clean[0].start_frame = 0
    clean[-1].end_frame = last
    # Renumber to be safe.
    for i, seg in enumerate(clean):
        seg.segment_idx = i
    return clean
