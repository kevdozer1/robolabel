"""Subgoal keyframes: the achieved sub-state at the end of each subtask.

A subgoal frame is, by default, the last frame of each subtask segment — the
visual goal that subtask reaches. Images are extracted to a folder on request.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..episode import Episode
from ..schema import Subgoal, SubtaskSegment


def derive_subgoals(
    subtasks: list[SubtaskSegment],
    num_frames: int,
    source: str = "subtask_end",
) -> list[Subgoal]:
    """Pick one subgoal frame index per subtask. Only ``subtask_end`` is built-in."""
    if source != "subtask_end":
        raise ValueError(f"Unknown subgoal source {source!r}; supported: 'subtask_end'.")
    last = max(0, num_frames - 1)
    return [
        Subgoal(segment_idx=seg.segment_idx, frame_idx=min(max(seg.end_frame, 0), last))
        for seg in subtasks
    ]


def extract_subgoal_images(episode: Episode, subgoals: list[Subgoal], out_dir: str | Path) -> list[Subgoal]:
    """Write each subgoal frame to ``out_dir`` as PNG; set ``image_path`` in place."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for sg in subgoals:
        frame = episode.frame(sg.frame_idx)
        path = folder / f"{episode.episode_id}_seg{sg.segment_idx}_f{sg.frame_idx}.png"
        Image.fromarray(frame).save(path)
        sg.image_path = str(path)
    return subgoals
