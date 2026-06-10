"""Subtask temporal segmentation (two-stage observe -> label)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..episode import Episode
from ..providers.base import ProviderResponse, VLMProvider
from ..rubric import Rubric
from ..schema import SubtaskSegment
from . import select_keyframes, validate_segments


@dataclass
class SubtaskResult:
    segments: list[SubtaskSegment]
    observations: Any
    calls: list[ProviderResponse]
    keyframes: list[int]


def label_subtasks(
    episode: Episode,
    provider: VLMProvider,
    rubric: Rubric,
    receipt_dir: Path,
) -> SubtaskResult:
    """Segment an episode into ordered subtasks using the two-stage flow."""
    keyframes = select_keyframes(episode, rubric.keyframes)
    frames = episode.frames(keyframes)
    last_frame = episode.num_frames - 1
    task = episode.task or episode.episode_id

    def build_label_question(observations: Any) -> str:
        obs_text = json.dumps(observations, sort_keys=True) if observations is not None else "[]"
        return rubric.subtask_label_prompt(task=task, last_frame=last_frame, observations=obs_text)

    result = provider.observe_then_label(
        frames,
        keyframes,
        rubric.subtask_observe_prompt(task=task, last_frame=last_frame),
        build_label_question,
        receipt_dir / "subtasks_observe.json",
        receipt_dir / "subtasks_label.json",
    )
    from ..providers.base import try_extract_json

    segments = validate_segments(
        try_extract_json(result.label.answer),
        episode.num_frames,
        rubric.subtask_min_segments,
        rubric.subtask_max_segments,
    )
    return SubtaskResult(segments, result.observations, result.all_calls, keyframes)
