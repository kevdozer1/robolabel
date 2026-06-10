"""Episode quality / mistake / strategy metadata (two-stage observe -> label)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..episode import Episode
from ..providers.base import ProviderResponse, VLMProvider, try_extract_json
from ..rubric import Rubric
from ..schema import EpisodeMetadata
from . import select_keyframes


@dataclass
class MetadataResult:
    metadata: EpisodeMetadata
    observations: Any
    calls: list[ProviderResponse]
    keyframes: list[int]


def label_metadata(
    episode: Episode,
    provider: VLMProvider,
    rubric: Rubric,
    receipt_dir: Path,
) -> MetadataResult:
    """Rate an episode (quality/mistake) using the two-stage flow."""
    keyframes = select_keyframes(episode, rubric.keyframes)
    frames = episode.frames(keyframes)
    task = episode.task or episode.episode_id

    def build_label_question(observations: Any) -> str:
        obs_text = json.dumps(observations, sort_keys=True) if observations is not None else "{}"
        return rubric.metadata_label_prompt(task=task, num_frames=episode.num_frames, observations=obs_text)

    result = provider.observe_then_label(
        frames,
        keyframes,
        rubric.metadata_observe_prompt(task=task, num_frames=episode.num_frames),
        build_label_question,
        receipt_dir / "metadata_observe.json",
        receipt_dir / "metadata_label.json",
    )
    metadata = _parse_metadata(try_extract_json(result.label.answer))
    return MetadataResult(metadata, result.observations, result.all_calls, keyframes)


def _parse_metadata(data: Any) -> EpisodeMetadata:
    if not isinstance(data, dict):
        return EpisodeMetadata(reason="unparseable VLM metadata answer")
    task_success = _clamp_quality(data.get("task_success_quality", data.get("task_success_quality_1_to_5")))
    curation = _clamp_quality(data.get("curation_quality", data.get("curation_quality_1_to_5")))
    quality = curation if curation is not None else _clamp_quality(data.get("quality", data.get("quality_1_to_5")))
    mistake = data.get("mistake", data.get("mistake_boolean"))
    if isinstance(mistake, str):
        mistake = mistake.strip().lower() in {"true", "yes", "1"}
    boundary = data.get("boundary_clarity")
    return EpisodeMetadata(
        quality=quality,
        task_success_quality=task_success if task_success is not None else quality,
        mistake=bool(mistake) if mistake is not None else None,
        boundary_clarity=str(boundary).strip().lower() if boundary else None,
        control_mode=str(data.get("control_mode")).strip() if data.get("control_mode") else None,
        reason=str(data.get("reason", "")).strip(),
    )


def _clamp_quality(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return None
