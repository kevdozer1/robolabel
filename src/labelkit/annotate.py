"""Orchestration: run every labeler over an :class:`EpisodeSource`.

Produces a list of :class:`~labelkit.schema.EpisodeAnnotation` and writes the
``annotations.parquet`` sidecar. Each episode's raw provider receipts and
extracted subgoal frames are written under ``out_dir``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .episode import Episode, EpisodeSource
from .labelers.metadata import label_metadata
from .labelers.subgoals import derive_subgoals, extract_subgoal_images
from .labelers.subtasks import label_subtasks
from .providers.base import VLMProvider
from .rubric import Rubric, load_rubric
from .schema import EpisodeAnnotation, write_annotations


def annotate_episode(
    episode: Episode,
    provider: VLMProvider,
    rubric: Rubric,
    out_dir: Path,
    *,
    extract_images: bool = True,
) -> EpisodeAnnotation:
    """Annotate a single episode: subtasks, metadata, subgoals."""
    receipt_dir = out_dir / "raw_receipts" / episode.episode_id
    subtasks = label_subtasks(episode, provider, rubric, receipt_dir)
    metadata = label_metadata(episode, provider, rubric, receipt_dir)
    subgoals = derive_subgoals(subtasks.segments, episode.num_frames, rubric.subgoal_source)
    if extract_images and subgoals:
        extract_subgoal_images(episode, subgoals, out_dir / "subgoal_frames")

    calls = [*subtasks.calls, *metadata.calls]
    costs = [c.estimated_cost_usd for c in calls if c.estimated_cost_usd is not None]
    return EpisodeAnnotation(
        episode_id=episode.episode_id,
        task=episode.task,
        num_frames=episode.num_frames,
        fps=episode.fps,
        provider=provider.name,
        model=provider.model,
        metadata=metadata.metadata,
        subtasks=subtasks.segments,
        subgoals=subgoals,
        cost_usd=round(sum(costs), 8) if costs else None,
        receipts=[str(receipt_dir)],
    )


def annotate_source(
    source: EpisodeSource,
    out_dir: str | Path,
    *,
    provider: VLMProvider,
    rubric: Rubric | None = None,
    extract_images: bool = True,
    limit: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[EpisodeAnnotation]:
    """Annotate every episode in ``source`` and write ``annotations.parquet``.

    Args:
        source: The dataset adapter.
        out_dir: Output directory for the sidecar, receipts, and subgoal frames.
        provider: A constructed VLM provider.
        rubric: Rubric (defaults to the bundled one).
        extract_images: Write subgoal frames as PNGs.
        limit: Annotate at most this many episodes (None = all).
        progress: Optional callback ``(index, total, episode_id)``.
    """
    rubric = rubric or load_rubric()
    out = Path(out_dir)
    total = len(source) if limit is None else min(limit, len(source))
    annotations: list[EpisodeAnnotation] = []
    for i, episode in enumerate(source):
        if limit is not None and i >= limit:
            break
        if progress is not None:
            progress(i, total, episode.episode_id)
        annotations.append(annotate_episode(episode, provider, rubric, out, extract_images=extract_images))
    write_annotations(annotations, out)
    return annotations
