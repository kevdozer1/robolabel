"""Orchestration: run every labeler over an :class:`EpisodeSource`.

Produces a list of :class:`~robovid_conditioner.schema.EpisodeAnnotation` and writes the
``annotations.parquet`` sidecar. Each episode's raw provider receipts and
extracted subgoal frames are written under ``out_dir``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .episode import Episode, EpisodeSource
from .labelers.metadata import label_metadata
from .labelers.subgoals import derive_subgoals, extract_subgoal_images
from .labelers.subtasks import label_subtasks
from .providers.base import VLMProvider
from .rubric import Rubric, load_rubric
from .schema import (
    ANNOTATIONS_FILENAME,
    EpisodeAnnotation,
    list_episode_ids,
    read_annotations,
    to_dataframe,
)


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
    resume: bool = True,
) -> list[EpisodeAnnotation]:
    """Annotate every episode in ``source`` and write ``annotations.parquet``.

    Resilient to per-episode failures: a transient provider error (rate limit,
    503, a single unparseable answer) on one episode is recorded and skipped,
    never discarding the episodes that already succeeded. The sidecar is
    checkpointed after every episode, so a hard crash loses at most one episode.
    Re-running the same command **resumes** — episodes already in the sidecar are
    skipped and only the missing/failed ones are retried.

    Args:
        source: The dataset adapter.
        out_dir: Output directory for the sidecar, receipts, and subgoal frames.
        provider: A constructed VLM provider.
        rubric: Rubric (defaults to the bundled one).
        extract_images: Write subgoal frames as PNGs.
        limit: Annotate at most this many episodes (None = all).
        progress: Optional callback ``(index, total, episode_id)``.
        resume: Skip episodes already present in an existing sidecar.

    Returns:
        The episode annotations produced in this run (excludes resumed ones).
    """
    rubric = rubric or load_rubric()
    out = Path(out_dir)
    total = len(source) if limit is None else min(limit, len(source))

    existing_df = None
    done_ids: set[str] = set()
    if resume:
        try:
            existing_df = read_annotations(out)
            done_ids = set(list_episode_ids(existing_df))
        except FileNotFoundError:
            existing_df = None

    new_annotations: list[EpisodeAnnotation] = []
    failures: list[dict[str, str]] = []
    for i, episode in enumerate(source):
        if limit is not None and i >= limit:
            break
        if episode.episode_id in done_ids:
            continue  # resume: already annotated in a previous run
        if progress is not None:
            progress(i, total, episode.episode_id)
        try:
            ann = annotate_episode(episode, provider, rubric, out, extract_images=extract_images)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not sink the run
            failures.append({"episode_id": episode.episode_id, "error": str(exc)[:500]})
            if progress is not None:
                progress(i, total, f"{episode.episode_id} FAILED: {str(exc)[:120]}")
            continue
        new_annotations.append(ann)
        _checkpoint(out, existing_df, new_annotations)  # survive a hard crash

    _checkpoint(out, existing_df, new_annotations)  # ensure the file exists
    if failures:
        (out / "failures.json").write_text(
            json.dumps({"failed": failures, "count": len(failures)}, indent=2) + "\n", encoding="utf-8"
        )
    return new_annotations


def _checkpoint(out: Path, existing_df, new_annotations: list[EpisodeAnnotation]) -> None:
    """Write the sidecar = previously-resumed rows + this run's rows so far."""
    out.mkdir(parents=True, exist_ok=True)
    frame = to_dataframe(new_annotations)
    if existing_df is not None and not existing_df.empty:
        frame = pd.concat([existing_df, frame], ignore_index=True)
    frame.to_parquet(out / ANNOTATIONS_FILENAME, index=False)
