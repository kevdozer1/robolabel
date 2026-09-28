"""Tests for robolabel query (phase retrieval + needs_review)."""

from __future__ import annotations

from pathlib import Path

from robolabel.query import find_phase_segments, needs_review_episodes
from robolabel.schema import EpisodeAnnotation, EpisodeMetadata, SubtaskSegment, write_annotations


def _ann(eid, quality, phases):
    return EpisodeAnnotation(
        episode_id=eid, task="t", num_frames=100, fps=10.0, provider="mock", model="m", strategy="S2",
        metadata=EpisodeMetadata(quality=quality, task_success_quality=quality, mistake=False, reason="ok"),
        subtasks=[SubtaskSegment(i, i * 25, i * 25 + 24, f"{p} obj", phase=p, evidence=f"{p} ev")
                  for i, p in enumerate(phases)],
    )


def test_find_phase_segments(tmp_path: Path):
    write_annotations([_ann("0", 5, ["approach", "grasp", "transport", "retract"]),
                       _ann("1", 5, ["approach", "grasp", "release-place", "retract"])], tmp_path)
    grasps = find_phase_segments(tmp_path, "grasp")
    assert len(grasps) == 2                       # one grasp per episode
    assert {h["episode_id"] for h in grasps} == {"0", "1"}
    assert all(h["mid_frame"] == (h["start_frame"] + h["end_frame"]) // 2 for h in grasps)
    assert find_phase_segments(tmp_path, "GRASP") == grasps  # case-insensitive


def test_needs_review_episodes_worst_first(tmp_path: Path):
    spans = ["approach", "grasp", "transport", "release-place"]
    anns = [_ann(f"e{i}", 5, spans) for i in range(8)]
    anns.append(_ann("bad", 1, spans))            # hallucinated low score vs median 5
    write_annotations(anns, tmp_path)
    rows = needs_review_episodes(tmp_path)
    assert rows and rows[0]["episode_id"] == "bad"
    assert rows[0]["quality"] == 1
