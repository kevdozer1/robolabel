from __future__ import annotations

from pathlib import Path

from labelkit.demo import synthetic_episode, synthetic_source
from labelkit.labelers import validate_segments
from labelkit.labelers.metadata import label_metadata
from labelkit.labelers.subgoals import derive_subgoals
from labelkit.labelers.subtasks import label_subtasks
from labelkit.providers import build_provider
from labelkit.rubric import load_rubric
from labelkit.schema import (
    EpisodeAnnotation,
    EpisodeMetadata,
    SubtaskSegment,
    episode_records,
    read_annotations,
    write_annotations,
)


def test_validate_segments_makes_contiguous_full_coverage():
    raw = {"segments": [
        {"start_step": 0, "end_step": 5, "subtask_text": "a"},
        {"start_step": 6, "end_step": 99, "subtask_text": "b"},  # end clamped to num_frames-1
    ]}
    segs = validate_segments(raw, num_frames=20, min_seg=2, max_seg=5)
    assert segs[0].start_frame == 0
    assert segs[-1].end_frame == 19
    # contiguous, non-overlapping
    for prev, nxt in zip(segs, segs[1:]):
        assert nxt.start_frame == prev.end_frame + 1


def test_validate_segments_empty_falls_back_to_single_segment():
    segs = validate_segments({"segments": []}, num_frames=10, min_seg=2, max_seg=5)
    assert len(segs) == 1
    assert segs[0].start_frame == 0 and segs[0].end_frame == 9


def test_subtask_and_metadata_labelers_with_mock(tmp_path: Path):
    rubric = load_rubric()
    provider = build_provider("mock")
    ep = synthetic_episode(0)
    sub = label_subtasks(ep, provider, rubric, tmp_path / "r")
    meta = label_metadata(ep, provider, rubric, tmp_path / "r")
    assert 2 <= len(sub.segments) <= 5
    assert meta.metadata.quality in range(1, 6)
    subgoals = derive_subgoals(sub.segments, ep.num_frames, rubric.subgoal_source)
    assert [sg.frame_idx for sg in subgoals] == [s.end_frame for s in sub.segments]


def test_annotations_parquet_roundtrip(tmp_path: Path):
    ann = EpisodeAnnotation(
        episode_id="ep0", task="put block in bowl", num_frames=12, fps=10.0,
        provider="mock", model="mock-vlm",
        metadata=EpisodeMetadata(quality=4, task_success_quality=3, mistake=False,
                                 boundary_clarity="clear", reason="clear boundaries"),
        subtasks=[SubtaskSegment(0, 0, 5, "grasp block"), SubtaskSegment(1, 6, 11, "place block")],
    )
    write_annotations([ann], tmp_path)
    df = read_annotations(tmp_path)
    rec = episode_records(df, "ep0")
    assert rec["task"] == "put block in bowl"
    assert len(rec["subtasks"]) == 2
    assert int(rec["metadata"]["quality"]) == 4


def test_write_is_deterministic(tmp_path: Path):
    src = synthetic_source(3)
    from labelkit.annotate import annotate_source

    a = annotate_source(src, tmp_path / "a", provider=build_provider("mock"))
    b = annotate_source(synthetic_source(3), tmp_path / "b", provider=build_provider("mock"))
    # Absolute output paths legitimately differ by output dir; compare logical content.
    path_cols = ["receipt_path", "subgoal_image_path"]
    da = read_annotations(tmp_path / "a").drop(columns=path_cols)
    db = read_annotations(tmp_path / "b").drop(columns=path_cols)
    assert da.equals(db)
    assert len(a) == len(b) == 3
