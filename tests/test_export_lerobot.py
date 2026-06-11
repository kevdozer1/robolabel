"""LeRobot subtask-convention export + round-trip.

The round-trip reloads ``meta/subtasks.parquet`` exactly as the pinned lerobot does
(string-indexed table; a frame's subtask is ``subtasks.iloc[subtask_index].name``) and
checks that the dense per-frame ``subtask_index`` resolves every frame to the subtask
segment it actually falls in. If lerobot is installed, it also reloads through the real
``lerobot.datasets.utils.load_subtasks`` reader.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from robolabel.export_lerobot import (
    SUBTASKS_REL_PATH,
    build_subtask_vocabulary,
    export_lerobot_subtasks,
    frame_subtask_indices,
)
from robolabel.schema import (
    EpisodeAnnotation,
    SubtaskSegment,
    episode_records,
    read_annotations,
    write_annotations,
)


def _annotations(tmp_path: Path) -> Path:
    anns = [
        EpisodeAnnotation(
            episode_id="0", task="put brick in box", num_frames=40, fps=10.0,
            provider="mock", model="m", strategy="S2",
            subtasks=[
                SubtaskSegment(0, 0, 14, "move to the brick", phase="approach", evidence="gripper above brick"),
                SubtaskSegment(1, 15, 29, "grasp the brick", phase="grasp", evidence="gripper closes"),
                SubtaskSegment(2, 30, 39, "place the brick in the box", phase="release-place", evidence="brick released"),
            ],
        ),
        EpisodeAnnotation(
            episode_id="1", task="put brick in box", num_frames=20, fps=10.0,
            provider="mock", model="m", strategy="S2",
            subtasks=[
                SubtaskSegment(0, 0, 9, "move to the brick", phase="approach", evidence="x"),  # shared phrase
                SubtaskSegment(1, 10, 19, "retract the arm", phase="retract", evidence="y"),
            ],
        ),
    ]
    write_annotations(anns, tmp_path)
    return tmp_path


def test_subtasks_parquet_mirrors_tasks_schema(tmp_path: Path):
    ann_dir = _annotations(tmp_path / "ann")
    out = export_lerobot_subtasks(ann_dir, tmp_path / "export")
    sub = pd.read_parquet(Path(out["out_dir"]) / SUBTASKS_REL_PATH)
    # Exactly like meta/tasks.parquet: string index + a single subtask_index column.
    assert list(sub.columns) == ["subtask_index"]
    assert sub.index.name is None
    assert list(sub["subtask_index"]) == list(range(len(sub)))
    # Shared phrase "move to the brick" is deduplicated across episodes.
    assert "move to the brick" in sub.index
    assert len(sub) == 4  # move/grasp/place/retract — "move" appears once
    # The lerobot resolution semantics: subtasks.iloc[idx].name is the subtask string.
    assert sub.iloc[0].name == "move to the brick"


def test_frame_subtask_index_resolves_to_correct_frames(tmp_path: Path):
    ann_dir = _annotations(tmp_path / "ann")
    out = export_lerobot_subtasks(ann_dir, tmp_path / "export")
    sub = pd.read_parquet(Path(out["out_dir"]) / SUBTASKS_REL_PATH)
    df = read_annotations(ann_dir)
    rec = episode_records(df, "0")
    per_frame = frame_subtask_indices(rec["subtasks"], rec["num_frames"], sub)
    assert len(per_frame) == 40
    # Every frame resolves (via the lerobot .iloc[idx].name rule) to the segment it's in.
    expected = {**{f: "move to the brick" for f in range(0, 15)},
                **{f: "grasp the brick" for f in range(15, 30)},
                **{f: "place the brick in the box" for f in range(30, 40)}}
    for f, idx in enumerate(per_frame):
        assert sub.iloc[idx].name == expected[f]


def test_manifest_documents_dropped_fields(tmp_path: Path):
    ann_dir = _annotations(tmp_path / "ann")
    out = export_lerobot_subtasks(ann_dir, tmp_path / "export")
    assert "boundary_evidence" in out["fields_sidecar_only"]
    assert "phase" in out["fields_sidecar_only"]
    assert any("high_level" in s for s in out["not_emitted"])
    assert out["n_episodes"] == 2


def test_export_via_phase_field(tmp_path: Path):
    ann_dir = _annotations(tmp_path / "ann")
    out = export_lerobot_subtasks(ann_dir, tmp_path / "export", subtask_field="phase")
    sub = pd.read_parquet(Path(out["out_dir"]) / SUBTASKS_REL_PATH)
    assert set(sub.index) == {"approach", "grasp", "release-place", "retract"}


def test_roundtrip_through_real_lerobot_loader_if_available(tmp_path: Path):
    """If lerobot is installed, reload meta/subtasks.parquet via its own loader."""
    load_subtasks = pytest.importorskip("lerobot.datasets.utils").load_subtasks
    ann_dir = _annotations(tmp_path / "ann")
    out = export_lerobot_subtasks(ann_dir, tmp_path / "export")
    reloaded = load_subtasks(Path(out["out_dir"]))
    assert reloaded is not None
    # Same resolution the dataset uses: iloc[idx].name -> subtask string.
    assert reloaded.iloc[0].name == "move to the brick"
    assert list(reloaded["subtask_index"]) == list(range(len(reloaded)))


def test_vocabulary_first_seen_order_is_deterministic(tmp_path: Path):
    ann_dir = _annotations(tmp_path / "ann")
    v1 = build_subtask_vocabulary(read_annotations(ann_dir))
    v2 = build_subtask_vocabulary(read_annotations(ann_dir))
    assert list(v1.index) == list(v2.index)
    assert list(v1.index)[:2] == ["move to the brick", "grasp the brick"]
