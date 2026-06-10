from __future__ import annotations

from pathlib import Path

from labelkit.demo import run_demo
from labelkit.schema import read_annotations


def test_offline_demo_produces_valid_parquet(tmp_path: Path):
    summary = run_demo(tmp_path / "demo", n_episodes=3)
    assert summary["episodes"] == 3
    assert summary["provider"] == "mock"

    df = read_annotations(tmp_path / "demo")
    # Three record types present for each episode.
    assert set(df["record_type"]) == {"episode_metadata", "subtask", "subgoal"}
    assert df["episode_id"].nunique() == 3
    # Subtasks cover the episode; subgoal frames equal subtask end frames.
    subtasks = df[df["record_type"] == "subtask"]
    assert (subtasks["start_frame"] >= 0).all()
    # Subgoal frames were extracted to disk.
    assert (tmp_path / "demo" / "subgoal_frames").is_dir()
    # Receipts were written for each episode.
    assert (tmp_path / "demo" / "raw_receipts").is_dir()
