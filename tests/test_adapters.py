from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robolabel.adapters import DirectoryAdapter, build_source
from robolabel.adapters.lerobot import LeRobotAdapter


def _write_frame_episodes(root: Path, n_eps: int = 2, n_frames: int = 10) -> None:
    for ep in range(n_eps):
        d = root / f"ep_{ep:03d}"
        d.mkdir(parents=True)
        for i in range(n_frames):
            Image.fromarray(np.full((32, 32, 3), (ep * 30 + i) % 255, dtype=np.uint8)).save(d / f"{i:04d}.png")


def test_directory_adapter_frame_dirs(tmp_path: Path):
    _write_frame_episodes(tmp_path)
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps({"episode_id": "ep_000", "task": "put cube in box", "fps": 12}) + "\n", encoding="utf-8"
    )
    src = DirectoryAdapter(tmp_path)
    episodes = list(src)
    assert len(src) == 2
    assert episodes[0].episode_id == "ep_000"
    assert episodes[0].task == "put cube in box"
    assert episodes[0].fps == 12.0
    assert episodes[0].num_frames == 10
    frame = episodes[0].frame(0)
    assert frame.shape == (32, 32, 3) and frame.dtype.name == "uint8"


def test_directory_adapter_empty_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No episodes"):
        DirectoryAdapter(tmp_path)


def test_build_source_dispatch(tmp_path: Path):
    _write_frame_episodes(tmp_path, n_eps=1)
    src = build_source("directory", str(tmp_path))
    assert isinstance(src, DirectoryAdapter)
    with pytest.raises(ValueError, match="Unknown source"):
        build_source("nope", "x")


def test_lerobot_adapter_imports_without_dataset():
    # The module imports cleanly even though constructing one needs a real dataset.
    assert LeRobotAdapter.name == "lerobot"
