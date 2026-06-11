"""Offline demo: synthetic episodes + mock provider, end to end.

``robolabel demo`` runs entirely offline with no API key. It generates tiny
synthetic episodes (procedurally — no real dataset files are bundled), annotates
them with the meaningless :class:`MockProvider`, writes a valid
``annotations.parquet``, and runs the gate. It exercises the whole pipeline shape
in seconds; it proves nothing about label quality.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .episode import Episode, EpisodeSource

_TASKS = [
    "put the red block in the bowl",
    "stack the green cube on the blue cube",
    "move the yellow cup to the left",
    "place the spoon on the plate",
]


class InMemorySource(EpisodeSource):
    """An :class:`EpisodeSource` over a fixed list of episodes (tests/demo)."""

    name = "in_memory"

    def __init__(self, episodes: list[Episode]):
        self._episodes = episodes

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)


def synthetic_episode(index: int, num_frames: int = 24, fps: float = 10.0) -> Episode:
    """Make one deterministic synthetic episode (a colored square that moves).

    A gripper-like action channel toggles once, so keyframe selection and subgoal
    derivation have something realistic to bite on.
    """
    task = _TASKS[index % len(_TASKS)]
    color = np.array([(index * 53) % 255, (index * 97) % 255, (index * 151) % 255], dtype=np.uint8)
    h = w = 96

    def get_frame(i: int) -> np.ndarray:
        frame = np.full((h, w, 3), 32, dtype=np.uint8)
        # background gradient (deterministic, no randomness)
        frame[:, :, 1] = (np.linspace(20, 90, w).astype(np.uint8))[None, :]
        x = int(8 + (w - 24) * (i / max(1, num_frames - 1)))
        y = int(h // 2 - 8)
        frame[y:y + 16, x:x + 16] = color
        return frame

    actions = np.zeros((num_frames, 7), dtype=np.float32)
    actions[num_frames // 2:, -1] = 1.0  # gripper closes halfway through
    return Episode(
        episode_id=f"synthetic_{index:03d}",
        num_frames=num_frames,
        fps=fps,
        task=task,
        get_frame=get_frame,
        actions=actions,
    )


def synthetic_source(n_episodes: int = 3) -> InMemorySource:
    return InMemorySource([synthetic_episode(i) for i in range(n_episodes)])


def run_demo(out_dir: str | Path, n_episodes: int = 3) -> dict[str, Any]:
    """Run the offline pipeline end to end and return a small summary."""
    from .annotate import annotate_source
    from .gate import run_gate
    from .providers.base import build_provider
    from .rubric import load_rubric
    from .schema import read_annotations

    out = Path(out_dir)
    provider = build_provider("mock")
    rubric = load_rubric()
    annotations = annotate_source(synthetic_source(n_episodes), out, provider=provider, rubric=rubric)
    df = read_annotations(out)
    gate = run_gate(out, rubric=rubric)
    return {
        "out_dir": str(out),
        "annotations_parquet": str(out / "annotations.parquet"),
        "episodes": len(annotations),
        "rows": int(len(df)),
        "subtasks": int((df["record_type"] == "subtask").sum()),
        "subgoals": int((df["record_type"] == "subgoal").sum()),
        "gate_passed": gate.passed,
        "gate_issue_count": len(gate.issues),
        "provider": provider.name,
        "note": "Mock provider — labels are structurally valid and semantically meaningless.",
    }
