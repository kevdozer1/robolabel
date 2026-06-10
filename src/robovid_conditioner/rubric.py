"""Rubric loading. The rubric is config, not code.

Every prompt template, score definition, and gate threshold lives in
``rubric.yaml``. :func:`load_rubric` returns a :class:`Rubric` that the labelers
and the gate consult; nothing scoring-related is hardcoded in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Rubric:
    """A parsed rubric. Wraps the raw mapping with typed accessors."""

    data: dict[str, Any]
    source: str

    # --- top level -------------------------------------------------------- #
    @property
    def name(self) -> str:
        return str(self.data.get("name", "unnamed"))

    @property
    def schema_version(self) -> str:
        return str(self.data.get("schema_version", "robovid_conditioner/rubric/v1"))

    @property
    def keyframes(self) -> int:
        return int(self.data.get("keyframes", 6))

    # --- subtasks --------------------------------------------------------- #
    @property
    def subtask_min_segments(self) -> int:
        return int(self.data.get("subtasks", {}).get("min_segments", 2))

    @property
    def subtask_max_segments(self) -> int:
        return int(self.data.get("subtasks", {}).get("max_segments", 5))

    def subtask_observe_prompt(self, *, task: str, last_frame: int) -> str:
        return _fill(self.data["subtasks"]["observe_prompt"], task=task, last_frame=last_frame)

    def subtask_label_prompt(self, *, task: str, last_frame: int, observations: str) -> str:
        return _fill(
            self.data["subtasks"]["label_prompt"],
            task=task,
            last_frame=last_frame,
            observations=observations,
            min_segments=self.subtask_min_segments,
            max_segments=self.subtask_max_segments,
        )

    # --- metadata --------------------------------------------------------- #
    @property
    def quality_scale(self) -> dict[int, str]:
        raw = self.data.get("metadata", {}).get("quality_scale", {})
        return {int(k): str(v) for k, v in raw.items()}

    def quality_scale_text(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in sorted(self.quality_scale.items()))

    def metadata_observe_prompt(self, *, task: str, num_frames: int) -> str:
        return _fill(self.data["metadata"]["observe_prompt"], task=task, num_frames=num_frames)

    def metadata_label_prompt(self, *, task: str, num_frames: int, observations: str) -> str:
        return _fill(
            self.data["metadata"]["label_prompt"],
            task=task, num_frames=num_frames, observations=observations,
            quality_scale=self.quality_scale_text(),
        )

    # --- subgoals --------------------------------------------------------- #
    @property
    def subgoal_source(self) -> str:
        return str(self.data.get("subgoals", {}).get("source", "subtask_end"))

    # --- gate ------------------------------------------------------------- #
    @property
    def gate(self) -> dict[str, Any]:
        return dict(self.data.get("gate", {}))


def _fill(template: str, **values: object) -> str:
    """Substitute ``{key}`` placeholders without touching literal JSON braces.

    The prompts contain literal ``{...}`` JSON examples, so ``str.format`` cannot
    be used. Only the explicitly provided keys are replaced.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def load_rubric(path: str | Path | None = None) -> Rubric:
    """Load a rubric from ``path`` or the bundled default."""
    if path is None:
        text = resources.files("robovid_conditioner").joinpath("rubric.yaml").read_text(encoding="utf-8")
        return Rubric(data=yaml.safe_load(text), source="bundled:rubric.yaml")
    p = Path(path)
    return Rubric(data=yaml.safe_load(p.read_text(encoding="utf-8")), source=str(p))
