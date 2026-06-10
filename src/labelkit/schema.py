"""The annotations sidecar: a deterministic, versioned ``annotations.parquet``.

This replaces the monorepo's content-addressed snapshot store. One long-format
parquet holds three record types per episode — ``episode_metadata``, ``subtask``,
and ``subgoal`` — keyed by ``episode_id``. The schema is documented in
``SCHEMA.md`` and versioned by :data:`labelkit.SCHEMA_VERSION`.

Human (calibration) labels are never written here; they live in a separate gold
file (see :mod:`labelkit.gold`). This module only reads/writes VLM output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import SCHEMA_VERSION

ANNOTATIONS_FILENAME = "annotations.parquet"

# Stable column order for the long-format sidecar.
COLUMNS: list[str] = [
    "schema_version", "source", "episode_id", "task", "num_frames", "fps",
    "record_type", "segment_idx", "start_frame", "end_frame", "subtask_text",
    "quality", "task_success_quality", "mistake", "boundary_clarity", "control_mode", "reason",
    "subgoal_frame_idx", "subgoal_image_path",
    "provider", "model", "cost_usd", "receipt_path",
]


@dataclass
class SubtaskSegment:
    segment_idx: int
    start_frame: int
    end_frame: int
    subtask_text: str


@dataclass
class EpisodeMetadata:
    quality: int | None = None              # curation / training-usefulness, 1-5
    task_success_quality: int | None = None
    mistake: bool | None = None
    boundary_clarity: str | None = None
    control_mode: str | None = None         # strategy metadata, e.g. end_effector
    reason: str = ""


@dataclass
class Subgoal:
    segment_idx: int
    frame_idx: int
    image_path: str | None = None


@dataclass
class EpisodeAnnotation:
    """All VLM annotations for one episode."""

    episode_id: str
    task: str | None
    num_frames: int
    fps: float
    provider: str
    model: str
    metadata: EpisodeMetadata | None = None
    subtasks: list[SubtaskSegment] = field(default_factory=list)
    subgoals: list[Subgoal] = field(default_factory=list)
    cost_usd: float | None = None
    receipts: list[str] = field(default_factory=list)

    def to_rows(self) -> list[dict[str, Any]]:
        base = {
            "schema_version": SCHEMA_VERSION, "source": "vlm",
            "episode_id": self.episode_id, "task": self.task,
            "num_frames": int(self.num_frames), "fps": float(self.fps),
            "provider": self.provider, "model": self.model,
        }
        receipt = self.receipts[0] if self.receipts else None
        rows: list[dict[str, Any]] = []
        if self.metadata is not None:
            m = self.metadata
            rows.append({**base, "record_type": "episode_metadata",
                         "quality": m.quality, "task_success_quality": m.task_success_quality,
                         "mistake": m.mistake, "boundary_clarity": m.boundary_clarity,
                         "control_mode": m.control_mode, "reason": m.reason,
                         "cost_usd": self.cost_usd, "receipt_path": receipt})
        for seg in self.subtasks:
            rows.append({**base, "record_type": "subtask", "segment_idx": seg.segment_idx,
                         "start_frame": seg.start_frame, "end_frame": seg.end_frame,
                         "subtask_text": seg.subtask_text})
        for sg in self.subgoals:
            rows.append({**base, "record_type": "subgoal", "segment_idx": sg.segment_idx,
                         "subgoal_frame_idx": sg.frame_idx, "subgoal_image_path": sg.image_path})
        return rows


def to_dataframe(annotations: list[EpisodeAnnotation]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ann in annotations:
        rows.extend(ann.to_rows())
    frame = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    # Deterministic row order: episode, then record type, then segment.
    record_order = {"episode_metadata": 0, "subtask": 1, "subgoal": 2}
    frame["_ro"] = frame["record_type"].map(record_order).fillna(9)
    frame["_seg"] = pd.to_numeric(frame["segment_idx"], errors="coerce").fillna(-1)
    frame = frame.sort_values(["episode_id", "_ro", "_seg"]).drop(columns=["_ro", "_seg"])
    return frame[COLUMNS].reset_index(drop=True)


def write_annotations(annotations: list[EpisodeAnnotation], out_dir: str | Path) -> Path:
    """Write ``annotations.parquet`` under ``out_dir`` and return its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ANNOTATIONS_FILENAME
    to_dataframe(annotations).to_parquet(path, index=False)
    return path


def read_annotations(path: str | Path) -> pd.DataFrame:
    """Read an annotations sidecar (file or directory containing it)."""
    p = Path(path)
    if p.is_dir():
        p = p / ANNOTATIONS_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"No annotations parquet at {p}")
    return pd.read_parquet(p)


# --------------------------------------------------------------------------- #
# Convenience views for reliability / gate / gold.
# --------------------------------------------------------------------------- #
def episode_records(df: pd.DataFrame, episode_id: str) -> dict[str, Any]:
    """Return ``{metadata, subtasks, subgoals, task, num_frames}`` for one episode."""
    ep = df[df["episode_id"].astype(str) == str(episode_id)]
    meta_rows = ep[ep["record_type"] == "episode_metadata"]
    metadata = meta_rows.iloc[0].to_dict() if not meta_rows.empty else {}
    subtasks = (
        ep[ep["record_type"] == "subtask"]
        .sort_values("segment_idx")
        [["segment_idx", "start_frame", "end_frame", "subtask_text"]]
        .to_dict("records")
    )
    subgoals = (
        ep[ep["record_type"] == "subgoal"]
        .sort_values("segment_idx")
        [["segment_idx", "subgoal_frame_idx", "subgoal_image_path"]]
        .to_dict("records")
    )
    num_frames = int(ep["num_frames"].iloc[0]) if not ep.empty else 0
    task = ep["task"].iloc[0] if not ep.empty else None
    return {"metadata": metadata, "subtasks": subtasks, "subgoals": subgoals,
            "task": task, "num_frames": num_frames}


def list_episode_ids(df: pd.DataFrame) -> list[str]:
    return [str(x) for x in df["episode_id"].drop_duplicates().tolist()]


def export_jsonl(annotations_dir: str | Path, out_path: str | Path) -> Path:
    """Export the sidecar as one consolidated JSON object per episode (JSONL).

    A portable, human-readable view: each line is an episode with its metadata,
    ordered subtasks, and subgoal frames. Round-trippable into other tools without
    a parquet reader.
    """
    import json

    df = read_annotations(annotations_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for episode_id in list_episode_ids(df):
        rec = episode_records(df, episode_id)
        meta = rec["metadata"]
        record = {
            "episode_id": episode_id,
            "task": rec["task"],
            "num_frames": rec["num_frames"],
            "metadata": {
                "quality": _opt_int(meta.get("quality")),
                "task_success_quality": _opt_int(meta.get("task_success_quality")),
                "mistake": _opt_bool(meta.get("mistake")),
                "boundary_clarity": meta.get("boundary_clarity"),
                "control_mode": meta.get("control_mode"),
                "reason": meta.get("reason"),
            },
            "subtasks": [
                {"segment_idx": _opt_int(s["segment_idx"]), "start_frame": _opt_int(s["start_frame"]),
                 "end_frame": _opt_int(s["end_frame"]), "subtask_text": s["subtask_text"]}
                for s in rec["subtasks"]
            ],
            "subgoals": [
                {"segment_idx": _opt_int(s["segment_idx"]), "frame_idx": _opt_int(s["subgoal_frame_idx"]),
                 "image_path": s["subgoal_image_path"]}
                for s in rec["subgoals"]
            ],
        }
        lines.append(json.dumps(record, sort_keys=True))
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def _opt_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return bool(value)
