"""The annotations sidecar: a deterministic, versioned ``annotations.parquet``.

This replaces the monorepo's content-addressed snapshot store. One long-format
parquet holds three record types per episode — ``episode_metadata``, ``subtask``,
and ``subgoal`` — keyed by ``episode_id``. The schema is documented in
``SCHEMA.md`` and versioned by :data:`robolabel.SCHEMA_VERSION`.

Human (calibration) labels are never written here; they live in a separate gold
file (see :mod:`robolabel.gold`). This module only reads/writes VLM output.
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
    "phase", "target", "boundary_evidence", "active_dof",
    "quality", "task_success_quality", "mistake", "boundary_clarity", "control_mode",
    "control_modality", "reason",
    "speed", "speed_norm", "novelty", "curation_value", "curation_tier",
    "active_frames", "active_seconds", "active_fraction",
    "subgoal_frame_idx", "subgoal_image_path",
    "retrieved_subgoal_episode_id", "retrieved_subgoal_frame_idx",
    "provider", "model", "strategy", "cost_usd", "receipt_path",
]


@dataclass
class SubtaskSegment:
    segment_idx: int
    start_frame: int
    end_frame: int
    subtask_text: str
    # v2+, populated only by grounded strategies (S1+). Baseline S0 leaves them None.
    phase: str | None = None       # closed-vocabulary phase label (S2+)
    evidence: str | None = None    # one-line visual evidence for the boundary (S1+)
    target: str | None = None      # v3: grounded object/destination ("red cube"); "phase -> target"
    active_dof: str | None = None  # v4: set of moving component groups, '+'-joined ('arm', 'gripper', 'arm+gripper', 'none'); deterministic smoothed-excursion


@dataclass
class EpisodeMetadata:
    quality: int | None = None              # curation / training-usefulness, 1-5
    task_success_quality: int | None = None
    mistake: bool | None = None
    boundary_clarity: str | None = None
    control_mode: str | None = None         # legacy strategy metadata (superseded by control_modality)
    control_modality: str | None = None     # v4: 'joint'|'end-effector' — the action COORDINATE FRAME (deterministic)
    reason: str = ""
    # v5: deterministic metadata + curation signals (no VLM).
    speed: str | None = None                # 'fast'|'medium'|'slow' tier — corpus-relative; null if insufficient population
    speed_norm: float | None = None         # raw scalar: mean per-step action velocity
    novelty: float | None = None            # raw: distance to nearest neighbours (corpus-pooled); higher = more novel
    curation_value: float | None = None     # raw: f(quality, novelty); weights in the run config
    curation_tier: str | None = None        # 'full'|'reduced'|'minimal'/'keep'/'cut'; null if insufficient population
    # v6: continuous, phase-agnostic motion descriptor (the primary speed signal).
    active_frames: int | None = None        # frames from motion onset to offset (motion-defined, not phase-tied)
    active_seconds: float | None = None      # active_frames / fps
    active_fraction: float | None = None     # active_frames / num_frames


@dataclass
class Subgoal:
    segment_idx: int
    frame_idx: int                          # the REAL same-episode end-of-sub-step keyframe (ground truth)
    image_path: str | None = None
    # v4: an OPTIONAL retrieved subgoal — the same-phase end frame from a DIFFERENT episode.
    # Stored alongside the real keyframe (never replaces it) to support copy-shortcut-free
    # policy training/eval. None when no same-phase candidate exists or retrieval wasn't run.
    retrieved_episode_id: str | None = None
    retrieved_frame_idx: int | None = None


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
    strategy: str | None = None     # annotation strategy name (S0..S4); None == baseline

    def to_rows(self) -> list[dict[str, Any]]:
        base = {
            "schema_version": SCHEMA_VERSION, "source": "vlm",
            "episode_id": self.episode_id, "task": self.task,
            "num_frames": int(self.num_frames), "fps": float(self.fps),
            "provider": self.provider, "model": self.model,
            "strategy": self.strategy,
        }
        receipt = self.receipts[0] if self.receipts else None
        rows: list[dict[str, Any]] = []
        if self.metadata is not None:
            m = self.metadata
            rows.append({**base, "record_type": "episode_metadata",
                         "quality": m.quality, "task_success_quality": m.task_success_quality,
                         "mistake": m.mistake, "boundary_clarity": m.boundary_clarity,
                         "control_mode": m.control_mode, "control_modality": m.control_modality,
                         "reason": m.reason, "speed": m.speed, "speed_norm": m.speed_norm,
                         "novelty": m.novelty, "curation_value": m.curation_value,
                         "curation_tier": m.curation_tier, "active_frames": m.active_frames,
                         "active_seconds": m.active_seconds, "active_fraction": m.active_fraction,
                         "cost_usd": self.cost_usd, "receipt_path": receipt})
        for seg in self.subtasks:
            rows.append({**base, "record_type": "subtask", "segment_idx": seg.segment_idx,
                         "start_frame": seg.start_frame, "end_frame": seg.end_frame,
                         "subtask_text": seg.subtask_text, "phase": seg.phase, "target": seg.target,
                         "boundary_evidence": seg.evidence, "active_dof": seg.active_dof})
        for sg in self.subgoals:
            rows.append({**base, "record_type": "subgoal", "segment_idx": sg.segment_idx,
                         "subgoal_frame_idx": sg.frame_idx, "subgoal_image_path": sg.image_path,
                         "retrieved_subgoal_episode_id": sg.retrieved_episode_id,
                         "retrieved_subgoal_frame_idx": sg.retrieved_frame_idx})
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


def save_dataframe(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Write an already-built sidecar DataFrame back to ``annotations.parquet``.

    Used by enrichment passes (``robolabel enrich``) that add columns to an existing sidecar
    without rebuilding the annotation objects. Missing schema columns are added as null.
    """
    out = Path(out_dir)
    path = out / ANNOTATIONS_FILENAME if out.is_dir() or not out.suffix else out
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = df.copy()
    for col in COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    frame[COLUMNS].to_parquet(path, index=False)
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
    subtask_cols = ["segment_idx", "start_frame", "end_frame", "subtask_text"]
    for optional in ("phase", "target", "boundary_evidence", "active_dof"):  # v2/v3/v4; absent in older files
        if optional in ep.columns:
            subtask_cols.append(optional)
    subtasks = (
        ep[ep["record_type"] == "subtask"]
        .sort_values("segment_idx")
        [subtask_cols]
        .to_dict("records")
    )
    subgoal_cols = ["segment_idx", "subgoal_frame_idx", "subgoal_image_path"]
    for optional in ("retrieved_subgoal_episode_id", "retrieved_subgoal_frame_idx"):  # v4
        if optional in ep.columns:
            subgoal_cols.append(optional)
    subgoals = (
        ep[ep["record_type"] == "subgoal"]
        .sort_values("segment_idx")
        [subgoal_cols]
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
                "control_modality": meta.get("control_modality"),
                "reason": meta.get("reason"),
            },
            "subtasks": [
                {"segment_idx": _opt_int(s["segment_idx"]), "start_frame": _opt_int(s["start_frame"]),
                 "end_frame": _opt_int(s["end_frame"]), "subtask_text": s["subtask_text"],
                 "active_dof": s.get("active_dof")}
                for s in rec["subtasks"]
            ],
            "subgoals": [
                {"segment_idx": _opt_int(s["segment_idx"]), "frame_idx": _opt_int(s["subgoal_frame_idx"]),
                 "image_path": s["subgoal_image_path"],
                 "retrieved_episode_id": s.get("retrieved_subgoal_episode_id"),
                 "retrieved_frame_idx": _opt_int(s.get("retrieved_subgoal_frame_idx"))}
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
