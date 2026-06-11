"""Export our sidecar subtask annotations into the LeRobot subtask convention.

This targets the convention the **pinned lerobot (0.4.x)** actually reads back, so a
round-trip resolves through the real API. Verified against the installed source:

* ``meta/subtasks.parquet`` — a **string-indexed** table of unique subtask phrases
  with a single ``subtask_index`` column, mirroring ``meta/tasks.parquet`` exactly.
  ``LeRobotDataset.__getitem__`` resolves a frame's subtask with
  ``self.meta.subtasks.iloc[subtask_index].name`` (lerobot_dataset.py), i.e. the
  table's *index* is the subtask string and ``subtask_index`` is its row position.
* a per-frame ``subtask_index`` (parallel to the existing per-frame ``task_index``).
  We materialise this as a compact per-episode boundary table,
  ``meta/episodes_subtasks.parquet`` (one row per episode: the ordered
  ``subtask_index`` / ``start_frame`` / ``end_frame`` / ``subtask_text`` lists, plus
  the SARM-style ``subtask_start_frames`` / ``subtask_end_frames`` / ``subtask_names``
  columns), from which the dense per-frame ``subtask_index`` column is reconstructable
  by :func:`frame_subtask_indices`. We do **not** rewrite the dataset's binary
  ``data/`` parquet files — the export is a non-destructive metadata overlay.

What survives export: the subtask temporal boundaries and the subtask phrase. What
stays **sidecar-only** (the LeRobot subtask convention has no slot for it): the
per-boundary ``evidence`` string, the closed-vocabulary ``phase`` tag, the
episode ``quality`` / ``mistake`` scores, provider receipts, and cost.

High-level tasks (``meta/tasks_high_level.parquet`` / ``task_index_high_level``) are a
**newer-LeRobot** concept that does **not** exist in the pinned version, so they are
intentionally not emitted; see ``SCHEMA.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .schema import episode_records, list_episode_ids, read_annotations

SUBTASKS_REL_PATH = "meta/subtasks.parquet"
EPISODES_SUBTASKS_REL_PATH = "meta/episodes_subtasks.parquet"


def build_subtask_vocabulary(df: pd.DataFrame, subtask_field: str = "subtask_text") -> pd.DataFrame:
    """Unique subtask phrases as a string-indexed table, exactly like meta/tasks.parquet.

    Index = the subtask string; one column ``subtask_index`` = 0-based row position.
    First-seen order is preserved so the mapping is deterministic.
    """
    seen: list[str] = []
    for episode_id in list_episode_ids(df):
        for s in episode_records(df, episode_id)["subtasks"]:
            text = _phrase(s, subtask_field)
            if text and text not in seen:
                seen.append(text)
    table = pd.DataFrame({"subtask_index": range(len(seen))}, index=pd.Index(seen, name=None))
    return table


def frame_subtask_indices(subtasks: list[dict], num_frames: int, vocab: pd.DataFrame,
                          subtask_field: str = "subtask_text") -> list[int]:
    """Dense per-frame ``subtask_index`` for one episode (length ``num_frames``).

    Each frame is labelled with the index of the subtask segment it falls in. This is
    the column LeRobot would carry per frame; we reconstruct it on demand rather than
    rewriting the data parquet.
    """
    idx_of = {str(k): int(v) for k, v in vocab["subtask_index"].items()}
    out = [0] * max(0, num_frames)
    for s in subtasks:
        text = _phrase(s, subtask_field)
        si = idx_of.get(text)
        if si is None:
            continue
        start = max(0, int(s.get("start_frame") or 0))
        end = min(num_frames - 1, int(s.get("end_frame") if s.get("end_frame") is not None else num_frames - 1))
        for f in range(start, end + 1):
            out[f] = si
    return out


def export_lerobot_subtasks(annotations_dir: str | Path, out_dir: str | Path,
                            subtask_field: str = "subtask_text") -> dict[str, Any]:
    """Write the LeRobot subtask-convention overlay from our sidecar.

    Returns a manifest describing what was written and which fields were dropped.
    """
    df = read_annotations(annotations_dir)
    out = Path(out_dir)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    vocab = build_subtask_vocabulary(df, subtask_field)
    vocab.to_parquet(out / SUBTASKS_REL_PATH)

    rows: list[dict[str, Any]] = []
    idx_of = {str(k): int(v) for k, v in vocab["subtask_index"].items()}
    for episode_id in list_episode_ids(df):
        rec = episode_records(df, episode_id)
        segs = rec["subtasks"]
        names = [_phrase(s, subtask_field) for s in segs]
        start_frames = [int(s.get("start_frame") or 0) for s in segs]
        end_frames = [int(s.get("end_frame") or 0) for s in segs]
        fps = _fps_for(df, episode_id)
        rows.append({
            "episode_index": int(episode_id) if str(episode_id).isdigit() else episode_id,
            "episode_id": str(episode_id),
            "subtask_indices": [idx_of.get(n, 0) for n in names],
            "subtask_names": names,                                   # SARM-compatible
            "subtask_start_frames": start_frames,                    # SARM-compatible
            "subtask_end_frames": end_frames,                        # SARM-compatible
            "subtask_start_times": [round(f / fps, 4) for f in start_frames],
            "subtask_end_times": [round(f / fps, 4) for f in end_frames],
        })
    episodes_subtasks = pd.DataFrame(rows)
    episodes_subtasks.to_parquet(out / EPISODES_SUBTASKS_REL_PATH, index=False)

    return {
        "out_dir": str(out),
        "files": [SUBTASKS_REL_PATH, EPISODES_SUBTASKS_REL_PATH],
        "n_episodes": len(rows),
        "n_subtasks_vocab": int(len(vocab)),
        "subtask_field": subtask_field,
        "fields_exported": ["subtask boundaries (start/end frame)", "subtask phrase (as the subtask vocabulary)"],
        "fields_sidecar_only": ["boundary_evidence", "phase", "quality", "task_success_quality",
                                "mistake", "reason", "receipts", "cost_usd"],
        "not_emitted": ["meta/tasks_high_level.parquet", "task_index_high_level (not in pinned lerobot)"],
    }


def _phrase(subtask: dict, subtask_field: str) -> str:
    val = subtask.get(subtask_field)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        val = subtask.get("subtask_text") or subtask.get("phase")
    return str(val or "").strip()


def _fps_for(df: pd.DataFrame, episode_id: str) -> float:
    ep = df[df["episode_id"].astype(str) == str(episode_id)]
    try:
        return float(ep["fps"].iloc[0]) or 30.0
    except (IndexError, ValueError, TypeError):
        return 30.0
