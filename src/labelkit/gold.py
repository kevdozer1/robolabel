"""Gold sets: human calibration labels, stored separately from VLM labels.

A gold file is JSON. For every episode it records the ``auto`` labels (a snapshot
of the VLM output from ``annotations.parquet``) and a ``gold`` block the human
fills in. The two are never merged into one field and the tool never overwrites
one with the other — that separation is the whole point. The reliability report
reads only the gold file (it carries both sides).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import episode_records, list_episode_ids, read_annotations

GOLD_SCHEMA_VERSION = "labelkit/gold/v1"


def build_gold_template(annotations_dir: str | Path) -> dict[str, Any]:
    """Build a gold template from an annotations sidecar (auto filled, gold empty)."""
    df = read_annotations(annotations_dir)
    episodes = []
    for episode_id in list_episode_ids(df):
        rec = episode_records(df, episode_id)
        auto_subtasks = [
            {"segment_idx": _int(s["segment_idx"]), "start_frame": _int(s["start_frame"]),
             "end_frame": _int(s["end_frame"]), "subtask_text": s["subtask_text"]}
            for s in rec["subtasks"]
        ]
        auto_subgoals = [
            {"segment_idx": _int(s["segment_idx"]), "frame_idx": _int(s["subgoal_frame_idx"])}
            for s in rec["subgoals"]
        ]
        meta = rec["metadata"]
        auto_metadata = {
            "quality": _int(meta.get("quality")),
            "task_success_quality": _int(meta.get("task_success_quality")),
            "mistake": _bool(meta.get("mistake")),
            "boundary_clarity": meta.get("boundary_clarity"),
            "reason": meta.get("reason"),
        }
        episodes.append({
            "episode_id": episode_id,
            "task": rec["task"],
            "num_frames": rec["num_frames"],
            "auto": {"subtasks": auto_subtasks, "metadata": auto_metadata, "subgoals": auto_subgoals},
            "gold": {
                "subtasks": [{"segment_idx": s["segment_idx"], "start_frame": None, "end_frame": None,
                              "subtask_text": None, "accept_auto": None} for s in auto_subtasks],
                "metadata": {"quality": None, "mistake": None, "reason": None, "accept_auto": None},
                "subgoals": [{"segment_idx": s["segment_idx"], "frame_idx": None, "accept_auto": None}
                             for s in auto_subgoals],
            },
            "review_notes": "",
        })
    return {"schema_version": GOLD_SCHEMA_VERSION, "episodes": episodes}


def load_or_sync_gold(annotations_dir: str | Path, gold_path: str | Path) -> dict[str, Any]:
    """Create the gold file if missing, else re-sync ``auto`` while keeping human edits."""
    path = Path(gold_path)
    template = build_gold_template(annotations_dir)
    if not path.exists():
        _write(path, template)
        return template
    existing = _read(path)
    by_id = {str(e["episode_id"]): e for e in existing.get("episodes", [])}
    merged = []
    for entry in template["episodes"]:
        prior = by_id.get(str(entry["episode_id"]))
        if prior is not None:
            entry["gold"] = prior.get("gold", entry["gold"])  # keep human edits
            entry["review_notes"] = prior.get("review_notes", "")
        merged.append(entry)
    template["episodes"] = merged
    _write(path, template)
    return template


def update_episode_review(
    gold_path: str | Path,
    episode_id: str,
    *,
    quality: int | None = None,
    mistake: bool | None = None,
    reason: str | None = None,
    accept_auto_metadata: bool = False,
    gold_subtasks: list[dict[str, Any]] | None = None,
    gold_subgoals: list[dict[str, Any]] | None = None,
    review_notes: str | None = None,
) -> dict[str, Any]:
    """Write one episode's human labels into the gold ``gold`` block."""
    path = Path(gold_path)
    gold = _read(path)
    entry = _entry(gold, episode_id)
    auto_meta = entry.get("auto", {}).get("metadata", {})
    gm = entry["gold"]["metadata"]
    if quality is not None:
        gm["quality"] = int(max(1, min(5, quality)))
    gm["mistake"] = bool(mistake) if mistake is not None else _bool(auto_meta.get("mistake"))
    gm["reason"] = reason if reason is not None else gm.get("reason")
    gm["accept_auto"] = bool(accept_auto_metadata)
    if gold_subtasks is not None:
        _apply_indexed(entry["gold"]["subtasks"], gold_subtasks, ("start_frame", "end_frame", "subtask_text", "accept_auto"))
    if gold_subgoals is not None:
        _apply_indexed(entry["gold"]["subgoals"], gold_subgoals, ("frame_idx", "accept_auto"))
    if review_notes is not None:
        entry["review_notes"] = review_notes
    _write(path, gold)
    return entry


def _apply_indexed(targets: list[dict], updates: list[dict], fields: tuple[str, ...]) -> None:
    by_idx = {u.get("segment_idx"): u for u in updates}
    for item in targets:
        upd = by_idx.get(item.get("segment_idx"))
        if upd is None:
            continue
        for field in fields:
            if field in upd:
                item[field] = upd[field]


def _entry(gold: dict[str, Any], episode_id: str) -> dict[str, Any]:
    for entry in gold.get("episodes", []):
        if str(entry.get("episode_id")) == str(episode_id):
            return entry
    raise KeyError(f"Episode not in gold file: {episode_id}")


def _int(value: Any) -> int | None:
    try:
        import pandas as pd
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    try:
        import pandas as pd
        if isinstance(value, float) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
