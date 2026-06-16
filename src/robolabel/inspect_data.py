"""Assemble the data the ``robolabel inspect`` viewer renders.

A viewer payload is ``{dataset, source_kind, track_order, track_colors, episodes:[...]}``.
Each episode carries several **tracks** (gold + each strategy/baseline), where a track is
a list of ``{start,end,phase,target,text,evidence}`` segments, plus per-track **metrics**
against the gold track. Pure functions here; the heavy reconstruction lives in
``scripts/build_inspect_data.py`` and the serving in ``inspect_server.py``.
"""

from __future__ import annotations

from typing import Any

from .metrics import boundaries, boundary_pr_mae, episode_iou

TRACK_COLORS = {
    "gold": "#111827",
    "S0-Flash": "#e8752a",
    "grounded": "#2563eb",
    "S_grip": "#59a14f",
    "uniform-fifths": "#9aa5b1",
}


def _clean(v) -> str | None:
    """None for missing/empty/NaN (empty parquet columns read back as float NaN)."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None


def segments_from_records(subtasks: list[dict]) -> list[dict]:
    """Convert episode_records()/parquet subtask rows into viewer segments.

    Empty ``phase`` / ``boundary_evidence`` columns (e.g. for the baseline S0, which
    has neither) come out of parquet as float ``NaN``; coerce those to ``None`` so the
    viewer doesn't render the literal string "nan".
    """
    out = []
    for s in sorted(subtasks, key=lambda r: int(r.get("segment_idx") or 0)):
        out.append({
            "start": int(s.get("start_frame") or 0),
            "end": int(s.get("end_frame") or 0),
            "phase": _clean(s.get("phase")),
            "target": _clean(s.get("target")),
            "text": _clean(s.get("subtask_text")) or "",
            "evidence": _clean(s.get("boundary_evidence")),
        })
    return out


def episode_metrics(track_segs: list[dict], gold_segs: list[dict]) -> dict[str, Any]:
    """Per-track metrics vs gold: IoU + boundary precision/recall@±5 + MAE."""
    b = boundary_pr_mae(boundaries(track_segs), boundaries(gold_segs))
    return {
        "iou": _r(episode_iou(track_segs, gold_segs)),
        "boundary_precision": _r(b["precision"]),
        "boundary_recall": _r(b["recall"]),
        "mae": _r(b["mae"]),
        "n_segments": len(track_segs),
        "n_gold_boundaries": b["n_gold"],
    }


def _num(v) -> float | None:
    """Coerce to float; None for missing/NaN."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    try:
        return round(float(v), 5)
    except (TypeError, ValueError):
        return None


def module_block(meta: dict) -> dict[str, Any]:
    """The per-episode module fields (the run modules' outputs), cleaned for the viewer."""
    q = _num(meta.get("quality"))
    return {
        "quality": int(q) if q is not None else None,
        "speed": _clean(meta.get("speed")),
        "novelty": _num(meta.get("novelty")),
        "curation_value": _num(meta.get("curation_value")),
        "curation_tier": _clean(meta.get("curation_tier")),
        # control_modality is the action COORDINATE FRAME (joint vs end-effector), not gripper use
        "control_modality": _clean(meta.get("control_modality")),
    }


def subgoal_block(subgoals: list[dict]) -> list[dict[str, Any]]:
    """Per-segment subgoal pointers: the real same-episode frame + the retrieved cross-episode one."""
    out = []
    for s in sorted(subgoals, key=lambda r: int(r.get("segment_idx") or 0)):
        rep = _clean(s.get("retrieved_subgoal_episode_id"))
        rf = s.get("retrieved_subgoal_frame_idx")
        out.append({
            "segment_idx": int(s.get("segment_idx") or 0),
            "frame": int(s.get("subgoal_frame_idx") or 0),
            "retrieved_episode": rep,
            "retrieved_frame": int(rf) if (rf is not None and not (isinstance(rf, float) and rf != rf)) else None,
        })
    return out


def enabled_modules(mods: dict, subgoals: list[dict], segments: list[dict]) -> list[str]:
    """Infer which run modules were enabled, from which fields are populated (honest)."""
    on = []
    if segments:
        on.append("segmentation")
    if mods.get("quality") is not None:
        on.append("quality")
    if mods.get("speed"):
        on.append("speed")
    if subgoals:
        on.append("subgoals")
    if mods.get("control_modality"):
        on.append("control")
    if mods.get("novelty") is not None:
        on.append("novelty")
    if mods.get("curation_value") is not None:
        on.append("curation")
    return on


def _thumb_frame(subgoals: list[dict], num_frames: int) -> int:
    """Representative frame: first subgoal keyframe, else a mid-episode frame."""
    for s in sorted(subgoals, key=lambda r: r.get("segment_idx", 0)):
        f = s.get("frame")
        if f is not None:
            return int(f)
    return max(0, int(num_frames) // 2)


def build_episode(episode_id: str, num_frames: int, fps: float, task: str,
                  tracks: dict[str, dict], *, gold_name: str = "gold",
                  quality: dict | None = None, gate_flags: list | None = None,
                  modules: dict | None = None, subgoals: list | None = None) -> dict[str, Any]:
    gold_segs = tracks.get(gold_name, {}).get("segments", [])
    metrics = {
        name: episode_metrics(t.get("segments", []), gold_segs)
        for name, t in tracks.items() if name != gold_name
    }
    primary = next((t for n, t in tracks.items() if n != gold_name), {})
    sgs = subgoal_block(subgoals or [])
    return {
        "episode_id": str(episode_id),
        "num_frames": int(num_frames),
        "fps": float(fps),
        "task": task or "",
        "tracks": tracks,
        "metrics": metrics,
        "quality": quality or {},
        "gate_flags": gate_flags or [],
        # gallery: module outputs, subgoal pointers, representative thumbnail, enabled modules
        "modules": modules or {},
        "subgoals": sgs,
        "thumb": _thumb_frame(sgs, num_frames),
        "enabled": enabled_modules(modules or {}, sgs, primary.get("segments", [])),
        # convenience sort keys (worst grounded IoU, total gate flags, gold-vs-grounded gap)
        "sort_iou": _min_iou(metrics),
        "n_flags": len(gate_flags or []),
    }


def assemble(dataset: str, source_kind: str, track_order: list[str],
             episodes: list[dict], *, blind: bool = False) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source_kind": source_kind,
        "track_order": track_order,
        "track_colors": {t: TRACK_COLORS.get(t, "#888") for t in track_order},
        "blind": blind,
        "episodes": episodes,
    }


def _min_iou(metrics: dict) -> float:
    vals = [m["iou"] for m in metrics.values() if m.get("iou") is not None]
    return min(vals) if vals else 1.0


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)
