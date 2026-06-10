"""Reliability: how far apart are the VLM labels and the human gold labels?

Computes, over the reviewed episodes in a gold file:

* subtask boundary temporal IoU (mean over matched segments),
* quality-score exact agreement and within-one agreement,
* subgoal frame agreement.

These are the numbers that justify the tool's existence: they say how wrong the
VLM was on your data. Output is a dict (also written as JSON by the CLI).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def reliability_report(gold_path: str | Path) -> dict[str, Any]:
    """Compare ``auto`` vs ``gold`` labels in a gold file and summarize agreement."""
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    boundary_ious: list[float] = []
    exact_quality: list[bool] = []
    within_one_quality: list[bool] = []
    subgoal_matches: list[bool] = []
    reviewed = 0
    per_episode: list[dict[str, Any]] = []

    for entry in gold.get("episodes", []):
        episode_id = str(entry.get("episode_id"))
        auto = entry.get("auto", {})
        auto_subtasks = auto.get("subtasks", [])
        auto_metadata = auto.get("metadata", {})
        auto_subgoals = {_int(s.get("segment_idx")): _int(s.get("frame_idx")) for s in auto.get("subgoals", [])}

        gold_subtasks = _reviewed_subtasks(entry)
        gold_metadata = _reviewed_metadata(entry)
        gold_subgoals = _reviewed_subgoals(entry)
        if gold_subtasks or gold_metadata or gold_subgoals:
            reviewed += 1

        ep_ious = [
            iou for iou in (_temporal_iou(a, g) for a, g in zip(auto_subtasks, gold_subtasks, strict=False)) if iou is not None
        ]
        boundary_ious.extend(ep_ious)

        q_auto = _int(auto_metadata.get("quality"))
        q_gold = _int(gold_metadata.get("quality"))
        q_exact = q_within = None
        if q_auto is not None and q_gold is not None:
            q_exact = q_auto == q_gold
            q_within = abs(q_auto - q_gold) <= 1
            exact_quality.append(q_exact)
            within_one_quality.append(q_within)

        for sg in gold_subgoals:
            idx = _int(sg.get("segment_idx"))
            frame = _int(sg.get("frame_idx"))
            if idx is not None and frame is not None:
                subgoal_matches.append(auto_subgoals.get(idx) == frame)

        per_episode.append({
            "episode_id": episode_id,
            "boundary_iou_mean": _mean(ep_ious),
            "quality_auto": q_auto,
            "quality_gold": q_gold,
            "quality_exact": q_exact,
            "quality_within_one": q_within,
        })

    return {
        "gold_file": str(Path(gold_path).resolve()),
        "episode_count": len(gold.get("episodes", [])),
        "reviewed_episode_count": reviewed,
        "subtask_boundary_temporal_iou_mean": _mean(boundary_ious),
        "quality_exact_agreement": _bool_mean(exact_quality),
        "quality_within_one_agreement": _bool_mean(within_one_quality),
        "subgoal_frame_agreement": _bool_mean(subgoal_matches),
        "per_episode": per_episode,
    }


def format_report(report: dict[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    return "\n".join([
        f"Reliability over {report['reviewed_episode_count']}/{report['episode_count']} reviewed episodes",
        f"  subtask boundary temporal IoU (mean): {pct(report['subtask_boundary_temporal_iou_mean'])}",
        f"  quality exact agreement:              {pct(report['quality_exact_agreement'])}",
        f"  quality within-one agreement:         {pct(report['quality_within_one_agreement'])}",
        f"  subgoal frame agreement:              {pct(report['subgoal_frame_agreement'])}",
    ])


def _reviewed_subtasks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        s for s in entry.get("gold", {}).get("subtasks", [])
        if s.get("accept_auto") or s.get("start_frame") is not None or s.get("end_frame") is not None
    ]


def _reviewed_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    meta = entry.get("gold", {}).get("metadata", {})
    return meta if (meta.get("accept_auto") or meta.get("quality") is not None) else {}


def _reviewed_subgoals(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        s for s in entry.get("gold", {}).get("subgoals", [])
        if s.get("accept_auto") or s.get("frame_idx") is not None
    ]


def _temporal_iou(auto: dict[str, Any], gold: dict[str, Any]) -> float | None:
    if gold.get("accept_auto"):
        gold = {**gold, "start_frame": auto.get("start_frame"), "end_frame": auto.get("end_frame")}
    if gold.get("quality") is not None and gold.get("start_frame") is None:  # metadata-only review
        return None
    a0, a1 = _int(auto.get("start_frame")), _int(auto.get("end_frame"))
    g0, g1 = _int(gold.get("start_frame")), _int(gold.get("end_frame"))
    if None in (a0, a1, g0, g1):
        return None
    intersection = max(0, min(a1, g1) - max(a0, g0) + 1)
    union = max(a1, g1) - min(a0, g0) + 1
    return intersection / union if union > 0 else None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _bool_mean(values: list[bool]) -> float | None:
    return (sum(1 for v in values if v) / len(values)) if values else None
