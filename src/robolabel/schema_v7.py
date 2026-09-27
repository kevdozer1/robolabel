"""Schema v7 (PLAN 4.7), additive: new columns and new record types in the same long-format parquet.

v1 to v6 files still read (``schema.read_annotations``), and older readers ignore the new record types
because they filter on ``record_type``. This module writes v7 files only; ``schema.py`` and the legacy
pipeline (including the offline demo) are unchanged and keep writing v6.

New record types: ``coarse_subtask``, ``attempt``, ``scene_fact``, ``requirement``, ``check``. Nested
values (boxes, points, visibility, evidence) are stored as JSON text so every column has one type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import ANNOTATIONS_FILENAME, COLUMNS

SCHEMA_VERSION_V7 = "robolabel/annotations/v7"

SUBTASK_V7 = ["phase_class", "target_object_id", "destination_object_id", "attempt_idx", "outcome", "failure_type",
              "boundary_source", "boundary_confidence", "evidence_frame", "evidence_camera", "evidence_visibility",
              "coarse_idx", "confidence", "evidence_json"]
EPISODE_V7 = ["goal_objective", "primary_target", "primary_destination", "goal_source", "episode_outcome",
              "outcome_confidence", "n_attempts", "n_failed_attempts", "speed_steps", "speed_bin_pi07",
              "execution_quality", "label_risk", "review_status", "cameras_used", "pipeline_version",
              "layer_models_json"]
RECORD_V7 = ["coarse_text", "object_id", "object_name", "category", "evident_frame", "attempt_source",
             "frame_idx", "camera", "box_json", "point_json", "mask_path", "predicate", "ref_object_id", "value",
             "visibility", "source_model", "req_id", "kind", "status", "unsure_kind", "basis", "achieved",
             "deciding_frame", "deciding_camera", "visibility_json", "consensus_present", "consensus_of",
             "check_id", "target_record", "rule_or_question", "checker", "verdict", "probability", "check_note",
             "arm"]
COLUMNS_V7 = COLUMNS + [c for c in SUBTASK_V7 + EPISODE_V7 + RECORD_V7 if c not in COLUMNS]
RECORD_ORDER = {"episode_metadata": 0, "subtask": 1, "coarse_subtask": 2, "attempt": 3, "scene_fact": 4,
                "requirement": 5, "check": 6, "subgoal": 7}

INT_COLS = {"segment_idx", "start_frame", "end_frame", "num_frames", "attempt_idx", "coarse_idx", "evident_frame",
            "frame_idx", "deciding_frame", "n_attempts", "n_failed_attempts", "speed_steps", "evidence_frame",
            "consensus_present", "consensus_of", "check_id"}
FLOAT_COLS = {"fps", "cost_usd", "label_risk", "boundary_confidence", "confidence", "outcome_confidence",
              "probability"}


def to_dataframe_v7(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in COLUMNS_V7:
        if col not in frame.columns:
            frame[col] = None
    frame["_ro"] = frame["record_type"].map(RECORD_ORDER).fillna(9)
    frame["_seg"] = pd.to_numeric(frame["segment_idx"], errors="coerce").fillna(-1)
    frame["_i"] = range(len(frame))
    frame = frame.sort_values(["episode_id", "arm", "_ro", "_seg", "_i"], na_position="first")
    frame = frame.drop(columns=["_ro", "_seg", "_i"])[COLUMNS_V7].reset_index(drop=True)
    for col in COLUMNS_V7:
        if col in INT_COLS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
        elif col in FLOAT_COLS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        elif col == "mistake":
            frame[col] = frame[col].astype("boolean")
        else:
            frame[col] = frame[col].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))
    return frame


def write_v7(rows: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ANNOTATIONS_FILENAME
    to_dataframe_v7(rows).to_parquet(path, index=False)
    return path


def _j(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def episode_rows(*, arm: str, episode: Any, provider: str, model: str, pipeline_version: str,
                 objects: list[dict[str, Any]], facts: list[dict[str, Any]], segments: list[dict[str, Any]],
                 coarse: list[dict[str, Any]], goal: dict[str, Any] | None, l1: dict[str, Any],
                 checks: dict[str, Any], cost_usd: float, source_model: str) -> list[dict[str, Any]]:
    """All v7 rows of one (arm, episode)."""
    base = {"schema_version": SCHEMA_VERSION_V7, "source": "vlm", "episode_id": episode.episode_id,
            "task": episode.task, "num_frames": int(episode.num_frames), "fps": float(episode.fps),
            "provider": provider, "model": model, "strategy": "v-lite", "arm": arm}
    failed_attempts = {s.get("attempt_idx") for s in segments if s.get("outcome") == "failed"}
    rows: list[dict[str, Any]] = [{
        **base, "record_type": "episode_metadata",
        "goal_objective": (goal or {}).get("objective_text") if goal else None,
        "primary_target": (goal or {}).get("primary_target") if goal else None,
        "primary_destination": (goal or {}).get("primary_destination") if goal else None,
        "goal_source": "single_episode" if goal else None,
        "episode_outcome": (goal or {}).get("episode_outcome") if goal else "unknown",
        "n_attempts": len({s.get("attempt_idx") for s in segments}), "n_failed_attempts": len(failed_attempts),
        "speed_steps": int(episode.num_frames), "label_risk": checks.get("risk"),
        "review_status": "routed" if checks.get("routed") else "auto",
        "cameras_used": ",".join(episode.extra.get("camera_order", [])), "pipeline_version": pipeline_version,
        "layer_models_json": _j({"L1": "signal", "L2": source_model, "L3": source_model, "L4": source_model,
                                 "L5": "rules"}),
        "cost_usd": cost_usd}]
    for i, s in enumerate(segments):
        ev = s.get("evidence") or []
        rows.append({**base, "record_type": "subtask", "segment_idx": i, "start_frame": s["start_frame"],
                     "end_frame": s["end_frame"], "subtask_text": s.get("phase_text"), "phase": s.get("phase_class"),
                     "phase_class": s.get("phase_class"), "target": s.get("target"),
                     "target_object_id": s.get("target"), "destination_object_id": s.get("destination"),
                     "attempt_idx": s.get("attempt_idx"), "outcome": s.get("outcome"),
                     "failure_type": s.get("failure_type"), "mistake": bool(s.get("mistake")),
                     "boundary_source": s.get("boundary_source"),
                     "boundary_confidence": 0.9 if s.get("boundary_source") == "signal" else 0.5,
                     "evidence_frame": ev[0].get("frame") if ev else None,
                     "evidence_camera": ev[0].get("camera") if ev else None, "evidence_json": _j(ev),
                     "coarse_idx": next((c["coarse_idx"] for c in coarse if i in c.get("segment_indices", [])), None)})
    for c in coarse:
        rows.append({**base, "record_type": "coarse_subtask", "coarse_idx": c["coarse_idx"],
                     "start_frame": c["start_frame"], "end_frame": c["end_frame"], "coarse_text": c["text"],
                     "subtask_text": c["text"], "target_object_id": c.get("target"),
                     "destination_object_id": c.get("destination"), "mistake": bool(c.get("mistake"))})
    for a in l1.get("attempts", []):
        rows.append({**base, "record_type": "attempt", "attempt_idx": a["attempt_idx"], "object_id": None,
                     "start_frame": a["closing_onset"], "end_frame": a["end_frame"], "outcome": a["outcome"],
                     "failure_type": a["failure_type"], "evident_frame": a["event_frame"], "attempt_source": "signal",
                     "confidence": 0.8})
    for f in facts:
        for oid in f["visible"] + f["partial"]:
            box = next((b["box"] for b in f.get("boxes", []) if b["object_id"] == oid), None)
            rows.append({**base, "record_type": "scene_fact", "frame_idx": f["frame"], "camera": f["camera"],
                         "object_id": oid, "predicate": "visible", "value": "true",
                         "visibility": "visible" if oid in f["visible"] else "partial",
                         "box_json": _j(box) if box else None, "source_model": source_model})
        rows.append({**base, "record_type": "scene_fact", "frame_idx": f["frame"], "camera": f["camera"],
                     "object_id": f["in_gripper"], "predicate": "in_gripper", "value": "true",
                     "source_model": source_model})
        for r in f["relations"]:
            rows.append({**base, "record_type": "scene_fact", "frame_idx": f["frame"], "camera": f["camera"],
                         "object_id": r["subject"], "predicate": r["relation"], "ref_object_id": r["object"],
                         "value": r["value"], "source_model": source_model})
    for o in objects:
        for v in o.get("views", []):
            if v.get("visible"):
                rows.append({**base, "record_type": "scene_fact", "frame_idx": 0, "camera": v["camera"],
                             "object_id": o["object_id"], "object_name": o["name"], "category": o["category"],
                             "predicate": "inventory", "value": "true", "point_json": _j([v["x"], v["y"]]),
                             "box_json": _j(v["box"]), "visibility": "visible", "source_model": source_model})
    for r in (goal or {}).get("requirements", []) if goal else []:
        rows.append({**base, "record_type": "requirement", "req_id": r["req_id"], "kind": r["kind"],
                     "object_id": r["object"], "predicate": r["predicate"], "ref_object_id": r["ref_object"],
                     "value": str(r["value"]).lower(), "status": r["status"], "unsure_kind": r.get("unsure_kind"),
                     "basis": r["basis"], "achieved": str(r["achieved"]).lower(),
                     "deciding_frame": r.get("deciding_frame"), "deciding_camera": r.get("deciding_camera"),
                     "visibility_json": _j(r.get("visibility") or {}), "reason": r.get("reason")})
    for c in checks.get("checks", []):
        rows.append({**base, "record_type": "check", "check_id": c["rule_id"], "target_record": "episode",
                     "rule_or_question": f"rule {c['rule_id']}", "checker": "rule", "verdict": c["verdict"],
                     "check_note": c["note"], "cost_usd": 0.0})
    return rows
