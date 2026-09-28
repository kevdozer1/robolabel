"""Schema v7 (PLAN 4.7), additive: new columns and new record types in the same long-format parquet.

v1 to v6 files still read (``schema.read_annotations``), and older readers ignore the new record types
because they filter on ``record_type``. This module writes v7 files only; ``schema.py`` and the legacy
pipeline (including the offline demo) are unchanged and keep writing v6.

New record types: ``coarse_subtask``, ``attempt``, ``scene_fact``, ``requirement``, ``check``. Nested
values (boxes, points, visibility, evidence) are stored as JSON text so every column has one type.

The v1.1 fields (SPEC_V1_1 3.5) are additive and optional; they live in their own section at the end of
this module (``COLUMNS_V11``, ``add_v11_fields``, ``write_v11``, ``read_v11``) and leave the v7 writer
as it was.
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


# ------------------------------------------------------------------------------------------------ v1.1 (additive)
# SPEC_V1_1 3.5: optional fields on subtask and episode_metadata rows. The v7 writer above is unchanged
# (``COLUMNS_V7``, ``to_dataframe_v7``, ``write_v7`` and ``episode_rows`` produce the same files as
# before); a v1.1 run adds the fields to the rows with :func:`add_v11_fields` and writes them with
# :func:`write_v11`. Readers (:func:`read_v11`, :func:`subtask_fields_v11`, :func:`episode_fields_v11`,
# :func:`subtask_records_v11`, :func:`episode_record_v11`) accept rows and files without the fields and
# give None for each missing one. ``boundary_source`` is a v7 column; v1.1 adds the values ``coarse`` and
# ``crawl``. ``event_sources`` is stored as comma-joined text (like ``cameras_used``) and read back as a
# list. The rows keep ``schema_version`` v7: the new fields are additive (SPEC_V1_1 3.5).

SUBTASK_V11 = ["end_event", "coarse_end_frame", "crawl_calls", "attempt_outcome", "event_sources"]
EPISODE_V11 = ["has_end_state", "goal_command", "event_sources", "coarse_mode", "coarse_fps", "crawl_enabled",
               "crawl_model"]
V11_FIELDS = list(dict.fromkeys(SUBTASK_V11 + EPISODE_V11))
COLUMNS_V11 = COLUMNS_V7 + [c for c in V11_FIELDS if c not in COLUMNS_V7]
INT_COLS_V11 = INT_COLS | {"coarse_end_frame", "crawl_calls"}
FLOAT_COLS_V11 = FLOAT_COLS | {"coarse_fps"}
BOOL_COLS_V11 = {"mistake", "has_end_state", "crawl_enabled"}
LIST_COLS_V11 = {"event_sources"}
_TYPED_COLS_V11 = INT_COLS_V11 | FLOAT_COLS_V11 | BOOL_COLS_V11 | LIST_COLS_V11

END_EVENTS = ("close_start", "open_start", "contact_start", "contact_end", "other")
BOUNDARY_SOURCES_V11 = ("signal", "vlm", "coarse", "crawl")
ATTEMPT_OUTCOMES = ("success", "failed", "aborted")
COARSE_MODES = ("frames", "video")
SUBTASK_READ_V11 = ["end_event", "boundary_source", "coarse_end_frame", "crawl_calls", "attempt_outcome",
                    "event_sources"]


def _missing(v: Any) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    return isinstance(v, float) and v != v


def _join(values: Any) -> str | None:
    """A list of names as comma-joined text (a string passes through; None and missing stay None)."""
    if _missing(values):
        return None
    if isinstance(values, str):
        return values
    return ",".join(str(v) for v in values)


def _split(value: Any) -> list[str] | None:
    if _missing(value):
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [p for p in str(value).split(",") if p]


def _as_int(v: Any) -> int | None:
    if _missing(v) or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float(v: Any) -> float | None:
    if _missing(v) or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_bool(v: Any) -> bool | None:
    if _missing(v):
        return None
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        return None
    return bool(v)


def _typed(col: str, v: Any) -> Any:
    if col in INT_COLS_V11:
        return _as_int(v)
    if col in FLOAT_COLS_V11:
        return _as_float(v)
    if col in BOOL_COLS_V11:
        return _as_bool(v)
    if col in LIST_COLS_V11:
        return _split(v)
    return None if _missing(v) else v


def _getter(row: Any):
    if hasattr(row, "get"):
        return row.get
    return lambda k, d=None: getattr(row, k, d)


# ------------------------------------------------------------------------------------------------ v1.1 writing
def subtask_row_fields_v11(segment: dict[str, Any], event_sources: Any = None) -> dict[str, Any]:
    """The v1.1 fields of one subtask row, from a v1.1 segment dict (absent keys stay None)."""
    return {"end_event": segment.get("end_event"), "boundary_source": segment.get("boundary_source"),
            "coarse_end_frame": segment.get("coarse_end_frame"), "crawl_calls": segment.get("crawl_calls"),
            "attempt_outcome": segment.get("attempt_outcome"), "event_sources": _join(event_sources)}


def episode_row_fields_v11(*, has_end_state: bool | None = None, goal_command: str | None = None,
                           event_sources: Any = None, coarse_mode: str | None = None,
                           coarse_fps: float | None = None, crawl_enabled: bool | None = None,
                           crawl_model: str | None = None) -> dict[str, Any]:
    """The v1.1 fields of an ``episode_metadata`` row."""
    return {"has_end_state": has_end_state, "goal_command": goal_command, "event_sources": _join(event_sources),
            "coarse_mode": coarse_mode, "coarse_fps": coarse_fps, "crawl_enabled": crawl_enabled,
            "crawl_model": crawl_model}


def add_v11_fields(rows: list[dict[str, Any]], segments: list[dict[str, Any]], *, event_sources: Any = None,
                   episode_fields: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Copies of one episode's v7 rows (from :func:`episode_rows`) with the v1.1 fields added.

    Subtask rows take their fields from ``segments[segment_idx]``; the ``episode_metadata`` row takes
    ``episode_fields`` (the keyword arguments of :func:`episode_row_fields_v11`), with ``event_sources``
    from the argument when ``episode_fields`` does not name it. Other rows are copied unchanged.
    """
    ep = dict(episode_fields or {})
    ep.setdefault("event_sources", event_sources)
    ep_fields = episode_row_fields_v11(**ep)
    out = []
    for r in rows:
        row = dict(r)
        kind = row.get("record_type")
        if kind == "subtask":
            i = _as_int(row.get("segment_idx"))
            seg = segments[i] if i is not None and 0 <= i < len(segments) else {}
            row.update(subtask_row_fields_v11(seg, event_sources))
        elif kind == "episode_metadata":
            row.update(ep_fields)
        out.append(row)
    return out


def validate_v11_row(row: dict[str, Any]) -> list[str]:
    """Problems with the v1.1 fields of one row (None is always allowed: the fields are optional)."""
    problems = []
    for col, allowed in (("end_event", END_EVENTS), ("boundary_source", BOUNDARY_SOURCES_V11),
                         ("attempt_outcome", ATTEMPT_OUTCOMES), ("coarse_mode", COARSE_MODES)):
        v = row.get(col)
        if not _missing(v) and v not in allowed:
            problems.append(f"{col} {v!r} is not one of {', '.join(allowed)}")
    for col in ("coarse_end_frame", "crawl_calls"):
        v = row.get(col)
        if not _missing(v) and (_as_int(v) is None or _as_int(v) < 0):
            problems.append(f"{col} {v!r} is not a non-negative integer")
    return problems


def to_dataframe_v11(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """As :func:`to_dataframe_v7`, with the v1.1 columns (``COLUMNS_V11``) and their types."""
    frame = pd.DataFrame(rows)
    for col in COLUMNS_V11:
        if col not in frame.columns:
            frame[col] = None
    frame["_ro"] = frame["record_type"].map(RECORD_ORDER).fillna(9)
    frame["_seg"] = pd.to_numeric(frame["segment_idx"], errors="coerce").fillna(-1)
    frame["_i"] = range(len(frame))
    frame = frame.sort_values(["episode_id", "arm", "_ro", "_seg", "_i"], na_position="first")
    frame = frame.drop(columns=["_ro", "_seg", "_i"])[COLUMNS_V11].reset_index(drop=True)
    for col in COLUMNS_V11:
        if col in INT_COLS_V11:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
        elif col in FLOAT_COLS_V11:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        elif col == "mistake":
            frame[col] = frame[col].astype("boolean")
        elif col in BOOL_COLS_V11:
            frame[col] = frame[col].map(_as_bool).astype("boolean")
        elif col in LIST_COLS_V11:
            frame[col] = frame[col].map(_join)
        else:
            frame[col] = frame[col].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))
    return frame


def write_v11(rows: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ANNOTATIONS_FILENAME
    to_dataframe_v11(rows).to_parquet(path, index=False)
    return path


# ------------------------------------------------------------------------------------------------ v1.1 reading
def read_v11(path: str | Path) -> pd.DataFrame:
    """A v7 or v1.1 annotations file (or its folder), with every ``COLUMNS_V11`` column present."""
    from .schema import read_annotations

    frame = read_annotations(path)
    for col in COLUMNS_V11:
        if col not in frame.columns:
            frame[col] = None
    return frame


def subtask_fields_v11(row: Any) -> dict[str, Any]:
    """The v1.1 fields of a subtask row (a dict or a pandas row), typed, None where absent."""
    get = _getter(row)
    return {c: _typed(c, get(c, None)) for c in SUBTASK_READ_V11}


def episode_fields_v11(row: Any) -> dict[str, Any]:
    """The v1.1 fields of an ``episode_metadata`` row (a dict or a pandas row), typed, None where absent."""
    get = _getter(row)
    return {c: _typed(c, get(c, None)) for c in EPISODE_V11}


def _clean_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: _typed(k, v) if k in _TYPED_COLS_V11 else (None if _missing(v) else v) for k, v in rec.items()}


def _select(frame: pd.DataFrame, episode_id: str, arm: str | None, kind: str) -> list[dict[str, Any]]:
    sel = (frame["episode_id"].astype(str) == str(episode_id)) & (frame["record_type"] == kind)
    if arm is not None and "arm" in frame.columns:
        sel &= frame["arm"].astype(str) == str(arm)
    return frame[sel].to_dict("records")


def subtask_records_v11(frame: pd.DataFrame, episode_id: str, arm: str | None = None, *,
                        derive: bool = True) -> list[dict[str, Any]]:
    """Subtask rows of one episode (and arm) in segment order, typed, with every v1.1 field (None in a v7
    file, except ``attempt_outcome``).

    A row without ``attempt_outcome`` (a v7 file) gets it derived by the old rule (SPEC_V1_1 4: a reader
    that finds none derives it; :func:`fill_attempt_outcome`, per arm); present values are kept.
    ``derive=False`` leaves it None, as the file has it."""
    out = []
    for rec in _select(frame, episode_id, arm, "subtask"):
        clean = _clean_record(rec)
        clean.update(subtask_fields_v11(rec))
        out.append(clean)
    out.sort(key=lambda r: (r.get("segment_idx") is None, r.get("segment_idx") or 0))
    if not derive:
        return out
    by_arm: dict[Any, list[int]] = {}
    for k, r in enumerate(out):
        by_arm.setdefault(r.get("arm"), []).append(k)
    for ks in by_arm.values():
        for k, filled in zip(ks, fill_attempt_outcome([out[k] for k in ks]), strict=True):
            out[k] = filled
    return out


def episode_record_v11(frame: pd.DataFrame, episode_id: str, arm: str | None = None) -> dict[str, Any] | None:
    """The ``episode_metadata`` row of one episode (and arm), typed, with every v1.1 field (None in v7)."""
    recs = _select(frame, episode_id, arm, "episode_metadata")
    if not recs:
        return None
    clean = _clean_record(recs[0])
    clean.update(episode_fields_v11(recs[0]))
    return clean


def fill_attempt_outcome(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of segments with ``attempt_outcome`` derived where it is missing (SPEC_V1_1 4).

    The one rule is :func:`robolabel.eval.failure.derive_attempt_outcome` (this function calls it): the v7
    rule marked every phase of a failed attempt ``outcome: failed``, so an attempt is ``failed`` when any of
    its phases failed (or has ``mistake: true``), else ``aborted`` when any was aborted, else ``success``; a
    segment without an ``attempt_idx`` is an attempt of its own. Segments that already carry an
    ``attempt_outcome`` keep it; items that are not mappings are dropped.
    """
    from .eval.failure import derive_attempt_outcome

    return derive_attempt_outcome(segments)
