"""Free and legacy arms of the sweep as view records (V_LITE "The other arms").

* ``sig_only``: L1 events only. Phases by event order (approach, grasp, transport, release, retract);
  empty, aborted or slipped attempts become failed approach and grasp segments; no targets; template
  text with "the object"; the compiled goal (spec 4.0) plus the L1 robot end state; outcome unknown.
  Close to baseline B3 and to A-segvlm without scene targets.
* ``uniform5``: baseline B4, five equal segments with canonical phases.
* ``b2b@<model>``: today's robolabel default (S2-open segmentation plus the quality call) turned into
  a view record: phase names mapped with the lexicon, targets as free text, the compiled goal, and the
  episode outcome from the quality call's task success.
"""

from __future__ import annotations

from typing import Any

from .eval.lexicon import coarse_groups, compiled_goal, map_phase

CANONICAL5 = ["approach", "grasp", "transport", "release", "retract"]


def _seg(start: int, end: int, phase: str, *, outcome: str = "success", failure: str = "none", attempt: int = 1,
         source: str = "signal", target: str = "none", destination: str = "none", text: str | None = None
         ) -> dict[str, Any]:
    return {"start": int(start), "end": int(end), "phase_class": phase,
            "phase_text": text or f"{phase} the object" if phase != "retract" else (text or "move the arm away"),
            "target_name": target, "destination_name": destination, "attempt_idx": attempt, "outcome": outcome,
            "failure_type": failure, "mistake": outcome == "failed", "boundary_source": source}


def _contiguous(segs: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in sorted(segs, key=lambda s: s["start"]):
        if s["end"] < s["start"]:
            continue
        if out and s["start"] <= out[-1]["start"]:
            continue
        if out:
            out[-1]["end"] = s["start"] - 1
        out.append(s)
    if not out:
        return [_seg(0, n - 1, "other", text="no gripper event")]
    out[0]["start"] = 0
    out[-1]["end"] = n - 1
    return out


def sig_only_segments(l1: dict[str, Any]) -> list[dict[str, Any]]:
    n = int(l1["num_frames"])
    cands = {(c["attempt_idx"], c["transition"]): c["frame"] for c in l1.get("candidates", [])}
    segs: list[dict[str, Any]] = []
    cursor = 0
    attempts = l1.get("attempts", [])
    for a in attempts:
        i = a["attempt_idx"]
        on = int(a["closing_onset"])
        if on > cursor:
            segs.append(_seg(cursor, on - 1, "approach", attempt=i))
        if a["outcome"] in ("empty", "aborted", "slip", "unknown"):
            failed = a["outcome"] != "unknown"
            if segs and segs[-1]["phase_class"] == "approach" and failed:
                segs[-1].update(outcome="failed", failure_type=a["failure_type"], mistake=True)
            end = int(a["opening_onset"]) - 1 if a.get("opening_onset") is not None else max(on, int(a["event_frame"]))
            segs.append(_seg(on, max(on, end), "grasp", attempt=i, outcome="failed" if failed else "success",
                             failure=a["failure_type"] if failed else "none"))
            cursor = max(on, end) + 1
            continue
        g2t = cands.get((i, "grasp->transport"))
        grasp_end = int(g2t) if g2t is not None else int(a["closing_offset"])
        segs.append(_seg(on, grasp_end, "grasp", attempt=i))
        if a["outcome"] == "released":
            op = int(a["opening_onset"])
            if op - 1 > grasp_end:
                segs.append(_seg(grasp_end + 1, op - 1, "transport", attempt=i))
            r2r = cands.get((i, "release->retract"))
            rel_end = int(r2r) if r2r is not None else int(a["opening_offset"])
            segs.append(_seg(op, max(op, rel_end), "release", attempt=i))
            cursor = max(op, rel_end) + 1
        else:  # hold to the end
            if n - 1 > grasp_end:
                segs.append(_seg(grasp_end + 1, n - 1, "transport", attempt=i))
            cursor = n
    withdrawn = any(it["predicate"] == "withdrawn" and it["value"] for it in l1.get("end_state", []))
    if cursor < n:
        if attempts and attempts[-1]["outcome"] == "released" and withdrawn:
            segs.append(_seg(cursor, n - 1, "retract", attempt=attempts[-1]["attempt_idx"]))
        elif segs:
            segs[-1]["end"] = n - 1
    return _contiguous(segs, n)


def _goal_view(goal: dict[str, Any], l1: dict[str, Any] | None) -> dict[str, Any]:
    reqs = []
    for r in goal.get("requirements", []):
        reqs.append({"text": _req_text(r), "kind": r.get("kind"), "predicate": r.get("predicate"),
                     "object_name": r.get("object") or "none", "ref_name": r.get("ref_object") or "none",
                     "value": r.get("value"), "status": r.get("status", "required"),
                     "unsure_kind": r.get("unsure_kind") or "none", "basis": r.get("basis", "compiled"),
                     "achieved": r.get("achieved", "unknown")})
    if l1 is not None:
        have = {r["predicate"] for r in reqs if r["kind"] == "robot_end_state"}
        for it in l1.get("end_state", []):
            if it["predicate"] in have:
                continue
            reqs.append({"text": f"robot {it['predicate'].replace('_', ' ')}: {it['value']}", "kind": "robot_end_state",
                         "predicate": it["predicate"], "object_name": "none", "ref_name": "none",
                         "value": it["value"], "status": "required", "unsure_kind": "none", "basis": "signal",
                         "achieved": True})
    return {"objective": _objective(goal), "primary_target_name": goal.get("primary_target") or "none",
            "primary_destination_name": goal.get("primary_destination") or "none", "requirements": reqs}


def _req_text(r: dict[str, Any]) -> str:
    p = str(r.get("predicate", "")).replace("_", " ")
    if r.get("kind") == "robot_end_state":
        return f"the robot is {p}"
    ref = f" {r['ref_object']}" if r.get("ref_object") else ""
    return f"{r.get('object') or 'the object'} is {p}{ref}"


def _objective(goal: dict[str, Any]) -> str:
    for r in goal.get("requirements", []):
        if r.get("kind") == "object_end_state":
            return _req_text(r)
    return "not stated (compiled goal has no object requirement)"


def _base_view(arm: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {"arm": arm, "episode_key": meta["episode_key"], "family": meta["family"], "fps": meta["fps"],
            "num_frames": meta["num_frames"], "cameras": meta["cameras"], "camera_sizes": meta.get("camera_sizes", {}),
            "task": meta.get("task"), "objects": [], "checks": [], "risk": None, "routed": False, "cost_usd": 0.0,
            "calls": 0, "wall_s": 0.0, "valid": True, "repairs": [], "no_output": False, "cache_hits": 0}


def _attempts_from_segments(segs: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    by: dict[int, list[dict[str, Any]]] = {}
    for s in segs:
        by.setdefault(int(s.get("attempt_idx") or 1), []).append(s)
    out = []
    for i in sorted(by):
        ss = by[i]
        failed = [s for s in ss if s["outcome"] == "failed"]
        out.append({"start": min(s["start"] for s in ss), "end": max(s["end"] for s in ss),
                    "outcome": "failed" if failed else "success",
                    "failure_type": failed[0]["failure_type"] if failed else "none", "source": source})
    return out


def _coarse(segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = coarse_groups(segs, lambda ref: None if ref in (None, "none", "unsure") else str(ref))
    return [{"start": g["start_frame"], "end": g["end_frame"], "text": g["text"], "mistake": bool(g["mistake"])}
            for g in groups]


def sig_only_view(l1: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    segs = sig_only_segments(l1)
    goal = compiled_goal(segs)
    v = _base_view("sig_only", meta)
    v.update(segments=segs, coarse=_coarse(segs), attempts=_attempts_from_segments(segs, "signal"),
             goal=_goal_view(goal, l1), episode_outcome="unknown")
    return v


def uniform5_view(meta: dict[str, Any]) -> dict[str, Any]:
    n = int(meta["num_frames"])
    edges = [round(i * n / 5) for i in range(6)]
    segs = [_seg(edges[i], edges[i + 1] - 1, CANONICAL5[i], source="uniform") for i in range(5)]
    segs[-1]["end"] = n - 1
    goal = compiled_goal(segs)
    v = _base_view("uniform5", meta)
    v.update(segments=segs, coarse=_coarse(segs), attempts=_attempts_from_segments(segs, "uniform"),
             goal=_goal_view(goal, None), episode_outcome="unknown")
    return v


def outcome_from_task_success(q: Any) -> str:
    """Legacy quality call's task success (1 to 5): 4 or 5 success, 3 partial, 1 or 2 failure."""
    try:
        v = int(q)
    except (TypeError, ValueError):
        return "unknown"
    return "success" if v >= 4 else "partial" if v == 3 else "failure"


def legacy_view(arm: str, meta: dict[str, Any], subtasks: list[Any], metadata: Any, *, cost: float, calls: int,
                wall_s: float, valid: bool, repairs: list[str], cache_hits: int = 0) -> dict[str, Any]:
    """B2b: legacy SubtaskSegment list plus EpisodeMetadata to a view record (compiled goal)."""
    segs = []
    for s in subtasks:
        phase = map_phase(getattr(s, "phase", None) or getattr(s, "subtask_text", "")) if (
            getattr(s, "phase", None) or getattr(s, "subtask_text", None)) else "unmapped"
        tgt = getattr(s, "target", None) or "none"
        segs.append({"start": int(s.start_frame), "end": int(s.end_frame), "phase_class": phase,
                     "phase_text": getattr(s, "phase", None) or s.subtask_text, "target_name": tgt,
                     "destination_name": "none", "attempt_idx": 1, "outcome": "success", "failure_type": "none",
                     "mistake": False, "boundary_source": "vlm", "subtask_text": s.subtask_text})
    goal = compiled_goal(segs)
    v = _base_view(arm, meta)
    v.update(segments=segs, coarse=[{"start": s["start"], "end": s["end"], "text": s["subtask_text"],
                                     "mistake": bool(getattr(metadata, "mistake", False))} for s in segs],
             attempts=_attempts_from_segments(segs, "vlm"), goal=_goal_view(goal, None),
             episode_outcome=outcome_from_task_success(getattr(metadata, "task_success_quality", None)),
             cost_usd=round(cost, 8), calls=calls, wall_s=round(wall_s, 3), valid=valid, repairs=repairs,
             no_output=not segs, cache_hits=cache_hits)
    return v
