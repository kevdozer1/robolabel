"""L4 goal layer (V_LITE L4): what the episode was for, from one call that never sees segment evidence.

Inputs: the task string, the inventory, scene facts at frame 0 and at the last keyframe (plain lines),
the L1 robot end state (marked as measured by the robot), a one-line attempt summary, and frames 0 and
N-1 from the two external cameras. Post-processing adds any missing mandatory robot slot from L1
(``basis: signal``, ``status: unsure``, ``unsure_kind: intent``, flagged ``added_by: postprocess``),
rejects requirements that refer only to something inside a failed attempt (the G2 rule), keeps the
canonical holding form (spec 3.2), and derives ``episode_outcome`` over the required items.
"""

from __future__ import annotations

from typing import Any

from ..prompts.v7 import MAX_TOKENS, PREDICATES, SCHEMAS, load_prompt
from ..providers.base import CallRequest, TextPart
from .frames import camera_label, image_parts
from .scene import fact_lines, inventory_lines, label_to_camera
from .signal import attempt_summary, end_state_lines

ROBOT_POSITION = ("withdrawn", "at_home_pose", "near_object")


def goal_request(episode: Any, l1: dict[str, Any], objects: list[dict[str, Any]], facts: list[dict[str, Any]],
                 context: dict[str, Any], reasoning: dict[str, Any] | None) -> tuple[CallRequest, list[dict[str, Any]]]:
    ext = episode.extra["external_cameras"][:2]
    last = episode.num_frames - 1
    items = [(0, c) for c in ext] + [(last, c) for c in ext]
    parts, manifest = image_parts(episode, items)
    kf = sorted({f["frame"] for f in facts})
    last_kf = kf[-1] if kf else last
    text = load_prompt("goal").format(
        task=episode.task or "", inventory_lines=inventory_lines(objects),
        facts_first=fact_lines(facts, 0, objects) if facts else "(no scene facts)",
        facts_last=fact_lines(facts, last_kf, objects) if facts else "(no scene facts)",
        end_state_lines="\n".join(end_state_lines(l1)) or "(no robot measurement)",
        attempt_summary=attempt_summary(l1))
    req = CallRequest(step="goal", system=load_prompt("system").strip(), parts=[TextPart(text), *parts],
                      schema=SCHEMAS["goal"], schema_name="goal_v7", max_tokens=MAX_TOKENS["goal"], reasoning=reasoning,
                      context={**context, "frame_indices": [0, last], "cameras": [camera_label(c) for c in ext]})
    return req, manifest


def _bool_or_text(value: Any, predicate: str) -> Any:
    v = str(value).strip().lower()
    if predicate == "state":
        return v or "unsure"
    if v in ("true", "yes"):
        return True
    if v in ("false", "no"):
        return False
    return "unsure"


def _failed_only_objects(segments: list[dict[str, Any]]) -> set[str]:
    ok = {s.get("target") for s in segments if s.get("outcome") == "success"} | \
        {s.get("destination") for s in segments if s.get("outcome") == "success"}
    bad = {s.get("target") for s in segments if s.get("outcome") == "failed"}
    return {o for o in bad - ok if o not in (None, "none", "unsure")}


def postprocess_goal(data: Any, episode: Any, l1: dict[str, Any], objects: list[dict[str, Any]],
                     segments: list[dict[str, Any]], repairs: list[str]) -> dict[str, Any]:
    ids = {o["object_id"] for o in objects}
    cams = label_to_camera(episode)

    def ref(v: Any) -> str:
        s = str(v or "none").strip().lower()
        return s if s in ids or s in ("none", "unsure") else "unsure"

    primary_target = ref(data.get("primary_target"))
    primary_destination = ref(data.get("primary_destination"))
    failed_only = _failed_only_objects(segments) - {primary_target, primary_destination}
    reqs: list[dict[str, Any]] = []
    for r in data.get("requirements") or []:
        if not isinstance(r, dict):
            continue
        pred = r.get("predicate") if r.get("predicate") in PREDICATES else "other"
        kind = r.get("kind") if r.get("kind") in ("object_end_state", "robot_end_state") else "object_end_state"
        obj, refo = ref(r.get("object")), ref(r.get("ref_object"))
        if kind == "object_end_state" and (obj in failed_only or (refo in failed_only and obj in failed_only)):
            repairs.append(f"goal: requirement on {obj} rejected (refers only to a failed attempt, G2 rule)")
            continue
        status = r.get("status") if r.get("status") in ("required", "incidental", "unsure") else "unsure"
        uk = r.get("unsure_kind") if r.get("unsure_kind") in ("perception", "intent") else None
        if status == "unsure" and uk is None:
            uk = "intent"
            repairs.append(f"goal: unsure item {pred} had no kind, set to intent")
        if status != "unsure":
            uk = None
        vis = {}
        for v in r.get("visibility") or []:
            if isinstance(v, dict) and cams.get(str(v.get("camera", ""))) and v.get("class") in (
                    "visible", "partial", "not_visible"):
                vis[cams[str(v["camera"])]] = v["class"]
        try:
            dframe = int(r.get("deciding_frame"))
        except (TypeError, ValueError):
            dframe = episode.num_frames - 1
        ach = str(r.get("achieved") or "unknown").lower()
        reqs.append({"req_id": f"r{len(reqs) + 1}", "kind": kind, "object": obj if kind == "object_end_state" else "none",
                     "predicate": pred, "ref_object": refo, "value": _bool_or_text(r.get("value"), pred),
                     "status": status, "unsure_kind": uk,
                     "basis": r.get("basis") if r.get("basis") in ("task_string", "physical_necessity", "observed")
                     else "observed",
                     "achieved": True if ach == "true" else False if ach == "false" else "unknown",
                     "deciding_frame": max(0, min(dframe, episode.num_frames - 1)),
                     "deciding_camera": cams.get(str(r.get("deciding_camera", "")), ""),
                     "visibility": vis, "reason": str(r.get("reason", ""))[:240], "added_by": "model"})
    # canonical holding form: "holding nothing" is holding, ref none, value false
    for r in reqs:
        if r["predicate"] == "holding" and r["ref_object"] == "unsure":
            r["ref_object"] = "none"
    # mandatory robot slots from L1
    sig = {it["predicate"]: it for it in l1.get("end_state", [])}
    have = {r["predicate"] for r in reqs if r["kind"] == "robot_end_state"}
    need: list[dict[str, Any]] = []
    if "holding" not in have and "holding" in sig:
        need.append(sig["holding"])
    if not have & {"gripper_open", "gripper_closed"}:
        g = sig.get("gripper_open") or sig.get("gripper_closed")
        if g:
            need.append(g)
    if not have & set(ROBOT_POSITION) and "withdrawn" in sig:
        need.append(sig["withdrawn"])
    for it in need:
        repairs.append(f"goal: mandatory robot slot {it['predicate']} missing, added from L1")
        reqs.append({"req_id": f"r{len(reqs) + 1}", "kind": "robot_end_state", "object": "none",
                     "predicate": it["predicate"], "ref_object": "none", "value": bool(it["value"]),
                     "status": "unsure", "unsure_kind": "intent", "basis": "signal",
                     "achieved": True, "deciding_frame": episode.num_frames - 1, "deciding_camera": "",
                     "visibility": {}, "reason": "added from the robot's own measurement", "added_by": "postprocess"})
    return {"objective_text": str(data.get("objective_text") or "")[:300], "primary_target": primary_target,
            "primary_destination": primary_destination, "requirements": reqs,
            "episode_outcome": episode_outcome(reqs)}


def episode_outcome(reqs: list[dict[str, Any]]) -> str:
    """success / failure / partial / unknown over the required items (unsure-perception counts unknown)."""
    vals = []
    for r in reqs:
        if r["status"] == "required":
            vals.append(r["achieved"])
        elif r["status"] == "unsure" and r.get("unsure_kind") == "perception":
            vals.append("unknown")
    if not vals:
        return "unknown"
    t = sum(1 for v in vals if v is True)
    f = sum(1 for v in vals if v is False)
    if t == len(vals):
        return "success"
    if f and not t:
        return "failure"
    if t and f:
        return "partial"
    return "unknown"


def requirement_text(r: dict[str, Any], names: dict[str, str]) -> str:
    def nm(x: str) -> str:
        return names.get(x, x)

    p = r["predicate"].replace("_", " ")
    v = r["value"]
    neg = v is False
    if r["kind"] == "robot_end_state":
        if r["predicate"] == "holding":
            what = "nothing" if (r["ref_object"] in ("none", "unsure") and neg) else nm(r["ref_object"]) \
                if r["ref_object"] not in ("none", "unsure") else "an object"
            return f"the robot is holding {what}" if not (neg and r["ref_object"] not in ("none", "unsure")) \
                else f"the robot is not holding {nm(r['ref_object'])}"
        return f"the robot is {'not ' if neg else ''}{p}"
    ref = f" {nm(r['ref_object'])}" if r["ref_object"] not in ("none", "unsure") else ""
    if r["predicate"] == "state":
        return f"{nm(r['object'])} is {v}"
    return f"{nm(r['object'])} is {'not ' if neg else ''}{p}{ref}"
