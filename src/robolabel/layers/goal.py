"""L4 goal layer (V_LITE L4): what the episode was for, from one call that never sees segment evidence.

Inputs: the task string, the inventory, scene facts at frame 0 and at the last keyframe (plain lines),
the L1 robot end state (marked as measured by the robot), a one-line attempt summary, and frames 0 and
N-1 from the two external cameras. Post-processing maps object names and aliases to inventory IDs
(other text stays as the model wrote it), adds any missing mandatory robot slot from L1
(``basis: signal``, ``status: unsure``, ``unsure_kind: intent``, flagged ``added_by: postprocess``),
rejects requirements that refer only to something inside a failed attempt (the G2 rule), converts
holding items to the canonical form (spec 3.2), and derives ``episode_outcome`` over the required
items. Every repair, clamp and coercion is recorded.
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


def goal_ref(value: Any, objects: list[dict[str, Any]], repairs: list[str], label: str) -> str:
    """An object reference of the goal call: an inventory ID, ``none`` or ``unsure``; a name or alias maps
    to its ID (as in the segment layer). Any other text stays as the model wrote it, so the view shows
    the name and the metrics can still resolve it by string (spec 4.0 rule 2)."""
    v = str(value if value is not None else "").strip()
    low = v.lower()
    ids = {o["object_id"] for o in objects}
    if not v:
        return "none"
    if low in ids or low in ("none", "unsure"):
        return low
    for o in objects:
        if low == o["name"].lower() or low in [a.lower() for a in o.get("aliases", [])]:
            repairs.append(f"goal: {label} {v!r} given as a name, mapped to {o['object_id']}")
            return o["object_id"]
    repairs.append(f"goal: {label} {v!r} is not an inventory ID or name, kept as the model's text")
    return v


def raw_goal_refs(data: Any) -> list[str]:
    """The goal call's object references as the model wrote them (for rule 9). The ``object`` of a robot
    item is not a reference (post-processing sets it to none), so it is left out."""
    if not isinstance(data, dict):
        return []
    out = [str(data.get("primary_target", "")), str(data.get("primary_destination", ""))]
    for r in data.get("requirements") or []:
        if isinstance(r, dict):
            if r.get("kind") != "robot_end_state":
                out.append(str(r.get("object", "")))
            out.append(str(r.get("ref_object", "")))
    return out


def postprocess_goal(data: Any, episode: Any, l1: dict[str, Any], objects: list[dict[str, Any]],
                     segments: list[dict[str, Any]], repairs: list[str]) -> dict[str, Any]:
    cams = label_to_camera(episode)
    last = episode.num_frames - 1

    def ref(v: Any, label: str) -> str:
        return goal_ref(v, objects, repairs, label)

    primary_target = ref(data.get("primary_target"), "primary_target")
    primary_destination = ref(data.get("primary_destination"), "primary_destination")
    failed_only = _failed_only_objects(segments) - {primary_target, primary_destination}
    reqs: list[dict[str, Any]] = []
    for r in data.get("requirements") or []:
        if not isinstance(r, dict):
            repairs.append("goal: a requirement that is not an object was dropped")
            continue
        pred = r.get("predicate")
        if pred not in PREDICATES:
            repairs.append(f"goal: predicate {pred!r} is not a predicate, set to other")
            pred = "other"
        kind = r.get("kind")
        if kind not in ("object_end_state", "robot_end_state"):
            repairs.append(f"goal: kind {kind!r} of {pred} is not a kind, set to object_end_state")
            kind = "object_end_state"
        if kind == "robot_end_state":
            obj = "none"
            if str(r.get("object") or "none").strip().lower() != "none":
                repairs.append(f"goal: object {r.get('object')!r} of robot item {pred} set to none")
        else:
            obj = ref(r.get("object"), f"{pred} object")
        refo = ref(r.get("ref_object"), f"{pred} ref_object")
        if kind == "object_end_state" and (obj in failed_only or (refo in failed_only and obj in failed_only)):
            repairs.append(f"goal: requirement on {obj} rejected (refers only to a failed attempt, G2 rule)")
            continue
        status = r.get("status")
        if status not in ("required", "incidental", "unsure"):
            repairs.append(f"goal: status {status!r} of {pred} is not a status, set to unsure")
            status = "unsure"
        uk = r.get("unsure_kind") if r.get("unsure_kind") in ("perception", "intent") else None
        if status == "unsure" and uk is None:
            uk = "intent"
            repairs.append(f"goal: unsure item {pred} had no kind, set to intent")
        if status != "unsure" and uk is not None:
            repairs.append(f"goal: unsure_kind {uk} of a {status} item {pred} dropped")
            uk = None
        vis = {}
        raw_vis = r.get("visibility") or []
        dropped = 0 if isinstance(raw_vis, list) else 1
        for v in raw_vis if isinstance(raw_vis, list) else []:
            if isinstance(v, dict) and cams.get(str(v.get("camera", ""))) and v.get("class") in (
                    "visible", "partial", "not_visible"):
                vis[cams[str(v["camera"])]] = v["class"]
            else:
                dropped += 1
        if dropped:
            repairs.append(f"goal: visibility of {pred}: {dropped} entry or entries with an unknown camera or class "
                           "dropped")
        dcam = str(r.get("deciding_camera", ""))
        if dcam.strip().lower() not in ("", "none") and not cams.get(dcam):
            repairs.append(f"goal: deciding_camera {dcam!r} of {pred} is not a camera, left empty")
        if len(str(r.get("reason", ""))) > 240:
            repairs.append(f"goal: reason of {pred} cut to 240 characters")
        try:
            dframe = int(r.get("deciding_frame"))
        except (TypeError, ValueError, OverflowError):
            repairs.append(f"goal: deciding_frame {r.get('deciding_frame')!r} of {pred} set to {last}")
            dframe = last
        if not 0 <= dframe <= last:
            repairs.append(f"goal: deciding_frame {dframe} of {pred} clamped to [0, {last}]")
            dframe = max(0, min(dframe, last))
        value = _bool_or_text(r.get("value"), pred)
        if pred != "state" and str(r.get("value")).strip().lower() not in ("true", "false", "unsure"):
            repairs.append(f"goal: value {r.get('value')!r} of {pred} set to {value}")
        elif pred == "state" and value == "unsure" and str(r.get("value")).strip().lower() != "unsure":
            repairs.append(f"goal: value {r.get('value')!r} of {pred} set to unsure")
        ach = "unknown" if r.get("achieved") in (None, "") else str(r.get("achieved")).strip().lower()
        if ach not in ("true", "false", "unknown"):
            repairs.append(f"goal: achieved {r.get('achieved')!r} of {pred} set to unknown")
        basis = r.get("basis")
        if basis not in ("task_string", "physical_necessity", "observed"):
            repairs.append(f"goal: basis {basis!r} of {pred} set to observed")
            basis = "observed"
        reqs.append({"req_id": f"r{len(reqs) + 1}", "kind": kind, "object": obj,
                     "predicate": pred, "ref_object": refo, "value": value,
                     "status": status, "unsure_kind": uk, "basis": basis,
                     "achieved": True if ach == "true" else False if ach == "false" else "unknown",
                     "deciding_frame": dframe,
                     "deciding_camera": cams.get(str(r.get("deciding_camera", "")), ""),
                     "visibility": vis, "reason": str(r.get("reason", ""))[:240], "added_by": "model"})
    # canonical holding form (spec 3.2): "holding nothing" is holding, ref none, value false; "holding o3"
    # is holding, ref o3, value true
    for r in reqs:
        if r["predicate"] != "holding":
            continue
        if r["ref_object"] != "none" and (r["value"] is False or r["ref_object"] == "unsure"):
            repairs.append(f"goal: holding {r['ref_object']} value {r['value']} converted to the canonical form "
                           "(ref_object none)")
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
    if len(str(data.get("objective_text") or "")) > 300:
        repairs.append("goal: objective_text cut to 300 characters")
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
    """Plain text of a requirement for the rating page. A value ``unsure`` reads "unsure whether ...", and
    an ``unsure`` object reads "an unidentified object"."""
    def nm(x: str) -> str:
        return "an unidentified object" if x == "unsure" else names.get(x, x)

    p = r["predicate"].replace("_", " ")
    v = r["value"]
    neg = v is False
    unsure = v == "unsure"
    if r["kind"] == "robot_end_state":
        if r["predicate"] == "holding" and r["ref_object"] == "none":
            if neg:
                return "the robot is holding nothing"
            claim = "the robot is holding an object"
        elif r["predicate"] == "holding":
            claim = f"the robot is {'not ' if neg else ''}holding {nm(r['ref_object'])}"
        else:
            claim = f"the robot is {'not ' if neg else ''}{p}"
        return f"unsure whether {claim}" if unsure else claim
    if r["predicate"] == "state":
        return f"unsure what state {nm(r['object'])} is in" if unsure else f"{nm(r['object'])} is {v}"
    ref = f" {nm(r['ref_object'])}" if r["ref_object"] != "none" else ""
    claim = f"{nm(r['object'])} is {'not ' if neg else ''}{p}{ref}"
    return f"unsure whether {claim}" if unsure else claim
