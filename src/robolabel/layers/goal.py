"""L4 goal layer (V_LITE L4): what the episode was for, from one call that never sees segment evidence.

Inputs: the task string, the inventory, scene facts at frame 0 and at the last keyframe (plain lines),
the L1 robot end state (marked as measured by the robot), a one-line attempt summary, and frames 0 and
N-1 from the two external cameras. Post-processing maps object names and aliases to inventory IDs
(other text stays as the model wrote it), adds any missing mandatory robot slot from L1
(``basis: signal``, ``status: unsure``, ``unsure_kind: intent``, flagged ``added_by: postprocess``),
rejects requirements that refer only to something inside a failed attempt (the G2 rule), converts
holding items to the canonical form (spec 3.2), and derives ``episode_outcome`` over the required
items. Every repair, clamp and coercion is recorded.

v1.1 (SPEC_V1_1 5 and D1a) adds, next to the v7 functions (which are unchanged):

- :func:`goal_request_v11`: the goal call with prompt v8 (``goal.txt``, schema ``goal_v8``), from one or more
  cameras of a clip; the robot's own measurement goes into the prompt only when ``signal`` is true.
- :func:`goal_command`: the command form of the goal, rendered deterministically from the required object
  end states (the text for training prompts).
- :func:`is_state_objective` and :func:`render_objective`: an objective that starts with an imperative verb is
  not a state, and is re-rendered as a state sentence from the requirements.
- :func:`postprocess_goal_v11`: the v7 post-processing plus ``has_end_state``, ``goal_command``, the state
  check, the G2 rule by attempt (SPEC 4), and D1a when a signal exists: a required robot item that the model
  left unknown, or marked unsure / perception, is decided by L1 (``basis: signal``). Without a signal the L1
  record is never read and robot items come from the model.
- :func:`episode_outcome_v11`: ``episode_outcome`` with D1a.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

from ..prompts.v7 import MAX_TOKENS, PREDICATES, SCHEMAS, load_prompt
from ..providers.base import CallRequest, ImagePart, TextPart
from ..schema_v7 import fill_attempt_outcome
from .frames import camera_label, frame_line, image_parts, model_jpeg
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


# ------------------------------------------------------------------------------------------ v1.1 (SPEC_V1_1 5, D1a)
OBJECT_KIND = "object_end_state"
ROBOT_KIND = "robot_end_state"
OBJECTIVE_MAX = 300
NO_END_STATE_OBJECTIVE = "No object has a required end state."

# A sentence that starts with one of these words is a command, not a state (unless a state verb or "of"
# follows it: see _STATE_FOLLOW).
IMPERATIVE_VERBS = frozenset("""
add adjust align approach arrange assemble attach bake boil bring build carry collect connect deliver detach
disassemble disconnect dispose drag draw erase extract fetch fill find fix flip fold fry gather get give go grab
grasp hang hold hover insert keep knock lay leave let lift loosen make move organize pick place polish pour press
pull push put raise rearrange release relocate remove reorganize replace reposition retract retrieve rinse rotate
rub shut slide sort squeeze stir store stow straighten swap sweep take throw tighten toss transfer transport
uncover unfold unload unlock unpack unplug unscrew untie unwrap unzip use wash wipe write
""".split())
# These also start state sentences as nouns or adjectives ("clean dishes ...", "stack of ...", "dance routine
# ...", "wave of greeting ..."): they make a command only when a determiner, a pronoun, a particle, a
# preposition or adverb of motion, a greeting or a number follows, or nothing does, or (when the caller gives
# the object nouns) an object noun follows in a text with no state verb ("open drawer").
AMBIGUOUS_VERBS = frozenset("""
bend brush button catch check chop clap clean clear close color colour cook cover crush cut dance drive drop dry
dust empty file grip hand heat hit iron jump kick level light load lock lower mark mix mop nod open pack paint
pass peel pile play plug point position reach return roll run scoop screw seal separate serve set shake shift
show sit sketch slice spin spread squash stack stand start stop switch tap tidy tie toast touch trace turn twist
type vacuum walk water wave wrap zip
""".split())
_COMMAND_FOLLOW = frozenset("""
the a an all both each every either its his her their my your our this that these those some any it them him me
us up down on off in out into onto away over back together apart open closed shut one two three four five six
seven eight nine ten
at to toward towards with for around across through along past forward forwards backward backwards left right
aside about again here there home twice once slowly quickly fast gently carefully hello hi goodbye bye
""".split())
# A state verb right after the first word makes the text a state whatever that word is ("dance routine is
# finished" aside, "walk ends at the door", "turn is complete"), and so does "of" ("wave of greeting ...").
_STATE_VERBS = frozenset({"is", "are", "was", "were", "has", "have", "had", "remains", "remain", "stays", "stay",
                          "ends", "ended"})
_STATE_FOLLOW = _STATE_VERBS | {"of"}
_LEAD_WORDS = frozenset({"first", "then", "now", "next", "finally", "just", "simply", "carefully", "gently",
                         "slowly", "quickly", "and", "also"})
_REQUEST_WORDS = frozenset({"please", "kindly", "do", "don", "dont"})
_CONTROL_WORDS = frozenset({"button", "buttons", "key", "keys", "keypad", "switch", "switches", "pedal", "trigger"})
_STATE_PHRASES = {"inside": "inside {ref}", "on_top_of": "on top of {ref}", "touching": "touching {ref}",
                  "at_location": "at {ref}", "in_gripper": "held", "lifted": "lifted", "tool_lifted": "lifted",
                  "unchanged": "unchanged", "activated": "activated"}
_MISSING_REFS = ("", "none")


def _words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text if text is not None else "").lower())


def is_state_objective(text: Any, *, nouns: Any = None) -> bool:
    """False when ``text`` reads as a command: it starts (after words like "first" or "then") with an
    imperative verb, with "please", "do not" or "never" and a verb, or with "to" and a verb.

    - A first word followed by a state verb (is, are, was, were, has, have, remains, stays, ends) or by "of"
      makes a state, whatever the word ("dance routine" aside: "walk ends at the door", "wave of greeting").
    - An ambiguous word ("open", "clean", "stack", "dance", "wave", "turn") counts as a verb only when a
      determiner, a pronoun, a particle, a preposition or adverb of motion ("at", "to", "forward"), a greeting
      ("hello") or a number follows it, or nothing does; so "dance performance by one person" is a state and
      "wave hello" a command.
    - ``nouns`` (optional; the words of the object names, for example from the inventory): an ambiguous word
      followed by one of them, in a text with no state verb, is a command ("open drawer", "close lid").
      Without ``nouns`` such a text counts as a state.

    An empty text states nothing, so it is not a state either."""
    words = _words(text)
    i = 0
    while i < len(words) and words[i] in _LEAD_WORDS:
        i += 1
    if i >= len(words):
        return False
    w = words[i]
    nxt = words[i + 1] if i + 1 < len(words) else ""
    if w in _REQUEST_WORDS:
        return False
    if w in ("to", "never") and (nxt in IMPERATIVE_VERBS or nxt in AMBIGUOUS_VERBS):
        return False
    if nxt in _STATE_FOLLOW:
        return True
    if w in IMPERATIVE_VERBS:
        return False
    if w in AMBIGUOUS_VERBS:
        if not nxt or nxt in _COMMAND_FOLLOW or nxt.isdigit():
            return False
        known = {str(x).lower() for x in (nouns or ())}
        if nxt in known and not any(x in _STATE_VERBS for x in words[i + 1:]):
            return False
    return True


def objective_nouns(objects: list[dict[str, Any]] | None, requirements: list[dict[str, Any]] | None = None,
                    names: dict[str, str] | None = None) -> frozenset[str]:
    """The words of the object names and aliases of the inventory, and of the objects the requirements name
    (an ID read through ``names``, else the plain words), for :func:`is_state_objective`. Determiners, particles
    and "none" or "unsure" are left out."""
    texts: list[str] = []
    for o in objects or []:
        if isinstance(o, dict):
            texts.append(str(o.get("name") or ""))
            texts += [str(a) for a in (o.get("aliases") or []) if isinstance(a, str)]
    nm = names or {}
    for r in requirements or []:
        if isinstance(r, dict):
            for key in ("object", "ref_object"):
                v = str(r.get(key) if r.get(key) is not None else "")
                texts.append(str(nm.get(v, v)))
    skip = _COMMAND_FOLLOW | _STATE_FOLLOW | {"none", "unsure", "unknown"}
    return frozenset(x for t in texts for x in _words(t) if x not in skip and not x.isdigit())


def _names_and_categories(names: Any, categories: dict[str, str] | None) -> tuple[dict[str, str], dict[str, str]]:
    """``names`` is an ID to name map, or an inventory (a list of objects), which also gives the categories."""
    if isinstance(names, dict):
        nm = {str(k): str(v) for k, v in names.items()}
        cats: dict[str, str] = {}
    else:
        objs = [o for o in (names or []) if isinstance(o, dict) and o.get("object_id")]
        nm = {str(o["object_id"]): str(o.get("name") or o["object_id"]) for o in objs}
        cats = {str(o["object_id"]): str(o["category"]) for o in objs if o.get("category")}
    if categories:
        cats = {**cats, **{str(k): str(v) for k, v in categories.items()}}
    return nm, cats


def _phrase(ref: Any, names: dict[str, str], *, unsure: str | None) -> str | None:
    """"the <name>" for an object reference (a leading article of the name is replaced); ``unsure`` for an
    unsure reference; None for none or an empty reference."""
    v = str(ref if ref is not None else "").strip()
    if v.lower() in _MISSING_REFS:
        return None
    if v.lower() == "unsure":
        return unsure
    words = str(names.get(v, names.get(v.lower(), v))).split()
    if words and words[0].lower() in ("the", "a", "an"):
        words = words[1:]
    return "the " + " ".join(words) if words else None


def _truth(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    return True if v in ("true", "yes") else False if v in ("false", "no") else None


def _state_word(value: Any) -> str | None:
    v = " ".join(str(value if value is not None else "").split())
    return None if v.lower() in ("", "unsure", "unknown", "none") else v


def _is_control(ref: Any, names: dict[str, str], categories: dict[str, str]) -> bool:
    """The inventory category decides when it is known; else a name ending in button, key or switch."""
    cat = categories.get(str(ref))
    if cat is not None:
        return cat.lower() == "control"
    words = re.findall(r"[a-z]+", names.get(str(ref), str(ref)).lower())
    return bool(words) and words[-1] in _CONTROL_WORDS


def _command(r: dict[str, Any], names: dict[str, str], categories: dict[str, str],
             extra_forms: bool = False) -> str | None:
    p = r.get("predicate")
    o = _phrase(r.get("object"), names, unsure=None)
    if o is None:
        return None
    if p == "state":
        word = _state_word(r.get("value"))
        return f"set {o} to {word}" if word else None
    val = _truth(r.get("value"))
    if val is None or (not val and not extra_forms):
        return None
    ref = _phrase(r.get("ref_object"), names, unsure=None)
    if p == "inside":
        return (f"put {o} in {ref}" if val else f"take {o} out of {ref}") if ref else None
    if p == "on_top_of":
        return (f"put {o} on {ref}" if val else f"take {o} off {ref}") if ref else None
    if p == "activated":
        if not val:
            return f"turn off {o}"
        return f"press {o}" if _is_control(r.get("object"), names, categories) else f"turn on {o}"
    if not extra_forms:
        return None
    if val and p == "at_location":
        return f"move {o} to {ref}" if ref else None
    if val and p == "lifted":
        return f"lift {o}"
    if val and p == "in_gripper":
        return f"pick up {o}"
    return None


def goal_command(requirements: list[dict[str, Any]], names: Any, *,
                 categories: dict[str, str] | None = None, extra_forms: bool = False) -> str:
    """The goal as a command, for training prompts (SPEC_V1_1 5), from the required object end states in
    their order, joined with " then "; robot items never appear.

    The four forms of SPEC_V1_1 5: ``inside`` gives "Put <object> in <ref>", ``on_top_of`` "Put <object> on
    <ref>", ``activated`` "Press <object>" for a control (inventory category ``control``, or with no category a
    name ending in button, key or switch) and "Turn on <object>" otherwise, ``state`` "Set <object> to
    <value>". ``inside``, ``on_top_of`` and ``activated`` render only with value true.

    ``extra_forms=True`` (off by default: the spec lists only the four forms, and this text becomes training
    prompts, so more verbs wait for sign-off) also renders value false of ``inside``, ``on_top_of`` and
    ``activated`` ("Take <object> out of <ref>", "Take <object> off <ref>", "Turn off <object>"), and
    ``at_location``, ``lifted`` and ``in_gripper`` ("Move <object> to <ref>", "Lift <object>", "Pick up
    <object>").

    Any other item, an unsure value, and an object or a needed reference that is none or unsure are left out.
    Objects read "the <name>" (``names`` maps IDs to names, or is the inventory); other text is the model's own
    words. Empty when nothing renders. Deterministic."""
    nm, cats = _names_and_categories(names, categories)
    cmds = []
    for r in requirements or []:
        if isinstance(r, dict) and r.get("kind") == OBJECT_KIND and r.get("status") == "required":
            c = _command(r, nm, cats, extra_forms)
            if c:
                cmds.append(c)
    if not cmds:
        return ""
    text = " then ".join(cmds)
    return text[0].upper() + text[1:]


def _clause(r: dict[str, Any], names: dict[str, str]) -> str | None:
    p = r.get("predicate")
    o = _phrase(r.get("object"), names, unsure="an unidentified object")
    if o is None:
        return None
    if p == "state":
        word = _state_word(r.get("value"))
        return f"{o} is {word}" if word else None
    val = _truth(r.get("value"))
    if val is None or p not in _STATE_PHRASES:
        return None
    phrase = _STATE_PHRASES[p]
    if "{ref}" in phrase:
        ref = _phrase(r.get("ref_object"), names, unsure="an unidentified object")
        if ref is None:
            return None
        phrase = phrase.replace("{ref}", ref)
    return f"{o} is {'' if val else 'not '}{phrase}"


def render_objective(requirements: list[dict[str, Any]], names: Any) -> str:
    """A state sentence from the required object end states, in their order ("The pink brick is inside the
    transparent box."); :data:`NO_END_STATE_OBJECTIVE` when none renders. Deterministic."""
    nm, _ = _names_and_categories(names, None)
    clauses = []
    for r in requirements or []:
        if isinstance(r, dict) and r.get("kind") == OBJECT_KIND and r.get("status") == "required":
            c = _clause(r, nm)
            if c:
                clauses.append(c)
    if not clauses:
        return NO_END_STATE_OBJECTIVE
    text = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + " and " + clauses[-1]
    return text[0].upper() + text[1:] + "."


# ---------------------------------------------------------------- D1a: L1 decides robot items (signal only)
def needs_signal_decision(r: dict[str, Any]) -> bool:
    """A robot item from the model that D1a hands to L1: required with achieved unknown, or unsure /
    perception. Items that post-processing added from L1 already carry the signal's value."""
    if r.get("kind") != ROBOT_KIND or r.get("added_by") == "postprocess":
        return False
    achieved = r.get("achieved")
    unknown = achieved is None or (not isinstance(achieved, bool) and str(achieved).strip().lower() == "unknown")
    return (r.get("status") == "required" and unknown) or \
        (r.get("status") == "unsure" and r.get("unsure_kind") == "perception")


def signal_achieved(r: dict[str, Any], l1: dict[str, Any] | None) -> bool | None:
    """Whether the robot item is achieved at the last frame as L1 measures it, or None when L1 does not decide
    it. L1 decides what L5 rule 6 compares: holding (L1 says whether any object is held, so "holding <a named
    object>" is decided only when L1 says nothing is held), gripper_open and gripper_closed at high confidence,
    and withdrawn. A value that is not true or false (unsure) is never decided."""
    if not l1 or r.get("kind") != ROBOT_KIND:
        return None
    sig = {it.get("predicate"): it for it in l1.get("end_state", []) if isinstance(it, dict)}
    p = r.get("predicate")
    val = _truth(r.get("value"))
    if val is None:
        return None
    if p == "holding" and "holding" in sig:
        held = bool(sig["holding"].get("value"))
        if str(r.get("ref_object") or "none").strip().lower() not in _MISSING_REFS:
            if val:  # "holding <object>": L1 can only say that nothing is held
                return False if not held else None
            return True if not held else None  # "not holding <object>" (not the canonical form)
        return held == val
    if p in ("gripper_open", "gripper_closed"):
        sp = "gripper_open" if "gripper_open" in sig else "gripper_closed" if "gripper_closed" in sig else None
        if sp is None or sig[sp].get("confidence") != "high":
            return None
        return (p == sp) == val
    if p == "withdrawn" and "withdrawn" in sig:
        return bool(sig["withdrawn"].get("value")) == val
    return None


def _need_l1(signal: bool, l1: Any) -> None:
    if signal and l1 is None:
        raise ValueError("signal=True needs the L1 record (l1=...)")


def episode_outcome_v11(reqs: list[dict[str, Any]], l1: dict[str, Any] | None = None, signal: bool = False) -> str:
    """``episode_outcome`` with D1a: when a signal exists, a robot item that the model left unknown or marked
    unsure / perception counts as L1 decides it (when L1 decides it). Without a signal it equals
    ``episode_outcome`` and ``l1`` is not read."""
    _need_l1(signal, l1)
    counted = []
    for r in reqs:
        decided = signal_achieved(r, l1) if signal and needs_signal_decision(r) else None
        if decided is not None:
            counted.append({"status": "required", "achieved": decided})
        else:
            counted.append(r)
    return episode_outcome(counted)


def _has_end_state(value: Any, repairs: list[str]) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        repairs.append("goal: has_end_state missing, left unknown")
        return None
    v = str(value).strip().lower()
    if v in ("true", "false"):
        repairs.append(f"goal: has_end_state {value!r} read as {v == 'true'}")
        return v == "true"
    repairs.append(f"goal: has_end_state {value!r} is not true or false, left unknown")
    return None


def _goal_episode(episode: Any) -> Any:
    """The episode as ``postprocess_goal`` reads it: a clip without ``camera_order`` gets its cameras."""
    extra = getattr(episode, "extra", None) or {}
    if "camera_order" in extra:
        return episode
    cams = list((extra.get("cameras") or {}).keys())
    key = getattr(episode, "camera_key", None)
    if not cams and key:
        cams = [key]
    return SimpleNamespace(episode_id=getattr(episode, "episode_id", ""), num_frames=episode.num_frames,
                           extra={**extra, "camera_order": cams})


def postprocess_goal_v11(data: Any, episode: Any, *, l1: dict[str, Any] | None = None,
                         objects: list[dict[str, Any]], segments: list[dict[str, Any]], repairs: list[str],
                         signal: bool) -> dict[str, Any]:
    """The goal call's answer (schema ``goal_v8``) as the v1.1 goal record: the ``postprocess_goal`` fields plus
    ``has_end_state`` (None when the answer has no true or false), ``goal_command`` and ``objective_rendered``.

    - With ``signal`` true: the v7 post-processing with L1 (mandatory robot slots from L1), then D1a: a robot
      item that the model left unknown (status required) or marked unsure / perception, and that L1 decides
      (:func:`signal_achieved`), gets ``achieved`` from L1 and ``basis: signal``; an unsure / perception item
      becomes ``required``. Every decision is recorded in ``repairs``.
    - With ``signal`` false: ``l1`` is never read (a hidden signal stays hidden); no robot slot is added and
      robot items are the model's own.
    - The G2 rule reads each segment's ``attempt_outcome`` (SPEC 4: under the v1.1 convention only the failing
      phase has outcome failed), derived by the v7 rule where it is missing.
    - An objective that is not a state (:func:`is_state_objective`, with the object nouns of the inventory and
      the requirements, :func:`objective_nouns`) is re-rendered from the requirements (:func:`render_objective`),
      recorded in ``repairs`` and flagged ``objective_rendered``.
    - ``goal_command`` uses the four forms of SPEC_V1_1 5 only (``goal_command`` with ``extra_forms`` off).
    - ``has_end_state`` false with object end-state items keeps the items: L5 rule 13 reports them.
    """
    _need_l1(signal, l1)
    if not isinstance(data, dict):
        repairs.append("goal: the answer is not an object, read as empty")
        data = {}
    by_attempt = [{**s, "outcome": s.get("attempt_outcome")} for s in fill_attempt_outcome(list(segments or []))]
    goal = postprocess_goal(data, _goal_episode(episode), l1 if signal else {}, objects, by_attempt, repairs)
    reqs = goal["requirements"]
    if signal:
        for r in reqs:
            if not needs_signal_decision(r):
                continue
            why = "unsure / perception" if r["status"] == "unsure" else "required, achieved unknown"
            decided = signal_achieved(r, l1)
            if decided is None:
                repairs.append(f"goal: robot item {r['predicate']} ({why}) is not decided by L1, kept (D1a)")
                continue
            r["achieved"] = decided
            r["basis"] = "signal"
            if r["status"] == "unsure":
                r["status"], r["unsure_kind"] = "required", None
            repairs.append(f"goal: robot item {r['predicate']} ({why}) decided by L1: achieved {decided} "
                           "(basis signal, D1a)")
    names = {o["object_id"]: o["name"] for o in objects}
    categories = {o["object_id"]: o["category"] for o in objects if o.get("category")}
    text = goal["objective_text"]
    rendered = not is_state_objective(text, nouns=objective_nouns(objects, reqs, names))
    if rendered:
        new = render_objective(reqs, names)[:OBJECTIVE_MAX]
        why = "is empty" if not text.strip() else "is not a state (it starts with an imperative verb)"
        repairs.append(f"goal: objective_text {text!r} {why}, rendered from the requirements: {new!r}")
        goal["objective_text"] = new
    goal["episode_outcome"] = episode_outcome_v11(reqs, l1 if signal else None, signal)
    goal["has_end_state"] = _has_end_state(data.get("has_end_state"), repairs)
    goal["goal_command"] = goal_command(reqs, names, categories=categories)
    goal["objective_rendered"] = rendered
    return goal


def _frame_getter(episode: Any, camera: str) -> Any:
    cams = (getattr(episode, "extra", None) or {}).get("cameras") or {}
    if camera in cams:
        return cams[camera]
    if not cams or camera == getattr(episode, "camera_key", None):
        return episode.frame
    raise KeyError(f"camera {camera!r} is not a camera of episode {getattr(episode, 'episode_id', '?')}")


def _fill(template: str, values: dict[str, Any]) -> str:
    """Replace ``{name}`` placeholders in one pass, so text inside the values is never read as a placeholder."""
    return re.sub(r"\{([a-z_]+)\}", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0),
                  template)


def goal_request_v11(episode: Any, *, camera: str | list[str], objects: list[dict[str, Any]],
                     facts: list[dict[str, Any]], context: dict[str, Any], reasoning: dict[str, Any] | None,
                     l1: dict[str, Any] | None = None, signal: bool = False
                     ) -> tuple[CallRequest, list[dict[str, Any]]]:
    """(request, manifest) for the v1.1 goal call (prompt v8 ``goal.txt``, schema ``goal_v8``, step ``goal``), as
    the v7 ``goal_request`` returns them; the manifest lists the (frame, camera) of each image.

    Images: the first and the last frame of each camera given (first frames, then last frames), each after
    its caption; ``context`` gains ``frame_indices`` and ``cameras``. Text: the task (or that there is none),
    the inventory lines (or plain words), the scene facts at frame 0 and at the last keyframe, and the robot's
    own measurement (``end_state_lines`` and the attempt summary) only when ``signal`` is true; otherwise the
    prompt says robot items come from the images, and ``l1`` is not read. Never segment evidence or phase text
    (V_LITE L4). Deterministic."""
    from ..prompts.v8 import GOAL_MAX_TOKENS, GOAL_SCHEMAS, prompt_sections
    from ..prompts.v8 import load_prompt as v8_load_prompt

    _need_l1(signal, l1)
    cams = [camera] if isinstance(camera, str) else list(camera)
    if not cams:
        raise ValueError("goal_request_v11 needs at least one camera")
    n, fps = int(episode.num_frames), float(episode.fps)
    last = max(0, n - 1)
    frames = [0] if last == 0 else [0, last]
    parts: list[Any] = []
    manifest: list[dict[str, Any]] = []
    getters = {c: _frame_getter(episode, c) for c in cams}
    for f in frames:
        for c in cams:
            parts.append(TextPart(frame_line(f, n, fps, c)))
            parts.append(ImagePart(model_jpeg(getters[c], f), f"{c}@{f}"))
            manifest.append({"frame": f, "camera": c})
    sec = prompt_sections("goal")
    labels = [camera_label(c) for c in cams]
    camera_text = f"camera {labels[0]}" if len(labels) == 1 else \
        "cameras " + ", ".join(labels[:-1]) + " and " + labels[-1]
    blocks = [_fill(sec["intro"], {"camera_text": camera_text})]
    task = (episode.task or "").strip()
    blocks.append(_fill(sec["task"], {"task": task}) if task else sec["no_task"])
    blocks.append(_fill(sec["objects"], {"inventory_lines": inventory_lines(objects)}) if objects else
                  sec["no_objects"])
    kf = sorted({int(f["frame"]) for f in facts})
    last_kf = kf[-1] if kf else last
    blocks.append(_fill(sec["facts"], {
        "facts_first": fact_lines(facts, 0, objects) if facts else "(no scene facts)",
        "facts_last": fact_lines(facts, last_kf, objects) if facts else "(no scene facts)"}))
    if signal:
        blocks.append(_fill(sec["robot_measured"], {
            "end_state_lines": "\n".join(end_state_lines(l1 or {})) or "(no robot measurement)",
            "attempt_summary": attempt_summary(l1 or {})}))
    else:
        blocks.append(sec["robot_not_measured"])
    kind = "inventory" if objects else "words"
    blocks.append(_fill(sec["rules"], {"target_rule": sec[f"target_{kind}"],
                                       "destination_rule": sec[f"destination_{kind}"],
                                       "ref_rule": sec[f"refs_{kind}"]}))
    text = "\n\n".join(blocks)
    req = CallRequest(step="goal", system=v8_load_prompt("system").strip(), parts=[TextPart(text), *parts],
                      schema=GOAL_SCHEMAS["goal_v8"], schema_name="goal_v8", max_tokens=GOAL_MAX_TOKENS["goal_v8"],
                      reasoning=reasoning, context={**context, "frame_indices": frames, "cameras": labels})
    return req, manifest
