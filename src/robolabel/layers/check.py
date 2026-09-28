"""L5 check layer (V_LITE L5, PLAN 4.2 rules 1 to 10): rules only, no model call, $0.

Each rule gives one ``check`` row per episode: ``pass``, ``fail`` or ``na`` with a short note. Episode
risk = failed rules / applicable rules. The episode is routed for review when a required item is
``unsure / perception`` and no signal decides it, when a claim contradicts a signal fact (rules 1, 2,
6, 7, 8), or when the object held at the end of the last grasp is not the primary target.

v1.1 (SPEC_V1_1 3.4 and D1b) adds :func:`run_checks_v11` next to :func:`run_checks` (unchanged): rules 1, 2,
6, 7 and 8 are ``na`` without a signal, three video-only rules check the crawl and the goal (11: crawl picks
lie inside their windows; 12: no boundary moved by the crawl crosses a neighbouring boundary; 13:
``has_end_state`` false implies no ``object_end_state`` item), and a failed, invalid, refused or unavailable
call, or no output, gives risk 1.0 and routes the episode with the reason.
"""

from __future__ import annotations

import re
from typing import Any

from .goal import signal_achieved

SIGNAL_RULES = {1, 2, 6, 7, 8}
GRASP = {"grasp"}
RELEASE = {"release"}
_TEMPLATE = re.compile(r"^(pick up |put |pour |insert |move the arm away|press |[a-z]+ )")
_NARRATIVE = re.compile(r"\b(fail|fails|failed|miss|missed|misses|retry|retries|again|twice|attempt|attempts|"
                        r"try|tries|tried|drop|drops|dropped|slip|slips|slipped)\b")


def _row(rule: int, verdict: str, note: str) -> dict[str, Any]:
    return {"rule_id": rule, "verdict": verdict, "note": note[:200]}


def run_checks(segments: list[dict[str, Any]], coarse: list[dict[str, Any]], goal: dict[str, Any] | None,
               l1: dict[str, Any], objects: list[dict[str, Any]], facts: list[dict[str, Any]],
               raw_refs: list[str], *, have_inventory: bool, have_facts: bool, tol: int = 5) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    closings = [int(e["onset"]) for e in l1.get("events", []) if e["type"] == "closing"]
    openings = [int(e["onset"]) for e in l1.get("events", []) if e["type"] == "opening"]
    grasps = [s for s in segments if s.get("phase_class") in GRASP]
    releases = [s for s in segments if s.get("phase_class") in RELEASE]

    # 1: each grasp-class segment starts within 5 frames of a closing onset
    if not grasps:
        rows.append(_row(1, "na", "no grasp segment"))
    else:
        bad = [s["start_frame"] for s in grasps if not any(abs(s["start_frame"] - c) <= tol for c in closings)]
        rows.append(_row(1, "fail" if bad else "pass",
                         f"grasp starts {bad} not within {tol} frames of a closing onset {closings}" if bad else
                         f"{len(grasps)} grasp segment(s) start at a closing onset"))
    # 2: each release-class segment contains an opening onset
    if not releases:
        rows.append(_row(2, "na", "no release segment"))
    else:
        bad = [(s["start_frame"], s["end_frame"]) for s in releases
               if not any(s["start_frame"] <= o <= s["end_frame"] for o in openings)]
        rows.append(_row(2, "fail" if bad else "pass",
                         f"release segments {bad} contain no opening onset" if bad else "every release contains an opening"))
    # 3: the object claimed in a (successful) grasp is in the gripper at the next keyframe, from the end of
    # the grasp segment on, in some camera where it is visible; a failed grasp claims no hold
    held_grasps = [s for s in grasps if s.get("outcome") == "success"]
    if not (have_inventory and have_facts) or not held_grasps:
        rows.append(_row(3, "na", "no inventory, no scene facts or no successful grasp"))
    else:
        fails, checked = [], 0
        for s in held_grasps:
            tgt = s.get("target")
            if tgt in (None, "none", "unsure"):
                continue
            later = sorted({f["frame"] for f in facts if f["frame"] >= s["end_frame"]})
            if not later:
                continue
            nxt = [f for f in facts if f["frame"] == later[0]]
            seen = [f for f in nxt if tgt in f["visible"] or f["in_gripper"] == tgt]
            if not seen:
                continue
            checked += 1
            if not any(f["in_gripper"] == tgt for f in seen):
                fails.append(f"{tgt} at frame {later[0]}")
        rows.append(_row(3, "na" if not checked else ("fail" if fails else "pass"),
                         f"claimed grasp targets not in the gripper: {fails}" if fails else f"{checked} grasp(s) checked"))
    # 4: one target per attempt
    by_attempt: dict[int, set[str]] = {}
    for s in segments:
        t = s.get("target")
        if t not in (None, "none", "unsure") and s.get("phase_class") in ("approach", "grasp", "transport", "release"):
            by_attempt.setdefault(int(s.get("attempt_idx") or 1), set()).add(t)
    multi = {a: sorted(t) for a, t in by_attempt.items() if len(t) > 1}
    rows.append(_row(4, "na" if not by_attempt else ("fail" if multi else "pass"),
                     f"attempts with several targets: {multi}" if multi else "one target per attempt"))
    # 5: every required object end state has a visible scene fact at the last keyframe, or is unsure/perception
    reqs = (goal or {}).get("requirements", []) if goal else []
    obj_req = [r for r in reqs if r["kind"] == "object_end_state" and r["status"] == "required"]
    if not (have_facts and have_inventory and goal) or not obj_req:
        rows.append(_row(5, "na", "no facts, no goal or no required object item"))
    else:
        last_kf = max(f["frame"] for f in facts)
        at_last = [f for f in facts if f["frame"] == last_kf]
        bad = [r["object"] for r in obj_req
               if not any(r["object"] in f["visible"] for f in at_last)
               and not (r["status"] == "unsure" and r.get("unsure_kind") == "perception")]
        rows.append(_row(5, "fail" if bad else "pass",
                         f"required items with no visible fact at frame {last_kf}: {bad}" if bad else "all supported"))
    # 6: robot end-state items equal the L1 facts. The model's claim about the last frame is the value when
    # achieved is true and its negation when achieved is false; an item with achieved unknown claims nothing.
    # L1 holding means holding any object, so a claim of not holding one named object is not compared.
    sig = {it["predicate"]: it for it in l1.get("end_state", [])}
    robot = [r for r in reqs if r["kind"] == "robot_end_state" and r.get("added_by") != "postprocess"]
    if not goal or not robot:
        rows.append(_row(6, "na", "no robot items from the model"))
    else:
        bad, compared = [], 0
        for r in robot:
            p, v, ach = r["predicate"], r["value"], r.get("achieved")
            if not isinstance(v, bool) or not isinstance(ach, bool):
                continue
            claim = v if ach else not v
            if p == "holding" and "holding" in sig:
                if not claim and r.get("ref_object", "none") != "none":
                    continue  # "not holding o3" leaves open holding something else; L1 cannot tell which object
                compared += 1
                if claim != bool(sig["holding"]["value"]):
                    bad.append(f"holding claimed {claim} vs signal {sig['holding']['value']}")
            elif p in ("gripper_open", "gripper_closed"):
                sp = "gripper_open" if "gripper_open" in sig else "gripper_closed"
                if sig.get(sp, {}).get("confidence") == "high":
                    compared += 1
                    if (p == sp) != claim:
                        bad.append(f"{p} claimed {claim} vs signal {sp}")
            elif p == "withdrawn" and "withdrawn" in sig:
                compared += 1
                if claim != bool(sig["withdrawn"]["value"]):
                    bad.append(f"withdrawn claimed {claim} vs signal {sig['withdrawn']['value']}")
        rows.append(_row(6, "na" if not compared else ("fail" if bad else "pass"),
                         "; ".join(bad) if bad else f"{compared} robot item(s) match the signal" if compared else
                         "no robot claim the signal decides (achieved unknown, value unsure, low confidence, or "
                         "not holding a named object)"))
    # 7: a retract is claimed only if L1 says the arm moved away. L1's withdrawn flag is about the end of
    # the episode, so only a retract that is the last segment is checked; a retract mid-episode is na.
    if not segments or segments[-1].get("phase_class") != "retract":
        mid = sum(1 for s in segments if s.get("phase_class") == "retract")
        rows.append(_row(7, "na", f"no final retract ({mid} retract(s) mid-episode not checked)" if mid else
                         "no retract claimed"))
    else:
        moved = bool(sig.get("withdrawn", {}).get("value"))
        rows.append(_row(7, "pass" if moved else "fail",
                         "the signal shows the arm moving away" if moved else "retract claimed but the arm did not move away"))
    # 8: every empty or slip attempt in L1 is matched by a failed segment
    bad_attempts = [a for a in l1.get("attempts", []) if a["outcome"] in ("empty", "slip")]
    if not bad_attempts:
        rows.append(_row(8, "na", "no empty or slip attempt in the signal"))
    else:
        missed = []
        for a in bad_attempts:
            lo, hi = int(a["closing_onset"]), int(a["event_frame"])
            if not any(s.get("outcome") == "failed" and s["start_frame"] <= hi and s["end_frame"] >= lo
                       for s in segments):
                missed.append(a["attempt_idx"])
        rows.append(_row(8, "fail" if missed else "pass",
                         f"signal attempts {missed} have no failed segment" if missed else
                         f"{len(bad_attempts)} failed attempt(s) matched"))
    # 9: every target and destination ID exists in the inventory
    if not have_inventory:
        rows.append(_row(9, "na", "no inventory"))
    else:
        ids = {o["object_id"] for o in objects}
        bad = sorted({r for r in raw_refs if r.lower() not in ids and r.lower() not in ("none", "unsure", "")})
        rows.append(_row(9, "fail" if bad else "pass", f"unknown references {bad}" if bad else "all references known"))
    # 10: coarse text equals the template rendering (no failure narrative)
    texts = [c.get("text", "") for c in coarse]
    names = sorted({str(o.get("name", "")).lower() for o in objects if o.get("name")}, key=len, reverse=True)

    def without_names(t: str) -> str:  # an object called "eye drops" is not a failure narrative
        t = t.lower()
        for nm in names:
            t = t.replace(nm, " object ")
        return t

    bad = [t for t in texts if _NARRATIVE.search(without_names(t)) or not _TEMPLATE.match(t.lower())]
    rows.append(_row(10, "na" if not texts else ("fail" if bad else "pass"),
                     f"non-template coarse text: {bad[:2]}" if bad else "coarse text is the template rendering"))

    applicable = [r for r in rows if r["verdict"] != "na"]
    failed = [r for r in applicable if r["verdict"] == "fail"]
    risk = round(len(failed) / len(applicable), 4) if applicable else None
    reasons = []
    for r in reqs:
        if r["status"] == "unsure" and r.get("unsure_kind") == "perception" and r["kind"] == "object_end_state":
            reasons.append(f"required-or-unsure item {r['predicate']} {r['object']} cannot be seen and no signal decides it")
            break
    if any(r["rule_id"] in SIGNAL_RULES for r in failed):
        reasons.append("a claim contradicts a signal fact (rules " +
                       ", ".join(str(r["rule_id"]) for r in failed if r["rule_id"] in SIGNAL_RULES) + ")")
    held = _held_after_last_grasp(l1, facts)
    pt = (goal or {}).get("primary_target") if goal else None
    ids = {o["object_id"] for o in objects}
    if held and pt in ids and held != pt:  # a primary target the inventory cannot resolve is not compared
        reasons.append(f"the held object {held} is not the primary target {pt}")
    return {"checks": rows, "risk": risk, "routed": bool(reasons), "route_reasons": reasons}


def _held_after_last_grasp(l1: dict[str, Any], facts: list[dict[str, Any]]) -> str | None:
    holds = [a for a in l1.get("attempts", []) if a.get("hold_frame") is not None]
    if not holds or not facts:
        return None
    f0 = int(holds[-1]["hold_frame"])
    later = sorted({f["frame"] for f in facts if f["frame"] >= f0})
    if not later:
        return None
    vals = [f["in_gripper"] for f in facts if f["frame"] == later[0] and f["in_gripper"] not in ("none", "unsure")]
    return vals[0] if vals else None


# ------------------------------------------------------------------------------------------ v1.1 (SPEC_V1_1 3.4, D1b)
VIDEO_RULES = (11, 12, 13)
# statuses of a call that D1b counts: the four of SPEC D1(b), plus "stopped" (the spend guard stopped paid
# calls, so the step has no output) and "not run" (a step skipped after a stop)
FAILED_STATUSES = ("invalid", "refused", "failed", "unavailable", "stopped", "not run")


def call_failures(failed_calls: Any) -> list[str]:
    """Notes for the calls D1b counts. Each item is a note (a string), a ``(step, result)`` pair, a dict with
    ``status`` (and ``step``, ``error``), or a call result with ``status`` (its receipt may name the step). A
    result or dict whose status is not a failed one (``ok``) is not counted; a string always is."""
    out: list[str] = []
    for item in failed_calls or []:
        if isinstance(item, str):
            if item.strip():
                out.append(item.strip())
            continue
        step, res = None, item
        if isinstance(item, tuple) and len(item) == 2:
            step, res = item
        if isinstance(res, str):
            status, error = res, None
        elif isinstance(res, dict):
            status, error, step = res.get("status"), res.get("error"), step or res.get("step")
        else:
            status, error = getattr(res, "status", None), getattr(res, "error", None)
            receipt = getattr(res, "receipt", None)
            step = step or (receipt.get("step") if isinstance(receipt, dict) else None) or getattr(res, "step", None)
        if status is not None and status not in FAILED_STATUSES:
            continue
        note = f"{step or 'a'} call {status or 'failed'}"
        if error:
            note += f": {str(error)[:80]}"
        out.append(note)
    return out


def _windows(entry: dict[str, Any]) -> list[list[int]]:
    wins = [entry.get(k) for k in ("stage1_frames", "retry_frames", "stage2_frames")]
    wins += [c.get("frames") for c in entry.get("call_log") or [] if isinstance(c, dict)]
    out = []
    for w in wins:
        try:
            frames = [int(f) for f in w or []]
        except (TypeError, ValueError):
            continue
        if frames:
            out.append(frames)
    return out


def _rule11(crawl_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Each crawl pick (the frame of the image the deciding answer points at) lies inside a window the model
    saw for that boundary. The onset is not checked: after a 9 at an edge it is one frame after the window by
    design."""
    picks = [e for e in crawl_log if e.get("pick") is not None]
    if not picks:
        return _row(11, "na", "no crawl pick")
    bad = []
    for e in picks:
        try:
            p = int(e["pick"])
        except (TypeError, ValueError):
            bad.append(f"boundary {e.get('boundary_index')} pick {e.get('pick')!r}")
            continue
        if not any(min(w) <= p <= max(w) for w in _windows(e)):
            bad.append(f"boundary {e.get('boundary_index')} pick {p}")
    return _row(11, "fail" if bad else "pass", f"crawl picks outside their windows: {bad}" if bad else
                f"{len(picks)} crawl pick(s) inside their windows")


def _rule12(segments: list[dict[str, Any]], crawl_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Every boundary the crawl moved (log onset other than the coarse frame, or a crawl-sourced segment whose
    end differs from its coarse end) lies strictly between its neighbouring boundaries in the final segments
    (the start of the segment before it and the start of the segment after the next one, or the clip end),
    and the segments meet there."""
    moved: dict[int, int | None] = {}
    for e in crawl_log:
        try:
            i, onset, coarse = int(e["boundary_index"]), int(e["onset"]), int(e["coarse_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if onset != coarse:
            moved[i] = onset
    for i, s in enumerate(segments[:-1]):
        ce = s.get("coarse_end_frame")
        if s.get("boundary_source") == "crawl" and ce is not None and int(s["end_frame"]) != int(ce):
            moved.setdefault(i, None)
    if not moved:
        return _row(12, "na", "no boundary moved by the crawl")
    bad = []
    for i in sorted(moved):
        if not 0 <= i < len(segments) - 1:
            bad.append(f"boundary {i} is not a boundary of the segments")
            continue
        before, after = segments[i], segments[i + 1]
        pos = int(after["start_frame"])
        lo = int(before["start_frame"])
        hi = int(segments[i + 2]["start_frame"]) if i + 2 < len(segments) else int(after["end_frame"]) + 1
        if moved[i] is not None and moved[i] != pos:
            bad.append(f"boundary {i}: crawl onset {moved[i]} but the segment starts at {pos}")
        elif not lo < pos < hi or int(before["end_frame"]) != pos - 1:
            bad.append(f"boundary {i} at {pos} crosses a neighbour ({lo}, {hi})")
    return _row(12, "fail" if bad else "pass", "; ".join(bad) if bad else
                f"{len(moved)} moved boundary(ies) between their neighbours")


def _rule13(goal: dict[str, Any] | None) -> dict[str, Any]:
    if not goal:
        return _row(13, "na", "no goal record")
    hes = goal.get("has_end_state")
    if hes is None:
        return _row(13, "na", "has_end_state not given")
    if hes is not False:
        return _row(13, "na", "the goal has an end state")
    items = [r for r in goal.get("requirements", []) if r.get("kind") == "object_end_state"]
    if items:
        return _row(13, "fail", f"has_end_state false but {len(items)} object_end_state item(s): " +
                    ", ".join(f"{r.get('predicate')} {r.get('object')}" for r in items[:4]))
    return _row(13, "pass", "no end state and no object_end_state item")


def _rule6_view(goal: dict[str, Any] | None) -> dict[str, Any] | None:
    """The goal as rule 6 should see it: robot items that D1a decided from L1 (basis signal) are not compared
    with the signal they came from, like the slots post-processing added from L1."""
    if not goal:
        return goal
    reqs = [dict(r, added_by="postprocess") if r.get("kind") == "robot_end_state" and r.get("basis") == "signal"
            else r for r in goal.get("requirements", [])]
    return {**goal, "requirements": reqs}


def run_checks_v11(segments: list[dict[str, Any]], coarse: list[dict[str, Any]], goal: dict[str, Any] | None, *,
                   l1: dict[str, Any] | None = None, objects: list[dict[str, Any]], facts: list[dict[str, Any]],
                   raw_refs: list[str], crawl_log: list[dict[str, Any]] | None, have_inventory: bool,
                   have_facts: bool, failed_calls: Any = (), no_output: bool = False, signal: bool,
                   tol: int = 5) -> dict[str, Any]:
    """L5 for a v1.1 run: rules 1 to 10 as :func:`run_checks`, then rules 11, 12 and 13.

    - ``coarse`` is the coarse subtask list (``layers.segment.coarse_subtasks``) as in :func:`run_checks`;
      entries without ``text`` (coarse-pass segments) are ignored, so rule 10 is then ``na``.
    - ``signal`` false: rules 1, 2, 6, 7 and 8 are ``na`` and ``l1`` is never read (a hidden signal stays
      hidden), so no route reason comes from it. ``signal`` true: ``l1`` is required and the rules run as in
      v7, except that rule 6 does not compare the robot items D1a decided from L1.
    - Risk is failed rules over applicable rules; D1b: any failed call (:func:`call_failures`, statuses
      :data:`FAILED_STATUSES`) or ``no_output`` sets it to 1.0 and routes with the reason.
    - Route reasons: D1b; an ``unsure / perception`` item that no signal decides (object items always;
      robot items unless L1 decides them); with a signal, a claim that contradicts it (rules 1, 2, 6, 7, 8)
      and a held object that is not the primary target. Rules 11 to 13 add to the risk only.
    """
    if signal and l1 is None:
        raise ValueError("signal=True needs the L1 record (l1=...)")
    sig_l1 = l1 if signal else {}
    reqs = (goal or {}).get("requirements", []) if goal else []
    text_coarse = [c for c in coarse or [] if isinstance(c, dict) and "text" in c]
    base = run_checks(segments, text_coarse, _rule6_view(goal), sig_l1, objects, facts, list(raw_refs or []),
                      have_inventory=have_inventory, have_facts=have_facts, tol=tol)
    rows = []
    for r in base["checks"]:
        if not signal and r["rule_id"] in SIGNAL_RULES:
            r = _row(r["rule_id"], "na", "no signal (video only): the rule needs one")
        rows.append(r)
    log = [e for e in crawl_log or [] if isinstance(e, dict)]
    rows += [_rule11(log), _rule12(segments, log), _rule13(goal)]

    applicable = [r for r in rows if r["verdict"] != "na"]
    failed = [r for r in applicable if r["verdict"] == "fail"]
    risk = round(len(failed) / len(applicable), 4) if applicable else None
    reasons: list[str] = []
    fails = call_failures(failed_calls)
    if fails:
        reasons.append("failed call(s): " + "; ".join(fails) + " (risk 1.0, D1b)")
        risk = 1.0
    if no_output:
        reasons.append("no output (risk 1.0, D1b)")
        risk = 1.0
    for r in reqs:
        if r.get("status") != "unsure" or r.get("unsure_kind") != "perception":
            continue
        robot = r.get("kind") == "robot_end_state"
        if robot and signal and signal_achieved(r, l1) is not None:
            continue
        what = f"robot item {r.get('predicate')}" if robot else f"item {r.get('predicate')} {r.get('object')}"
        reasons.append(f"unsure / perception {what} cannot be seen and no signal decides it")
        break
    if signal:
        if any(r["rule_id"] in SIGNAL_RULES for r in failed):
            reasons.append("a claim contradicts a signal fact (rules " +
                           ", ".join(str(r["rule_id"]) for r in failed if r["rule_id"] in SIGNAL_RULES) + ")")
        held = _held_after_last_grasp(l1 or {}, facts)
        pt = (goal or {}).get("primary_target") if goal else None
        if held and pt in {o["object_id"] for o in objects} and held != pt:
            reasons.append(f"the held object {held} is not the primary target {pt}")
    return {"checks": rows, "risk": risk, "routed": bool(reasons), "route_reasons": reasons}
