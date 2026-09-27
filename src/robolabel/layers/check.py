"""L5 check layer (V_LITE L5, PLAN 4.2 rules 1 to 10): rules only, no model call, $0.

Each rule gives one ``check`` row per episode: ``pass``, ``fail`` or ``na`` with a short note. Episode
risk = failed rules / applicable rules. The episode is routed for review when a required item is
``unsure / perception`` and no signal decides it, when a claim contradicts a signal fact (rules 1, 2,
6, 7, 8), or when the object held at the end of the last grasp is not the primary target.
"""

from __future__ import annotations

import re
from typing import Any

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
