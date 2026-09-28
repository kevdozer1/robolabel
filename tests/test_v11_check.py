"""L5 for v1.1 (SPEC_V1_1 3.4 and Q155 b, D1b): signal rules na without a signal, the video-only rules 11 to 13, and
risk 1.0 with a route reason for failed calls or no output. Fake callers only; no network, no model calls."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from robolabel.episode import Episode
from robolabel.layers.check import FAILED_STATUSES, call_failures, run_checks, run_checks_v11
from robolabel.layers.crawl import crawl_boundaries
from robolabel.layers.goal import postprocess_goal_v11
from robolabel.providers.base import CallResult

CAM = "observation.images.up"
OBJECTS = [{"object_id": "o1", "name": "pink brick", "aliases": [], "category": "block", "views": []},
           {"object_id": "o2", "name": "transparent box", "aliases": [], "category": "container", "views": []}]


# ------------------------------------------------------------------------------------------ fixtures
def seg(start, end, phase="other", end_event="other", **kw):
    d = {"start_frame": start, "end_frame": end, "phase_class": phase, "phase_text": f"{phase} from {start}",
         "target": "o1", "destination": "none", "attempt_idx": 1, "outcome": "success", "attempt_outcome": "success",
         "failure_type": "none", "mistake": False, "end_event": end_event, "boundary_source": "coarse",
         "coarse_end_frame": end, "crawl_calls": 0, "candidate_id": "none", "evidence": []}
    d.update(kw)
    return d


def l1_with(closings=(), openings=(), end_state=None, attempts=None):
    events = [{"type": "closing", "onset": f} for f in closings] + [{"type": "opening", "onset": f} for f in openings]
    return {"events": events, "attempts": attempts or [], "end_state": end_state or [], "candidates": []}


def goal_with(reqs, has_end_state=True, target="o1"):
    for r in reqs:
        r.setdefault("added_by", "model")
    return {"objective_text": "The pink brick is inside the transparent box.", "primary_target": target,
            "primary_destination": "o2", "requirements": reqs, "episode_outcome": "unknown",
            "has_end_state": has_end_state, "goal_command": ""}


def req(pred, value=True, achieved=True, *, kind="robot_end_state", obj="none", ref="none", status="required",
        uk=None, basis="observed"):
    return {"req_id": "r1", "kind": kind, "object": obj, "predicate": pred, "ref_object": ref, "value": value,
            "status": status, "unsure_kind": uk, "basis": basis, "achieved": achieved, "deciding_frame": 99,
            "deciding_camera": "", "visibility": {}, "reason": ""}


def check(segments, goal=None, *, signal=False, l1=None, crawl_log=None, failed_calls=(), no_output=False,
          coarse=(), facts=(), objects=OBJECTS, raw_refs=()):
    return run_checks_v11(segments, list(coarse), goal, l1=l1, objects=objects, facts=list(facts),
                          raw_refs=list(raw_refs), crawl_log=crawl_log, have_inventory=bool(objects),
                          have_facts=bool(facts), failed_calls=failed_calls, no_output=no_output, signal=signal)


def rule(res, n):
    return next(c for c in res["checks"] if c["rule_id"] == n)


class Hidden(dict):
    """An L1 record that must never be read."""

    def _no(self, *a, **k):
        raise AssertionError("the L1 record was read although signal is False")

    get = __getitem__ = __iter__ = __contains__ = keys = items = values = __len__ = _no


# a grasp far from any closing onset, a release without an opening, and a final retract the arm did not make
BAD_FOR_SIGNAL = [seg(0, 29, "approach", "close_start"), seg(30, 59, "grasp", "other"),
                  seg(60, 79, "release", "other"), seg(80, 99, "retract", "other", target="none")]
L1_BAD = l1_with(closings=[5], openings=[], end_state=[
    {"predicate": "holding", "value": True, "confidence": "high"},
    {"predicate": "withdrawn", "value": False, "confidence": "low"}],
    attempts=[{"attempt_idx": 1, "outcome": "empty", "closing_onset": 5, "event_frame": 12, "hold_frame": None}])


# ------------------------------------------------------------------------------------------ signal rules
def test_signal_rules_are_na_without_a_signal_and_l1_is_not_read():
    goal = goal_with([req("holding", False, True)])
    res = check(BAD_FOR_SIGNAL, goal, signal=False, l1=Hidden())
    for n in (1, 2, 6, 7, 8):
        assert rule(res, n)["verdict"] == "na" and "no signal" in rule(res, n)["note"]
    assert not res["routed"] and res["route_reasons"] == []
    assert [c["rule_id"] for c in res["checks"]] == list(range(1, 14))


def test_with_a_signal_rules_1_to_10_are_the_v7_rules():
    goal = goal_with([req("holding", False, True)])
    res = check(BAD_FOR_SIGNAL, goal, signal=True, l1=L1_BAD)
    v7 = run_checks(BAD_FOR_SIGNAL, [], goal, L1_BAD, OBJECTS, [], [], have_inventory=True, have_facts=False)
    assert res["checks"][:10] == v7["checks"]
    assert {n: rule(res, n)["verdict"] for n in (1, 2, 6, 7, 8)} == \
        {1: "fail", 2: "fail", 6: "fail", 7: "fail", 8: "fail"}
    assert res["routed"] and any("contradicts a signal fact (rules 1, 2, 6, 7, 8)" in r for r in res["route_reasons"])
    with pytest.raises(ValueError):
        check(BAD_FOR_SIGNAL, goal, signal=True, l1=None)


def test_rule6_does_not_compare_items_d1a_decided_from_l1():
    l1 = l1_with(end_state=[{"predicate": "holding", "value": True, "confidence": "high"}])
    decided = req("holding", False, False, basis="signal")  # D1a wrote achieved from L1 (still holding)
    res = check([seg(0, 99)], goal_with([decided]), signal=True, l1=l1)
    assert rule(res, 6)["verdict"] == "na"
    claimed = req("holding", False, True)  # the model's own claim: holding nothing, contradicted by L1
    res = check([seg(0, 99)], goal_with([decided, claimed]), signal=True, l1=l1)
    assert rule(res, 6)["verdict"] == "fail" and res["routed"]


# ------------------------------------------------------------------------------------------ rule 11
def entry(i, coarse, onset, pick, s1, stage2=None, retry=None):
    calls = [{"stage": "stage1", "frames": s1}]
    if retry:
        calls.append({"stage": "retry", "frames": retry})
    if stage2:
        calls.append({"stage": "stage2", "frames": stage2})
    return {"boundary_index": i, "event_type": "close_start", "coarse_frame": coarse, "stage1_frames": s1,
            "retry_frames": retry, "stage2_frames": stage2, "pick": pick, "onset": onset, "flags": [],
            "calls": len(calls), "call_log": calls}


S1 = [20, 29, 37, 46, 54, 63, 71, 80]


def test_rule11_picks_lie_inside_their_windows():
    segs = [seg(0, 45, end_event="close_start"), seg(46, 99)]
    ok = check(segs, crawl_log=[entry(0, 50, 46, 46, S1)])
    assert rule(ok, 11)["verdict"] == "pass"
    stage2 = check(segs, crawl_log=[entry(0, 50, 42, 42, S1, stage2=[37, 38, 39, 40, 41, 42, 43, 44, 45, 46][:8])])
    assert rule(stage2, 11)["verdict"] == "pass"
    edge9 = check([seg(0, 80, end_event="close_start"), seg(81, 99)], crawl_log=[entry(0, 50, 81, 80, S1)])
    assert rule(edge9, 11)["verdict"] == "pass"  # the onset after a 9 is one frame after the window, by design
    outside = check(segs, crawl_log=[entry(0, 50, 46, 90, S1)])
    assert rule(outside, 11)["verdict"] == "fail" and "pick 90" in rule(outside, 11)["note"]
    assert rule(outside, 11)["note"] and outside["risk"] > 0 and not outside["routed"]
    none = check(segs, crawl_log=[dict(entry(0, 50, 50, None, S1), flags=["crawl_none"])])
    assert rule(none, 11)["verdict"] == "na" and rule(check(segs), 11)["verdict"] == "na"


# ------------------------------------------------------------------------------------------ rule 12
def test_rule12_moved_boundaries_stay_between_their_neighbours():
    segs = [seg(0, 39), seg(40, 69, end_event="close_start"), seg(70, 99)]
    moved = copy.deepcopy(segs)
    moved[1].update(end_frame=64, boundary_source="crawl", coarse_end_frame=69)
    moved[2]["start_frame"] = 65
    res = check(moved, crawl_log=[entry(1, 70, 65, 65, S1)])
    assert rule(res, 12)["verdict"] == "pass"
    crossed = copy.deepcopy(segs)  # boundary 1 moved from 70 to 30, before boundary 0 at 40
    crossed[1].update(end_frame=29, boundary_source="crawl", coarse_end_frame=69)
    crossed[2]["start_frame"] = 30
    res = check(crossed, crawl_log=[entry(1, 70, 30, 30, [40, 50, 60, 70])])
    assert rule(res, 12)["verdict"] == "fail" and "crosses a neighbour" in rule(res, 12)["note"]
    last = copy.deepcopy(segs)  # the last boundary moved past the clip end
    last[1].update(end_frame=99, boundary_source="crawl", coarse_end_frame=69)
    last[2]["start_frame"] = 100
    assert rule(check(last, crawl_log=[entry(1, 70, 100, 99, S1)]), 12)["verdict"] == "fail"
    mismatch = check(moved, crawl_log=[entry(1, 70, 66, 66, S1)])
    assert rule(mismatch, 12)["verdict"] == "fail" and "crawl onset 66" in rule(mismatch, 12)["note"]
    from_segments = check(moved, crawl_log=[])  # a crawl-sourced end that differs from its coarse end counts
    assert rule(from_segments, 12)["verdict"] == "pass"
    assert rule(check(segs, crawl_log=[entry(1, 70, 70, None, S1)]), 12)["verdict"] == "na"
    assert rule(check(segs, crawl_log=[entry(7, 70, 65, 65, S1)]), 12)["verdict"] == "fail"


def moving_clip(n=300, fps=30.0):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(n, 48, 64, 3), dtype=np.uint8)

    def get(i):
        return frames[int(i)]

    return Episode(episode_id="F1/0", num_frames=n, fps=fps, task="put the block in the box", get_frame=get,
                   camera_key=CAM, extra={"cameras": {CAM: get}, "camera_order": [CAM]})


class Scripted:
    name = "scripted"
    model = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)

    def call(self, req):
        a = self.answers.pop(0)
        return CallResult(True, {"answer": a}, json.dumps({"answer": a}), "ok", usd=0.001, wall_s=0.1)


def test_rules_11_and_12_pass_on_a_real_crawl():
    """Stage 1 then stage 2 on the first boundary; 9 twice on the second (its onset is one frame after the
    retry window)."""
    ep = moving_clip()
    coarse = [seg(0, 99, "approach", "close_start"), seg(100, 199, "grasp", "open_start"), seg(200, 299, "retract")]
    caller = Scripted([4, 5, 9, 9])
    segments, log, _ = crawl_boundaries(ep, coarse, caller, camera=CAM, context={}, reasoning=None)
    assert caller.answers == []
    assert [e["pick"] is not None for e in log] == [True, True]
    assert log[1]["onset"] == log[1]["pick"] + 1 and log[1]["onset"] > max(log[1]["retry_frames"])
    res = check(segments, crawl_log=log)
    assert rule(res, 11)["verdict"] == "pass" and rule(res, 12)["verdict"] == "pass"
    assert "2 moved" in rule(res, 12)["note"]


# ------------------------------------------------------------------------------------------ rule 13
def test_rule13_no_end_state_means_no_object_end_state_item():
    invented = goal_with([req("on_top_of", kind="object_end_state", obj="o1", ref="o2")], has_end_state=False)
    res = check([seg(0, 99)], invented)
    assert rule(res, 13)["verdict"] == "fail" and "on_top_of o1" in rule(res, 13)["note"]
    assert res["risk"] > 0 and not res["routed"]
    clean = goal_with([req("holding", False, True)], has_end_state=False)
    assert rule(check([seg(0, 99)], clean), 13)["verdict"] == "pass"
    with_state = goal_with([req("inside", kind="object_end_state", obj="o1", ref="o2")], has_end_state=True)
    assert rule(check([seg(0, 99)], with_state), 13)["verdict"] == "na"
    assert rule(check([seg(0, 99)], goal_with([], has_end_state=None)), 13)["verdict"] == "na"
    assert rule(check([seg(0, 99)], None), 13)["verdict"] == "na"


def test_rule13_rejects_a_postprocessed_goal_with_an_invented_end_state():
    data = {"objective_text": "The person waves.", "has_end_state": False, "primary_target": "none",
            "primary_destination": "none",
            "requirements": [{"req_id": "r1", "kind": "object_end_state", "object": "o1", "predicate": "lifted",
                              "ref_object": "none", "value": "true", "status": "required", "unsure_kind": "none",
                              "basis": "observed", "achieved": "true", "deciding_frame": 99, "deciding_camera": "",
                              "visibility": [], "reason": ""}]}

    class Ep:
        episode_id, num_frames, fps, task = "C/no_end_state", 100, 20.0, ""
        extra = {"camera_order": ["video"]}

    goal = postprocess_goal_v11(data, Ep(), objects=OBJECTS, segments=[], repairs=[], signal=False)
    assert rule(check([seg(0, 99)], goal), 13)["verdict"] == "fail"


# ------------------------------------------------------------------------------------------ D1b
def test_a_failed_call_gives_risk_one_and_routes_with_the_reason():
    clean = check([seg(0, 99)], goal_with([]))
    assert clean["risk"] != 1.0 and not clean["routed"]
    for status in ("invalid", "refused", "failed", "unavailable"):
        res = check([seg(0, 99)], goal_with([]), failed_calls=[{"step": "coarse", "status": status}])
        assert res["risk"] == 1.0 and res["routed"]
        assert res["route_reasons"][0] == f"failed call(s): coarse call {status} (risk 1.0, D1b)"
    results = [("scene_inventory", CallResult(True, {}, "{}", "ok")),
               ("goal", CallResult(False, None, "", "refused", error="spend guard: bucket cap")),
               CallResult(False, None, "", "failed", receipt={"step": "crawl"}),
               "scene_facts invalid after its repair retry"]
    res = check([seg(0, 99)], goal_with([]), failed_calls=results)
    assert res["risk"] == 1.0
    assert res["route_reasons"][0] == ("failed call(s): goal call refused: spend guard: bucket cap; crawl call failed; "
                                       "scene_facts invalid after its repair retry (risk 1.0, D1b)")
    only_ok = check([seg(0, 99)], goal_with([]), failed_calls=[CallResult(True, {}, "{}", "ok")])
    assert only_ok["risk"] != 1.0 and not only_ok["routed"]
    assert call_failures([{"step": "goal", "status": "stopped"}, {"step": "goal", "status": "not run"},
                          {"step": "x", "status": "ok"}]) == ["goal call stopped", "goal call not run"]
    assert set(FAILED_STATUSES) >= {"invalid", "refused", "failed", "unavailable"}


def test_no_output_gives_risk_one_and_routes():
    res = check([seg(0, 99)], None, no_output=True, objects=[])
    assert res["risk"] == 1.0 and res["routed"] and res["route_reasons"] == ["no output (risk 1.0, D1b)"]
    both = check([seg(0, 99)], None, no_output=True, failed_calls=[("coarse", "invalid"), ("goal", "invalid")],
                 objects=[])
    assert both["route_reasons"] == ["failed call(s): coarse call invalid; goal call invalid (risk 1.0, D1b)",
                                     "no output (risk 1.0, D1b)"]


# ------------------------------------------------------------------------------------------ other route reasons
def test_unsure_perception_items_route_unless_a_signal_decides_them():
    obj = req("inside", "unsure", "unknown", kind="object_end_state", obj="o1", ref="o2", status="unsure",
              uk="perception")
    for signal, l1 in ((False, None), (True, l1_with())):
        res = check([seg(0, 99)], goal_with([obj]), signal=signal, l1=l1)
        assert res["routed"] and "item inside o1 cannot be seen" in res["route_reasons"][0]
    robot = req("withdrawn", True, "unknown", status="unsure", uk="perception")
    res = check([seg(0, 99)], goal_with([robot]))
    assert res["routed"] and "robot item withdrawn" in res["route_reasons"][0]
    decides = l1_with(end_state=[{"predicate": "withdrawn", "value": True, "confidence": "low"}])
    assert not check([seg(0, 99)], goal_with([robot]), signal=True, l1=decides)["routed"]
    home = req("at_home_pose", True, "unknown", status="unsure", uk="perception")
    assert check([seg(0, 99)], goal_with([home]), signal=True, l1=decides)["routed"]


def test_rule10_reads_coarse_subtasks_only():
    segs = [seg(0, 99)]
    passing = check(segs, coarse=[{"coarse_idx": 0, "start_frame": 0, "end_frame": 99, "text": "put the brick in the box",
                                   "mistake": False}])
    assert rule(passing, 10)["verdict"] == "pass"
    assert rule(check(segs, coarse=segs), 10)["verdict"] == "na"  # coarse-pass segments have no text


def test_run_checks_v11_is_deterministic_and_leaves_its_inputs_alone():
    segs = copy.deepcopy(BAD_FOR_SIGNAL)
    goal = goal_with([req("holding", False, True, basis="signal")])
    log = [entry(0, 50, 46, 46, S1)]
    before = json.dumps([segs, goal, log], sort_keys=True)
    runs = [json.dumps(check(segs, goal, signal=True, l1=L1_BAD, crawl_log=log,
                             failed_calls=[("goal", "invalid")]), sort_keys=True) for _ in range(3)]
    assert len(set(runs)) == 1
    assert json.dumps([segs, goal, log], sort_keys=True) == before
