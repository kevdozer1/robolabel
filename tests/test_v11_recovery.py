"""v1.1 recovery after a failed close (SPEC_V1_1 6) and the keyframe plan over every event (Q154).

L1 gains two additive fields, ``recovery_candidates``: ``open_start`` at the reopening onset minus 1 after
an ``empty`` or ``aborted`` close, and ``back_off`` where the arm then starts moving away (low confidence);
and ``recovery_version``, the version of that rule.
The existing fields of the L1 record stay byte-identical (checked against digests taken with the phase 2
code, 4b6832b). The gripper event source turns the candidates into ``gripper_recovery`` events.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from robolabel.events import (
    EVENT_TYPES,
    GripperSource,
    candidate_lines,
    events_from_l1,
    get_source,
    recovery_events,
    sort_events,
    validate_event,
)
from robolabel.layers.signal import (
    BACK_OFF_MIN_DELAY_S,
    LAYOUTS,
    RECOVERY_VERSION,
    Calibration,
    arm_speed,
    back_off_frame,
    dumps_record,
    gripper_signals,
    keyframe_plan,
    keyframe_plan_every_event,
    recovery_candidates,
    run_l1,
)

NEW_L1_FIELDS = ("recovery_candidates", "recovery_version")


# ------------------------------------------------------------------------------------------------ traces
# BEGIN TRACE BUILDERS (numpy only: the digests below were taken by running these builders with 4b6832b)
def _ramp(a, b, n):
    return list(np.linspace(a, b, n))


def _cal(**kw):
    base = Calibration(layout="so101", fps=30.0, cmd_open=30.0, cmd_closed=0.0, meas_open=30.0, meas_closed=1.0,
                       pause_speed=20.0, withdraw_threshold=20.0)
    return replace(base, **kw)


def _so101(cmd, meas, arm):
    arm = np.asarray(arm, dtype=np.float64)
    return (np.column_stack([arm, np.asarray(meas, dtype=np.float64)]),
            np.column_stack([arm, np.asarray(cmd, dtype=np.float64)]))


def trace_back_off(move_at=110):
    """Close on nothing at 40, rest, reopen at about 89, back off (joint 0 from 0 to 60 from ``move_at``),
    come back to 10, close at 180 on the object (hold), carry, release at 250, retract."""
    cmd = np.array([30.0] * 40 + _ramp(30, 1, 10) + [1.0] * 40 + _ramp(1, 30, 10) + [30.0] * 80
                   + _ramp(30, 1, 10) + [1.0] * 60 + _ramp(1, 30, 10) + [30.0] * 60)
    meas = cmd.copy()
    meas[180:260] = np.maximum(cmd[180:260], 10.0)  # the fingers stall on the object: a hold
    n = len(cmd)
    arm = np.zeros((n, 5))
    arm[move_at:move_at + 30, 0] = np.linspace(0.0, 60.0, 30)
    arm[move_at + 30:150, 0] = 60.0
    arm[150:175, 0] = np.linspace(60.0, 10.0, 25)
    arm[175:, 0] = 10.0
    arm[200:230, 1] = np.linspace(0.0, 60.0, 30)
    arm[230:270, 1] = 60.0
    arm[270:300, 1] = np.linspace(60.0, 0.0, 30)
    return _so101(cmd, meas, arm)


def trace_reset_then_reopen_on_the_way_back(rest_until=99):
    """F3/1821's pattern: close on nothing at 40, move away with the fingers closed (55 to 84), rest, then
    move back toward the object (from ``rest_until + 1``) and reopen at about 119, close again at 170."""
    cmd = np.array([30.0] * 40 + _ramp(30, 1, 10) + [1.0] * 70 + _ramp(1, 30, 10) + [30.0] * 40
                   + _ramp(30, 1, 10) + [1.0] * 40 + _ramp(1, 30, 10) + [30.0] * 30)
    meas = cmd.copy()
    meas[170:230] = np.maximum(cmd[170:230], 10.0)
    n = len(cmd)
    arm = np.zeros((n, 5))
    arm[55:85, 0] = np.linspace(0.0, -80.0, 30)
    arm[85:rest_until + 1, 0] = -80.0
    arm[rest_until + 1:165, 0] = np.linspace(-80.0, 5.0, 164 - rest_until)
    arm[165:, 0] = 5.0
    return _so101(cmd, meas, arm)


def trace_libero():
    """LIBERO: attempt 1 closes on nothing, attempt 2 stalls on a rim (from test_signal_l1)."""
    cmd = [-1.0] * 10 + [1.0] * 30 + [-1.0] * 10 + [1.0] * 40 + [-1.0] * 10
    w = ([0.079] * 10 + list(np.linspace(0.079, 0.0013, 20)) + [0.0013] * 10 + list(np.linspace(0.0013, 0.079, 10))
         + list(np.linspace(0.079, 0.0035, 15)) + [0.0035] * 25 + list(np.linspace(0.0035, 0.079, 10)))
    n = len(cmd)
    state = np.zeros((n, 8))
    state[:, 6] = np.array(w) / 2
    state[:, 7] = -np.array(w) / 2
    state[:, 0] = np.linspace(0, 0.3, n)
    action = np.zeros((n, 7))
    action[:, 6] = cmd
    cal = Calibration(layout="libero", fps=10.0, cmd_open=-1.0, cmd_closed=1.0, meas_open=0.08,
                      meas_closed=0.0012, hold_margin=0.015, pause_speed=0.02, withdraw_threshold=0.02)
    return state, action, cal


def existing_fields_digest(rec):
    old = {k: v for k, v in rec.items() if k not in NEW_L1_FIELDS}
    return hashlib.sha256(dumps_record(old).encode("utf-8")).hexdigest()
# END TRACE BUILDERS


# SHA-256 of dumps_record(record without recovery_candidates), taken with the phase 2 code (4b6832b).
DIGESTS_4B6832B = {
    "back_off": "3b9fac11786a6367fe053bba7401623f4285a4559236bf598cd410bbadb93d77",
    "back_off_early": "041b96ec5e9fbae06cb82427a7e27c272ee03af16942716a61b4b7564dee7dd8",
    "reset": "9975142e68646b4829c18c3e833af86a38491395dd7d6a40a6eafbb389f26443",
    "reset_rest": "edc53a9f451b6a92ac9f22a6ced2c71a0fcd477eea09717a53d252eac4064c05",
    "libero": "fdb2a71abf97cf3da67b8ff8c3b59afad2523457a524c7590017f6be986cee23",
}


def _records():
    s, a = trace_back_off()
    s2, a2 = trace_back_off(move_at=92)
    s3, a3 = trace_reset_then_reopen_on_the_way_back()
    s4, a4 = trace_reset_then_reopen_on_the_way_back(rest_until=139)
    s5, a5, cal5 = trace_libero()
    return {
        "back_off": run_l1(s, a, _cal(), episode_key="F1/1", family="F1"),
        "back_off_early": run_l1(s2, a2, _cal(), episode_key="F1/2", family="F1"),
        "reset": run_l1(s3, a3, _cal(), episode_key="F3/3", family="F3"),
        "reset_rest": run_l1(s4, a4, _cal(), episode_key="F3/4", family="F3"),
        "libero": run_l1(s5, a5, cal5, episode_key="F2/5", family="F2"),
    }


# ------------------------------------------------------------------------------------------------ L1 field
def test_existing_l1_fields_are_byte_identical_to_phase_2():
    recs = _records()
    assert {k: existing_fields_digest(r) for k, r in recs.items()} == DIGESTS_4B6832B
    for rec in recs.values():
        assert tuple(rec)[-2:] == NEW_L1_FIELDS  # appended: the order of the old keys is unchanged
        assert rec["recovery_version"] == RECOVERY_VERSION and rec["code_version"] == "l1-2026-09-27.2"


def test_back_off_after_a_rest_at_the_reopen():
    rec = _records()["back_off"]
    assert [x["outcome"] for x in rec["attempts"]] == ["empty", "released"]
    on = rec["attempts"][0]["opening_onset"]
    rc = rec["recovery_candidates"]
    assert [(c["type"], c["confidence"], c["attempt_idx"]) for c in rc] == [
        ("open_start", "high", 1), ("back_off", "low", 1)]
    assert rc[0]["frame"] == on - 1
    assert 105 <= rc[1]["frame"] <= 110  # the arm starts backing off at frame 110 (smoothed speed: a frame early)
    assert all(set(c) == {"type", "frame", "confidence", "attempt_idx"} for c in rc)
    # the released attempt 2 gives no recovery candidate; the v7 candidates have nothing after the failed
    # close of attempt 1 (the gap SPEC_V1_1 6 fills)
    assert all(c["attempt_idx"] == 1 for c in rc)
    assert [c["transition"] for c in rec["candidates"] if c["attempt_idx"] == 1] == ["approach->grasp"]


def test_a_back_off_that_starts_with_the_reopen_is_not_listed():
    rec = _records()["back_off_early"]
    on = rec["attempts"][0]["opening_onset"]
    assert 92 - on < _cal().frames(BACK_OFF_MIN_DELAY_S)
    s, _ = trace_back_off(move_at=92)
    mv = back_off_frame(s[:, :5], arm_speed(s[:, :5], 30.0), s[50, :5], on, rec["num_frames"], _cal())
    assert mv is not None and mv - on < _cal().frames(BACK_OFF_MIN_DELAY_S)  # found, then dropped
    assert [(c["type"], c["frame"]) for c in rec["recovery_candidates"]] == [("open_start", on - 1)]


def test_reset_with_closed_fingers_then_reopen_on_the_way_back():
    """F3/1821: the arm leaves with the fingers closed and reopens while it comes back. The re-open ends
    the failed grasp; the arm never rests after it before the next close, so there is no back-off."""
    rec = _records()["reset"]
    assert [x["outcome"] for x in rec["attempts"]] == ["empty", "released"]
    on = rec["attempts"][0]["opening_onset"]
    assert 115 <= on <= 121
    assert [(c["type"], c["frame"], c["attempt_idx"]) for c in rec["recovery_candidates"]] == [
        ("open_start", on - 1, 1)]


def test_moving_back_toward_the_grasp_pose_is_not_a_back_off():
    rec = _records()["reset_rest"]  # reopens while resting far away, then moves back toward the object
    assert [c["type"] for c in rec["recovery_candidates"]] == ["open_start"]


def test_aborted_close_and_libero():
    cmd = [20.0] * 30 + _ramp(20, 1, 8) + [1.0] * 4 + _ramp(1, 20, 8) + [20.0] * 50
    meas = [20.0] * 32 + _ramp(20, 10, 8) + [10.0] * 4 + _ramp(10, 20, 6) + [20.0] * 50
    s, a = _so101(cmd, meas, np.zeros((len(cmd), 5)))
    rec = run_l1(s, a, _cal(cmd_open=20.0, meas_open=20.0))
    assert [x["outcome"] for x in rec["attempts"]] == ["aborted"]
    assert rec["recovery_candidates"] == [{"type": "open_start", "frame": rec["attempts"][0]["opening_onset"] - 1,
                                           "confidence": "high", "attempt_idx": 1}]
    lib = _records()["libero"]
    assert [x["outcome"] for x in lib["attempts"]] == ["empty", "released"]
    assert [(c["type"], c["frame"]) for c in lib["recovery_candidates"]] == [
        ("open_start", lib["attempts"][0]["opening_onset"] - 1)]  # the arm never rests: no back-off


def test_no_recovery_without_a_failed_close_that_reopens():
    # hold then release, then a rest close; a slip; an empty close that never reopens
    for cmd, meas in [
        ([20.0] * 30 + _ramp(20, 1, 10) + [1.0] * 60 + _ramp(1, 20, 10) + [20.0] * 40,
         [20.0] * 30 + _ramp(20, 8, 10) + [8.0] * 60 + _ramp(8, 20, 10) + [20.0] * 40),
        ([20.0] * 30 + _ramp(20, 1, 10) + [1.0] * 90, [20.0] * 30 + _ramp(20, 8, 10) + [8.0] * 40
         + _ramp(8, 1, 5) + [1.0] * 45),
        ([20.0] * 30 + _ramp(20, 1, 10) + [1.0] * 90, [20.0] * 30 + _ramp(20, 1, 10) + [1.0] * 90),
    ]:
        s, a = _so101(cmd, meas, np.zeros((len(cmd), 5)))
        rec = run_l1(s, a, _cal(cmd_open=20.0, meas_open=20.0))
        assert rec["attempts"] and all(x["outcome"] in ("released", "slip", "empty") for x in rec["attempts"])
        assert rec["recovery_candidates"] == []


def test_recovery_is_deterministic_and_bounded():
    s, a = trace_back_off()
    one = dumps_record(run_l1(s, a, _cal(), episode_key="F1/1"))
    two = dumps_record(run_l1(s.copy(), a.copy(), _cal(), episode_key="F1/1"))
    assert one == two and '"recovery_candidates":[{' in one
    rec = run_l1(s, a, _cal())
    n = rec["num_frames"]
    assert all(0 <= c["frame"] < n - 1 for c in rec["recovery_candidates"])
    # the function alone, on the record's own inputs, gives the field
    sig = gripper_signals(s, a, LAYOUTS["so101"])
    speed = arm_speed(sig["arm"], 30.0)
    assert recovery_candidates(rec["attempts"], rec["events"], sig, speed, _cal()) == rec["recovery_candidates"]
    assert recovery_candidates([], rec["events"], sig, speed, _cal()) == []


def test_back_off_frame_rules():
    cal = _cal()
    n = 120
    arm = np.zeros((n, 5))
    arm[60:90, 0] = np.linspace(0.0, 60.0, 30)
    arm[90:, 0] = 60.0
    speed = np.zeros(n)
    speed[60:90] = 60.0
    ref = np.zeros(5)
    assert back_off_frame(arm, speed, ref, 10, n, cal) == 60
    assert back_off_frame(arm, speed, ref, 10, 61, cal) is None  # the next close comes first
    assert back_off_frame(arm, speed, np.array([80.0, 0, 0, 0, 0]), 10, n, cal) is None  # toward the reference
    moving = np.full(n, 60.0)
    assert back_off_frame(arm, moving, ref, 10, n, cal) is None  # never rests
    assert back_off_frame(arm, speed, ref, 10, n, replace(cal, pause_speed=0.0)) is None  # no pause speed


# ------------------------------------------------------------------------------------------------ events
def test_gripper_source_adds_recovery_events():
    rec = _records()["back_off"]
    events = events_from_l1(rec)
    assert events == sort_events(events) and all(validate_event(e) == [] for e in events)
    rc = rec["recovery_candidates"]
    rec_events = [e for e in events if e["source"] == "gripper_recovery"]
    assert [(e["type"], e["frame"], e["attempt_idx"], e["confidence"]) for e in rec_events] == [
        ("open_start", rc[0]["frame"] + 1, 1, 0.9), ("back_off", rc[1]["frame"] + 1, 1, 0.3)]
    # the re-open is one event, labelled as recovery; the release of attempt 2 stays a plain gripper event
    opens = [e for e in events if e["type"] == "open_start"]
    assert [e["source"] for e in opens] == ["gripper_recovery", "gripper"]
    assert recovery_events(rec) == rec_events
    plain = events_from_l1(rec, include_recovery=False)
    assert plain == events_from_l1({k: v for k, v in rec.items() if k != "recovery_candidates"})
    assert {e["source"] for e in plain} == {"gripper"} and "back_off" not in {e["type"] for e in plain}
    assert [e for e in events if e["type"] != "back_off"] == [
        dict(e, source="gripper_recovery") if (e["type"], e["frame"]) == ("open_start", rc[0]["frame"] + 1) else e
        for e in plain]
    assert GripperSource().events(None, l1=rec) == events
    assert get_source("gripper", include_recovery=False).events(None, l1=rec) == plain
    lines = candidate_lines(events, rec["num_frames"], rec["fps"])
    assert any(line.endswith("open_start (gripper_recovery)") for line in lines)
    assert any(line.endswith("back_off (gripper_recovery)") for line in lines)
    assert "back_off" not in EVENT_TYPES  # the v1.1 core types are unchanged


def test_recovery_events_of_old_records_and_edges():
    assert recovery_events({"num_frames": 50}) == []
    rec = {"num_frames": 50, "events": [], "attempts": [], "candidates": [],
           "recovery_candidates": [{"type": "open_start", "frame": 49, "confidence": "high", "attempt_idx": 1},
                                   {"type": "back_off", "frame": 20, "confidence": "low", "attempt_idx": 1}]}
    # onset 50 is past the last frame and is dropped; onset 21 stays
    assert [(e["type"], e["frame"]) for e in events_from_l1(rec)] == [("back_off", 21)]


# ------------------------------------------------------------------------------------------------ Q154
def _l1(events, attempts, n=300):
    return {"num_frames": n, "events": events, "attempts": attempts}


def test_every_event_plan_equals_the_old_plan_when_every_event_is_an_attempt_onset():
    for rec in _records().values():
        if rec["layout"] != "so101":
            continue
        cal = _cal()
        attempt_onsets = {a["closing_onset"] for a in rec["attempts"]} | {
            a["opening_onset"] for a in rec["attempts"] if a["opening_onset"] is not None}
        if {e["onset"] for e in rec["events"]} == attempt_onsets and not any(a["recloses"] for a in rec["attempts"]):
            assert keyframe_plan_every_event(rec, rec["num_frames"], cal) == rec["keyframes"]
    cal = _cal(fps=10.0)
    attempts = [{"attempt_idx": 1, "closing_onset": 37, "closing_offset": 40, "opening_onset": 57, "opening_offset": 60},
                {"attempt_idx": 2, "closing_onset": 69, "closing_offset": 72, "opening_onset": 113,
                 "opening_offset": 116}]
    events = [{"type": "closing", "onset": 37, "offset": 40}, {"type": "opening", "onset": 57, "offset": 60},
              {"type": "closing", "onset": 69, "offset": 72}, {"type": "opening", "onset": 113, "offset": 116}]
    assert keyframe_plan_every_event(_l1(events, attempts, 128), 128, cal) == keyframe_plan(attempts, 128, cal)


def test_every_event_plan_adds_early_openings_recloses_and_rest_closes():
    cal = _cal()
    attempts = [{"attempt_idx": 1, "closing_onset": 60, "closing_offset": 70,
                 "recloses": [{"onset": 110, "offset": 118}], "opening_onset": 160, "opening_offset": 170}]
    events = [{"type": "opening", "onset": 10, "offset": 20}, {"type": "closing", "onset": 60, "offset": 70},
              {"type": "closing", "onset": 110, "offset": 118}, {"type": "opening", "onset": 160, "offset": 170},
              {"type": "closing", "onset": 240, "offset": 250}]
    old = keyframe_plan(attempts, 300, cal)
    new = keyframe_plan_every_event(_l1(events, attempts), 300, cal)
    assert old == [0, 60, 78, 160, 178, 299]
    # 12 frames for 8 slots: as in keyframe_plan, the frames of the first and last attempts come first
    # (here the re-close at 110 and its settled frame, which the old plan never had), then the rest
    assert new == [0, 60, 78, 110, 126, 160, 178, 299]
    # with free slots, the opening before the first close and the rest close come in too
    assert keyframe_plan_every_event(_l1(events, attempts), 300, cal, max_frames=20) == [
        0, 10, 28, 60, 78, 110, 126, 160, 178, 240, 258, 299]
    assert keyframe_plan_every_event(_l1(events, attempts), 300, cal, max_frames=10) == [
        0, 10, 28, 60, 78, 110, 126, 160, 178, 299]  # time order among the frames of no attempt
    assert keyframe_plan_every_event(_l1([], []), 300, cal) == [0, 299]


def test_every_event_plan_trims_to_max_frames_like_the_old_plan():
    cal = _cal()
    attempts = [{"attempt_idx": i, "closing_onset": 40 * i, "closing_offset": 40 * i + 5, "recloses": [],
                 "opening_onset": 40 * i + 20, "opening_offset": 40 * i + 25} for i in range(1, 6)]
    events = [{"type": "opening", "onset": 5, "offset": 12}]
    for a in attempts:
        events += [{"type": "closing", "onset": a["closing_onset"], "offset": a["closing_offset"]},
                   {"type": "opening", "onset": a["opening_onset"], "offset": a["opening_offset"]}]
    plan = keyframe_plan_every_event(_l1(events, attempts, 260), 260, cal)
    assert len(plan) == 8 and plan[0] == 0 and plan[-1] == 259 and plan == sorted(set(plan))
    assert {40, 60, 200, 220} <= set(plan)  # the onsets of the first and last attempts come first
    assert plan == keyframe_plan_every_event(_l1(events, attempts, 260), 260, cal)
