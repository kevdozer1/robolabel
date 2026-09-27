"""L1 signal layer: probe P5 numbers, synthetic gripper traces, determinism and speed."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from robolabel.layers.signal import (
    Calibration,
    attempt_summary,
    dumps_record,
    gripper_runs,
    keyframe_plan,
    run_l1,
)

DEMO = Path(__file__).resolve().parent.parent / "demo" / "pickplace.json"


def so101_cal(**kw) -> Calibration:
    base = Calibration(layout="so101", fps=30.0, cmd_open=30.0, cmd_closed=0.0, meas_open=30.0, meas_closed=1.0,
                       pause_speed=20.0, withdraw_threshold=20.0)
    return replace(base, **kw)


def libero_cal(**kw) -> Calibration:
    base = Calibration(layout="libero", fps=10.0, cmd_open=-1.0, cmd_closed=1.0, meas_open=0.08,
                       meas_closed=0.0012, hold_margin=0.015, pause_speed=0.02, withdraw_threshold=0.02)
    return replace(base, **kw)


def ramp(a: float, b: float, n: int) -> list[float]:
    return list(np.linspace(a, b, n))


def so101_episode(cmd: list[float], meas: list[float], arm_moves: bool = True):
    n = len(cmd)
    state = np.zeros((n, 6))
    action = np.zeros((n, 6))
    action[:, 5] = cmd
    state[:, 5] = meas
    if arm_moves:
        t = np.arange(n, dtype=float)
        state[:, 0] = 40.0 * np.sin(t / 25.0)
        state[:, 1] = 30.0 * np.cos(t / 30.0)
    return state, action


# ------------------------------------------------------------------------------------------ probe P5
def test_p5_numbers_on_demo_pickplace():
    """probes.md P5: opening 41-50, closing 94-102, opening 148-154, closing 155-161 (episode-range method)."""
    actions = np.array(json.loads(DEMO.read_text(encoding="utf-8"))["actions"], dtype=float)
    g = actions[:, 5]
    cal = so101_cal(cmd_open=float(g.max()), cmd_closed=float(g.min()), merge_gap_s=0.0)
    runs = [(r["type"], r["onset"], r["offset"]) for r in gripper_runs(g, cal, close_sign=-1)]
    assert runs == [("opening", 41, 50), ("closing", 94, 102), ("opening", 148, 154), ("closing", 155, 161)]


def test_p5_with_f1_dataset_calibration_within_two_frames():
    """With the F1 dataset range (stats.json 0.0 to 33.0) the same events move by at most 2 frames."""
    actions = np.array(json.loads(DEMO.read_text(encoding="utf-8"))["actions"], dtype=float)
    cal = so101_cal(cmd_open=32.998, cmd_closed=0.0)
    runs = gripper_runs(actions[:, 5], cal, close_sign=-1)
    ref = [("opening", 41, 50), ("closing", 94, 102), ("opening", 148, 154), ("closing", 155, 161)]
    assert [r["type"] for r in runs] == [t for t, _, _ in ref]
    for r, (_, on, off) in zip(runs, ref, strict=True):
        assert abs(r["onset"] - on) <= 2 and abs(r["offset"] - off) <= 2


# ------------------------------------------------------------------------------------------ synthetic traces
def test_hold_then_release_then_rest_close():
    # open 0-29 at 20, close to 1 over 30-39 (fingers stop at 8: object), hold, open at 100, close to rest at 150
    cmd = [20.0] * 30 + ramp(20, 1, 10) + [1.0] * 60 + ramp(1, 20, 10) + [20.0] * 40 + ramp(20, 1, 10) + [1.0] * 30
    meas = [20.0] * 30 + ramp(20, 8, 10) + [8.0] * 60 + ramp(8, 20, 10) + [20.0] * 40 + ramp(20, 1, 12)[:10] + [1.0] * 30
    s, a = so101_episode(cmd, meas)
    rec = run_l1(s, a, so101_cal())
    assert [x["outcome"] for x in rec["attempts"]] == ["released"]
    assert len(rec["rest_closes"]) == 1
    assert rec["attempts"][0]["closing_onset"] in (29, 30, 31)
    assert rec["attempts"][0]["opening_onset"] in (99, 100, 101)
    holding = [x for x in rec["end_state"] if x["predicate"] == "holding"][0]
    assert holding["value"] is False
    kinds = [c["transition"] for c in rec["candidates"]]
    assert "approach->grasp" in kinds and "transport->release" in kinds
    assert [c["candidate_id"] for c in rec["candidates"]] == [f"c{i}" for i in range(1, len(kinds) + 1)]


def test_slip_after_a_release_stays_an_attempt():
    """review:vlite 4: hold and release at 99, close again at 149 and hold, the gap collapses at about 204
    and the episode ends closed. The slip is a failed attempt, not the gripper going to rest."""
    cmd = [20.0] * 30 + ramp(20, 1, 10) + [1.0] * 59 + ramp(1, 20, 10) + [20.0] * 40 + ramp(20, 1, 10) + [1.0] * 81
    meas = ([20.0] * 30 + ramp(20, 8, 10) + [8.0] * 59 + ramp(8, 20, 10) + [20.0] * 40 + ramp(20, 8, 10)
            + [8.0] * 45 + ramp(8, 1, 5) + [1.0] * 31)
    s, a = so101_episode(cmd, meas)
    rec = run_l1(s, a, so101_cal())
    assert [x["outcome"] for x in rec["attempts"]] == ["released", "slip"]
    assert rec["rest_closes"] == []
    assert 200 <= rec["attempts"][1]["event_frame"] <= 210
    assert "lost the grip" in attempt_summary(rec)


def test_empty_closure_is_missed_grasp_then_retry_holds():
    cmd = [20.0] * 30 + ramp(20, 1, 10) + [1.0] * 30 + ramp(1, 20, 10) + [20.0] * 20 + ramp(20, 1, 10) + [1.0] * 60
    meas = [20.0] * 30 + ramp(20, 1, 10) + [1.0] * 30 + ramp(1, 20, 10) + [20.0] * 20 + ramp(20, 9, 10) + [9.0] * 60
    s, a = so101_episode(cmd, meas)
    rec = run_l1(s, a, so101_cal())
    assert [x["outcome"] for x in rec["attempts"]] == ["empty", "hold"]
    assert rec["attempts"][0]["failure_type"] == "missed_grasp"
    assert "missed grasp" in attempt_summary(rec)
    holding = [x for x in rec["end_state"] if x["predicate"] == "holding"][0]
    assert holding["value"] is True


def test_slip_is_hold_then_gap_collapse_before_opening():
    cmd = [20.0] * 30 + ramp(20, 1, 10) + [1.0] * 90
    meas = [20.0] * 30 + ramp(20, 8, 10) + [8.0] * 40 + ramp(8, 1, 5) + [1.0] * 45
    s, a = so101_episode(cmd, meas)
    rec = run_l1(s, a, so101_cal())
    assert [x["outcome"] for x in rec["attempts"]] == ["slip"]
    assert 78 <= rec["attempts"][0]["event_frame"] <= 86


def test_quick_reopen_is_aborted_not_a_grasp():
    cmd = [20.0] * 30 + ramp(20, 1, 8) + [1.0] * 4 + ramp(1, 20, 8) + [20.0] * 50
    meas = [20.0] * 32 + ramp(20, 10, 8) + [10.0] * 4 + ramp(10, 20, 6) + [20.0] * 50
    s, a = so101_episode(cmd, meas)
    rec = run_l1(s, a, so101_cal())
    assert [x["outcome"] for x in rec["attempts"]] == ["aborted"]


def test_libero_stall_hold_and_empty():
    # attempt 1 closes on nothing (width creeps to the closed value), attempt 2 stalls on a rim at 0.0035
    n1, n2 = 30, 40
    cmd = [-1.0] * 10 + [1.0] * n1 + [-1.0] * 10 + [1.0] * n2 + [-1.0] * 10
    w = ([0.079] * 10 + list(np.linspace(0.079, 0.0013, 20)) + [0.0013] * 10 + list(np.linspace(0.0013, 0.079, 10))
         + list(np.linspace(0.079, 0.0035, 15)) + [0.0035] * 25 + list(np.linspace(0.0035, 0.079, 10)))
    n = len(cmd)
    assert len(w) == n
    state = np.zeros((n, 8))
    state[:, 6] = np.array(w) / 2
    state[:, 7] = -np.array(w) / 2
    state[:, 0] = np.linspace(0, 0.3, n)
    action = np.zeros((n, 7))
    action[:, 6] = cmd
    rec = run_l1(state, action, libero_cal())
    assert [x["outcome"] for x in rec["attempts"]] == ["empty", "released"]


def test_keyframe_plan_at_most_eight_and_merges_close_frames():
    cal = so101_cal()
    attempts = [{"attempt_idx": i, "closing_onset": 40 * i, "closing_offset": 40 * i + 5,
                 "opening_onset": 40 * i + 20, "opening_offset": 40 * i + 25} for i in range(1, 6)]
    plan = keyframe_plan(attempts, 260, cal)
    assert len(plan) <= 8 and plan[0] == 0 and plan[-1] == 259
    assert 40 in plan and 200 in plan  # first and last attempts kept
    assert all(b - a >= 1 for a, b in zip(plan, plan[1:], strict=False))


# ------------------------------------------------------------------------------------------ determinism, speed
def test_byte_identical_and_fast():
    rng = np.random.default_rng(0)
    cmd = [20.0] * 100 + ramp(20, 1, 10) + [1.0] * 200 + ramp(1, 20, 10) + [20.0] * 280
    meas = [20.0] * 100 + ramp(20, 8, 10) + [8.0] * 200 + ramp(8, 20, 10) + [20.0] * 280
    s, a = so101_episode(cmd, meas)
    s[:, :5] += rng.normal(0, 0.2, size=(len(cmd), 5))
    t0 = time.process_time()
    one = dumps_record(run_l1(s, a, so101_cal(), episode_key="F1/99", family="F1"))
    elapsed = time.process_time() - t0
    two = dumps_record(run_l1(s.copy(), a.copy(), so101_cal(), episode_key="F1/99", family="F1"))
    assert one == two
    assert elapsed < 1.0


def test_keyframe_plan_keeps_first_and_last_frames_when_truncating():
    cal = so101_cal(fps=10.0)
    attempts = [{"attempt_idx": 1, "closing_onset": 37, "closing_offset": 40, "opening_onset": 57, "opening_offset": 60},
                {"attempt_idx": 2, "closing_onset": 69, "closing_offset": 72, "opening_onset": 113, "opening_offset": 116}]
    plan = keyframe_plan(attempts, 128, cal)
    assert plan == [0, 37, 57, 69, 74, 113, 118, 127]
