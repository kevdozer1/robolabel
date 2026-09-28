"""Phase 3 pieces working together, with fakes only (no network).

- The gripper source's re-open after a failed close (SPEC_V1_1 6, source label ``gripper_recovery``) snaps its
  boundary in the full pipeline (SPEC 3.3 item 10) and is not crawled, and the failed attempt follows SPEC 4.
- A view written by the full pipeline scores under the v1.1 failure rule (S4 from its attempt records).
"""

from __future__ import annotations

import copy
import json
import re

import numpy as np

from robolabel.episode import Episode
from robolabel.eval.score import score_view
from robolabel.events import events_from_l1
from robolabel.layers.signal import Calibration, run_l1
from robolabel.providers.base import CallResult, TextPart
from robolabel.vfirst import run_episode_v11

CAM = "video"
CAND_RE = re.compile(r"^(c\d+): frame (\d+) \([\d.]+ s\), (\w+) \((\w+)\)$", re.MULTILINE)
OBJECTS = [{"object_id": "o1", "name": "pink brick", "aliases": ["brick"], "category": "block",
            "views": [{"camera": CAM, "visible": True, "x": 0.3, "y": 0.5, "x0": 0.2, "y0": 0.4, "x1": 0.4, "y1": 0.6}]},
           {"object_id": "o2", "name": "box", "aliases": [], "category": "container",
            "views": [{"camera": CAM, "visible": True, "x": 0.7, "y": 0.5, "x0": 0.6, "y0": 0.4, "x1": 0.8, "y1": 0.6}]}]


# ------------------------------------------------------------------------------------------ fixtures
def clip(n: int, fps: float, key: str) -> Episode:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[10:30, 10:30] = (200, 30, 30)
    return Episode(episode_id=key, num_frames=n, fps=fps, task="put the pink brick in the box",
                   get_frame=lambda i: frame, camera_key=CAM)


def _ramp(a, b, n):
    return list(np.linspace(a, b, n))


def failed_close_l1() -> dict:
    """Close on nothing at 40 and reopen at about 89, back off, close on the object at 180 (hold), release at 250."""
    cmd = np.array([30.0] * 40 + _ramp(30, 1, 10) + [1.0] * 40 + _ramp(1, 30, 10) + [30.0] * 80
                   + _ramp(30, 1, 10) + [1.0] * 60 + _ramp(1, 30, 10) + [30.0] * 60)
    meas = cmd.copy()
    meas[180:260] = np.maximum(cmd[180:260], 10.0)
    n = len(cmd)
    arm = np.zeros((n, 5))
    arm[110:140, 0] = np.linspace(0.0, 60.0, 30)
    arm[140:150, 0] = 60.0
    arm[150:175, 0] = np.linspace(60.0, 10.0, 25)
    arm[175:, 0] = 10.0
    state = np.column_stack([arm, meas])
    action = np.column_stack([arm, cmd])
    cal = Calibration(layout="so101", fps=30.0, cmd_open=30.0, cmd_closed=0.0, meas_open=30.0, meas_closed=1.0,
                      pause_speed=20.0, withdraw_threshold=20.0)
    return run_l1(state, action, cal, episode_key="C/synthetic", family="C")


def seg(start, end, end_event, phase_class, *, target="o1", destination="none", candidate_id="none", attempt_idx=1,
        outcome="success", attempt_outcome="success", failure_type="none"):
    return {"start_frame": start, "end_frame": end, "phase_class": phase_class, "phase_text": f"{phase_class} (fake)",
            "end_event": end_event, "target": target, "destination": destination, "attempt_idx": attempt_idx,
            "outcome": outcome, "attempt_outcome": attempt_outcome, "failure_type": failure_type,
            "candidate_id": candidate_id}


def texts(req) -> str:
    return "\n".join(p.text for p in req.parts if isinstance(p, TextPart))


class PlanCaller:
    """Answers every step; the coarse plan is a list of segments or a function of the request."""

    name = "fake"
    model = "fake-model"

    def __init__(self, plan):
        self.plan = plan
        self.requests = []

    def call(self, req):
        self.requests.append(req)
        if req.step == "scene_inventory":
            data = {"objects": OBJECTS}
        elif req.step == "coarse":
            data = {"segments": self.plan(req) if callable(self.plan) else self.plan}
        elif req.step == "crawl":
            data = {"answer": 4}
        elif req.step == "scene_facts":
            data = {"facts": [{"frame": f, "camera": CAM, "visible": ["o1", "o2"], "partial": [], "in_gripper": "none",
                               "relations": [], "boxes": []} for f in req.context["frame_indices"]]}
        elif req.step == "goal":
            data = {"objective_text": "the pink brick is inside the box", "has_end_state": True,
                    "primary_target": "o1", "primary_destination": "o2", "requirements": []}
        else:
            raise AssertionError(req.step)
        return CallResult(True, copy.deepcopy(data), json.dumps(data), "ok", usd=0.001, wall_s=0.1)


# ------------------------------------------------------------------------------------------ recovery in the pipeline
def test_the_reopen_after_a_failed_close_snaps_its_boundary_in_the_full_pipeline():
    l1 = failed_close_l1()
    n = int(l1["num_frames"])
    events = events_from_l1(l1)
    reopen = [e["frame"] for e in events if e["type"] == "open_start" and e["source"] == "gripper_recovery"]
    close = [e["frame"] for e in events if e["type"] == "close_start" and e["source"] == "gripper"]
    assert reopen and close and close[0] < reopen[0] < 150

    def plan(req):
        cands = CAND_RE.findall(texts(req))
        cid_close = next(c for c, f, t, s in cands if t == "close_start" and s == "gripper" and int(f) == close[0])
        cid_reopen = next(c for c, f, t, s in cands if t == "open_start" and s == "gripper_recovery")
        return [seg(0, close[0] - 1, "close_start", "approach", target="none", candidate_id=cid_close,
                    attempt_outcome="failed"),
                seg(close[0], reopen[0] + 2, "open_start", "grasp", candidate_id=cid_reopen, outcome="failed",
                    attempt_outcome="failed", failure_type="missed_grasp"),
                seg(reopen[0] + 3, 150, "other", "retract", target="none", attempt_outcome="failed"),
                seg(151, n - 1, "other", "approach", attempt_idx=2)]

    caller = PlanCaller(plan)
    out = run_episode_v11(clip(n, 30.0, "C/synthetic"), camera=CAM, caller=caller, event_source="gripper", l1=l1,
                          context={"arm": "E2-gripper", "episode_key": "C/synthetic"})
    segs = out["segments"]
    # the re-open boundary takes the L1 frame (boundary_source signal) and is not crawled
    assert segs[2]["start_frame"] == reopen[0] and segs[1]["end_frame"] == reopen[0] - 1
    assert segs[1]["boundary_source"] == "signal" and segs[1]["coarse_end_frame"] == reopen[0] + 2
    assert any("snapped to" in r and f"at {reopen[0]}" in r for r in out["repairs"])
    by_boundary = {e["boundary_index"]: e for e in out["crawl_log"]}
    assert by_boundary[1]["flags"] == ["skipped_signal"] and by_boundary[1]["calls"] == 0
    assert "crawl" not in [r.step for r in caller.requests]
    # SPEC 4 on the failed attempt: only the grasp failed, every phase carries the attempt's outcome
    view = out["view"]
    assert view["event_sources"] == ["gripper"]
    assert [(s["outcome"], s["mistake"], s["attempt_outcome"]) for s in view["segments"][:3]] == [
        ("success", False, "failed"), ("failed", True, "failed"), ("success", False, "failed")]
    first = view["attempts"][0]
    assert (first["start"], first["end"], first["outcome"], first["evident_frame"]) == (0, 150, "failed",
                                                                                     reopen[0] - 1)
    assert any(e["source"] == "gripper_recovery" for e in view["events"])
    json.dumps(view)


# ------------------------------------------------------------------------------------------ scoring a pipeline view
def gold_missed_grasp():
    cam = CAM

    def gseg(idx, start, end, phase, attempt, outcome="success", failure_type=None, last=False):
        return {"segment_idx": idx, "start_frame": start, "end_frame": end, "phase_class": phase,
                "phase_text": phase, "target": "o1" if phase != "retract" else None, "destination": None,
                "attempt_idx": attempt, "outcome": outcome, "failure_type": failure_type,
                "end_boundary_quality": None if last else "sharp"}

    return {
        "episode_key": "F3/544", "episode_index": 544, "split": "dev", "pass": 1, "num_frames": 100,
        "cameras": [cam], "task_string": "put the pink brick in the box",
        "objects": [{"object_id": "o1", "name": "pink brick", "aliases": ["brick"], "category": "block",
                     "first_frame_point": {"camera": cam, "xy": [0.4, 0.6]}},
                    {"object_id": "o2", "name": "box", "aliases": [], "category": "container",
                     "first_frame_point": {"camera": cam, "xy": [0.7, 0.3]}}],
        "primary_target": "o1", "primary_destination": "o2",
        "segments": [gseg(0, 0, 29, "approach", 1), gseg(1, 30, 39, "grasp", 1, "failed", "missed_grasp"),
                     gseg(2, 40, 45, "retract", 1), gseg(3, 46, 59, "approach", 2), gseg(4, 60, 69, "grasp", 2),
                     gseg(5, 70, 84, "transport", 2), gseg(6, 85, 99, "release", 2, last=True)],
        "failed_attempts": [{"span": [0, 45], "failure_type": "missed_grasp", "object": "o1", "evident_frame": 38,
                             "evident_camera": cam, "recovery_start": 46}],
        "goal": {"objective_text": "the pink brick is inside the box", "requirements": []},
        "episode_outcome": "success", "hard_tags": ["failed_attempt"],
    }


def test_a_full_pipeline_view_scores_under_the_v11_failure_rule():
    plan = [seg(0, 29, "close_start", "approach", attempt_outcome="failed"),
            seg(30, 39, "open_start", "grasp", outcome="failed", attempt_outcome="failed", failure_type="missed_grasp"),
            seg(40, 45, "other", "retract", target="none", attempt_outcome="failed"),
            seg(46, 59, "close_start", "approach", attempt_idx=2),
            seg(60, 69, "other", "grasp", attempt_idx=2),
            seg(70, 84, "open_start", "transport", destination="o2", attempt_idx=2),
            seg(85, 99, "other", "release", destination="o2", attempt_idx=2)]
    out = run_episode_v11(clip(100, 20.0, "F3/544"), camera=CAM, caller=PlanCaller(plan), crawl=False,
                          context={"arm": "E2-main", "episode_key": "F3/544"})
    view = out["view"]
    assert [(a["start"], a["end"], a["outcome"]) for a in view["attempts"]] == [(0, 45, "failed"), (46, 99, "success")]
    r = score_view(view, gold_missed_grasp())
    assert r["details"]["S4"]["convention"] == "v11" and r["details"]["S4"]["pred_span_source"] == "attempts"
    assert r["details"]["S4"]["pred_spans"] == [[0, 45]]
    c = r["counts"]["S4-F1"]
    assert (c["numerator"], c["denominator"], c["pending"]) == (2, 2, 0)
    assert r["details"]["goal"]["G2"]["convention"] == "v11"
