"""The full v1.1 pipeline (robolabel.vfirst.run_episode_v11) and its no-signal scene calls, with fakes only."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import robolabel.layers.check as check_layer
import robolabel.layers.goal as goal_layer
from robolabel.episode import Episode
from robolabel.events import candidate_map, get_source
from robolabel.layers.scene import (
    facts_keyframes_v11,
    facts_request_v11,
    inventory_frames_v11,
    inventory_request_v11,
    parse_facts,
    parse_inventory,
)
from robolabel.layers.signal import Calibration, run_l1
from robolabel.prompts.v7 import SCHEMAS as SCHEMAS_V7
from robolabel.prompts.v8 import prompt_sha256
from robolabel.providers.base import CallResult, ImagePart, TextPart, VideoPart
from robolabel.providers.mock import MockProvider
from robolabel.schema_v7 import episode_record_v11, read_v11, subtask_records_v11, validate_v11_row, write_v11

HAS_V11_GOAL_AND_CHECKS = all(hasattr(goal_layer, f) for f in ("goal_request_v11", "postprocess_goal_v11")) and \
    hasattr(check_layer, "run_checks_v11")
needs_goal_and_checks = pytest.mark.skipif(not HAS_V11_GOAL_AND_CHECKS,
                                           reason="layers.goal / layers.check v1.1 functions not present yet")

CAM = "video"
N, FPS = 120, 30.0
CTX = {"arm": "E2-main", "episode_key": "C/synthetic", "bucket": "e2_luna", "model_key": "luna"}
V11_VIEW_FIELDS = {"has_end_state", "goal_command", "event_sources", "coarse_mode", "coarse_fps", "crawl_enabled",
                   "crawl_model", "crawl_log", "crawl_calls"}
V11_SEGMENT_FIELDS = {"end_event", "boundary_source", "coarse_end_frame", "crawl_calls", "attempt_outcome"}


# ------------------------------------------------------------------------------------------ fixtures
def clip_episode(n: int = N, fps: float = FPS, task: str | None = "put the red block in the blue box",
                 key: str = "C/synthetic") -> Episode:
    """A block that moves right one pixel per frame, except in a pause (frames 40 to 59), one camera."""
    frames = np.zeros((n, 48, 64, 3), dtype=np.uint8)
    x = 0
    for i in range(n):
        if not 40 <= i < 60:
            x += 1
        frames[i, 10:30, x % 50:x % 50 + 12] = (200, 30, 30)
        frames[i, 34:46, 40:60] = (30, 30, 200)

    def get(i):
        return frames[int(i)]

    return Episode(episode_id=key, num_frames=n, fps=fps, task=task, get_frame=get, camera_key=CAM)


def gripper_l1(n: int = N, fps: float = FPS) -> dict:
    """An L1 record with one close (onset about frame 40) and one open (about frame 85)."""
    cmd = np.array([20.0] * 40 + list(np.linspace(20, 1, 8)) + [1.0] * 37 + list(np.linspace(1, 20, 8))
                   + [20.0] * (n - 93))
    meas = np.array([20.0] * 40 + list(np.linspace(20, 8, 8)) + [8.0] * 37 + list(np.linspace(8, 20, 8))
                    + [20.0] * (n - 93))
    state, action = np.zeros((n, 6)), np.zeros((n, 6))
    state[:, 5], action[:, 5] = meas, cmd
    state[:, 0] = np.linspace(0, 90, n)
    cal = Calibration(layout="so101", fps=fps, cmd_open=20.0, cmd_closed=1.0, meas_open=20.0, meas_closed=1.0,
                      pause_speed=5.0, withdraw_threshold=5.0)
    return run_l1(state, action, cal, episode_key="C/synthetic", family="C")


def seg(start, end, end_event, phase_class, text, target="o1", destination="none", candidate_id="none",
        attempt_idx=1, outcome="success", attempt_outcome="success", failure_type="none"):
    return {"start_frame": start, "end_frame": end, "phase_class": phase_class, "phase_text": text,
            "end_event": end_event, "target": target, "destination": destination, "attempt_idx": attempt_idx,
            "outcome": outcome, "attempt_outcome": attempt_outcome, "failure_type": failure_type,
            "candidate_id": candidate_id}


PLAN = [seg(0, 39, "close_start", "approach", "reach for the red block"),
        seg(40, 54, "other", "grasp", "close on the red block"),
        seg(55, 84, "open_start", "transport", "carry the red block to the blue box", destination="o2"),
        seg(85, 99, "contact_end", "release", "let go of the red block", destination="o2"),
        seg(100, 119, "other", "retract", "move away", target="none")]

OBJECTS = [{"object_id": "o1", "name": "red block", "aliases": ["block"], "category": "block",
            "views": [{"camera": CAM, "visible": True, "x": 0.2, "y": 0.4, "x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.6}]},
           {"object_id": "o2", "name": "blue box", "aliases": [], "category": "container",
            "views": [{"camera": CAM, "visible": True, "x": 0.7, "y": 0.8, "x0": 0.6, "y0": 0.7, "x1": 0.9, "y1": 0.95}]}]


def goal_answer(has_end_state=True):
    def item(kind, obj, pred, ref, value, status="required", achieved="true"):
        return {"req_id": "r", "kind": kind, "object": obj, "predicate": pred, "ref_object": ref, "value": value,
                "status": status, "unsure_kind": "none", "basis": "task_string", "achieved": achieved,
                "deciding_frame": N - 1, "deciding_camera": CAM,
                "visibility": [{"camera": CAM, "class": "visible"}], "reason": "seen at the end"}

    reqs = [item("robot_end_state", "none", "holding", "none", "false"),
            item("robot_end_state", "none", "gripper_open", "none", "true"),
            item("robot_end_state", "none", "withdrawn", "none", "true")]
    if has_end_state:
        reqs.insert(0, item("object_end_state", "o1", "inside", "o2", "true"))
    return {"objective_text": "the red block is inside the blue box" if has_end_state else "a person waves",
            "has_end_state": has_end_state, "primary_target": "o1" if has_end_state else "none",
            "primary_destination": "o2" if has_end_state else "none", "requirements": reqs}


class StepCaller:
    """A fake caller answering by step: fixed inventory, coarse plan, crawl answers, facts and goal."""

    name = "fake"
    model = "fake-model"

    def __init__(self, *, objects=OBJECTS, plan=PLAN, crawl=None, goal=None, statuses=None, usd=0.001):
        self.objects, self.plan, self.goal = objects, plan, goal if goal is not None else goal_answer()
        self.crawl = crawl or {"stage1": 4, "retry": -1, "stage2": 2}
        self.statuses = dict(statuses or {})
        self.usd = usd
        self.requests = []

    def call(self, req):
        self.requests.append(req)
        st = self.statuses.get(req.step)
        if st:
            return CallResult(False, None, "", st, error=f"fake {st}", usd=0.0, wall_s=0.1)
        data = getattr(self, "_" + req.step)(req)
        return CallResult(True, copy.deepcopy(data), json.dumps(data), "ok", usd=self.usd, wall_s=0.5)

    def _scene_inventory(self, req):
        return {"objects": self.objects}

    def _coarse(self, req):
        if req.schema_name == "coarse_video_v8":
            return {"segments": [{**{k: v for k, v in s.items() if k not in ("start_frame", "end_frame")},
                                  "start_s": s["start_frame"] / FPS, "end_s": (s["end_frame"] + 1) / FPS}
                                 for s in self.plan]}
        return {"segments": self.plan}

    def _crawl(self, req):
        return {"answer": self.crawl[req.context["crawl_stage"]]}

    def _scene_facts(self, req):
        out = []
        for f in req.context["frame_indices"]:
            out.append({"frame": f, "camera": CAM, "visible": ["o1", "o2"], "partial": [],
                        "in_gripper": "o1" if 45 <= f <= 84 else "none",
                        "relations": [{"subject": "o1", "relation": "inside", "object": "o2",
                                       "value": "true" if f >= 85 else "false"}],
                        "boxes": [{"object_id": "o1", "x0": 0.6, "y0": 0.7, "x1": 0.7, "y1": 0.8}] if f == N - 1 else []})
        return {"facts": out}

    def _goal(self, req):
        return self.goal

    def steps(self):
        return [r.step for r in self.requests]


def run(ep=None, caller=None, **kw):
    from robolabel.vfirst import run_episode_v11

    kw.setdefault("context", CTX)
    return run_episode_v11(ep or clip_episode(), camera=CAM, caller=caller or StepCaller(), **kw)


def texts(req):
    return "\n".join(p.text for p in req.parts if isinstance(p, TextPart))


def onsets(segments):
    return [s["start_frame"] for s in segments[1:]]


# ------------------------------------------------------------------------------------------ scene calls (no signal)
def test_inventory_request_v11_frames_prompt_and_schema():
    ep = clip_episode()
    from robolabel.vfirst import with_camera

    ep = with_camera(ep, CAM)
    req, manifest = inventory_request_v11(ep, camera=CAM, context=CTX, reasoning={"effort": "low"})
    frames = inventory_frames_v11(N)
    assert frames[0] == 0 and frames[-1] == N - 1 and len(frames) == 8 and frames == sorted(set(frames))
    assert [m["frame"] for m in manifest] == frames and {m["camera"] for m in manifest} == {CAM}
    assert req.step == "scene_inventory" and req.schema == SCHEMAS_V7["scene_inventory"]
    assert req.context["frame_indices"] == frames and req.context["cameras"] == [CAM]
    assert req.context["bucket"] == "e2_luna" and req.reasoning == {"effort": "low"}
    assert sum(isinstance(p, ImagePart) for p in req.parts) == 8
    text = req.parts[0].text
    assert '"put the red block in the blue box"' in text and "8 frames of one video clip from camera video" in text
    assert "{" not in text and "robot measurement" not in text.lower() and "measured" not in text
    assert "a robot, a humanoid or a person" in req.system
    ep2 = clip_episode(task=None, n=5)
    req2, _ = inventory_request_v11(with_camera(ep2, CAM), camera=CAM, context=CTX, reasoning=None)
    assert "gives no task description" in req2.parts[0].text and req2.context["frame_indices"] == [0, 1, 2, 3, 4]
    assert inventory_frames_v11(1) == [0]


def test_facts_keyframes_v11_rule():
    segs = [dict(s) for s in PLAN]
    assert facts_keyframes_v11(segs, N) == [0, 40, 55, 85, 100, N - 1]
    # more boundaries than fit: the typed ones first, then others, evenly spaced within each group
    many = []
    for i in range(12):
        many.append({"start_frame": i * 10, "end_frame": i * 10 + 9,
                     "end_event": "close_start" if i % 3 == 0 else "other"})
    kf = facts_keyframes_v11(many, 120)
    assert len(kf) == 8 and kf[0] == 0 and kf[-1] == 119
    typed = [s["start_frame"] for b, s in zip(many, many[1:], strict=False) if b["end_event"] == "close_start"]
    assert set(typed) <= set(kf)  # 10, 40, 70, 100: every typed boundary fits in the 6 free places
    assert facts_keyframes_v11(many, 120) == kf  # deterministic
    assert facts_keyframes_v11([{"start_frame": 0, "end_frame": 0, "end_event": "other"}], 1) == [0]
    assert facts_keyframes_v11(many, 120, max_frames=3) == [0, 40, 119]  # one place: the middle typed boundary
    assert facts_keyframes_v11(many, 120, max_frames=4) == [0, 10, 100, 119]  # two: the first and last typed


def test_facts_request_v11_and_parsers_reuse_v7():
    from robolabel.vfirst import with_camera

    ep = with_camera(clip_episode(), CAM)
    repairs: list[str] = []
    objects = parse_inventory({"objects": OBJECTS}, ep, repairs)
    assert [o["object_id"] for o in objects] == ["o1", "o2"] and objects[0]["views"][0]["camera"] == CAM
    req, manifest = facts_request_v11(ep, camera=CAM, keyframes=[119, 0, 40, 40, 500], objects=objects,
                                      context=CTX, reasoning=None)
    assert req.context["frame_indices"] == [0, 40, 119] and req.schema == SCHEMAS_V7["scene_facts"]
    text = req.parts[0].text
    assert "o1: red block (block)" in text and "Task objects: o1, o2." in text and "frame 119" in text
    facts = parse_facts(StepCaller()._scene_facts(req), manifest, ep, objects, repairs)
    assert [f["frame"] for f in facts] == [0, 40, 119] and facts[-1]["boxes"] and not repairs
    req2, _ = facts_request_v11(with_camera(clip_episode(task=None), CAM), camera=CAM, keyframes=[0], objects=objects,
                                context=CTX, reasoning=None)
    assert "the objects that are handled in the clip" in req2.parts[0].text


def test_scene_prompts_are_v8_files_with_their_hashes():
    from robolabel.vfirst import prompt_hashes_v11

    hashes = prompt_hashes_v11()
    for name in ("scene_inventory", "scene_facts", "coarse", "crawl", "system"):
        assert hashes[name] == prompt_sha256(name) and len(hashes[name]) == 64


def test_with_camera_copies_and_never_changes_the_episode():
    from robolabel.vfirst import with_camera

    ep = clip_episode()
    out = with_camera(ep, CAM)
    assert out is not ep and ep.extra == {}
    assert out.extra["camera_order"] == [CAM] and out.extra["external_cameras"] == [CAM]
    assert out.extra["wrist_cameras"] == [] and out.extra["camera_sizes"] == {CAM: [64, 48]}
    assert with_camera(out, CAM) is out
    with pytest.raises(KeyError):
        with_camera(out, "observation.images.wrist")


def test_attempt_records_v11_span_outcome_and_evident_frame():
    from robolabel.vfirst import attempt_records_v11

    segs = [seg(0, 9, "close_start", "approach", "a", attempt_outcome="failed"),
            seg(10, 19, "open_start", "grasp", "g", outcome="failed", attempt_outcome="failed",
                failure_type="missed_grasp"),
            seg(20, 29, "other", "retract", "r", attempt_outcome="failed"),
            seg(30, 49, "other", "approach", "a2", attempt_idx=2)]
    recs = attempt_records_v11(segs)
    assert recs == [{"attempt_idx": 1, "start": 0, "end": 29, "outcome": "failed", "failure_type": "missed_grasp",
                     "evident_frame": 19, "source": "vlm"},
                    {"attempt_idx": 2, "start": 30, "end": 49, "outcome": "success", "failure_type": "none",
                     "evident_frame": None, "source": "vlm"}]
    old = [{k: v for k, v in s.items() if k != "attempt_outcome"} for s in segs]  # a v7 reader: derived
    assert attempt_records_v11(old)[0]["outcome"] == "failed"


def test_pipeline_code_v11_is_a_stable_hash():
    from robolabel.vfirst import PIPELINE_CODE_V11, PIPELINE_VERSION_V11, pipeline_code_v11

    assert len(PIPELINE_CODE_V11) == 12 and int(PIPELINE_CODE_V11, 16) >= 0
    assert pipeline_code_v11() == PIPELINE_CODE_V11 and PIPELINE_VERSION_V11.startswith("v1.1 v8")


def test_signal_rules_are_enforced_before_any_call():
    caller = StepCaller()
    with pytest.raises(ValueError, match="needs the episode's L1"):
        run(caller=caller, event_source="gripper")
    with pytest.raises(ValueError, match="only with the gripper source"):
        run(caller=caller, event_source="none", l1=gripper_l1())
    with pytest.raises(ValueError, match="only with the gripper source"):
        run(caller=caller, event_source="motion", l1=gripper_l1())
    # an L1 record for another frame count would snap boundaries to the wrong frames
    with pytest.raises(ValueError, match="would not line up"):
        run(caller=caller, event_source="gripper", l1={**gripper_l1(), "num_frames": N + 1})
    with pytest.raises(ValueError, match="would not line up"):
        run(clip_episode(n=N + 10), caller=caller, event_source="gripper", l1=gripper_l1())
    no_count = {k: v for k, v in gripper_l1().items() if k != "num_frames"}
    with pytest.raises(ValueError, match="would not line up"):
        run(caller=caller, event_source="gripper", l1=no_count)
    assert caller.requests == []


# ------------------------------------------------------------------------------------------ the pipeline
@needs_goal_and_checks
def test_full_pipeline_no_signal_order_and_outputs():
    caller = StepCaller()
    out = run(caller=caller, reasoning={"effort": "low"}, image_tokens_per_image=133.0, crawl_image_tokens=140.0)
    steps = caller.steps()
    assert steps[:2] == ["scene_inventory", "coarse"] and steps[-2:] == ["scene_facts", "goal"]
    assert set(steps[2:-2]) == {"crawl"} and len(steps) - 4 == out["view"]["crawl_calls"] > 0
    req = {r.step: r for r in caller.requests}
    # the coarse pass sees the inventory IDs and no candidate list
    assert "o1: red block (block)" in texts(req["coarse"]) and "Candidate events" not in texts(req["coarse"])
    assert req["scene_inventory"].image_tokens_per_image == 133.0 and req["coarse"].image_tokens_per_image == 133.0
    assert all(r.image_tokens_per_image == 140.0 for r in caller.requests if r.step == "crawl")
    # the crawl refined the three typed boundaries (40 close, 85 open, 100 contact end)
    log = out["crawl_log"]
    assert [e["event_type"] for e in log] == ["close_start", "open_start", "contact_end"]
    assert onsets(out["segments_coarse"]) == [40, 55, 85, 100]
    refined = onsets(out["segments"])
    assert refined == [e["onset"] for e in log][:1] + [55] + [e["onset"] for e in log][1:]
    for e in log:  # picks lie inside a window the model saw
        seen = set(e["stage1_frames"] or []) | set(e["stage2_frames"] or []) | set(e["retry_frames"] or [])
        assert e["pick"] in seen
    # facts on keyframes from the refined boundaries: frame 0, each boundary, the last frame
    assert req["scene_facts"].context["frame_indices"] == [0, *refined, N - 1] == out["view"]["keyframes"]
    assert out["view"]["keyframe_source"] == "boundaries"
    # no robot measurement reaches any prompt without a signal
    for r in caller.requests:
        assert "measured by the robot" not in texts(r)
    view = out["view"]
    assert V11_VIEW_FIELDS <= set(view)
    assert view["event_sources"] == ["none"] and view["coarse_mode"] == "frames" and view["crawl_enabled"] is True
    assert view["crawl_model"] == "fake-model" and view["coarse_fps"] == pytest.approx(2.0, abs=0.1)
    assert view["has_end_state"] is True and isinstance(view["goal_command"], str) and view["goal_command"]
    assert view["step_status"] == {"scene_inventory": "ok", "coarse": "ok", "crawl": "ok", "scene_facts": "ok",
                                   "goal": "ok"}
    assert view["failed_calls"] == [] and view["valid"] is True and view["no_output"] is False
    for vs in view["segments"]:
        assert V11_SEGMENT_FIELDS <= set(vs)
    assert view["segments"][0]["boundary_source"] == "crawl" and view["segments"][0]["coarse_end_frame"] == 39
    assert [a["outcome"] for a in view["attempts"]] == ["success"] and view["attempts_signal"] is None
    assert view["cost_usd"] == pytest.approx(0.001 * len(steps)) and view["calls"] == len(steps)
    assert out["objects"][0]["name"] == "red block" and out["facts"] and out["goal"]["has_end_state"] is True
    json.dumps(view)
    # rows: v7 rows plus the v1.1 fields
    rows = out["rows"]
    meta = [r for r in rows if r["record_type"] == "episode_metadata"]
    assert len(meta) == 1 and meta[0]["has_end_state"] is True and meta[0]["event_sources"] == "none"
    assert meta[0]["coarse_mode"] == "frames" and meta[0]["crawl_enabled"] is True
    assert meta[0]["crawl_model"] == "fake-model" and meta[0]["goal_command"] == view["goal_command"]
    assert json.loads(meta[0]["layer_models_json"])["L1"] == "none"
    subs = [r for r in rows if r["record_type"] == "subtask"]
    assert [r["end_event"] for r in subs] == [s["end_event"] for s in out["segments"]]
    attempts = [r for r in rows if r["record_type"] == "attempt"]
    assert len(attempts) == 1 and attempts[0]["attempt_source"] == "vlm"
    assert all(not validate_v11_row(r) for r in rows) and all(r["strategy"] == "v1.1" for r in rows)


@needs_goal_and_checks
def test_rows_round_trip_through_the_v11_writer(tmp_path):
    out = run()
    write_v11(out["rows"], tmp_path)
    frame = read_v11(tmp_path)
    meta = episode_record_v11(frame, "C/synthetic", "E2-main")
    assert meta["has_end_state"] is True and meta["event_sources"] == ["none"] and meta["coarse_mode"] == "frames"
    subs = subtask_records_v11(frame, "C/synthetic", "E2-main")
    assert [s["coarse_end_frame"] for s in subs] == [s["coarse_end_frame"] for s in out["segments"]]
    assert [s["boundary_source"] for s in subs] == [s["boundary_source"] for s in out["segments"]]


@needs_goal_and_checks
def test_same_input_gives_the_same_outputs():
    a, b = run(), run()
    for key in ("segments_coarse", "segments", "crawl_log", "goal", "objects", "facts", "repairs", "checks"):
        assert json.dumps(a[key], sort_keys=True) == json.dumps(b[key], sort_keys=True)

    def strip(view):
        return {k: v for k, v in view.items() if k not in ("wall_s",)}

    assert json.dumps(strip(a["view"]), sort_keys=True) == json.dumps(strip(b["view"]), sort_keys=True)
    assert json.dumps(a["rows"], sort_keys=True, default=str) == json.dumps(b["rows"], sort_keys=True, default=str)


@needs_goal_and_checks
def test_gripper_source_snaps_signal_boundaries_and_measures_the_goal():
    ep, l1 = clip_episode(), gripper_l1()
    events = get_source("gripper").events(ep, camera=CAM, l1=l1)
    cmap = candidate_map(events)
    close_id = next(k for k, e in cmap.items() if e["type"] == "close_start")
    open_id = next(k for k, e in cmap.items() if e["type"] == "open_start")
    plan = copy.deepcopy(PLAN)
    plan[0]["candidate_id"], plan[2]["candidate_id"] = close_id, open_id
    caller = StepCaller(plan=plan)
    out = run(ep, caller, event_source="gripper", l1=l1)
    req = {r.step: r for r in caller.requests}
    assert "Candidate events" in texts(req["coarse"]) and f"{close_id}: frame" in texts(req["coarse"])
    segs = out["segments"]
    assert segs[1]["start_frame"] == cmap[close_id]["frame"] and segs[0]["boundary_source"] == "signal"
    assert segs[3]["start_frame"] == cmap[open_id]["frame"] and segs[2]["boundary_source"] == "signal"
    flags = {e["event_type"]: e["flags"] for e in out["crawl_log"]}
    assert flags["close_start"] == ["skipped_signal"] and flags["open_start"] == ["skipped_signal"]
    assert [r.context["event_type"] for r in caller.requests if r.step == "crawl"] == \
        ["contact_end"] * sum(1 for r in caller.requests if r.step == "crawl")
    assert "measured by the robot" in texts(req["goal"])  # the gripper_on extra is shown the signal by design
    view = out["view"]
    assert view["event_sources"] == ["gripper"] and view["keyframe_source"] == "signal"
    from robolabel.layers.signal import keyframe_plan_every_event

    cal = Calibration(**l1["calibration"])
    assert view["keyframes"] == keyframe_plan_every_event(l1, N, cal) == req["scene_facts"].context["frame_indices"]
    assert json.loads(next(r for r in out["rows"] if r["record_type"] == "episode_metadata")["layer_models_json"])[
        "L1"] == "signal"
    # one attempt source in the view and the rows: the segments' attempts (vlm), and L1's kept beside them
    attempt_rows = [r for r in out["rows"] if r["record_type"] == "attempt"]
    assert [r["attempt_source"] for r in attempt_rows] == ["signal"] * len(l1["attempts"]) + \
        ["vlm"] * len(view["attempts"])
    vlm_rows = [r for r in attempt_rows if r["attempt_source"] == "vlm"]
    assert [(r["start_frame"], r["end_frame"], r["outcome"]) for r in vlm_rows] == \
        [(a["start"], a["end"], a["outcome"]) for a in view["attempts"]]
    sig_rows = [r for r in attempt_rows if r["attempt_source"] == "signal"]
    assert [(r["start_frame"], r["end_frame"], r["outcome"], r["evident_frame"]) for r in sig_rows] == \
        [(a["start"], a["end"], a["outcome"], a["evident_frame"]) for a in view["attempts_signal"]]
    assert all(a["source"] == "vlm" for a in view["attempts"])
    assert all(a["source"] == "signal" for a in view["attempts_signal"]) and view["attempts_signal"]
    # rules 1, 2, 6, 7 and 8 are not na for lack of a signal
    by_rule = {c["rule_id"]: c for c in out["checks"]["checks"]}
    assert by_rule[1]["verdict"] in ("pass", "fail") and by_rule[2]["verdict"] in ("pass", "fail")


@needs_goal_and_checks
def test_motion_source_sends_pause_candidates_and_stays_video_only():
    caller = StepCaller()
    out = run(caller=caller, event_source="motion")
    coarse_req = next(r for r in caller.requests if r.step == "coarse")
    assert "Candidate events" in texts(coarse_req) and "pause_start (motion)" in texts(coarse_req)
    assert out["events"] and {e["source"] for e in out["events"]} == {"motion"}
    assert out["view"]["event_sources"] == ["motion"] and out["view"]["keyframe_source"] == "boundaries"
    by_rule = {c["rule_id"]: c for c in out["checks"]["checks"]}
    assert all(by_rule[r]["verdict"] == "na" for r in (1, 2, 6, 7, 8))
    assert "measured by the robot" not in texts(next(r for r in caller.requests if r.step == "goal"))


@needs_goal_and_checks
def test_no_signal_rules_are_na_and_no_l1_is_read():
    out = run()
    by_rule = {c["rule_id"]: c for c in out["checks"]["checks"]}
    for rule in (1, 2, 6, 7, 8):
        assert by_rule[rule]["verdict"] == "na"
    assert all(r["basis"] != "signal" for r in out["goal"]["requirements"])


@needs_goal_and_checks
def test_a_failed_coarse_call_gives_the_missing_output_segment_and_routes():
    from robolabel.layers.coarse import missing_output_segments

    for status in ("invalid", "failed", "refused", "unavailable"):
        caller = StepCaller(statuses={"coarse": status})
        out = run(caller=caller)
        assert caller.steps() == ["scene_inventory", "coarse", "scene_facts", "goal"]  # no crawl, the rest runs
        assert out["segments"] == out["segments_coarse"] == missing_output_segments(N)
        assert out["view"]["step_status"]["crawl"] == "not run" and out["crawl_log"] == []
        assert out["failed_calls"] == [{"step": "coarse", "status": status, "error": f"fake {status}"}]
        assert out["view"]["no_output"] is True and out["view"]["valid"] is False
        assert out["checks"]["risk"] == 1.0 and out["checks"]["routed"] is True  # D1b
        assert any(r.startswith(f"coarse {status}: missing-output segment") for r in out["repairs"])


@needs_goal_and_checks
def test_a_failed_goal_or_facts_call_routes_with_the_reason():
    out = run(caller=StepCaller(statuses={"goal": "invalid"}))
    assert out["goal"] is None and out["view"]["goal"] is None and out["view"]["has_end_state"] is None
    assert out["checks"]["risk"] == 1.0 and out["checks"]["routed"] is True
    assert any("goal" in r for r in out["checks"]["route_reasons"])
    out = run(caller=StepCaller(statuses={"scene_facts": "failed"}))
    assert out["facts"] == [] and out["goal"] is not None and out["checks"]["routed"] is True


@needs_goal_and_checks
def test_a_stopped_call_ends_the_model_calls():
    caller = StepCaller(statuses={"scene_inventory": "stopped"})
    out = run(caller=caller)
    assert caller.steps() == ["scene_inventory"] and out["stopped"] is True
    assert out["view"]["step_status"] == {"scene_inventory": "stopped", "coarse": "not run", "crawl": "not run",
                                          "scene_facts": "not run", "goal": "not run"}
    assert [f["step"] for f in out["failed_calls"]] == ["scene_inventory", "coarse", "crawl", "scene_facts", "goal"]
    assert [f["status"] for f in out["failed_calls"]] == ["stopped"] + ["not run"] * 4
    assert out["checks"]["risk"] == 1.0 and out["checks"]["routed"] is True


@needs_goal_and_checks
def test_empty_inventory_uses_plain_words_and_skips_facts():
    words = [dict(s, target="the red block", destination="none") for s in PLAN]
    caller = StepCaller(objects=[], plan=words, goal=goal_answer(has_end_state=False))
    out = run(caller=caller)
    assert "facts" not in " ".join(caller.steps()) and out["view"]["step_status"]["scene_facts"] == "skipped"
    coarse_req = next(r for r in caller.requests if r.step == "coarse")
    assert "Objects in the scene" not in texts(coarse_req)
    assert out["segments"][0]["target"] == "the red block"
    assert out["view"]["has_end_state"] is False
    assert not [r for r in out["goal"]["requirements"] if r["kind"] == "object_end_state"]


@needs_goal_and_checks
def test_crawl_off_and_a_separate_crawl_caller():
    caller = StepCaller()
    out = run(caller=caller, crawl=False)
    assert "crawl" not in caller.steps() and out["view"]["crawl_enabled"] is False
    assert out["view"]["crawl_model"] is None and out["view"]["step_status"]["crawl"] == "off"
    assert out["segments"] == out["segments_coarse"]
    other = StepCaller()
    other.model = "crawl-model"
    caller = StepCaller()
    out = run(caller=caller, crawl_caller=other, crawl_context={**CTX, "bucket": "e2_crawl"})
    assert "crawl" not in caller.steps() and set(other.steps()) == {"crawl"}
    assert out["view"]["crawl_model"] == "crawl-model"
    assert {r.context["bucket"] for r in other.requests} == {"e2_crawl"}


@needs_goal_and_checks
def test_native_video_coarse_pass():
    video = VideoPart(data=b"\x00\x00\x00\x18ftypmp42", seconds=N / FPS, label="clip.mp4")
    caller = StepCaller()
    out = run(caller=caller, coarse_mode="video", video=video)
    coarse_req = next(r for r in caller.requests if r.step == "coarse")
    assert any(isinstance(p, VideoPart) for p in coarse_req.parts) and coarse_req.schema_name == "coarse_video_v8"
    assert onsets(out["segments_coarse"]) == [40, 55, 85, 100]
    assert out["view"]["coarse_mode"] == "video" and out["view"]["coarse_fps"] is None
    with pytest.raises(ValueError):
        run(caller=StepCaller(), coarse_mode="video")  # the package never encodes video


@needs_goal_and_checks
@pytest.mark.parametrize("source", ["none", "motion", "gripper"])
@pytest.mark.parametrize("mode", ["frames", "video"])
def test_mock_provider_end_to_end(source, mode):
    ep = clip_episode()
    video = VideoPart(data=b"mock", seconds=N / FPS) if mode == "video" else None
    out = run(ep, MockProvider(), event_source=source, l1=gripper_l1() if source == "gripper" else None,
              coarse_mode=mode, video=video)
    view = out["view"]
    assert V11_VIEW_FIELDS <= set(view) and view["event_sources"] == [source] and view["coarse_mode"] == mode
    assert out["segments"][0]["start_frame"] == 0 and out["segments"][-1]["end_frame"] == N - 1
    assert all(not validate_v11_row(r) for r in out["rows"])
    json.dumps(view)
