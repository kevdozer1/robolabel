"""The E1 timing pipeline (robolabel.vfirst.run_timing): coarse pass plus crawl, with fake callers only."""

from __future__ import annotations

import copy
import json

import numpy as np

from robolabel.episode import Episode
from robolabel.events import candidate_map, get_source
from robolabel.layers.coarse import missing_output_segments
from robolabel.layers.signal import Calibration, run_l1
from robolabel.prompts.v8 import VERSION, prompt_sha256
from robolabel.providers.base import CallResult, TextPart, VideoPart
from robolabel.vfirst import run_timing

CAM = "observation.images.up"
OUT_KEYS = {"segments_coarse", "segments", "crawl_log", "calls", "events", "coarse_info", "coarse_status",
            "repairs", "cost_usd", "wall_s", "event_sources", "crawl_enabled", "prompt_version", "prompt_hashes"}
COARSE_CTX = {"arm": "L-A", "episode_key": "F1/0", "bucket": "e1_luna", "model_key": "luna"}
CRAWL_CTX = {"arm": "L-B", "episode_key": "F1/0", "bucket": "e1_luna_g", "model_key": "luna"}


# ------------------------------------------------------------------------------------------ fixtures
def moving_episode(n: int = 300, fps: float = 30.0) -> Episode:
    """A white block that moves one pixel per frame, except in two pauses (frames 80-129 and 200-239)."""
    x, xs = 0, []
    for i in range(n):
        if not (80 <= i < 130 or 200 <= i < 240):
            x += 1
        xs.append(x)
    frames = np.zeros((n, 48, 64, 3), dtype=np.uint8)
    for i, x in enumerate(xs):
        frames[i, 10:30, x % 50:x % 50 + 12] = 255

    def get(i):
        return frames[int(i)]

    return Episode(episode_id="F1/0", num_frames=n, fps=fps, task="put the block in the box", get_frame=get,
                   camera_key=CAM, extra={"family": "F1", "cameras": {CAM: get}, "camera_order": [CAM],
                                          "external_cameras": [CAM], "wrist_cameras": []})


def gripper_episode():
    """120 frames at 30 fps with one close (about frame 30) and one open (about frame 80), plus its L1 record."""
    n = 120
    ep = moving_episode(n=n)
    cmd = np.array([20.0] * 30 + list(np.linspace(20, 1, 10)) + [1.0] * 40 + list(np.linspace(1, 20, 10)) + [20.0] * 30)
    meas = np.array([20.0] * 30 + list(np.linspace(20, 8, 10)) + [8.0] * 40 + list(np.linspace(8, 20, 10)) + [20.0] * 30)
    state, action = np.zeros((n, 6)), np.zeros((n, 6))
    state[:, 5], action[:, 5] = meas, cmd
    state[:, 0] = np.linspace(0, 90, n)
    ep.extra.update(state=state, action=action)
    cal = Calibration(layout="so101", fps=30.0, cmd_open=20.0, cmd_closed=1.0, meas_open=20.0, meas_closed=1.0,
                      pause_speed=5.0, withdraw_threshold=5.0)
    return ep, run_l1(state, action, cal, episode_key="F1/0", family="F1")


def item(start, end, end_event="other", phase_class="other", text=None, target="the block", destination="none",
         candidate_id="none", attempt_idx=1, outcome="success", attempt_outcome="success", failure_type="none"):
    return {"start_frame": start, "end_frame": end, "phase_class": phase_class,
            "phase_text": text or f"{phase_class} from {start}", "end_event": end_event, "target": target,
            "destination": destination, "attempt_idx": attempt_idx, "outcome": outcome,
            "attempt_outcome": attempt_outcome, "failure_type": failure_type, "candidate_id": candidate_id}


PLAN = {"segments": [item(0, 99, "contact_start", "approach", "move to the block"),
                     item(100, 159, "close_start", "grasp", "close on the block"),
                     item(160, 219, "open_start", "transport", "carry the block to the box", destination="the box"),
                     item(220, 299, "other", "retract", "move away", target="none")]}


class Fixed:
    """A fake coarse caller: one fixed answer (or a status), recorded requests."""

    name = "fake"
    model = "fake-coarse"

    def __init__(self, data=None, status="ok", usd=0.01):
        self.data, self.status, self.usd = data, status, usd
        self.requests = []

    def call(self, req):
        self.requests.append(req)
        if self.status != "ok":
            return CallResult(False, None, "", self.status, error=f"fake {self.status}", usd=0.0, wall_s=1.0)
        return CallResult(True, copy.deepcopy(self.data), json.dumps(self.data), "ok", usd=self.usd, wall_s=2.0)


class Scripted:
    """A fake crawl caller answering from a list, one item per call."""

    name = "scripted"
    model = "fake-crawl"

    def __init__(self, answers, usd=0.001):
        self.answers, self.usd = list(answers), usd
        self.requests = []

    def call(self, req):
        self.requests.append(req)
        a = self.answers.pop(0)
        return CallResult(True, {"answer": a}, json.dumps({"answer": a}), "ok", usd=self.usd, wall_s=0.5)


def timing(ep, coarse, crawl=None, **kw):
    return run_timing(ep, camera=CAM, coarse_caller=coarse, crawl_caller=crawl, coarse_context=COARSE_CTX,
                      crawl_context=CRAWL_CTX, **kw)


def onsets(segments):
    return [s["start_frame"] for s in segments[1:]]


# ------------------------------------------------------------------------------------------ tests
def test_coarse_only_arm():
    ep = moving_episode()
    coarse = Fixed(PLAN)
    out = timing(ep, coarse, reasoning={"effort": "low"}, coarse_image_tokens=140.0, start_mode="json_object_with_schema_in_prompt")
    assert OUT_KEYS <= set(out)
    assert out["coarse_status"] == "ok" and out["crawl_enabled"] is False and out["crawl_log"] == []
    assert out["segments"] == out["segments_coarse"] and out["segments"] is not out["segments_coarse"]
    assert onsets(out["segments"]) == [100, 160, 220]
    assert [s["end_event"] for s in out["segments"]] == ["contact_start", "close_start", "open_start", "other"]
    assert all(s["boundary_source"] == "coarse" and s["crawl_calls"] == 0 for s in out["segments"])
    assert len(out["calls"]) == 1 and out["cost_usd"] == 0.01 and out["wall_s"] == 2.0
    assert out["events"] == [] and out["event_sources"] == ["none"]
    assert out["coarse_info"]["mode"] == "frames" and out["coarse_info"]["frame_indices"][0] == 0
    assert out["prompt_version"] == VERSION
    assert out["prompt_hashes"]["coarse"] == prompt_sha256("coarse")
    assert out["prompt_hashes"]["crawl"] == prompt_sha256("crawl")
    (req,) = coarse.requests
    assert req.step == "coarse" and req.context["bucket"] == "e1_luna" and req.reasoning == {"effort": "low"}
    assert req.image_tokens_per_image == 140.0 and req.start_mode == "json_object_with_schema_in_prompt"
    assert "Candidate events" not in req.parts[0].text  # no candidate list without an event source
    json.dumps({k: v for k, v in out.items() if k != "calls"})


def test_frames_plus_crawl():
    ep = moving_episode()
    # contact_start: pick then stage 2; close_start: cannot be judged; open_start: a 9 at frame 250, then a
    # retry pick after it (frame 256), then stage 2 from 250 (stage 1's last image) to 256
    crawl = Scripted([4, 5, -1, 9, 3, 5])
    out = timing(ep, Fixed(PLAN), crawl, crawl_image_tokens=130.0)
    assert crawl.answers == []
    assert out["crawl_enabled"] is True and out["crawl_ran"] is True
    log = out["crawl_log"]
    assert [e["event_type"] for e in log] == ["contact_start", "close_start", "open_start"]
    assert log[0]["onset"] == 92 and log[0]["object"] == "block" and log[1]["flags"] == ["crawl_none"]
    assert log[2]["stage1_frames"][-1] == 250
    assert log[2]["retry_frames"] == [239, 248, 256, 265, 273, 282, 290, 299]
    assert log[2]["stage2_frames"] == [250, 251, 252, 253, 254, 255, 256] and log[2]["onset"] == 254
    segs = out["segments"]
    assert onsets(segs) == [92, 160, 254]
    assert onsets(out["segments_coarse"]) == [100, 160, 220]  # the coarse proposal is kept as it was
    assert segs[0]["boundary_source"] == "crawl" and segs[0]["coarse_end_frame"] == 99 and segs[0]["end_frame"] == 91
    assert segs[1]["boundary_source"] == "coarse" and segs[1]["crawl_calls"] == 1
    assert segs[2]["boundary_source"] == "crawl" and segs[2]["crawl_calls"] == 3 and segs[2]["end_frame"] == 253
    assert len(out["calls"]) == 1 + 6
    assert out["cost_usd"] == round(0.01 + 6 * 0.001, 8) and out["wall_s"] == 2.0 + 6 * 0.5
    assert out["crawl_calls"] == 6
    for r in crawl.requests:
        assert r.step == "crawl" and r.context["bucket"] == "e1_luna_g" and r.image_tokens_per_image == 130.0
    assert "touches the block?" in crawl.requests[0].parts[0].text
    assert "have started to close?" in crawl.requests[2].parts[0].text
    # a retry pick before frame 250 (image 2, frame 248) contradicts the 9 of stage 1: the stage-1 edge stays
    # (the frame after 250) and stage 2 does not run
    crawl = Scripted([4, 5, -1, 9, 2])
    out = timing(ep, Fixed(PLAN), crawl)
    e = out["crawl_log"][2]
    assert e["flags"] == ["crawl_edge", "crawl_inconsistent"] and e["onset"] == 251 and e["pick"] == 250
    assert e["stage2_frames"] is None and onsets(out["segments"]) == [92, 160, 251] and len(out["calls"]) == 6


def test_missing_output_when_the_coarse_call_fails():
    ep = moving_episode()
    for status in ("invalid", "failed", "refused", "unavailable", "stopped"):
        crawl = Scripted([])
        out = timing(ep, Fixed(status=status), crawl)
        assert out["coarse_status"] == status and out["coarse_error"] == f"fake {status}"
        assert out["segments"] == missing_output_segments(300) == out["segments_coarse"]
        assert out["segments"][0]["end_event"] == "other" and len(out["segments"]) == 1
        assert crawl.requests == [] and out["crawl_log"] == [] and out["crawl_ran"] is False
        assert any(r.startswith(f"coarse {status}: missing-output segment") for r in out["repairs"])
        assert len(out["calls"]) == 1 and out["cost_usd"] == 0.0


def test_gripper_source_snaps_close_and_open_and_crawls_contact():
    ep, l1 = gripper_episode()
    events = get_source("gripper").events(ep, camera=CAM, l1=l1)
    cmap = candidate_map(events)
    close_id = next(k for k, e in cmap.items() if e["type"] == "close_start")
    open_id = next(k for k, e in cmap.items() if e["type"] == "open_start")
    close_f, open_f = cmap[close_id]["frame"], cmap[open_id]["frame"]
    plan = {"segments": [item(0, 35, "close_start", "approach", candidate_id=close_id),
                         item(36, 59, "contact_start", "grasp"),
                         item(60, 84, "open_start", "transport", candidate_id=open_id),
                         item(85, 119, "other", "retract", target="none")]}
    coarse = Fixed(plan)
    crawl = Scripted([-1])
    out = timing(ep, coarse, crawl, event_source="gripper", l1=l1)
    assert crawl.answers == [] and len(crawl.requests) == 1
    assert out["events"] == events and out["event_sources"] == ["gripper"]
    assert f"{close_id}: frame {close_f}" in coarse.requests[0].parts[0].text
    segs = out["segments"]
    assert onsets(segs) == [close_f, 60, open_f]
    assert segs[0]["boundary_source"] == "signal" and segs[2]["boundary_source"] == "signal"
    by = {e["event_type"]: e for e in out["crawl_log"]}
    assert by["close_start"]["flags"] == ["skipped_signal"] and by["open_start"]["flags"] == ["skipped_signal"]
    assert by["close_start"]["calls"] == 0 and by["open_start"]["calls"] == 0
    assert by["contact_start"]["calls"] == 1 and by["contact_start"]["flags"] == ["crawl_none"]


def test_gripper_source_crawls_a_gripper_boundary_tied_to_no_event():
    """SPEC 3.3 item 10 as read here: only a snapped boundary (tied to a gripper event) skips the crawl."""
    ep, l1 = gripper_episode()
    events = get_source("gripper").events(ep, camera=CAM, l1=l1)
    cmap = candidate_map(events)
    close_id = next(k for k, e in cmap.items() if e["type"] == "close_start")
    close_f = cmap[close_id]["frame"]
    plan = {"segments": [item(0, 35, "close_start", "approach", candidate_id=close_id),
                         item(36, 84, "open_start", "transport"),
                         item(85, 119, "other", "retract", target="none")]}
    crawl = Scripted([-1])
    out = timing(ep, Fixed(plan), crawl, event_source="gripper", l1=l1)
    assert len(crawl.requests) == 1 and crawl.requests[0].context["event_type"] == "open_start"
    by = {e["event_type"]: e for e in out["crawl_log"]}
    assert by["close_start"]["flags"] == ["skipped_signal"] and by["close_start"]["onset"] == close_f
    assert by["open_start"]["calls"] == 1 and by["open_start"]["flags"] == ["crawl_none"]
    assert out["segments"][0]["boundary_source"] == "signal" and out["segments"][1]["boundary_source"] == "coarse"


def test_motion_candidates_are_hints_and_are_crawled():
    ep = moving_episode()
    events = get_source("motion").events(ep, camera=CAM)
    assert events and events[0]["type"] == "pause_start"
    plan = copy.deepcopy(PLAN)
    plan["segments"][1]["candidate_id"] = "c1"  # a close_start tied to a motion event: never snapped
    coarse = Fixed(plan)
    crawl = Scripted([-1, -1, -1])
    out = timing(ep, coarse, crawl, event_source="motion")
    assert out["events"] == events and out["event_sources"] == ["motion"]
    assert "c1: frame" in coarse.requests[0].parts[0].text and "pause_start (motion)" in coarse.requests[0].parts[0].text
    assert onsets(out["segments_coarse"]) == [100, 160, 220]
    assert out["segments_coarse"][1]["boundary_source"] == "coarse" and out["segments_coarse"][1]["candidate_id"] == "c1"
    assert [e["flags"] for e in out["crawl_log"]] == [["crawl_none"]] * 3  # close and open crawled too


def test_video_mode_then_crawl_on_frames():
    ep = moving_episode()
    plan = {"segments": [dict(item(0, 0, "close_start", "approach"), start_s=0.0, end_s=3.3),
                         dict(item(0, 0, "open_start", "grasp"), start_s=3.3, end_s=6.7),
                         dict(item(0, 0, "other", "retract"), start_s=6.7, end_s=10.0)]}
    for s in plan["segments"]:
        del s["start_frame"], s["end_frame"]
    video = VideoPart(b"fake mp4 bytes, never decoded", label="F1/0", seconds=10.0)
    coarse = Fixed(plan)
    crawl = Scripted([-1, -1])
    out = timing(ep, coarse, crawl, coarse_mode="video", video=video)
    assert onsets(out["segments_coarse"]) == [99, 201]  # round(t * fps)
    assert out["coarse_info"]["mode"] == "video" and out["coarse_info"]["coarse_fps"] is None
    assert out["coarse_mode"] == "video" and out["coarse_fps"] is None
    assert any(isinstance(p, VideoPart) for p in coarse.requests[0].parts)
    assert [r.context["frame_indices"][0] for r in crawl.requests] == [69, 171]


def test_run_timing_is_deterministic():
    ep = moving_episode()

    def once():
        crawl = Scripted([4, 5, 0, 3, 5, 9, 9])
        out = timing(ep, Fixed(PLAN), crawl, event_source="motion")
        keep = {k: v for k, v in out.items() if k not in ("calls", "episode_wall_s")}
        texts = [[p.text for p in r.parts if isinstance(p, TextPart)] for r in crawl.requests]
        return json.dumps({"out": keep, "texts": texts}, sort_keys=True).encode("utf-8")

    assert once() == once()


def test_mock_provider_works_as_both_callers():
    from robolabel.layers.crawl import crawl_boundaries
    from robolabel.providers.mock import MockProvider

    ep = moving_episode()
    out = timing(ep, MockProvider(), MockProvider())
    assert out["coarse_status"] == "ok" and len(out["segments"]) >= 1
    assert out["segments"][0]["start_frame"] == 0 and out["segments"][-1]["end_frame"] == 299
    new, log, calls = crawl_boundaries(ep, out["segments_coarse"] if len(out["segments_coarse"]) > 1 else
                                       [dict(out["segments"][0], end_frame=149, end_event="close_start"),
                                        dict(out["segments"][0], start_frame=150)],
                                       MockProvider(), camera=CAM, context={}, reasoning=None)
    assert log and log[0]["stage1_answer"] == 0 and "crawl_edge" in log[0]["flags"]  # the mock answers 0
    assert all(c.valid for c in calls)
