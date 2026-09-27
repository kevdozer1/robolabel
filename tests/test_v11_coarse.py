"""v1.1 coarse pass and prompt v8 (SPEC_V1_1 3.2, 3.3 item 10, 4 and 6): the request in frames and video
mode, the v8 schemas, and the deterministic post-processing. No network and no paid call: MockProvider
and hand-written model answers only."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robolabel.episode import Episode
from robolabel.events import candidate_lines, make_event
from robolabel.layers.coarse import (
    coarse_frame_indices,
    coarse_request,
    effective_fps,
    missing_output_segments,
    postprocess_coarse,
)
from robolabel.layers.frames import frame_line
from robolabel.prompts import v7, v8
from robolabel.providers.base import CallResult, ImagePart, TextPart, VideoPart
from robolabel.providers.mock import MockProvider

CAM = "observation.images.up"
SEGMENT_KEYS = {"start_frame", "end_frame", "phase_class", "phase_text", "target", "destination", "attempt_idx",
                "outcome", "attempt_outcome", "failure_type", "mistake", "end_event", "boundary_source",
                "coarse_end_frame", "crawl_calls", "candidate_id", "evidence"}
OBJECTS = [{"object_id": "o1", "name": "pink brick", "aliases": ["brick"], "category": "block", "views": []},
           {"object_id": "o2", "name": "transparent box", "aliases": [], "category": "container", "views": []}]


def episode(n: int = 303, fps: float = 30.0, task: str | None = "put the pink brick in the box") -> Episode:
    rng = np.random.default_rng(7)
    frames = rng.integers(0, 255, size=(8, 480, 640, 3), dtype=np.uint8)

    def get(i: int) -> np.ndarray:
        return frames[int(i) % 8]

    return Episode(episode_id="F1/999", num_frames=n, fps=fps, task=task, get_frame=get, camera_key=CAM,
                   extra={"family": "F1", "cameras": {CAM: get, "observation.images.wrist": get}})


def seg(start, end, pc="other", ev="other", *, idx=1, outcome="success", att="success", ft="none", cid="none",
        target="none", dest="none", text=None, times=False):
    s = {"phase_class": pc, "phase_text": text if text is not None else f"{pc} phase", "end_event": ev,
         "target": target, "destination": dest, "attempt_idx": idx, "outcome": outcome, "attempt_outcome": att,
         "failure_type": ft, "candidate_id": cid}
    return {**({"start_s": start, "end_s": end} if times else {"start_frame": start, "end_frame": end}), **s}


def post(segments, ep=None, *, mode="frames", objects=None, candidates=None):
    ep = ep or episode()
    repairs: list[str] = []
    out = postprocess_coarse({"segments": segments}, ep, mode=mode, info={"mode": mode}, objects=objects,
                             candidates=candidates, repairs=repairs)
    return out, repairs


def onsets(segs):
    return [s["start_frame"] for s in segs[1:]]


def assert_contiguous(segs, n):
    assert segs[0]["start_frame"] == 0 and segs[-1]["end_frame"] == n - 1
    assert all(b["start_frame"] == a["end_frame"] + 1 for a, b in zip(segs, segs[1:], strict=False))
    assert all(s["start_frame"] <= s["end_frame"] for s in segs)
    assert all(set(s) == SEGMENT_KEYS for s in segs)


class AnswerCaller:
    """A caller with a hand-written model answer (the shape OpenRouterProvider.call returns)."""

    name, model = "fake", "fake"

    def __init__(self, data):
        self.data = data
        self.requests = []

    def call(self, req):
        self.requests.append(req)
        return CallResult(True, copy.deepcopy(self.data), json.dumps(self.data), "ok", mode="json_schema_strict")


# --------------------------------------------------------------------------- prompt v8 and schemas
def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_v8_schemas_follow_the_v7_schema_rules():
    banned = {"minItems", "maxItems", "minimum", "maximum", "pattern", "format", "$ref", "oneOf", "anyOf", "allOf",
              "minLength", "maxLength"}
    for schema in v8.SCHEMAS.values():
        for node in _walk(schema):
            if not isinstance(node, dict) or "type" not in node:
                continue
            assert not banned & set(node)
            assert node["type"] in ("object", "array", "string", "integer", "number", "boolean")
            if node["type"] == "object":
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
    item = v8.SCHEMAS["coarse_frames"]["properties"]["segments"]["items"]
    assert list(item["properties"]) == ["start_frame", "end_frame", "phase_class", "phase_text", "end_event", "target",
                                        "destination", "attempt_idx", "outcome", "attempt_outcome", "failure_type",
                                        "candidate_id"]
    assert item["properties"]["start_frame"] == {"type": "integer"}
    assert item["properties"]["end_event"]["enum"] == v8.END_EVENTS
    assert v8.END_EVENTS == ["close_start", "open_start", "contact_start", "contact_end", "other"]
    assert item["properties"]["phase_class"]["enum"] == v7.PHASE_CLASSES
    assert item["properties"]["failure_type"]["enum"] == v7.FAILURE_TYPES
    assert item["properties"]["outcome"]["enum"] == item["properties"]["attempt_outcome"]["enum"] == \
        ["success", "failed", "aborted"]
    vitem = v8.SCHEMAS["coarse_video"]["properties"]["segments"]["items"]
    assert list(vitem["properties"])[:2] == ["start_s", "end_s"]
    assert vitem["properties"]["start_s"] == vitem["properties"]["end_s"] == {"type": "number"}
    assert list(vitem["properties"])[2:] == list(item["properties"])[2:]
    assert v8.SCHEMAS["crawl"] == {"type": "object", "additionalProperties": False, "required": ["answer"],
                                   "properties": {"answer": {"type": "integer"}}}
    assert v8.MAX_TOKENS == {"coarse": 8000, "crawl": 2500}
    assert v8.PHASE_CLASSES is v7.PHASE_CLASSES and v8.FAILURE_TYPES is v7.FAILURE_TYPES


def test_v8_version_and_prompt_hashes():
    assert v8.VERSION == "v8-2026-09-27.1" and v7.VERSION == "v7-2026-09-27.1"
    for name in ("system", "coarse"):
        assert v8.prompt_sha256(name) == hashlib.sha256(v8.load_prompt(name).encode("utf-8")).hexdigest()
    assert v8.prompt_sha256("system") != v7.prompt_sha256("system")
    sections = v8.prompt_sections("coarse")
    assert {"input_frames", "input_video", "task", "no_task", "objects", "candidates", "units_frames", "units_video",
            "rules", "candidate_id_list", "candidate_id_none"} <= set(sections)
    assert all(text and "===" not in text for text in sections.values())
    sections["rules"] = "changed"
    assert v8.prompt_sections("coarse")["rules"] != "changed"  # a fresh dict each time


def test_coarse_prompt_says_what_spec_v1_1_asks():
    low = " ".join(v8.load_prompt("coarse").lower().split())
    for phrase in [
        "the phase list is a vocabulary, not a template: use it only where it fits the activity, otherwise describe "
        "the phase in plain words and use other",
        "phase_text: a short description of the phase in your own plain words",
        "close_start: the fingers of a gripper or hand start to close",
        "open_start: the fingers of a gripper or hand start to open",
        "contact_start: the hand or a held tool starts touching an object or surface",
        "contact_end: the hand or a held tool stops touching an object or surface",
        "other: any other kind of boundary",
        "outcome is the result of this phase alone",
        "in a missed grasp, the approach that reached the object is success and the grasp is failed with "
        "failure_type missed_grasp",
        "copy it onto every phase of the attempt",
        "only the phase that failed has outcome failed",
        "a retry is a new attempt with attempt_idx one higher",
        "the failed grasp ends when the fingers start to reopen",
        "then comes a retract segment if the arm or hand backs off",
        "then comes the approach of the next attempt, with attempt_idx one higher",
        "the candidates are hints, not boundaries to copy",
        "in steps of 0.1 s",
    ]:
        assert phrase in low, phrase


def test_v8_files_follow_the_style_rules():
    pkg = Path(v8.__file__).resolve().parents[2]
    files = [*sorted((pkg / "prompts" / "v8").glob("*.txt")), pkg / "prompts" / "v8" / "__init__.py",
             pkg / "layers" / "coarse.py", Path(__file__).resolve()]
    banned_words = ["cave" + "at", "side" + "car", "smoke" + " test"]
    machine = re.compile("|".join(["[A-Za-z]:" + r"\\" + "Users", "D:" + r"\\", "D:" + "/", "/c/" + "Users"]))
    for p in files:
        text = p.read_text(encoding="utf-8")
        assert chr(0x2014) not in text, p.name
        assert not any(w in text.lower() for w in banned_words), p.name
        assert not machine.search(text), p.name
    for name in ("system", "coarse"):  # no example phrases in the prompts (Q153)
        low = v8.load_prompt(name).lower()
        assert "for example" not in low and "e.g." not in low and "such as" not in low


# --------------------------------------------------------------------------- the request
def test_frame_plan_is_two_per_second_capped_at_48_with_both_ends():
    assert coarse_frame_indices(303, 30.0) == coarse_frame_indices(303, 30.0)
    idx = coarse_frame_indices(303, 30.0)
    assert len(idx) == 21 and idx[0] == 0 and idx[-1] == 302
    assert effective_fps(idx, 30.0) == pytest.approx(2.0, abs=0.05)
    long = coarse_frame_indices(1800, 30.0)  # 60 s: 121 frames at 2 per second, capped at 48
    assert len(long) == 48 and long[0] == 0 and long[-1] == 1799
    assert effective_fps(long, 30.0) == pytest.approx(47 / (1799 / 30.0), abs=1e-3)
    f3 = coarse_frame_indices(587, 20.0)
    assert len(f3) == 48 and f3[-1] == 586  # 29.3 s at 2 per second would be 60 frames
    assert coarse_frame_indices(10, 30.0) == [0, 9]
    assert coarse_frame_indices(1, 30.0) == [0]
    assert coarse_frame_indices(303, 30.0, max_frames=10, fps_target=1.0) == \
        sorted({int(round(x)) for x in np.linspace(0, 302, 10)})


def test_frames_request_sends_captioned_frames_at_448_px():
    ep = episode()
    ctx = {"arm": "L-A", "episode_key": ep.episode_id, "bucket": "e1_luna", "model_key": "luna"}
    req, info = coarse_request(ep, camera=CAM, mode="frames", context=ctx, reasoning={"effort": "low"})
    idx = info["frame_indices"]
    assert info["mode"] == "frames" and set(info) == {"frame_indices", "coarse_fps", "mode"}
    assert idx == coarse_frame_indices(303, 30.0) and info["coarse_fps"] == effective_fps(idx, 30.0)
    assert req.step == "coarse" and req.schema == v8.SCHEMAS["coarse_frames"] and req.max_tokens == 8000
    assert req.schema_name == "coarse_frames_v8" and req.reasoning == {"effort": "low"}
    assert req.system == v8.load_prompt("system").strip()
    assert {**ctx, "frame_indices": idx, "cameras": ["up"], "coarse_mode": "frames"} == req.context
    assert ctx == {"arm": "L-A", "episode_key": ep.episode_id, "bucket": "e1_luna", "model_key": "luna"}
    media = req.parts[1:]
    assert len(media) == 2 * len(idx)
    for k, f in enumerate(idx):
        caption, image = media[2 * k], media[2 * k + 1]
        assert isinstance(caption, TextPart) and caption.text == frame_line(f, 303, 30.0, CAM)
        assert isinstance(image, ImagePart) and image.label == f"{CAM}@{f}"
        assert Image.open(io.BytesIO(image.jpeg)).size == (448, 336)
    text = req.parts[0].text
    assert text.startswith("Label the phases of one clip of 303 frames (10.1 s at 30 frames per second).")
    assert "21 frames of the clip from camera up, evenly spaced at about 2.0 per second" in text
    assert 'The dataset\'s task description is: "put the pink brick in the box".' in text
    assert "the last segment ends at frame 302" in text and "start_s" not in text
    assert "in a few plain words" in text and "Objects in the scene" not in text
    assert "{" not in text and "}" not in text  # every placeholder filled


def test_requests_are_deterministic():
    ep = episode()
    a, _ = coarse_request(ep, camera=CAM, mode="frames", context={}, reasoning=None)
    b, _ = coarse_request(ep, camera=CAM, mode="frames", context={}, reasoning=None)
    assert [p.text if isinstance(p, TextPart) else p.jpeg for p in a.parts] == \
        [p.text if isinstance(p, TextPart) else p.jpeg for p in b.parts]


def test_video_request_sends_the_callers_video_and_no_frames():
    ep = episode()
    video = VideoPart(b"\x00\x00\x00\x18ftypisom not a real clip", seconds=10.1, label="F1/999 up")
    req, info = coarse_request(ep, camera=CAM, mode="video", context={"arm": "G-V"}, reasoning=None, video=video)
    assert info == {"frame_indices": [], "coarse_fps": None, "mode": "video"}
    assert len(req.parts) == 2 and req.parts[1] is video
    assert not any(isinstance(p, ImagePart) for p in req.parts)
    assert req.context["frame_indices"] == [] and req.context["video_seconds"] == 10.1
    assert req.context["arm"] == "G-V" and req.context["coarse_mode"] == "video"
    assert req.schema == v8.SCHEMAS["coarse_video"] and req.schema_name == "coarse_video_v8"
    text = req.parts[0].text
    assert text.startswith("Label the phases of one video clip of 10.1 s from camera up.")
    assert "in steps of 0.1 s" in text and "the last segment ends at 10.1." in text
    assert "start_frame" not in text and "{" not in text
    with pytest.raises(ValueError):
        coarse_request(ep, camera=CAM, mode="video", context={}, reasoning=None)
    with pytest.raises(ValueError):
        coarse_request(ep, camera=CAM, mode="slides", context={}, reasoning=None)


@pytest.mark.parametrize("cands", [None, []])
def test_no_candidate_list_without_candidates(cands):
    req, _ = coarse_request(episode(), camera=CAM, mode="frames", context={}, reasoning=None, candidates=cands)
    text = req.parts[0].text
    assert "Candidate events" not in text and "hints, not boundaries" not in text
    assert "candidate_id: none (there is no candidate list)." in text


def test_candidate_lines_go_in_as_plain_hints():
    ep = episode()
    evs = [make_event("pause_start", 60, 0.8, "motion"), make_event("pause_end", 150, 0.7, "motion")]
    req, _ = coarse_request(ep, camera=CAM, mode="frames", context={}, reasoning=None, candidates=evs)
    text = req.parts[0].text
    lines = candidate_lines(evs, ep.num_frames, ep.fps)
    assert len(lines) == 2 and "\n".join(lines) in text
    assert "The candidates are hints, not boundaries to copy." in text
    assert "the part of its line before the colon" in text
    # candidates that give no line (every frame outside the clip) give no candidate list
    req, _ = coarse_request(ep, camera=CAM, mode="frames", context={}, reasoning=None,
                            candidates=[make_event("pause_start", 900, 0.8, "motion")])
    assert "Candidate events" not in req.parts[0].text
    assert "candidate_id: none (there is no candidate list)." in req.parts[0].text


def test_inventory_ids_or_plain_words_and_the_task_string():
    req, _ = coarse_request(episode(), camera=CAM, mode="frames", context={}, reasoning=None, objects=OBJECTS)
    text = req.parts[0].text
    assert "Objects in the scene (use these IDs):\no1: pink brick (block)\no2: transparent box (container)" in text
    assert "the ID of the object the phase acts on, from the object list above" in text
    assert "in a few plain words" not in text
    req, _ = coarse_request(episode(task=None), camera=CAM, mode="frames", context={}, reasoning=None)
    assert "The dataset gives no task description." in req.parts[0].text


def test_single_stream_episode_uses_its_own_frames():
    ep = episode()
    plain = Episode(episode_id="C/clip", num_frames=90, fps=30.0, task=None, get_frame=ep.get_frame)
    req, info = coarse_request(plain, camera="clip", mode="frames", context={}, reasoning=None)
    assert len(info["frame_indices"]) == 7 and sum(isinstance(p, ImagePart) for p in req.parts) == 7
    with pytest.raises(KeyError):
        coarse_request(ep, camera="observation.images.side", mode="frames", context={}, reasoning=None)


# --------------------------------------------------------------------------- post-processing
def test_mock_provider_answer_becomes_one_valid_segment():
    ep = episode()
    for mode, video in (("frames", None), ("video", VideoPart(b"x", seconds=10.1))):
        req, info = coarse_request(ep, camera=CAM, mode=mode, context={}, reasoning=None, video=video)
        res = MockProvider().call(req)
        assert res.valid
        repairs: list[str] = []
        segs = postprocess_coarse(res.data, ep, mode=mode, info=info, repairs=repairs)
        assert len(segs) == 1 and repairs
        assert_contiguous(segs, ep.num_frames)
        assert segs[0]["end_event"] == "other" and segs[0]["candidate_id"] == "none"
        assert segs[0]["boundary_source"] == "coarse" and segs[0]["crawl_calls"] == 0


def test_hand_written_answer_through_a_caller():
    ep = episode()
    answer = {"segments": [seg(0, 99, "approach", "close_start", target="the pink brick"),
                           seg(100, 302, "grasp", "other", target="the pink brick")]}
    caller = AnswerCaller(answer)
    req, info = coarse_request(ep, camera=CAM, mode="frames", context={}, reasoning=None)
    res = caller.call(req)
    repairs: list[str] = []
    segs = postprocess_coarse(res.data, ep, mode="frames", info=info, repairs=repairs)
    assert repairs == [] and onsets(segs) == [100]
    assert [s["target"] for s in segs] == ["the pink brick", "the pink brick"]
    assert [s["coarse_end_frame"] for s in segs] == [99, 302]


def test_contiguity_and_bounds_are_repaired():
    segs, repairs = post([
        seg(95, 200, "transport", "open_start"),
        seg(45, 100, "grasp", "other"),
        "not a segment",
        seg("x", 3, "approach"),
        seg(5, 40, "approach", "close_start"),
        seg(400, 210, "release", "contact_end"),  # swapped and past the end
    ])
    assert_contiguous(segs, 303)
    assert [s["phase_class"] for s in segs] == ["approach", "grasp", "transport", "release"]
    assert onsets(segs) == [45, 95, 210]
    assert [s["coarse_end_frame"] for s in segs] == [44, 94, 209, 302]
    assert segs[-1]["end_event"] == "other" and segs[1]["end_event"] == "other"
    for text in ("not an object was dropped", "without integer frames was dropped", "clamped to (302, 210)",
                 "start and end swapped", "gap before frame 45", "overlap before frame 95", "gap before frame 210",
                 "first segment extended to frame 0", "end_event 'contact_end' of the last segment set to other"):
        assert any(text in r for r in repairs), text
    assert all(s["boundary_source"] == "coarse" and s["crawl_calls"] == 0 and s["evidence"] == [] for s in segs)


def test_bad_values_are_coerced_and_recorded():
    segs, repairs = post([
        {**seg(0, 150, "reach", "touch", outcome="meh", att="partial", ft="oops", idx="2nd", text=""),
         "start_frame": 0.0},
        seg(151, 302, "grasp", "other", idx=-3, text="x" * 300),
    ])
    a, b = segs
    assert a["phase_class"] == "other" and a["end_event"] == "other" and a["outcome"] == "success"
    assert a["phase_text"] == "other" and a["attempt_idx"] == 1 and b["attempt_idx"] == 1
    assert a["failure_type"] == "none" and a["attempt_outcome"] == "success"
    assert len(b["phase_text"]) == 120
    for text in ("converted to integers", "is not a phase class", "end_event 'touch'", "outcome 'meh'",
                 "attempt_outcome 'partial'", "failure_type 'oops'", "attempt_idx '2nd'", "attempt_idx -3",
                 "empty phase_text", "cut to 120 characters"):
        assert any(text in r for r in repairs), text


def test_seconds_become_frames_with_round_t_times_fps():
    ep = episode()
    answer = [seg(0.0, 2.3, "approach", "close_start", times=True),
              seg(2.3, 3.1, "grasp", "other", times=True),
              seg(3.1, 7.0, "transport", "open_start", times=True),
              seg(7.0, 10.1, "retract", "other", times=True)]
    segs, repairs = post(answer, ep, mode="video")
    assert repairs == []
    assert onsets(segs) == [round(2.3 * 30), round(3.1 * 30), round(7.0 * 30)] == [69, 93, 210]
    assert_contiguous(segs, 303)
    assert [s["coarse_end_frame"] for s in segs] == [68, 92, 209, 302]
    # rounding past the end by at most 0.1 s is part of the conversion; far past it is a repair
    segs, repairs = post([seg(0.0, 5.0, times=True), seg(5.0, 10.2, times=True)], ep, mode="video")
    assert repairs == [] and segs[-1]["end_frame"] == 302
    segs, repairs = post([seg(0.0, 5.0, times=True), seg(5.0, 14.0, times=True)], ep, mode="video")
    assert any("outside the clip, frames clamped" in r for r in repairs) and segs[-1]["end_frame"] == 302
    # a gap in time is a gap in frames; a segment of no length or past the clip is dropped
    segs, repairs = post([seg(0.0, 4.0, times=True), seg(4.0, 4.0, times=True), seg(5.0, 10.1, times=True),
                          seg(11.0, 12.0, times=True), seg("a", 1.0, times=True)], ep, mode="video")
    assert onsets(segs) == [150] and len(segs) == 2
    for text in ("gap before frame 150", "covers no frame, dropped", "lies outside the clip, dropped",
                 "without numeric times was dropped"):
        assert any(text in r for r in repairs), text
    # 20 fps (F3): steps of 0.1 s are whole frames
    f3 = episode(n=587, fps=20.0)
    segs, repairs = post([seg(0.0, 12.4, "approach", "close_start", times=True),
                          seg(12.4, 29.4, "grasp", times=True)], f3, mode="video")
    assert onsets(segs) == [248] and segs[-1]["end_frame"] == 586 and repairs == []


def test_missed_grasp_and_recovery_follow_spec_4_unchanged():
    answer = [
        seg(0, 40, "approach", "close_start", att="failed", target="brick"),
        seg(41, 80, "grasp", "open_start", outcome="failed", att="failed", ft="missed_grasp", target="brick"),
        seg(81, 110, "retract", "other", att="failed"),
        seg(111, 160, "approach", "close_start", idx=2, target="brick"),
        seg(161, 302, "grasp", "other", idx=2, target="brick"),
    ]
    segs, repairs = post(answer)
    assert repairs == []
    assert [s["outcome"] for s in segs] == ["success", "failed", "success", "success", "success"]
    assert [s["attempt_outcome"] for s in segs] == ["failed"] * 3 + ["success"] * 2
    assert [s["mistake"] for s in segs] == [False, True, False, False, False]
    assert [s["failure_type"] for s in segs] == ["none", "missed_grasp", "none", "none", "none"]
    assert [s["attempt_idx"] for s in segs] == [1, 1, 1, 2, 2]


def test_old_convention_keeps_failed_only_on_the_phase_that_failed():
    answer = [
        seg(0, 40, "approach", "close_start", outcome="failed", att="failed", ft="missed_grasp"),
        seg(41, 80, "grasp", "open_start", outcome="failed", att="failed", ft="missed_grasp"),
        seg(81, 110, "retract", "other", outcome="failed", att="failed", ft="missed_grasp"),
        seg(111, 302, "approach", "other", idx=2),
    ]
    segs, repairs = post(answer)
    assert [s["outcome"] for s in segs] == ["success", "failed", "success", "success"]
    assert [s["failure_type"] for s in segs] == ["none", "missed_grasp", "none", "none"]
    assert [s["mistake"] for s in segs] == [False, True, False, False]
    assert [s["attempt_outcome"] for s in segs] == ["failed", "failed", "failed", "success"]
    assert any("attempt 1 has 3 failed phases; only the grasp at 41 keeps outcome failed" in r for r in repairs)
    # no phase matches the failure type: the last failed phase that is not a retract
    segs, _ = post([seg(0, 100, "push", outcome="failed", att="failed", ft="other"),
                    seg(101, 200, "rotate", outcome="failed", att="failed", ft="other"),
                    seg(201, 302, "retract", outcome="failed", att="failed", ft="other")])
    assert [s["mistake"] for s in segs] == [False, True, False]


def test_attempt_outcome_is_derived_and_copied():
    answer = [
        seg(0, 40, "approach", "close_start", att="success"),
        seg(41, 80, "grasp", "open_start", outcome="failed", att="failed", ft="slip"),
        seg(81, 150, "approach", "close_start", idx=2, att="failed"),
        seg(151, 200, "grasp", "other", idx=2, att="failed"),
        seg(201, 250, "approach", "other", idx=3, outcome="aborted", att="success"),
        seg(251, 302, "retract", "other", idx=1),  # attempt_idx goes back: raised to 3
    ]
    segs, repairs = post(answer)
    assert [s["attempt_idx"] for s in segs] == [1, 1, 2, 2, 3, 3]
    # attempt 2: every phase says attempt_outcome failed, no phase says outcome failed. The failure is kept
    # (not derived away to success) and the last phase that is not a retract carries it.
    assert [s["attempt_outcome"] for s in segs] == ["failed", "failed", "failed", "failed", "aborted", "aborted"]
    assert [s["outcome"] for s in segs] == ["success", "failed", "success", "failed", "aborted", "success"]
    assert [s["failure_type"] for s in segs] == ["none", "slip", "none", "other", "aborted", "none"]
    assert [s["mistake"] for s in segs] == [False, True, False, True, False, False]
    for text in ("attempt 1 attempt_outcome failed/success set to failed", "attempt 2 attempt_outcome failed but no "
                 "failed phase; kept failed, the grasp at 151 set to failed (other)", "attempt 3 attempt_outcome success "
                 "set to aborted", "attempt_idx 1 at 251 follows attempt 3, set to 3", "aborted phase at 201 without a "
                 "failure_type, set to aborted"):
        assert any(text in r for r in repairs), text
    assert not any("set to success" in r for r in repairs)
    # the phase the answer gave a failure type to carries the failure, with that type
    segs, repairs = post([seg(0, 50, "approach", "close_start", att="failed"),
                          seg(51, 120, "grasp", "open_start", att="failed", ft="missed_grasp"),
                          seg(121, 200, "retract", att="failed"), seg(201, 302, "approach", idx=2)])
    assert [s["outcome"] for s in segs] == ["success", "failed", "success", "success"]
    assert [s["failure_type"] for s in segs] == ["none", "missed_grasp", "none", "none"]
    assert [s["attempt_outcome"] for s in segs] == ["failed", "failed", "failed", "success"]
    assert [s["mistake"] for s in segs] == [False, True, False, False]
    # an aborted attempt named only in attempt_outcome stays aborted (no mistake: nothing failed)
    segs, repairs = post([seg(0, 100, "approach", att="aborted"), seg(101, 200, "retract", att="aborted"),
                          seg(201, 302, "approach", idx=2)])
    assert [s["attempt_outcome"] for s in segs] == ["aborted", "aborted", "success"]
    assert [s["outcome"] for s in segs] == ["aborted", "success", "success"]
    assert [s["failure_type"] for s in segs] == ["aborted", "none", "none"]
    assert [s["mistake"] for s in segs] == [False, False, False]
    # mixed answers (failed on one phase, success on another) and no failed phase: derived, as before
    segs, repairs = post([seg(0, 100, "approach", att="failed"), seg(101, 302, "grasp", att="success")])
    assert [s["attempt_outcome"] for s in segs] == ["success", "success"]
    assert any("attempt 1 attempt_outcome failed/success set to success" in r for r in repairs)
    segs, repairs = post([seg(0, 100, "grasp", ft="drop"), seg(101, 302, "grasp", outcome="failed")])
    assert [s["failure_type"] for s in segs] == ["none", "other"]
    assert any("of a successful phase at 0 set to none" in r for r in repairs)
    assert any("failed phase at 101 without a failure_type, set to other" in r for r in repairs)


GRIPPER = [make_event("close_start", 52, 1.0, "gripper", 1), make_event("open_start", 150, 1.0, "gripper", 1),
           make_event("arm_move", 170, 0.3, "gripper", 1)]


def test_gripper_candidates_snap_close_and_open_boundaries():
    answer = [seg(0, 47, "approach", "close_start", cid="c1"),
              seg(48, 139, "transport", "open_start", cid="c2"),
              seg(140, 179, "release", "other", cid="c3"),
              seg(180, 302, "retract", "other")]
    segs, repairs = post(answer, candidates=GRIPPER)
    assert onsets(segs) == [52, 150, 180]
    assert [s["boundary_source"] for s in segs] == ["signal", "signal", "coarse", "coarse"]
    assert [s["coarse_end_frame"] for s in segs] == [47, 139, 179, 302]
    assert [s["candidate_id"] for s in segs] == ["c1", "c2", "c3", "none"]
    assert_contiguous(segs, 303)
    assert any("close_start onset 48 snapped to c1 at 52" in r for r in repairs)
    assert any("open_start onset 140 snapped to c2 at 150" in r for r in repairs)


def test_gripper_snap_needs_a_matching_type_and_a_frame_in_reach():
    segs, repairs = post([seg(0, 47, "approach", "close_start", cid="c2"), seg(48, 302, "grasp")],
                         candidates=GRIPPER)
    assert onsets(segs) == [48] and segs[0]["boundary_source"] == "coarse"
    assert any("c2 is a open_start event" in r for r in repairs)
    segs, repairs = post([seg(0, 47, "approach", "close_start", cid="c1"), seg(48, 50, "grasp", "open_start"),
                          seg(51, 302, "transport")], candidates=GRIPPER)
    assert onsets(segs) == [48, 51] and segs[0]["boundary_source"] == "coarse"
    assert any("lies outside the segments around" in r for r in repairs)


def test_motion_candidates_are_hints_and_never_snapped():
    motion = [make_event("pause_start", 52, 0.9, "motion"), make_event("pause_end", 150, 0.9, "motion")]
    answer = [seg(0, 47, "approach", "close_start", cid="c1"), seg(48, 139, "transport", "open_start", cid="c2"),
              seg(140, 302, "release")]
    segs, repairs = post(answer, candidates=motion)
    assert onsets(segs) == [48, 140] and repairs == []
    assert all(s["boundary_source"] == "coarse" for s in segs)
    assert [s["candidate_id"] for s in segs] == ["c1", "c2", "none"]
    # without a candidate list a candidate ID is unknown
    segs, repairs = post(answer)
    assert [s["candidate_id"] for s in segs] == ["none", "none", "none"]
    assert any("unknown candidate 'c1'" in r for r in repairs)


def test_targets_are_inventory_ids_or_plain_words():
    answer = [seg(0, 50, "approach", target="pink brick", dest="none"),
              seg(51, 100, "grasp", target="O1", dest=""),
              seg(101, 200, "transport", target="banana", dest="o2"),
              seg(201, 302, "release", target="Unsure", dest="brick")]
    segs, repairs = post(answer, objects=OBJECTS)
    assert [s["target"] for s in segs] == ["o1", "o1", "unsure", "unsure"]
    assert [s["destination"] for s in segs] == ["none", "none", "o2", "o1"]
    assert any("'pink brick' at frame 0 given as a name, mapped to o1" in r for r in repairs)
    assert any("'banana' at frame 101 is not an inventory ID" in r for r in repairs)
    segs, repairs = post([seg(0, 150, target="  the pink brick ", dest="NONE"), seg(151, 302, target="y" * 200)])
    assert [s["target"] for s in segs] == ["the pink brick", "y" * 120]
    assert segs[0]["destination"] == "none"
    assert any("target at frame 151 cut to 120 characters" in r for r in repairs)


def test_no_usable_answer_gives_the_missing_output_segment():
    ep = episode()
    for data in (None, {}, {"segments": []}, {"segments": ["x", {"start_frame": None}]}):
        repairs: list[str] = []
        segs = postprocess_coarse(data, ep, mode="frames", info=None, repairs=repairs)
        assert segs == missing_output_segments(303) and repairs
    miss = missing_output_segments(303)[0]
    assert set(miss) == SEGMENT_KEYS and miss["end_event"] == "other" and miss["end_frame"] == 302
    assert miss["attempt_outcome"] == "success" and miss["mistake"] is False and miss["coarse_end_frame"] == 302
    with pytest.raises(ValueError):
        postprocess_coarse({}, ep, mode="frames", info={"mode": "video"}, repairs=[])


def test_postprocessing_is_deterministic_and_leaves_the_answer_alone():
    answer = {"segments": [seg(95, 200, "transport", "open_start", outcome="failed", att="success", ft="drop"),
                           seg(5, 47, "approach", "close_start", cid="c1"), seg(45, 100, "grasp", "other"),
                           seg(210, 400, "release", "contact_end")]}
    before = json.dumps(answer, sort_keys=True)
    runs = []
    for _ in range(2):
        repairs: list[str] = []
        segs = postprocess_coarse(answer, episode(), mode="frames", info={"mode": "frames"}, candidates=GRIPPER,
                                  repairs=repairs)
        runs.append(json.dumps({"segments": segs, "repairs": repairs}, sort_keys=True))
    assert runs[0] == runs[1]
    assert json.dumps(answer, sort_keys=True) == before
