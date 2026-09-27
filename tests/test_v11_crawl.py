"""The crawl (SPEC_V1_1 3.3): windows, answers, edges, stage 2, caps, crossing, logs; scripted fake callers only."""

from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
from PIL import Image

from robolabel.episode import Episode
from robolabel.layers import crawl as C
from robolabel.layers.crawl import (
    contact_object,
    crawl_boundaries,
    crawl_request,
    interpret,
    shift_window,
    stage1_frames,
    stage2_frames,
)
from robolabel.prompts.v8 import MAX_TOKENS, SCHEMAS, prompt_sha256
from robolabel.providers.base import CallResult, ImagePart, TextPart

CAM = "observation.images.up"
LOG_KEYS = {"boundary_index", "event_type", "coarse_frame", "stage1_frames", "stage1_answer", "retry_frames",
            "retry_answer", "stage2_frames", "stage2_answer", "onset", "flags", "calls", "usd"}


# ------------------------------------------------------------------------------------------ fixtures
def make_episode(n: int = 300, fps: float = 30.0, h: int = 48, w: int = 64, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 255, size=(n, h, w, 3), dtype=np.uint8)

    def get(i):
        return frames[int(i)]

    return Episode(episode_id="F1/0", num_frames=n, fps=fps, task="put the block in the box", get_frame=get,
                   camera_key=CAM, extra={"family": "F1", "cameras": {CAM: get}, "camera_order": [CAM],
                                          "external_cameras": [CAM], "wrist_cameras": []})


def seg(start: int, end: int, end_event: str = "other", **kw) -> dict:
    d = {"start_frame": start, "end_frame": end, "phase_class": "other", "phase_text": f"phase from {start}",
         "target": "none", "destination": "none", "attempt_idx": 1, "outcome": "success",
         "attempt_outcome": "success", "failure_type": "none", "mistake": False, "end_event": end_event,
         "boundary_source": "coarse", "coarse_end_frame": end, "crawl_calls": 0, "candidate_id": "none",
         "evidence": []}
    d.update(kw)
    return d


def chain(onsets: list[int], types: list[str], n: int) -> list[dict]:
    """Contiguous segments with boundaries at ``onsets`` (start frames of the later segment), typed ``types``."""
    starts = [0, *onsets]
    ends = [o - 1 for o in onsets] + [n - 1]
    return [seg(s, e, (types[i] if i < len(types) else "other")) for i, (s, e) in enumerate(zip(starts, ends, strict=True))]


class Scripted:
    """A fake caller answering from a list, one item per call: an int answer or a status string."""

    name = "scripted"
    model = "scripted"

    def __init__(self, answers, usd: float = 0.001):
        self.answers = list(answers)
        self.usd = usd
        self.requests = []

    def call(self, req):
        self.requests.append(req)
        a = self.answers.pop(0)
        if isinstance(a, str):
            return CallResult(False, None, "", a, error=f"scripted {a}", usd=0.0005, wall_s=0.5)
        data = {"answer": a}
        return CallResult(True, data, json.dumps(data), "ok", usd=self.usd, wall_s=0.25)

    def seen(self) -> list[list[int]]:
        return [r.context["frame_indices"] for r in self.requests]


def run(segments, answers, *, ep=None, **kw):
    ep = ep or make_episode()
    caller = Scripted(answers)
    out = crawl_boundaries(ep, segments, caller, camera=CAM, context={"arm": "t", "bucket": "debug"},
                           reasoning={"effort": "low"}, **kw)
    assert caller.answers == [], "every scripted answer must be used"
    return (*out, caller)


S1_100 = [70, 79, 87, 96, 104, 113, 121, 130]  # stage 1 around frame 100 at 30 fps


# ------------------------------------------------------------------------------------------ windows
def test_stage1_is_8_frames_over_plus_minus_one_second():
    assert stage1_frames(100, 300, 30.0) == S1_100
    assert stage1_frames(100, 300, 20.0) == [80, 86, 91, 97, 103, 109, 114, 120]
    assert stage1_frames(100, 300, 29.97) == S1_100  # 1.0 s rounds to 30 frames


def test_stage1_is_clamped_by_shifting_not_shrinking():
    start = stage1_frames(5, 300, 30.0)
    end = stage1_frames(295, 300, 30.0)
    assert start == [0, 9, 17, 26, 34, 43, 51, 60]
    assert end == [239, 248, 256, 265, 273, 282, 290, 299]
    assert start[-1] - start[0] == end[-1] - end[0] == 60  # full width at both edges
    assert stage1_frames(-40, 300, 30.0) == start and stage1_frames(900, 300, 30.0) == end


def test_stage1_on_short_clips():
    assert stage1_frames(10, 20, 30.0) == [0, 3, 5, 8, 11, 14, 16, 19]  # the clip is shorter than the window
    assert stage1_frames(2, 5, 30.0) == [0, 1, 2, 3, 4]  # fewer frames than 8: unique frames only
    assert stage1_frames(0, 1, 30.0) == [0]


def test_shift_window_moves_by_its_own_width():
    assert shift_window(S1_100, -1, 300) == [10, 19, 27, 36, 44, 53, 61, 70]
    assert shift_window(S1_100, 1, 300) == [130, 139, 147, 156, 164, 173, 181, 190]
    at_start = stage1_frames(0, 300, 30.0)
    assert shift_window(at_start, -1, 300) == at_start  # nowhere to go
    assert shift_window([20, 29, 37, 46, 54, 63, 71, 80], -1, 300) == at_start  # shifted back inside at full width
    assert shift_window([239, 248, 256, 265, 273, 282, 290, 299], 1, 300) == [239, 248, 256, 265, 273, 282, 290, 299]
    with pytest.raises(ValueError):
        shift_window(S1_100, 0, 300)


def test_stage2_native_spacing_or_finest_that_fits():
    assert stage2_frames(96, 101) == [96, 97, 98, 99, 100, 101]
    assert stage2_frames(96, 103) == list(range(96, 104))  # exactly 8
    assert stage2_frames(96, 104) == [96, 97, 98, 99, 101, 102, 103, 104]
    assert stage2_frames(87, 96) == [87, 88, 90, 91, 92, 93, 95, 96]
    assert stage2_frames(0, 70) == [0, 10, 20, 30, 40, 50, 60, 70]
    assert stage2_frames(5, 5) == [5]
    for lo, hi in ((0, 7), (3, 50), (100, 131)):
        f = stage2_frames(lo, hi)
        assert f[0] == lo and f[-1] == hi and len(f) == min(8, hi - lo + 1) and f == sorted(set(f))


def test_window_helpers_are_byte_identical_on_repeat():
    def dump():
        out = []
        for n, fps in ((300, 30.0), (211, 20.0), (40, 15.0), (7, 30.0)):
            for c in range(-5, n + 5, 3):
                s1 = stage1_frames(c, n, fps)
                out.append([s1, shift_window(s1, -1, n), shift_window(s1, 1, n),
                            [stage2_frames(a, b) for a, b in zip(s1, s1[1:], strict=False)]])
        return json.dumps(out, separators=(",", ":")).encode("utf-8")

    first = dump()
    assert first == dump()
    assert b"." not in first  # integers only


# ------------------------------------------------------------------------------------------ answers
def test_interpret_every_answer():
    fr = S1_100
    assert interpret(4, fr) == {"answer": 4, "read_as": 4, "kind": "pick", "index": 3, "frame": 96}
    for k in range(2, 9):
        assert interpret(k, fr)["frame"] == fr[k - 1]
    assert interpret(0, fr)["kind"] == "already" and interpret(0, fr)["frame"] == 70
    one = interpret(1, fr)
    assert one["kind"] == "already" and one["read_as"] == 0 and one["answer"] == 1
    assert interpret(9, fr)["kind"] == "not_begun" and interpret(9, fr)["frame"] == 130
    assert interpret(-1, fr)["kind"] == "none" and interpret(-1, fr)["frame"] is None
    for bad in (10, -2, 7, None, "4", True, 3.5):
        assert interpret(bad, fr[:6] if bad == 7 else fr)["kind"] == "invalid"
    assert interpret(5.0, fr)["kind"] == "pick"


def test_request_validator_allows_only_the_answer_set():
    ep = make_episode()
    req = crawl_request(ep, [96, 97, 98, 99, 100], camera=CAM, event_type="close_start")
    for ok in (-1, 0, 1, 2, 5, 9):
        assert req.validate({"answer": ok}) == []
    for bad in (6, 8, 10, -2, "3", None):
        assert req.validate({"answer": bad})
    assert SCHEMAS["crawl"] == {"type": "object", "additionalProperties": False, "required": ["answer"],
                                "properties": {"answer": {"type": "integer"}}}


# ------------------------------------------------------------------------------------------ the request
def test_request_parts_captions_question_and_settings():
    ep = make_episode()
    req = crawl_request(ep, S1_100, camera=CAM, event_type="close_start", phases=("approach the block", "grasp it"),
                        context={"arm": "L-B", "bucket": "e1_luna"}, reasoning={"effort": "low"}, stage="stage1",
                        image_tokens_per_image=160.0, start_mode="json_object_with_schema_in_prompt")
    assert req.step == "crawl" and req.schema == SCHEMAS["crawl"] and req.max_tokens == MAX_TOKENS["crawl"] == 2500
    assert req.reasoning == {"effort": "low"} and req.image_tokens_per_image == 160.0
    assert req.start_mode == "json_object_with_schema_in_prompt"
    assert req.context["frame_indices"] == S1_100 and req.context["cameras"] == ["up"]
    assert req.context["arm"] == "L-B" and req.context["bucket"] == "e1_luna" and req.context["crawl_stage"] == "stage1"
    head, *mid, tail = req.parts
    assert isinstance(head, TextPart) and isinstance(tail, TextPart)
    assert "Which of these frames is the first where the fingers (gripper or hand) have started to close?" in head.text
    assert '"approach the block"' in head.text and '"grasp it"' in head.text
    assert "2 to 8" in head.text and "9:" in head.text and "-1:" in head.text and "camera up" in head.text
    assert "{" not in head.text and "{" not in tail.text
    captions = [p.text for p in mid[0::2]]
    assert captions == [f"image {k} of 8 (frame {f})" for k, f in enumerate(S1_100, start=1)]
    assert all(isinstance(p, ImagePart) for p in mid[1::2]) and len(mid) == 16
    assert [p.label for p in mid[1::2]] == [f"{CAM}@{f}" for f in S1_100]


def test_request_questions_per_event_type():
    ep = make_episode()
    texts = {t: crawl_request(ep, S1_100, camera=CAM, event_type=t, obj="paper").parts[0].text
             for t in C.CRAWL_EVENTS}
    assert "first where the fingers (gripper or hand) have started to close?" in texts["close_start"]
    assert "first where the fingers have started to open?" in texts["open_start"]
    assert "first where the hand or the tool touches the paper?" in texts["contact_start"]
    assert "first where the hand or the tool no longer touches the paper?" in texts["contact_end"]


def test_request_fewer_images_says_so():
    ep = make_episode()
    req = crawl_request(ep, [96, 97, 98, 99, 100, 101], camera=CAM, event_type="open_start")
    assert "2 to 6" in req.parts[0].text and "by image 6" in req.parts[0].text
    assert req.parts[1].text == "image 1 of 6 (frame 96)" and len(req.parts) == 1 + 12 + 1


def test_request_images_are_448_px_and_byte_identical():
    ep = make_episode(n=40, h=480, w=640)
    a = crawl_request(ep, [1, 2, 3], camera=CAM, event_type="close_start")
    b = crawl_request(ep, [1, 2, 3], camera=CAM, event_type="close_start")
    img = Image.open(io.BytesIO(a.parts[2].jpeg))
    assert max(img.size) == 448
    assert [getattr(p, "jpeg", None) or p.text for p in a.parts] == [getattr(p, "jpeg", None) or p.text for p in b.parts]


def test_placeholders_in_model_text_are_not_expanded():
    ep = make_episode()
    req = crawl_request(ep, S1_100, camera=CAM, event_type="contact_start", obj="{question} box",
                        phases=("reach {phases}", "touch {n}"))
    text = req.parts[0].text
    assert "touches the {question} box?" in text and '"reach {phases}"' in text and '"touch {n}"' in text


def test_contact_object_from_target_and_destination():
    approach = seg(0, 9, "contact_start", target="the cup")
    grasp = seg(10, 19, "other", target="cup")
    assert contact_object(approach, grasp, "contact_start") == "cup"
    to_paper = seg(0, 9, "contact_start", target="the pencil", destination="the paper")
    draw = seg(10, 19, "contact_end", target="pencil", destination="paper")
    lift = seg(20, 29, "other", target="the pencil")
    assert contact_object(to_paper, draw, "contact_start") == "paper"
    assert contact_object(draw, lift, "contact_end") == "paper"
    place = seg(0, 9, "contact_end", target="a cup", destination="the shelf")
    withdraw = seg(10, 19, "other", target="none")
    assert contact_object(place, withdraw, "contact_end") == "cup"
    assert contact_object(seg(0, 9, target="unsure"), seg(10, 19, target="none"), "contact_start") == "object"
    names = {"o1": "red block"}
    assert contact_object(seg(0, 9, target="o1"), seg(10, 19, target="o1"), "contact_start", names) == "red block"


# ------------------------------------------------------------------------------------------ the driver
def test_stage1_pick_then_stage2_pick():
    segs = chain([100], ["close_start"], 300)
    before = copy.deepcopy(segs)
    new, log, calls, caller = run(segs, [4, 5])
    assert segs == before  # the input is not changed
    assert caller.seen() == [S1_100, [87, 88, 90, 91, 92, 93, 95, 96]]
    assert new[0]["end_frame"] == 91 and new[1]["start_frame"] == 92
    assert new[0]["coarse_end_frame"] == 99 and new[0]["boundary_source"] == "crawl" and new[0]["crawl_calls"] == 2
    assert new[1]["boundary_source"] == "coarse" and new[1]["crawl_calls"] == 0
    (e,) = log
    assert LOG_KEYS <= set(e)
    assert e["stage1_frames"] == S1_100 and e["stage1_answer"] == 4 and e["retry_frames"] is None
    assert e["stage2_frames"] == [87, 88, 90, 91, 92, 93, 95, 96] and e["stage2_answer"] == 5
    assert e["onset"] == 92 and e["pick"] == 92 and e["flags"] == [] and e["calls"] == 2 and e["usd"] == 0.002
    assert e["coarse_frame"] == 100 and e["event_type"] == "close_start" and e["boundary_index"] == 0
    assert [c["frames"] for c in e["call_log"]] == caller.seen()
    assert [c["answer"] for c in e["call_log"]] == [4, 5] and all(c["usd"] == 0.001 for c in e["call_log"])
    assert len(calls) == 2 and all(c.valid for c in calls)


@pytest.mark.parametrize("k", range(2, 9))
def test_every_stage1_pick_runs_stage2_from_the_frame_before(k):
    segs = chain([100], ["open_start"], 300)
    new, log, calls, caller = run(segs, [k, 8])
    lo, hi = S1_100[k - 2], S1_100[k - 1]
    assert caller.seen()[1] == stage2_frames(lo, hi)
    assert log[0]["onset"] == hi == new[1]["start_frame"]  # answer 8 of stage 2 is the pick itself


def test_stage2_skipped_when_frames_are_adjacent():
    ep = make_episode(n=100, fps=4.0)  # 1 s is 4 frames: stage 1 frames are nearly adjacent
    segs = chain([50], ["close_start"], 100)
    assert stage1_frames(50, 100, 4.0) == [46, 47, 48, 49, 51, 52, 53, 54]
    new, log, calls, caller = run(segs, [2], ep=ep)
    assert len(calls) == 1 and log[0]["onset"] == 47 and log[0]["stage2_frames"] is None
    assert new[1]["start_frame"] == 47 and new[0]["crawl_calls"] == 1


def test_stage1_minus_one_keeps_the_coarse_frame():
    segs = chain([100], ["close_start"], 300)
    new, log, calls, caller = run(segs, [-1])
    e = log[0]
    assert e["flags"] == ["crawl_none"] and e["onset"] == 100 and e["pick"] is None and e["calls"] == 1
    assert new[1]["start_frame"] == 100 and new[0]["boundary_source"] == "coarse" and new[0]["crawl_calls"] == 1


@pytest.mark.parametrize("first", [0, 1])
def test_answer_0_or_1_shifts_earlier_and_asks_once_more(first):
    segs = chain([100], ["close_start"], 300)
    new, log, calls, caller = run(segs, [first, 3, 5])
    retry = [10, 19, 27, 36, 44, 53, 61, 70]
    s2 = stage2_frames(19, 27)
    assert caller.seen() == [S1_100, retry, s2]
    e = log[0]
    assert e["stage1_answer"] == first and e["retry_frames"] == retry and e["retry_answer"] == 3
    assert e["onset"] == s2[4] and e["flags"] == [] and e["calls"] == 3
    assert new[1]["start_frame"] == s2[4] and new[0]["boundary_source"] == "crawl"


def test_answer_9_then_pick_runs_stage2():
    segs = chain([100], ["open_start"], 300)
    new, log, calls, caller = run(segs, [9, 2, 4])
    retry = [130, 139, 147, 156, 164, 173, 181, 190]
    s2 = stage2_frames(130, 139)
    assert caller.seen() == [S1_100, retry, s2]
    assert log[0]["onset"] == s2[3] and new[1]["start_frame"] == s2[3]


def test_edge_twice_keeps_the_farthest_frame_and_skips_stage2():
    segs = chain([100], ["close_start"], 300)
    new, log, calls, caller = run(segs, [0, 0])
    e = log[0]
    assert e["flags"] == ["crawl_edge"] and e["onset"] == 10 and e["stage2_frames"] is None and e["calls"] == 2
    assert e["pick"] == 10  # a 0: the event began at or before the image's own frame
    assert new[1]["start_frame"] == 10 and new[0]["end_frame"] == 9 and new[0]["boundary_source"] == "crawl"

    segs = chain([100], ["open_start"], 300)
    new, log, calls, caller = run(segs, [9, 9])
    # a 9: not begun by frame 190, so the onset is the frame after it; the pick is the image the model saw
    assert log[0]["flags"] == ["crawl_edge"] and log[0]["onset"] == 191 and log[0]["pick"] == 190
    assert new[1]["start_frame"] == 191 and new[0]["end_frame"] == 190


def test_edge_then_opposite_or_none_keeps_the_stage1_edge_frame():
    cases = (([0, 9], 70, ["crawl_edge", "crawl_inconsistent"]), ([9, 0], 131, ["crawl_edge", "crawl_inconsistent"]),
             ([0, -1], 70, ["crawl_edge"]), ([9, -1], 131, ["crawl_edge"]))
    for answers, onset, flags in cases:
        segs = chain([100], ["close_start"], 300)
        new, log, calls, caller = run(segs, answers)
        assert log[0]["flags"] == flags and log[0]["onset"] == onset, answers
        assert len(calls) == 2 and log[0]["stage2_frames"] is None


def test_edge_at_the_clip_start_is_like_minus_one():
    segs = chain([5, 150], ["close_start", "other"], 300)
    new, log, calls, caller = run(segs, [0])
    # the window is already at frame 0 and frame 0 shows the event: it did not begin inside the clip
    assert len(calls) == 1 and log[0]["flags"] == ["crawl_edge", "crawl_none"] and log[0]["onset"] == 5
    assert log[0]["pick"] is None and new[1]["start_frame"] == 5 and new[0]["boundary_source"] == "coarse"


def test_edge_at_the_clip_end_is_like_minus_one():
    segs = chain([290], ["open_start"], 300)
    new, log, calls, caller = run(segs, [9])
    # not begun by the last frame: no onset inside the clip, so the coarse frame stays (no one-frame segment)
    assert len(calls) == 1 and log[0]["flags"] == ["crawl_edge", "crawl_none"] and log[0]["onset"] == 290
    assert new[1]["start_frame"] == 290 and new[0]["end_frame"] == 289 and new[0]["boundary_source"] == "coarse"
    # a retry that reaches the last frame and is still not begun ends the same way
    segs = chain([250], ["open_start"], 300)
    new, log, calls, caller = run(segs, [9, 9])
    assert log[0]["retry_frames"][-1] == 299 and log[0]["flags"] == ["crawl_edge", "crawl_none"]
    assert log[0]["onset"] == 250 and new[1]["start_frame"] == 250 and log[0]["calls"] == 2


def test_clamped_retry_pick_that_contradicts_stage1_keeps_the_stage1_edge():
    # stage 1 around 40 is 10..70; a 0 says frame 10 already shows the event, so the onset is at or before 10.
    # The shifted window is clamped at frame 0 and overlaps stage 1: a retry pick of frame 43 contradicts it.
    segs = chain([40], ["close_start"], 300)
    assert stage1_frames(40, 300, 30.0)[0] == 10
    new, log, calls, caller = run(segs, [0, 6])
    e = log[0]
    assert e["retry_frames"] == [0, 9, 17, 26, 34, 43, 51, 60]
    assert e["flags"] == ["crawl_edge", "crawl_inconsistent"] and e["onset"] == 10 and e["pick"] == 10
    assert e["stage2_frames"] is None and e["calls"] == 2 and new[1]["start_frame"] == 10
    # a retry pick at or before frame 10 agrees and runs stage 2
    new, log, calls, caller = run(segs, [0, 2, 5])
    assert log[0]["flags"] == [] and log[0]["stage2_frames"] == stage2_frames(0, 9)
    assert log[0]["onset"] == stage2_frames(0, 9)[4]


def test_clamped_retry_after_a_9_and_the_stage2_floor():
    # stage 1 around 255 is 225..285; the shifted window is clamped at the last frame: 239..299
    segs = chain([255], ["open_start"], 300)
    s1 = stage1_frames(255, 300, 30.0)
    assert s1[-1] == 285
    new, log, calls, caller = run(segs, [9, 3])  # retry image 3 is frame 256: stage 1 said not begun by 285
    e = log[0]
    assert e["retry_frames"] == [239, 248, 256, 265, 273, 282, 290, 299]
    assert e["flags"] == ["crawl_edge", "crawl_inconsistent"] and e["onset"] == 286 and e["pick"] == 285
    # retry image 7 (frame 290) agrees; stage 2 starts at 285 (stage 1's last image), not at 282
    new, log, calls, caller = run(segs, [9, 7, 3])
    e = log[0]
    assert e["flags"] == [] and e["stage2_frames"] == [285, 286, 287, 288, 289, 290]
    assert e["onset"] == 287 and new[1]["start_frame"] == 287


def test_call_cap_on_a_9_gives_the_frame_after_the_edge():
    segs = chain([100], ["open_start"], 300)
    new, log, calls, caller = run(segs, [9], max_calls_per_boundary=1)
    assert log[0]["flags"] == ["crawl_edge", "crawl_call_cap"] and log[0]["onset"] == 131 and log[0]["pick"] == 130


@pytest.mark.parametrize("second", [0, 1, 9])
def test_stage2_edge_answer_contradicts_stage1(second):
    segs = chain([100], ["close_start"], 300)
    new, log, calls, caller = run(segs, [5, second])
    e = log[0]
    assert e["flags"] == ["crawl_inconsistent"] and e["onset"] == 104 and e["stage2_answer"] == second
    assert new[1]["start_frame"] == 104 and new[0]["boundary_source"] == "crawl"


def test_stage2_minus_one_keeps_the_stage1_pick():
    segs = chain([100], ["close_start"], 300)
    new, log, calls, caller = run(segs, [5, -1])
    assert log[0]["flags"] == [] and log[0]["onset"] == 104 and log[0]["stage2_answer"] == -1


def test_failed_calls_and_bad_answers():
    segs = chain([100], ["close_start"], 300)
    new, log, calls, caller = run(segs, ["failed"])
    assert log[0]["flags"] == ["crawl_failed"] and log[0]["onset"] == 100 and log[0]["stage1_answer"] is None
    assert new[0]["boundary_source"] == "coarse" and log[0]["call_log"][0]["status"] == "failed"
    new, log, calls, caller = run(segs, [12])  # outside the answer set
    assert log[0]["flags"] == ["crawl_failed"] and log[0]["onset"] == 100 and log[0]["stage1_answer"] == 12
    new, log, calls, caller = run(segs, [5, "invalid"])
    assert log[0]["flags"] == ["crawl_failed"] and log[0]["onset"] == 104
    new, log, calls, caller = run(segs, [0, "refused"])
    assert log[0]["flags"] == ["crawl_edge", "crawl_failed"] and log[0]["onset"] == 70


def test_refined_onset_never_crosses_a_neighbour():
    segs = chain([100, 110], ["close_start", "open_start"], 300)
    # boundary 0 picks frame 130, past the next boundary at 110: the coarse frame stays
    new, log, calls, caller = run(segs, [8, 9, -1])
    assert log[0]["flags"] == ["crawl_inconsistent", "crawl_cross"] and log[0]["onset"] == 100
    assert log[0]["pick"] == 130 and new[0]["boundary_source"] == "coarse" and new[0]["crawl_calls"] == 2
    assert new[1]["start_frame"] == 100 and new[1]["end_frame"] == 109


def test_crossing_checks_the_refined_previous_boundary():
    segs = chain([100, 140], ["close_start", "open_start"], 300)
    # b0: stage 1 pick 8 -> 130, stage 2 (121..130) answer 8 -> 130: inside (0, 140), accepted
    # b1: stage 1 around 140 = [110, ...]; answer 2 -> 119, stage 2 answer 2 -> the frame after 110
    new, log, calls, caller = run(segs, [8, 8, 2, 2])
    assert log[0]["onset"] == 130 and new[1]["start_frame"] == 130
    s1b = stage1_frames(140, 300, 30.0)
    assert s1b[0] == 110 and log[1]["stage1_frames"] == s1b
    assert log[1]["pick"] < 130 and log[1]["flags"] == ["crawl_cross"] and log[1]["onset"] == 140
    starts = [s["start_frame"] for s in new]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_only_typed_boundaries_are_crawled_and_skips_are_logged():
    types = ["other", "close_start", "contact_start", "open_start", "contact_end"]
    segs = chain([50, 100, 150, 200, 250], types, 300)
    segs[3]["boundary_source"] = "signal"  # a snapped gripper boundary
    new, log, calls, caller = run(segs, [-1, -1], skip_types={"close_start"})
    by = {e["boundary_index"]: e for e in log}
    assert 0 not in by  # "other" is not a crawl boundary and is not logged
    assert by[1]["flags"] == ["skipped_type"] and by[1]["calls"] == 0
    assert by[3]["flags"] == ["skipped_signal"] and by[3]["calls"] == 0
    assert by[2]["flags"] == ["crawl_none"] and by[4]["flags"] == ["crawl_none"]
    assert len(calls) == 2


def test_gripper_skip_types_leave_contact_boundaries_crawled():
    types = ["close_start", "contact_start", "open_start"]
    segs = chain([80, 150, 220], types, 300)
    new, log, calls, caller = run(segs, [5, 8], skip_types=frozenset({"close_start", "open_start"}))
    crawled = [e for e in log if e["calls"]]
    assert [e["event_type"] for e in crawled] == ["contact_start"]
    assert new[0]["start_frame"] == 0 and new[1]["start_frame"] == 80 and new[3]["start_frame"] == 220


def test_boundary_cap_12_in_time_order():
    onsets = list(range(20, 300, 20))  # 14 boundaries
    segs = chain(onsets, ["close_start"] * len(onsets), 300)
    new, log, calls, caller = run(segs, [-1] * 12)
    assert len(log) == 14 and len(calls) == 12
    assert [e["flags"] for e in log[:12]] == [["crawl_none"]] * 12
    assert [e["flags"] for e in log[12:]] == [["skipped_cap"], ["skipped_cap"]]
    assert [e["coarse_frame"] for e in log[12:]] == [260, 280]
    new, log, calls, caller = run(segs, [-1] * 3, max_boundaries=3)
    assert sum(1 for e in log if e["flags"] == ["skipped_cap"]) == 11


def test_call_cap_per_boundary():
    segs = chain([100], ["close_start"], 300)
    new, log, calls, caller = run(segs, [4], max_calls_per_boundary=1)
    assert log[0]["flags"] == ["crawl_call_cap"] and log[0]["onset"] == 96 and log[0]["calls"] == 1
    new, log, calls, caller = run(segs, [0], max_calls_per_boundary=1)
    assert log[0]["flags"] == ["crawl_edge", "crawl_call_cap"] and log[0]["onset"] == 70
    new, log, calls, caller = run(segs, [0, 4], max_calls_per_boundary=2)
    assert log[0]["flags"] == ["crawl_call_cap"] and log[0]["onset"] == 36
    # never more than 3 calls per boundary, whatever the answers
    new, log, calls, caller = run(segs, [9, 3, 9])
    assert log[0]["calls"] == 3


def test_stopped_calls_skip_the_rest():
    segs = chain([100, 200], ["close_start", "open_start"], 300)
    new, log, calls, caller = run(segs, ["stopped"])
    assert log[0]["flags"] == ["crawl_failed"] and log[1]["flags"] == ["skipped_stopped"] and len(calls) == 1


def test_contact_question_names_the_object():
    segs = [seg(0, 99, "contact_start", phase_text="move the pencil to the paper", target="pencil",
                destination="the paper"),
            seg(100, 199, "contact_end", phase_text="draw a line", target="pencil", destination="paper"),
            seg(200, 299, "other", phase_text="lift the pencil", target="pencil")]
    new, log, calls, caller = run(segs, [-1, -1])
    q0, q1 = (r.parts[0].text for r in caller.requests)
    assert "the hand or the tool touches the paper?" in q0
    assert "the hand or the tool no longer touches the paper?" in q1
    assert '"move the pencil to the paper" ends and the phase "draw a line"' in q0
    assert log[0]["object"] == "paper" and log[1]["object"] == "paper"


def test_inventory_ids_become_names():
    segs = [seg(0, 99, "contact_start", target="o1"), seg(100, 299, "other", target="o1")]
    new, log, calls, caller = run(segs, [-1], objects=[{"object_id": "o1", "name": "red block"}])
    assert "touches the red block?" in caller.requests[0].parts[0].text


def test_crawl_is_deterministic_and_json():
    segs = chain([60, 120, 200], ["close_start", "contact_start", "open_start"], 300)
    answers = [4, 5, 0, 3, 5, 9, 9]

    def once():
        new, log, calls, caller = run(copy.deepcopy(segs), list(answers))
        texts = [[p.text if isinstance(p, TextPart) else p.jpeg.hex()[:64] for p in r.parts] for r in caller.requests]
        return json.dumps({"segments": new, "log": log, "texts": texts}, sort_keys=True).encode("utf-8")

    assert once() == once()


def test_segments_stay_contiguous_after_any_crawl():
    rng = np.random.default_rng(7)
    for _ in range(30):
        onsets = sorted(rng.choice(np.arange(5, 295), size=5, replace=False).tolist())
        types = rng.choice(list(C.CRAWL_EVENTS) + ["other"], size=5).tolist()
        answers = rng.choice([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=60).tolist()
        caller = Scripted(answers)
        new, log, calls = crawl_boundaries(make_episode(), chain(onsets, types, 300), caller, camera=CAM,
                                           context={}, reasoning=None)
        assert new[0]["start_frame"] == 0 and new[-1]["end_frame"] == 299
        for a, b in zip(new, new[1:], strict=False):
            assert b["start_frame"] == a["end_frame"] + 1 and a["start_frame"] <= a["end_frame"]
        for e in log:
            assert e["calls"] <= 3
            seen = [f for c in e["call_log"] for f in c["frames"]]
            if e["pick"] is not None:
                assert e["pick"] in seen  # a crawl pick lies inside a window the model saw


def test_prompt_hash_and_style():
    from pathlib import Path

    import robolabel.prompts.v8 as v8

    assert len(prompt_sha256("crawl")) == 64
    text = (Path(v8.__file__).parent / "crawl.txt").read_text(encoding="utf-8")
    src = Path(C.__file__).read_text(encoding="utf-8")
    for body in (text, src):
        assert chr(0x2014) not in body
        for word in ("cave" + "at", "side" + "car", "smoke" + " test"):
            assert word not in body.lower()
