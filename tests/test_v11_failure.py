"""The failure convention of SPEC_V1_1 section 4: derive_attempt_outcome, S4 and G2 under v1.1, the gold v2
attempt_outcome field and its warnings, and the baselines' failure_convention keyword. No network."""

from __future__ import annotations

import copy
import json

import pytest

from robolabel.baselines import legacy_view, sig_only_segments, sig_only_view, uniform5_view
from robolabel.eval import derive_attempt_outcome
from robolabel.eval.failure import (
    failed_spans_by_attempt,
    failure_convention_warnings,
    failure_marked,
    inside_share,
    resolve_convention,
)
from robolabel.eval.goal import g2_episode
from robolabel.eval.gold_v2 import validate_gold, warnings_for_episode, with_attempt_outcome
from robolabel.eval.score import score_view
from robolabel.eval.semantic import s4_episode
from robolabel.eval.temporal import failed_spans_from_segments
from robolabel.schema_v7 import fill_attempt_outcome

CAM = "observation.images.front"


# --------------------------------------------------------------------------- helpers
def vseg(start, end, phase, attempt, outcome="success", failure_type="none", attempt_outcome=None, mistake=None,
         target="pink brick"):
    s = {"start": start, "end": end, "phase_class": phase, "phase_text": phase, "target_name": target,
         "destination_name": "none", "attempt_idx": attempt, "outcome": outcome, "failure_type": failure_type,
         "mistake": outcome == "failed" if mistake is None else mistake, "boundary_source": "crawl"}
    if attempt_outcome is not None:
        s["attempt_outcome"] = attempt_outcome
    return s


def gseg(idx, start, end, phase, attempt, outcome="success", failure_type=None, last=False, **extra):
    return {"segment_idx": idx, "start_frame": start, "end_frame": end, "phase_class": phase, "phase_text": phase,
            "target": "o1" if phase != "retract" else None, "destination": None, "attempt_idx": attempt,
            "outcome": outcome, "failure_type": failure_type, "end_boundary_quality": None if last else "sharp",
            **extra}


def gold_missed_grasp(span=(0, 45)):
    """A missed grasp (approach, grasp failed, retract), then a second attempt that succeeds. The failed
    attempt's span is the whole attempt, from its approach to its retract (SPEC_V1_1 4)."""
    return {
        "episode_key": "F3/544", "episode_index": 544, "split": "dev", "pass": 1, "num_frames": 100,
        "cameras": [CAM], "task_string": "put the pink brick in the box",
        "objects": [{"object_id": "o1", "name": "pink brick", "aliases": ["brick"], "category": "block",
                     "first_frame_point": {"camera": CAM, "xy": [0.4, 0.6]}},
                    {"object_id": "o2", "name": "box", "aliases": [], "category": "container",
                     "first_frame_point": {"camera": CAM, "xy": [0.7, 0.3]}}],
        "primary_target": "o1", "primary_destination": "o2",
        "segments": [
            gseg(0, 0, 29, "approach", 1),
            gseg(1, 30, 39, "grasp", 1, "failed", "missed_grasp"),
            gseg(2, 40, 45, "retract", 1),
            gseg(3, 46, 59, "approach", 2),
            gseg(4, 60, 69, "grasp", 2),
            gseg(5, 70, 84, "transport", 2),
            gseg(6, 85, 99, "release", 2, last=True),
        ],
        "failed_attempts": [{"span": list(span), "failure_type": "missed_grasp", "object": "o1", "evident_frame": 38,
                             "evident_camera": CAM, "recovery_start": 46}],
        "goal": {"objective_text": "the pink brick is inside the box", "requirements": [
            greq("r1", "robot_end_state", "holding", False), greq("r2", "robot_end_state", "gripper_open", True),
            greq("r3", "robot_end_state", "withdrawn", True)]},
        "episode_outcome": "success", "hard_tags": ["failed_attempt"],
    }


def greq(rid, kind, predicate, value):
    return {"req_id": rid, "kind": kind, "object": None, "predicate": predicate, "ref_object": None, "value": value,
            "status": "required", "unsure_kind": None, "basis": "physical_necessity", "achieved": True,
            "deciding_frame": 99, "deciding_camera": CAM, "visibility": {CAM: "visible"}, "reason": ""}


def family(ep):
    return {"schema_version": "robolabel/gold/v2", "family": "F3", "guide_version": "gold_guide_v1.0 (SPEC_V1_1 4)",
            "episodes": [ep]}


def view_v11(**changes):
    """A v1.1 view of ``gold_missed_grasp``: per-phase outcomes, attempt_outcome on every phase, mistake only
    on the failed grasp, and attempt records that keep the whole span."""
    segments = [
        vseg(0, 29, "approach", 1, attempt_outcome="failed"),
        vseg(30, 39, "grasp", 1, "failed", "missed_grasp", attempt_outcome="failed"),
        vseg(40, 45, "retract", 1, attempt_outcome="failed", target="none"),
        vseg(46, 59, "approach", 2, attempt_outcome="success"),
        vseg(60, 69, "grasp", 2, attempt_outcome="success"),
        vseg(70, 84, "transport", 2, attempt_outcome="success"),
        vseg(85, 99, "release", 2, attempt_outcome="success"),
    ]
    view = {"arm": "v@test", "episode_key": "F3/544", "family": "F3", "fps": 20.0, "num_frames": 100,
            "cameras": [CAM], "camera_sizes": {CAM: [640, 480]}, "objects": [], "segments": segments,
            "coarse": [{"start": 0, "end": 39, "text": "pick up the pink brick", "mistake": True}],
            "attempts": [{"start": 0, "end": 45, "outcome": "failed", "failure_type": "missed_grasp", "source": "vlm"},
                         {"start": 46, "end": 99, "outcome": "success", "failure_type": "none", "source": "vlm"}],
            "goal": None, "episode_outcome": "success", "no_output": False}
    view.update(changes)
    return view


def counts(result, key):
    c = result["counts"][key]
    return c["numerator"], c["denominator"], c["pending"]


# --------------------------------------------------------------------------- derive_attempt_outcome
def test_derive_fills_absent_values_from_the_v7_rule_and_keeps_present_ones():
    segs = [
        {"start": 0, "end": 9, "attempt_idx": 1, "outcome": "success"},
        {"start": 10, "end": 19, "attempt_idx": 1, "outcome": "failed"},
        {"start": 20, "end": 29, "attempt_idx": 2, "outcome": "aborted"},
        {"start": 30, "end": 39, "attempt_idx": 3, "outcome": "success"},
        {"start": 40, "end": 49, "attempt_idx": 3, "outcome": "success", "attempt_outcome": "failed"},
        {"start": 50, "end": 59, "attempt_idx": 4, "outcome": "success", "attempt_outcome": float("nan")},
    ]
    before = copy.deepcopy(segs)
    out = derive_attempt_outcome(segs)
    assert [s["attempt_outcome"] for s in out] == ["failed", "failed", "aborted", "success", "failed", "success"]
    assert segs[:5] == before[:5] and "attempt_outcome" not in segs[0]  # the input is not modified
    assert derive_attempt_outcome([]) == [] and derive_attempt_outcome(["x", None]) == []


def test_derive_agrees_with_schema_v7_fill_attempt_outcome_on_numbered_segments():
    segs = [{"attempt_idx": i // 3 + 1, "outcome": o, "start": i, "end": i}
            for i, o in enumerate(["success", "failed", "success", "success", "success", "aborted",
                                   "success", "success", "success", "failed", "failed", "success"])]
    segs[4]["attempt_outcome"] = "failed"  # present values are kept by both
    assert [s["attempt_outcome"] for s in derive_attempt_outcome(segs)] == \
        [s["attempt_outcome"] for s in fill_attempt_outcome(segs)]


def test_derive_reads_mistake_and_unnumbered_segments_as_their_own_attempts():
    segs = [{"start": 0, "end": 9, "outcome": "success", "mistake": True},
            {"start": 10, "end": 19, "outcome": "success"},
            {"start": 20, "end": 29, "attempt_idx": "2", "outcome": "success", "mistake": "true"},
            {"start": 30, "end": 39, "attempt_idx": 2, "outcome": "success"}]
    assert [s["attempt_outcome"] for s in derive_attempt_outcome(segs)] == ["failed", "success", "failed", "failed"]
    # schema_v7.fill_attempt_outcome is the same rule (it calls derive_attempt_outcome)
    assert fill_attempt_outcome(segs) == derive_attempt_outcome(segs)
    assert [s["attempt_outcome"] for s in fill_attempt_outcome(segs[:2])] == ["failed", "success"]


def test_derive_reads_pandas_missing_values_as_absent():
    import pandas as pd

    segs = [{"start": 0, "end": 9, "attempt_idx": pd.NA, "outcome": "failed", "attempt_outcome": pd.NA,
             "mistake": pd.NA},
            {"start": 10, "end": 19, "attempt_idx": 2, "outcome": "success", "attempt_outcome": None}]
    assert [s["attempt_outcome"] for s in derive_attempt_outcome(segs)] == ["failed", "success"]
    assert resolve_convention(segs) == "v7"


def test_convention_and_marks():
    assert resolve_convention([vseg(0, 9, "grasp", 1)]) == "v7"
    assert resolve_convention([vseg(0, 9, "grasp", 1), vseg(10, 19, "grasp", 1, attempt_outcome="success")]) == "v11"
    assert resolve_convention([vseg(0, 9, "grasp", 1)], "v11") == "v11"
    with pytest.raises(ValueError):
        resolve_convention([], "v8")
    assert failure_marked({"attempt_outcome": "failed", "outcome": "success", "mistake": False})
    assert failure_marked({"attempt_outcome": "success", "outcome": "aborted"})
    assert not failure_marked({"attempt_outcome": "success", "outcome": "success", "mistake": False})
    assert inside_share({"start": 30, "end": 39}, (0, 45)) == 1.0
    assert inside_share({"start": 40, "end": 59}, (0, 45)) == pytest.approx(0.3)
    assert inside_share({"start": 5, "end": None}, (0, 45)) == 0.0


# --------------------------------------------------------------------------- S4
def test_s4_v11_uses_attempt_records_with_their_whole_span():
    view, gold = view_v11(), gold_missed_grasp()
    r = s4_episode(view["segments"], gold["failed_attempts"], pred_attempts=view["attempts"])
    assert (r["convention"], r["pred_span_source"]) == ("v11", "attempts")
    assert r["pred_spans"] == [[0, 45]] and r["pairs"][0]["iou"] == 1.0
    assert (r["S4-F1"]["numerator"], r["S4-F1"]["denominator"]) == (2, 2)
    assert r["pairs"][0]["pred_type"] == "missed_grasp" and r["S4-type"]["numerator"] == 1
    # without records: consecutive phases with the same attempt_idx and attempt_outcome failed
    r = s4_episode(view["segments"], gold["failed_attempts"])
    assert (r["pred_span_source"], r["pred_spans"]) == ("segments", [[0, 45]])
    assert r["pairs"][0]["pred_type"] == "missed_grasp"
    # an empty or span-less attempt list counts as no records
    assert s4_episode(view["segments"], [], pred_attempts=[{"outcome": "failed"}])["pred_span_source"] == "segments"


def test_s4_v11_splits_failed_attempts_by_attempt_idx():
    segs = [vseg(0, 9, "approach", 1, attempt_outcome="failed"),
            vseg(10, 19, "grasp", 1, "failed", "missed_grasp", attempt_outcome="failed"),
            vseg(20, 29, "approach", 2, attempt_outcome="failed"),
            vseg(30, 39, "grasp", 2, "failed", "slip", attempt_outcome="failed"),
            vseg(40, 49, "grasp", 3, attempt_outcome="success")]
    assert failed_spans_by_attempt(segs) == [[0, 19], [20, 39]]
    assert failed_spans_from_segments(segs) == [[10, 19], [30, 39]]  # the v7 rule reads own outcomes only


def test_s4_old_outputs_score_exactly_as_before():
    pred = [vseg(0, 29, "approach", 1), vseg(30, 40, "grasp", 1, "failed", "missed_grasp"),
            vseg(41, 44, "approach", 1, mistake=True), vseg(45, 79, "transport", 1),
            vseg(80, 90, "grasp", 1, "failed", "drop"), vseg(91, 99, "retract", 1)]
    gold = [{"span": [30, 45], "failure_type": "missed_grasp"}, {"span": [60, 70], "failure_type": "slip"}]
    records = [{"start": 0, "end": 99, "outcome": "success"}]
    r = s4_episode(pred, gold, pred_attempts=records)
    assert (r["convention"], r["pred_spans"]) == ("v7", [[30, 44], [80, 90]])  # attempt records are not read
    forced = s4_episode(pred, gold, convention="v11")
    assert forced["pred_spans"] == [[0, 99]]  # the v7 rule derives failed for the whole attempt 1


# --------------------------------------------------------------------------- G2
def test_g2_v11_attempt_outcome_marks_every_phase_of_the_failed_attempt():
    view, gold = view_v11(), gold_missed_grasp()
    r = g2_episode(view["segments"], [], gold["failed_attempts"], [], episode_key="F3/544")
    assert r["convention"] == "v11" and r["part_i_hits"] == [] and r["counts"]["G2-i"]["numerator"] == 0
    # the approach inside the failed attempt, left as part of a successful attempt, is copied
    segs = copy.deepcopy(view["segments"])
    segs[0]["attempt_outcome"] = "success"
    r = g2_episode(segs, [], gold["failed_attempts"], [], episode_key="F3/544")
    assert [(h["index"], h["inside_share"]) for h in r["part_i_hits"]] == [(0, 1.0)]
    assert r["counts"]["G2-i"]["numerator"] == 1 and r["ok"] is False


def test_g2_v11_reads_segments_inside_the_attempt_not_only_by_iou():
    gold = gold_missed_grasp()
    segs = [vseg(0, 29, "approach", 1, attempt_outcome="success"), vseg(30, 39, "grasp", 1, attempt_outcome="success"),
            vseg(40, 99, "transport", 1, attempt_outcome="success")]
    r = g2_episode(segs, [], gold["failed_attempts"], [])
    # the grasp's IoU with [0, 45] is 10 / 46, but all of it lies inside the failed attempt
    assert [h["index"] for h in r["part_i_hits"]] == [0, 1]
    assert r["part_i_hits"][1]["iou"] == pytest.approx(10 / 46, abs=1e-6)
    # the transport has 6 of its 60 frames inside: not inside
    assert all(h["index"] != 2 for h in r["part_i_hits"])


def test_g2_old_outputs_score_exactly_as_before():
    gold = [{"span": [30, 45]}]
    segs = [vseg(0, 29, "approach", 1), vseg(30, 45, "grasp", 1)]
    r = g2_episode(segs, [], gold, [])
    assert r["convention"] == "v7"
    assert r["part_i_hits"] == [{"source": "segment", "index": 1, "gold_span": [30, 45], "iou": 1.0}]


# --------------------------------------------------------------------------- score_view
def test_score_view_v11_view_against_whole_attempt_gold():
    gold = gold_missed_grasp()
    assert validate_gold(family(gold)) == []
    r = score_view(view_v11(), gold)
    assert counts(r, "S4-F1") == (2, 2, 0) and counts(r, "S4-type") == (1, 1, 0)
    assert counts(r, "G2-i") == (0, 1, 0)
    assert r["details"]["S4"]["convention"] == "v11" and r["details"]["S4"]["pred_span_source"] == "attempts"
    assert r["details"]["goal"]["G2"]["convention"] == "v11"
    # forcing the v7 rule reads the failed grasp only: [30, 39] against [0, 45] is below IoU 0.3
    old = score_view(view_v11(), gold, failure_convention="v7")
    assert old["details"]["S4"]["pred_spans"] == [[30, 39]] and counts(old, "S4-F1") == (0, 2, 0)
    # a missing output has no failed attempt under either rule
    miss = score_view({"arm": "v@test", "episode_key": "F3/544", "no_output": True}, gold)
    assert counts(miss, "S4-P") == (0, 0, 0) and miss["details"]["S4"]["convention"] == "v7"


# --------------------------------------------------------------------------- gold v2
def test_gold_schema_takes_an_optional_attempt_outcome():
    ep = gold_missed_grasp()
    for s in ep["segments"]:
        s["attempt_outcome"] = "failed" if s["attempt_idx"] == 1 else "success"
    assert validate_gold(family(ep)) == [] and warnings_for_episode(ep) == []
    ep["segments"][0]["attempt_outcome"] = "partial"
    errors = validate_gold(family(ep))
    assert len(errors) == 1 and errors[0].startswith("schema: $.episodes[0].segments[0].attempt_outcome:")


def test_with_attempt_outcome_derives_for_readers():
    ep = gold_missed_grasp()
    derived = with_attempt_outcome(ep)
    assert [s["attempt_outcome"] for s in derived["segments"]] == ["failed"] * 3 + ["success"] * 4
    assert "attempt_outcome" not in ep["segments"][0]
    doc = with_attempt_outcome(family(ep))
    assert doc["episodes"][0]["segments"][1]["attempt_outcome"] == "failed"
    export = with_attempt_outcome({"schema": "robolabel/gold-export/v1", "families": {"F3": family(ep)}})
    assert export["families"]["F3"]["episodes"][0]["segments"][6]["attempt_outcome"] == "success"
    assert validate_gold(doc) == []


def test_gold_warnings_for_the_per_phase_rule():
    ep = gold_missed_grasp()
    assert warnings_for_episode(ep) == []
    old = copy.deepcopy(ep)  # the v7 habit: every phase of the failed attempt marked failed
    for s in old["segments"][:3]:
        s.update(outcome="failed", failure_type="missed_grasp")
    assert warnings_for_episode(old)[0].startswith("attempt 1: 3 phases have outcome failed (segments 0, 1, 2)")
    bad = copy.deepcopy(ep)
    bad["segments"][4]["failure_type"] = "slip"
    bad["segments"][0]["attempt_outcome"] = "failed"
    bad["segments"][1]["attempt_outcome"] = "success"
    bad["segments"][3]["attempt_outcome"] = "failed"
    assert failure_convention_warnings(bad["segments"]) == [
        "segment 4: failure_type slip on a phase whose own outcome is success",
        "attempt 1: its phases disagree on attempt_outcome (failed, success); copy one value onto every phase "
        "of the attempt",
        "attempt 2: attempt_outcome failed but its phases say success",
    ]


def test_gold_warns_when_a_failed_attempt_span_is_not_the_whole_attempt():
    part = gold_missed_grasp(span=(30, 38))  # the v1.0 span: failing phase to evident frame
    assert validate_gold(family(part)) == []
    assert warnings_for_episode(part) == [
        "failed attempt 0: span [30, 38] is not the whole attempt 1 (frames 0 to 45, from its first phase to its last)"]
    across = gold_missed_grasp(span=(30, 50))
    assert warnings_for_episode(across) == ["failed attempt 0: span [30, 50] overlaps the phases of attempts 1, 2; "
                                            "a span covers one attempt"]


# --------------------------------------------------------------------------- baselines
L1 = {
    "num_frames": 120,
    "attempts": [
        {"attempt_idx": 1, "outcome": "empty", "failure_type": "missed_grasp", "closing_onset": 20, "closing_offset": 25,
         "opening_onset": 30, "opening_offset": 34, "event_frame": 27},
        {"attempt_idx": 2, "outcome": "released", "failure_type": "none", "closing_onset": 50, "closing_offset": 55,
         "opening_onset": 80, "opening_offset": 85, "event_frame": 55},
    ],
    "candidates": [{"attempt_idx": 2, "transition": "grasp->transport", "frame": 58},
                   {"attempt_idx": 2, "transition": "release->retract", "frame": 88}],
    "end_state": [{"predicate": "withdrawn", "value": True}],
}
META = {"episode_key": "F3/1821", "family": "F3", "fps": 20.0, "num_frames": 120, "cameras": [CAM]}

# the v7 output of sig_only on L1, written out so that any change to the default shows
SIG_ONLY_V7 = [
    (0, 19, "approach", 1, "failed", "missed_grasp", True),
    (20, 29, "grasp", 1, "failed", "missed_grasp", True),
    (30, 49, "approach", 2, "success", "none", False),
    (50, 58, "grasp", 2, "success", "none", False),
    (59, 79, "transport", 2, "success", "none", False),
    (80, 88, "release", 2, "success", "none", False),
    (89, 119, "retract", 2, "success", "none", False),
]


def _rows(segs):
    return [(s["start"], s["end"], s["phase_class"], s["attempt_idx"], s["outcome"], s["failure_type"], s["mistake"])
            for s in segs]


def test_sig_only_default_is_the_v7_output():
    segs = sig_only_segments(copy.deepcopy(L1))
    assert _rows(segs) == SIG_ONLY_V7 and all("attempt_outcome" not in s for s in segs)
    view = sig_only_view(copy.deepcopy(L1), META)
    assert json.dumps(view, sort_keys=True) == json.dumps(sig_only_view(copy.deepcopy(L1), META, failure_convention="v7"),
                                                          sort_keys=True)
    assert view["attempts"][0] == {"start": 0, "end": 29, "outcome": "failed", "failure_type": "missed_grasp",
                                   "source": "signal"}
    with pytest.raises(ValueError):
        sig_only_segments(L1, failure_convention="v8")


def test_sig_only_v11_marks_the_failed_phase_only():
    segs = sig_only_segments(copy.deepcopy(L1), failure_convention="v11")
    rows = _rows(segs)
    assert rows[0] == (0, 19, "approach", 1, "success", "none", False)  # the approach reached the object
    assert rows[1:] == SIG_ONLY_V7[1:]
    assert [s["attempt_outcome"] for s in segs] == ["failed", "failed"] + ["success"] * 5
    view = sig_only_view(copy.deepcopy(L1), META, failure_convention="v11")
    assert view["attempts"][0] == {"start": 0, "end": 29, "outcome": "failed", "failure_type": "missed_grasp",
                                   "source": "signal", "evident_frame": 27}
    assert "evident_frame" not in view["attempts"][1]
    old = sig_only_view(copy.deepcopy(L1), META)
    assert view["coarse"] == old["coarse"] and view["goal"] == old["goal"]
    # both conventions give S4 the same predicted failed attempt: the whole attempt 1
    gold = [{"span": [0, 29], "failure_type": "missed_grasp"}]
    new_s4 = s4_episode(view["segments"], gold, pred_attempts=view["attempts"])
    old_s4 = s4_episode(old["segments"], gold, pred_attempts=old["attempts"])
    assert new_s4["pred_spans"] == old_s4["pred_spans"] == [[0, 29]]
    assert (new_s4["convention"], old_s4["convention"]) == ("v11", "v7")


def test_uniform5_and_legacy_views_take_the_keyword():
    meta = dict(META, num_frames=100)
    assert all("attempt_outcome" not in s for s in uniform5_view(meta)["segments"])
    assert [s["attempt_outcome"] for s in uniform5_view(meta, failure_convention="v11")["segments"]] == ["success"] * 5

    class Seg:
        def __init__(self, start, end, text):
            self.start_frame, self.end_frame, self.subtask_text, self.phase, self.target = start, end, text, None, None

    subtasks = [Seg(0, 49, "pick up the brick"), Seg(50, 99, "put it in the box")]
    old = legacy_view("b2b@test", meta, subtasks, None, cost=0.0, calls=1, wall_s=1.0, valid=True, repairs=[])
    new = legacy_view("b2b@test", meta, subtasks, None, cost=0.0, calls=1, wall_s=1.0, valid=True, repairs=[],
                      failure_convention="v11")
    assert all("attempt_outcome" not in s for s in old["segments"])
    assert [s["attempt_outcome"] for s in new["segments"]] == ["success", "success"]
    assert [{k: v for k, v in s.items() if k != "attempt_outcome"} for s in new["segments"]] == old["segments"]
