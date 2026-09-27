"""Tests for object resolution, S1, S2, S4, G1 to G7, the view scorer and the bootstrap statistics.

Appendix F items 6, 7 and 8 of MEASUREMENT_SPEC are checked with the hand-computed numbers; every
other expected number is worked out in the comments next to it.
"""

from __future__ import annotations

import copy
import json

import pytest

from robolabel.eval import semantic, stats
from robolabel.eval.goal import (
    PENDING_REF,
    could_match,
    g6_episode,
    gold_requirements,
    items_equal,
    match_requirements,
    narrates_failure,
)
from robolabel.eval.gold_v2 import validate_gold
from robolabel.eval.heldout import HeldoutRefused
from robolabel.eval.score import aggregate, score_legacy_boundaries, score_view, split_metric_key, to_records
from robolabel.eval.semantic import ObjectResolver, resolve_object, s1_episode, s2_episode, s4_episode

UP = "observation.images.up"
SIDE = "observation.images.side"
SIZES = {UP: [640, 480], SIDE: [640, 480]}
IMG = "observation.images.image"
IMG2 = "observation.images.image2"
FRONT = "observation.images.front"
TOP = "observation.images.top"


# --------------------------------------------------------------------------- #
# Fixtures: gold v2 episodes (validated) and view records
# --------------------------------------------------------------------------- #
def _gseg(idx, start, end, phase, target, destination=None, outcome="success", failure_type=None,
          attempt=1, last=False):
    return {"segment_idx": idx, "start_frame": start, "end_frame": end, "phase_class": phase,
            "phase_text": phase, "target": target, "destination": destination, "attempt_idx": attempt,
            "outcome": outcome, "failure_type": failure_type,
            "end_boundary_quality": None if last else "sharp"}


def _greq(rid, kind, obj, predicate, ref, value, status, unsure_kind=None, achieved=True, cams=(UP, SIDE),
          frame=90):
    return {"req_id": rid, "kind": kind, "object": obj, "predicate": predicate, "ref_object": ref,
            "value": value, "status": status, "unsure_kind": unsure_kind, "basis": "task_string",
            "achieved": achieved, "deciding_frame": frame, "deciding_camera": cams[0],
            "visibility": {c: "visible" for c in cams}, "reason": ""}


def _brick_objects(cam=UP):
    return [
        {"object_id": "o1", "name": "pink lego brick", "aliases": ["pink brick", "brick"], "category": "block",
         "first_frame_point": {"camera": cam, "xy": [0.4, 0.6]}},
        {"object_id": "o2", "name": "transparent box", "aliases": ["box", "clear box"], "category": "container",
         "first_frame_point": {"camera": cam, "xy": [0.7, 0.35]}},
    ]


def _gold_f1():
    """F1/0: approach, grasp, transport, release, retract over 100 frames."""
    return {
        "episode_key": "F1/0", "episode_index": 0, "split": "dev", "pass": 1, "annotator_id": "rater_a",
        "blind": True, "tool": "robolabel-gold-ui 0.1", "active_seconds": 300, "num_frames": 100,
        "cameras": [UP, SIDE], "task_string": "pink lego brick into the transparent box",
        "objects": _brick_objects(),
        "primary_target": "o1", "primary_destination": "o2",
        "segments": [
            _gseg(0, 0, 29, "approach", "o1"),
            _gseg(1, 30, 39, "grasp", "o1"),
            _gseg(2, 40, 69, "transport", "o1", "o2"),
            _gseg(3, 70, 79, "release", "o1", "o2"),
            _gseg(4, 80, 99, "retract", None, last=True),
        ],
        "coarse_subtasks": [
            {"coarse_idx": 0, "start_frame": 0, "end_frame": 39, "text": "pick up the pink lego brick",
             "target": "o1", "destination": None, "mistake": False},
            {"coarse_idx": 1, "start_frame": 40, "end_frame": 79, "text": "put the brick in the box",
             "target": "o1", "destination": "o2", "mistake": False},
            {"coarse_idx": 2, "start_frame": 80, "end_frame": 99, "text": "move the arm away",
             "target": None, "destination": None, "mistake": False},
        ],
        "failed_attempts": [],
        "goal": {"objective_text": "the pink brick is inside the transparent box", "requirements": [
            _greq("r1", "object_end_state", "o1", "inside", "o2", True, "required"),
            _greq("r2", "robot_end_state", None, "holding", None, False, "required"),
            _greq("r3", "robot_end_state", None, "withdrawn", None, True, "unsure", "intent"),
        ]},
        "episode_outcome": "success", "quality": 5, "hard_tags": [], "notes": "",
    }


def _gold_f3():
    """F3/544: a missed grasp in [30, 45], then a second attempt that succeeds."""
    return {
        "episode_key": "F3/544", "episode_index": 544, "split": "dev", "pass": 1, "annotator_id": "rater_a",
        "blind": True, "tool": "robolabel-gold-ui 0.1", "active_seconds": 400, "num_frames": 100,
        "cameras": [FRONT, TOP], "task_string": "put the pink brick in the box",
        "objects": _brick_objects(FRONT),
        "primary_target": "o1", "primary_destination": "o2",
        "segments": [
            _gseg(0, 0, 29, "approach", "o1"),
            _gseg(1, 30, 45, "grasp", "o1", outcome="failed", failure_type="missed_grasp"),
            _gseg(2, 46, 59, "approach", "o1", attempt=2),
            _gseg(3, 60, 69, "grasp", "o1", attempt=2),
            _gseg(4, 70, 84, "transport", "o1", "o2", attempt=2),
            _gseg(5, 85, 92, "release", "o1", "o2", attempt=2),
            _gseg(6, 93, 99, "retract", None, attempt=2, last=True),
        ],
        "failed_attempts": [{"span": [30, 45], "failure_type": "missed_grasp", "object": "o1",
                             "evident_frame": 45, "evident_camera": FRONT, "recovery_start": 46}],
        "goal": {"objective_text": "the pink brick is inside the box", "requirements": [
            _greq("r1", "object_end_state", "o1", "inside", "o2", True, "required", cams=(FRONT, TOP)),
            _greq("r2", "robot_end_state", None, "holding", None, False, "required", cams=(FRONT, TOP)),
            _greq("r3", "robot_end_state", None, "withdrawn", None, True, "unsure", "intent", cams=(FRONT, TOP)),
        ]},
        "episode_outcome": "success", "quality": 3, "hard_tags": ["failed_attempt"], "notes": "",
    }


def _gold_f2():
    """F2/0: two identical black bowls; the left one goes on the plate."""
    return {
        "episode_key": "F2/0", "episode_index": 0, "split": "dev", "pass": 1, "annotator_id": "rater_a",
        "blind": True, "tool": "robolabel-gold-ui 0.1", "active_seconds": 200, "num_frames": 80,
        "cameras": [IMG, IMG2], "task_string": "put the left black bowl on the plate",
        "objects": [
            {"object_id": "o1", "name": "left black bowl", "aliases": ["black bowl", "bowl"], "category": "bowl",
             "first_frame_point": {"camera": IMG, "xy": [0.3, 0.5]}},
            {"object_id": "o2", "name": "right black bowl", "aliases": ["black bowl", "bowl"], "category": "bowl",
             "first_frame_point": {"camera": IMG, "xy": [0.7, 0.5]}},
            {"object_id": "o3", "name": "plate", "aliases": [], "category": "plate",
             "first_frame_point": {"camera": IMG, "xy": [0.5, 0.8]}},
        ],
        "primary_target": "o1", "primary_destination": "o3",
        "segments": [
            _gseg(0, 0, 19, "approach", "o1"),
            _gseg(1, 20, 29, "grasp", "o1"),
            _gseg(2, 30, 49, "transport", "o1", "o3"),
            _gseg(3, 50, 59, "release", "o1", "o3"),
            _gseg(4, 60, 79, "retract", None, last=True),
        ],
        "failed_attempts": [],
        "goal": {"objective_text": "the left black bowl is on the plate", "requirements": [
            _greq("r1", "object_end_state", "o1", "on_top_of", "o3", True, "required", cams=(IMG, IMG2), frame=79),
            _greq("r2", "robot_end_state", None, "holding", None, False, "required", cams=(IMG, IMG2), frame=79),
        ]},
        "episode_outcome": "success", "hard_tags": ["distractor"], "notes": "",
    }


def _vseg(start, end, phase, target="pink brick", destination="none", outcome="success", failure_type="none",
          mistake=False, text=None, attempt=1):
    return {"start": start, "end": end, "phase_class": phase, "phase_text": text or phase, "target_name": target,
            "destination_name": destination, "attempt_idx": attempt, "outcome": outcome,
            "failure_type": failure_type, "mistake": mistake, "boundary_source": "vlm"}


def _vreq(kind, predicate, obj, ref, value, status, unsure_kind="none", achieved=True, text=""):
    return {"text": text, "kind": kind, "predicate": predicate, "object_name": obj, "ref_name": ref,
            "value": value, "status": status, "unsure_kind": unsure_kind, "basis": "task_string",
            "achieved": achieved}


def _brick_view_objects(cam=UP):
    # "clear container" is no gold name: it resolves by its point (rule 1) at o2's gold point.
    return [
        {"object_id": "o1", "name": "pink brick", "category": "block",
         "points": [{"camera": cam, "x": 0.41, "y": 0.61}],
         "boxes": [{"camera": cam, "x0": 0.35, "y0": 0.55, "x1": 0.45, "y1": 0.65}]},
        {"object_id": "o2", "name": "clear container", "category": "container",
         "points": [{"camera": cam, "x": 0.7, "y": 0.35}], "boxes": []},
    ]


def _goal_reqs():
    return [
        _vreq("object_end_state", "inside", "pink brick", "clear container", True, "required"),
        _vreq("robot_end_state", "holding", "none", "none", False, "required"),
        _vreq("robot_end_state", "withdrawn", "none", "none", True, "unsure", "intent"),
    ]


def _view_f1():
    """A view that agrees with ``_gold_f1`` everywhere."""
    return {
        "arm": "v@test", "episode_key": "F1/0", "family": "F1", "fps": 30.0, "num_frames": 100,
        "cameras": [UP, SIDE], "camera_sizes": SIZES, "task": "pink lego brick into the transparent box",
        "objects": _brick_view_objects(),
        "segments": [
            _vseg(0, 29, "approach"),
            _vseg(30, 39, "grasp"),
            _vseg(40, 69, "transport", destination="clear container"),
            _vseg(70, 79, "release", destination="clear container"),
            _vseg(80, 99, "retract", target="none"),
        ],
        "coarse": [{"start": 0, "end": 39, "text": "pick up the pink brick", "mistake": False},
                   {"start": 40, "end": 79, "text": "put the pink brick in the clear container", "mistake": False},
                   {"start": 80, "end": 99, "text": "move the arm away", "mistake": False}],
        "attempts": [{"start": 0, "end": 99, "outcome": "success", "failure_type": "none", "source": "vlm"}],
        "goal": {"objective": "the pink brick is in the clear container", "primary_target_name": "pink brick",
                 "primary_destination_name": "clear container", "requirements": _goal_reqs()},
        "episode_outcome": "success", "checks": [], "risk": 0.0, "routed": False, "cost_usd": 0.01, "calls": 4,
        "wall_s": 10.0, "valid": True, "repairs": [], "no_output": False,
    }


def _view_f3():
    """A view of ``_gold_f3`` that marks the missed grasp as failed and keeps the coarse text clean."""
    view = _view_f1()
    view.update(arm="v@test", episode_key="F3/544", family="F3", fps=20.0, cameras=[FRONT, TOP],
                camera_sizes={FRONT: [640, 480], TOP: [640, 480]}, objects=_brick_view_objects(FRONT))
    view["segments"] = [
        _vseg(0, 29, "approach"),
        _vseg(30, 45, "grasp", outcome="failed", failure_type="missed_grasp", mistake=True),
        _vseg(46, 59, "approach", attempt=2),
        _vseg(60, 69, "grasp", attempt=2),
        _vseg(70, 84, "transport", destination="clear container", attempt=2),
        _vseg(85, 92, "release", destination="clear container", attempt=2),
        _vseg(93, 99, "retract", target="none", attempt=2),
    ]
    view["coarse"] = [{"start": 0, "end": 45, "text": "pick up the pink brick", "mistake": True},
                      {"start": 46, "end": 69, "text": "pick up the pink brick", "mistake": False},
                      {"start": 70, "end": 92, "text": "put the pink brick in the clear container", "mistake": False},
                      {"start": 93, "end": 99, "text": "move the arm away", "mistake": False}]
    return view


def _view_f2():
    """A view of ``_gold_f2`` with an empty inventory that names the target only "black bowl"."""
    return {
        "arm": "v@test", "episode_key": "F2/0", "family": "F2", "fps": 10.0, "num_frames": 80,
        "cameras": [IMG, IMG2], "camera_sizes": {IMG: [256, 256], IMG2: [256, 256]}, "task": "",
        "objects": [],
        "segments": [
            _vseg(0, 19, "approach", target="black bowl"),
            _vseg(20, 29, "grasp", target="black bowl"),
            _vseg(30, 49, "transport", target="black bowl", destination="plate"),
            _vseg(50, 59, "release", target="black bowl", destination="plate"),
            _vseg(60, 79, "retract", target="none"),
        ],
        "coarse": [],
        "attempts": [],
        "goal": {"objective": "", "primary_target_name": "black bowl", "primary_destination_name": "plate",
                 "requirements": [
                     _vreq("object_end_state", "on_top_of", "black bowl", "plate", True, "required"),
                     _vreq("robot_end_state", "holding", "none", "none", False, "required"),
                 ]},
        "episode_outcome": "success", "no_output": False, "valid": True, "repairs": [],
    }


def _family_doc(ep):
    fam = ep["episode_key"].split("/")[0]
    return {"schema_version": "robolabel/gold/v2", "family": fam, "guide_version": "pilot (gold_guide_v1.0 draft)",
            "episodes": [ep]}


def _c(result, key):
    c = result["counts"][key]
    return c["numerator"], c["denominator"], c["pending"]


def test_fixtures_are_valid_gold_v2():
    for ep in (_gold_f1(), _gold_f2(), _gold_f3()):
        assert validate_gold(_family_doc(ep)) == []


# --------------------------------------------------------------------------- #
# Object resolution (spec 4.0)
# --------------------------------------------------------------------------- #
def test_rule1_box_resolves():
    golds = _brick_objects()
    res = resolve_object("thing", [], [{"camera": UP, "x0": 0.35, "y0": 0.55, "x1": 0.45, "y1": 0.65}],
                         golds, SIZES)
    assert (res["object_id"], res["rule"], res["status"]) == ("o1", 1, "resolved")
    # a short camera name is the same camera
    res = resolve_object("thing", [], [{"camera": "up", "x0": 0.45, "y0": 0.65, "x1": 0.35, "y1": 0.55}],
                         golds, SIZES)
    assert (res["object_id"], res["rule"]) == ("o1", 1)


def test_rule1_point_uses_pixels_of_the_diagonal():
    golds = _brick_objects()
    # 640 x 480: diagonal 800 px, limit 40 px. 0.08 of the height is 38.4 px: a match.
    near = [{"camera": UP, "x": 0.4, "y": 0.68}]
    res = resolve_object("thing", near, [], golds, SIZES)
    assert (res["object_id"], res["rule"]) == ("o1", 1)
    # In the unit square the same offset is 0.08 > 0.05 * sqrt(2) = 0.0707: no rule-1 match, and
    # "thing" names no gold object, so it goes to the judge.
    res = resolve_object("thing", near, [], golds, {})
    assert (res["object_id"], res["rule"], res["status"]) == (None, 3, "pending")
    # 0.09 of the height is 43.2 px > 40: no match
    res = resolve_object("thing", [{"camera": UP, "x": 0.4, "y": 0.69}], [], golds, SIZES)
    assert res["status"] == "pending"
    # the right spot in another camera does not count
    res = resolve_object("thing", [{"camera": SIDE, "x": 0.4, "y": 0.6}], [], golds, SIZES)
    assert res["status"] == "pending"


def test_rule1_several_or_zero_matches_fall_through_to_rule2(monkeypatch):
    golds = _brick_objects()
    both = [{"camera": UP, "x0": 0.3, "y0": 0.3, "x1": 0.8, "y1": 0.7}]
    res = resolve_object("pink brick", [], both, golds, SIZES)
    assert (res["object_id"], res["rule"]) == ("o1", 2)
    none = [{"camera": UP, "x0": 0.0, "y0": 0.0, "x1": 0.1, "y1": 0.1}]
    res = resolve_object("the box", [], none, golds, SIZES)
    assert (res["object_id"], res["rule"]) == ("o2", 2)
    # with the literal reading (straight to the judge) the same case waits for the judge
    monkeypatch.setattr(semantic, "RULE1_MISS_NEXT_RULE", 3)
    res = resolve_object("pink brick", [], both, golds, SIZES)
    assert (res["rule"], res["status"], res["candidates"]) == (3, "pending", ["o1", "o2"])


def test_rule2_exact_and_content_words_and_rule3():
    golds = _brick_objects()
    assert resolve_object("The Pink Brick!", None, None, golds, SIZES)["object_id"] == "o1"
    # content words {pink, lego} are a subset of o1's words only
    res = resolve_object("pink lego", None, None, golds, SIZES)
    assert (res["object_id"], res["rule"]) == ("o1", 2)
    # function words do not count: {brick} after dropping "of", "the"
    assert resolve_object("the brick of the scene", None, None, golds, SIZES)["status"] == "pending"
    assert resolve_object("brick on the left", None, None, golds, SIZES)["status"] == "pending"
    res = resolve_object("red block", None, None, golds, SIZES)
    assert (res["rule"], res["status"], res["judge_key"]) == (3, "pending", "red block")
    res = resolve_object("red block", None, None, golds, SIZES, {"red block": "o2"})
    assert (res["object_id"], res["rule"], res["status"]) == ("o2", 3, "resolved")
    res = resolve_object("red block", None, None, golds, SIZES, {"red block": "o9"})
    assert (res["object_id"], res["status"]) == (None, "resolved")


def test_no_reference_and_unsure():
    golds = _brick_objects()
    for name in (None, "none", "None", "", "nothing", "n/a", "the"):
        res = resolve_object(name, None, None, golds, SIZES)
        assert (res["object_id"], res["rule"], res["status"]) == (None, None, "resolved"), name
    res = resolve_object("unsure", None, None, golds, SIZES)
    assert (res["object_id"], res["rule"], res["reason"]) == (None, None, "unsure reference")


def test_appendix_f6_two_black_bowls_go_to_the_judge():
    golds = _gold_f2()["objects"]
    res = resolve_object("black bowl", None, None, golds, None)
    assert (res["object_id"], res["rule"], res["status"]) == (None, 3, "pending")
    assert res["candidates"] == ["o1", "o2"]
    res = resolve_object("black bowl", None, None, golds, None, {"black bowl": "ambiguous"})
    assert (res["object_id"], res["rule"], res["status"]) == (None, 3, "resolved")
    assert res["reason"] == "judge: ambiguous"

    # In S2 the ambiguous answer is wrong; without an answer the item is pending.
    gold_segs = _gold_f2()["segments"]
    pred = [_vseg(20, 29, "grasp", target="black bowl")]
    ids = ["o1", "o2", "o3"]
    s2 = s2_episode(pred, gold_segs[1:2], ObjectResolver(golds, [], None, None, "F2/0"), ids)
    assert s2["counts"] == {"numerator": 0, "denominator": 1, "pending": 1}
    s2 = s2_episode(pred, gold_segs[1:2], ObjectResolver(golds, [], None, {"black bowl": "ambiguous"}), ids)
    assert s2["counts"] == {"numerator": 0, "denominator": 1, "pending": 0}
    s2 = s2_episode(pred, gold_segs[1:2], ObjectResolver(golds, [], None, {"black bowl": "o1"}), ids)
    assert s2["counts"] == {"numerator": 1, "denominator": 1, "pending": 0}


def test_resolver_uses_the_view_inventory_and_queues_judge_items():
    golds = _gold_f2()["objects"]
    view_objects = [{"object_id": "o7", "name": "bowl", "points": [{"camera": IMG, "x": 0.31, "y": 0.5}],
                     "boxes": []},
                    {"object_id": "o8", "name": "black bowl", "points": [], "boxes": []}]
    resolver = ObjectResolver(golds, view_objects, {IMG: [256, 256]}, None, "F2/0")
    # "bowl" alone is ambiguous by name, but its point sits on o1's gold point (rule 1)
    assert (resolver("bowl")["object_id"], resolver("bowl")["rule"]) == ("o1", 1)
    # the arm's own inventory ID works too
    assert resolver("o7")["object_id"] == "o1"
    # "black bowl" has no point: rule 2 finds two objects, so it is queued once
    assert resolver("black bowl")["status"] == "pending"
    assert resolver("o8")["status"] == "pending"
    queue = resolver.judge_queue
    assert [q["judge_id"] for q in queue] == ["object|F2/0|black bowl"]
    assert queue[0]["gold_object_ids"] == ["o1", "o2", "o3"] and queue[0]["answered"] is False


# --------------------------------------------------------------------------- #
# S1, S2, S4
# --------------------------------------------------------------------------- #
def _s1_pred():
    return [
        _vseg(0, 34, "approach", target="pink brick"),
        _vseg(35, 69, None, target="transparent box", text="carry the brick"),
        _vseg(70, 99, "dance", target="unsure", destination="box"),
    ]


def test_s1_framewise_seg_and_unmapped():
    gold = _gold_f1()["segments"]
    r = s1_episode(_s1_pred(), gold, 100)
    # frames 0-29 approach = approach (30), 30-39 wrong, 40-69 transport = transport (30), 70-99 unmapped
    assert r["S1"] == {"numerator": 60, "denominator": 100, "pending": 0}
    assert r["S1-unmapped"] == {"numerator": 30, "denominator": 100, "pending": 0}
    # T3 pairs: (p0, g0) 30/35, (p1, g2) 30/35, (p2, g4) 20/30, all >= 0.5; the third differs in class
    assert r["S1-seg"] == {"numerator": 2, "denominator": 3, "pending": 0}
    assert [(p["pred_idx"], p["gold_idx"]) for p in r["seg_pairs"]] == [(0, 0), (1, 2), (2, 4)]


def test_s1_uncovered_frames_are_wrong_and_unlabeled_gold_frames_are_skipped():
    gold = _gold_f1()["segments"]
    gold[4]["phase_class"] = None                      # 20 gold frames without a class
    r = s1_episode([_vseg(0, 29, "approach")], gold, 100)
    assert r["S1"]["numerator"] == 30 and r["S1"]["denominator"] == 80
    assert r["uncovered_frames"] == 70 and r["gold_unlabeled_frames"] == 20
    assert r["S1-unmapped"] == {"numerator": 0, "denominator": 30, "pending": 0}


def test_s2_targets_no_counterpart_and_destinations():
    gold = _gold_f1()
    resolver = ObjectResolver(gold["objects"], [], SIZES)
    s2 = s2_episode(_s1_pred(), gold["segments"], resolver, ["o1", "o2"])
    # g0 -> p0 pink brick (o1) right; g1 best IoU 5/40 < 0.2: no counterpart; g2 -> p1 box (o2) wrong;
    # g3 -> p2 "unsure" wrong. The retract has no target.
    assert s2["counts"] == {"numerator": 1, "denominator": 4, "pending": 0}
    assert [i["correct"] for i in s2["items"]] == [True, False, False, False]
    assert s2["items"][1]["reason"] == "no counterpart"
    d = s2_episode(_s1_pred(), gold["segments"], resolver, ["o1", "o2"], destination=True)
    # only the release segment counts: p2's destination "box" is o2's alias
    assert d["counts"] == {"numerator": 1, "denominator": 1, "pending": 0}


def test_s2_region_destination_is_compared_as_text():
    gold = _gold_f1()
    gold["segments"][3]["destination"] = "left side of the table"
    resolver = ObjectResolver(gold["objects"], [], SIZES)
    pred = [_vseg(70, 79, "release", destination="the left side of table")]
    d = s2_episode(pred, gold["segments"], resolver, ["o1", "o2"], destination=True)
    assert d["counts"] == {"numerator": 1, "denominator": 1, "pending": 0}


def _s4_pred():
    return [
        _vseg(0, 29, "approach"),
        _vseg(30, 40, "grasp", outcome="failed", failure_type="missed_grasp"),
        _vseg(41, 44, "approach", mistake=True),
        _vseg(45, 79, "transport"),
        _vseg(80, 90, "grasp", outcome="failed", failure_type="drop"),
        _vseg(91, 99, "retract"),
    ]


def test_s4_spans_types_and_episode_level():
    gold_fa = [{"span": [30, 45], "failure_type": "missed_grasp"}, {"span": [60, 70], "failure_type": "slip"}]
    r = s4_episode(_s4_pred(), gold_fa)
    # predicted spans [30, 44] (two merged segments) and [80, 90]; [30, 44] vs [30, 45] IoU 15/16
    assert r["pred_spans"] == [[30, 44], [80, 90]]
    assert r["pairs"][0]["iou"] == pytest.approx(0.9375)
    assert r["S4-P"]["numerator"] == 1 and r["S4-P"]["denominator"] == 2
    assert r["S4-R"]["numerator"] == 1 and r["S4-R"]["denominator"] == 2
    assert (r["S4-F1"]["numerator"], r["S4-F1"]["denominator"]) == (2, 4)
    assert (r["S4-type"]["numerator"], r["S4-type"]["denominator"]) == (1, 1)
    assert (r["S4-ep-P"]["numerator"], r["S4-ep-P"]["denominator"]) == (1, 1)
    # a predicted failure where the gold has none counts against episode-level precision
    r2 = s4_episode(_s4_pred(), [])
    assert (r2["S4-ep-P"]["numerator"], r2["S4-ep-P"]["denominator"]) == (0, 1)
    assert (r2["S4-ep-R"]["numerator"], r2["S4-ep-R"]["denominator"]) == (0, 0)


# --------------------------------------------------------------------------- #
# Requirement matching and G metrics
# --------------------------------------------------------------------------- #
def test_matching_prefers_status_agreement_then_order():
    gold = gold_requirements([
        _greq("r1", "robot_end_state", None, "withdrawn", None, True, "unsure", "intent"),
        _greq("r2", "robot_end_state", None, "withdrawn", None, True, "required"),
    ], [])
    pred = [dict(gold[1], position=0), dict(gold[0], position=1)]   # required first, then unsure
    assert match_requirements(pred, gold) == [(1, 0), (0, 1)]
    # without status agreement the order decides
    pred2 = [dict(gold[1], status="incidental"), dict(gold[1], status="incidental")]
    assert match_requirements(pred2, gold) == [(0, 0), (1, 1)]


def test_could_match_treats_pending_names_as_object_ids_only():
    g = gold_requirements([_greq("r1", "object_end_state", "o1", "inside", "o2", True, "required")],
                          ["o1", "o2"])[0]
    p = dict(g, object=PENDING_REF)
    assert not items_equal(p, g) and could_match(p, g)
    g_null = gold_requirements([_greq("r2", "robot_end_state", None, "holding", None, False, "required")], [])[0]
    assert not could_match(dict(g_null, object=PENDING_REF), g_null)


def test_appendix_f7_incidental_as_required_makes_g6_false():
    gold = _gold_f1()
    gold["goal"]["requirements"] = [
        _greq("r1", "object_end_state", "o1", "inside", "o2", True, "required"),
        _greq("r2", "robot_end_state", None, "holding", None, False, "required"),
        _greq("r3", "robot_end_state", None, "at_home_pose", None, True, "incidental"),
    ]
    assert validate_gold(_family_doc(gold)) == []
    view = _view_f1()
    view["goal"]["requirements"] = [
        _vreq("object_end_state", "inside", "pink brick", "clear container", True, "required"),
        _vreq("robot_end_state", "holding", "none", "none", False, "unsure", "intent"),
        _vreq("robot_end_state", "at_home_pose", "none", "none", True, "required"),
    ]
    r = score_view(view, gold)
    assert _c(r, "G1a") == (2, 2, 0)          # both gold ending items stated
    assert _c(r, "G1b") == (1, 1, 0)
    assert _c(r, "G1c") == (0, 1, 0)          # holding is stated unsure, not required
    assert _c(r, "G4") == (1, 2, 0)           # at_home_pose is incidental in gold
    assert _c(r, "G4-h") == (0, 1, 0)
    assert _c(r, "G3") == (0, 1, 0)
    assert _c(r, "G6") == (0, 1, 0)
    assert r["details"]["goal"]["G6"]["G4_none_incidental_or_hallucinated"] is False
    # G5: one pair predicted unsure (intent) against gold none
    assert _c(r, "G5-P") == (0, 1, 0) and _c(r, "G5-R") == (0, 0, 0)
    assert r["details"]["goal"]["g5_confusion"]["none"] == {"none": 2, "perception": 0, "intent": 1}


def test_goal_perfect_view_scores_everything_right():
    r = score_view(_view_f1(), _gold_f1())
    for tau in (3, 5, 10):
        assert _c(r, f"T1-F1@{tau}") == (8, 8, 0)
        assert _c(r, f"T6-F1@{tau}") == (4, 4, 0)       # coarse boundaries 39 and 79
    assert _c(r, "T3") == (1.0, 1, 0) and _c(r, "T5") == (0, 1, 0) and _c(r, "T4") == (0, 1, 0)
    assert _c(r, "T2-MAE") == (0, 4, 0)
    assert _c(r, "S1") == (100, 100, 0) and _c(r, "S1-seg") == (5, 5, 0) and _c(r, "S1-unmapped") == (0, 100, 0)
    assert _c(r, "S2") == (4, 4, 0) and _c(r, "S2-dest") == (1, 1, 0)
    assert _c(r, "S4-F1") == (0, 0, 0) and _c(r, "S4-ep-R") == (0, 0, 0)
    assert _c(r, "G1a") == (2, 2, 0) and _c(r, "G1b") == (1, 1, 0) and _c(r, "G1c") == (1, 1, 0)
    assert _c(r, "G2") == (0, 0, 0)
    assert _c(r, "G3") == (0, 1, 0) and _c(r, "G4") == (0, 2, 0) and _c(r, "G4-h") == (0, 1, 0)
    assert _c(r, "G5-P") == (1, 1, 0) and _c(r, "G5-R") == (1, 1, 0) and _c(r, "G5-P-intent") == (1, 1, 0)
    assert _c(r, "G5-dropped") == (0, 1, 0)
    assert _c(r, "G6") == (1, 1, 0) and _c(r, "G7") == (2, 2, 0) and _c(r, "G7-outcome") == (1, 1, 0)
    assert r["pending"] == [] and r["judge_queue"] == []
    assert r["values"]["S2-dest"] == 1.0 and r["values"]["S4-P"] is None
    # "clear container" resolved by its point (rule 1)
    assert r["details"]["S2-dest"][0]["rule"] == 1


def test_canonical_holding_form_compares_ref_object_with_null():
    gold = _gold_f1()
    view = _view_f1()
    view["goal"]["requirements"][1] = _vreq("robot_end_state", "holding", "robot", "nothing", False, "required")
    r = score_view(view, gold)
    assert _c(r, "G1a") == (2, 2, 0)          # "robot" subject and "nothing" held are the canonical form
    view["goal"]["requirements"][1] = _vreq("robot_end_state", "holding", "none", "pink brick", False, "required")
    r = score_view(view, gold)
    assert _c(r, "G1a") == (1, 2, 0)          # "not holding the brick" is not "holding nothing"
    assert _c(r, "G4-h") == (1, 1, 0) and _c(r, "G4-h-any") == (1, 1, 0)
    assert _c(r, "G6") == (0, 1, 0)


def test_g5_kinds_dropped_items_and_g7_achieved():
    gold = _gold_f1()
    view = _view_f1()
    view["goal"]["requirements"][0]["achieved"] = False
    view["goal"]["requirements"][2] = _vreq("robot_end_state", "withdrawn", "none", "none", True, "unsure",
                                            "perception")
    r = score_view(view, gold)
    assert _c(r, "G5-P") == (1, 1, 0)                 # unsure in both
    assert _c(r, "G5-P-perception") == (0, 1, 0)      # but the kinds differ
    assert _c(r, "G5-R-intent") == (0, 1, 0)
    assert r["details"]["goal"]["g5_confusion"]["intent"]["perception"] == 1
    assert _c(r, "G7") == (1, 2, 0)                   # inside achieved false against gold true
    del view["goal"]["requirements"][2]
    r = score_view(view, gold)
    assert _c(r, "G5-dropped") == (1, 1, 0) and _c(r, "G5-R") == (0, 0, 0)


def test_goal_null_fails_g1_g3_and_g6():
    view = _view_f1()
    view["goal"] = None
    view["episode_outcome"] = "unknown"
    r = score_view(view, _gold_f1())
    assert r["goal_missing"] is True and r["no_output"] is False
    assert _c(r, "G1a") == (0, 2, 0) and _c(r, "G1b") == (0, 1, 0)
    assert _c(r, "G3") == (1, 1, 0) and _c(r, "G6") == (0, 1, 0)
    assert _c(r, "G4") == (0, 0, 0) and _c(r, "G4-h") == (0, 1, 0) and _c(r, "G7") == (0, 0, 0)
    assert _c(r, "G7-outcome") == (0, 1, 0)
    assert _c(r, "S2") == (4, 4, 0)                   # the segments are still scored


def test_missing_output_rule():
    view = {"arm": "v@test", "episode_key": "F1/0", "no_output": True, "segments": [], "goal": None}
    r = score_view(view, _gold_f1())
    assert r["no_output"] is True and r["missing_output_reason"] == "no_output"
    assert _c(r, "T1-R@5") == (0, 4, 0) and _c(r, "T1-P@5") == (0, 0, 0)
    assert _c(r, "T4") == (1, 1, 0) and _c(r, "T5") == (-4, 1, 0)
    assert _c(r, "S1") == (0, 100, 0) and _c(r, "S1-unmapped") == (100, 100, 0)
    assert _c(r, "S2") == (0, 4, 0) and _c(r, "S4-P") == (0, 0, 0)
    assert _c(r, "G1b") == (0, 1, 0) and _c(r, "G3") == (1, 1, 0) and _c(r, "G6") == (0, 1, 0)
    assert _c(r, "G7-outcome") == (0, 1, 0)
    # segments missing from an otherwise present view take the same rule
    view2 = _view_f1()
    view2["segments"] = None
    r2 = score_view(view2, _gold_f1())
    assert r2["no_output"] is True and r2["missing_output_reason"] == "no segments"
    assert _c(r2, "G6") == (0, 1, 0)


def test_appendix_f9_missing_output_on_a_three_segment_gold():
    gold = _gold_f1()
    gold["segments"] = [_gseg(0, 0, 39, "grasp", "o1"), _gseg(1, 40, 79, "transport", "o1", "o2"),
                        _gseg(2, 80, 99, "retract", None, last=True)]
    del gold["coarse_subtasks"]
    assert validate_gold(_family_doc(gold)) == []
    r = score_view({"arm": "v@test", "episode_key": "F1/0", "no_output": True}, gold)
    # one predicted segment against 3 gold: degenerate; gold boundaries 39 and 79, none predicted
    assert _c(r, "T4") == (1, 1, 0) and r["details"]["T4"]["single_segment"] is True
    for tau in (3, 5, 10):
        assert _c(r, f"T1-R@{tau}") == (0, 2, 0)
    assert r["values"]["T1-R@5"] == 0.0 and _c(r, "G6") == (0, 1, 0)
    assert r["goal_missing"] is True and "T6-F1@5" not in r["counts"]


def test_rule1_several_point_matches_fall_through():
    golds = _brick_objects()
    golds[1]["first_frame_point"]["xy"] = [0.42, 0.6]
    # (0.41, 0.6) is 6.4 px from both gold points (limit 40 px): two matches, then rule 2 finds nothing
    res = resolve_object("thing", [{"camera": UP, "x": 0.41, "y": 0.6}], [], golds, SIZES)
    assert (res["rule"], res["status"], res["candidates"]) == (3, "pending", ["o1", "o2"])
    res = resolve_object("transparent box", [{"camera": UP, "x": 0.41, "y": 0.6}], [], golds, SIZES)
    assert (res["object_id"], res["rule"]) == ("o2", 2)


def test_duplicate_name_suffix_is_not_part_of_the_name():
    # the scene layer names duplicates "black bowl (o1)", "black bowl (o2)" after its own IDs
    view_objects = [{"object_id": "o1", "name": "black bowl (o1)", "points": [], "boxes": []},
                    {"object_id": "o2", "name": "black bowl (o2)", "points": [], "boxes": []}]
    resolver = ObjectResolver(_gold_f2()["objects"], view_objects, None, None, "F2/0")
    assert resolver("black bowl (o1)")["judge_key"] == "black bowl"
    assert resolver("black bowl (o2)")["status"] == "pending"
    # one judge item, and the judge never sees the arm's own ID
    assert [(q["judge_id"], q["text"]) for q in resolver.judge_queue] == [("object|F2/0|black bowl", "black bowl")]
    # with one black bowl in the gold, the name resolves by rule 2
    single = [o for o in _gold_f2()["objects"] if o["object_id"] != "o2"]
    resolver = ObjectResolver(single, view_objects, None, None, "F2/0")
    assert (resolver("black bowl (o2)")["object_id"], resolver("black bowl (o2)")["rule"]) == ("o1", 2)
    # a name that only ends like an ID is kept
    other = [{"object_id": "o3", "name": "bin (o1)", "points": [], "boxes": []}]
    assert ObjectResolver(single, other)("bin (o1)")["judge_key"] == "bin o1"


def test_robot_subject_names_mean_no_object():
    gold = _gold_f1()
    for subject in ("robot's gripper", "the gripper fingers", "Robot arm"):
        view = _view_f1()
        view["goal"]["requirements"][1] = _vreq("robot_end_state", "holding", subject, "none", False, "required")
        r = score_view(view, gold)
        assert _c(r, "G1a") == (2, 2, 0) and r["judge_queue"] == [], subject
    # an object name is still resolved on a robot item
    view = _view_f1()
    view["goal"]["requirements"][1] = _vreq("robot_end_state", "holding", "pink brick", "none", False, "required")
    assert _c(score_view(view, gold), "G1a") == (1, 2, 0)


def test_g2_coarse_text_narrating_both_attempts():
    # "grasp twice" over one merged segment: its IoU with the failed span [30, 45] is 16 / 40 = 0.4,
    # so only the coarse text marks the copy (spec 4.3 G2 part i example)
    gold, view = _gold_f3(), _view_f3()
    view["segments"] = [_vseg(0, 29, "approach"), _vseg(30, 69, "grasp"), *view["segments"][4:]]
    view["coarse"] = [{"start": 0, "end": 69, "text": "grasp the pink brick twice", "mistake": False},
                      *view["coarse"][2:]]
    r = score_view(view, gold)
    assert _c(r, "G2-i") == (1, 1, 0) and _c(r, "G2") == (1, 1, 0) and _c(r, "G6") == (0, 1, 0)
    assert [h["source"] for h in r["details"]["goal"]["G2"]["part_i_hits"]] == ["coarse"]
    view["coarse"][0]["text"] = "pick up the pink brick"
    assert _c(score_view(view, gold), "G2-i") == (0, 1, 0)


# --------------------------------------------------------------------------- #
# Judge-dependent items: pending and G6
# --------------------------------------------------------------------------- #
def test_two_bowls_pending_until_judged():
    gold, view = _gold_f2(), _view_f2()
    r = score_view(view, gold)
    assert _c(r, "S2") == (0, 4, 4) and r["values"]["S2"] == "pending"
    assert _c(r, "S2-dest") == (1, 1, 0)                      # "plate" resolves by name
    assert r["values"]["G3"] == "pending" and r["values"]["G1a"] == "pending"
    assert r["values"]["G6"] == "pending"
    assert [q["judge_id"] for q in r["judge_queue"]] == ["object|F2/0|black bowl"]

    r = score_view(view, gold, {"object|F2/0|black bowl": "ambiguous"})
    assert r["pending"] == []
    assert _c(r, "S2") == (0, 4, 0) and _c(r, "G3") == (1, 1, 0)
    assert _c(r, "G1a") == (1, 2, 0) and _c(r, "G4-h") == (1, 1, 0) and _c(r, "G6") == (0, 1, 0)

    # the contract name of the third argument, and judge_answers.jsonl records
    r = score_view(view, gold, gold_objects_resolution_judge={"object|F2/0|black bowl": "o1"})
    assert _c(r, "S2") == (4, 4, 0) and _c(r, "G3") == (0, 1, 0) and _c(r, "G6") == (1, 1, 0)
    r = score_view(view, gold, [{"judge_id": "object|F2/0|black bowl", "answer": "o1"}])
    assert _c(r, "G6") == (1, 1, 0)
    # an answer for another episode does not leak in
    r = score_view(view, gold, {"object|F2/1|black bowl": "o1"})
    assert r["values"]["S2"] == "pending"


def test_g6_false_despite_pending_when_a_required_item_cannot_match():
    gold, view = _gold_f2(), _view_f2()
    view["goal"]["primary_target_name"] = "left black bowl"          # rule 2, exact name
    view["goal"]["requirements"].append(_vreq("object_end_state", "lifted", "plate", "none", True, "required"))
    r = score_view(view, gold)
    assert r["values"]["G1a"] == "pending" and _c(r, "G3") == (0, 1, 0)
    assert _c(r, "G6") == (0, 1, 0)                                  # lifted(plate) is surely hallucinated


def test_g2_part_ii_waits_for_the_judge_and_g6_with_it():
    gold, view = _gold_f3(), _view_f3()
    assert validate_gold(_family_doc(gold)) == []
    r = score_view(view, gold)
    assert _c(r, "G2-i") == (0, 1, 0)
    assert _c(r, "G2") == (0, 1, 3) and r["values"]["G2"] == "pending"
    assert r["values"]["G2-ii"] == "pending" and r["values"]["G6"] == "pending"
    g2_items = [q for q in r["judge_queue"] if q["type"] == "g2_requirement"]
    assert len(g2_items) == 3 and g2_items[0]["failed_spans"] == [[30, 45]]
    # S4: the marked span matches the gold span exactly
    assert _c(r, "S4-F1") == (2, 2, 0) and _c(r, "S4-type") == (1, 1, 0)

    answers = {q["judge_id"]: "no" for q in g2_items}
    r = score_view(view, gold, answers)
    assert _c(r, "G2") == (0, 1, 0) and _c(r, "G2-ii") == (0, 1, 0) and _c(r, "G6") == (1, 1, 0)
    answers[g2_items[0]["judge_id"]] = "yes"
    r = score_view(view, gold, answers)
    assert _c(r, "G2") == (1, 1, 0) and _c(r, "G2-ii") == (1, 1, 0) and _c(r, "G6") == (0, 1, 0)

    # a wrong primary target makes G6 false without the judge
    wrong = copy.deepcopy(view)
    wrong["goal"]["primary_target_name"] = "clear container"
    r = score_view(wrong, gold)
    assert _c(r, "G6") == (0, 1, 0) and r["values"]["G2"] == "pending"


def test_g2_part_i_segment_or_coarse_text():
    gold = _gold_f3()
    view = _view_f3()
    view["coarse"][0]["text"] = "try to grasp the pink brick"
    r = score_view(view, gold)
    assert _c(r, "G2-i") == (1, 1, 0) and _c(r, "G2") == (1, 1, 0) and _c(r, "G6") == (0, 1, 0)
    assert r["details"]["goal"]["G2"]["part_i_hits"][0]["source"] == "coarse"
    view = _view_f3()
    view["segments"][1].update(outcome="success", mistake=False, failure_type="none")
    r = score_view(view, gold)
    assert _c(r, "G2-i") == (1, 1, 0)
    assert r["details"]["goal"]["G2"]["part_i_hits"][0] == {"source": "segment", "index": 1,
                                                            "gold_span": [30, 45], "iou": 1.0}
    assert narrates_failure("withdraw after missing") and narrates_failure("grasp twice")
    assert not narrates_failure("put the cube in the tray") and not narrates_failure("drop it in the box")


def test_g6_from_conditions():
    assert g6_episode({"a": True, "b": "pending"})["correct"] == "pending"
    assert g6_episode({"a": False, "b": "pending"})["correct"] is False
    assert g6_episode({"a": True, "b": True})["counts"] == {"numerator": 1, "denominator": 1, "pending": 0}


# --------------------------------------------------------------------------- #
# score_view guards, legacy continuity, aggregation, determinism
# --------------------------------------------------------------------------- #
def test_score_view_refuses_heldout_and_mismatched_keys():
    gold = _gold_f1()
    gold["split"] = "heldout"
    with pytest.raises(HeldoutRefused):
        score_view(_view_f1(), gold)
    assert score_view(_view_f1(), gold, allow_heldout=True)["episode_key"] == "F1/0"
    view = _view_f1()
    view["episode_key"] = "F1/1"
    with pytest.raises(ValueError):
        score_view(view, _gold_f1())


def test_score_legacy_boundaries_continuity_only():
    legacy = [{"start_frame": 0, "end_frame": 159}, {"start_frame": 160, "end_frame": 174},
              [175, 215], {"start": 216, "end": 299}]
    view = {"arm": "v@test", "episode_key": "F1/0", "num_frames": 300,
            "segments": [_vseg(0, 158, "approach"), _vseg(159, 180, "grasp"), _vseg(181, 215, "transport"),
                         _vseg(216, 299, "release")]}
    r = score_legacy_boundaries(view, legacy)
    assert r["label"] == "continuity only"
    assert r["gold_boundaries"] == [159, 174, 215] and r["pred_boundaries"] == [158, 180, 215]
    assert r["T1"]["3"]["matched"] == 2 and r["T1"]["5"]["matched"] == 2 and r["T1"]["10"]["matched"] == 3
    assert r["T1"]["5"]["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert r["T5"] == 0
    assert r["T3"] == pytest.approx((159 / 160 + 15 / 22 + 35 / 41 + 1) / 4, abs=1e-6)
    r = score_legacy_boundaries({"no_output": True}, legacy, num_frames=300)
    assert r["no_output"] is True and r["T1"]["10"]["matched"] == 0 and r["T5"] == -3


def test_aggregate_micro_macro_subsets_and_pending():
    f1 = score_view(_view_f1(), _gold_f1())
    f3 = score_view({"arm": "v@test", "episode_key": "F3/544", "no_output": True}, _gold_f3())
    f2 = score_view(_view_f2(), _gold_f2())
    agg = aggregate([f3, f1, f2])
    m = agg["metrics"]
    # T1-R@5: F1 4/4, F3 0/6, F2 4/4 -> micro 8/14; macro mean(1, 0, 1)
    assert (m["T1-R@5"]["numerator"], m["T1-R@5"]["denominator"]) == (8, 14)
    assert m["T1-R@5"]["value"] == pytest.approx(8 / 14, abs=1e-6)
    assert agg["macro"]["T1-R@5"] == pytest.approx(2 / 3, abs=1e-6)
    assert agg["by_family"]["F3"]["metrics"]["T1-R@5"]["value"] == 0.0
    assert (m["T1-R@5"]["metric_id"], m["T1-R@5"]["tau"]) == ("T1-R", 5)
    # T3 is a mean of per-episode scores
    assert m["T3"]["denominator"] == 3
    # S2 waits for the judge on F2/0, so the pooled S2 and its macro read pending
    assert m["S2"]["value"] == "pending" and m["S2"]["pending_items"] == 4 and agg["macro"]["S2"] == "pending"
    assert "S2" in agg["pending_metrics"] and agg["g5_confusion"] == "pending"
    assert [q["judge_id"] for q in agg["judge_queue"]][0] == "object|F2/0|black bowl"
    assert agg["n_no_output"] == 1 and agg["families"] == {"F1": 1, "F2": 1, "F3": 1}
    assert set(agg["by_subset"]) == {"ordinary", "distractor", "failed_attempt"}
    assert agg["by_subset"]["ordinary"]["n_episodes"] == 1
    # a subset filter
    sub = aggregate([f3, f1, f2], subset="failed_attempt")
    assert sub["n_episodes"] == 1 and sub["metrics"]["G6"]["value"] == 0.0
    # records for metrics.json
    recs = to_records(agg, "v@test")
    row = next(r for r in recs if r["family"] == "F1" and r["metric_id"] == "T1-F1" and r["tau"] == 5)
    assert (row["numerator"], row["denominator"], row["value"]) == (8, 8, 1.0)
    assert any(r["family"] == "macro" and r["metric_id"] == "G6" for r in recs)
    with pytest.raises(ValueError):
        aggregate([f1, dict(f3, arm="v@other")])
    with pytest.raises(ValueError):
        aggregate([f1, f1])
    assert split_metric_key("S4-ep-P") == ("S4-ep-P", None)


def test_scoring_is_deterministic():
    def run():
        eps = [score_view(_view_f1(), _gold_f1()), score_view(_view_f3(), _gold_f3()),
               score_view(_view_f2(), _gold_f2(), {"object|F2/0|black bowl": "o2"})]
        return json.dumps({"eps": eps, "agg": aggregate(eps)}, sort_keys=True)
    assert run() == run()


# --------------------------------------------------------------------------- #
# Statistics (spec 6)
# --------------------------------------------------------------------------- #
def _eps(values, family="F1", key="G6", start=0):
    return [{"episode_key": f"{family}/{start + i}", "family": family,
             "counts": {key: {"numerator": v, "denominator": 1, "pending": 0}}} for i, v in enumerate(values)]


def test_metric_value_is_micro_within_family_and_macro_across():
    eps = _eps([1, 0], "F1") + _eps([1], "F3")
    v = stats.metric_value(eps, "G6")
    assert v["by_family"] == {"F1": 0.5, "F3": 1.0} and v["value"] == 0.75
    pend = _eps([1], "F1")
    pend[0]["counts"]["G6"]["pending"] = 1
    assert stats.metric_value(pend, "G6")["value"] == "pending"


def test_bootstrap_is_identical_for_the_same_seed():
    a = _eps([1, 0, 1, 1, 0, 1, 0, 1], "F1") + _eps([0, 1, 1], "F3")
    b = _eps([0, 0, 1, 0, 0, 1, 1, 0], "F1") + _eps([0, 0, 1], "F3")
    one = stats.paired_difference(a, b, "G6", contrast_index=3, n_boot=2000, mei=0.10)
    two = stats.paired_difference(list(reversed(a)), b, "G6", contrast_index=3, n_boot=2000, mei=0.10)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    assert one["seed"] == 20261001 + 3
    boot = stats.bootstrap_metric(a, "G6", contrast_index=3, n_boot=2000)
    assert boot == stats.bootstrap_metric(a, "G6", contrast_index=3, n_boot=2000)
    assert stats.N_BOOT == 10_000 and stats.bootstrap_metric(a, "G6", n_boot=100)["seed"] == 20261001


def test_clear_difference():
    a = _eps([1] * 20, "F1") + _eps([1] * 10, "F3")
    b = _eps([0] * 20, "F1") + _eps([0] * 10, "F3")
    r = stats.paired_difference(a, b, "G6", contrast_index=0, n_boot=2000, mei=0.10)
    assert (r["diff"], r["interval"], r["p_boot"]) == (1.0, [1.0, 1.0], 0.0)
    assert r["outcome_word"] == "clear difference"
    # the same data below the planned n is inconclusive by rule
    r = stats.paired_difference(a, b, "G6", contrast_index=0, n_boot=500, mei=0.10, planned_n=74)
    assert r["outcome_word"] == "inconclusive"


def test_interval_straddling_zero():
    a = _eps([1, 0] * 10)
    b = _eps([0, 1] * 10)
    r = stats.paired_difference(a, b, "G6", contrast_index=1, n_boot=2000, mei=0.10)
    lo, hi = r["interval"]
    assert r["diff"] == 0.0 and lo < 0 < hi and (hi - lo) / 2 > 0.10
    assert r["p_boot"] == 1.0 and r["outcome_word"] == "inconclusive"
    # nearly identical systems on many episodes: the interval is narrow, so no clear difference
    a = _eps([1] * 396 + [1, 1, 0, 0])
    b = _eps([1] * 396 + [0, 0, 1, 1])
    r = stats.paired_difference(a, b, "G6", contrast_index=2, n_boot=2000, mei=0.05)
    lo, hi = r["interval"]
    assert lo <= 0 <= hi and (hi - lo) / 2 <= 0.05 and r["outcome_word"] == "no clear difference"


def test_paired_difference_compares_families_defined_for_both():
    def eps(family, rows, start=0):
        return [{"episode_key": f"{family}/{start + i}", "family": family,
                 "counts": {"S4-P": {"numerator": n, "denominator": d, "pending": 0}}} for i, (n, d) in enumerate(rows)]
    # A predicts no failed span in F1 (precision undefined there); B has F1 0.5 and F3 0.0
    a = eps("F1", [(0, 0), (0, 0)]) + eps("F3", [(1, 2), (1, 2)])
    b = eps("F1", [(1, 1), (0, 1)]) + eps("F3", [(0, 2), (0, 2)])
    r = stats.paired_difference(a, b, "S4-P", contrast_index=0, n_boot=500, mei=None)
    # each system's own value keeps its families (A 0.5, B mean(0.5, 0) = 0.25), the difference uses F3 only
    assert (r["value_a"], r["value_b"]) == (0.5, 0.25)
    assert r["families_compared"] == ["F3"] and r["diff"] == 0.5 and r["interval"] == [0.5, 0.5]
    assert r["outcome_word"] == "clear difference"
    none = stats.paired_difference(eps("F1", [(0, 0)]), eps("F1", [(1, 1)]), "S4-P", contrast_index=0, n_boot=10)
    assert none["diff"] is None and none["outcome_word"] == "inconclusive"


def test_paired_difference_needs_identical_episodes_and_no_pending():
    with pytest.raises(ValueError):
        stats.paired_difference(_eps([1, 0]), _eps([1, 0], start=5), "G6", contrast_index=0, n_boot=10)
    b = _eps([1, 0])
    b[0]["counts"]["G6"]["pending"] = 2
    r = stats.paired_difference(_eps([1, 0]), b, "G6", contrast_index=0, n_boot=10)
    assert r["outcome_word"] == "pending" and r["interval"] is None


def test_p_boot_outcome_words_and_holm():
    assert stats.p_boot([-1, 0, 1, 2], 1.0) == 1.0                 # 2 of 4 at or below 0
    assert stats.p_boot([-1] + [1] * 9, 0.5) == 0.2
    assert stats.p_boot([-1] * 9 + [0.5], -1.0) == 0.2
    assert stats.p_boot([1, 2], 0.0) == 1.0
    assert stats.outcome_word([0.02, 0.10], 0.05) == "clear difference"
    assert stats.outcome_word([-0.10, -0.01], 0.05) == "clear difference"
    assert stats.outcome_word([-0.03, 0.04], 0.05) == "no clear difference"
    assert stats.outcome_word([-0.10, 0.20], 0.05) == "inconclusive"
    assert stats.outcome_word([-0.03, 0.04], None) == "inconclusive"
    assert stats.outcome_word([0.02, 0.10], 0.05, n_ok=False) == "inconclusive"
    assert stats.outcome_word(None, 0.05) == "inconclusive"
    # Holm: sorted 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.03, 0.06, max(0.06, 0.04)
    assert stats.holm([0.01, 0.04, 0.03, 0.005]) == [0.03, 0.06, 0.06, 0.02]
    assert stats.holm_reject([0.01, 0.04, 0.03, 0.005]) == [True, False, False, True]


def test_mei_table():
    assert stats.MEI["G6"] == 0.10 and stats.MEI["T3"] == 0.05 and stats.MEI["S3"] == 0.03
    assert stats.mei_for("T1-F1@5") == 0.05
    assert stats.mei_for("G1b") == 0.10 and stats.mei_for("G5-P-intent") == 0.10
    assert stats.mei_for("S2") == 0.05 and stats.mei_for("S2", subset="distractor") == 0.10
    assert stats.mei_for("G3", subset="distractor") == 0.10 and stats.mei_for("G3", subset="ordinary") == 0.05
    assert stats.mei_for("T2-MAE") == 1.0 and stats.mei_for("C1-R@10") == 0.10
    assert stats.mei_for("C3", reference=12.0) == 2.4 and stats.mei_for("C3") is None
    assert stats.mei_for("S1-seg") is None and stats.mei_for("G7") is None


def test_appendix_f8_ece():
    assert stats.ece([0.9, 0.9, 0.1, 0.1], [1, 0, 0, 0]) == 0.25
    table = stats.reliability_table([0.9, 0.9, 0.1, 0.1], [True, False, False, False])
    assert table == [{"n": 2, "mean_confidence": 0.1, "accuracy": 0.0},
                     {"n": 2, "mean_confidence": 0.9, "accuracy": 0.5}]
    assert stats.ece([], []) is None
    # 30 items: 15 equal-count bins of 2; perfectly calibrated pairs give 0
    conf = [i / 30 for i in range(30) for _ in (0,)]
    assert len(stats.reliability_table(conf, [0] * 30)) == 15
    assert stats.ece([0.5] * 30, [1, 0] * 15) == 0.0
    with pytest.raises(ValueError):
        stats.ece([0.5], [1, 0])
