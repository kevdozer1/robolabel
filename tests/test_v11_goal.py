"""v1.1 goals (SPEC_V1_1 5) and Q155 a (D1a): goal_command, the state check of the objective, render_objective,
has_end_state, the v8 goal prompt and schema, and postprocess_goal_v11 with and without a signal. No network,
no model calls."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from robolabel.episode import Episode
from robolabel.layers.goal import (
    NO_END_STATE_OBJECTIVE,
    episode_outcome,
    episode_outcome_v11,
    goal_command,
    goal_request_v11,
    is_state_objective,
    postprocess_goal,
    postprocess_goal_v11,
    render_objective,
    signal_achieved,
)
from robolabel.prompts import v7, v8
from robolabel.providers.base import ImagePart, TextPart

CAM = "video"
OBJECTS = [{"object_id": "o1", "name": "pink brick", "aliases": ["brick"], "category": "block", "views": []},
           {"object_id": "o2", "name": "transparent box", "aliases": [], "category": "container", "views": []},
           {"object_id": "o3", "name": "red button", "aliases": [], "category": "control", "views": []},
           {"object_id": "o4", "name": "stove", "aliases": [], "category": "other", "views": []},
           {"object_id": "o5", "name": "drawer", "aliases": [], "category": "container", "views": []}]
NAMES = {o["object_id"]: o["name"] for o in OBJECTS}
CATS = {o["object_id"]: o["category"] for o in OBJECTS}

# E1's frozen v8 state (RUNBOOK phase 2 item 5; tools/v11/run_e1.py prompt_state): the coarse, crawl and system
# prompt files and the coarse and crawl schemas must not change by a single byte.
FROZEN_PROMPTS = {"system": "c4c06e6179c6e33d245110ee675ed7eff6e52411206727b23f13018bb1bf7886",
                  "coarse": "c99abac24f4ec5c259d5c50f5624cc02acf3529eac6142dc17845810c362b2da",
                  "crawl": "36bdde7b4b92384cf09506aecdef354e278274ee176e46a181cba620abe80322"}
FROZEN_SCHEMAS = "d43efb9e14188e7e4c282c9d7aaeb532eb5dd8f67e567a9ddc2b239ccaa9c7e6"


def req(pred, obj="o1", ref="none", value=True, *, kind="object_end_state", status="required", achieved=True,
        uk=None, basis="observed", added_by="model"):
    """A requirement in post-processed form."""
    return {"req_id": "r1", "kind": kind, "object": obj, "predicate": pred, "ref_object": ref, "value": value,
            "status": status, "unsure_kind": uk, "basis": basis, "achieved": achieved, "deciding_frame": 99,
            "deciding_camera": CAM, "visibility": {}, "reason": "", "added_by": added_by}


def raw(pred, value="true", achieved="true", *, kind="object_end_state", obj="o1", ref="none", status="required",
        uk="none", basis="observed"):
    """A requirement as the goal call returns it (schema goal_v8)."""
    return {"req_id": "r1", "kind": kind, "object": obj, "predicate": pred, "ref_object": ref, "value": value,
            "status": status, "unsure_kind": uk, "basis": basis, "achieved": achieved, "deciding_frame": 99,
            "deciding_camera": CAM, "visibility": [{"camera": CAM, "class": "visible"}], "reason": "seen"}


def answer(objective, reqs, has_end_state=True, target="o1", destination="o2"):
    return {"objective_text": objective, "has_end_state": has_end_state, "primary_target": target,
            "primary_destination": destination, "requirements": reqs}


def clip(n: int = 100, fps: float = 20.0, task: str = "put the pink brick in the transparent box") -> Episode:
    """A clip as the clip-folder adapter gives it: one camera named video, no camera_order."""
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 255, size=(n, 36, 48, 3), dtype=np.uint8)

    def get(i):
        return frames[int(i)]

    return Episode(episode_id="C/robot", num_frames=n, fps=fps, task=task, get_frame=get, camera_key=CAM,
                   extra={"cameras": {CAM: get}})


def l1_record(holding=False, gripper=("gripper_open", "high"), withdrawn=True, attempts=None):
    return {"end_state": [
        {"predicate": "holding", "ref_object": "none", "value": holding, "basis": "signal", "confidence": "high",
         "frame": 99},
        {"predicate": gripper[0], "ref_object": "none", "value": True, "basis": "signal", "confidence": gripper[1],
         "frame": 99},
        {"predicate": "withdrawn", "ref_object": "none", "value": withdrawn, "basis": "signal", "confidence": "low",
         "frame": 99}],
        "attempts": attempts or [], "events": [], "candidates": []}


class Hidden(dict):
    """An L1 record that must never be read: any access fails the test (a hidden signal stays hidden)."""

    def _no(self, *a, **k):
        raise AssertionError("the L1 record was read although signal is False")

    get = __getitem__ = __iter__ = __contains__ = keys = items = values = __len__ = _no


# ------------------------------------------------------------------------------------------ goal_command
def test_goal_command_renders_each_predicate():
    assert goal_command([req("inside", "o1", "o2")], NAMES) == "Put the pink brick in the transparent box"
    assert goal_command([req("on_top_of", "o1", "o2")], NAMES) == "Put the pink brick on the transparent box"
    assert goal_command([req("activated", "o3")], NAMES, categories=CATS) == "Press the red button"
    assert goal_command([req("activated", "o4")], NAMES, categories=CATS) == "Turn on the stove"
    assert goal_command([req("state", "o5", value="open")], NAMES) == "Set the drawer to open"


def test_goal_command_joins_required_object_items_in_order_with_then():
    reqs = [req("inside", "o1", "o2"),
            req("holding", "none", value=False, kind="robot_end_state"),
            req("gripper_open", "none", kind="robot_end_state"),
            req("on_top_of", "o2", "o5", status="incidental"),
            req("activated", "o4", status="unsure", uk="intent"),
            req("state", "o5", value="closed"),
            req("activated", "o3")]
    assert goal_command(reqs, OBJECTS) == \
        "Put the pink brick in the transparent box then set the drawer to closed then press the red button"
    assert goal_command([r for r in reqs if r["kind"] == "robot_end_state"], NAMES) == ""
    assert goal_command([], NAMES) == ""


def test_goal_command_is_byte_identical_on_repeated_input():
    reqs = [req("inside", "o1", "o2"), req("state", "o5", value="half open"), req("activated", "o3")]
    first = goal_command(copy.deepcopy(reqs), NAMES, categories=CATS)
    runs = [goal_command(copy.deepcopy(reqs), NAMES, categories=CATS) for _ in range(5)]
    assert all(r.encode("utf-8") == first.encode("utf-8") for r in runs)
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == \
        hashlib.sha256(goal_command(reqs, OBJECTS).encode("utf-8")).hexdigest()  # the inventory gives the same
    assert reqs == [req("inside", "o1", "o2"), req("state", "o5", value="half open"), req("activated", "o3")]


def test_goal_command_decides_press_by_category_else_by_name():
    assert goal_command([req("activated", "o3")], NAMES) == "Press the red button"  # no categories: by name
    assert goal_command([req("activated", "k1")], {"k1": "stove knob"}, categories={"k1": "control"}) == \
        "Press the stove knob"
    assert goal_command([req("activated", "b1")], {"b1": "big button"}, categories={"b1": "other"}) == \
        "Turn on the big button"  # a known category decides
    assert goal_command([req("activated", "the light switch")], {}) == "Press the light switch"
    assert goal_command([req("activated", "the lamp")], {}) == "Turn on the lamp"


EXTRA_FORMS = {
    "Take the pink brick out of the transparent box": req("inside", "o1", "o2", value=False),
    "Take the pink brick off the transparent box": req("on_top_of", "o1", "o2", value=False),
    "Turn off the stove": req("activated", "o4", value=False),
    "Move the pink brick to the transparent box": req("at_location", "o1", "o2"),
    "Lift the pink brick": req("lifted", "o1"),
    "Pick up the pink brick": req("in_gripper", "o1"),
}


def test_goal_command_uses_only_the_four_spec_forms_by_default():
    for r in EXTRA_FORMS.values():  # SPEC_V1_1 5 lists four forms; the others wait for sign-off
        assert goal_command([r], NAMES) == ""
    mixed = [req("at_location", "o1", "o2"), req("inside", "o1", "o2"), req("activated", "o4", value=False)]
    assert goal_command(mixed, NAMES) == "Put the pink brick in the transparent box"
    assert goal_command(mixed, NAMES, extra_forms=True) == \
        "Move the pink brick to the transparent box then put the pink brick in the transparent box then turn off the stove"


def test_goal_command_negations_extras_and_what_is_left_out():
    for text, r in EXTRA_FORMS.items():
        assert goal_command([r], NAMES, extra_forms=True) == text
    for left_out in (req("inside", "o1", "o2", value="unsure"), req("inside", "o1", "none"),
                     req("inside", "unsure", "o2"), req("inside", "o1", "unsure"), req("touching", "o1", "o2"),
                     req("unchanged", "o1"), req("other", "o1"), req("state", "o5", value="unsure")):
        assert goal_command([left_out], NAMES) == ""
        assert goal_command([left_out], NAMES, extra_forms=True) == ""
    # plain words (no inventory): a leading article is not doubled
    assert goal_command([req("inside", "the pink brick", "a clear box")], {}) == "Put the pink brick in the clear box"
    assert goal_command([req("inside", "o1", "o2", value="true")], NAMES) == \
        "Put the pink brick in the transparent box"  # raw string values read the same


# ------------------------------------------------------------------------------------------ state check, rendering
@pytest.mark.parametrize("text", [
    "Put the pink brick in the transparent box.", "put the brick in the box", "Place the cup on the plate",
    "Pick up the eye drops and put them in the basket", "Pick-up the cup", "Stack the red block on the blue one",
    "Open the drawer", "Close it", "Clean the table", "Empty the bowl into the sink", "Stack 3 cups", "Open",
    "Please put the brick in the box", "Do not move the cup", "Don't move the cup", "First, put the brick in the box",
    "Then press the button", "To put the brick in the box", "Never touch the stove", "Dance.", "Wave hello",
    "Draw three lines on the paper", "Fold the T-shirt", "Turn on the stove", "Toast the bread", "", "   "])
def test_commands_are_not_states(text):
    assert is_state_objective(text) is False


@pytest.mark.parametrize("text", [
    "The pink brick is inside the transparent box.", "the brick is in the box", "Pink brick inside the box",
    "Open drawer is empty", "Clean dishes are in the rack", "Stack of cups is on the tray", "Set of blocks is sorted",
    "Water is in the glass", "Light is on", "The person dances in place.", "No object has a required end state.",
    "A person waves at the camera", "Three strokes are on the paper", "The T-shirt is folded",
    # words that are also nouns start state sentences (the no_end_state clip's objective)
    "Dance performance by one person, no object end state", "Dance routine is finished",
    "Wave of greeting toward the camera", "Walk ends at the door", "Turn is complete", "Run of the robot is over",
    "Return of the cup to the tray is done", "Reach is limited to the table", "Spin sequence by two dancers"])
def test_statements_are_states(text):
    assert is_state_objective(text) is True


@pytest.mark.parametrize("text", [
    "Walk to the door", "Turn left", "Turn around", "Reach for the cup", "Return to the start", "Spin around twice",
    "Wave at the camera", "Dance with a partner", "Jump forward", "Clap twice", "Sit at the table", "Nod",
    "Run to the tree", "Drive to the garage", "Shake it"])
def test_words_that_are_also_nouns_are_commands_with_a_command_follower(text):
    assert is_state_objective(text) is False


def test_object_nouns_make_short_commands():
    nouns = {"drawer", "lid", "pink", "brick"}
    for text in ("Open drawer", "Close lid", "Open drawer and close lid", "Clean pink brick"):
        assert is_state_objective(text) is True  # without the nouns: a noun phrase, read as a state
        assert is_state_objective(text, nouns=nouns) is False
    for text in ("Open drawer is empty", "Clean pink brick is in the box", "Dance routine is finished",
                 "Close lid stays shut"):
        assert is_state_objective(text, nouns=nouns) is True  # a state verb keeps it a state
    from robolabel.layers.goal import objective_nouns

    got = objective_nouns(OBJECTS, [req("inside", "o1", "o2"), req("state", "the lid", value="closed")], NAMES)
    assert {"pink", "brick", "transparent", "box", "drawer", "lid", "red", "button", "stove"} <= got
    assert not got & {"the", "none", "is", "of"}


def test_a_headline_command_on_an_inventory_object_is_rerendered():
    data = answer("Open drawer", [raw("state", value="open", obj="o5")], target="o5", destination="none")
    goal = postprocess_goal_v11(data, clip(), objects=OBJECTS, segments=[], repairs=[], signal=False)
    assert goal["objective_rendered"] is True and goal["objective_text"] == "The drawer is open."
    dance = answer("Dance performance by one person, no object end state", [], has_end_state=False,
                   target="none", destination="none")
    kept = postprocess_goal_v11(dance, clip(), objects=OBJECTS, segments=[], repairs=[], signal=False)
    assert kept["objective_rendered"] is False
    assert kept["objective_text"] == "Dance performance by one person, no object end state"


def test_render_objective_is_a_state_sentence():
    one = render_objective([req("inside", "o1", "o2")], NAMES)
    assert one == "The pink brick is inside the transparent box."
    reqs = [req("inside", "o1", "o2"), req("on_top_of", "o5", "o2", value=False), req("state", "o4", value="off"),
            req("activated", "o3"), req("holding", "none", value=False, kind="robot_end_state"),
            req("lifted", "o1", status="incidental")]
    many = render_objective(reqs, OBJECTS)
    assert many == ("The pink brick is inside the transparent box, the drawer is not on top of the transparent box, "
                    "the stove is off and the red button is activated.")
    assert render_objective([req("inside", "unsure", "o2")], NAMES) == \
        "An unidentified object is inside the transparent box."
    assert render_objective([], NAMES) == NO_END_STATE_OBJECTIVE
    assert render_objective([req("gripper_open", "none", kind="robot_end_state")], NAMES) == NO_END_STATE_OBJECTIVE
    for text in (one, many, NO_END_STATE_OBJECTIVE):
        assert is_state_objective(text)
    assert render_objective(reqs, OBJECTS) == many  # deterministic


# ------------------------------------------------------------------------------------------ postprocess, no signal
def test_an_imperative_objective_is_rerendered_as_a_state():
    data = answer("Put the pink brick in the transparent box.", [raw("inside", ref="o2")])
    repairs: list[str] = []
    goal = postprocess_goal_v11(data, clip(), objects=OBJECTS, segments=[], repairs=repairs, signal=False)
    assert goal["objective_text"] == "The pink brick is inside the transparent box."
    assert goal["objective_rendered"] is True
    assert goal["goal_command"] == "Put the pink brick in the transparent box"
    assert goal["has_end_state"] is True and goal["episode_outcome"] == "success"
    assert any("starts with an imperative verb" in r and "rendered from the requirements" in r for r in repairs)
    kept = postprocess_goal_v11(answer("The pink brick is in the box.", [raw("inside", ref="o2")]), clip(),
                                objects=OBJECTS, segments=[], repairs=[], signal=False)
    assert kept["objective_text"] == "The pink brick is in the box." and kept["objective_rendered"] is False
    empty: list[str] = []
    g = postprocess_goal_v11(answer("", [raw("inside", ref="o2")]), clip(), objects=OBJECTS, segments=[],
                             repairs=empty, signal=False)
    assert g["objective_text"] == "The pink brick is inside the transparent box." and any("is empty" in r for r in empty)


def test_has_end_state_is_read_and_coerced():
    def has(v, **kw):
        repairs: list[str] = []
        data = answer("The person dances.", [], target="none", destination="none")
        if v is not ...:
            data["has_end_state"] = v
        else:
            del data["has_end_state"]
        g = postprocess_goal_v11(data, clip(), objects=[], segments=[], repairs=repairs, signal=False)
        return g["has_end_state"], repairs

    assert has(True) == (True, []) and has(False) == (False, [])
    value, repairs = has("false")
    assert value is False and repairs == ["goal: has_end_state 'false' read as False"]
    assert has(...)[0] is None and has(...)[1] == ["goal: has_end_state missing, left unknown"]
    assert has("maybe")[0] is None


def test_no_end_state_goal_keeps_invented_object_items_for_rule_13():
    data = answer("The person waves at the camera.", [raw("on_top_of", ref="o2")], has_end_state=False)
    goal = postprocess_goal_v11(data, clip(), objects=OBJECTS, segments=[], repairs=[], signal=False)
    assert goal["has_end_state"] is False
    assert [r["kind"] for r in goal["requirements"]] == ["object_end_state"]  # kept: L5 rule 13 reports it
    clean = postprocess_goal_v11(answer("The person waves at the camera.", [], has_end_state=False, target="none",
                                        destination="none"), clip(), objects=[], segments=[], repairs=[], signal=False)
    assert clean["requirements"] == [] and clean["goal_command"] == "" and clean["episode_outcome"] == "unknown"
    assert clean["objective_text"] == "The person waves at the camera."


def test_without_a_signal_the_l1_record_is_never_read():
    robot = [raw("holding", "false", "unknown", kind="robot_end_state", obj="none"),
             raw("gripper_open", "true", "true", kind="robot_end_state", obj="none", status="unsure", uk="perception")]
    data = answer("The pink brick is inside the transparent box.", [raw("inside", ref="o2"), *robot])
    repairs: list[str] = []
    goal = postprocess_goal_v11(data, clip(), l1=Hidden(), objects=OBJECTS, segments=[], repairs=repairs,
                                signal=False)
    reqs = goal["requirements"]
    assert len(reqs) == 3 and all(r.get("added_by") == "model" for r in reqs)  # no slot added from L1
    assert reqs[1]["achieved"] == "unknown" and reqs[2]["status"] == "unsure"  # the model's own values
    assert all(r["basis"] != "signal" for r in reqs)
    assert goal["episode_outcome"] == episode_outcome(reqs) == "unknown"
    assert not any("D1a" in r for r in repairs)
    assert episode_outcome_v11(reqs, Hidden(), False) == "unknown"
    req_, _ = goal_request_v11(clip(), camera=CAM, objects=OBJECTS, facts=[], context={}, reasoning=None,
                               l1=Hidden(), signal=False)
    text = req_.parts[0].text
    assert "No measurement from the robot is given" in text and "measured by the robot itself" not in text


def test_signal_true_needs_the_l1_record():
    with pytest.raises(ValueError):
        postprocess_goal_v11(answer("x", []), clip(), objects=[], segments=[], repairs=[], signal=True)
    with pytest.raises(ValueError):
        episode_outcome_v11([], None, True)
    with pytest.raises(ValueError):
        goal_request_v11(clip(), camera=CAM, objects=[], facts=[], context={}, reasoning=None, signal=True)


# ------------------------------------------------------------------------------------------ D1a
def test_d1a_decides_unknown_and_unseen_robot_items_from_l1():
    l1 = l1_record(holding=False, gripper=("gripper_open", "high"), withdrawn=True)
    robot = [raw("holding", "false", "unknown", kind="robot_end_state", obj="none"),
             raw("gripper_open", "true", "unknown", kind="robot_end_state", obj="none", status="unsure",
                 uk="perception"),
             raw("withdrawn", "true", "unknown", kind="robot_end_state", obj="none"),
             raw("at_home_pose", "true", "unknown", kind="robot_end_state", obj="none")]
    data = answer("The pink brick is inside the transparent box.", [raw("inside", ref="o2"), *robot])
    repairs: list[str] = []
    goal = postprocess_goal_v11(data, clip(), l1=l1, objects=OBJECTS, segments=[], repairs=repairs, signal=True)
    by = {r["predicate"]: r for r in goal["requirements"]}
    assert by["holding"]["achieved"] is True and by["holding"]["basis"] == "signal"
    assert by["gripper_open"]["achieved"] is True and by["gripper_open"]["status"] == "required"
    assert by["gripper_open"]["unsure_kind"] is None and by["gripper_open"]["basis"] == "signal"
    assert by["withdrawn"]["achieved"] is True and by["withdrawn"]["basis"] == "signal"
    assert by["at_home_pose"]["achieved"] == "unknown" and by["at_home_pose"]["basis"] == "observed"
    assert by["inside"]["basis"] == "observed"
    assert sum("decided by L1: achieved" in r and "D1a" in r for r in repairs) == 3
    assert any("at_home_pose" in r and "not decided by L1" in r for r in repairs)
    assert goal["episode_outcome"] == "unknown"  # at_home_pose stays unknown
    # without the undecidable item, the signal settles the outcome that v7 would leave unknown
    data2 = answer("The pink brick is inside the transparent box.", [raw("inside", ref="o2"), *robot[:3]])
    goal2 = postprocess_goal_v11(data2, clip(), l1=l1, objects=OBJECTS, segments=[], repairs=[], signal=True)
    assert goal2["episode_outcome"] == "success"
    v7_goal = postprocess_goal(copy.deepcopy(data2), clip_v7(), l1, OBJECTS, [], [])
    assert v7_goal["episode_outcome"] == "unknown"
    # episode_outcome_v11 applies D1a to undecided items too, and is idempotent on decided ones
    assert episode_outcome_v11(v7_goal["requirements"], l1, True) == "success"
    assert episode_outcome_v11(goal2["requirements"], l1, True) == "success"
    assert episode_outcome_v11(v7_goal["requirements"], l1, False) == "unknown"


def clip_v7() -> Episode:
    ep = clip()
    ep.extra["camera_order"] = [CAM]
    return ep


def test_d1a_contradicting_signal_and_what_l1_cannot_decide():
    held = l1_record(holding=True, gripper=("gripper_closed", "low"))
    # "holding the brick": L1 says something is held but not which object
    assert signal_achieved(req("holding", "none", "o1", True, kind="robot_end_state"), held) is None
    assert signal_achieved(req("holding", "none", "o1", True, kind="robot_end_state"), l1_record(holding=False)) is False
    assert signal_achieved(req("holding", "none", "none", True, kind="robot_end_state"), held) is True
    assert signal_achieved(req("holding", "none", "none", False, kind="robot_end_state"), held) is False
    # a low-confidence gripper state decides nothing; a high one decides open and closed claims
    assert signal_achieved(req("gripper_closed", "none", kind="robot_end_state"), held) is None
    open_high = l1_record(gripper=("gripper_open", "high"))
    assert signal_achieved(req("gripper_open", "none", kind="robot_end_state"), open_high) is True
    assert signal_achieved(req("gripper_closed", "none", kind="robot_end_state"), open_high) is False
    assert signal_achieved(req("gripper_open", "none", value="unsure", kind="robot_end_state"), open_high) is None
    assert signal_achieved(req("near_object", "none", kind="robot_end_state"), open_high) is None
    assert signal_achieved(req("inside", "o1", "o2"), open_high) is None  # object items are never decided
    # D1a writes the signal's answer, even when it says the robot item was not achieved
    data = answer("The pink brick is inside the transparent box.",
                  [raw("inside", ref="o2"), raw("holding", "false", "unknown", kind="robot_end_state", obj="none")])
    goal = postprocess_goal_v11(data, clip(), l1=held, objects=OBJECTS, segments=[], repairs=[], signal=True)
    h = next(r for r in goal["requirements"] if r["predicate"] == "holding" and r["added_by"] == "model")
    assert h["achieved"] is False and h["basis"] == "signal"
    assert goal["episode_outcome"] == "partial"


def test_mandatory_robot_slots_come_from_l1_only_with_a_signal():
    data = answer("The pink brick is inside the transparent box.", [raw("inside", ref="o2")])
    with_sig = postprocess_goal_v11(copy.deepcopy(data), clip(), l1=l1_record(), objects=OBJECTS, segments=[],
                                    repairs=[], signal=True)
    added = [r for r in with_sig["requirements"] if r["added_by"] == "postprocess"]
    assert [r["predicate"] for r in added] == ["holding", "gripper_open", "withdrawn"]
    assert all(r["basis"] == "signal" and r["status"] == "unsure" for r in added)
    without = postprocess_goal_v11(copy.deepcopy(data), clip(), l1=l1_record(), objects=OBJECTS, segments=[],
                                   repairs=[], signal=False)
    assert [r["kind"] for r in without["requirements"]] == ["object_end_state"]


# ------------------------------------------------------------------------------------------ G2 by attempt (SPEC 4)
def seg(start, end, phase, target, attempt, outcome, attempt_outcome=None, destination="none"):
    s = {"start_frame": start, "end_frame": end, "phase_class": phase, "phase_text": phase, "target": target,
         "destination": destination, "attempt_idx": attempt, "outcome": outcome,
         "failure_type": "missed_grasp" if outcome == "failed" else "none", "mistake": outcome == "failed"}
    if attempt_outcome is not None:
        s["attempt_outcome"] = attempt_outcome
    return s


def test_g2_rejects_requirements_on_objects_of_failed_attempts_only():
    """Under the v1.1 convention the approach of a failed attempt is a success, so the rule reads the attempt."""
    v11 = [seg(0, 19, "approach", "o4", 1, "success", "failed"), seg(20, 39, "grasp", "o4", 1, "failed", "failed"),
           seg(40, 59, "approach", "o1", 2, "success", "success"),
           seg(60, 99, "transport", "o1", 2, "success", "success", destination="o2")]
    data = answer("The pink brick is inside the transparent box.",
                  [raw("inside", ref="o2"), raw("unchanged", obj="o4")])
    repairs: list[str] = []
    goal = postprocess_goal_v11(copy.deepcopy(data), clip(), objects=OBJECTS, segments=v11, repairs=repairs,
                                signal=False)
    assert [r["object"] for r in goal["requirements"]] == ["o1"]
    assert any("o4 rejected" in r for r in repairs)
    # v7 reads each phase's outcome, so the successful approach would keep the o4 item
    v7_goal = postprocess_goal(copy.deepcopy(data), clip_v7(), {}, OBJECTS, v11, [])
    assert [r["object"] for r in v7_goal["requirements"]] == ["o1", "o4"]
    # old outputs without attempt_outcome: derived by the v7 rule, same result
    old = [{k: v for k, v in s.items() if k != "attempt_outcome"} for s in v11]
    old[0]["outcome"] = "failed"
    g_old = postprocess_goal_v11(copy.deepcopy(data), clip(), objects=OBJECTS, segments=old, repairs=[],
                                 signal=False)
    assert [r["object"] for r in g_old["requirements"]] == ["o1"]


# ------------------------------------------------------------------------------------------ prompt and schema
def test_goal_v8_schema_is_the_v7_schema_plus_has_end_state():
    g8 = v8.GOAL_SCHEMAS["goal_v8"]
    assert list(g8["properties"]) == ["objective_text", "has_end_state", "primary_target", "primary_destination",
                                      "requirements"]
    assert g8["properties"]["has_end_state"] == {"type": "boolean"}
    assert g8["required"] == list(g8["properties"]) and g8["additionalProperties"] is False
    rest = {k: v for k, v in g8["properties"].items() if k != "has_end_state"}
    assert rest == v7.SCHEMAS["goal"]["properties"]
    assert v8.GOAL_MAX_TOKENS == {"goal_v8": 5000}


def test_frozen_e1_prompt_state_is_unchanged():
    """tools/v11/run_e1.py prompt_state hashes the whole v8 SCHEMAS dict and the whole MAX_TOKENS dict, so the
    goal entries live in GOAL_SCHEMAS and GOAL_MAX_TOKENS and these two dicts keep exactly E1's entries."""
    for name, sha in FROZEN_PROMPTS.items():
        assert v8.prompt_sha256(name) == sha
    assert set(v8.SCHEMAS) == {"coarse_frames", "coarse_video", "crawl"}
    assert hashlib.sha256(json.dumps(v8.SCHEMAS, sort_keys=True).encode("utf-8")).hexdigest() == FROZEN_SCHEMAS
    assert dict(sorted(v8.MAX_TOKENS.items())) == {"coarse": 8000, "crawl": 2500}
    assert v8.VERSION == "v8-2026-09-27.1"


def test_goal_prompt_text_rules():
    text = v8.load_prompt("goal")
    low = text.lower()
    for banned in ("for example", "e.g.", "such as", "for instance", chr(0x2014), chr(0x2013)):
        assert banned not in low
    assert text.isascii()
    assert "has_end_state" in text and "never a command" in text
    assert "dance, waving or a gesture" in text and "no object_end_state item" in text
    assert "judge the robot items from the images alone" in text
    assert set(v8.prompt_sections("goal")) == {
        "intro", "task", "no_task", "objects", "no_objects", "facts", "robot_measured", "robot_not_measured",
        "target_inventory", "target_words", "destination_inventory", "destination_words", "refs_inventory",
        "refs_words", "rules"}


def test_goal_request_with_and_without_a_signal():
    ep = clip(task="put the {pink} brick in the box")
    l1 = l1_record(holding=False, attempts=[{"attempt_idx": 1, "outcome": "released", "hold_frame": 30,
                                             "opening_onset": 70, "closing_onset": 20, "event_frame": 70}])
    facts = [{"frame": 0, "camera": CAM, "visible": ["o1", "o2"], "partial": [], "in_gripper": "none",
              "relations": []},
             {"frame": 99, "camera": CAM, "visible": ["o1", "o2"], "partial": [], "in_gripper": "none",
              "relations": [{"subject": "o1", "relation": "inside", "object": "o2", "value": "true"}]}]
    ctx = {"arm": "main", "episode_key": "C/robot", "bucket": "debug", "model_key": "luna"}
    r_off, manifest = goal_request_v11(ep, camera=CAM, objects=OBJECTS, facts=facts, context=ctx, reasoning=None)
    r_on, _ = goal_request_v11(ep, camera=CAM, objects=OBJECTS, facts=facts, context=ctx, reasoning=None, l1=l1,
                               signal=True)
    for r in (r_off, r_on):
        assert r.step == "goal" and r.schema == v8.GOAL_SCHEMAS["goal_v8"] and r.schema_name == "goal_v8"
        assert r.max_tokens == 5000 and r.system == v8.load_prompt("system").strip()
        assert r.context["frame_indices"] == [0, 99] and r.context["cameras"] == ["video"]
        assert r.context["arm"] == "main"
        text = r.parts[0].text
        assert "{" not in text.replace("{pink}", "") and "put the {pink} brick in the box" in text
        assert "camera video" in text and "o1: pink brick (block)" in text and "pink brick inside transparent box" in text
        assert [p.text for p in r.parts[1::2]] == ["frame 0 of 100 (0.00 s), camera video",
                                                   "frame 99 of 100 (4.95 s), camera video"]
        assert all(isinstance(p, ImagePart) for p in r.parts[2::2])
    assert manifest == [{"frame": 0, "camera": CAM}, {"frame": 99, "camera": CAM}]
    off, on = r_off.parts[0].text, r_on.parts[0].text
    assert "No measurement from the robot is given" in off and "attempt 1" not in off
    assert "measured by the robot itself" in on and "holding an object at the last frame: no" in on
    assert "attempt 1: held from 30, released at 70" in on
    # no inventory: plain words; no task: said so
    bare = Episode(episode_id="C/x", num_frames=100, fps=20.0, task="", get_frame=ep.get_frame, camera_key=CAM,
                   extra={"cameras": {CAM: ep.get_frame}})
    r_bare, _ = goal_request_v11(bare, camera=CAM, objects=[], facts=[], context={}, reasoning=None)
    t = r_bare.parts[0].text
    assert "no task description" in t and "no object list" in t and "in a few plain words" in t
    assert "(no scene facts)" in t
    # byte-identical on repeat
    again, _ = goal_request_v11(ep, camera=CAM, objects=OBJECTS, facts=facts, context=ctx, reasoning=None)
    assert [p.text if isinstance(p, TextPart) else p.jpeg for p in again.parts] == \
        [p.text if isinstance(p, TextPart) else p.jpeg for p in r_off.parts]
