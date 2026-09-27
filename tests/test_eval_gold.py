"""Tests for gold v2 (schema, validator, export wrapper, merge) and the draft lexicons."""

from __future__ import annotations

import copy
import json

import pytest

from robolabel.eval.gold_v2 import (
    EXPORT_SCHEMA,
    SCHEMA_VERSION,
    dump_gold,
    load_gold,
    load_schema,
    merge_episodes,
    save_gold,
    validate_gold,
    validate_gold_export,
    warnings_for_episode,
)
from robolabel.eval.lexicon import (
    coarse_groups,
    compiled_goal,
    load_phase_lexicon,
    load_predicate_lexicon,
    map_phase,
    map_predicate,
    object_categorizer,
    object_namer,
    relation_for_category,
    relation_words,
)

UP = "observation.images.up"
SIDE = "observation.images.side"


def _seg(idx, start, end, phase, text, target, destination=None, quality="sharp", coarse=0, **extra):
    seg = {"segment_idx": idx, "start_frame": start, "end_frame": end, "phase_class": phase,
           "phase_text": text, "target": target, "destination": destination, "attempt_idx": 1,
           "outcome": "success", "failure_type": None, "end_boundary_quality": quality, "coarse_idx": coarse}
    seg.update(extra)
    return seg


def _req(rid, kind, obj, predicate, ref, value, status, unsure_kind, basis, frame, camera, vis, reason=""):
    return {"req_id": rid, "kind": kind, "object": obj, "predicate": predicate, "ref_object": ref,
            "value": value, "status": status, "unsure_kind": unsure_kind, "basis": basis, "achieved": True,
            "deciding_frame": frame, "deciding_camera": camera, "visibility": vis, "reason": reason}


def spec_example() -> dict:
    """The spec 3.2 example, extended to cover [0, 244] with five valid segments."""
    episode = {
        "episode_key": "F1/12",
        "episode_index": 12,
        "split": "dev",
        "pass": 1,
        "annotator_id": "A1",
        "blind": True,
        "tool": "robolabel-gold-ui 0.1",
        "active_seconds": 512,
        "num_frames": 245,
        "cameras": [UP, SIDE],
        "task_string": "pink lego brick into the transparent box",
        "objects": [
            {"object_id": "o1", "name": "pink lego brick", "aliases": ["pink brick", "lego brick", "brick"],
             "category": "block", "first_frame_point": {"camera": UP, "xy": [0.41, 0.62]}},
            {"object_id": "o2", "name": "transparent box", "aliases": ["box", "clear box", "container"],
             "category": "container", "first_frame_point": {"camera": UP, "xy": [0.70, 0.35]}},
        ],
        "primary_target": "o1",
        "primary_destination": "o2",
        "segments": [
            _seg(0, 0, 81, "approach", "reach toward the pink brick", "o1", coarse=0),
            _seg(1, 82, 97, "grasp", "close on the pink brick", "o1", quality="fuzzy", coarse=0),
            _seg(2, 98, 180, "transport", "carry the pink brick to the box", "o1", "o2", coarse=1),
            _seg(3, 181, 215, "release", "open over the box", "o1", "o2", coarse=1),
            _seg(4, 216, 244, "retract", "move the arm away", None, quality=None, coarse=2),
        ],
        "coarse_subtasks": [
            {"coarse_idx": 0, "start_frame": 0, "end_frame": 97, "text": "pick up the pink brick",
             "target": "o1", "destination": None, "mistake": False},
            {"coarse_idx": 1, "start_frame": 98, "end_frame": 215, "text": "put the pink brick in the box",
             "target": "o1", "destination": "o2", "mistake": False},
            {"coarse_idx": 2, "start_frame": 216, "end_frame": 244, "text": "move the arm away",
             "target": None, "destination": None, "mistake": False},
        ],
        "failed_attempts": [],
        "goal": {
            "objective_text": "the pink brick is inside the transparent box",
            "requirements": [
                _req("r1", "object_end_state", "o1", "inside", "o2", True, "required", None, "task_string",
                     240, UP, {UP: "visible", SIDE: "partial"}),
                _req("r2", "robot_end_state", None, "holding", None, False, "required", None,
                     "physical_necessity", 240, SIDE, {UP: "partial", SIDE: "visible"},
                     "a placed object must be released"),
                _req("r3", "robot_end_state", None, "withdrawn", None, True, "unsure", "intent", "observed",
                     244, SIDE, {UP: "visible", SIDE: "visible"},
                     "arm leaves the box in this demo; the task string does not say whether it must"),
                _req("r4", "robot_end_state", None, "gripper_open", None, True, "incidental", None,
                     "observed", 244, SIDE, {}),
            ],
        },
        "episode_outcome": "success",
        "quality": 5,
        "hard_tags": [],
        "notes": "",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "family": "F1",
        "dataset": {"repo_id": "lerobot/svla_so101_pickplace", "revision": "abc123", "fps": 30},
        "guide_version": "gold_guide_v1.0",
        "episodes": [episode],
    }


def _ep(doc: dict) -> dict:
    return doc["episodes"][0]


# --------------------------------------------------------------------------- #
# Schema and validator
# --------------------------------------------------------------------------- #
def test_schema_is_a_valid_draft_2020_12_schema():
    from jsonschema import Draft202012Validator

    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["schema_version"] == {"const": "robolabel/gold/v2"}


def test_schema_enums_match_the_lexicons():
    defs = load_schema()["$defs"]
    classes = [c for c in defs["phase_class_or_null"]["enum"] if c is not None]
    assert classes == load_phase_lexicon()["classes"]
    predicates = [p["name"] for p in load_predicate_lexicon()["predicates"]]
    assert defs["predicate"]["enum"] == [*predicates, "other"]


def test_spec_example_validates_without_warnings():
    doc = spec_example()
    assert validate_gold(doc) == []
    assert warnings_for_episode(_ep(doc)) == []


def test_gap_between_segments_fails_with_a_clear_message():
    doc = spec_example()
    _ep(doc)["segments"][1]["start_frame"] = 83
    errors = validate_gold(doc)
    assert errors == ["episodes[0] F1/12 pass 1: segments: gap between segment 0 (ends at 81) and segment 1 "
                      "(starts at 83); frames 82 to 82 are not covered"]


def test_overlap_unsorted_and_coverage_errors():
    doc = spec_example()
    _ep(doc)["segments"][2]["start_frame"] = 90
    assert any("overlaps segment 1" in e for e in validate_gold(doc))

    doc = spec_example()
    _ep(doc)["segments"][0]["start_frame"] = 3
    assert any("first segment starts at 3, expected 0" in e for e in validate_gold(doc))

    doc = spec_example()
    _ep(doc)["segments"][-1]["end_frame"] = 243
    assert any("last segment ends at 243, expected num_frames - 1 = 244" in e for e in validate_gold(doc))

    doc = spec_example()
    segs = _ep(doc)["segments"]
    segs[0], segs[1] = segs[1], segs[0]
    assert any("segments not sorted" in e for e in validate_gold(doc))


def test_last_segment_must_have_null_boundary_quality():
    doc = spec_example()
    _ep(doc)["segments"][-1]["end_boundary_quality"] = "sharp"
    assert validate_gold(doc) == ["episodes[0] F1/12 pass 1: segment 4: the last segment must have "
                                  "end_boundary_quality null (got 'sharp')"]


def test_unsure_without_kind_fails():
    doc = spec_example()
    _ep(doc)["goal"]["requirements"][2]["unsure_kind"] = None
    assert validate_gold(doc) == ["episodes[0] F1/12 pass 1: requirement r3: status unsure needs "
                                  "unsure_kind perception or intent"]


def test_unsure_kind_without_unsure_status_fails():
    doc = spec_example()
    _ep(doc)["goal"]["requirements"][0]["unsure_kind"] = "perception"
    assert validate_gold(doc) == ["episodes[0] F1/12 pass 1: requirement r1: unsure_kind 'perception' is "
                                  "set but status is 'required'"]


def test_unknown_predicate_fails_in_the_schema():
    doc = spec_example()
    _ep(doc)["goal"]["requirements"][0]["predicate"] = "inside_of"
    errors = validate_gold(doc)
    assert len(errors) == 1
    assert errors[0].startswith("schema: $.episodes[0].goal.requirements[0].predicate: 'inside_of' is not one of")


def test_bad_visibility_camera_fails():
    doc = spec_example()
    _ep(doc)["goal"]["requirements"][0]["visibility"] = {UP: "visible", "observation.images.front": "partial"}
    errors = validate_gold(doc)
    assert errors == ["episodes[0] F1/12 pass 1: requirement r1: visibility camera 'observation.images.front' "
                      f"is not an episode camera ({SIDE}, {UP})"]


def test_bad_visibility_class_fails_in_the_schema():
    doc = spec_example()
    _ep(doc)["goal"]["requirements"][0]["visibility"][UP] = "hidden"
    errors = validate_gold(doc)
    assert errors == ['schema: $.episodes[0].goal.requirements[0].visibility["observation.images.up"]: '
                      "'hidden' is not one of ['visible', 'partial', 'not_visible']"]


def test_typo_key_and_bad_enums_are_caught():
    doc = spec_example()
    seg = _ep(doc)["segments"][0]
    seg["phase_clas"] = seg.pop("phase_class")
    _ep(doc)["objects"][0]["category"] = "cube"
    _ep(doc)["goal"]["requirements"][1]["achieved"] = 1
    errors = validate_gold(doc)
    assert any("objects[0].category: 'cube' is not one of" in e for e in errors)
    assert any("requirements[1].achieved: 1 is not one of" in e for e in errors)
    assert any("segments[0]: 'phase_class' is a required property" in e for e in errors)
    assert any("segments[0]: Additional properties are not allowed ('phase_clas' was unexpected)" in e
               for e in errors)
    assert all(e.startswith("schema: ") for e in errors)  # no semantic checks on a broken episode
    assert validate_gold(doc) == errors


def test_object_references_and_ids():
    doc = spec_example()
    ep = _ep(doc)
    ep["segments"][0]["target"] = "o9"
    ep["objects"][1]["object_id"] = "o1"
    ep["goal"]["requirements"][0]["ref_object"] = "o7"
    errors = validate_gold(doc)
    assert "episodes[0] F1/12 pass 1: objects: duplicate object_id 'o1'" in errors
    assert "episodes[0] F1/12 pass 1: segment 0: target 'o9' is not an object_id of this episode" in errors
    assert "episodes[0] F1/12 pass 1: requirement r1: ref_object 'o7' is not an object_id of this episode" \
        in errors


def test_region_names_are_allowed_where_a_region_can_be():
    doc = spec_example()
    req = _ep(doc)["goal"]["requirements"][0]
    req["predicate"], req["ref_object"] = "at_location", "left side of the table"
    assert validate_gold(doc) == []
    _ep(doc)["segments"][1]["target"] = "brick"
    assert validate_gold(doc) == ["episodes[0] F1/12 pass 1: segment 1: target 'brick' must be an object_id "
                                  "of this episode or null"]


def test_robot_requirement_may_have_null_object_but_object_requirement_may_not():
    doc = spec_example()
    _ep(doc)["goal"]["requirements"][0]["object"] = None
    assert validate_gold(doc) == ["episodes[0] F1/12 pass 1: requirement r1: an object_end_state requirement "
                                  "needs an object"]


def test_duplicate_req_id_episode_key_and_value_type():
    doc = spec_example()
    ep = _ep(doc)
    ep["goal"]["requirements"][1]["req_id"] = "r1"
    ep["episode_key"] = "F1/13"
    ep["goal"]["requirements"][0]["value"] = "yes"
    errors = validate_gold(doc)
    assert "episodes[0] F1/13 pass 1: episode_key 'F1/13' is not 'F1/12'" in errors
    assert "episodes[0] F1/13 pass 1: requirements: duplicate req_id 'r1'" in errors
    assert "episodes[0] F1/13 pass 1: requirement r1: predicate inside takes a boolean value, got 'yes'" in errors


def test_failed_attempt_span_inside_episode_and_camera_known():
    doc = spec_example()
    ep = _ep(doc)
    ep["failed_attempts"] = [{"span": [82, 300], "failure_type": "slip", "object": "o1", "evident_frame": 95,
                              "evident_camera": "observation.images.wrist", "recovery_start": None}]
    ep["hard_tags"] = ["failed_attempt"]
    errors = validate_gold(doc)
    assert errors == [
        "episodes[0] F1/12 pass 1: failed attempt 0: span [82, 300] is outside [0, 244]",
        "episodes[0] F1/12 pass 1: failed attempt 0: evident_camera 'observation.images.wrist' is not an "
        "episode camera",
    ]


def test_coarse_subtasks_must_be_contiguous_and_are_optional():
    doc = spec_example()
    _ep(doc)["coarse_subtasks"][1]["start_frame"] = 100
    assert validate_gold(doc) == ["episodes[0] F1/12 pass 1: coarse subtasks: coarse subtask 1 starts at 100, "
                                  "expected 98 (right after coarse subtask 0)"]
    doc = spec_example()
    del _ep(doc)["coarse_subtasks"]
    for seg in _ep(doc)["segments"]:
        del seg["coarse_idx"]
    assert validate_gold(doc) == []


def test_duplicate_episode_and_pass_fails():
    doc = spec_example()
    doc["episodes"].append(copy.deepcopy(_ep(doc)))
    assert validate_gold(doc) == ["episodes[1] F1/12 pass 1: duplicate of episodes[0] (same episode_key and pass)"]


def test_unfilled_phase_is_a_warning_not_an_error():
    doc = spec_example()
    ep = _ep(doc)
    ep["segments"][1]["phase_class"] = None
    ep["primary_target"] = None
    ep["goal"]["requirements"] = ep["goal"]["requirements"][:1]
    assert validate_gold(doc) == []
    assert warnings_for_episode(ep) == [
        "segment 1: missing phase_class",
        "missing primary_target",
        "missing robot ending slot holding: holding",
        "missing robot ending slot gripper: gripper_open or gripper_closed",
        "missing robot ending slot position: withdrawn or at_home_pose or near_object",
    ]


def test_validate_is_deterministic_and_non_dict_input():
    doc = spec_example()
    ep = _ep(doc)
    ep["segments"][3]["phase_class"] = "drop"
    ep["split"] = "test"
    ep["quality"] = 7
    first = validate_gold(doc)
    assert first == validate_gold(copy.deepcopy(doc))
    assert [e.split(":")[1] for e in first] == [" $.episodes[0].quality", " $.episodes[0].segments[3].phase_class",
                                                " $.episodes[0].split"]
    assert validate_gold([]) == ["$: a gold v2 document must be a JSON object"]


# --------------------------------------------------------------------------- #
# Export wrapper, dump, merge
# --------------------------------------------------------------------------- #
def _export(doc: dict) -> dict:
    return {"schema": EXPORT_SCHEMA, "run_id": "sweep-20260927T0641Z", "saved_at": "2026-09-28T08:00:00Z",
            "families": {"F1": doc}}


def test_export_wrapper_validates():
    assert validate_gold_export(_export(spec_example())) == []


def test_export_wrapper_errors():
    bad = _export(spec_example())
    bad["schema"] = "robolabel/gold-export/v0"
    del bad["run_id"]
    _ep(bad["families"]["F1"])["goal"]["requirements"][2]["unsure_kind"] = None
    bad["families"]["F3"] = bad["families"].pop("F1")
    errors = validate_gold_export(bad)
    assert errors[0] == "$.schema: expected 'robolabel/gold-export/v1', got 'robolabel/gold-export/v0'"
    assert errors[1] == "$.run_id: a non-empty string is required"
    assert "families.F3: the document's family is 'F1'" in errors
    assert "families.F3: episodes[0] F1/12 pass 1: requirement r3: status unsure needs unsure_kind " \
           "perception or intent" in errors
    assert validate_gold_export([]) == ["$: a gold export must be a JSON object"]


def test_dump_gold_is_canonical_and_round_trips(tmp_path):
    doc = spec_example()
    _ep(doc)["objects"][0]["first_frame_point"]["xy"] = [0.123456789, 1 / 3]
    _ep(doc)["task_string"] = "pink brick into the box (caf\u00e9)"
    text = dump_gold(doc)
    assert text.endswith("}\n") and "\r" not in text
    assert text == dump_gold(copy.deepcopy(doc))
    assert '"episodes": [' in text and text.index('"dataset"') < text.index('"episodes"') < text.index('"family"')
    assert "0.123457" in text and "0.333333" in text and "caf\u00e9" in text
    path = save_gold(doc, tmp_path / "gold" / "F1.gold.json")
    assert path.read_bytes() == text.encode("utf-8")
    loaded = load_gold(path)
    assert validate_gold(loaded) == []
    assert dump_gold(loaded) == text


def test_merge_episodes_replaces_same_key_and_pass():
    existing = spec_example()
    other = copy.deepcopy(_ep(existing))
    other.update(episode_key="F1/3", episode_index=3)
    existing["episodes"].insert(0, other)
    new = spec_example()
    replacement = _ep(new)
    replacement["notes"] = "second look"
    second_pass = copy.deepcopy(replacement)
    second_pass["pass"] = 2
    fifth = copy.deepcopy(replacement)
    fifth.update(episode_key="F1/5", episode_index=5)
    new["episodes"] = [second_pass, fifth, replacement]
    new["guide_version"] = "pilot (gold_guide_v1.0 draft)"
    before = dump_gold(existing)

    merged = merge_episodes(existing, new)
    keys = [(e["episode_key"], e["pass"]) for e in merged["episodes"]]
    assert keys == [("F1/3", 1), ("F1/5", 1), ("F1/12", 1), ("F1/12", 2)]
    assert merged["episodes"][2]["notes"] == "second look"
    assert merged["guide_version"] == "pilot (gold_guide_v1.0 draft)"
    assert validate_gold(merged) == []
    assert dump_gold(existing) == before  # inputs untouched
    assert merge_episodes(None, new)["episodes"][0]["episode_key"] == "F1/5"


def test_merge_episodes_refuses_mixed_family_or_revision():
    existing, new = spec_example(), spec_example()
    new["family"] = "F3"
    with pytest.raises(ValueError, match="family"):
        merge_episodes(existing, new)
    new = spec_example()
    new["dataset"]["revision"] = "def456"
    with pytest.raises(ValueError, match="revision"):
        merge_episodes(existing, new)


# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #
def test_lexicon_files_are_drafts_with_versions():
    phase, pred = load_phase_lexicon(), load_predicate_lexicon()
    assert phase["status"] == "draft" and phase["version"] == "phase_lexicon_v1"
    assert pred["status"] == "draft" and pred["version"] == "predicate_lexicon_v1"
    assert "Appendix A" in phase["source"] and "Appendix B" in pred["source"]
    phase["classes"].append("mutated")
    assert "mutated" not in load_phase_lexicon()["classes"]  # callers get copies


@pytest.mark.parametrize("text, expected", [
    ("open gripper", "release"), ("open the gripper", "release"), ("open fingers", "release"),
    ("let go", "release"), ("close gripper", "grasp"), ("close the gripper", "grasp"),
    ("close fingers", "grasp"), ("move away", "retract"), ("go home", "retract"), ("back away", "retract"),
    ("return to home", "retract"), ("push button", "press"), ("push the button", "press"),
])
def test_map_phase_multi_word_patterns(text, expected):
    assert map_phase(text) == expected
    assert map_phase(f"then {text.upper()} slowly") == expected


@pytest.mark.parametrize("text, expected", [
    ("pick and place", "release"),          # release (2) outranks grasp (3)
    ("move to the brick", "transport"),
    ("reach toward", "approach"),
    ("release-place", "release"),
    ("idle", "unmapped"),
    ("", "unmapped"),
    (None, "unmapped"),
    ("grasp and lift", "grasp"),            # grasp (3) outranks transport (14)
    ("retract and release", "retract"),
    ("go to the bowl", "approach"),
    ("toggle the switch", "press"),
    ("shut the drawer", "close"),
    ("turn the knob", "rotate"),
    ("drag the cloth", "pull"),
    ("slide the block", "push"),
    ("wipe the table", "wipe"),
    ("fold the towel", "fold"),
    ("insert the peg", "insert"),
    ("tilt the cup", "pour"),
    ("lower the brick", "transport"),
    ("put down the cube", "release"),
])
def test_map_phase_keyword_priority(text, expected):
    assert map_phase(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("picks up the cube", "grasp"), ("picking", "grasp"), ("picked", "grasp"),
    ("placing the brick", "release"), ("placed", "release"), ("dropped it", "release"),
    ("gripping", "grasp"), ("carries the cup", "transport"), ("opening the gripper", "release"),
    ("closing the gripper", "grasp"), ("moving away", "retract"), ("letting go", "release"),
    ("pushes the button", "press"),
])
def test_map_phase_inflected_forms(text, expected):
    assert map_phase(text) == expected


def test_map_phase_whole_words_only_and_exact_names():
    assert map_phase("reopen") == "unmapped"         # "open" is not a whole word here
    assert map_phase("gripper") == "unmapped"        # "grip" is not a whole word here
    assert map_phase("homework") == "unmapped"
    for cls in load_phase_lexicon()["classes"]:
        assert map_phase(cls) == cls
    for legacy in ["approach", "grasp", "transport", "retract", "other"]:
        assert map_phase(legacy) == legacy
    assert map_phase("Release_Place") == "release"


@pytest.mark.parametrize("text, expected", [
    ("inside", "inside"), ("in the box", "inside"), ("into", "inside"), ("inside of", "inside"),
    ("on_top_of", "on_top_of"), ("on top of the plate", "on_top_of"), ("stacked on", "on_top_of"),
    ("onto", "on_top_of"), ("touching", "touching"), ("in contact with", "touching"),
    ("in the gripper", "in_gripper"), ("held", "in_gripper"), ("off the table", "lifted"),
    ("next to the bowl", "at_location"), ("at", "at_location"), ("untouched", "unchanged"),
    ("switched on", "activated"), ("lit", "activated"), ("the drawer is open", "state"),
    ("position 3", "state"), ("holding nothing", "holding"), ("carrying", "holding"),
    ("gripper open", "gripper_open"), ("fingers open", "gripper_open"), ("gripper closed", "gripper_closed"),
    ("away from the tray", "withdrawn"), ("rest pose", "at_home_pose"), ("at the object", "near_object"),
    ("above", "near_object"), ("pen up", "tool_lifted"), ("tool off the surface", "tool_lifted"),
    ("flying", "other"), ("", "other"), (None, "other"), ("other", "other"),
])
def test_map_predicate(text, expected):
    assert map_predicate(text) == expected


def test_relations():
    for cat in ["container", "bowl", "basket", "box", "cup", "bin", "drawer"]:
        assert relation_for_category(cat) == "inside"
    for cat in ["plate", "tray", "block", "cloth", "shelf", "table", "board"]:
        assert relation_for_category(cat) == "on_top_of"
    assert relation_for_category("Bowl") == "inside"
    assert relation_for_category("bottle") == "at_location"
    assert relation_for_category(None) == "at_location"
    assert [relation_words(p) for p in ["inside", "on_top_of", "at_location"]] == ["in", "on", "at"]
    assert relation_words("holding") == "at"


# --------------------------------------------------------------------------- #
# Coarse grouping and compiled goal
# --------------------------------------------------------------------------- #
CUBE_OBJECTS = [
    {"object_id": "o1", "name": "red cube", "aliases": ["cube"], "category": "block"},
    {"object_id": "o2", "name": "tray", "aliases": [], "category": "container"},
]


def test_coarse_groups_worked_example_slip_and_regrasp():
    segs = [
        {"start_frame": 0, "end_frame": 40, "phase_class": "approach", "target": "o1", "outcome": "success",
         "attempt_idx": 1},
        {"start_frame": 41, "end_frame": 60, "phase_class": "grasp", "target": "o1", "outcome": "failed",
         "failure_type": "slip", "attempt_idx": 1},
        {"start_frame": 61, "end_frame": 90, "phase_class": "approach", "target": "o1", "outcome": "success",
         "attempt_idx": 2},
        {"start_frame": 91, "end_frame": 110, "phase_class": "grasp", "target": "o1", "outcome": "success",
         "attempt_idx": 2},
        {"start_frame": 111, "end_frame": 170, "phase_class": "transport", "target": "o1", "destination": "o2",
         "outcome": "success", "attempt_idx": 2},
        {"start_frame": 171, "end_frame": 199, "phase_class": "release", "target": "o1", "destination": "o2",
         "outcome": "success", "attempt_idx": 2},
    ]
    groups = coarse_groups(segs, object_namer(CUBE_OBJECTS), category_of=object_categorizer(CUBE_OBJECTS))
    assert [g["text"] for g in groups] == ["pick up the red cube", "pick up the red cube",
                                           "put the red cube in the tray"]
    assert [g["mistake"] for g in groups] == [True, False, False]
    assert [(g["start_frame"], g["end_frame"]) for g in groups] == [(0, 60), (61, 110), (111, 199)]
    assert [g["segment_indices"] for g in groups] == [[0, 1], [2, 3], [4, 5]]
    assert [g["phase_classes"] for g in groups] == [["approach", "grasp"], ["approach", "grasp"],
                                                    ["transport", "release"]]
    assert [(g["target"], g["destination"]) for g in groups] == [("o1", None), ("o1", None), ("o1", "o2")]
    assert [g["coarse_idx"] for g in groups] == [0, 1, 2]


def test_coarse_groups_press_pour_retract_and_aliases():
    press = [
        {"start": 0, "end": 30, "phase_class": "approach", "target_name": "green button", "outcome": "success"},
        {"start": 31, "end": 40, "phase_class": "press", "target_name": "green button", "outcome": "failed",
         "failure_type": "press_no_effect"},
        {"start": 41, "end": 60, "phase_class": "retract", "target_name": None, "outcome": "success"},
        {"start": 61, "end": 80, "phase_class": "approach", "target_name": "green button", "outcome": "success"},
        {"start": 81, "end": 90, "phase_class": "press", "target_name": "green button", "outcome": "success"},
    ]
    groups = coarse_groups(press)
    assert [(g["text"], g["mistake"]) for g in groups] == [
        ("press the green button", True), ("move the arm away", False), ("press the green button", False)]

    pour = [
        {"start_frame": 0, "end_frame": 9, "phase_class": "reach for the cup", "target": "cup"},
        {"start_frame": 10, "end_frame": 19, "phase_class": "grasp", "target": "cup"},
        {"start_frame": 20, "end_frame": 29, "phase_class": "transport", "target": "cup", "destination": "bowl"},
        {"start_frame": 30, "end_frame": 39, "phase_class": "pour", "target": "cup", "destination": "bowl"},
        {"start_frame": 40, "end_frame": 49, "phase_class": "release", "target": "cup", "destination": "bowl"},
        {"start_frame": 50, "end_frame": 59, "phase_class": "wipe", "target": None},
    ]
    groups = coarse_groups(pour)
    assert [g["text"] for g in groups] == ["pick up the cup", "pour the cup into the bowl", "wipe the object"]
    assert groups[1]["segment_indices"] == [2, 3, 4]


def test_coarse_put_relation_from_category_or_name():
    segs = [
        {"start_frame": 0, "end_frame": 9, "phase_class": "transport", "target": "o1", "destination": "o2"},
        {"start_frame": 10, "end_frame": 19, "phase_class": "release", "target": "o1", "destination": "o2"},
    ]
    objects = [{"object_id": "o1", "name": "red cube", "category": "block"},
               {"object_id": "o2", "name": "blue plate", "category": "plate"}]
    assert coarse_groups(segs, object_namer(objects), category_of=object_categorizer(objects))[0]["text"] == \
        "put the red cube on the blue plate"
    # No category: a category word in the name decides, else at_location.
    assert coarse_groups(segs, {"o1": "red cube", "o2": "wooden tray"}.get)[0]["text"] == \
        "put the red cube on the wooden tray"
    assert coarse_groups(segs, {"o1": "red cube"}.get)[0]["text"] == "put the red cube at the object"


def test_compiled_goal_from_segments():
    objects = [{"object_id": "o1", "name": "pink lego brick", "aliases": ["brick"], "category": "block"},
               {"object_id": "o2", "name": "transparent box", "aliases": ["box"], "category": "container"}]
    doc = spec_example()
    goal = compiled_goal(_ep(doc)["segments"], objects)
    assert goal["primary_target"] == "o1" and goal["primary_destination"] == "o2"
    assert goal["basis"] == "compiled"
    assert goal["objective_text"] == "the pink lego brick is inside the transparent box and the arm is withdrawn"
    reqs = goal["requirements"]
    assert [(r["req_id"], r["kind"], r["object"], r["predicate"], r["ref_object"], r["value"]) for r in reqs] == [
        ("r1", "object_end_state", "o1", "inside", "o2", True),
        ("r2", "robot_end_state", None, "withdrawn", None, True),
    ]
    assert all(r["status"] == "required" and r["unsure_kind"] is None and r["basis"] == "compiled" for r in reqs)
    assert compiled_goal(_ep(doc)["segments"], objects) == goal  # deterministic


def test_compiled_goal_legacy_segments_and_no_retract():
    legacy = [
        {"start": 0, "end": 20, "phase": "approach", "target": "red cube"},
        {"start": 21, "end": 40, "phase": "grasp", "target": "red cube"},
        {"start": 41, "end": 80, "phase": "transport", "target": "red cube"},
        {"start": 81, "end": 99, "phase": "release-place", "target": "the plate"},
    ]
    goal = compiled_goal(legacy, [])
    assert goal["primary_target"] == "red cube" and goal["primary_destination"] == "the plate"
    assert [(r["predicate"], r["ref_object"]) for r in goal["requirements"]] == [("on_top_of", "the plate")]
    unknown = copy.deepcopy(legacy)
    unknown[-1]["target"] = "somewhere"
    assert compiled_goal(unknown)["requirements"][0]["predicate"] == "at_location"
    assert compiled_goal([]) == {"objective_text": "", "primary_target": None, "primary_destination": None,
                                 "requirements": [], "basis": "compiled"}


# --------------------------------------------------------------------------- #
# Verifier additions: empty inputs, single segments, ties, strict references
# --------------------------------------------------------------------------- #
def test_empty_and_degenerate_documents():
    doc = spec_example()
    doc["episodes"] = []
    assert validate_gold(doc) == []
    doc = spec_example()
    _ep(doc)["segments"] = []
    assert validate_gold(doc) == ["schema: $.episodes[0].segments: [] should be non-empty"]
    doc = spec_example()
    del doc["episodes"]
    assert validate_gold(doc) == ["schema: $: 'episodes' is a required property"]
    export = _export(spec_example())
    export["families"] = {}
    assert validate_gold_export(export) == []


def test_single_frame_single_segment_episode_validates():
    doc = spec_example()
    ep = _ep(doc)
    ep["num_frames"] = 1
    ep["segments"] = [_seg(0, 0, 0, "approach", "reach toward the pink brick", "o1", quality=None)]
    ep["coarse_subtasks"] = [{"coarse_idx": 0, "start_frame": 0, "end_frame": 0, "text": "approach the pink brick",
                              "target": "o1", "destination": None, "mistake": False}]
    for req in ep["goal"]["requirements"]:
        req["deciding_frame"] = 0
    assert validate_gold(doc) == []
    ep["segments"][0]["end_frame"] = 1
    assert validate_gold(doc) == ["episodes[0] F1/12 pass 1: segments: the last segment ends at 1, "
                                  "expected num_frames - 1 = 0"]


def test_frames_and_indices_must_be_json_integers():
    doc = spec_example()
    _ep(doc)["segments"][0]["start_frame"] = 0.0
    _ep(doc)["episode_index"] = 12.0
    _ep(doc)["quality"] = 5.0
    assert validate_gold(doc) == [
        "schema: $.episodes[0].episode_index: 12.0 is not of type 'integer'",
        "schema: $.episodes[0].quality: 5.0 is not of type 'integer', 'null'",
        "schema: $.episodes[0].segments[0].start_frame: 0.0 is not of type 'integer'",
    ]


def test_segment_and_primary_destinations_are_object_ids():
    doc = spec_example()
    _ep(doc)["segments"][3]["destination"] = "left side of the table"
    _ep(doc)["primary_destination"] = "table"
    assert validate_gold(doc) == [
        "episodes[0] F1/12 pass 1: primary_destination 'table' must be an object_id of this episode or null",
        "episodes[0] F1/12 pass 1: segment 3: destination 'left side of the table' must be an object_id of "
        "this episode or null",
    ]


def test_coarse_subtask_references_and_segment_coarse_idx():
    doc = spec_example()
    _ep(doc)["coarse_subtasks"][1]["destination"] = "o5"
    _ep(doc)["segments"][4]["coarse_idx"] = 7
    assert validate_gold(doc) == [
        "episodes[0] F1/12 pass 1: segment 4: coarse_idx 7 is not a coarse subtask of this episode",
        "episodes[0] F1/12 pass 1: coarse subtask 1: destination 'o5' is not an object_id of this episode",
    ]


def test_warnings_on_empty_or_non_object_episode():
    assert warnings_for_episode({}) == [
        "missing primary_target",
        "missing robot ending slot holding: holding",
        "missing robot ending slot gripper: gripper_open or gripper_closed",
        "missing robot ending slot position: withdrawn or at_home_pose or near_object",
    ]
    assert warnings_for_episode([]) == ["the episode is not a JSON object"]


def test_dump_gold_normalizes_negative_zero_and_refuses_nan():
    doc = spec_example()
    _ep(doc)["objects"][0]["first_frame_point"]["xy"] = [-0.0000001, 0.5]
    text = dump_gold(doc)
    assert "-0.0" not in text
    assert _ep(json.loads(text))["objects"][0]["first_frame_point"]["xy"] == [0.0, 0.5]
    _ep(doc)["active_seconds"] = float("nan")
    with pytest.raises(ValueError):
        dump_gold(doc)


def test_merge_into_nothing_and_empty_new_document():
    new = {"schema_version": SCHEMA_VERSION, "family": "F1", "guide_version": "g", "episodes": []}
    assert merge_episodes(None, new)["episodes"] == []
    merged = merge_episodes(spec_example(), new)
    assert [e["episode_key"] for e in merged["episodes"]] == ["F1/12"]
    assert merged["guide_version"] == "g" and merged["dataset"]["revision"] == "abc123"


def test_map_phase_ties_and_pattern_order():
    assert map_phase("open the gripper and move away") == "release"   # first listed pattern wins
    assert map_phase("move away from the box") == "retract"           # pattern before keyword "move"
    assert map_phase("move the arm away") == "transport"              # spec tables as written
    assert map_phase("carry and place") == "release"                  # priority 2 before 14
    assert map_phase("go to home") == "retract"                       # "home" (1) before "go to" (15)
    assert map_phase("let us go") == "unmapped"                       # "let" and "go" not adjacent


@pytest.mark.parametrize("text, expected", [
    ("at home", "at_home_pose"),        # one word each: the longer alias "home" wins over "at"
    ("on top of the box", "on_top_of"),
    ("in contact with the bowl", "touching"),
    ("held in the gripper", "in_gripper"),
    ("turned on", "activated"),         # two words beat "on"
    ("at the object", "near_object"),
])
def test_map_predicate_longest_alias_first(text, expected):
    assert map_predicate(text) == expected


def _worked_segments() -> list[dict]:
    """Spec 3.4.9 slip and regrasp, then a retract, as gold segments over 230 frames."""
    return [
        _seg(0, 0, 40, "approach", "reach toward the red cube", "o1", coarse=0),
        _seg(1, 41, 60, "grasp", "close on the red cube", "o1", coarse=0, outcome="failed",
             failure_type="slip"),
        _seg(2, 61, 90, "approach", "reach toward the red cube again", "o1", coarse=1, attempt_idx=2),
        _seg(3, 91, 110, "grasp", "close on the red cube", "o1", coarse=1, attempt_idx=2),
        _seg(4, 111, 170, "transport", "carry the red cube to the tray", "o1", "o2", coarse=2, attempt_idx=2),
        _seg(5, 171, 199, "release", "open over the tray", "o1", "o2", coarse=2, attempt_idx=2),
        _seg(6, 200, 229, "retract", "move away", None, quality=None, coarse=3, attempt_idx=2),
    ]


def test_worked_example_coarse_output_is_valid_gold():
    segs = _worked_segments()
    groups = coarse_groups(segs, object_namer(CUBE_OBJECTS), category_of=object_categorizer(CUBE_OBJECTS))
    assert [(g["text"], g["mistake"], g["start_frame"], g["end_frame"]) for g in groups] == [
        ("pick up the red cube", True, 0, 60), ("pick up the red cube", False, 61, 110),
        ("put the red cube in the tray", False, 111, 199), ("move the arm away", False, 200, 229)]
    doc = spec_example()
    ep = _ep(doc)
    ep.update(num_frames=230, objects=copy.deepcopy(CUBE_OBJECTS), segments=segs, coarse_subtasks=groups,
              hard_tags=["failed_attempt"], task_string="put the red cube in the tray")
    ep["failed_attempts"] = [{"span": [41, 55], "failure_type": "slip", "object": "o1", "evident_frame": 55,
                              "evident_camera": UP, "recovery_start": 61}]
    for req in ep["goal"]["requirements"]:
        req["deciding_frame"] = 229
    assert validate_gold(doc) == []


def test_coarse_misplace_keeps_transport_with_the_failed_release():
    segs = [
        {"start_frame": 0, "end_frame": 9, "phase_class": "approach", "target": "o1", "attempt_idx": 1},
        {"start_frame": 10, "end_frame": 19, "phase_class": "grasp", "target": "o1", "attempt_idx": 1},
        {"start_frame": 20, "end_frame": 29, "phase_class": "transport", "target": "o1", "destination": "o2",
         "attempt_idx": 1},
        {"start_frame": 30, "end_frame": 39, "phase_class": "release", "target": "o1", "destination": "o2",
         "outcome": "failed", "failure_type": "misplace", "attempt_idx": 1},
        {"start_frame": 40, "end_frame": 49, "phase_class": "approach", "target": "o1", "attempt_idx": 2},
        {"start_frame": 50, "end_frame": 59, "phase_class": "grasp", "target": "o1", "attempt_idx": 2},
        {"start_frame": 60, "end_frame": 69, "phase_class": "transport", "target": "o1", "destination": "o2",
         "attempt_idx": 2},
        {"start_frame": 70, "end_frame": 79, "phase_class": "release", "target": "o1", "destination": "o2",
         "attempt_idx": 2},
    ]
    groups = coarse_groups(segs, object_namer(CUBE_OBJECTS), category_of=object_categorizer(CUBE_OBJECTS))
    assert [(g["text"], g["mistake"], g["segment_indices"]) for g in groups] == [
        ("pick up the red cube", False, [0, 1]),
        ("put the red cube in the tray", True, [2, 3]),
        ("pick up the red cube", False, [4, 5]),
        ("put the red cube in the tray", False, [6, 7]),
    ]


def test_coarse_failed_attempt_intent_ignores_retract_and_placeholders():
    aborted = [
        {"start": 0, "end": 9, "phase_class": "approach", "target_name": "red cube", "outcome": "aborted",
         "attempt_idx": 1},
        {"start": 10, "end": 19, "phase_class": "retract", "target_name": "none", "outcome": "aborted",
         "attempt_idx": 1},
        {"start": 20, "end": 29, "phase_class": "approach", "target_name": "red cube", "attempt_idx": 2},
        {"start": 30, "end": 39, "phase_class": "grasp", "target_name": "unsure", "attempt_idx": 2},
    ]
    groups = coarse_groups(aborted)
    assert [(g["text"], g["mistake"], g["target"]) for g in groups] == [
        ("pick up the red cube", True, "red cube"), ("pick up the red cube", False, "red cube")]
    assert coarse_groups([{"start": 0, "end": 5, "phase_class": "grasp", "target_name": "unsure"}])[0]["text"] \
        == "pick up the object"


def test_single_segment_coarse_and_compiled_goal():
    only_retract = [{"start_frame": 0, "end_frame": 29, "phase_class": "retract", "target": None}]
    assert [g["text"] for g in coarse_groups(only_retract)] == ["move the arm away"]
    goal = compiled_goal(only_retract)
    assert goal["primary_target"] is None and goal["primary_destination"] is None
    assert [(r["predicate"], r["status"], r["deciding_frame"]) for r in goal["requirements"]] == [
        ("withdrawn", "required", 29)]
    only_grasp = [{"start_frame": 0, "end_frame": 29, "phase_class": "grasp", "target": "o1"}]
    assert coarse_groups(only_grasp, object_namer(CUBE_OBJECTS))[0]["text"] == "pick up the red cube"
    assert compiled_goal(only_grasp, CUBE_OBJECTS)["requirements"] == []
    assert compiled_goal(only_grasp, CUBE_OBJECTS)["primary_target"] == "o1"


def test_compiled_goal_uses_the_last_grasp_and_release():
    segs = [
        {"start_frame": 0, "end_frame": 9, "phase_class": "grasp", "target": "o1"},
        {"start_frame": 10, "end_frame": 19, "phase_class": "release", "target": "o1", "destination": "o2"},
        {"start_frame": 20, "end_frame": 29, "phase_class": "grasp", "target": "o3"},
        {"start_frame": 30, "end_frame": 39, "phase_class": "release", "target": "o3", "destination": "o1",
         "outcome": "failed"},
    ]
    objects = [*CUBE_OBJECTS, {"object_id": "o3", "name": "blue cube", "aliases": [], "category": "block"}]
    goal = compiled_goal(segs, objects)
    assert goal["primary_target"] == "o3" and goal["primary_destination"] == "o1"
    assert [(r["object"], r["predicate"], r["ref_object"], r["achieved"]) for r in goal["requirements"]] == [
        ("o3", "on_top_of", "o1", False)]
    assert goal["objective_text"] == "the blue cube is on top of the red cube"


@pytest.mark.parametrize("name, relation", [
    ("bowl between the plate and the ramekin", "in"),
    ("plate to the left of the bowl", "on"),
    ("red cube near the tray", "at"),
    ("the transparent box", "in"),
])
def test_put_relation_uses_the_head_of_the_destination_name(name, relation):
    segs = [
        {"start_frame": 0, "end_frame": 9, "phase_class": "transport", "target": "cube", "destination": name},
        {"start_frame": 10, "end_frame": 19, "phase_class": "release", "target": "cube", "destination": name},
    ]
    assert coarse_groups(segs)[0]["text"].split()[3] == relation
