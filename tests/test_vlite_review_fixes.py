"""Review fixes of the V-lite layers (review:vlite findings 0 to 12): segment snapping, goal references and
canonical forms, rules 6, 7 and 9, routing of failed calls, the episode outcome, requirement text, the
sig_only baseline and the pipeline code stamp. No network, no model calls."""

from __future__ import annotations

import shutil

from robolabel import vlite
from robolabel.baselines import sig_only_segments
from robolabel.layers.check import run_checks
from robolabel.layers.goal import episode_outcome, postprocess_goal, raw_goal_refs, requirement_text
from robolabel.layers.segment import postprocess_segments

OBJECTS = [{"object_id": "o1", "name": "black bowl", "aliases": ["bowl"], "category": "bowl", "views": []},
           {"object_id": "o2", "name": "plate", "aliases": [], "category": "plate", "views": []}]


class Ep:
    episode_id = "F2/1"
    num_frames = 120
    fps = 10.0
    task = "put the black bowl on the plate"
    extra = {"camera_order": ["observation.images.image", "observation.images.wrist_image"],
             "external_cameras": ["observation.images.image"], "wrist_cameras": ["observation.images.wrist_image"]}


def l1_with(cands: list[tuple[str, int]], end_state: list[dict] | None = None, attempts: list[dict] | None = None,
            events: list[dict] | None = None) -> dict:
    return {"candidates": [{"candidate_id": c, "frame": f, "transition": "approach->grasp", "confidence": "high",
                            "attempt_idx": 1} for c, f in cands],
            "end_state": end_state or [], "attempts": attempts or [], "events": events or []}


def seg(start: int, end: int, cid: str = "none", phase: str = "approach", src: str = "vlm", **kw) -> dict:
    return {"start_frame": start, "end_frame": end, "phase_class": phase, "phase_text": f"{phase} the bowl",
            "target": "o1", "destination": "none", "attempt_idx": 1, "outcome": kw.get("outcome", "success"),
            "failure_type": kw.get("failure_type", "none"), "boundary_source": src, "candidate_id": cid,
            "evidence": []}


def req(pred: str, value, achieved, *, kind: str = "robot_end_state", status: str = "required", obj: str = "none",
        ref: str = "none", uk: str = "none") -> dict:
    return {"req_id": "r1", "kind": kind, "object": obj, "predicate": pred, "ref_object": ref, "value": value,
            "status": status, "unsure_kind": uk, "basis": "observed", "achieved": achieved, "deciding_frame": 119,
            "deciding_camera": "scene", "visibility": [], "reason": ""}


# ------------------------------------------------------------------------------------------ vlite 1, 11, 12
def test_snap_to_confirmed_candidates_after_contiguity():
    """Finding 1 scenario: c1=45 and c2=80 confirmed, the model says [0-50 c1], [51-70 c2], [71-119]."""
    l1 = l1_with([("c1", 45), ("c2", 80)])
    data = {"segments": [seg(0, 50, "c1", src="signal"), seg(51, 70, "c2", "grasp", src="signal"),
                         seg(71, 119, phase="transport")],
            "candidates": [{"candidate_id": "c1", "verdict": "confirm", "note": ""},
                           {"candidate_id": "c2", "verdict": "confirm", "note": ""}]}
    repairs: list[str] = []
    out, _ = postprocess_segments(data, 120, l1, OBJECTS, repairs)
    assert [(s["start_frame"], s["end_frame"], s["boundary_source"]) for s in out] == [
        (0, 45, "signal"), (46, 80, "signal"), (81, 119, "vlm")]
    assert any("snapped to confirmed c1 at 45" in r for r in repairs)
    assert any("snapped to confirmed c2 at 80" in r for r in repairs)


def test_signal_source_only_when_the_end_is_the_candidate_frame():
    """Finding 11: a final segment carrying a confirmed candidate, and a segment carrying the candidate
    that begins it, are vlm boundaries (with a repair)."""
    l1 = l1_with([("c1", 40), ("c4", 90)])
    data = {"segments": [seg(0, 40, "c1", src="signal"), seg(41, 90, "c1", "grasp", src="signal"),
                         seg(91, 119, "c4", "retract", src="signal")],
            "candidates": [{"candidate_id": "c1", "verdict": "confirm", "note": ""},
                           {"candidate_id": "c4", "verdict": "confirm", "note": ""}]}
    repairs: list[str] = []
    out, _ = postprocess_segments(data, 120, l1, OBJECTS, repairs)
    assert [(s["start_frame"], s["end_frame"], s["boundary_source"]) for s in out] == [
        (0, 40, "signal"), (41, 90, "vlm"), (91, 119, "vlm")]
    assert any("confirmed c1 at 40 cannot end the segment 41-90" in r for r in repairs)
    assert any("confirmed c4 at 90 cannot end the segment 91-119" in r for r in repairs)
    assert sum("boundary_source" in r and "set to vlm" in r for r in repairs) == 2


def test_segment_clamps_and_coercions_are_recorded():
    """Finding 12: end_frame 303 on a 303-frame episode, an unknown phase class and outcome, and a
    failure type on a successful segment are all repairs."""
    s = seg(0, 303, phase="dance", outcome="great", failure_type="slip")
    repairs: list[str] = []
    out, _ = postprocess_segments({"segments": [s], "candidates": []}, 303, l1_with([]), OBJECTS, repairs)
    assert out[0]["end_frame"] == 302 and out[0]["phase_class"] == "other" and out[0]["outcome"] == "success"
    assert out[0]["failure_type"] == "none"
    text = " | ".join(repairs)
    assert "frames (0, 303) clamped to (0, 302)" in text
    assert "phase_class 'dance'" in text and "outcome 'great'" in text and "failure_type 'slip'" in text


# ------------------------------------------------------------------------------------------ vlite 6, 7
def test_goal_references_map_names_and_keep_unresolved_text():
    """Finding 6: 'black bowl' maps to o1; 'saucer' is not in the inventory and stays as text, with a
    repair; rule 9 sees the goal's raw references."""
    data = {"objective_text": "the bowl is on the saucer", "primary_target": "black bowl",
            "primary_destination": "saucer",
            "requirements": [req("on_top_of", "true", "true", kind="object_end_state", obj="bowl", ref="saucer")]}
    repairs: list[str] = []
    goal = postprocess_goal(data, Ep(), l1_with([]), OBJECTS, [], repairs)
    assert goal["primary_target"] == "o1" and goal["primary_destination"] == "saucer"
    r = goal["requirements"][0]
    assert r["object"] == "o1" and r["ref_object"] == "saucer"
    assert any("'saucer' is not an inventory ID or name, kept" in x for x in repairs)
    assert requirement_text(r, {"o1": "black bowl", "o2": "plate"}) == "black bowl is on top of saucer"
    raw = raw_goal_refs(data)
    checks = run_checks([], [], goal, l1_with([]), OBJECTS, [], raw, have_inventory=True, have_facts=False)
    rule9 = next(c for c in checks["checks"] if c["rule_id"] == 9)
    assert rule9["verdict"] == "fail" and "saucer" in rule9["note"]


def test_holding_false_takes_the_canonical_form():
    """Finding 7: 'holding o1, value false' becomes 'holding none, value false'; 'holding o1 true' stays."""
    data = {"objective_text": "", "primary_target": "o1", "primary_destination": "o2",
            "requirements": [req("holding", "false", "true", ref="o1"), req("gripper_open", "true", "true")]}
    repairs: list[str] = []
    goal = postprocess_goal(data, Ep(), l1_with([]), OBJECTS, [], repairs)
    h = next(r for r in goal["requirements"] if r["predicate"] == "holding")
    assert h["ref_object"] == "none" and h["value"] is False
    assert any("canonical form" in x for x in repairs)
    data["requirements"][0] = req("holding", "true", "true", ref="o1")
    goal = postprocess_goal(data, Ep(), l1_with([]), OBJECTS, [], [])
    assert next(r for r in goal["requirements"] if r["predicate"] == "holding")["ref_object"] == "o1"


# ------------------------------------------------------------------------------------------ vlite 2, 8
def _rule(checks: dict, rule: int) -> dict:
    return next(c for c in checks["checks"] if c["rule_id"] == rule)


def _goal(reqs: list[dict]) -> dict:
    for r in reqs:
        r.setdefault("added_by", "model")
    return {"primary_target": "o1", "primary_destination": "o2", "requirements": reqs, "episode_outcome": "success"}


def test_rule6_compares_the_claimed_end_state():
    """Finding 2: 'gripper_open, value true, achieved false' agrees with L1 gripper_closed; 'holding,
    value false, achieved false' (still holding) contradicts L1 holding false."""
    l1 = l1_with([], end_state=[{"predicate": "holding", "value": False, "confidence": "high"},
                                {"predicate": "gripper_closed", "value": True, "confidence": "high"}])
    ok = run_checks([], [], _goal([req("gripper_open", True, False)]), l1, OBJECTS, [], [],
                    have_inventory=True, have_facts=False)
    assert _rule(ok, 6)["verdict"] == "pass"
    bad = run_checks([], [], _goal([req("holding", False, False)]), l1, OBJECTS, [], [],
                     have_inventory=True, have_facts=False)
    assert _rule(bad, 6)["verdict"] == "fail" and bad["routed"]
    skip = run_checks([], [], _goal([req("holding", True, "unknown")]), l1, OBJECTS, [], [],
                      have_inventory=True, have_facts=False)
    assert _rule(skip, 6)["verdict"] == "na"


def test_rule7_checks_only_a_final_retract():
    """Finding 8: a retract mid-episode is na for rule 7; a final retract is checked against withdrawn."""
    l1 = l1_with([], end_state=[{"predicate": "withdrawn", "value": False, "confidence": "low"}])
    mid = [seg(0, 30), seg(31, 50, phase="retract"), seg(51, 119, phase="grasp")]
    res = run_checks(mid, [], None, l1, OBJECTS, [], [], have_inventory=True, have_facts=False)
    assert _rule(res, 7)["verdict"] == "na"
    final = [seg(0, 30), seg(31, 119, phase="retract")]
    res = run_checks(final, [], None, l1, OBJECTS, [], [], have_inventory=True, have_facts=False)
    assert _rule(res, 7)["verdict"] == "fail"


# ------------------------------------------------------------------------------------------ vlite 9
def test_failed_calls_are_not_routed_by_l5():
    """Finding 9 was not adopted (SPEC_QUESTIONS Q155): V_LITE lists three route conditions and risk is
    failed rules over applicable rules, so a failed call is recorded in repairs and validity only."""
    segs = [seg(0, 119, phase="other")]
    res = run_checks(segs, [], None, l1_with([]), [], [], [], have_inventory=False, have_facts=False)
    assert res["routed"] is False and res["risk"] != 1.0


# ------------------------------------------------------------------------------------------ vlite 3
def test_episode_outcome_follows_v_lite():
    """Finding 3 was not adopted (SPEC_QUESTIONS Q155): items with unsure / perception count as unknown,
    robot items included, and L1 does not fill in achieved."""
    reqs = [req("inside", True, True, kind="object_end_state", obj="o1", ref="o2"),
            req("holding", False, True)]
    assert episode_outcome(reqs) == "success"
    robot_unsure = req("withdrawn", True, "unknown", status="unsure", uk="perception")
    assert episode_outcome([*reqs, robot_unsure]) == "unknown"
    assert episode_outcome([reqs[0], req("holding", False, "unknown")]) == "unknown"
    assert episode_outcome([reqs[0], req("holding", False, False)]) == "partial"


# ------------------------------------------------------------------------------------------ vlite 10
def test_requirement_text_for_unsure_values_and_refs():
    names = {"o1": "black bowl", "o2": "plate"}
    assert requirement_text(req("holding", "unsure", True), names) == "unsure whether the robot is holding an object"
    assert requirement_text(req("inside", "unsure", True, kind="object_end_state", obj="o1", ref="o2"), names) == \
        "unsure whether black bowl is inside plate"
    assert requirement_text(req("on_top_of", True, True, kind="object_end_state", obj="o1", ref="unsure"), names) == \
        "black bowl is on top of an unidentified object"
    assert requirement_text(req("holding", False, True), names) == "the robot is holding nothing"
    assert requirement_text(req("withdrawn", "unsure", True), names) == "unsure whether the robot is withdrawn"


# ------------------------------------------------------------------------------------------ vlite 5
def test_sig_only_marks_only_empty_and_slip_as_failed():
    base = {"closing_offset": 0, "opening_offset": None, "hold_frame": None, "failure_type": "aborted"}
    attempts = [{**base, "attempt_idx": 1, "outcome": "aborted", "closing_onset": 20, "closing_offset": 23,
                 "opening_onset": 30, "opening_offset": 33, "event_frame": 30},
                {**base, "attempt_idx": 2, "outcome": "empty", "failure_type": "missed_grasp", "closing_onset": 50,
                 "closing_offset": 53, "opening_onset": 60, "opening_offset": 63, "event_frame": 55},
                {**base, "attempt_idx": 3, "outcome": "unknown", "failure_type": "none", "closing_onset": 110,
                 "closing_offset": 113, "event_frame": 113}]
    segs = sig_only_segments({"num_frames": 120, "attempts": attempts, "candidates": [], "end_state": []})
    failed = {s["attempt_idx"] for s in segs if s["outcome"] == "failed"}
    assert failed == {2}
    assert all(not s["mistake"] for s in segs if s["attempt_idx"] in (1, 3))


# ------------------------------------------------------------------------------------------ vlite 0
def test_pipeline_code_is_a_stable_hash_of_the_pipeline_files(tmp_path):
    from pathlib import Path

    pkg = Path(vlite.__file__).resolve().parent
    assert len(vlite.PIPELINE_CODE) == 12 and int(vlite.PIPELINE_CODE, 16) >= 0
    assert vlite.pipeline_code(pkg) == vlite.PIPELINE_CODE
    copy = tmp_path / "robolabel"
    shutil.copytree(pkg / "layers", copy / "layers", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(pkg / "prompts" / "v7", copy / "prompts" / "v7", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("vlite.py", "schema_v7.py"):
        shutil.copy2(pkg / name, copy / name)
    (copy / "unrelated.py").write_text("x = 1\n", encoding="utf-8")  # outside the hashed set
    assert vlite.pipeline_code(copy) == vlite.PIPELINE_CODE
    check = copy / "layers" / "check.py"
    check.write_bytes(check.read_bytes().replace(b"\n", b"\r\n"))  # line endings do not count
    assert vlite.pipeline_code(copy) == vlite.PIPELINE_CODE
    check.write_bytes(check.read_bytes() + b"# changed\r\n")
    assert vlite.pipeline_code(copy) != vlite.PIPELINE_CODE


# ------------------------------------------------------------------------------------------ second-pass checks
def test_rule6_does_not_compare_not_holding_a_named_object():
    """L1 holding means holding any object: 'holding o1, value true, achieved false' (not holding o1 at the
    end) is not a contradiction of L1 holding true, but 'holding o1' achieved against L1 holding false is."""
    held = l1_with([], end_state=[{"predicate": "holding", "value": True, "confidence": "high"}])
    res = run_checks([], [], _goal([req("holding", True, False, ref="o1")]), held, OBJECTS, [], [],
                     have_inventory=True, have_facts=False)
    assert _rule(res, 6)["verdict"] == "na" and not res["routed"]
    empty = l1_with([], end_state=[{"predicate": "holding", "value": False, "confidence": "high"}])
    res = run_checks([], [], _goal([req("holding", True, True, ref="o1")]), empty, OBJECTS, [], [],
                     have_inventory=True, have_facts=False)
    assert _rule(res, 6)["verdict"] == "fail"


def test_segment_parsing_survives_odd_numbers():
    """An attempt_idx like '--5' and a frame of 1e400 (JSON infinity) are repairs, not crashes."""
    odd = {**seg(0, 60), "attempt_idx": "--5"}
    inf = {**seg(61, 119), "end_frame": float("inf")}
    repairs: list[str] = []
    out, _ = postprocess_segments({"segments": [odd, inf], "candidates": []}, 120, l1_with([]), OBJECTS, repairs)
    assert [(s["start_frame"], s["end_frame"], s["attempt_idx"]) for s in out] == [(0, 119, 1)]
    assert any("attempt_idx '--5'" in r for r in repairs)
    assert any("without integer frames" in r for r in repairs)
    two = {**seg(0, 119), "attempt_idx": "2"}
    out, _ = postprocess_segments({"segments": [two], "candidates": []}, 120, l1_with([]), OBJECTS, [])
    assert out[0]["attempt_idx"] == 2


def test_goal_coercions_are_recorded():
    """Finding 12 on the goal layer: a 'yes' value, an unknown deciding camera, a visibility entry with an
    unknown camera, a long reason and objective, and an infinite deciding frame are recorded; a JSON false
    for achieved is read as false."""
    r = {**req("lifted", "yes", False, kind="object_end_state", obj="o1"), "deciding_camera": "banana",
         "visibility": [{"camera": "zzz", "class": "visible"}, {"camera": "scene", "class": "visible"}],
         "reason": "r" * 300, "deciding_frame": float("inf")}
    data = {"objective_text": "x" * 400, "primary_target": "o1", "primary_destination": "none", "requirements": [r]}
    repairs: list[str] = []
    goal = postprocess_goal(data, Ep(), l1_with([]), OBJECTS, [], repairs)
    g = goal["requirements"][0]
    assert g["value"] is True and g["achieved"] is False and g["deciding_frame"] == 119
    assert g["visibility"] == {"observation.images.image": "visible"} and g["deciding_camera"] == ""
    text = " | ".join(repairs)
    assert "value 'yes' of lifted set to True" in text
    assert "deciding_camera 'banana'" in text and "visibility of lifted: 1 entry" in text
    assert "reason of lifted cut" in text and "objective_text cut" in text and "deciding_frame inf" in text
    clean: list[str] = []
    ok = {**req("lifted", "true", "true", kind="object_end_state", obj="o1"), "deciding_camera": "scene",
          "visibility": [{"camera": "scene", "class": "visible"}]}
    postprocess_goal({"objective_text": "x", "primary_target": "o1", "primary_destination": "none",
                      "requirements": [ok]}, Ep(), l1_with([]), OBJECTS, [], clean)
    assert not [x for x in clean if "lifted" in x or "objective" in x]
