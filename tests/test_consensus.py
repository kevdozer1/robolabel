"""PLAN 3.7 consensus counting: Wilson bound, thresholds, per-episode fill."""

from __future__ import annotations

import pytest

from robolabel.layers.consensus import count_predicates, episode_goal, status_for, wilson_lower


def test_wilson_lower_known_values():
    assert wilson_lower(0, 0) == 0.0
    assert wilson_lower(10, 10) == pytest.approx(0.7225, abs=1e-4)
    assert wilson_lower(28, 30) == pytest.approx(0.7868, abs=1e-3)


def test_status_thresholds():
    assert status_for(0.95, 0.8, False) == ("required", None, "consensus")
    assert status_for(0.95, 0.7, False) == ("unsure", "intent", "consensus")  # p high but too few episodes
    assert status_for(0.5, 0.3, False) == ("unsure", "intent", "consensus")
    assert status_for(0.3, 0.1, False) == ("incidental", None, "consensus")
    assert status_for(0.0, 0.0, True) == ("required", None, "task_string")


def test_count_and_fill():
    inside = {"kind": "object_end_state", "object": "target", "predicate": "inside", "ref_object": "destination"}
    held = {"kind": "robot_end_state", "object": "none", "predicate": "holding", "ref_object": "none"}
    wd = {"kind": "robot_end_state", "object": "none", "predicate": "withdrawn", "ref_object": "none"}
    recs = []
    for i in range(30):
        facts = [dict(inside, value=True), dict(held, value=False), dict(wd, value=i % 2 == 0)]
        recs.append({"episode_key": f"F1/{i}", "successful": True, "facts": facts})
    recs.append({"episode_key": "F1/99", "successful": False, "facts": [dict(inside, value=False)]})
    rows = count_predicates(recs, stated=[("object_end_state", "target", "inside", "destination")])
    by = {r["predicate"]: r for r in rows}
    assert by["inside"]["status"] == "required" and by["inside"]["basis"] == "task_string"
    assert by["withdrawn"]["p"] == 0.5 and by["withdrawn"]["status"] == "unsure"
    assert "holding" not in by  # holding true never observed
    goal = episode_goal(rows, recs[1])
    assert {g["predicate"]: g["achieved"] for g in goal} == {"inside": True, "withdrawn": False}
