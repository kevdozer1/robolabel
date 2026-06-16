"""Tests for the retrieval subgoal (same-phase end frame from a different episode)."""

from __future__ import annotations

from robolabel.retrieve import retrieve_subgoals
from robolabel.schema import (
    EpisodeAnnotation,
    Subgoal,
    SubtaskSegment,
    episode_records,
    to_dataframe,
)


def _ann(eid: str, phases_ends: list[tuple[str, int]]) -> EpisodeAnnotation:
    subs, sgs = [], []
    for i, (ph, end) in enumerate(phases_ends):
        start = 0 if i == 0 else phases_ends[i - 1][1] + 1
        subs.append(SubtaskSegment(i, start, end, ph, phase=ph))
        sgs.append(Subgoal(i, end))                 # real keyframe = segment end
    return EpisodeAnnotation(episode_id=eid, task="t", num_frames=phases_ends[-1][1] + 1,
                             fps=30.0, provider="mock", model="mock", subtasks=subs, subgoals=sgs)


def test_retrieve_picks_same_phase_other_episode():
    df = to_dataframe([
        _ann("0", [("approach", 10), ("grasp", 20)]),
        _ann("1", [("approach", 15), ("grasp", 25)]),
    ])
    out = retrieve_subgoals(df, method="random", seed=1)
    r0 = episode_records(out, "0")["subtasks"]  # noqa: F841 (subtask phases checked via subgoals)
    sg0 = episode_records(out, "0")["subgoals"]
    by_seg = {int(s["segment_idx"]): s for s in sg0}
    # ep0 'approach' subgoal must point at ep1's 'approach' end frame (15), not its own.
    assert by_seg[0]["retrieved_subgoal_episode_id"] == "1"
    assert int(by_seg[0]["retrieved_subgoal_frame_idx"]) == 15
    assert by_seg[0]["subgoal_frame_idx"] == 10        # real keyframe untouched
    assert int(by_seg[1]["retrieved_subgoal_frame_idx"]) == 25  # grasp -> ep1 grasp end


def test_retrieve_no_same_phase_candidate_leaves_null():
    df = to_dataframe([
        _ann("0", [("approach", 10)]),
        _ann("1", [("pour", 12)]),       # no shared phase
    ])
    out = retrieve_subgoals(df, method="random", seed=1)
    sg = episode_records(out, "0")["subgoals"][0]
    assert sg.get("retrieved_subgoal_episode_id") in (None,) or _isnull(sg.get("retrieved_subgoal_episode_id"))


def test_retrieve_only_from_allowed_sources():
    df = to_dataframe([
        _ann("0", [("approach", 10), ("grasp", 20)]),
        _ann("1", [("approach", 15), ("grasp", 25)]),
        _ann("2", [("approach", 18), ("grasp", 28)]),
    ])
    # allow retrieving only FROM episode "2" (e.g. the only gate-passed one)
    out = retrieve_subgoals(df, method="random", seed=1, allowed_sources={"2"})
    for e in ("0", "1"):
        for s in episode_records(out, e)["subgoals"]:
            rid = s.get("retrieved_subgoal_episode_id")
            if not _isnull(rid):
                assert rid == "2"
    # empty allowed set -> nothing may be retrieved from
    out2 = retrieve_subgoals(df, method="random", seed=1, allowed_sources=set())
    assert all(_isnull(s.get("retrieved_subgoal_episode_id"))
               for e in ("0", "1", "2") for s in episode_records(out2, e)["subgoals"])


def test_retrieve_is_deterministic():
    df = to_dataframe([
        _ann("0", [("approach", 10), ("grasp", 20)]),
        _ann("1", [("approach", 15), ("grasp", 25)]),
        _ann("2", [("approach", 18), ("grasp", 28)]),
    ])
    a = retrieve_subgoals(df, method="random", seed=7)
    b = retrieve_subgoals(df, method="random", seed=7)
    ga = [s["retrieved_subgoal_frame_idx"] for s in episode_records(a, "0")["subgoals"]]
    gb = [s["retrieved_subgoal_frame_idx"] for s in episode_records(b, "0")["subgoals"]]
    assert ga == gb


def _isnull(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)
