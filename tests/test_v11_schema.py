"""v1.1 schema additions (SPEC_V1_1 3.5): additive optional fields, readers that accept rows without them,
and the v7 writer left as it was."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from robolabel import schema, schema_v7
from robolabel.schema_v7 import (
    COLUMNS_V7,
    COLUMNS_V11,
    END_EVENTS,
    EPISODE_V11,
    SUBTASK_V11,
    add_v11_fields,
    episode_fields_v11,
    episode_record_v11,
    episode_row_fields_v11,
    episode_rows,
    fill_attempt_outcome,
    read_v11,
    subtask_fields_v11,
    subtask_records_v11,
    to_dataframe_v7,
    to_dataframe_v11,
    validate_v11_row,
    write_v7,
    write_v11,
)

NEW = ["end_event", "coarse_end_frame", "crawl_calls", "attempt_outcome", "event_sources", "has_end_state",
       "goal_command", "coarse_mode", "coarse_fps", "crawl_enabled", "crawl_model"]


def _episode():
    return SimpleNamespace(episode_id="F1/0", task="put the cube in the box", num_frames=120, fps=30.0,
                           extra={"camera_order": ["observation.images.up"], "family": "F1"})


def _seg(start, end, phase, end_event, source, *, coarse_end=None, calls=0, attempt=1, outcome="success",
         attempt_outcome="success", failure="none", mistake=False):
    return {"start_frame": start, "end_frame": end, "phase_class": phase, "phase_text": f"{phase} the cube",
            "target": "the red cube", "destination": "the box", "attempt_idx": attempt, "outcome": outcome,
            "attempt_outcome": attempt_outcome, "failure_type": failure, "mistake": mistake, "end_event": end_event,
            "boundary_source": source, "coarse_end_frame": end if coarse_end is None else coarse_end,
            "crawl_calls": calls, "candidate_id": "none", "evidence": []}


def _segments():
    return [_seg(0, 39, "approach", "close_start", "coarse"),
            _seg(40, 79, "grasp", "open_start", "crawl", coarse_end=37, calls=2),
            _seg(80, 119, "release", "other", "crawl", coarse_end=81, calls=3)]


def _rows(segments=None):
    segs = _segments() if segments is None else segments
    return episode_rows(arm="L-B", episode=_episode(), provider="mock", model="mock-model",
                        pipeline_version="v1.1 test", objects=[], facts=[], segments=segs, coarse=[], goal=None,
                        l1={"attempts": []}, checks={"checks": [], "risk": 0.0, "routed": False}, cost_usd=0.01,
                        source_model="mock-model")


def _v11_rows():
    return add_v11_fields(_rows(), _segments(), event_sources=["none"],
                          episode_fields={"has_end_state": True, "goal_command": "Put the red cube in the box",
                                          "coarse_mode": "frames", "coarse_fps": 2.0, "crawl_enabled": True,
                                          "crawl_model": "luna"})


def test_v7_columns_and_writer_are_unchanged():
    assert not set(NEW) & set(COLUMNS_V7)
    assert "boundary_source" in COLUMNS_V7  # v1.1 only adds values (coarse, crawl) to this column
    assert COLUMNS_V11[: len(COLUMNS_V7)] == COLUMNS_V7 and set(COLUMNS_V11) - set(COLUMNS_V7) == set(NEW)
    assert set(SUBTASK_V11) | set(EPISODE_V11) == set(NEW)
    frame = to_dataframe_v7(_rows())
    assert list(frame.columns) == COLUMNS_V7
    # the v7 rows of a v1.1 segment carry no v1.1 field: episode_rows is unchanged
    assert not any(k in r for r in _rows() for k in NEW)


def test_v11_rows_round_trip_through_parquet(tmp_path):
    rows = _v11_rows()
    assert all(validate_v11_row(r) == [] for r in rows)
    path = write_v11(rows, tmp_path)
    frame = read_v11(path)
    assert list(frame.columns) == COLUMNS_V11
    subs = subtask_records_v11(frame, "F1/0", arm="L-B")
    assert [(s["segment_idx"], s["start_frame"], s["end_frame"]) for s in subs] == [(0, 0, 39), (1, 40, 79),
                                                                                     (2, 80, 119)]
    assert [s["end_event"] for s in subs] == ["close_start", "open_start", "other"]
    assert [s["boundary_source"] for s in subs] == ["coarse", "crawl", "crawl"]
    assert [s["coarse_end_frame"] for s in subs] == [39, 37, 81]
    assert [s["crawl_calls"] for s in subs] == [0, 2, 3]
    assert all(type(s["coarse_end_frame"]) is int and type(s["crawl_calls"]) is int for s in subs)
    assert [s["attempt_outcome"] for s in subs] == ["success"] * 3
    assert [s["event_sources"] for s in subs] == [["none"]] * 3
    meta = episode_record_v11(frame, "F1/0", arm="L-B")
    assert meta["has_end_state"] is True and meta["crawl_enabled"] is True
    assert meta["goal_command"] == "Put the red cube in the box"
    assert meta["event_sources"] == ["none"] and meta["coarse_mode"] == "frames"
    assert meta["coarse_fps"] == 2.0 and meta["crawl_model"] == "luna"
    assert meta["n_attempts"] == 1 and meta["episode_id"] == "F1/0"
    assert episode_record_v11(frame, "F1/0", arm="other arm") is None
    # dtypes: integers nullable, floats float64, flags nullable booleans, lists as comma-joined text
    assert str(frame["coarse_end_frame"].dtype) == "Int64" and str(frame["crawl_calls"].dtype) == "Int64"
    assert str(frame["coarse_fps"].dtype) == "float64"
    assert str(frame["has_end_state"].dtype) == "boolean" and str(frame["crawl_enabled"].dtype) == "boolean"
    assert set(frame["event_sources"].dropna()) == {"none"}


def test_v7_files_read_with_the_v11_fields_absent(tmp_path):
    write_v7(_rows(), tmp_path)
    frame = read_v11(tmp_path)
    assert set(NEW) <= set(frame.columns)
    subs = subtask_records_v11(frame, "F1/0")
    assert len(subs) == 3
    for s in subs:
        assert {k: s[k] for k in ("end_event", "coarse_end_frame", "crawl_calls", "attempt_outcome",
                                  "event_sources")} == dict.fromkeys(SUBTASK_V11)
    assert [s["boundary_source"] for s in subs] == ["coarse", "crawl", "crawl"]  # the v7 column
    meta = episode_record_v11(frame, "F1/0")
    assert {k: meta[k] for k in EPISODE_V11} == dict.fromkeys(EPISODE_V11)
    # the legacy reader still reads both files
    assert len(schema.episode_records(schema.read_annotations(tmp_path), "F1/0")["subtasks"]) == 3
    v11 = tmp_path / "v11"
    write_v11(_v11_rows(), v11)
    assert len(schema.episode_records(schema.read_annotations(v11), "F1/0")["subtasks"]) == 3


def test_readers_default_to_none_for_rows_without_the_fields():
    assert subtask_fields_v11({}) == {"end_event": None, "boundary_source": None, "coarse_end_frame": None,
                                      "crawl_calls": None, "attempt_outcome": None, "event_sources": None}
    assert episode_fields_v11({}) == dict.fromkeys(EPISODE_V11)
    row = pd.Series({"end_event": "open_start", "coarse_end_frame": pd.NA, "crawl_calls": 2.0,
                     "event_sources": "gripper,motion", "boundary_source": float("nan")})
    got = subtask_fields_v11(row)
    assert got == {"end_event": "open_start", "boundary_source": None, "coarse_end_frame": None, "crawl_calls": 2,
                   "attempt_outcome": None, "event_sources": ["gripper", "motion"]}
    ep = episode_fields_v11(SimpleNamespace(has_end_state="false", coarse_fps="1.5", crawl_enabled=None))
    assert ep["has_end_state"] is False and ep["coarse_fps"] == 1.5 and ep["crawl_enabled"] is None


def test_episode_fields_and_validation():
    f = episode_row_fields_v11(event_sources=("gripper", "motion"), coarse_mode="video", crawl_enabled=False)
    assert f["event_sources"] == "gripper,motion" and f["coarse_mode"] == "video" and f["crawl_enabled"] is False
    assert f["has_end_state"] is None and f["coarse_fps"] is None
    bad = {"end_event": "grasp_start", "boundary_source": "model", "attempt_outcome": "partial",
           "coarse_mode": "gif", "coarse_end_frame": -1, "crawl_calls": "two"}
    assert len(validate_v11_row(bad)) == 6
    assert validate_v11_row({}) == []
    assert validate_v11_row({"boundary_source": "signal", "end_event": "contact_end", "crawl_calls": 0}) == []


def test_fill_attempt_outcome_uses_the_v7_rule_where_missing():
    old = [{"attempt_idx": 1, "outcome": "failed"}, {"attempt_idx": 1, "outcome": "failed"},
           {"attempt_idx": 2, "outcome": "success"}, {"attempt_idx": 3, "outcome": "aborted"},
           {"attempt_idx": 4, "outcome": "success", "attempt_outcome": "failed"}]
    got = fill_attempt_outcome(old)
    assert [s["attempt_outcome"] for s in got] == ["failed", "failed", "success", "aborted", "failed"]
    assert "attempt_outcome" not in old[0]  # copies, not edits


def test_v11_dataframe_is_deterministic():
    a, b = to_dataframe_v11(_v11_rows()), to_dataframe_v11(_v11_rows())
    assert a.equals(b)
    assert to_dataframe_v11(list(reversed(_v11_rows()))).equals(a)  # row order does not matter


def test_end_events_match_the_v8_prompts():
    v8 = pytest.importorskip("robolabel.prompts.v8")
    assert list(v8.END_EVENTS) == list(END_EVENTS)


def test_schema_module_keeps_its_v7_names():
    for name in ("SCHEMA_VERSION_V7", "SUBTASK_V7", "EPISODE_V7", "RECORD_V7", "COLUMNS_V7", "RECORD_ORDER",
                 "INT_COLS", "FLOAT_COLS", "to_dataframe_v7", "write_v7", "episode_rows"):
        assert hasattr(schema_v7, name)
    assert schema_v7.SCHEMA_VERSION_V7 == "robolabel/annotations/v7"
