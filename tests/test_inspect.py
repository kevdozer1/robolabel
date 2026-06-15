"""Tests for the inspect viewer data layer + an HTTP round-trip."""

from __future__ import annotations

import json
import urllib.request
from contextlib import closing
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from robolabel.inspect_data import assemble, build_episode, segments_from_records
from robolabel.inspect_server import InspectSession, make_handler, parse_episodes


def test_parse_episodes():
    assert parse_episodes(None) is None
    assert parse_episodes("") is None
    assert parse_episodes("0-7") == [0, 1, 2, 3, 4, 5, 6, 7]
    assert parse_episodes("0,2,5") == [0, 2, 5]
    assert parse_episodes("3") == [3]
    assert parse_episodes("0-2,5") == [0, 1, 2, 5]


def _tracks():
    gold = [{"start": 0, "end": 10}, {"start": 11, "end": 20}, {"start": 21, "end": 30}]
    good = [{"start": 0, "end": 10, "phase": "approach", "text": "x", "evidence": "gripper above"},
            {"start": 11, "end": 20, "phase": "grasp", "text": "y", "evidence": "closes"},
            {"start": 21, "end": 30, "phase": "transport", "text": "z", "evidence": "lifted"}]
    return {"gold": {"segments": gold}, "grounded": {"segments": good}}


def test_build_episode_metrics_vs_gold():
    ep = build_episode("0", 31, 30.0, "t", _tracks(), gate_flags=["uniform_split"])
    m = ep["metrics"]["grounded"]
    assert m["iou"] == 1.0                       # identical to gold
    assert m["boundary_recall"] == 1.0           # both internal boundaries (10, 20) matched
    assert ep["n_flags"] == 1
    assert ep["sort_iou"] == 1.0


def test_segments_from_records_maps_fields():
    rows = [{"segment_idx": 1, "start_frame": 5, "end_frame": 9, "subtask_text": "b", "phase": "grasp",
             "boundary_evidence": "closes"},
            {"segment_idx": 0, "start_frame": 0, "end_frame": 4, "subtask_text": "a", "phase": "approach",
             "boundary_evidence": None}]
    segs = segments_from_records(rows)
    assert [s["start"] for s in segs] == [0, 5]   # sorted by segment_idx
    assert segs[0]["phase"] == "approach" and segs[0]["evidence"] is None
    assert segs[1]["evidence"] == "closes"


def test_segments_from_records_coerces_nan():
    # Regression: empty parquet columns (e.g. baseline S0's phase/evidence) read back as
    # float NaN, which is truthy — must become None, not the string "nan".
    nan = float("nan")
    segs = segments_from_records([{"segment_idx": 0, "start_frame": 0, "end_frame": 9,
                                   "subtask_text": "reach", "phase": nan, "boundary_evidence": nan}])
    assert segs[0]["phase"] is None
    assert segs[0]["evidence"] is None
    assert segs[0]["text"] == "reach"


def test_inspect_http_roundtrip(tmp_path: Path):
    payload = assemble("ds", "lerobot", ["gold", "grounded"],
                       [build_episode("0", 31, 30.0, "t", _tracks())])
    data_path = tmp_path / "inspect.json"
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    session = InspectSession(data_path)  # no source -> no frames, fine
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with closing(urllib.request.urlopen(base + "/api/state")) as r:
            state = json.loads(r.read())
        assert state["track_order"] == ["gold", "grounded"]
        assert len(state["episodes"]) == 1
        with closing(urllib.request.urlopen(base + "/api/episode/0")) as r:
            ep = json.loads(r.read())
        assert ep["metrics"]["grounded"]["iou"] == 1.0
    finally:
        server.shutdown()
        server.server_close()


def test_blind_grade_roundtrip(tmp_path: Path):
    payload = assemble("ds", "lerobot", ["model"],
                       [build_episode("T0", 31, 30.0, "t", {"model": _tracks()["grounded"]})], blind=True)
    data_path = tmp_path / "blind.json"
    data_path.write_text(json.dumps(payload), encoding="utf-8")
    grades = tmp_path / "grades.json"
    session = InspectSession(data_path, grades_path=grades)
    res = session.save_grade({"episode_id": "T0", "track": "model",
                              "marks": {"b0": True, "p0": True, "e0": False}, "verdict": "usable"})
    assert res["saved"] is True
    saved = json.loads(grades.read_text())
    assert saved["T0"]["verdict"] == "usable"
    assert session.state()["graded_count"] == 1
