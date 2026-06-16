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


def test_build_episode_carries_module_block():
    from robolabel.inspect_data import build_episode, module_block
    meta = {"quality": 4, "speed": "fast", "novelty": 0.42, "curation_value": 0.8,
            "curation_tier": "full", "control_modality": "joint"}
    tracks = {"grounded": {"segments": [{"start": 0, "end": 10, "phase": "grasp", "target": "cube"}]},
              "gold": {"segments": [{"start": 0, "end": 10}]}}
    subgoals = [{"segment_idx": 0, "subgoal_frame_idx": 10,
                 "retrieved_subgoal_episode_id": "3", "retrieved_subgoal_frame_idx": 7}]
    ep = build_episode("0", 40, 30.0, "stack", tracks, modules=module_block(meta), subgoals=subgoals)
    assert ep["modules"]["quality"] == 4 and ep["modules"]["speed"] == "fast"
    assert ep["modules"]["control_modality"] == "joint"
    assert ep["thumb"] == 10                              # first subgoal keyframe
    assert ep["subgoals"][0]["retrieved_episode"] == "3" and ep["subgoals"][0]["retrieved_frame"] == 7
    # enabled inferred from populated fields
    assert set(ep["enabled"]) == {"segmentation", "quality", "speed", "subgoals", "control", "novelty", "curation"}


def test_module_block_coerces_nan():
    import numpy as np

    from robolabel.inspect_data import module_block
    m = module_block({"quality": np.nan, "novelty": np.nan, "speed": None, "curation_tier": "nan"})
    assert m["quality"] is None and m["novelty"] is None and m["curation_tier"] is None


def test_gallery_state_exposes_card_fields(tmp_path: Path):
    from robolabel.inspect_data import assemble, build_episode, module_block
    from robolabel.inspect_server import InspectSession, merge_gallery_payloads
    meta = {"quality": 5, "speed": "slow", "novelty": 0.2, "curation_value": 0.75, "curation_tier": "full"}
    ep = build_episode("0", 30, 30.0, "t", {"grounded": {"segments": [{"start": 0, "end": 9, "phase": "grasp"}]}},
                       modules=module_block(meta), subgoals=[{"segment_idx": 0, "subgoal_frame_idx": 9}])
    payload = assemble("ds", "lerobot", ["grounded"], [ep])
    combined, _ = merge_gallery_payloads([{"task": "pour", "payload": payload, "episodes": {}}])
    st = InspectSession(combined).state()
    assert st["gallery"] is True
    e0 = st["episodes"][0]
    assert e0["gallery_task"] == "pour" and e0["modules"]["curation_tier"] == "full" and e0["thumb"] == 9


def test_merge_gallery_payloads():
    from robolabel.inspect_server import merge_gallery_payloads
    pa = {"track_order": ["grounded"], "track_colors": {"grounded": "#2563eb"},
          "episodes": [{"episode_id": "0", "task": "pour water", "num_frames": 10,
                        "tracks": {"grounded": {"segments": []}}}]}
    pb = {"track_order": ["gold", "grounded"], "track_colors": {"gold": "#111", "grounded": "#2563eb"},
          "episodes": [{"episode_id": "0", "num_frames": 5, "tracks": {}}]}
    dummy = object()
    combined, emap = merge_gallery_payloads([
        {"task": "pour", "payload": pa, "episodes": {"0": dummy}},
        {"task": "fold", "payload": pb, "episodes": {}},
    ])
    assert combined["gallery"] is True
    assert [e["episode_id"] for e in combined["episodes"]] == ["pour::0", "fold::0"]
    assert [e["gallery_task"] for e in combined["episodes"]] == ["pour", "fold"]
    assert combined["episodes"][0]["frame_ep"] == "pour::0"
    assert combined["track_order"] == ["grounded", "gold"]   # union, first-seen order
    assert emap["pour::0"] is dummy                            # frames route by prefixed id


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
