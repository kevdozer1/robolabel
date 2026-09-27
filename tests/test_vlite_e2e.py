"""V-lite end to end with the mock provider (no network): L1 to L5, view record, schema v7 rows."""

from __future__ import annotations

import json

import numpy as np

from robolabel.episode import Episode
from robolabel.layers.signal import Calibration, run_l1
from robolabel.providers.mock import MockProvider
from robolabel.schema import read_annotations
from robolabel.schema_v7 import write_v7
from robolabel.vlite import run_episode

VIEW_KEYS = {"arm", "episode_key", "family", "fps", "num_frames", "cameras", "camera_sizes", "task", "objects",
             "segments", "coarse", "attempts", "goal", "episode_outcome", "checks", "risk", "routed", "cost_usd",
             "calls", "wall_s", "valid", "repairs", "no_output"}


def synthetic_episode(n: int = 120):
    rng = np.random.default_rng(3)
    frames = {c: rng.integers(0, 255, size=(n, 60, 80, 3), dtype=np.uint8)
              for c in ("observation.images.up", "observation.images.side", "observation.images.wrist")}
    cams = {c: (lambda i, _c=c: frames[_c][int(i)]) for c in frames}
    cmd = np.array([20.0] * 30 + list(np.linspace(20, 1, 10)) + [1.0] * 40 + list(np.linspace(1, 20, 10)) + [20.0] * 30)
    meas = np.array([20.0] * 30 + list(np.linspace(20, 8, 10)) + [8.0] * 40 + list(np.linspace(8, 20, 10)) + [20.0] * 30)
    state = np.zeros((n, 6))
    action = np.zeros((n, 6))
    state[:, 5], action[:, 5] = meas, cmd
    state[:, 0] = np.linspace(0, 90, n)
    ep = Episode(episode_id="F1/999", num_frames=n, fps=30.0, task="pink lego brick into the transparent box",
                 get_frame=cams["observation.images.up"], camera_key="observation.images.up",
                 extra={"family": "F1", "cameras": cams, "camera_order": list(frames),
                        "external_cameras": ["observation.images.up", "observation.images.side"],
                        "wrist_cameras": ["observation.images.wrist"],
                        "camera_sizes": {c: [80, 60] for c in frames}, "state": state, "action": action})
    cal = Calibration(layout="so101", fps=30.0, cmd_open=20.0, cmd_closed=1.0, meas_open=20.0, meas_closed=1.0,
                      pause_speed=5.0, withdraw_threshold=5.0)
    return ep, run_l1(state, action, cal, episode_key="F1/999", family="F1")


def test_vlite_mock_end_to_end(tmp_path):
    ep, l1 = synthetic_episode()
    out = run_episode(ep, l1, MockProvider(), arm="v@mock", model_key="mock", bucket="sweep")
    view = out["view"]
    assert VIEW_KEYS <= set(view)
    assert view["calls"] == 4 and view["valid"] is True and view["cost_usd"] == 0.0
    segs = view["segments"]
    assert segs[0]["start"] == 0 and segs[-1]["end"] == ep.num_frames - 1
    assert all(b["start"] == a["end"] + 1 for a, b in zip(segs, segs[1:], strict=False))
    assert {c["rule_id"] for c in view["checks"]} == set(range(1, 11))
    assert view["goal"] is not None
    robot = [r for r in view["goal"]["requirements"] if r["kind"] == "robot_end_state"]
    assert {r["predicate"] for r in robot} >= {"holding"}  # mandatory robot slots present
    json.dumps(view)  # the view record is plain JSON
    write_v7(out["rows"], tmp_path)
    df = read_annotations(tmp_path)
    assert {"episode_metadata", "subtask", "coarse_subtask", "check"} <= set(df["record_type"])
    legacy_types = df[df["record_type"].isin(["episode_metadata", "subtask", "subgoal"])]
    assert len(legacy_types) >= 2  # older readers filter on record_type and still find their rows


def test_vlite_is_deterministic():
    ep, l1 = synthetic_episode()
    a = run_episode(ep, l1, MockProvider(), arm="v@mock", model_key="mock")["view"]
    b = run_episode(ep, l1, MockProvider(), arm="v@mock", model_key="mock")["view"]
    a.pop("wall_s")
    b.pop("wall_s")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


class FailingCaller:
    name = "fake"
    model = "fake"

    def __init__(self, fail_steps):
        self.fail = set(fail_steps)
        self.mock = MockProvider()

    def call(self, req):
        from robolabel.providers.base import CallResult

        if req.step in self.fail:
            return CallResult(False, None, "", "invalid", error="invalid after repair")
        return self.mock.call(req)


def test_failed_calls_follow_vlite_rules():
    ep, l1 = synthetic_episode()
    out = run_episode(ep, l1, FailingCaller({"segments", "scene_inventory"}), arm="v@fake", model_key="fake")
    view = out["view"]
    assert len(view["segments"]) == 1 and view["segments"][0]["end"] == ep.num_frames - 1  # missing-output segment
    assert view["valid"] is False and view["calls"] == 4
    rule9 = next(c for c in view["checks"] if c["rule_id"] == 9)
    assert rule9["verdict"] == "na"  # no inventory
    out2 = run_episode(ep, l1, FailingCaller({"goal"}), arm="v@fake", model_key="fake")
    assert out2["view"]["goal"] is None and out2["view"]["episode_outcome"] == "unknown"


def test_failed_calls_are_recorded_not_routed():
    """V_LITE "When one call of an episode fails": every case is counted in repairs and validity; L5 routes
    only on its three listed conditions (review:vlite 9 not adopted, SPEC_QUESTIONS Q155)."""
    ep, l1 = synthetic_episode()
    view = run_episode(ep, l1, FailingCaller({"segments", "scene_facts"}), arm="v@fake", model_key="fake")["view"]
    assert view["valid"] is False
    assert any(r.startswith("segments invalid") for r in view["repairs"])
    assert any(r.startswith("scene_facts invalid") for r in view["repairs"])
    assert not any("missing model output" in r for r in view["route_reasons"])
    both = run_episode(ep, l1, FailingCaller({"segments", "goal"}), arm="v@fake", model_key="fake")["view"]
    assert both["no_output"] is True


def test_view_and_rows_carry_the_pipeline_code():
    """review:vlite 0: the view and the episode_metadata row name the code that wrote them."""
    from robolabel.vlite import PIPELINE_CODE, PIPELINE_VERSION

    ep, l1 = synthetic_episode()
    out = run_episode(ep, l1, MockProvider(), arm="v@mock", model_key="mock")
    assert out["view"]["pipeline_code"] == PIPELINE_CODE and out["view"]["pipeline_version"] == PIPELINE_VERSION
    meta = [r for r in out["rows"] if r["record_type"] == "episode_metadata"]
    assert len(meta) == 1 and meta[0]["pipeline_code"] == PIPELINE_CODE
    assert meta[0]["pipeline_version"] == PIPELINE_VERSION
