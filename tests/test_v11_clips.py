"""Clip folders (robolabel.adapters.clip_folder) and the held-out guard's clip allowlist.

The real-clip tests read the clip folder named by ``ROBOLABEL_CLIPS`` (or ``$ROBOLABEL_DATA/clips``) and
skip when it is not there; they only read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import robolabel.layers.check as check_layer
import robolabel.layers.goal as goal_layer
from robolabel.adapters.clip_folder import ClipFolderSource, clip_key, probe_clip
from robolabel.eval.heldout import (
    HeldoutGuard,
    HeldoutRefused,
    filter_dev,
    is_clip_key,
    load_clip_keys,
    normalize_clip_key,
    require_explicit_episodes,
    write_heldout_ids,
)
from robolabel.layers.signal import Calibration, run_l1
from robolabel.providers.mock import MockProvider

av = pytest.importorskip("av")

HAS_V11_GOAL_AND_CHECKS = all(hasattr(goal_layer, f) for f in ("goal_request_v11", "postprocess_goal_v11")) and \
    hasattr(check_layer, "run_checks_v11")
needs_goal_and_checks = pytest.mark.skipif(not HAS_V11_GOAL_AND_CHECKS,
                                           reason="layers.goal / layers.check v1.1 functions not present yet")


def write_clip(path: Path, n: int = 30, fps: int = 25, size: tuple[int, int] = (64, 48)) -> None:
    """A small mpeg4 clip (a block moving right) written with PyAV, for the tests only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[10:30, i % (w - 10):i % (w - 10) + 10] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@pytest.fixture
def clips(tmp_path):
    root = tmp_path / "clips"
    write_clip(root / "wave" / "clip.mp4", n=30, fps=25)
    write_clip(root / "stack" / "clip.mp4", n=12, fps=10, size=(96, 40))
    (root / "stack" / "task.txt").write_text("\n  stack the cups  \nsecond line\n", encoding="utf-8")
    (root / "stack" / "source.json").write_text(json.dumps({"task": "not this one"}), encoding="utf-8")
    (root / "wave" / "source.json").write_text(json.dumps({"task": "wave at the camera", "truth": {"x": 1}}),
                                               encoding="utf-8")
    (root / "empty").mkdir()  # no clip.mp4: not a clip
    return root


# ------------------------------------------------------------------------------------------ clip folders
def test_clip_folder_discovery_keys_and_tasks(clips):
    src = ClipFolderSource(clips)
    assert src.clip_ids == ["stack", "wave"] and len(src) == 2
    assert src.episode_ids() == ["C/stack", "C/wave"] and clip_key("wave") == "C/wave"
    assert src.task("stack") == "stack the cups"  # task.txt wins over source.json
    assert src.task("wave") == "wave at the camera"
    mapped = ClipFolderSource(clips, ["wave"], keys={"wave": "F3/1821"})
    assert mapped.episode_ids() == ["F3/1821"] and mapped.episode("wave").extra["family"] == "F3"
    assert not src.ready("empty") and src.ready("wave")
    with pytest.raises(FileNotFoundError):
        ClipFolderSource(clips, ["empty"]).episode("empty")
    for bad in ("../x", "a/b", ""):
        with pytest.raises(ValueError):
            ClipFolderSource(clips, [bad])
    with pytest.raises(ValueError):
        ClipFolderSource(clips, "wave")
    with pytest.raises(ValueError):
        ClipFolderSource(clips, ["wave", "wave"])
    with pytest.raises(FileNotFoundError):
        ClipFolderSource(clips / "missing")


def test_clip_episode_frames_rate_and_size(clips):
    src = ClipFolderSource(clips)
    ep = src.episode("wave")
    assert ep.episode_id == "C/wave" and ep.num_frames == 30 and ep.fps == 25.0 and ep.camera_key == "video"
    assert ep.extra["camera_order"] == ["video"] and ep.extra["external_cameras"] == ["video"]
    assert ep.extra["wrist_cameras"] == [] and ep.extra["camera_sizes"] == {"video": [64, 48]}
    frame = ep.frame(5)
    assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
    assert np.array_equal(ep.extra["cameras"]["video"](5), frame)
    assert ep.frame(10_000).shape == (48, 64, 3)  # clamped to the last frame
    report = ep.extra["clip_frames"].report
    assert report["exact"] is True and report["expected"] == report["decoded"] == 30 and report["missing"] == 0
    assert frame[20, 5:15].mean() > frame[20, 40:50].mean()  # the block is at x = 5 in frame 5
    small = ClipFolderSource(clips, max_side=32).episode("stack")
    assert small.num_frames == 12 and small.fps == 10.0 and small.frame(0).shape == (13, 32, 3)
    info = probe_clip(clips / "wave" / "clip.mp4")
    assert info["num_frames"] == 30 and info["codec"] == "mpeg4" and info["duration_s"] == pytest.approx(1.2)
    src.release("wave")


def test_video_part_only_for_short_h264_mp4(clips):
    src = ClipFolderSource(clips)
    assert src.video_part("wave") is None  # mpeg4, not H.264: the caller encodes one
    items = [ep.episode_id for ep in src]
    assert items == ["C/stack", "C/wave"]


def test_decoding_is_repeatable(clips):
    a = ClipFolderSource(clips).episode("wave")
    b = ClipFolderSource(clips).episode("wave")
    assert all(np.array_equal(a.frame(i), b.frame(i)) for i in (0, 7, 29))


# ------------------------------------------------------------------------------------------ the guard
@pytest.fixture
def seal(tmp_path):
    ids = tmp_path / "eval" / "heldout_ids.json"
    write_heldout_ids(ids, {"F1": {"repo_id": "r", "revision": "s", "rule": "x", "episode_keys": ["F1/2"]},
                            "F3": {"repo_id": "r", "revision": "s", "rule": "y", "episode_keys": ["F3/9"]}},
                      created_utc="2026-09-27T09:00:00Z")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("# prereg\n", encoding="utf-8")
    return ids, prereg, tmp_path / "eval" / "heldout_access_log.jsonl"


def lines(log):
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []


def test_clip_keys_on_the_allowlist_are_dev(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg, clip_keys=["C/no_end_state", " C/draw "])
    assert guard.clip_keys == {"C/no_end_state", "C/draw"}
    guard.check(["C/no_end_state", "F1/0", "F3/1821", "C/draw"], command="e2")
    assert not log.exists()
    assert guard.heldout_in(["C/no_end_state", "F1/2"]) == ["F1/2"]
    assert guard.clips_refused(["C/no_end_state", "C/other", "F1/0"]) == ["C/other"]


def test_other_clip_keys_are_refused_and_logged(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg, clip_keys=["C/no_end_state"])
    with pytest.raises(HeldoutRefused, match="clip_not_allowed"):
        guard.check(["C/no_end_state", "C/secret", "F1/0"], command="e2", commit="abc")
    from robolabel.eval.receipts import sha256_bytes

    with pytest.raises(HeldoutRefused):  # a valid final does not unlock a clip off the allowlist
        guard.check(["C/secret", "F1/2"], command="e2 --final", final=sha256_bytes(prereg.read_bytes()), commit="c")
    with pytest.raises(HeldoutRefused):  # no allowlist: every clip key is refused, a numeric one too
        HeldoutGuard(ids, log, prereg).check(["C/5"], command="e2", commit="c")
    first, second, third = lines(log)
    assert first["outcome"] == "refused" and first["reason"] == "clip_not_allowed"
    assert first["clip_keys"] == ["C/secret"] and first["heldout_keys"] == [] and first["n_keys"] == 3
    assert second["clip_keys"] == ["C/secret"] and second["heldout_keys"] == ["F1/2"]
    assert third["clip_keys"] == ["C/5"]


def test_other_families_behave_as_before(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg, clip_keys=["C/no_end_state"])
    with pytest.raises(HeldoutRefused, match="1 held-out"):
        guard.check(["F1/2", "C/no_end_state"], command="e2", commit="abc")
    entry = lines(log)[0]
    assert entry["reason"] == "no_final" and "clip_keys" not in entry and entry["heldout_keys"] == ["F1/2"]
    with pytest.warns(RuntimeWarning, match="F2"):
        guard.check(["F2/5"], command="e2")
    with pytest.raises(ValueError):
        guard.check(["C/"], command="e2")  # a malformed clip key
    with pytest.raises(ValueError):
        guard.check("C/no_end_state", command="e2")


def test_allowlist_method_and_bad_entries(seal):
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg)
    guard.allow_clip_keys(["C/a"])
    guard.allow_clip_keys(("C/b",))
    assert guard.clip_keys == {"C/a", "C/b"}
    guard.check(["C/a", "C/b"], command="e2")
    for bad in (["F3/1821"], "C/a", ["C/"], [3]):
        with pytest.raises(ValueError):
            guard.allow_clip_keys(bad)
    with pytest.raises(ValueError):
        HeldoutGuard(ids, log, prereg, clip_keys=["F1/0"])


def test_the_clip_family_is_case_insensitive(seal):
    ids, prereg, log = seal
    assert is_clip_key("c/12") and is_clip_key(" c/x") and normalize_clip_key("c/No_End") == "C/No_End"
    guard = HeldoutGuard(ids, log, prereg, clip_keys=["c/no_end_state"])
    assert guard.clip_keys == {"C/no_end_state"}
    guard.check(["c/no_end_state", "C/no_end_state"], command="e2")  # the same allowed clip
    assert not log.exists()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "unlisted family c" warning: it is a clip key
        with pytest.raises(HeldoutRefused, match="clip_not_allowed"):
            guard.check(["c/12"], command="e2", commit="abc")
    assert lines(log)[-1]["clip_keys"] == ["C/12"]
    assert guard.clips_refused(["c/12", "c/no_end_state"]) == ["C/12"]
    assert filter_dev(["c/12", "c/no_end_state", "F1/0"], guard) == ["c/no_end_state", "F1/0"]
    with pytest.raises(ValueError):
        filter_dev(["c/12"], {"F1/2"})  # a clip key needs a guard, in either case
    assert require_explicit_episodes(["c/x"], allow_clip_keys=True) == ["C/x"]


def test_clip_key_helpers(tmp_path, seal):
    assert is_clip_key("C/x") and is_clip_key(" C/x") and not is_clip_key("F1/0") and not is_clip_key("C")
    assert normalize_clip_key(" C/no_end_state ") == "C/no_end_state"
    for bad in ("C/", "C/a/b", "C/-x", 5, "F1/0"):
        with pytest.raises(ValueError):
            normalize_clip_key(bad)
    assert require_explicit_episodes(["C/x", "F1/03"], allow_clip_keys=True) == ["C/x", "F1/3"]
    with pytest.raises(ValueError):
        require_explicit_episodes(["C/x"])  # the default keeps the old behavior
    with pytest.raises(ValueError):
        require_explicit_episodes(["C/x", " C/x"], allow_clip_keys=True)
    yml = tmp_path / "clips.yaml"
    yml.write_text("clips:\n  - {id: r, key: F3/1821}\n  - {id: a, key: C/no_end_state}\n  - {id: b, key: C/ego}\n",
                   encoding="utf-8")
    assert load_clip_keys(yml) == ["C/no_end_state", "C/ego"]
    ids, prereg, log = seal
    guard = HeldoutGuard(ids, log, prereg, clip_keys=load_clip_keys(yml))
    assert filter_dev(["C/ego", "C/zzz", "F1/2", "F1/0"], guard) == ["C/ego", "F1/0"]
    with pytest.raises(ValueError):
        filter_dev(["C/ego"], {"F1/2"})


# ------------------------------------------------------------------------------------------ the pipeline on clips
def gripper_l1(n: int, fps: float) -> dict:
    """A synthetic L1 record over n frames (one close at 40 percent, one open at 70 percent): plumbing only."""
    a, b = int(0.4 * n), int(0.7 * n)
    ramp = 8
    cmd = np.array([20.0] * a + list(np.linspace(20, 1, ramp)) + [1.0] * (b - a - ramp)
                   + list(np.linspace(1, 20, ramp)) + [20.0] * (n - b - ramp))
    state, action = np.zeros((n, 6)), np.zeros((n, 6))
    state[:, 5], action[:, 5] = np.clip(cmd, 8.0, 20.0), cmd
    state[:, 0] = np.linspace(0, 90, n)
    cal = Calibration(layout="so101", fps=fps, cmd_open=20.0, cmd_closed=1.0, meas_open=20.0, meas_closed=1.0,
                      pause_speed=5.0, withdraw_threshold=5.0)
    return run_l1(state, action, cal, episode_key="C/clip", family="C")


@needs_goal_and_checks
@pytest.mark.parametrize("source", ["none", "gripper"])
def test_pipeline_on_a_clip_folder(clips, source):
    from robolabel.vfirst import run_episode_v11

    ep = ClipFolderSource(clips).episode("wave")
    out = run_episode_v11(ep, camera="video", caller=MockProvider(), event_source=source,
                          l1=gripper_l1(ep.num_frames, ep.fps) if source == "gripper" else None,
                          context={"arm": "E2-main", "episode_key": ep.episode_id})
    assert out["view"]["episode_key"] == "C/wave" and out["view"]["family"] == "C"
    assert out["view"]["cameras"] == ["video"] and out["view"]["event_sources"] == [source]
    assert out["segments"][-1]["end_frame"] == 29


def real_clips_root() -> Path | None:
    for value in (os.environ.get("ROBOLABEL_CLIPS"),
                  os.environ.get("ROBOLABEL_DATA") and str(Path(os.environ["ROBOLABEL_DATA"]) / "clips")):
        if value and (Path(value) / "robot_f3_1821" / "clip.mp4").is_file():
            return Path(value)
    return None


needs_real_clip = pytest.mark.skipif(real_clips_root() is None,
                                     reason="no clip folder robot_f3_1821 (set ROBOLABEL_CLIPS or ROBOLABEL_DATA)")


@needs_real_clip
def test_real_robot_clip_reads_as_one_episode():
    src = ClipFolderSource(real_clips_root(), ["robot_f3_1821"], keys={"robot_f3_1821": "F3/1821"})
    ep = src.episode("robot_f3_1821")
    assert ep.episode_id == "F3/1821" and ep.num_frames == 474 and ep.fps == 20.0
    assert ep.extra["camera_sizes"] == {"video": [1024, 576]} and ep.task
    assert ep.frame(473).shape == (288, 512, 3)
    assert ep.extra["clip_frames"].report["exact"] is True
    part = src.video_part("robot_f3_1821")
    assert part is not None and part.mime == "video/mp4" and part.seconds == pytest.approx(23.7, abs=0.05)
    assert part.data[4:8] == b"ftyp"


@needs_real_clip
@needs_goal_and_checks
@pytest.mark.parametrize("source,mode", [("none", "frames"), ("gripper", "frames"), ("none", "video")])
def test_real_robot_clip_through_the_pipeline_with_the_mock_provider(source, mode):
    from robolabel.vfirst import run_episode_v11

    src = ClipFolderSource(real_clips_root(), ["robot_f3_1821"], keys={"robot_f3_1821": "F3/1821"})
    ep = src.episode("robot_f3_1821")
    video = src.video_part("robot_f3_1821") if mode == "video" else None
    out = run_episode_v11(ep, camera="video", caller=MockProvider(), event_source=source,
                          l1=gripper_l1(ep.num_frames, ep.fps) if source == "gripper" else None,
                          coarse_mode=mode, video=video, context={"arm": "E2-main", "episode_key": ep.episode_id})
    view = out["view"]
    assert view["episode_key"] == "F3/1821" and view["num_frames"] == 474 and view["coarse_mode"] == mode
    coarse_calls = [c for c in out["calls"] if c.receipt.get("step") == "coarse"]
    assert len(coarse_calls) == 1
    inv = view["inventory_frames"]
    assert inv[0] == 0 and inv[-1] == 473 and len(inv) == 8
    json.dumps(view)
    src.release("robot_f3_1821")
