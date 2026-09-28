"""v1.1 event sources (SPEC_V1_1 3.1): none, motion and gripper, and the candidate lines."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robolabel.episode import Episode
from robolabel.events import (
    EVENT_TYPES,
    GripperSource,
    MotionSource,
    NoneSource,
    candidate_lines,
    candidate_map,
    dumps_events,
    events_from_l1,
    get_source,
    make_event,
    motion_signal,
    sort_events,
    validate_event,
)
from robolabel.events.motion import (
    diff_sums,
    gray_small,
    min_pause_frames,
    pause_runs,
    percentile,
    smooth,
)
from robolabel.layers.signal import Calibration, run_l1

H, W = 96, 160  # long side above 128, so the resize runs with a non-integer factor
PERIOD = 16


# ------------------------------------------------------------------------------------------------ fixtures
def _grating(phase: int, noise_seed: int | None = None, noise: int = 0) -> np.ndarray:
    """A vertical sinusoidal grating shifted by ``phase`` pixels, optional per-frame noise (seeded)."""
    x = np.arange(W, dtype=np.float64)
    row = 128.0 + 100.0 * np.sin(2.0 * np.pi * (x - phase) / PERIOD)
    img = np.repeat(np.round(row)[None, :], H, axis=0)
    rgb = np.stack([img, np.roll(img, 3, axis=1), img[:, ::-1]], axis=-1)
    if noise and noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        rgb = rgb + rng.integers(-noise, noise + 1, size=rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _phases(n: int, still: list[tuple[int, int]], step: int | list[int] = 3) -> list[int]:
    """Grating phase per frame: it advances on every frame except inside the still stretches, so the
    frames of a stretch ``(a, b)`` are identical to frame ``a - 1``."""
    out, p = [], 0
    for i in range(n):
        if i > 0 and not any(a <= i <= b for a, b in still):
            p += step if isinstance(step, int) else step[i]
        out.append(p)
    return out


class _Guarded(dict):
    """A dict that fails the test when the robot signal (truth for E1) is read from it."""

    FORBIDDEN = ("state", "action")

    def __getitem__(self, key):
        if key in self.FORBIDDEN:
            raise AssertionError(f"read episode.extra[{key!r}]")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.FORBIDDEN:
            raise AssertionError(f"read episode.extra.get({key!r})")
        return super().get(key, default)


class _NoTouch(dict):
    """An L1 record that must never be read."""

    def __getitem__(self, key):
        raise AssertionError(f"read l1[{key!r}]")

    def get(self, key, default=None):
        raise AssertionError(f"read l1.get({key!r})")


def clip(n: int, still: list[tuple[int, int]], *, fps: float = 30.0, noise: int = 0, step=3,
         calls: list[int] | None = None, episode_id: str = "T/0") -> Episode:
    phases = _phases(n, still, step)

    def get(i: int) -> np.ndarray:
        if calls is not None:
            calls.append(int(i))
        return _grating(phases[int(i)], noise_seed=1000 + int(i), noise=noise)

    extra = _Guarded({"cameras": {"observation.images.up": get, "observation.images.wrist": lambda i: get(0)},
                      "camera_order": ["observation.images.up", "observation.images.wrist"],
                      "state": np.zeros((n, 6)), "action": np.zeros((n, 6))})
    return Episode(episode_id=episode_id, num_frames=n, fps=fps, task="t", get_frame=get,
                   camera_key="observation.images.up", extra=extra)


# ------------------------------------------------------------------------------------------------ motion
def test_motion_finds_the_starts_and_ends_of_two_still_stretches():
    # Frames 29..49 and 79..99 are identical, so the per-frame difference is 0 on frames 30..49 and
    # 80..99. With 3-frame smoothing the smoothed value is 0 on frames 31..48 and 81..98, which are 36
    # of 120 frames, so the 20th percentile is 0 and those runs (18 frames >= round(0.3 * 30) = 9) are
    # the pauses: pause_start at each run's first frame, pause_end at the frame after its last.
    ep = clip(120, [(30, 49), (80, 99)])
    events = MotionSource().events(ep, camera="observation.images.up")
    assert [(e["type"], e["frame"]) for e in events] == [
        ("pause_start", 31), ("pause_end", 49), ("pause_start", 81), ("pause_end", 99)]
    assert all(e["source"] == "motion" and e["attempt_idx"] is None and e["confidence"] == 1.0 for e in events)
    assert all(validate_event(e) == [] for e in events)


def test_motion_with_noise_lands_near_the_stretch_edges():
    # Every frame carries seeded pixel noise and the grating moves by 2 to 5 px per frame; the two still
    # stretches cover 15 percent of the clip. The pauses start and end within 2 frames of the stretches.
    n, still = 200, [(50, 64), (130, 144)]
    steps = [int(s) for s in np.random.default_rng(7).integers(2, 6, size=n)]
    ep = clip(n, still, fps=20.0, noise=2, step=steps)
    events = MotionSource().events(ep, camera="observation.images.up")
    starts = [e["frame"] for e in events if e["type"] == "pause_start"]
    ends = [e["frame"] for e in events if e["type"] == "pause_end"]
    assert len(starts) == 2 and len(ends) == 2, events
    for (a, b), s, e in zip(still, starts, ends, strict=True):
        assert abs(s - a) <= 2 and abs(e - (b + 1)) <= 2, (a, b, s, e)
    assert all(0.0 < ev["confidence"] <= 1.0 for ev in events)


def test_motion_is_byte_identical_on_repeated_runs():
    n, still = 200, [(50, 64), (130, 144)]
    steps = [int(s) for s in np.random.default_rng(7).integers(2, 6, size=n)]
    runs = []
    for _ in range(2):
        ep = clip(n, still, fps=20.0, noise=2, step=steps)  # a fresh episode and source each time
        src = get_source("motion")
        runs.append((json.dumps(src.events(ep, camera="observation.images.up")),
                     dumps_events(src.events(ep, camera="observation.images.up")),
                     json.dumps(motion_signal(ep, "observation.images.up"), sort_keys=True)))
    assert runs[0] == runs[1]
    assert runs[0][0] != "[]"


def _int_frame(phase: int, i: int) -> np.ndarray:
    """A sawtooth grating shifted by ``phase`` pixels, with a fixed pattern of small offsets per frame ``i``. Integer
    arithmetic only, so the frames are the same bytes on every machine and numpy version."""
    y = np.arange(H, dtype=np.int64)[:, None]
    x = np.arange(W, dtype=np.int64)[None, :]
    base = ((x + phase) % 32) * 8 + (y % 4)
    offset = (x * 73 + y * 151 + i * 199) % 7 - 3
    g = np.clip(base + offset, 0, 255)
    return np.stack([g, g // 2 + 40, 255 - g], axis=-1).astype(np.uint8)


def _int_clip(n: int, still: list[tuple[int, int]], fps: float) -> Episode:
    phases, p = [], 0
    for i in range(n):
        if i > 0 and not any(a <= i <= b for a, b in still):
            p += 2 + (i * 5) % 4
        phases.append(p)

    def get(i: int) -> np.ndarray:
        return _int_frame(phases[int(i)], int(i))

    return Episode(episode_id="T/9", num_frames=n, fps=fps, task="t", get_frame=get,
                   camera_key="observation.images.up", extra={"cameras": {"observation.images.up": get}})


def test_motion_matches_pinned_digests_on_an_integer_clip():
    """Golden values: the events and the whole signal of a clip built with integer arithmetic only. The motion
    source is integer arithmetic until its final divisions, so these digests hold across machines and numpy
    versions; a change here is a change of the source (bump MOTION_VERSION)."""
    import hashlib

    ep = _int_clip(150, [(40, 55), (100, 114)], 20.0)
    events = get_source("motion").events(ep, camera="observation.images.up")
    assert [(e["type"], e["frame"], e["confidence"]) for e in events] == [
        ("pause_start", 40, 0.8238), ("pause_end", 56, 0.8238), ("pause_start", 101, 0.8397),
        ("pause_end", 115, 0.8397)]
    sig = motion_signal(ep, "observation.images.up")
    assert sig["version"] == "motion-2026-09-27.1" and sig["pixels"] == 9856
    assert hashlib.sha256(dumps_events(events).encode("utf-8")).hexdigest() == \
        "b39f53fc436018e767c1622a5367e734ee454ab7c35fcd3b692211c9b950949d"
    assert hashlib.sha256(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest() == \
        "3cdd6a28320cb1293027c5e0b598f6ee0fa0398bd11e392f220c98f5b865a159"


def test_motion_pause_at_the_clip_edges_gives_no_boundary_at_frame_0_or_past_the_end():
    ep = clip(120, [(1, 15), (105, 119)])
    events = MotionSource().events(ep)
    assert [(e["type"], e["frame"]) for e in events] == [("pause_end", 15), ("pause_start", 106)]


def test_motion_constant_signal_has_no_pauses():
    a, b = _grating(0), _grating(5)
    ep = Episode(episode_id="T/2", num_frames=60, fps=30.0, task=None,
                 get_frame=lambda i: a if i % 2 == 0 else b)  # the same change on every frame
    sig = motion_signal(ep)
    assert len(set(sig["smoothed"])) == 1 and sig["smoothed"][0] > 0
    assert MotionSource().events(ep) == []
    still = clip(40, [(1, 39)])  # nothing moves at all
    assert MotionSource().events(still) == []


def test_motion_reads_frames_once_in_order_and_never_the_robot_signal():
    calls: list[int] = []
    ep = clip(90, [(30, 49)], calls=calls)
    MotionSource().events(ep, camera="observation.images.up", l1=_NoTouch())
    assert calls == list(range(90))
    assert NoneSource().events(ep, l1=_NoTouch()) == []


def test_motion_camera_selection():
    ep = clip(40, [(10, 30)])
    wrist = motion_signal(ep, "observation.images.wrist")  # a constant frame: no motion at all
    assert set(wrist["diff_sums"]) == {0} and wrist["camera"] == "observation.images.wrist"
    assert motion_signal(ep)["camera"] == "observation.images.up"  # the episode's own camera
    with pytest.raises(KeyError):
        MotionSource().events(ep, camera="observation.images.side")
    bare = Episode(episode_id="T/1", num_frames=3, fps=10.0, task=None, get_frame=lambda i: _grating(i))
    assert motion_signal(bare)["num_frames"] == 3  # no camera dict: the episode's own getter


def test_gray_small_resizes_deterministically_with_integer_box_averaging():
    assert gray_small(np.zeros((96, 160, 3), np.uint8)).shape == (77, 128)
    assert gray_small(np.zeros((480, 640, 3), np.uint8)).shape == (96, 128)
    assert gray_small(np.zeros((576, 1024, 3), np.uint8)).shape == (72, 128)
    small = np.full((50, 100, 3), 200, np.uint8)
    assert gray_small(small).shape == (50, 100) and int(gray_small(small)[0, 0]) == 200  # never upscaled
    # a 2 x 4 block of pixels 0 and 255 averages to 127.5, rounded half up to 128
    img = np.zeros((2, 256, 3), np.uint8)
    img[:, 1::2] = 255
    out = gray_small(img, long_side=128)
    assert out.shape == (1, 128) and set(out.ravel().tolist()) == {128}
    # grayscale weights 299, 587, 114 over 1000
    px = np.array([[[255, 0, 0]]], np.uint8)
    assert int(gray_small(px)[0, 0]) == 76  # 255 * 0.299 = 76.2
    frame = np.random.default_rng(3).integers(0, 256, size=(480, 640, 3)).astype(np.uint8)
    assert gray_small(frame).tobytes() == gray_small(frame.copy()).tobytes()


def test_percentile_matches_numpy_linear():
    rng = np.random.default_rng(11)
    for n in (1, 2, 5, 17, 100, 301):
        v = rng.random(n).tolist()
        for q in (0, 20, 50, 100):
            assert percentile(v, q) == pytest.approx(float(np.percentile(v, q)), abs=1e-12)


def test_smoothing_runs_and_min_length():
    assert smooth([3, 3, 6, 0], 3) == [1.0, 4 / 3, 1.0, 1.0]
    assert pause_runs([5, 1, 1, 1, 5, 1, 5], 1.0, 2) == [(1, 3)]
    assert pause_runs([2, 2, 2], 2.0, 1) == []  # threshold equals the largest value: no pauses
    assert min_pause_frames(30) == 9 and min_pause_frames(20) == 6 and min_pause_frames(10) == 3
    sums, npix = diff_sums(lambda i: np.zeros((4, 4, 3), np.uint8), 1)
    assert sums == [0] and npix == 16


# ------------------------------------------------------------------------------------------------ gripper
def _record() -> dict:
    return {
        "num_frames": 200,
        "events": [{"type": "closing", "onset": 0, "offset": 2}, {"type": "opening", "onset": 3, "offset": 8},
                   {"type": "closing", "onset": 40, "offset": 45}, {"type": "closing", "onset": 55, "offset": 58},
                   {"type": "opening", "onset": 80, "offset": 86}, {"type": "closing", "onset": 120, "offset": 125},
                   {"type": "opening", "onset": 140, "offset": 146},
                   {"type": "closing", "onset": 180, "offset": 185}],
        "attempts": [{"attempt_idx": 1, "closing_onset": 40, "closing_offset": 45,
                      "recloses": [{"onset": 55, "offset": 58}], "opening_onset": 80, "end_frame": 79},
                     {"attempt_idx": 2, "closing_onset": 120, "closing_offset": 125, "recloses": [],
                      "opening_onset": 140, "end_frame": 139}],
        "rest_closes": [{"closing_onset": 180}],
        "candidates": [{"frame": 39, "transition": "approach->grasp", "confidence": "high", "attempt_idx": 1},
                       {"frame": 64, "transition": "grasp->transport", "confidence": "low", "attempt_idx": 1},
                       {"frame": 79, "transition": "transport->release", "confidence": "high", "attempt_idx": 1},
                       {"frame": 150, "transition": "release->retract", "confidence": "low", "attempt_idx": 2}],
    }


def test_gripper_events_from_an_l1_record():
    events = events_from_l1(_record())
    assert [(e["type"], e["frame"], e["attempt_idx"]) for e in events] == [
        ("open_start", 3, None),       # an opening before any close
        ("close_start", 40, 1),
        ("close_start", 55, 1),        # a re-close inside attempt 1
        ("arm_move", 65, 1),           # candidate frame 64 ends the earlier segment: onset 65
        ("open_start", 80, 1),
        ("close_start", 120, 2),
        ("open_start", 140, 2),
        ("arm_move", 151, 2),
        ("close_start", 180, None),    # the rest close belongs to no attempt
    ]                                  # the close at frame 0 cannot start a segment and is dropped
    assert all(e["source"] == "gripper" and validate_event(e) == [] for e in events)
    assert {e["confidence"] for e in events if e["type"] != "arm_move"} == {0.9}
    assert {e["confidence"] for e in events if e["type"] == "arm_move"} == {0.3}
    no_move = events_from_l1(_record(), include_arm_move=False)
    assert "arm_move" not in {e["type"] for e in no_move}
    assert GripperSource().events(None, l1=_record()) == events
    assert get_source("gripper", include_arm_move=False).events(None, l1=_record()) == no_move


def _so101_episode(n: int = 200) -> tuple[Episode, Calibration]:
    cmd = np.full(n, 40.0)
    cmd[40:50] = np.linspace(40.0, 5.0, 10)
    cmd[50:100] = 5.0
    cmd[100:110] = np.linspace(5.0, 40.0, 10)
    meas = np.maximum(cmd, 15.0)  # the fingers stall on the object: a hold
    arm = np.zeros((n, 5))
    arm[60:90, 0] = np.linspace(0.0, 60.0, 30)
    arm[90:, 0] = 60.0
    arm[120:150, 1] = np.linspace(0.0, 60.0, 30)
    arm[150:, 1] = 60.0
    state = np.column_stack([arm, meas])
    action = np.column_stack([arm, cmd])
    cal = Calibration(layout="so101", fps=30.0, cmd_open=40.0, cmd_closed=0.0, meas_open=40.0, meas_closed=0.0,
                      pause_speed=20.0, withdraw_threshold=20.0)
    ep = Episode(episode_id="F1/0", num_frames=n, fps=30.0, task="t", get_frame=lambda i: np.zeros((4, 4, 3)),
                 extra={"state": state, "action": action, "family": "F1"})
    return ep, cal


def test_gripper_source_wraps_run_l1():
    ep, cal = _so101_episode()
    rec = run_l1(ep.extra["state"], ep.extra["action"], cal, episode_key="F1/0", family="F1")
    from_record = get_source("gripper").events(ep, l1=rec)
    from_calibration = GripperSource(cal).events(ep)
    assert from_record == from_calibration
    closes = [e for e in from_record if e["type"] == "close_start"]
    opens = [e for e in from_record if e["type"] == "open_start"]
    assert [e["frame"] for e in closes] == [r["onset"] for r in rec["events"] if r["type"] == "closing"]
    assert [e["frame"] for e in opens] == [r["onset"] for r in rec["events"] if r["type"] == "opening"]
    assert [e["attempt_idx"] for e in closes + opens] == [1, 1]
    assert rec["attempts"][0]["outcome"] == "released"
    moves = [e for e in from_record if e["type"] == "arm_move"]
    low = [c for c in rec["candidates"] if c["confidence"] == "low"]
    assert [e["frame"] for e in moves] == [c["frame"] + 1 for c in low] and moves
    assert from_record == sort_events(from_record)
    with pytest.raises(ValueError):
        GripperSource().events(ep)  # neither an L1 record nor a calibration
    with pytest.raises(ValueError):
        GripperSource(cal).events(Episode("F1/9", 5, 30.0, None, lambda i: np.zeros((2, 2, 3))))


# ------------------------------------------------------------------------------------------------ contract
def test_get_source_names_and_the_none_source():
    for name in ("none", "motion", "gripper"):
        src = get_source(name)
        assert src.name == name and src.version
    with pytest.raises(ValueError):
        get_source("imu")

    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"read episode.{name}")

    assert get_source("none").events(Untouchable(), camera="x", l1=_NoTouch()) == []


def test_candidate_lines_and_ids():
    events = [make_event("pause_start", 152, 0.5, "motion"), make_event("close_start", 160, 0.9, "gripper", 1),
              make_event("pause_end", 400, 0.5, "motion"), make_event("open_start", 210, 0.9, "gripper", 1)]
    lines = candidate_lines(events, 303, 30.0)
    assert lines == ["c1: frame 152 (5.07 s), pause_start (motion)",
                     "c2: frame 160 (5.33 s), close_start (gripper)",
                     "c4: frame 210 (7.00 s), open_start (gripper)"]  # frame 400 is past the end: no line
    assert candidate_map(events)["c4"]["frame"] == 210 and len(candidate_map(events)) == 4
    assert candidate_lines([], 100, 30.0) == []


def test_event_helpers():
    ev = make_event("pause_end", np.int64(12), 1.7, "motion")
    assert ev == {"type": "pause_end", "frame": 12, "confidence": 1.0, "source": "motion", "attempt_idx": None}
    assert type(ev["frame"]) is int
    with pytest.raises(ValueError):
        make_event("contact_start", 3, 0.5, "motion")
    assert validate_event({"type": "x"}) != []
    evs = [make_event("pause_end", 5, 0.2, "motion"), make_event("pause_start", 5, 0.2, "motion"),
           make_event("close_start", 2, 0.9, "gripper", 1)]
    assert [(e["frame"], e["type"]) for e in sort_events(evs)] == [
        (2, "close_start"), (5, "pause_start"), (5, "pause_end")]
    assert set(EVENT_TYPES) == {"close_start", "open_start", "arm_move", "pause_start", "pause_end"}
