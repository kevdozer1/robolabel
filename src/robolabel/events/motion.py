"""The ``motion`` event source (SPEC_V1_1 3.1): a free, deterministic pixel pseudo-signal from one camera.

Per frame, the mean absolute difference to the previous frame, on grayscale frames downscaled so the
long side is 128 px, then a 3-frame centered moving average. A pause is a run of at least
``round(0.3 * fps)`` frames whose smoothed value is at or below the clip's 20th percentile. Each pause
gives a ``pause_start`` event at its first frame and a ``pause_end`` event at the frame after its last.

Every step is integer arithmetic until the final divisions, so identical frames give byte-identical
events on any machine:

* grayscale: ``299 R + 587 G + 114 B`` (ITU-R BT.601 weights over 1000), kept as integers;
* resize: box averaging over integer bin edges (output pixel ``j`` of an axis of ``M`` input pixels and
  ``m`` output pixels covers input pixels ``floor(j M / m)`` to ``floor((j + 1) M / m) - 1``), each
  output pixel rounded half up to an integer from 0 to 255; frames whose long side is at most 128 px
  are not resized;
* difference: the sum of absolute pixel differences, an integer; frame 0 repeats frame 1's value;
* smoothing: the window is cut at the clip's ends (frame 0 averages frames 0 and 1);
* percentile: linear interpolation between order statistics (numpy's default method), computed here.

Conventions for edge cases (SPEC_QUESTIONS Q163): "below" means at or below the threshold, so frames
that are exactly still (identical frames, common in renders) count even when they make up more than 20
percent of the clip; a clip whose threshold equals its largest value (a constant signal) has no pauses;
a pause that starts at frame 0 gives no ``pause_start`` and one that reaches the last frame gives no
``pause_end`` (neither is a boundary). The confidence of both events of a pause is how far the pause's
mean smoothed value lies below the threshold, ``1 - mean / threshold`` (1.0 when the threshold is 0).

The source reads video frames only: never the robot state, the action or an L1 record.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..episode import _as_rgb_uint8
from .base import EventSource, make_event, sort_events

MOTION_VERSION = "motion-2026-09-27.1"
LONG_SIDE = 128
PERCENTILE = 20
MIN_PAUSE_S = 0.3
_LUMA = (299, 587, 114)


# ------------------------------------------------------------------------------------------------ frames
def frame_getter(episode: Any, camera: str | None = None) -> tuple[Callable[[int], Any], str | None]:
    """(getter, camera name) for one camera of an episode.

    With ``camera`` the getter is ``episode.extra["cameras"][camera]``. Without it, the episode's own
    camera (``camera_key``) when the adapter lists it, else ``episode.get_frame``.
    """
    cams = (getattr(episode, "extra", None) or {}).get("cameras") or {}
    if camera is not None:
        if camera not in cams:
            raise KeyError(f"camera {camera!r} is not one of the episode's cameras {sorted(cams)}")
        return cams[camera], camera
    key = getattr(episode, "camera_key", None)
    if key is not None and key in cams:
        return cams[key], key
    return episode.get_frame, key


def _bin_edges(size: int, out: int) -> np.ndarray:
    return (np.arange(out + 1, dtype=np.int64) * size) // out


def _out_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    long = max(h, w)
    if long <= long_side:
        return h, w

    def scaled(x: int) -> int:  # round half up, integers only
        return max(1, (2 * x * long_side + long) // (2 * long))

    return scaled(h), scaled(w)


def gray_small(frame: Any, long_side: int = LONG_SIDE) -> np.ndarray:
    """Grayscale frame (integers 0 to 255, int64) with its long side downscaled to ``long_side``."""
    rgb = _as_rgb_uint8(frame)
    r, g, b = (rgb[..., i].astype(np.int64) for i in range(3))
    gray = r * _LUMA[0] + g * _LUMA[1] + b * _LUMA[2]  # 0 .. 255000
    h, w = gray.shape
    oh, ow = _out_size(h, w, long_side)
    if (oh, ow) == (h, w):
        return (gray + 500) // 1000
    ey, ex = _bin_edges(h, oh), _bin_edges(w, ow)
    sums = np.add.reduceat(np.add.reduceat(gray, ey[:-1], axis=0), ex[:-1], axis=1)
    count = np.outer(np.diff(ey), np.diff(ex)) * 1000
    return (2 * sums + count) // (2 * count)


def diff_sums(get_frame: Callable[[int], Any], num_frames: int,
              long_side: int = LONG_SIDE) -> tuple[list[int], int]:
    """(sum of absolute differences to the previous frame per frame, pixels per frame).

    Frames are read once each, in order (adapters that decode a whole camera on first access, as the
    LeRobot v3 source does, then decode sequentially). Frame 0 repeats frame 1's value.
    """
    n = int(num_frames)
    if n <= 0:
        return [], 0
    prev = gray_small(get_frame(0), long_side)
    npix = int(prev.size)
    sums = [0] * n
    for i in range(1, n):
        cur = gray_small(get_frame(i), long_side)
        if cur.shape != prev.shape:
            raise ValueError(f"frame {i} has size {cur.shape} after downscaling, frame {i - 1} had {prev.shape}")
        sums[i] = int(np.abs(cur - prev).sum())
        prev = cur
    if n > 1:
        sums[0] = sums[1]
    return sums, npix


def smooth(sums: list[int], npix: int) -> list[float]:
    """3-frame centered mean of the per-frame mean absolute difference (window cut at the ends)."""
    n = len(sums)
    out = []
    for i in range(n):
        lo, hi = max(0, i - 1), min(n, i + 2)
        out.append(sum(sums[lo:hi]) / float((hi - lo) * max(npix, 1)))
    return out


def percentile(values: list[float], q: int = PERCENTILE) -> float:
    """Linear-interpolation percentile (numpy's default), with the position computed exactly."""
    if not values:
        raise ValueError("percentile of an empty list")
    v = sorted(values)
    num = (len(v) - 1) * int(q)
    lo, rem = divmod(num, 100)
    if rem == 0 or lo + 1 >= len(v):
        return float(v[lo])
    return float(v[lo] + (v[lo + 1] - v[lo]) * (rem / 100.0))


def pause_runs(smoothed: list[float], threshold: float, min_len: int) -> list[tuple[int, int]]:
    """(first, last) frames of runs of at least ``min_len`` frames at or below ``threshold``."""
    if not smoothed or threshold >= max(smoothed):
        return []
    runs = []
    i, n = 0, len(smoothed)
    while i < n:
        if smoothed[i] > threshold:
            i += 1
            continue
        j = i
        while j < n and smoothed[j] <= threshold:
            j += 1
        if j - i >= min_len:
            runs.append((i, j - 1))
        i = j
    return runs


def min_pause_frames(fps: float, min_pause_s: float = MIN_PAUSE_S) -> int:
    return max(1, int(round(float(min_pause_s) * float(fps))))


def pause_events(smoothed: list[float], fps: float, *, q: int = PERCENTILE,
                 min_pause_s: float = MIN_PAUSE_S) -> list[dict[str, Any]]:
    """``pause_start`` and ``pause_end`` events from a smoothed motion signal."""
    n = len(smoothed)
    if n < 2:
        return []
    thr = percentile(smoothed, q)
    events = []
    for first, last in pause_runs(smoothed, thr, min_pause_frames(fps, min_pause_s)):
        mean = sum(smoothed[first:last + 1]) / float(last - first + 1)
        conf = 1.0 if thr <= 0 else 1.0 - mean / thr
        if first > 0:
            events.append(make_event("pause_start", first, conf, "motion"))
        if last + 1 < n:
            events.append(make_event("pause_end", last + 1, conf, "motion"))
    return sort_events(events)


def motion_signal(episode: Any, camera: str | None = None, *, long_side: int = LONG_SIDE,
                  q: int = PERCENTILE) -> dict[str, Any]:
    """The motion signal of one camera: per-frame sums, smoothed means and the pause threshold."""
    get, cam = frame_getter(episode, camera)
    sums, npix = diff_sums(get, int(episode.num_frames), long_side)
    sm = smooth(sums, npix)
    return {"version": MOTION_VERSION, "camera": cam, "num_frames": len(sums), "pixels": npix,
            "long_side": int(long_side), "diff_sums": sums, "smoothed": sm,
            "threshold": percentile(sm, q) if sm else None, "percentile": int(q)}


class MotionSource(EventSource):
    """Pauses in the pixel motion of one camera. ``l1`` is accepted for the common signature and ignored."""

    name = "motion"
    version = MOTION_VERSION

    def __init__(self, *, long_side: int = LONG_SIDE, percentile: int = PERCENTILE,
                 min_pause_s: float = MIN_PAUSE_S):
        self.long_side = int(long_side)
        self.q = int(percentile)
        self.min_pause_s = float(min_pause_s)

    def signal(self, episode: Any, camera: str | None = None) -> dict[str, Any]:
        """The motion signal of one camera (see :func:`motion_signal`), computed afresh on every call."""
        return motion_signal(episode, camera, long_side=self.long_side, q=self.q)

    def events_from_signal(self, signal: dict[str, Any], fps: float) -> list[dict[str, Any]]:
        """Events from a signal already computed with :meth:`signal` (decodes nothing)."""
        return pause_events(signal["smoothed"], float(fps), q=self.q, min_pause_s=self.min_pause_s)

    def events(self, episode: Any, *, camera: str | None = None,
               l1: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.events_from_signal(self.signal(episode, camera), float(episode.fps))
