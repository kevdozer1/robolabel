"""The ``gripper`` event source (SPEC_V1_1 3.1): the L1 signal layer's gripper events as typed events.

Every L1 closing run gives a ``close_start`` at its onset and every opening run an ``open_start`` at its
onset (the command's onsets, as L1 measures them), with ``attempt_idx`` from the L1 attempt the event
belongs to: the attempt whose close or re-close starts there, or whose span contains the frame, for a
``close_start``; the attempt whose first opening starts there for an ``open_start``; otherwise None (a
rest close after the last release, an opening before any close). L1's low-confidence candidates ("the
arm starts moving after the close / the opening", and any other low-confidence candidate L1 adds later)
become ``arm_move`` events at the candidate's onset, which is the frame after the candidate frame (an L1
candidate frame ends the earlier segment).

With ``include_recovery`` (the default), L1's recovery candidates after a failed close (SPEC_V1_1 6, the
record's ``recovery_candidates``) become events with the source label ``gripper_recovery``: ``open_start``
at the re-open, which takes the place of the plain ``open_start`` L1 already gives at that onset (one event
per type and frame, and the label tells the model that the fingers reopen after a failed close), and
``back_off`` where the arm then starts moving away (low confidence). Both carry the failed attempt's
index and sit at the candidate's onset, the frame after the candidate frame. A record without the field
(an L1 record from before v1.1) gives the events it gave before.

Events keep only onsets in ``[1, num_frames - 1]``: an onset at frame 0 cannot start a new segment.

The record comes from the caller (``l1=``, the output of ``layers.signal.run_l1``), or is computed from
``episode.extra["state"]`` and ``episode.extra["action"]`` when the source was built with a
:class:`~robolabel.layers.signal.Calibration`. The source adds nothing to what L1 measured: this is the
robot signal, so an evaluation that scores against gripper truth must not use this source.
"""

from __future__ import annotations

from typing import Any

from ..layers.signal import CODE_VERSION, RECOVERY_VERSION, Calibration, run_l1
from .base import EventSource, make_event, sort_events

ONSET_CONFIDENCE = 0.9  # the same confidence the v7 rows give a signal boundary
ARM_MOVE_CONFIDENCE = 0.3
BACK_OFF_CONFIDENCE = 0.3  # a low-confidence heuristic, like arm_move
RECOVERY_SOURCE = "gripper_recovery"


def _attempt_for_close(attempts: list[dict[str, Any]], onset: int) -> int | None:
    for a in attempts:
        if int(a["closing_onset"]) == onset or any(int(r["onset"]) == onset for r in a.get("recloses") or []):
            return int(a["attempt_idx"])
    for a in attempts:
        end = a.get("end_frame")
        if int(a["closing_onset"]) <= onset and (end is None or onset <= int(end)):
            return int(a["attempt_idx"])
    return None


def _attempt_for_open(attempts: list[dict[str, Any]], onset: int) -> int | None:
    for a in attempts:
        if a.get("opening_onset") is not None and int(a["opening_onset"]) == onset:
            return int(a["attempt_idx"])
    return None


def recovery_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``gripper_recovery`` events of an L1 record's ``recovery_candidates`` (empty when absent)."""
    out = []
    for c in record.get("recovery_candidates") or []:
        kind = str(c.get("type"))
        conf = ONSET_CONFIDENCE if kind == "open_start" else BACK_OFF_CONFIDENCE
        idx = c.get("attempt_idx")
        out.append(make_event(kind, int(c["frame"]) + 1, conf, RECOVERY_SOURCE, None if idx is None else int(idx)))
    return out


def events_from_l1(record: dict[str, Any], *, include_arm_move: bool = True,
                   include_recovery: bool = True) -> list[dict[str, Any]]:
    """Typed events from an L1 record (``layers.signal.run_l1``), sorted by frame."""
    n = int(record.get("num_frames") or 0)
    attempts = list(record.get("attempts") or [])
    out: list[dict[str, Any]] = []
    for run in record.get("events") or []:
        onset = int(run["onset"])
        if run.get("type") == "closing":
            out.append(make_event("close_start", onset, ONSET_CONFIDENCE, "gripper",
                                  _attempt_for_close(attempts, onset)))
        elif run.get("type") == "opening":
            out.append(make_event("open_start", onset, ONSET_CONFIDENCE, "gripper",
                                  _attempt_for_open(attempts, onset)))
    if include_arm_move:
        for c in record.get("candidates") or []:
            if c.get("confidence") == "high":  # the gripper onsets themselves, already events above
                continue
            idx = c.get("attempt_idx")
            out.append(make_event("arm_move", int(c["frame"]) + 1, ARM_MOVE_CONFIDENCE, "gripper",
                                  None if idx is None else int(idx)))
    keep: dict[tuple[str, int], dict[str, Any]] = {}
    for ev in out:
        if ev["frame"] < 1 or (n and ev["frame"] > n - 1):
            continue
        keep.setdefault((ev["type"], ev["frame"]), ev)  # one event per type and frame
    if include_recovery:
        recovered: set[tuple[str, int]] = set()
        for ev in recovery_events(record):
            key = (ev["type"], ev["frame"])
            if ev["frame"] < 1 or (n and ev["frame"] > n - 1) or key in recovered:
                continue
            keep[key] = ev  # the recovery label replaces the plain gripper event at the same onset
            recovered.add(key)
    return sort_events(list(keep.values()))


class GripperSource(EventSource):
    """L1's gripper events. Pass ``l1=`` to :meth:`events`, or build the source with a calibration."""

    name = "gripper"
    version = f"{CODE_VERSION}+{RECOVERY_VERSION}"

    def __init__(self, calibration: Calibration | None = None, *, include_arm_move: bool = True,
                 include_recovery: bool = True):
        self.calibration = calibration
        self.include_arm_move = bool(include_arm_move)
        self.include_recovery = bool(include_recovery)

    def l1_record(self, episode: Any, l1: dict[str, Any] | None = None) -> dict[str, Any]:
        if l1 is not None:
            return l1
        if self.calibration is None:
            raise ValueError("the gripper source needs an L1 record (l1=...) or a Calibration")
        extra = getattr(episode, "extra", None) or {}
        state, action = extra.get("state"), extra.get("action")
        if state is None or action is None:
            raise ValueError(f"episode {getattr(episode, 'episode_id', '?')} has no state and action arrays "
                             "for the gripper source")
        return run_l1(state, action, self.calibration, episode_key=str(getattr(episode, "episode_id", "")),
                      family=str(extra.get("family") or ""))

    def events(self, episode: Any, *, camera: str | None = None,
               l1: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return events_from_l1(self.l1_record(episode, l1), include_arm_move=self.include_arm_move,
                              include_recovery=self.include_recovery)
