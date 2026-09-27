"""The event contract of v1.1 (SPEC_V1_1 3.1): typed candidate events from pluggable sources.

An event is a plain dict so it serializes as JSON without help::

    {"type": "pause_start", "frame": 152, "confidence": 0.8123, "source": "motion", "attempt_idx": None}

``frame`` is the event's onset frame: the first frame of whatever the event starts, which is also the
start frame of the segment after a boundary placed there. Types by source:

* ``gripper``: ``close_start`` and ``open_start`` (the onsets of L1's closing and opening runs) and
  ``arm_move`` (L1's low-confidence "the arm starts moving" candidates);
* ``motion``: ``pause_start`` and ``pause_end``;
* ``none``: no events.

Events are candidates for the model, never truth: no source here reads anything that a scorer uses as
ground truth unless the caller hands it over on purpose (the gripper source reads the robot signal).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class Event(TypedDict):
    type: str
    frame: int
    confidence: float
    source: str
    attempt_idx: int | None


EVENT_TYPES = ("close_start", "open_start", "arm_move", "pause_start", "pause_end")
SOURCE_NAMES = ("none", "motion", "gripper")
# Order of types at the same frame, so sorting is total and stable across runs.
_TYPE_ORDER = {t: i for i, t in enumerate(EVENT_TYPES)}
CONFIDENCE_DIGITS = 4


def make_event(type: str, frame: int, confidence: float, source: str,
               attempt_idx: int | None = None) -> dict[str, Any]:
    """One event with normalized value types (int frame, confidence rounded to 4 digits in [0, 1])."""
    if type not in _TYPE_ORDER:
        raise ValueError(f"unknown event type {type!r}; expected one of {', '.join(EVENT_TYPES)}")
    conf = min(1.0, max(0.0, float(confidence)))
    return {"type": str(type), "frame": int(frame), "confidence": round(conf, CONFIDENCE_DIGITS),
            "source": str(source), "attempt_idx": None if attempt_idx is None else int(attempt_idx)}


def sort_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events in time order; ties broken by type, then source, then attempt, so the order is total."""
    return sorted(events, key=lambda e: (int(e["frame"]), _TYPE_ORDER.get(e["type"], len(_TYPE_ORDER)),
                                         str(e["source"]), -1 if e.get("attempt_idx") is None
                                         else int(e["attempt_idx"])))


def validate_event(ev: Any) -> list[str]:
    """Problems with one event (empty when it follows the contract)."""
    if not isinstance(ev, dict):
        return [f"event is a {type(ev).__name__}, not a dict"]
    problems = []
    keys = {"type", "frame", "confidence", "source", "attempt_idx"}
    if set(ev) != keys:
        problems.append(f"keys {sorted(ev)} are not {sorted(keys)}")
    if ev.get("type") not in _TYPE_ORDER:
        problems.append(f"type {ev.get('type')!r} is not an event type")
    if not isinstance(ev.get("frame"), int) or isinstance(ev.get("frame"), bool):
        problems.append(f"frame {ev.get('frame')!r} is not an int")
    c = ev.get("confidence")
    if not isinstance(c, float) or not 0.0 <= c <= 1.0:
        problems.append(f"confidence {c!r} is not a float in [0, 1]")
    if ev.get("source") not in SOURCE_NAMES:
        problems.append(f"source {ev.get('source')!r} is not a source name")
    a = ev.get("attempt_idx")
    if a is not None and (not isinstance(a, int) or isinstance(a, bool)):
        problems.append(f"attempt_idx {a!r} is neither None nor an int")
    return problems


class EventSource(ABC):
    """A pluggable event source. ``events`` returns events sorted by frame (see :func:`sort_events`)."""

    name: str = "source"
    version: str = ""

    @abstractmethod
    def events(self, episode: Any, *, camera: str | None = None,
               l1: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ------------------------------------------------------------------------------------------------ prompt text
def candidate_ids(events: list[dict[str, Any]]) -> list[str]:
    """``c1, c2, ...``: the 1-based position of each event in the list it came in."""
    return [f"c{i}" for i in range(1, len(events) + 1)]


def candidate_map(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Candidate ID to event, numbered as :func:`candidate_lines` numbers them."""
    return dict(zip(candidate_ids(events), events, strict=True))


def candidate_lines(events: list[dict[str, Any]], num_frames: int, fps: float) -> list[str]:
    """Plain lines for prompts, one per event, for example ``c1: frame 152 (5.07 s), pause_start (motion)``.

    Events keep the ID of their position in the list (pass them sorted, as sources return them). An event
    whose frame lies outside ``[0, num_frames)`` gets no line, and the IDs of the others do not change.
    """
    rate = float(fps) if fps and float(fps) > 0 else 1.0
    lines = []
    for cid, ev in zip(candidate_ids(events), events, strict=True):
        f = int(ev["frame"])
        if not 0 <= f < int(num_frames):
            continue
        lines.append(f"{cid}: frame {f} ({f / rate:.2f} s), {ev['type']} ({ev['source']})")
    return lines
