"""Event sources of v1.1 (SPEC_V1_1 3.1): pluggable, typed candidate events for the coarse pass.

``get_source(name, **kw)`` returns a source with ``.name`` and
``.events(episode, *, camera=None, l1=None) -> list[Event]`` (sorted by frame):

* ``none``: no candidates (video alone);
* ``motion``: pauses in the pixel motion of one camera, deterministic and free;
* ``gripper``: the L1 gripper events (``close_start``, ``open_start``, ``arm_move``), plus the recovery
  events after a failed close (``open_start`` and ``back_off``, source label ``gripper_recovery``).

``candidate_lines(events, num_frames, fps)`` renders them as plain prompt lines with IDs ``c1, c2, ...``
and ``candidate_map(events)`` resolves those IDs back to events.
"""

from __future__ import annotations

import json
from typing import Any

from .base import (
    EVENT_SOURCE_LABELS,
    EVENT_TYPES,
    RECOVERY_EVENT_TYPES,
    SOURCE_NAMES,
    Event,
    EventSource,
    candidate_ids,
    candidate_lines,
    candidate_map,
    make_event,
    sort_events,
    validate_event,
)
from .gripper import GripperSource, events_from_l1, recovery_events
from .motion import MOTION_VERSION, MotionSource, motion_signal, pause_events
from .none import NoneSource

_SOURCES: dict[str, type[EventSource]] = {"none": NoneSource, "motion": MotionSource, "gripper": GripperSource}


def get_source(name: str, **kw: Any) -> EventSource:
    """The event source called ``name`` (``none``, ``motion`` or ``gripper``), built with ``kw``."""
    try:
        cls = _SOURCES[str(name)]
    except KeyError:
        raise ValueError(f"unknown event source {name!r}; expected one of {', '.join(SOURCE_NAMES)}") from None
    return cls(**kw)


def dumps_events(events: list[dict[str, Any]]) -> str:
    """Canonical JSON for a list of events (sorted keys, no spaces): byte-identical on identical input."""
    return json.dumps(events, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "EVENT_SOURCE_LABELS", "EVENT_TYPES", "RECOVERY_EVENT_TYPES", "SOURCE_NAMES", "MOTION_VERSION", "Event",
    "EventSource", "GripperSource", "MotionSource",
    "NoneSource", "candidate_ids", "candidate_lines", "candidate_map", "dumps_events", "events_from_l1",
    "get_source", "make_event", "motion_signal", "pause_events", "recovery_events", "sort_events",
    "validate_event",
]
