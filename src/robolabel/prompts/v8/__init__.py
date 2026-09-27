"""Prompt version v8 (robolabel v1.1, video first): the coarse pass and the crawl.

Prompts are plain text files next to this module. ``coarse.txt`` holds named sections, each opened by a
line ``=== name ===``; :func:`prompt_sections` splits them, and ``coarse_request`` in
``layers/coarse.py`` picks the sections that fit the call (frames or video, inventory or plain words,
candidate list or none) and fills their ``str.format`` placeholders. The hash of a prompt covers the
whole file, every section included.

Schemas follow the v7 rules for every provider (see ``prompts/v7``): no nulls, every property
required, ``additionalProperties: false``, no ``$ref``, ``oneOf`` or ``anyOf``, enums as strings,
frames as integers, and no ``minItems``, ``maxItems``, ``minimum``, ``maximum``, ``pattern`` or
``format``. Limits are stated in the prompt and enforced after parsing.
"""

from __future__ import annotations

import hashlib
import re
from functools import cache
from pathlib import Path
from typing import Any

from ..v7 import FAILURE_TYPES, PHASE_CLASSES

HERE = Path(__file__).resolve().parent
VERSION = "v8-2026-09-27.1"

END_EVENTS = ["close_start", "open_start", "contact_start", "contact_end", "other"]
OUTCOMES = ["success", "failed", "aborted"]

__all__ = ["END_EVENTS", "FAILURE_TYPES", "MAX_TOKENS", "OUTCOMES", "PHASE_CLASSES", "SCHEMAS", "VERSION",
           "load_prompt", "prompt_sections", "prompt_sha256"]

_SECTION = re.compile(r"^=== ([a-z0-9_]+) ===$")


@cache
def load_prompt(name: str) -> str:
    return (HERE / f"{name}.txt").read_text(encoding="utf-8")


def prompt_sha256(name: str) -> str:
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()


@cache
def _sections(name: str) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, list[str]]] = []
    for line in load_prompt(name).splitlines():
        m = _SECTION.match(line.strip())
        if m:
            out.append((m.group(1), []))
        elif out:
            out[-1][1].append(line)
    return tuple((k, "\n".join(v).strip()) for k, v in out)


def prompt_sections(name: str) -> dict[str, str]:
    """The named sections of a prompt file (a fresh dict; text stripped of surrounding blank lines)."""
    return dict(_sections(name))


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _enum(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


S = {"type": "string"}
I = {"type": "integer"}  # noqa: E741
N = {"type": "number"}


def _segment_item(start: str, end: str, kind: dict[str, Any]) -> dict[str, Any]:
    return _obj({
        start: kind, end: kind, "phase_class": _enum(PHASE_CLASSES), "phase_text": S, "end_event": _enum(END_EVENTS),
        "target": S, "destination": S, "attempt_idx": I, "outcome": _enum(OUTCOMES),
        "attempt_outcome": _enum(OUTCOMES), "failure_type": _enum(FAILURE_TYPES), "candidate_id": S,
    })


SCHEMAS: dict[str, dict[str, Any]] = {
    "coarse_frames": _obj({"segments": _arr(_segment_item("start_frame", "end_frame", I))}),
    "coarse_video": _obj({"segments": _arr(_segment_item("start_s", "end_s", N))}),
    "crawl": _obj({"answer": I}),
}

MAX_TOKENS = {"coarse": 8000, "crawl": 2500}
