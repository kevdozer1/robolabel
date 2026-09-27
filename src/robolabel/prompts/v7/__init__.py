"""Schema v7 layer prompts (V_LITE): scene inventory, scene facts, segments, goal.

Prompts are plain text files next to this module (str.format placeholders). Schemas follow the V_LITE
rules for every provider: no nulls, every property required, ``additionalProperties: false``, no
``$ref``, ``oneOf`` or ``anyOf``, object nesting at most 4, enums as strings, frames as integers, and no
``minItems``, ``maxItems``, ``minimum``, ``maximum``, ``pattern`` or ``format`` (limits are stated in
the prompt and enforced after parsing).
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VERSION = "v7-2026-09-27.1"

PHASE_CLASSES = ["approach", "grasp", "transport", "release", "retract", "press", "pour", "insert", "fold", "wipe",
                 "push", "pull", "rotate", "open", "close", "other"]
PREDICATES = ["inside", "on_top_of", "touching", "in_gripper", "lifted", "at_location", "unchanged", "activated",
              "state", "holding", "gripper_open", "gripper_closed", "withdrawn", "at_home_pose", "near_object",
              "tool_lifted", "other"]
CATEGORIES = ["block", "container", "bowl", "plate", "bottle", "tool", "control", "cloth", "other"]
FAILURE_TYPES = ["none", "missed_grasp", "slip", "drop", "wrong_object", "misplace", "press_no_effect", "aborted",
                 "other"]


@cache
def load_prompt(name: str) -> str:
    return (HERE / f"{name}.txt").read_text(encoding="utf-8")


def prompt_sha256(name: str) -> str:
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


S = {"type": "string"}
I = {"type": "integer"}  # noqa: E741
N = {"type": "number"}
B = {"type": "boolean"}


def _enum(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


SCHEMAS: dict[str, dict[str, Any]] = {
    "scene_inventory": _obj({"objects": _arr(_obj({
        "object_id": S, "name": S, "aliases": _arr(S), "category": _enum(CATEGORIES),
        "views": _arr(_obj({"camera": S, "visible": B, "x": N, "y": N, "x0": N, "y0": N, "x1": N, "y1": N})),
    }))}),
    "scene_facts": _obj({"facts": _arr(_obj({
        "frame": I, "camera": S, "visible": _arr(S), "partial": _arr(S), "in_gripper": S,
        "relations": _arr(_obj({"subject": S, "relation": _enum(["inside", "on_top_of", "touching"]), "object": S,
                                "value": _enum(["true", "false", "unsure"])})),
        "boxes": _arr(_obj({"object_id": S, "x0": N, "y0": N, "x1": N, "y1": N})),
    }))}),
    "segments": _obj({
        "segments": _arr(_obj({
            "start_frame": I, "end_frame": I, "phase_class": _enum(PHASE_CLASSES), "phase_text": S, "target": S,
            "destination": S, "attempt_idx": I, "outcome": _enum(["success", "failed", "aborted"]),
            "failure_type": _enum(FAILURE_TYPES), "boundary_source": _enum(["signal", "vlm"]), "candidate_id": S,
            "evidence": _arr(_obj({"frame": I, "camera": S, "statement": S})),
        })),
        "candidates": _arr(_obj({"candidate_id": S, "verdict": _enum(["confirm", "reject"]), "note": S})),
    }),
    "goal": _obj({
        "objective_text": S, "primary_target": S, "primary_destination": S,
        "requirements": _arr(_obj({
            "req_id": S, "kind": _enum(["object_end_state", "robot_end_state"]), "object": S,
            "predicate": _enum(PREDICATES), "ref_object": S, "value": S,
            "status": _enum(["required", "incidental", "unsure"]), "unsure_kind": _enum(["none", "perception", "intent"]),
            "basis": _enum(["task_string", "physical_necessity", "observed"]),
            "achieved": _enum(["true", "false", "unknown"]), "deciding_frame": I, "deciding_camera": S,
            "visibility": _arr(_obj({"camera": S, "class": _enum(["visible", "partial", "not_visible"])})),
            "reason": S,
        })),
    }),
}

MAX_TOKENS = {"scene_inventory": 3000, "scene_facts": 6000, "segments": 8000, "goal": 5000}
