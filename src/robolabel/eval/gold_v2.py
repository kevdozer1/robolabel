"""Gold v2 files (MEASUREMENT_SPEC 3.2): schema, validation, completeness warnings, merge and dump.

One JSON file per family holds the human gold for that family's episodes. :func:`validate_gold`
runs the bundled JSON Schema (``schemas/gold_v2.schema.json``) and then the rules a schema cannot
express: segments sorted, contiguous and covering the episode, the last segment without a boundary
quality, ``unsure_kind`` set exactly when the status is ``unsure``, unique ids, object references,
frames and cameras inside the episode, and ``episode_key`` equal to ``<family>/<episode_index>``.

The pilot annotation page (gold.html) saves a wrapper around several family files
(``robolabel/gold-export/v1``), checked by :func:`validate_gold_export`. :func:`warnings_for_episode`
lists the non-blocking completeness warnings the page also shows. :func:`dump_gold` writes the
canonical form: keys sorted, LF line endings, floats rounded to 6 decimals.

jsonschema comes from the ``eval`` extra and is imported only when a document is validated.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from .lexicon import load_predicate_lexicon, predicate_value_type

SCHEMA_VERSION = "robolabel/gold/v2"
EXPORT_SCHEMA = "robolabel/gold-export/v1"
SCHEMA_FILE = "gold_v2.schema.json"

OBJECT_ID_RE = re.compile(r"^o[0-9]+$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FLOAT_DECIMALS = 6


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@cache
def _schema_text() -> str:
    return resources.files("robolabel").joinpath("schemas").joinpath(SCHEMA_FILE).read_text(encoding="utf-8")


def load_schema() -> dict[str, Any]:
    """Return the gold v2 JSON Schema (draft 2020-12) bundled with the package."""
    return json.loads(_schema_text())


def _is_json_int(_checker: Any, instance: Any) -> bool:
    """True for a JSON integer. JSON Schema also accepts 12.0; frames and indices here must be ints."""
    return isinstance(instance, int) and not isinstance(instance, bool)


@cache
def _validator() -> Any:
    try:
        from jsonschema import Draft202012Validator, validators
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("gold v2 validation needs jsonschema: pip install 'robolabel[eval]'") from exc
    schema = load_schema()
    Draft202012Validator.check_schema(schema)
    checker = Draft202012Validator.TYPE_CHECKER.redefine("integer", _is_json_int)
    strict = validators.extend(Draft202012Validator, type_checker=checker)
    return strict(schema)


def _format_path(path: Iterable[Any]) -> str:
    """``$.episodes[0].segments[1].phase_class``; keys that are not identifiers go in brackets."""
    out = "$"
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        elif _IDENT_RE.match(str(part)):
            out += f".{part}"
        else:
            out += f"[{json.dumps(str(part), ensure_ascii=False)}]"
    return out


def _path_key(path: tuple[Any, ...]) -> tuple[tuple[int, int, str], ...]:
    """Sort key that orders list indices numerically and keys alphabetically."""
    return tuple((0, p, "") if isinstance(p, int) else (1, 0, str(p)) for p in path)


def _schema_errors(doc: Any) -> list[tuple[tuple[Any, ...], str]]:
    """(path, message) for every schema violation, sorted by path then message, without repeats."""
    found = {(tuple(err.absolute_path), err.message) for err in _validator().iter_errors(doc)}
    return sorted(found, key=lambda pm: (_path_key(pm[0]), pm[1]))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_gold(doc: Any) -> list[str]:
    """Errors of one gold v2 family document; an empty list means valid.

    Schema errors come first (``schema: <path>: <message>``, sorted by path), then the semantic
    errors in document order (``episodes[i] <episode_key> pass <n>: <message>``). An episode with
    a schema error gets no semantic checks, so every message describes a real problem.
    """
    if not isinstance(doc, Mapping):
        return ["$: a gold v2 document must be a JSON object"]
    schema_errors = _schema_errors(doc)
    errors = [f"schema: {_format_path(path)}: {message}" for path, message in schema_errors]
    broken = {
        path[1] for path, _ in schema_errors
        if len(path) >= 2 and path[0] == "episodes" and isinstance(path[1], int)
    }
    episodes = doc.get("episodes")
    if not isinstance(episodes, list):
        return errors
    family = doc.get("family") if isinstance(doc.get("family"), str) else None
    seen: dict[tuple[str, int], int] = {}
    for i, ep in enumerate(episodes):
        if i in broken or not isinstance(ep, Mapping):
            continue
        where = f"episodes[{i}] {ep.get('episode_key')} pass {ep.get('pass')}"
        key = (ep["episode_key"], ep["pass"])
        if key in seen:
            errors.append(f"{where}: duplicate of episodes[{seen[key]}] (same episode_key and pass)")
        seen.setdefault(key, i)
        errors.extend(f"{where}: {msg}" for msg in _episode_errors(ep, family))
    return errors


def _episode_errors(ep: Mapping[str, Any], family: str | None) -> list[str]:
    """Semantic errors of one schema-valid episode, in a fixed order."""
    errors: list[str] = []
    if family is not None and ep["episode_key"] != f"{family}/{ep['episode_index']}":
        errors.append(f"episode_key {ep['episode_key']!r} is not '{family}/{ep['episode_index']}'")
    num_frames = int(ep["num_frames"])
    cameras = set(ep["cameras"])
    object_ids = _object_errors(ep, cameras, errors)
    _check_object_ref(errors, "primary_target", ep["primary_target"], object_ids, strict=True)
    _check_object_ref(errors, "primary_destination", ep["primary_destination"], object_ids, strict=True)
    errors.extend(_segment_errors(ep["segments"], num_frames, object_ids, ep.get("coarse_subtasks")))
    if "coarse_subtasks" in ep:
        errors.extend(_coarse_errors(ep["coarse_subtasks"], num_frames, object_ids))
    errors.extend(_failed_attempt_errors(ep["failed_attempts"], num_frames, cameras, object_ids))
    errors.extend(_requirement_errors(ep["goal"]["requirements"], num_frames, cameras, object_ids))
    return errors


def _object_errors(ep: Mapping[str, Any], cameras: set[str], errors: list[str]) -> set[str]:
    """Check object ids and first-frame points; return the set of object ids."""
    ids: set[str] = set()
    for obj in ep["objects"]:
        oid = obj["object_id"]
        if oid in ids:
            errors.append(f"objects: duplicate object_id {oid!r}")
        ids.add(oid)
        point = obj.get("first_frame_point")
        if point is not None and point["camera"] not in cameras:
            errors.append(f"object {oid}: first_frame_point camera {point['camera']!r} "
                          "is not an episode camera")
    return ids


def _check_object_ref(errors: list[str], label: str, ref: Any, object_ids: set[str], *, strict: bool) -> None:
    """A reference must name an object of the episode.

    ``strict`` roles take only object ids or null: targets and destinations (spec 3.2 field rules),
    the primary target and destination, requirement and failed-attempt objects. ``ref_object`` may
    also hold a region name (``at_location``, Appendix B), so there only a string that looks like an
    object id (``o`` and digits) must refer to an existing object.
    """
    if ref is None:
        return
    if OBJECT_ID_RE.match(ref):
        if ref not in object_ids:
            errors.append(f"{label} {ref!r} is not an object_id of this episode")
    elif strict:
        errors.append(f"{label} {ref!r} must be an object_id of this episode or null")


def _frame_errors(label: str, frame: Any, num_frames: int) -> list[str]:
    if frame is None or 0 <= frame <= num_frames - 1:
        return []
    return [f"{label} {frame} is outside [0, {num_frames - 1}]"]


def _segment_errors(
    segments: list[Mapping[str, Any]], num_frames: int, object_ids: set[str], coarse: Any,
) -> list[str]:
    errors: list[str] = []
    last = num_frames - 1
    for k, seg in enumerate(segments):
        start, end = seg["start_frame"], seg["end_frame"]
        if start > end:
            errors.append(f"segment {k}: start_frame {start} is after end_frame {end}")
        if k == 0 and start != 0:
            errors.append(f"segments: the first segment starts at {start}, expected 0")
        if k > 0:
            prev = segments[k - 1]
            if start < prev["start_frame"]:
                errors.append(f"segments not sorted: segment {k} starts at {start}, before segment {k - 1} "
                              f"(starts at {prev['start_frame']})")
            elif start > prev["end_frame"] + 1:
                errors.append(f"segments: gap between segment {k - 1} (ends at {prev['end_frame']}) and "
                              f"segment {k} (starts at {start}); frames {prev['end_frame'] + 1} to "
                              f"{start - 1} are not covered")
            elif start <= prev["end_frame"]:
                errors.append(f"segments: segment {k} (starts at {start}) overlaps segment {k - 1} "
                              f"(ends at {prev['end_frame']})")
        _check_object_ref(errors, f"segment {k}: target", seg["target"], object_ids, strict=True)
        _check_object_ref(errors, f"segment {k}: destination", seg["destination"], object_ids, strict=True)
    final = segments[-1]
    if final["end_frame"] != last:
        errors.append(f"segments: the last segment ends at {final['end_frame']}, "
                      f"expected num_frames - 1 = {last}")
    if final["end_boundary_quality"] is not None:
        errors.append(f"segment {len(segments) - 1}: the last segment must have end_boundary_quality null "
                      f"(got {final['end_boundary_quality']!r})")
    if isinstance(coarse, list):
        coarse_ids = {c["coarse_idx"] for c in coarse}
        for k, seg in enumerate(segments):
            cidx = seg.get("coarse_idx")
            if cidx is not None and cidx not in coarse_ids:
                errors.append(f"segment {k}: coarse_idx {cidx} is not a coarse subtask of this episode")
    return errors


def _coarse_errors(coarse: list[Mapping[str, Any]], num_frames: int, object_ids: set[str]) -> list[str]:
    errors: list[str] = []
    seen: set[int] = set()
    for k, item in enumerate(coarse):
        start, end = item["start_frame"], item["end_frame"]
        if item["coarse_idx"] in seen:
            errors.append(f"coarse subtask {k}: duplicate coarse_idx {item['coarse_idx']}")
        seen.add(item["coarse_idx"])
        if start > end:
            errors.append(f"coarse subtask {k}: start_frame {start} is after end_frame {end}")
        if end > num_frames - 1:
            errors.append(f"coarse subtask {k}: end_frame {end} is outside [0, {num_frames - 1}]")
        if k > 0 and start != coarse[k - 1]["end_frame"] + 1:
            errors.append(f"coarse subtasks: coarse subtask {k} starts at {start}, expected "
                          f"{coarse[k - 1]['end_frame'] + 1} (right after coarse subtask {k - 1})")
        _check_object_ref(errors, f"coarse subtask {k}: target", item["target"], object_ids, strict=True)
        _check_object_ref(errors, f"coarse subtask {k}: destination", item["destination"], object_ids,
                          strict=True)
    return errors


def _failed_attempt_errors(
    attempts: list[Mapping[str, Any]], num_frames: int, cameras: set[str], object_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    for k, fa in enumerate(attempts):
        start, end = fa["span"]
        label = f"failed attempt {k}"
        if start > end:
            errors.append(f"{label}: span [{start}, {end}] ends before it starts")
        if end > num_frames - 1:
            errors.append(f"{label}: span [{start}, {end}] is outside [0, {num_frames - 1}]")
        errors.extend(_frame_errors(f"{label}: evident_frame", fa["evident_frame"], num_frames))
        errors.extend(_frame_errors(f"{label}: recovery_start", fa["recovery_start"], num_frames))
        if fa["evident_camera"] is not None and fa["evident_camera"] not in cameras:
            errors.append(f"{label}: evident_camera {fa['evident_camera']!r} is not an episode camera")
        _check_object_ref(errors, f"{label}: object", fa["object"], object_ids, strict=True)
    return errors


def _requirement_errors(
    requirements: list[Mapping[str, Any]], num_frames: int, cameras: set[str], object_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for req in requirements:
        rid = req["req_id"]
        label = f"requirement {rid}"
        if rid in seen:
            errors.append(f"requirements: duplicate req_id {rid!r}")
        seen.add(rid)
        if req["status"] == "unsure" and req["unsure_kind"] is None:
            errors.append(f"{label}: status unsure needs unsure_kind perception or intent")
        if req["status"] != "unsure" and req["unsure_kind"] is not None:
            errors.append(f"{label}: unsure_kind {req['unsure_kind']!r} is set but status is "
                          f"{req['status']!r}")
        if req["kind"] == "object_end_state" and req["object"] is None:
            errors.append(f"{label}: an object_end_state requirement needs an object")
        _check_object_ref(errors, f"{label}: object", req["object"], object_ids, strict=True)
        _check_object_ref(errors, f"{label}: ref_object", req["ref_object"], object_ids, strict=False)
        errors.extend(_value_type_errors(label, req["predicate"], req["value"]))
        errors.extend(_frame_errors(f"{label}: deciding_frame", req["deciding_frame"], num_frames))
        if req["deciding_camera"] is not None and req["deciding_camera"] not in cameras:
            errors.append(f"{label}: deciding_camera {req['deciding_camera']!r} is not an episode camera")
        for cam in sorted(set(req["visibility"]) - cameras):
            errors.append(f"{label}: visibility camera {cam!r} is not an episode camera "
                          f"({', '.join(sorted(cameras))})")
    return errors


def _value_type_errors(label: str, predicate: str, value: Any) -> list[str]:
    """``state`` takes a string value, the other lexicon predicates a boolean; null is allowed."""
    expected = predicate_value_type(predicate)
    if value is None or expected is None:
        return []
    if expected == "string" and not isinstance(value, str):
        return [f"{label}: predicate {predicate} takes a string value, got {value!r}"]
    if expected == "bool" and not isinstance(value, bool):
        return [f"{label}: predicate {predicate} takes a boolean value, got {value!r}"]
    return []


def validate_gold_export(doc: Any) -> list[str]:
    """Errors of a gold.html export: ``{"schema", "run_id", "saved_at", "families": {family: doc}}``.

    Each family document is checked with :func:`validate_gold`; its messages are prefixed with
    ``families.<family>:``. Extra wrapper keys are ignored.
    """
    if not isinstance(doc, Mapping):
        return ["$: a gold export must be a JSON object"]
    errors: list[str] = []
    if doc.get("schema") != EXPORT_SCHEMA:
        errors.append(f"$.schema: expected {EXPORT_SCHEMA!r}, got {doc.get('schema')!r}")
    for field in ("run_id", "saved_at"):
        if not isinstance(doc.get(field), str) or not doc.get(field):
            errors.append(f"$.{field}: a non-empty string is required")
    families = doc.get("families")
    if not isinstance(families, Mapping):
        errors.append("$.families: an object mapping family to gold v2 document is required")
        return errors
    for family in sorted(families):
        family_doc = families[family]
        if isinstance(family_doc, Mapping) and family_doc.get("family") != family:
            errors.append(f"families.{family}: the document's family is {family_doc.get('family')!r}")
        errors.extend(f"families.{family}: {msg}" for msg in validate_gold(family_doc))
    return errors


# --------------------------------------------------------------------------- #
# Completeness warnings (never block a save)
# --------------------------------------------------------------------------- #
def warnings_for_episode(ep: Mapping[str, Any]) -> list[str]:
    """Non-blocking completeness warnings for one episode, in a fixed order.

    Missing phase classes, boundary qualities and failure types; a missing ``primary_target``;
    missing robot ending slots (``holding``, ``gripper_open`` or ``gripper_closed``, a position
    item); requirements with neither a value nor an unsure status; visibility not given for every
    camera on required and unsure items; objects without a first-frame point; failed attempts
    without the ``failed_attempt`` tag; coarse subtasks that do not cover the episode.
    """
    if not isinstance(ep, Mapping):
        return ["the episode is not a JSON object"]
    warns: list[str] = []
    segments = [s for s in _as_list(ep.get("segments")) if isinstance(s, Mapping)]
    for k, seg in enumerate(segments):
        if not seg.get("phase_class"):
            warns.append(f"segment {k}: missing phase_class")
        if k < len(segments) - 1 and seg.get("end_boundary_quality") is None:
            warns.append(f"segment {k}: missing end_boundary_quality")
        if seg.get("outcome") == "failed" and not seg.get("failure_type"):
            warns.append(f"segment {k}: outcome failed without failure_type")
        if "segment_idx" in seg and seg.get("segment_idx") != k:
            warns.append(f"segment {k}: segment_idx is {seg.get('segment_idx')!r}, expected {k}")
    if not ep.get("primary_target"):
        warns.append("missing primary_target")

    goal = ep.get("goal") if isinstance(ep.get("goal"), Mapping) else {}
    requirements = [r for r in _as_list(goal.get("requirements")) if isinstance(r, Mapping)]
    robot_predicates = {r.get("predicate") for r in requirements if r.get("kind") == "robot_end_state"}
    slots = load_predicate_lexicon()["robot_ending_slots"]
    for slot_name, predicates in slots.items():
        if not robot_predicates & set(predicates):
            warns.append(f"missing robot ending slot {slot_name}: {' or '.join(predicates)}")
    cameras = [c for c in _as_list(ep.get("cameras")) if isinstance(c, str)]
    for req in requirements:
        rid = req.get("req_id")
        if req.get("value") is None and req.get("status") != "unsure":
            warns.append(f"requirement {rid}: no value and status is not unsure")
        if req.get("status") in ("required", "unsure"):
            visibility = req.get("visibility") if isinstance(req.get("visibility"), Mapping) else {}
            missing = [c for c in cameras if c not in visibility]
            if missing:
                warns.append(f"requirement {rid}: visibility missing for {', '.join(missing)}")

    for obj in _as_list(ep.get("objects")):
        if isinstance(obj, Mapping) and obj.get("first_frame_point") is None:
            warns.append(f"object {obj.get('object_id')}: no first-frame point")
    if _as_list(ep.get("failed_attempts")) and "failed_attempt" not in _as_list(ep.get("hard_tags")):
        warns.append("failed attempts recorded but hard tag failed_attempt missing")
    coarse = [c for c in _as_list(ep.get("coarse_subtasks")) if isinstance(c, Mapping)]
    num_frames = ep.get("num_frames")
    if coarse and isinstance(num_frames, int):
        if coarse[0].get("start_frame") != 0 or coarse[-1].get("end_frame") != num_frames - 1:
            warns.append(f"coarse subtasks do not cover frames 0 to {num_frames - 1}")
    return warns


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


# --------------------------------------------------------------------------- #
# Reading, writing, merging
# --------------------------------------------------------------------------- #
def _round_floats(value: Any) -> Any:
    """Round every float to 6 decimals (and turn -0.0 into 0.0) for byte-stable output."""
    if isinstance(value, float):
        rounded = round(value, _FLOAT_DECIMALS)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, Mapping):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(v) for v in value]
    return value


def dump_gold(doc: Mapping[str, Any]) -> str:
    """Canonical text of a gold document: keys sorted, indent 1, UTF-8 text, LF, final newline."""
    text = json.dumps(_round_floats(doc), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)
    return text + "\n"


def save_gold(doc: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``dump_gold(doc)`` to ``path`` as UTF-8 with LF line endings; return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_gold(doc), encoding="utf-8", newline="\n")
    return out


def load_gold(path: str | Path) -> dict[str, Any]:
    """Read a gold v2 file (or a gold.html export). A UTF-8 byte order mark is tolerated."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _merge_key(ep: Mapping[str, Any]) -> tuple[str, int]:
    try:
        return (str(ep["episode_key"]), int(ep["pass"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"episode without a usable episode_key and pass: {ep!r:.120}") from exc


def _sort_key(ep: Mapping[str, Any]) -> tuple[int, int, str]:
    key, n_pass = _merge_key(ep)
    index = ep.get("episode_index")
    if not isinstance(index, int):
        raise ValueError(f"episode {key} has no integer episode_index")
    return (index, n_pass, key)


def _merge_dataset(old: Any, new: Any) -> Any:
    """Keep one dataset record; refuse to mix repos, revisions or fps (spec 2.1)."""
    if not isinstance(old, Mapping):
        return copy.deepcopy(new)
    if not isinstance(new, Mapping):
        return copy.deepcopy(old)
    merged = dict(copy.deepcopy(old))
    for field, value in new.items():
        if value is None:
            continue
        if merged.get(field) is not None and merged[field] != value:
            raise ValueError(f"cannot merge gold files with different dataset {field}: "
                             f"{merged[field]!r} and {value!r}")
        merged[field] = copy.deepcopy(value)
    return merged


def merge_episodes(existing_doc: Mapping[str, Any] | None, new_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``new_doc`` into ``existing_doc`` by (episode_key, pass); return a new document.

    A new episode replaces the existing one with the same episode_key and pass; the others are
    kept. Episodes come out sorted by (episode_index, pass). Top-level fields of ``new_doc``
    (``guide_version``, ``notes``) replace the existing ones; ``schema_version`` and ``family``
    must agree, and the dataset records must not disagree on repo, revision or fps. Neither input
    is modified.
    """
    new = dict(copy.deepcopy(new_doc))
    merged = dict(copy.deepcopy(existing_doc)) if existing_doc else {}
    for field in ("schema_version", "family"):
        old_value, new_value = merged.get(field), new.get(field)
        if old_value is not None and new_value is not None and old_value != new_value:
            raise ValueError(f"cannot merge gold files with different {field}: "
                             f"{old_value!r} and {new_value!r}")
    if "dataset" in merged or "dataset" in new:
        merged["dataset"] = _merge_dataset(merged.get("dataset"), new.get("dataset"))
    for field, value in new.items():
        if field not in ("episodes", "dataset"):
            merged[field] = value
    by_key: dict[tuple[str, int], Any] = {}
    for ep in [*merged.get("episodes", []), *new.get("episodes", [])]:
        by_key[_merge_key(ep)] = ep
    merged["episodes"] = sorted(by_key.values(), key=_sort_key)
    return merged
