"""L2 scene layer (V_LITE L2): an object inventory and per-keyframe scene facts, from the model under test.

Two calls. ``scene_inventory`` sees frame 0 of each external camera and lists at most 10 objects with
IDs, names, categories and a point and box per camera. ``scene_facts`` sees the L1 keyframes from the
external cameras (plus the wrist camera after each grasp and at the end) and reports, per image, what is
visible, what is in the gripper, and a few relations among the task objects. Visibility here is the
model's own report (there is no detector yet), so it is stored with ``source_model`` and not scored as V1.
"""

from __future__ import annotations

import re
from typing import Any

from ..prompts.v7 import MAX_TOKENS, SCHEMAS, load_prompt
from ..providers.base import CallRequest
from .frames import camera_label, image_parts

MAX_OBJECTS = 10


def _clamp01(v: Any) -> float:
    try:
        return round(min(1.0, max(0.0, float(v))), 4)
    except (TypeError, ValueError):
        return 0.0


def label_to_camera(episode: Any) -> dict[str, str]:
    out = {}
    for cam in episode.extra["camera_order"]:
        out[camera_label(cam)] = cam
        out[cam] = cam
    return out


def inventory_request(episode: Any, context: dict[str, Any], reasoning: dict[str, Any] | None) -> tuple[CallRequest, list]:
    cams = episode.extra["external_cameras"][:2]
    parts, manifest = image_parts(episode, [(0, c) for c in cams])
    text = load_prompt("scene_inventory").format(camera_list=", ".join(camera_label(c) for c in cams),
                                                 task=episode.task or "")
    req = CallRequest(step="scene_inventory", system=load_prompt("system").strip(), parts=[_t(text), *parts],
                      schema=SCHEMAS["scene_inventory"], schema_name="scene_inventory_v7",
                      max_tokens=MAX_TOKENS["scene_inventory"], reasoning=reasoning,
                      context={**context, "frame_indices": [0], "cameras": cams})
    return req, manifest


def _t(text: str):
    from ..providers.base import TextPart

    return TextPart(text)


def parse_inventory(data: Any, episode: Any, repairs: list[str]) -> list[dict[str, Any]]:
    """Normalize the model's object list: at most 10, unique IDs o1..oN, known cameras, clamped coordinates."""
    items = data.get("objects", []) if isinstance(data, dict) else []
    if len(items) > MAX_OBJECTS:
        repairs.append(f"scene_inventory: {len(items)} objects, kept the first {MAX_OBJECTS}")
        items = items[:MAX_OBJECTS]
    cams = label_to_camera(episode)
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        oid = str(it.get("object_id") or "").strip().lower()
        if not re.fullmatch(r"o[0-9]+", oid) or oid in seen_ids:
            new = f"o{i + 1}"
            while new in seen_ids:
                new = f"o{int(new[1:]) + 1}"
            repairs.append(f"scene_inventory: object id {oid!r} renamed {new}")
            oid = new
        seen_ids.add(oid)
        views = []
        for v in it.get("views") or []:
            if not isinstance(v, dict):
                continue
            cam = cams.get(str(v.get("camera", "")).strip())
            if cam is None:
                continue
            x0, x1 = sorted((_clamp01(v.get("x0")), _clamp01(v.get("x1"))))
            y0, y1 = sorted((_clamp01(v.get("y0")), _clamp01(v.get("y1"))))
            views.append({"camera": cam, "visible": bool(v.get("visible")), "x": _clamp01(v.get("x")),
                          "y": _clamp01(v.get("y")), "box": [x0, y0, x1, y1]})
        name = str(it.get("name") or "").strip() or f"object {oid}"
        out.append({"object_id": oid, "name": name[:80],
                    "aliases": [str(a).strip()[:60] for a in (it.get("aliases") or []) if str(a).strip()][:6],
                    "category": str(it.get("category") or "other"), "views": views})
    names = [o["name"].lower() for o in out]
    for o in out:  # names must pick out one object: suffix duplicates with the ID
        if names.count(o["name"].lower()) > 1:
            repairs.append(f"scene_inventory: duplicate name {o['name']!r}, suffixed with its ID")
            o["name"] = f"{o['name']} ({o['object_id']})"
    return out


def inventory_lines(objects: list[dict[str, Any]]) -> str:
    if not objects:
        return "(no inventory: name objects in words and use \"unsure\" for IDs)"
    return "\n".join(f"{o['object_id']}: {o['name']} ({o['category']})" for o in objects)


def task_object_ids(objects: list[dict[str, Any]], task: str | None) -> list[str]:
    """Objects the task string names: every content word of the object's name or an alias appears in it."""
    words = set(re.findall(r"[a-z0-9]+", (task or "").lower()))
    stop = {"the", "a", "an", "of", "on", "in", "to", "and", "left", "right", "near", "next", "front", "back"}
    out = []
    for o in objects:
        for nm in [o["name"], *o.get("aliases", [])]:
            toks = [w for w in re.findall(r"[a-z0-9]+", nm.lower()) if w not in stop]
            if toks and all(w in words or w.rstrip("s") in words for w in toks):
                out.append(o["object_id"])
                break
    return out


def facts_request(episode: Any, l1: dict[str, Any], objects: list[dict[str, Any]], context: dict[str, Any],
                  reasoning: dict[str, Any] | None) -> tuple[CallRequest, list[dict[str, Any]]]:
    ext = episode.extra["external_cameras"][:2]
    wrist = episode.extra["wrist_cameras"][:1]
    last = episode.num_frames - 1
    keyframes = list(l1.get("keyframes") or [0, last])[:8]
    items = [(f, c) for f in keyframes for c in ext]
    if wrist:
        q = max(1, int(round(0.25 * episode.fps)))
        picks: list[int] = []
        for a in l1.get("attempts", []):  # after each grasp (first two), then the last frame: at most 3
            f = min(last, int(a["closing_offset"]) + q)
            if f not in picks and f != last:
                picks.append(f)
        wlist = sorted(picks[:2]) + [last]
        items += [(f, wrist[0]) for f in wlist]
    items.sort(key=lambda fc: (fc[0], episode.extra["camera_order"].index(fc[1])))
    parts, manifest = image_parts(episode, items)
    tids = task_object_ids(objects, episode.task)
    text = load_prompt("scene_facts").format(task=episode.task or "", inventory_lines=inventory_lines(objects),
                                             task_objects=", ".join(tids) if tids else "the objects the task names",
                                             last_frame=last)
    req = CallRequest(step="scene_facts", system=load_prompt("system").strip(), parts=[_t(text), *parts],
                      schema=SCHEMAS["scene_facts"], schema_name="scene_facts_v7", max_tokens=MAX_TOKENS["scene_facts"],
                      reasoning=reasoning,
                      context={**context, "frame_indices": sorted({f for f, _ in items}),
                               "cameras": sorted({c for _, c in items})})
    return req, manifest


def parse_facts(data: Any, manifest: list[dict[str, Any]], episode: Any, objects: list[dict[str, Any]],
                repairs: list[str]) -> list[dict[str, Any]]:
    """Keep one entry per shown image; drop unknown IDs; boxes only at the last frame; at most 4 relations."""
    ids = {o["object_id"] for o in objects}
    cams = label_to_camera(episode)
    shown = {(m["frame"], m["camera"]) for m in manifest}
    last = episode.num_frames - 1
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for it in (data.get("facts", []) if isinstance(data, dict) else []):
        if not isinstance(it, dict):
            continue
        try:
            frame = int(it.get("frame"))
        except (TypeError, ValueError):
            continue
        cam = cams.get(str(it.get("camera", "")).strip())
        if (frame, cam) not in shown or (frame, cam) in out:
            continue

        def keep(xs: Any) -> list[str]:
            return [str(x).lower() for x in (xs or []) if str(x).lower() in ids]

        held = str(it.get("in_gripper") or "unsure").strip().lower()
        if held not in ids and held not in ("none", "unsure"):
            held = "unsure"
        rels = []
        for r in (it.get("relations") or [])[:4]:
            if isinstance(r, dict) and str(r.get("subject", "")).lower() in ids and str(r.get("object", "")).lower() in ids:
                rels.append({"subject": str(r["subject"]).lower(), "relation": r.get("relation"),
                             "object": str(r["object"]).lower(), "value": r.get("value")})
        boxes = []
        if frame == last:
            for b in it.get("boxes") or []:
                if isinstance(b, dict) and str(b.get("object_id", "")).lower() in ids:
                    x0, x1 = sorted((_clamp01(b.get("x0")), _clamp01(b.get("x1"))))
                    y0, y1 = sorted((_clamp01(b.get("y0")), _clamp01(b.get("y1"))))
                    boxes.append({"object_id": str(b["object_id"]).lower(), "box": [x0, y0, x1, y1]})
        out[(frame, cam)] = {"frame": frame, "camera": cam, "visible": keep(it.get("visible")),
                             "partial": keep(it.get("partial")), "in_gripper": held, "relations": rels, "boxes": boxes}
    missing = len(shown) - len(out)
    if missing:
        repairs.append(f"scene_facts: {missing} of {len(shown)} images had no valid entry")
    return [out[k] for k in sorted(out, key=lambda fc: (fc[0], fc[1]))]


# ------------------------------------------------------------------------------------------------ v1.1 (additive)
# SPEC_V1_1 3.4: the scene calls of the full v1.1 pipeline work from one camera of the video alone. The
# inventory sees up to 8 evenly spaced frames (first and last included) before the coarse pass; the facts see
# keyframes from the refined boundaries (frame 0, each boundary, the last frame; at most 8) after the crawl.
# The prompts are v8 (``prompts/v8/scene_inventory.txt`` and ``scene_facts.txt``, no robot measurement
# lines); the JSON schemas and the parsers are the v7 ones above, unchanged.

SCENE_FRAMES_V11 = 8
_TYPED_BOUNDARIES = ("close_start", "open_start", "contact_start", "contact_end")


def inventory_frames_v11(num_frames: int, max_frames: int = SCENE_FRAMES_V11) -> list[int]:
    """Up to ``max_frames`` evenly spaced frames of the clip, the first and the last included."""
    from .frames import even_frames

    return even_frames(int(num_frames), max(1, int(max_frames)))


def _pick_even(values: list[int], k: int) -> list[int]:
    """``k`` of ``values`` at evenly spaced positions (all of them when they fit), first and last kept."""
    if k <= 0:
        return []
    if len(values) <= k:
        return list(values)
    if k == 1:
        return [values[(len(values) - 1) // 2]]
    span = len(values) - 1
    return [values[(2 * i * span + (k - 1)) // (2 * (k - 1))] for i in range(k)]


def facts_keyframes_v11(segments: list[dict[str, Any]], num_frames: int,
                        max_frames: int = SCENE_FRAMES_V11) -> list[int]:
    """Keyframes for the v1.1 facts call: frame 0, each boundary (the onset frame, the first frame of the
    segment after it), and the last frame; at most ``max_frames``.

    When the boundaries do not fit, the typed ones (``end_event`` close_start, open_start, contact_start,
    contact_end) come first, then the others; within each group the boundaries are taken at evenly spaced
    positions in time order. Deterministic (integer arithmetic only)."""
    last = max(0, int(num_frames) - 1)
    anchors = sorted({0, last})
    budget = max(0, int(max_frames) - len(anchors))
    typed, other = [], []
    seen: set[int] = set(anchors)
    for before, after in zip(segments, segments[1:], strict=False):
        f = int(after["start_frame"])
        if f in seen or not 0 < f < last:
            continue
        seen.add(f)
        (typed if before.get("end_event") in _TYPED_BOUNDARIES else other).append(f)
    chosen = _pick_even(sorted(typed), budget)
    chosen += _pick_even(sorted(other), budget - len(chosen))
    return sorted(set(anchors) | set(chosen))


def _v8_text(name: str, values: dict[str, Any], task: str | None, extra_sections: tuple[str, ...] = ()) -> str:
    from ..prompts.v8 import prompt_sections

    sec = prompt_sections(name)
    task_text = (task or "").strip()
    blocks = [sec["input"].format(**values),
              sec["task"].format(task=task_text) if task_text else sec["no_task"]]
    blocks += [sec[s].format(**values) for s in extra_sections]
    blocks.append(sec["rules"].format(**values))
    return "\n\n".join(blocks)


def _v8_system() -> str:
    from ..prompts.v8 import load_prompt as load_prompt_v8

    return load_prompt_v8("system").strip()


def inventory_request_v11(episode: Any, *, camera: str, context: dict[str, Any], reasoning: dict[str, Any] | None,
                          max_frames: int = SCENE_FRAMES_V11) -> tuple[CallRequest, list[dict[str, Any]]]:
    """(request, manifest) for the v1.1 inventory: one camera, up to 8 evenly spaced frames incl. first and
    last, the v8 prompt, the v7 ``scene_inventory`` schema (parse the answer with :func:`parse_inventory`)."""
    frames = inventory_frames_v11(episode.num_frames, max_frames)
    parts, manifest = image_parts(episode, [(f, camera) for f in frames])
    text = _v8_text("scene_inventory", {"n_images": len(frames), "camera": camera_label(camera),
                                        "first_frame": frames[0]}, episode.task)
    req = CallRequest(step="scene_inventory", system=_v8_system(), parts=[_t(text), *parts],
                      schema=SCHEMAS["scene_inventory"], schema_name="scene_inventory_v7",
                      max_tokens=MAX_TOKENS["scene_inventory"], reasoning=reasoning,
                      context={**context, "frame_indices": list(frames), "cameras": [camera]})
    return req, manifest


def facts_request_v11(episode: Any, *, camera: str, keyframes: list[int], objects: list[dict[str, Any]],
                      context: dict[str, Any], reasoning: dict[str, Any] | None
                      ) -> tuple[CallRequest, list[dict[str, Any]]]:
    """(request, manifest) for the v1.1 facts: one camera at the given keyframes (sorted, at most 8 in the
    pipeline), the v8 prompt, the v7 ``scene_facts`` schema (parse the answer with :func:`parse_facts`)."""
    last = int(episode.num_frames) - 1
    frames = sorted({min(max(int(f), 0), last) for f in keyframes})
    parts, manifest = image_parts(episode, [(f, camera) for f in frames])
    tids = task_object_ids(objects, episode.task)
    if tids:
        task_objects = ", ".join(tids)
    elif (episode.task or "").strip():
        task_objects = "the objects the task names"
    else:
        task_objects = "the objects that are handled in the clip"
    values = {"n_images": len(frames), "camera": camera_label(camera), "inventory_lines": inventory_lines(objects),
              "task_objects": task_objects, "last_frame": last}
    text = _v8_text("scene_facts", values, episode.task, ("objects",))
    req = CallRequest(step="scene_facts", system=_v8_system(), parts=[_t(text), *parts],
                      schema=SCHEMAS["scene_facts"], schema_name="scene_facts_v7", max_tokens=MAX_TOKENS["scene_facts"],
                      reasoning=reasoning, context={**context, "frame_indices": frames, "cameras": [camera]})
    return req, manifest


def fact_lines(facts: list[dict[str, Any]], frame: int, objects: list[dict[str, Any]]) -> str:
    """Plain lines for the goal call (never JSON inside a string)."""
    names = {o["object_id"]: o["name"] for o in objects}
    rows = [f for f in facts if f["frame"] == frame]
    if not rows:
        return "(no scene facts for this frame)"
    lines = []
    for f in rows:
        cam = camera_label(f["camera"])
        vis = ", ".join(names.get(i, i) for i in f["visible"]) or "nothing listed"
        part = ", ".join(names.get(i, i) for i in f["partial"])
        held = names.get(f["in_gripper"], f["in_gripper"])
        line = f"camera {cam}: visible {vis}"
        if part:
            line += f"; partly visible {part}"
        line += f"; in the gripper: {held}"
        for r in f["relations"]:
            line += f"; {names.get(r['subject'], r['subject'])} {str(r['relation']).replace('_', ' ')} " \
                    f"{names.get(r['object'], r['object'])}: {r['value']}"
        lines.append(line)
    return "\n".join(lines)
