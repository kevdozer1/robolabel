"""The crawl (SPEC_V1_1 3.3): refine the typed boundaries of the coarse pass to the exact onset frame.

Every boundary whose ``end_event`` is ``close_start``, ``open_start``, ``contact_start`` or ``contact_end`` is
crawled by rule (never because the model said it was unsure). Each call shows up to 8 frames of one camera,
long side 448 px, each preceded by a caption ``image k of 8 (frame F)``, and asks one narrow question for the
event type. The answer is one integer: 2 to 8 when image k is the first that shows the event (it began
between images k-1 and k), 0 when image 1 already shows it, 9 when the fingers, hand or tool are visible and
it has not begun by the last image, -1 when it cannot be judged in these images. An answer of 1 is read as 0.

- Stage 1: 8 frames evenly spaced over plus or minus 1.0 s around the proposed onset, clamped inside the clip
  by shifting the window, never by shrinking it.
- Edges: an answer of 0 or 9 in stage 1 shifts the window by its own width in that direction and asks once
  more. A retry that picks an image continues like a stage-1 pick when it agrees with stage 1 (after a 0 the
  retry pick is at or before stage 1's first image, after a 9 it is after stage 1's last image; near the clip
  edges the shifted window is clamped and overlaps stage 1, so a retry pick can disagree). A retry pick that
  disagrees, or a retry answer at the opposite edge, keeps the stage-1 edge (flags ``crawl_edge`` and
  ``crawl_inconsistent``). A retry that is still at the same edge keeps its farthest frame, and one that gives
  no usable answer keeps the stage-1 edge (flag ``crawl_edge``); stage 2 does not run after any of these.
- An edge gives the onset its answer supports: after a 0 the image's own frame (the event began at or
  before it), after a 9 the frame after the image (the event begins after it). When that onset would lie at
  frame 0 or past the last frame (the window could not move, or reached the clip's edge), the event is not
  inside the clip: the coarse frame stays, as for -1 (flags ``crawl_edge`` and ``crawl_none``).
- Stage 2: from the stage-1 frame before the pick to the pick, both included, at native spacing when they fit
  in 8 frames, else at the finest even spacing that fits. After a 9 and a retry pick it never starts before
  stage 1's last image, which stage 1 said shows no event yet. Skipped when the pick is image 1 or the two
  frames are adjacent. A stage-2 answer of 0 or 9 contradicts stage 1 and keeps the stage-1 pick
  (``crawl_inconsistent``); -1 in stage 2 keeps the stage-1 pick.
- -1 in stage 1 keeps the coarse frame (``crawl_none``). A call that fails, or an answer outside the allowed
  set, keeps the frame known at that point (``crawl_failed``).
- The onset is the pick frame (after a 9 at an edge, the frame after it): the segment after the boundary
  starts there and the one before ends one frame earlier. The log's ``pick`` is the frame of the image the
  deciding answer points at, so it always lies inside a window the model saw. A refined onset that would
  cross a neighbouring boundary keeps the coarse frame (``crawl_cross``); the check runs in time order,
  against the refined previous boundary and the coarse next one.
- Caps: at most 3 calls per boundary and 12 crawled boundaries per episode, in time order. A boundary that
  is not crawled is logged with a ``skipped_*`` flag (``skipped_cap``, ``skipped_type``, ``skipped_signal``,
  ``skipped_stopped``, ``skipped_short``).

The readings of SPEC 3.3 behind these rules are SPEC_QUESTIONS Q164 to Q167.

A segment whose onset the crawl decided gets ``boundary_source: crawl``; one that kept its coarse frame by rule
(``crawl_none``, ``crawl_cross``, a failed first call) keeps its source. Both record ``crawl_calls``, and
``coarse_end_frame`` keeps the coarse end. The window helpers are pure integer arithmetic, so identical input
gives byte-identical frames, and the request text depends only on its inputs.
"""

from __future__ import annotations

import copy
import functools
import math
import re
from collections.abc import Callable
from typing import Any

from ..providers.base import CallRequest, ImagePart, TextPart
from .frames import camera_label, model_jpeg

CRAWL_EVENTS = ("close_start", "open_start", "contact_start", "contact_end")
CONTACT_EVENTS = ("contact_start", "contact_end")
N_IMAGES = 8
HALF_WINDOW_S = 1.0
NOT_BEGUN = 9
FLAGS = ("crawl_edge", "crawl_none", "crawl_inconsistent", "crawl_cross", "crawl_failed", "crawl_call_cap",
         "skipped_cap", "skipped_type", "skipped_signal", "skipped_stopped", "skipped_short")
_MISSING = {"", "none", "unsure", "unknown", "n/a", "na", "null", "nothing"}


# ------------------------------------------------------------------------------------------ windows (pure)
def _even(lo: int, hi: int, n: int) -> list[int]:
    """n frames evenly spaced from lo to hi, both included, rounded half up; repeats dropped."""
    lo, hi = int(lo), int(hi)
    if n <= 1 or hi <= lo:
        return [lo]
    span = hi - lo
    out: list[int] = []
    for i in range(n):
        f = lo + (2 * i * span + (n - 1)) // (2 * (n - 1))
        if not out or f != out[-1]:
            out.append(f)
    return out


def stage1_frames(center: int, num_frames: int, fps: float, n: int = N_IMAGES,
                  half_s: float = HALF_WINDOW_S) -> list[int]:
    """n frames evenly spaced over ``center`` plus or minus ``half_s`` seconds (rounded to whole frames).

    A window that leaves the clip is shifted back inside at full width; only a clip shorter than the window
    gives the whole clip. Frames are sorted and unique.
    """
    last = int(num_frames) - 1
    if last <= 0:
        return [0]
    half = max(1, int(math.floor(float(half_s) * float(fps) + 0.5)))
    c = min(max(int(center), 0), last)
    lo, hi = c - half, c + half
    if lo < 0:
        hi -= lo
        lo = 0
    if hi > last:
        lo -= hi - last
        hi = last
    return _even(max(lo, 0), hi, n)


def shift_window(frames: list[int], direction: int, num_frames: int) -> list[int]:
    """The same window moved by its own width (last frame minus first) earlier (-1) or later (1).

    The moved window keeps its spacing and is shifted back inside the clip when it leaves it; at the clip's
    edge it can come back unchanged.
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1, not {direction!r}")
    fr = [int(f) for f in frames]
    if not fr:
        return []
    last = int(num_frames) - 1
    width = fr[-1] - fr[0]
    moved = [f + direction * width for f in fr]
    if moved[0] < 0:
        d = -moved[0]
        moved = [f + d for f in moved]
    if moved[-1] > last:
        d = moved[-1] - last
        moved = [f - d for f in moved]
    out: list[int] = []
    for f in moved:
        f = min(max(f, 0), max(last, 0))
        if not out or f != out[-1]:
            out.append(f)
    return out


def stage2_frames(lo: int, hi: int, n: int = N_IMAGES) -> list[int]:
    """Frames from ``lo`` to ``hi``, both included: every frame when they fit in n, else n evenly spaced."""
    lo, hi = int(lo), int(hi)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo + 1 <= n:
        return list(range(lo, hi + 1))
    return _even(lo, hi, n)


def interpret(answer: Any, frames: list[int]) -> dict[str, Any]:
    """What one answer means for the frames shown.

    ``kind`` is ``pick`` (image k, k from 2 to len(frames)), ``already`` (0, or 1 read as 0: image 1 shows
    it), ``not_begun`` (9), ``none`` (-1) or ``invalid``. ``index`` and ``frame`` are the image's position
    and frame index (image 1 for ``already``, the last image for ``not_begun``, None otherwise).
    """
    n = len(frames)
    a = answer
    if isinstance(a, float) and not isinstance(a, bool) and a.is_integer():
        a = int(a)
    if isinstance(a, bool) or not isinstance(a, int) or n == 0:
        return {"answer": answer, "read_as": None, "kind": "invalid", "index": None, "frame": None}
    if a == 1:
        a = 0
    if a == 0:
        return {"answer": answer, "read_as": 0, "kind": "already", "index": 0, "frame": int(frames[0])}
    if a == NOT_BEGUN:
        return {"answer": answer, "read_as": NOT_BEGUN, "kind": "not_begun", "index": n - 1,
                "frame": int(frames[-1])}
    if a == -1:
        return {"answer": answer, "read_as": -1, "kind": "none", "index": None, "frame": None}
    if 2 <= a <= n:
        return {"answer": answer, "read_as": a, "kind": "pick", "index": a - 1, "frame": int(frames[a - 1])}
    return {"answer": answer, "read_as": None, "kind": "invalid", "index": None, "frame": None}


# ------------------------------------------------------------------------------------------ the question
def _real(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in _MISSING


def _plain(ref: Any, names: dict[str, str]) -> str:
    v = str(ref).strip()
    v = names.get(v, names.get(v.lower(), v))
    low = v.lower()
    for article in ("the ", "a ", "an "):
        if low.startswith(article):
            v = v[len(article):]
            break
    return " ".join(v.split())


def contact_object(before: dict[str, Any], after: dict[str, Any], event_type: str,
                   names: dict[str, str] | None = None) -> str:
    """The object a contact question names, in plain words, from the segments' target and destination.

    The phase in contact is the one after a ``contact_start`` and the one before a ``contact_end``. When that
    phase's target also acts across the boundary (a held tool, or an object being set down) and it has a
    destination, the contact is with the destination (the pencil on the paper); otherwise with the target
    (the hand on the cup). Inventory IDs are replaced by their names.
    """
    names = names or {}
    inside, other = (after, before) if event_type == "contact_start" else (before, after)
    t, d = inside.get("target"), inside.get("destination")

    def same(a: Any, b: Any) -> bool:
        return _real(a) and _real(b) and _plain(a, names).lower() == _plain(b, names).lower()

    if _real(d) and not same(t, d) and same(t, other.get("target")):
        return _plain(d, names)
    for v in (t, other.get("target"), d, other.get("destination")):
        if _real(v):
            return _plain(v, names)
    return "object"


def _prompt_sections() -> dict[str, str]:
    from ..prompts.v8 import prompt_sections

    return prompt_sections("crawl")


def questions() -> dict[str, str]:
    """Event type to its question text (with an ``{object}`` placeholder for contact events)."""
    out = {}
    for line in _prompt_sections()["questions"].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _fill(template: str, values: dict[str, Any]) -> str:
    """Replace ``{name}`` placeholders in one pass, so text inside the values is never read as a placeholder."""
    return re.sub(r"\{([a-z_]+)\}", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0),
                  template)


def question_for(event_type: str, obj: str | None = None) -> str:
    q = questions()[event_type]
    return _fill(q, {"object": obj or "object"}) if event_type in CONTACT_EVENTS else q


def _clean(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def _answer_check(n: int) -> Callable[[Any], list[str]]:
    allowed = {-1, 0, 1, NOT_BEGUN, *range(2, n + 1)}

    def check(data: Any) -> list[str]:
        a = data.get("answer") if isinstance(data, dict) else None
        if isinstance(a, float) and a.is_integer():
            a = int(a)
        if isinstance(a, bool) or not isinstance(a, int) or a not in allowed:
            return [f"answer must be one integer: 2 to {n}, 0, 9 or -1 (got {a!r})"]
        return []

    return check


def _frame_getter(episode: Any, camera: str | None) -> Callable[[int], Any]:
    cams = (getattr(episode, "extra", None) or {}).get("cameras") or {}
    if camera in cams:
        return cams[camera]
    if not cams or camera is None or camera == getattr(episode, "camera_key", None):
        return episode.frame
    raise KeyError(f"camera {camera!r} is not a camera of episode {getattr(episode, 'episode_id', '?')}")


def crawl_request(episode: Any, frames: list[int], *, camera: str, event_type: str, obj: str | None = None,
                  phases: tuple[str, str] | None = None, context: dict[str, Any] | None = None,
                  reasoning: dict[str, Any] | None = None, stage: str = "stage1",
                  image_tokens_per_image: float = 1500.0, start_mode: str = "json_schema_strict",
                  get_frame: Callable[[int], Any] | None = None) -> CallRequest:
    """One crawl call: the question text, then each image after its caption, then the question again."""
    from ..prompts.v8 import MAX_TOKENS, SCHEMAS, load_prompt

    sec = _prompt_sections()
    n = len(frames)
    question = question_for(event_type, obj)
    phase_line = ""
    if phases is not None:
        before, after = (" ".join(str(p or "").split()) for p in phases)
        if before and after:
            phase_line = _fill(sec["phases"], {"before": before, "after": after})
    label = camera_label(camera) if camera else "main"
    values = {"n": n, "camera": label, "phases": phase_line, "question": question}
    head = _clean(_fill(sec["before_images"], values))
    tail = _clean(_fill(sec["after_images"], values))
    getter = get_frame or _frame_getter(episode, camera)
    parts: list[Any] = [TextPart(head)]
    for k, f in enumerate(frames, start=1):
        parts.append(TextPart(f"image {k} of {n} (frame {int(f)})"))
        parts.append(ImagePart(model_jpeg(getter, int(f)), f"{camera}@{int(f)}"))
    parts.append(TextPart(tail))
    ctx = {**(context or {}), "frame_indices": [int(f) for f in frames], "cameras": [label],
           "crawl_stage": stage, "event_type": event_type}
    return CallRequest(step="crawl", system=load_prompt("system").strip(), parts=parts, schema=SCHEMAS["crawl"],
                       schema_name="crawl_v8", max_tokens=MAX_TOKENS["crawl"], reasoning=reasoning, context=ctx,
                       image_tokens_per_image=image_tokens_per_image, start_mode=start_mode,
                       validate=_answer_check(n))


# ------------------------------------------------------------------------------------------ one boundary
def _edge_result(kind: str, frame: int, coarse: int, num_frames: int,
                 flags: list[str]) -> tuple[int, bool, int | None]:
    """(onset, decided, pick) for an edge answer at image frame ``frame``: 0 (``already``) supports an onset
    at that frame, 9 (``not_begun``) one at the frame after it. An onset at frame 0 or past the last frame
    means the event is not inside the clip, so the coarse frame stays (``crawl_none``)."""
    onset = int(frame) if kind == "already" else int(frame) + 1
    if onset <= 0 or onset >= int(num_frames):
        flags.append("crawl_none")
        return coarse, False, None
    return onset, True, int(frame)


def _crawl_one(ask: Callable[[list[int], str], dict[str, Any]], coarse: int, num_frames: int, fps: float,
               max_calls: int, rec: dict[str, Any]) -> tuple[int, bool, int | None]:
    """(onset, decided by the crawl, pick frame) for one boundary; fills the stage fields and flags of
    ``rec``. The pick frame is the frame of the image the deciding answer points at (None when undecided)."""
    flags = rec["flags"]
    s1 = stage1_frames(coarse, num_frames, fps)
    rec["stage1_frames"] = s1
    it = ask(s1, "stage1")
    rec["stage1_answer"] = it["answer"]
    if it["kind"] in ("invalid", "failed"):
        flags.append("crawl_failed")
        return coarse, False, None
    if it["kind"] == "none":
        flags.append("crawl_none")
        return coarse, False, None
    window, idx = s1, it["index"]
    floor = None  # after a 9 and a retry pick: stage 2 starts no earlier than stage 1's last image
    if it["kind"] in ("already", "not_begun"):
        kind = it["kind"]
        direction = -1 if kind == "already" else 1
        edge = int(it["frame"])
        shifted = shift_window(s1, direction, num_frames)
        if shifted == s1:  # at the clip's edge: nowhere to move
            flags.append("crawl_edge")
            return _edge_result(kind, edge, coarse, num_frames, flags)
        if rec["calls"] >= max_calls:
            flags += ["crawl_edge", "crawl_call_cap"]
            return _edge_result(kind, edge, coarse, num_frames, flags)
        rec["retry_frames"] = shifted
        it2 = ask(shifted, "retry")
        rec["retry_answer"] = it2["answer"]
        if it2["kind"] == "pick":
            f2 = int(shifted[it2["index"]])
            if (f2 > edge) if kind == "already" else (f2 <= edge):
                # the clamped retry window overlaps stage 1 and its pick contradicts stage 1's edge answer
                flags += ["crawl_edge", "crawl_inconsistent"]
                return _edge_result(kind, edge, coarse, num_frames, flags)
            window, idx = shifted, it2["index"]
            if kind == "not_begun":
                floor = edge
        else:
            flags.append("crawl_edge")
            if it2["kind"] in ("invalid", "failed"):
                flags.append("crawl_failed")
            if it2["kind"] == kind:  # still at the same edge: the farthest frame seen
                return _edge_result(kind, int(it2["frame"]), coarse, num_frames, flags)
            if it2["kind"] in ("already", "not_begun"):  # the opposite edge contradicts stage 1
                flags.append("crawl_inconsistent")
            return _edge_result(kind, edge, coarse, num_frames, flags)
    pick = int(window[idx])
    if idx == 0:
        return pick, True, pick
    lo = int(window[idx - 1]) if floor is None else max(int(window[idx - 1]), floor)
    if pick - lo <= 1:
        return pick, True, pick
    if rec["calls"] >= max_calls:
        flags.append("crawl_call_cap")
        return pick, True, pick
    s2 = stage2_frames(lo, pick)
    rec["stage2_frames"] = s2
    it3 = ask(s2, "stage2")
    rec["stage2_answer"] = it3["answer"]
    if it3["kind"] == "pick":
        return int(it3["frame"]), True, int(it3["frame"])
    if it3["kind"] in ("already", "not_begun"):
        flags.append("crawl_inconsistent")
    elif it3["kind"] in ("invalid", "failed"):
        flags.append("crawl_failed")
    return pick, True, pick


def _entry(i: int, event_type: str, coarse: int) -> dict[str, Any]:
    return {"boundary_index": i, "event_type": event_type, "object": None, "coarse_frame": coarse,
            "stage1_frames": None, "stage1_answer": None, "retry_frames": None, "retry_answer": None,
            "stage2_frames": None, "stage2_answer": None, "pick": None, "onset": coarse, "flags": [],
            "calls": 0, "usd": 0.0, "call_log": []}


class _Asker:
    """Sends the crawl calls of one episode and records each call on its boundary's log entry."""

    def __init__(self, episode: Any, caller: Any, *, camera: str, context: dict[str, Any],
                 reasoning: dict[str, Any] | None, image_tokens_per_image: float, start_mode: str):
        self.episode = episode
        self.caller = caller
        self.camera = camera
        self.context = context
        self.reasoning = reasoning
        self.image_tokens_per_image = image_tokens_per_image
        self.start_mode = start_mode
        self.getter: Callable[[int], Any] | None = None
        self.calls: list[Any] = []
        self.stopped = False

    def ask(self, rec: dict[str, Any], phases: tuple[str, str], frames: list[int], stage: str) -> dict[str, Any]:
        if self.getter is None:
            self.getter = _frame_getter(self.episode, self.camera)
        req = crawl_request(self.episode, frames, camera=self.camera, event_type=rec["event_type"],
                            obj=rec["object"], phases=phases,
                            context={**self.context, "boundary_index": rec["boundary_index"]},
                            reasoning=self.reasoning, stage=stage, image_tokens_per_image=self.image_tokens_per_image,
                            start_mode=self.start_mode, get_frame=self.getter)
        res = self.caller.call(req)
        self.calls.append(res)
        usd = float(getattr(res, "usd", 0.0) or 0.0)
        rec["calls"] += 1
        rec["usd"] = round(rec["usd"] + usd, 8)
        if getattr(res, "status", None) == "stopped":
            self.stopped = True
        if getattr(res, "valid", False) and isinstance(res.data, dict):
            it = interpret(res.data.get("answer"), frames)
        else:
            it = {"answer": None, "read_as": None, "kind": "failed", "index": None, "frame": None}
        rec["call_log"].append({"stage": stage, "frames": [int(f) for f in frames], "answer": it["answer"],
                                "read_as": it["read_as"], "status": getattr(res, "status", None), "usd": usd,
                                "wall_s": float(getattr(res, "wall_s", 0.0) or 0.0),
                                "cache_hit": bool(getattr(res, "cache_hit", False))})
        return it


def crawl_boundaries(episode: Any, segments: list[dict[str, Any]], caller: Any, *, camera: str,
                     context: dict[str, Any], reasoning: dict[str, Any] | None,
                     image_tokens_per_image: float = 1500.0, start_mode: str = "json_schema_strict",
                     max_boundaries: int = 12, max_calls_per_boundary: int = 3,
                     skip_types: set[str] | frozenset[str] = frozenset(),
                     objects: list[dict[str, Any]] | None = None
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
    """Crawl the typed boundaries of ``segments`` with ``caller.call(CallRequest) -> CallResult``.

    Returns (new segments, crawl log, call results). The input segments are not changed. ``objects`` (an
    inventory) only turns IDs into names for the contact questions.
    """
    segs = copy.deepcopy(list(segments))
    num_frames = int(episode.num_frames)
    fps = float(episode.fps)
    names = {str(o.get("object_id")): str(o.get("name")) for o in (objects or [])
             if isinstance(o, dict) and o.get("object_id") and o.get("name")}
    for s in segs:
        s.setdefault("coarse_end_frame", int(s["end_frame"]))
        s.setdefault("crawl_calls", 0)
    asker = _Asker(episode, caller, camera=camera, context=context, reasoning=reasoning,
                   image_tokens_per_image=image_tokens_per_image, start_mode=start_mode)
    log: list[dict[str, Any]] = []
    crawled = 0
    for i in range(len(segs) - 1):
        before, after = segs[i], segs[i + 1]
        etype = before.get("end_event")
        if etype not in CRAWL_EVENTS:
            continue
        coarse = int(after["start_frame"])
        rec = _entry(i, etype, coarse)
        log.append(rec)
        if etype in skip_types:
            rec["flags"].append("skipped_type")
            continue
        if before.get("boundary_source") == "signal":
            rec["flags"].append("skipped_signal")
            continue
        if asker.stopped:
            rec["flags"].append("skipped_stopped")
            continue
        if crawled >= max_boundaries:
            rec["flags"].append("skipped_cap")
            continue
        if num_frames < 3:
            rec["flags"].append("skipped_short")
            continue
        crawled += 1
        rec["object"] = contact_object(before, after, etype, names) if etype in CONTACT_EVENTS else None
        phases = (str(before.get("phase_text") or ""), str(after.get("phase_text") or ""))
        ask = functools.partial(asker.ask, rec, phases)
        onset, decided, pick = _crawl_one(ask, coarse, num_frames, fps, max_calls_per_boundary, rec)
        rec["pick"] = pick if decided else None
        prev_onset = int(before["start_frame"])
        next_onset = int(after["end_frame"]) + 1
        if decided and onset != coarse and not prev_onset < onset < next_onset:
            rec["flags"].append("crawl_cross")
            onset, decided = coarse, False
        rec["onset"] = onset
        if onset != coarse:
            before["end_frame"] = onset - 1
            after["start_frame"] = onset
        if decided:
            before["boundary_source"] = "crawl"
        before["crawl_calls"] = int(rec["calls"])
    return segs, log, asker.calls
