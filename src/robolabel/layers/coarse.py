"""Coarse pass (robolabel v1.1, SPEC_V1_1 3.2): one model call proposes the segments of a clip, then
deterministic post-processing.

Input, in one of two modes:

- ``frames``: frames of one camera evenly spaced at about ``fps_target`` (2) per second, first and last
  frame included, at most ``max_frames`` (48); long side at most 448 px; each image preceded by its
  caption (``frame 160 of 303 (5.33 s), camera up``).
- ``video``: one :class:`VideoPart` that the caller supplies (this package never encodes video); the
  model gives times in seconds, in 0.1 s steps.

The text is the v8 coarse prompt: the task string, the inventory lines when an inventory is given
(targets are then inventory IDs, else plain words), and the candidate events as plain lines when a
candidate list is given (hints, never boundaries to copy). Candidate IDs are ``c1, c2, ...`` in the
order of the list passed, the order ``robolabel.events.candidate_lines`` numbers them.

Post-processing (:func:`postprocess_coarse`) makes the segments contiguous over ``0..N-1``, converts
seconds to frames (``round(t * fps)``) in video mode, snaps gripper-tied boundaries to their event
(SPEC 3.3 item 10), and applies the failure convention of SPEC 4. Every repair is recorded. The same
input always gives the same output. The readings behind the repairs, the snap and the failure convention
are SPEC_QUESTIONS Q162, Q168 and Q169.
"""

from __future__ import annotations

import math
from itertools import groupby
from typing import Any

from ..events import candidate_lines, candidate_map
from ..prompts.v8 import (
    END_EVENTS,
    FAILURE_TYPES,
    MAX_TOKENS,
    OUTCOMES,
    PHASE_CLASSES,
    SCHEMAS,
    load_prompt,
    prompt_sections,
)
from ..providers.base import CallRequest, ImagePart, TextPart, VideoPart
from .frames import camera_label, even_frames, frame_line, model_jpeg

MODES = ("frames", "video")
SNAP_TYPES = ("close_start", "open_start")
# event sources a close or open boundary snaps to: the L1 gripper events, and the re-open after a failed
# close, which the gripper source labels gripper_recovery (SPEC_V1_1 6; SPEC_QUESTIONS Q178)
SNAP_SOURCES = ("gripper", "gripper_recovery")
TEXT_MAX = 120

# the phase that fails first for each failure type (MEASUREMENT_SPEC 3.4.5 and 3.4.9); used only to pick
# the one failed phase of an attempt whose answer marks several phases failed (the pre-v8 convention)
_FAILING_CLASS = {"missed_grasp": "grasp", "slip": "grasp", "wrong_object": "grasp", "drop": "transport",
                  "misplace": "release", "press_no_effect": "press", "aborted": "approach"}


# --------------------------------------------------------------------------- request
def coarse_frame_indices(num_frames: int, fps: float, *, max_frames: int = 48, fps_target: float = 2.0) -> list[int]:
    """Frames at about ``fps_target`` per second over the clip, first and last included, at most ``max_frames``.

    The count is ``round(span_s * fps_target) + 1`` (``span_s`` from the first to the last frame), capped at
    ``max_frames`` and at the number of frames; the frames are evenly spaced (``even_frames``).
    """
    n = int(num_frames)
    if n <= 1:
        return [0]
    span_s = (n - 1) / float(fps)
    k = int(math.floor(span_s * float(fps_target) + 0.5)) + 1
    k = min(k, int(max_frames), n)
    k = max(k, min(2, int(max_frames), n), 1)
    return even_frames(n, k)


def effective_fps(frame_indices: list[int], fps: float) -> float:
    """Frames per second of the spacing actually sent: (count - 1) over the seconds from first to last."""
    if len(frame_indices) < 2 or frame_indices[-1] == frame_indices[0]:
        return 0.0
    return round((len(frame_indices) - 1) * float(fps) / (frame_indices[-1] - frame_indices[0]), 4)


def _frame_getter(episode: Any, camera: str) -> Any:
    cams = (getattr(episode, "extra", None) or {}).get("cameras") or {}
    if camera in cams:
        return cams[camera]
    if not cams or camera == getattr(episode, "camera_key", None):
        return episode.frame
    raise KeyError(f"camera {camera!r} is not a camera of episode {episode.episode_id}")


def coarse_request(episode: Any, *, camera: str, mode: str, context: dict[str, Any],
                   reasoning: dict[str, Any] | None, candidates: list[dict[str, Any]] | None = None,
                   objects: list[dict[str, Any]] | None = None, video: VideoPart | None = None,
                   max_frames: int = 48, fps_target: float = 2.0) -> tuple[CallRequest, dict[str, Any]]:
    """(request, info) for the coarse call. ``info``: ``frame_indices``, ``coarse_fps`` (None in video mode),
    ``mode``."""
    if mode not in MODES:
        raise ValueError(f"coarse mode must be one of {MODES}, not {mode!r}")
    n, fps = int(episode.num_frames), float(episode.fps)
    sec = prompt_sections("coarse")
    duration = f"{n / fps:.1f}"
    cam = camera_label(camera)
    blocks: list[str] = []
    media: list[Any] = []
    if mode == "frames":
        idx = coarse_frame_indices(n, fps, max_frames=max_frames, fps_target=fps_target)
        rate: float | None = effective_fps(idx, fps)
        blocks.append(sec["input_frames"].format(num_frames=n, duration=duration, fps=f"{fps:g}", n_images=len(idx),
                                                 camera=cam, rate=f"{rate:.1f}"))
        units = sec["units_frames"].format(last_frame=n - 1)
        get = _frame_getter(episode, camera)
        for f in idx:
            media.append(TextPart(frame_line(f, n, fps, camera)))
            media.append(ImagePart(model_jpeg(get, f), f"{camera}@{f}"))
    else:
        if not isinstance(video, VideoPart):
            raise ValueError("coarse video mode needs a VideoPart from the caller (the package never encodes video)")
        idx, rate = [], None
        blocks.append(sec["input_video"].format(duration=duration, camera=cam))
        units = sec["units_video"].format(duration=duration)
        media.append(video)
    task = (episode.task or "").strip()
    blocks.append(sec["task"].format(task=task) if task else sec["no_task"])
    if objects:
        from .scene import inventory_lines

        blocks.append(sec["objects"].format(inventory_lines=inventory_lines(objects)))
        target_rule, destination_rule = sec["target_inventory"], sec["destination_inventory"]
    else:
        target_rule, destination_rule = sec["target_words"], sec["destination_words"]
    lines = candidate_lines(list(candidates), n, fps) if candidates else []
    if lines:
        blocks.append(sec["candidates"].format(candidate_lines="\n".join(lines)))
        candidate_rule = sec["candidate_id_list"]
    else:
        candidate_rule = sec["candidate_id_none"]
    blocks.append(sec["rules"].format(units_text=units, target_rule=target_rule, destination_rule=destination_rule,
                                      candidate_rule=candidate_rule))
    ctx = {**context, "frame_indices": list(idx), "cameras": [cam], "coarse_mode": mode}
    if mode == "video":
        ctx["video_seconds"] = float(video.seconds)  # type: ignore[union-attr]
    req = CallRequest(step="coarse", system=load_prompt("system").strip(), parts=[TextPart("\n\n".join(blocks)), *media],
                      schema=SCHEMAS[f"coarse_{mode}"], schema_name=f"coarse_{mode}_v8",
                      max_tokens=MAX_TOKENS["coarse"], reasoning=reasoning, context=ctx)
    return req, {"frame_indices": list(idx), "coarse_fps": rate, "mode": mode}


# --------------------------------------------------------------------------- post-processing
def missing_output_segments(num_frames: int) -> list[dict[str, Any]]:
    """One segment over the clip (spec 4.0) in the v1.1 form: no targets, no failed attempt."""
    last = max(0, int(num_frames) - 1)
    return [{"start_frame": 0, "end_frame": last, "phase_class": "other", "phase_text": "no output",
             "target": "none", "destination": "none", "attempt_idx": 1, "outcome": "success",
             "attempt_outcome": "success", "failure_type": "none", "mistake": False, "end_event": "other",
             "boundary_source": "coarse", "coarse_end_frame": last, "crawl_calls": 0, "candidate_id": "none",
             "evidence": []}]


def _span_frames(s: dict[str, Any], last: int, repairs: list[str]) -> tuple[int, int] | None:
    a, b = s.get("start_frame"), s.get("end_frame")
    try:
        start, end = int(a), int(b)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        repairs.append("coarse: a segment without integer frames was dropped")
        return None
    if type(a) is not int or type(b) is not int:
        repairs.append(f"coarse: frames {a!r}, {b!r} converted to integers ({start}, {end})")
    cs, ce = max(0, min(start, last)), max(0, min(end, last))
    if (cs, ce) != (start, end):
        repairs.append(f"coarse: frames ({start}, {end}) clamped to ({cs}, {ce})")
    if ce < cs:
        repairs.append(f"coarse: start and end swapped ({cs}, {ce})")
        cs, ce = ce, cs
    return cs, ce


def _span_seconds(s: dict[str, Any], fps: float, last: int, repairs: list[str]) -> tuple[int, int] | None:
    """Times to frames: a segment from ``start_s`` to ``end_s`` covers frames ``round(start_s * fps)`` up to
    the frame before ``round(end_s * fps)``, so segments that meet in time meet in frames. A segment that
    covers no frame of the clip is dropped. Rounding past the clip by at most 0.1 s is part of the
    conversion, not a repair."""
    a, b = s.get("start_s"), s.get("end_s")
    try:
        ta, tb = float(a), float(b)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        repairs.append("coarse: a segment without numeric times was dropped")
        return None
    if not (math.isfinite(ta) and math.isfinite(tb)):
        repairs.append("coarse: a segment with a time that is not finite was dropped")
        return None
    if type(a) not in (int, float) or type(b) not in (int, float):
        repairs.append(f"coarse: times {a!r}, {b!r} converted to numbers ({ta:g}, {tb:g})")
    if tb < ta:
        repairs.append(f"coarse: start and end times swapped ({ta:g}, {tb:g})")
        ta, tb = tb, ta
    start, stop = int(round(ta * fps)), int(round(tb * fps))
    if stop <= start:
        repairs.append(f"coarse: the segment from {ta:g} s to {tb:g} s covers no frame, dropped")
        return None
    end = stop - 1
    if start > last or end < 0:
        repairs.append(f"coarse: the segment from {ta:g} s to {tb:g} s lies outside the clip, dropped")
        return None
    slack = max(1, int(round(0.1 * fps)))
    cs, ce = max(0, min(start, last)), max(0, min(end, last))
    if min(start, end) < -slack or max(start, end) > last + slack:
        repairs.append(f"coarse: times ({ta:g}, {tb:g}) s outside the clip, frames clamped to ({cs}, {ce})")
    return cs, ce


def _ref(value: Any, objects: list[dict[str, Any]] | None, repairs: list[str], label: str, at: int) -> str:
    v = str(value if value is not None else "").strip()
    low = v.lower()
    if low in ("none", "unsure"):
        return low
    if not v:
        return "none"
    if objects:
        ids = {str(o["object_id"]).lower(): str(o["object_id"]) for o in objects}
        if low in ids:
            return ids[low]
        for o in objects:  # a name instead of an ID
            if low == str(o["name"]).lower() or low in [str(x).lower() for x in o.get("aliases", [])]:
                repairs.append(f"coarse: {label} {v!r} at frame {at} given as a name, mapped to {o['object_id']}")
                return str(o["object_id"])
        repairs.append(f"coarse: {label} {v!r} at frame {at} is not an inventory ID, set to unsure")
        return "unsure"
    if len(v) > TEXT_MAX:
        repairs.append(f"coarse: {label} at frame {at} cut to {TEXT_MAX} characters")
    return v[:TEXT_MAX]


def _attempt_idx(raw: Any, repairs: list[str], at: int) -> int:
    try:  # an integer or a string of digits; anything else (a float, "2nd", "--5") is attempt 1
        digits = type(raw) is int or (isinstance(raw, str) and raw.lstrip("-").isdecimal())
        idx = max(1, int(raw)) if digits else 1
    except ValueError:
        idx = 1
    if str(idx) != str(raw):
        repairs.append(f"coarse: attempt_idx {raw!r} at frame {at} set to {idx}")
    return idx


def _segment(s: dict[str, Any], span: tuple[int, int], objects: list[dict[str, Any]] | None,
             cmap: dict[str, dict[str, Any]], repairs: list[str]) -> dict[str, Any]:
    start, end = span
    pc = s.get("phase_class")
    if pc not in PHASE_CLASSES:
        repairs.append(f"coarse: phase_class {pc!r} at frame {start} is not a phase class, set to other")
        pc = "other"
    text = str(s.get("phase_text") if s.get("phase_text") is not None else "").strip()
    if not text:
        repairs.append(f"coarse: empty phase_text at frame {start}, set to {pc}")
        text = pc
    elif len(text) > TEXT_MAX:
        repairs.append(f"coarse: phase_text at frame {start} cut to {TEXT_MAX} characters")
    ev = s.get("end_event")
    if ev not in END_EVENTS:
        repairs.append(f"coarse: end_event {ev!r} of the segment at frame {start} set to other")
        ev = "other"
    outcome = s.get("outcome")
    if outcome not in OUTCOMES:
        repairs.append(f"coarse: outcome {outcome!r} at frame {start} is not an outcome, set to success")
        outcome = "success"
    att = s.get("attempt_outcome")
    if att not in OUTCOMES:
        repairs.append(f"coarse: attempt_outcome {att!r} at frame {start} is not an outcome, derived from the phases")
        att = None
    ft = s.get("failure_type")
    if ft not in FAILURE_TYPES:
        repairs.append(f"coarse: failure_type {ft!r} at frame {start} is not a failure type, set to other")
        ft = "other"
    cid = str(s.get("candidate_id") if s.get("candidate_id") is not None else "").strip().lower() or "none"
    if cid != "none" and cid not in cmap:
        repairs.append(f"coarse: unknown candidate {s.get('candidate_id')!r} at frame {start} set to none")
        cid = "none"
    return {"start_frame": start, "end_frame": end, "phase_class": pc, "phase_text": text[:TEXT_MAX],
            "target": _ref(s.get("target"), objects, repairs, "target", start),
            "destination": _ref(s.get("destination"), objects, repairs, "destination", start),
            "attempt_idx": _attempt_idx(s.get("attempt_idx"), repairs, start), "outcome": outcome,
            "attempt_outcome": att, "failure_type": ft, "mistake": False, "end_event": ev,
            "boundary_source": "coarse", "coarse_end_frame": end, "crawl_calls": 0, "candidate_id": cid,
            "evidence": []}


def _contiguous(segs: list[dict[str, Any]], last: int, repairs: list[str]) -> list[dict[str, Any]]:
    """Sort, then cover 0..last without gaps or overlaps: a gap extends the earlier segment, an overlap is
    clipped at the later start, and a segment that starts where an earlier one starts is dropped."""
    segs = sorted(segs, key=lambda s: (s["start_frame"], s["end_frame"]))
    out: list[dict[str, Any]] = []
    for s in segs:
        if out and s["start_frame"] <= out[-1]["start_frame"]:
            repairs.append(f"coarse: segment at {s['start_frame']} starts where the previous one starts, dropped")
            continue
        if out:
            prev = out[-1]
            if prev["end_frame"] != s["start_frame"] - 1:
                kind = "gap" if prev["end_frame"] < s["start_frame"] - 1 else "overlap"
                repairs.append(f"coarse: {kind} before frame {s['start_frame']} closed")
                prev["end_frame"] = s["start_frame"] - 1
        out.append(s)
    if out[0]["start_frame"] != 0:
        repairs.append("coarse: first segment extended to frame 0")
        out[0]["start_frame"] = 0
    if out[-1]["end_frame"] != last:
        repairs.append(f"coarse: last segment {'extended' if out[-1]['end_frame'] < last else 'cut'} to frame {last}")
        out[-1]["end_frame"] = last
    return out


def _snap_to_gripper(out: list[dict[str, Any]], cmap: dict[str, dict[str, Any]], repairs: list[str]) -> None:
    """SPEC 3.3 item 10: a close_start or open_start boundary tied to a gripper candidate of the same type
    (source ``gripper``, or ``gripper_recovery`` for the re-open after a failed close) takes that event's
    frame as its onset (``boundary_source: signal``). Other candidates are hints only."""
    for i in range(len(out) - 1):
        seg, nxt = out[i], out[i + 1]
        cid = seg["candidate_id"]
        ev = cmap.get(cid)
        if ev is None or ev.get("source") not in SNAP_SOURCES or seg["end_event"] not in SNAP_TYPES:
            continue
        if ev.get("type") != seg["end_event"]:
            repairs.append(f"coarse: {cid} is a {ev.get('type')} event, so the {seg['end_event']} boundary at "
                           f"{nxt['start_frame']} was not snapped")
            continue
        f = int(ev["frame"])
        if not seg["start_frame"] < f <= nxt["end_frame"]:
            repairs.append(f"coarse: {cid} at {f} lies outside the segments around the {seg['end_event']} boundary "
                           f"at {nxt['start_frame']}, boundary kept")
            continue
        if f != nxt["start_frame"]:
            repairs.append(f"coarse: {seg['end_event']} onset {nxt['start_frame']} snapped to {cid} at {f}")
        seg["end_frame"] = f - 1
        nxt["start_frame"] = f
        seg["boundary_source"] = "signal"


def _failing_phase(failed: list[dict[str, Any]]) -> dict[str, Any]:
    for s in failed:
        if s["phase_class"] == _FAILING_CLASS.get(s["failure_type"]):
            return s
    not_retract = [s for s in failed if s["phase_class"] != "retract"]
    return (not_retract or failed)[-1]


def _phase_to_mark(group: list[dict[str, Any]], hints: dict[int, str]) -> dict[str, Any]:
    """The phase that carries an attempt's failure when the answer named the failure only in attempt_outcome:
    the phase whose class fits a failure type the answer gave it, else the last phase given a failure type,
    else the last phase that is not a retract."""
    for s in group:
        if s["phase_class"] == _FAILING_CLASS.get(hints.get(id(s), "none")):
            return s
    hinted = [s for s in group if hints.get(id(s), "none") != "none"]
    return hinted[-1] if hinted else _failing_phase(group)


def _failure_convention(out: list[dict[str, Any]], repairs: list[str]) -> None:
    """SPEC 4: each phase's outcome is its own result; one failed phase per attempt; attempt_outcome is
    derived from the attempt's phases and copied onto each; mistake only on the phase that failed.

    When every attempt_outcome the answer gave for an attempt is failed (or aborted) but no phase of the
    attempt has that outcome, the answer's attempt_outcome is kept (deriving success would lose the
    failure) and one phase is marked with it (``_phase_to_mark``): its outcome becomes failed (or aborted),
    its failure_type the one the answer gave it, else other (or aborted), and the repair is recorded."""
    hints = {id(s): s["failure_type"] for s in out}  # before successful phases lose their failure_type
    for s in out:
        if s["outcome"] == "success" and s["failure_type"] != "none":
            repairs.append(f"coarse: failure_type {s['failure_type']!r} of a successful phase at {s['start_frame']} "
                           f"set to none")
            s["failure_type"] = "none"
        elif s["outcome"] != "success" and s["failure_type"] == "none":
            ft = "aborted" if s["outcome"] == "aborted" else "other"
            repairs.append(f"coarse: {s['outcome']} phase at {s['start_frame']} without a failure_type, set to {ft}")
            s["failure_type"] = ft
    top = 0
    for s in out:
        if s["attempt_idx"] < top:
            repairs.append(f"coarse: attempt_idx {s['attempt_idx']} at {s['start_frame']} follows attempt {top}, "
                           f"set to {top}")
            s["attempt_idx"] = top
        top = s["attempt_idx"]
    for idx, grp in groupby(out, key=lambda s: s["attempt_idx"]):
        group = list(grp)
        failed = [s for s in group if s["outcome"] == "failed"]
        if len(failed) > 1:
            keep = _failing_phase(failed)
            for s in failed:
                if s is not keep:
                    s["outcome"], s["failure_type"] = "success", "none"
            repairs.append(f"coarse: attempt {idx} has {len(failed)} failed phases; only the {keep['phase_class']} "
                           f"at {keep['start_frame']} keeps outcome failed")
            failed = [keep]
        derived = "failed" if failed else ("aborted" if any(s["outcome"] == "aborted" for s in group) else "success")
        given = sorted({s["attempt_outcome"] for s in group if s["attempt_outcome"] is not None})
        if derived == "success" and len(given) == 1 and given[0] != "success":
            kept = given[0]
            mark = _phase_to_mark(group, hints)
            hint = hints.get(id(mark), "none")
            mark["outcome"] = kept
            mark["failure_type"] = hint if hint != "none" else ("aborted" if kept == "aborted" else "other")
            repairs.append(f"coarse: attempt {idx} attempt_outcome {kept} but no {kept} phase; kept {kept}, the "
                           f"{mark['phase_class']} at {mark['start_frame']} set to {kept} ({mark['failure_type']})")
            derived = kept
        if given and given != [derived]:
            repairs.append(f"coarse: attempt {idx} attempt_outcome {'/'.join(given)} set to {derived}")
        for s in group:
            s["attempt_outcome"] = derived
    for s in out:
        s["mistake"] = s["outcome"] == "failed"


def postprocess_coarse(data: Any, episode: Any, *, mode: str, info: dict[str, Any] | None = None,
                       objects: list[dict[str, Any]] | None = None, candidates: list[dict[str, Any]] | None = None,
                       repairs: list[str]) -> list[dict[str, Any]]:
    """The coarse answer as v1.1 segment dicts, contiguous over ``0..N-1`` (deterministic; repairs recorded)."""
    if mode not in MODES:
        raise ValueError(f"coarse mode must be one of {MODES}, not {mode!r}")
    if info and info.get("mode") not in (None, mode):
        raise ValueError(f"coarse mode {mode!r} does not match the request's mode {info.get('mode')!r}")
    n, fps = int(episode.num_frames), float(episode.fps)
    last = max(0, n - 1)
    raw = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        repairs.append("coarse: the answer has no list of segments")
        raw = []
    cmap = candidate_map(list(candidates or []))
    segs: list[dict[str, Any]] = []
    for s in raw:
        if not isinstance(s, dict):
            repairs.append("coarse: a segment that is not an object was dropped")
            continue
        span = _span_frames(s, last, repairs) if mode == "frames" else _span_seconds(s, fps, last, repairs)
        if span is not None:
            segs.append(_segment(s, span, objects, cmap, repairs))
    if not segs:
        repairs.append("coarse: no usable segment, used the missing-output segment")
        return missing_output_segments(n)
    out = _contiguous(segs, last, repairs)
    tail = out[-1]
    if tail["end_event"] != "other":
        repairs.append(f"coarse: end_event {tail['end_event']!r} of the last segment set to other")
        tail["end_event"] = "other"
    if tail["candidate_id"] != "none":
        repairs.append(f"coarse: candidate {tail['candidate_id']} on the last segment set to none")
        tail["candidate_id"] = "none"
    for s in out:
        s["coarse_end_frame"] = s["end_frame"]
    _snap_to_gripper(out, cmap, repairs)
    _failure_convention(out, repairs)
    return out
