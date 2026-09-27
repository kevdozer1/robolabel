"""L3 segment layer (V_LITE L3): one model call labels the phases, then deterministic post-processing.

The model sees up to 28 frames of the first external camera (20 evenly spaced plus frames around the
L1 gripper events) and up to 4 wrist frames after grasps, the inventory, and the L1 candidates and
attempts as plain lines. Post-processing sorts the segments, makes them contiguous over 0..N-1 (a gap
extends the earlier segment, an overlap is clipped at the later start), snaps a segment whose
candidate is confirmed to that candidate's frame, checks targets against the inventory, renders coarse
subtasks with the Appendix A grouping rule and fixed templates, and sets ``mistake`` for failed
segments. Every repair is recorded.
"""

from __future__ import annotations

from typing import Any

from ..eval.lexicon import coarse_groups, object_categorizer, object_namer
from ..prompts.v7 import MAX_TOKENS, PHASE_CLASSES, SCHEMAS, load_prompt
from ..providers.base import CallRequest, TextPart
from .frames import camera_label, image_parts, segment_frame_plan
from .signal import attempt_lines, candidate_lines

OUTCOMES = ("success", "failed", "aborted")


def segments_request(episode: Any, l1: dict[str, Any], objects: list[dict[str, Any]], context: dict[str, Any],
                     reasoning: dict[str, Any] | None) -> tuple[CallRequest, list[dict[str, Any]]]:
    from .scene import inventory_lines

    ext = episode.extra["external_cameras"][0]
    wrist = episode.extra["wrist_cameras"][:1]
    ext_frames, wrist_frames = segment_frame_plan(episode.num_frames, episode.fps, l1)
    items = [(f, ext) for f in ext_frames] + ([(f, wrist[0]) for f in wrist_frames] if wrist else [])
    items.sort(key=lambda fc: (fc[0], 0 if fc[1] == ext else 1))
    parts, manifest = image_parts(episode, items)
    cl = candidate_lines(l1)
    al = attempt_lines(l1)
    text = load_prompt("segments").format(
        num_frames=episode.num_frames, duration=f"{episode.num_frames / episode.fps:.1f}", fps=f"{episode.fps:g}",
        task=episode.task or "", inventory_lines=inventory_lines(objects),
        candidate_lines="\n".join(cl) if cl else "(no gripper event detected)",
        attempt_lines="\n".join(al) if al else "", last_frame=episode.num_frames - 1)
    req = CallRequest(step="segments", system=load_prompt("system").strip(), parts=[TextPart(text), *parts],
                      schema=SCHEMAS["segments"], schema_name="segments_v7", max_tokens=MAX_TOKENS["segments"],
                      reasoning=reasoning,
                      context={**context, "frame_indices": [f for f, _ in items],
                               "cameras": sorted({camera_label(c) for _, c in items})})
    return req, manifest


def missing_output_segments(num_frames: int) -> list[dict[str, Any]]:
    """Spec 4.0: one segment over the episode, no targets, no failed attempts."""
    return [{"start_frame": 0, "end_frame": num_frames - 1, "phase_class": "other", "phase_text": "no output",
             "target": "none", "destination": "none", "attempt_idx": 1, "outcome": "success",
             "failure_type": "none", "boundary_source": "vlm", "candidate_id": "none", "evidence": [],
             "mistake": False}]


def _ref(value: Any, objects: list[dict[str, Any]], repairs: list[str], label: str) -> str:
    v = str(value or "").strip()
    low = v.lower()
    ids = {o["object_id"] for o in objects}
    if low in ids or low in ("none", "unsure"):
        return low
    for o in objects:  # a name instead of an ID
        if low and (low == o["name"].lower() or low in [a.lower() for a in o.get("aliases", [])]):
            repairs.append(f"segments: {label} {v!r} given as a name, mapped to {o['object_id']}")
            return o["object_id"]
    if v:
        repairs.append(f"segments: {label} {v!r} is not an inventory ID, set to unsure")
    return "unsure" if v else "none"


def postprocess_segments(data: Any, num_frames: int, l1: dict[str, Any], objects: list[dict[str, Any]],
                         repairs: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(segments, candidate verdicts) after the deterministic repairs of V_LITE L3."""
    last = num_frames - 1
    raw = data.get("segments", []) if isinstance(data, dict) else []
    cands = {c["candidate_id"]: c for c in l1.get("candidates", [])}
    verdicts = []
    for v in (data.get("candidates", []) if isinstance(data, dict) else []):
        if isinstance(v, dict) and v.get("candidate_id") in cands:
            verdicts.append({"candidate_id": v["candidate_id"], "verdict": v.get("verdict"),
                             "note": str(v.get("note", ""))[:200]})
    confirmed = {v["candidate_id"] for v in verdicts if v["verdict"] == "confirm"}
    segs: list[dict[str, Any]] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        try:
            start, end = int(s.get("start_frame")), int(s.get("end_frame"))
        except (TypeError, ValueError):
            repairs.append("segments: a segment without integer frames was dropped")
            continue
        start, end = max(0, min(start, last)), max(0, min(end, last))
        if end < start:
            start, end = end, start
            repairs.append(f"segments: start and end swapped ({end}, {start})")
        pc = str(s.get("phase_class") or "other")
        if pc not in PHASE_CLASSES:
            pc = "other"
        outcome = s.get("outcome") if s.get("outcome") in OUTCOMES else "success"
        ev = [e for e in (s.get("evidence") or []) if isinstance(e, dict)]
        if len(ev) > 3:
            repairs.append("segments: evidence cut to 3 items")
        segs.append({"start_frame": start, "end_frame": end, "phase_class": pc,
                     "phase_text": str(s.get("phase_text") or pc)[:120],
                     "target": _ref(s.get("target"), objects, repairs, "target"),
                     "destination": _ref(s.get("destination"), objects, repairs, "destination"),
                     "attempt_idx": max(1, int(s.get("attempt_idx") or 1)) if str(s.get("attempt_idx", "")).lstrip(
                         "-").isdigit() else 1,
                     "outcome": outcome, "failure_type": str(s.get("failure_type") or "none"),
                     "boundary_source": s.get("boundary_source") if s.get("boundary_source") in ("signal", "vlm")
                     else "vlm",
                     "candidate_id": str(s.get("candidate_id") or "none"),
                     "evidence": [{"frame": e.get("frame"), "camera": e.get("camera"),
                                   "statement": str(e.get("statement", ""))[:200]} for e in ev[:3]]})
    if not segs:
        repairs.append("segments: no usable segment, used the missing-output segment")
        return missing_output_segments(num_frames), verdicts
    segs.sort(key=lambda s: (s["start_frame"], s["end_frame"]))
    # snap confirmed candidates first (the signal is the timing authority), then make contiguous
    for s in segs:
        cid = s["candidate_id"]
        if cid in confirmed and cid in cands:
            f = int(cands[cid]["frame"])
            if f != s["end_frame"] and s["start_frame"] <= f < last:
                repairs.append(f"segments: end {s['end_frame']} snapped to confirmed {cid} at {f}")
                s["end_frame"] = f
            s["boundary_source"] = "signal"
        elif cid != "none" and s["boundary_source"] == "signal" and cid not in cands:
            s["candidate_id"] = "none"
            s["boundary_source"] = "vlm"
            repairs.append(f"segments: unknown candidate {cid} dropped")
    out: list[dict[str, Any]] = []
    for s in segs:
        if out and s["start_frame"] <= out[-1]["start_frame"]:
            repairs.append(f"segments: segment at {s['start_frame']} overlaps the previous start, dropped")
            continue
        if out:
            prev = out[-1]
            if prev["end_frame"] != s["start_frame"] - 1:
                kind = "gap" if prev["end_frame"] < s["start_frame"] - 1 else "overlap"
                repairs.append(f"segments: {kind} before frame {s['start_frame']} closed")
                prev["end_frame"] = s["start_frame"] - 1
        out.append(s)
    if out[0]["start_frame"] != 0:
        repairs.append("segments: first segment extended to frame 0")
        out[0]["start_frame"] = 0
    if out[-1]["end_frame"] != last:
        repairs.append(f"segments: last segment extended to frame {last}")
        out[-1]["end_frame"] = last
    for s in out:
        s["mistake"] = s["outcome"] == "failed"
        if s["outcome"] == "success":
            s["failure_type"] = "none"
    return out, verdicts


def coarse_subtasks(segments: list[dict[str, Any]], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Appendix A grouping and the fixed templates, naming objects by their inventory names."""
    namer = object_namer(objects)

    def name_of(ref: Any) -> str | None:
        if ref in (None, "none", "unsure"):
            return None
        return namer(ref)

    return coarse_groups(segments, name_of, category_of=object_categorizer(objects))
