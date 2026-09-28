"""L3 segment layer (V_LITE L3): one model call labels the phases, then deterministic post-processing.

The model sees up to 28 frames of the first external camera (20 evenly spaced plus frames around the
L1 gripper events) and up to 4 wrist frames after grasps, the inventory, and the L1 candidates and
attempts as plain lines. Post-processing sorts the segments, makes them contiguous over 0..N-1 (a gap
extends the earlier segment, an overlap is clipped at the later start), then snaps a segment whose
candidate is confirmed to that candidate's frame (``boundary_source: signal`` only when the final end
is that frame), checks targets against the inventory, renders coarse subtasks with the Appendix A
grouping rule and fixed templates, and sets ``mistake`` for failed segments. Every repair, clamp and
coercion is recorded.
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
            repairs.append("segments: a segment that is not an object was dropped")
            continue
        try:
            start, end = int(s.get("start_frame")), int(s.get("end_frame"))
        except (TypeError, ValueError, OverflowError):
            repairs.append("segments: a segment without integer frames was dropped")
            continue
        if type(s.get("start_frame")) is not int or type(s.get("end_frame")) is not int:
            repairs.append(f"segments: frames {s.get('start_frame')!r}, {s.get('end_frame')!r} converted to "
                           f"integers ({start}, {end})")
        cs, ce = max(0, min(start, last)), max(0, min(end, last))
        if (cs, ce) != (start, end):
            repairs.append(f"segments: frames ({start}, {end}) clamped to ({cs}, {ce})")
        start, end = cs, ce
        if end < start:
            start, end = end, start
            repairs.append(f"segments: start and end swapped ({end}, {start})")
        pc = s.get("phase_class")
        if pc not in PHASE_CLASSES:
            repairs.append(f"segments: phase_class {pc!r} at frame {start} is not a phase class, set to other")
            pc = "other"
        outcome = s.get("outcome")
        if outcome not in OUTCOMES:
            repairs.append(f"segments: outcome {outcome!r} at frame {start} is not an outcome, set to success")
            outcome = "success"
        text = str(s.get("phase_text") or pc)
        if not s.get("phase_text"):
            repairs.append(f"segments: empty phase_text at frame {start}, set to {pc}")
        elif len(text) > 120:
            repairs.append(f"segments: phase_text at frame {start} cut to 120 characters")
        raw_idx = s.get("attempt_idx")
        try:  # an integer or a string of digits; anything else (a float, "2nd", "--5") is attempt 1
            digits = type(raw_idx) is int or (isinstance(raw_idx, str) and raw_idx.lstrip("-").isdecimal())
            idx = max(1, int(raw_idx)) if digits else 1
        except ValueError:
            idx = 1
        if str(idx) != str(raw_idx):
            repairs.append(f"segments: attempt_idx {raw_idx!r} at frame {start} set to {idx}")
        src = s.get("boundary_source")
        if src not in ("signal", "vlm"):
            repairs.append(f"segments: boundary_source {src!r} at frame {start} set to vlm")
            src = "vlm"
        cid = str(s.get("candidate_id") or "none")
        if cid != "none" and cid not in cands:
            repairs.append(f"segments: unknown candidate {cid} dropped")
            cid = "none"
        all_ev = s.get("evidence") or []
        ev = [e for e in all_ev if isinstance(e, dict)] if isinstance(all_ev, list) else []
        if not isinstance(all_ev, list) or len(ev) < len(all_ev):
            repairs.append(f"segments: evidence that is not a list of objects dropped at frame {start}")
        if len(ev) > 3:
            repairs.append("segments: evidence cut to 3 items")
        if any(len(str(e.get("statement", ""))) > 200 for e in ev[:3]):
            repairs.append(f"segments: evidence statement at frame {start} cut to 200 characters")
        segs.append({"start_frame": start, "end_frame": end, "phase_class": pc,
                     "phase_text": text[:120],
                     "target": _ref(s.get("target"), objects, repairs, "target"),
                     "destination": _ref(s.get("destination"), objects, repairs, "destination"),
                     "attempt_idx": idx,
                     "outcome": outcome, "failure_type": str(s.get("failure_type") or "none"),
                     "boundary_source": src, "candidate_id": cid,
                     "evidence": [{"frame": e.get("frame"), "camera": e.get("camera"),
                                   "statement": str(e.get("statement", ""))[:200]} for e in ev[:3]]})
    if not segs:
        repairs.append("segments: no usable segment, used the missing-output segment")
        return missing_output_segments(num_frames), verdicts
    segs.sort(key=lambda s: (s["start_frame"], s["end_frame"]))
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
    # then snap to confirmed candidates (the signal is the timing authority): the boundary moves to the
    # candidate frame when that frame lies inside this segment or the next one, else it stays as it is
    for i, s in enumerate(out):
        cid = s["candidate_id"]
        f = int(cands[cid]["frame"]) if cid in confirmed else None
        if f is not None and f != s["end_frame"]:
            nxt = out[i + 1] if i + 1 < len(out) else None
            if nxt is not None and s["start_frame"] <= f < nxt["end_frame"]:
                repairs.append(f"segments: end {s['end_frame']} snapped to confirmed {cid} at {f}")
                s["end_frame"] = f
                nxt["start_frame"] = f + 1
            else:
                repairs.append(f"segments: confirmed {cid} at {f} cannot end the segment "
                               f"{s['start_frame']}-{s['end_frame']}, boundary kept")
        # signal only when the final end is the confirmed candidate's frame
        src = "signal" if f is not None and s["end_frame"] == f else "vlm"
        if src != s["boundary_source"]:
            repairs.append(f"segments: boundary_source of the segment ending at {s['end_frame']} set to {src}")
            s["boundary_source"] = src
    for s in out:
        s["mistake"] = s["outcome"] == "failed"
        if s["outcome"] == "success" and s["failure_type"] != "none":
            repairs.append(f"segments: failure_type {s['failure_type']!r} of a successful segment at "
                           f"{s['start_frame']} set to none")
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
