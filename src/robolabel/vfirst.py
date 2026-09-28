"""Video first (SPEC_V1_1 3): the timing pipeline of experiment E1, from video alone or with an event source.

``run_timing`` runs, for one episode and one camera:

1. the event source (``none``, ``motion`` or ``gripper``; SPEC 3.1), whose events go to the coarse pass as a
   candidate list unless the source is ``none``;
2. the coarse pass (one call, frames or native video; SPEC 3.2), post-processed into contiguous segments with
   typed boundaries (``robolabel.layers.coarse``);
3. the crawl (SPEC 3.3) on every boundary typed ``close_start``, ``open_start``, ``contact_start`` or
   ``contact_end``, when a crawl caller is given. With the ``gripper`` source, a ``close_start`` or
   ``open_start`` boundary that the coarse pass tied to a gripper event of the same type takes the L1 event
   frame in the coarse post-processing (``boundary_source: signal``) and is not crawled (flag
   ``skipped_signal``). A gripper-typed boundary tied to no gripper event has no signal frame, so it is
   crawled like any typed boundary (SPEC_QUESTIONS Q162: the reading of SPEC 3.3 item 10 chosen here).

There are no inventory, facts or goal calls: E1 scores timing only, and targets are plain words. When the
coarse call is not valid (invalid, failed, refused, unavailable or stopped), the episode gets the
missing-output segment (one segment over the clip, ``end_event: other``), no crawl runs, and
``coarse_status`` and ``repairs`` say why. Any object with ``.call(CallRequest) -> CallResult`` is a caller.

``run_episode_v11`` is the full v1.1 pipeline (SPEC_V1_1 3.4), in this order: L2 inventory on up to 8
evenly spaced frames of one camera (first and last included), the coarse pass with the inventory IDs, the
crawl, L2 facts on keyframes from the refined boundaries (frame 0, each boundary, the last frame; at most
8), the L4 goal (prompt v8, with ``has_end_state`` and ``goal_command``), then the L5 checks of v1.1. Its
view record is the V-lite view plus the SPEC 3.5 fields, and its rows are the v7 rows plus the v1.1
fields. See its docstring for the gripper source and for a call that fails.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .events import get_source
from .layers.coarse import coarse_request, missing_output_segments, postprocess_coarse
from .layers.crawl import crawl_boundaries
from .prompts.v8 import VERSION as PROMPT_VERSION
from .prompts.v8 import prompt_sha256

GRIPPER_TYPES = frozenset({"close_start", "open_start"})


def prompt_hashes() -> dict[str, str]:
    """SHA-256 of the v8 prompt files the timing pipeline sends (coarse, crawl and the shared system text)."""
    return {"coarse": prompt_sha256("coarse"), "crawl": prompt_sha256("crawl"), "system": prompt_sha256("system")}


def run_timing(episode: Any, *, camera: str, coarse_caller: Any, crawl_caller: Any = None,
               coarse_mode: str = "frames", video: Any = None, event_source: str = "none",
               l1: dict[str, Any] | None = None, coarse_context: dict[str, Any], crawl_context: dict[str, Any],
               reasoning: dict[str, Any] | None = None, coarse_image_tokens: float = 1500.0,
               crawl_image_tokens: float = 1500.0, start_mode: str = "json_schema_strict") -> dict[str, Any]:
    """Coarse pass plus crawl on one episode. Returns segments before and after the crawl, the crawl log,
    every call result, the events, and the run's cost, time and prompt hashes."""
    t0 = time.perf_counter()
    repairs: list[str] = []
    calls: list[Any] = []
    events = get_source(event_source).events(episode, camera=camera, l1=l1)
    candidates = list(events) if event_source != "none" else None
    req, info = coarse_request(episode, camera=camera, mode=coarse_mode, context=coarse_context,
                               reasoning=reasoning, candidates=candidates, video=video)
    req.image_tokens_per_image = coarse_image_tokens
    req.start_mode = start_mode
    res = coarse_caller.call(req)
    calls.append(res)
    coarse_ok = bool(getattr(res, "valid", False)) and res.data is not None
    if coarse_ok:
        segments_coarse = postprocess_coarse(res.data, episode, mode=coarse_mode, info=info,
                                             candidates=candidates, repairs=repairs)
    else:
        repairs.append(f"coarse {res.status}: missing-output segment (spec 4.0)"
                       + (f": {res.error}" if getattr(res, "error", None) else ""))
        segments_coarse = missing_output_segments(episode.num_frames)
    crawl_enabled = crawl_caller is not None
    crawl_log: list[dict[str, Any]] = []
    crawl_ran = False
    if crawl_enabled and coarse_ok:
        # snapped boundaries (boundary_source signal) are skipped inside crawl_boundaries; no type is skipped
        segments, crawl_log, crawl_calls = crawl_boundaries(
            episode, segments_coarse, crawl_caller, camera=camera, context=crawl_context, reasoning=reasoning,
            image_tokens_per_image=crawl_image_tokens, start_mode=start_mode, skip_types=frozenset())
        calls.extend(crawl_calls)
        crawl_ran = True
    else:
        segments = copy.deepcopy(segments_coarse)
    return {
        "segments_coarse": segments_coarse,
        "segments": segments,
        "crawl_log": crawl_log,
        "calls": calls,
        "events": events,
        "coarse_info": info,
        "coarse_status": res.status,
        "coarse_error": getattr(res, "error", None),
        "repairs": repairs,
        "cost_usd": round(sum(float(getattr(c, "usd", 0.0) or 0.0) for c in calls), 8),
        "unreconciled_usd": round(sum(float(getattr(c, "unreconciled_reserved_usd", 0.0) or 0.0) for c in calls), 8),
        "wall_s": round(sum(float(getattr(c, "wall_s", 0.0) or 0.0) for c in calls), 3),
        "episode_wall_s": round(time.perf_counter() - t0, 3),
        "event_sources": [event_source],
        "coarse_mode": coarse_mode,
        "coarse_fps": info.get("coarse_fps") if isinstance(info, dict) else None,
        "crawl_enabled": crawl_enabled,
        "crawl_ran": crawl_ran,
        "crawl_calls": sum(int(e.get("calls", 0)) for e in crawl_log),
        "prompt_version": PROMPT_VERSION,
        "prompt_hashes": prompt_hashes(),
    }


# ============================================================================================ the full v1.1 pipeline
FULL_STEPS = ("scene_inventory", "coarse", "crawl", "scene_facts", "goal")
BAD_STATUSES = ("invalid", "failed", "refused", "unavailable", "stopped")
PIPELINE_VERSION_V11 = f"v1.1 {PROMPT_VERSION}"
V11_PROMPTS = ("system", "coarse", "crawl", "scene_inventory", "scene_facts", "goal")


def pipeline_code_v11(root: Path | None = None) -> str:
    """First 12 hex of SHA-256 over the code that writes v1.1 views and rows: ``layers/*.py``,
    ``events/*.py``, ``vfirst.py``, ``vlite.py``, ``schema_v7.py`` and the files of ``prompts/v7`` and
    ``prompts/v8``, in sorted path order, each as its path relative to the package and its bytes with line
    endings normalized to LF (the rule of ``vlite.pipeline_code``)."""
    pkg = root or Path(__file__).resolve().parent
    files = [*pkg.glob("layers/*.py"), *pkg.glob("events/*.py"), pkg / "vfirst.py", pkg / "vlite.py",
             pkg / "schema_v7.py"]
    for sub in ("v7", "v8"):
        files += [p for p in (pkg / "prompts" / sub).glob("*") if p.is_file() and p.suffix != ".pyc"]
    h = hashlib.sha256()
    for rel, path in sorted((p.relative_to(pkg).as_posix(), p) for p in files):
        h.update(rel.encode("utf-8") + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


PIPELINE_CODE_V11 = pipeline_code_v11()


def prompt_hashes_v11() -> dict[str, str]:
    """SHA-256 of each v8 prompt file the full pipeline sends (those present)."""
    from .prompts.v8 import HERE

    return {name: prompt_sha256(name) for name in V11_PROMPTS if (HERE / f"{name}.txt").is_file()}


def with_camera(episode: Any, camera: str) -> Any:
    """The episode with the camera metadata the scene and goal layers read (``extra["cameras"]``,
    ``camera_order``, ``external_cameras``, ``wrist_cameras``, and ``camera_sizes`` from frame 0) filled in
    for ``camera`` where it is missing.

    The episode passed is never changed: a copy (``dataclasses.replace``) carries the added keys. A camera
    that is neither in ``extra["cameras"]`` nor the episode's own camera raises KeyError."""
    extra = dict(getattr(episode, "extra", None) or {})
    cams = dict(extra.get("cameras") or {})
    changed = False
    if camera not in cams:
        if cams and camera != getattr(episode, "camera_key", None):
            raise KeyError(f"camera {camera!r} is not a camera of episode {getattr(episode, 'episode_id', '?')}")
        cams[camera] = episode.frame
        extra["cameras"] = cams
        changed = True
    order = list(extra.get("camera_order") or [])
    if camera not in order:
        extra["camera_order"] = order + [camera]
        changed = True
    for key, default in (("external_cameras", [camera]), ("wrist_cameras", [])):
        if key not in extra:
            extra[key] = list(default)
            changed = True
    sizes = dict(extra.get("camera_sizes") or {})
    if camera not in sizes:  # [width, height] of the camera's frames, read from frame 0
        shape = getattr(cams[camera](0), "shape", None)
        if shape is not None and len(shape) >= 2:
            sizes[camera] = [int(shape[1]), int(shape[0])]
            extra["camera_sizes"] = sizes
            changed = True
    if not changed:
        return episode
    if not dataclasses.is_dataclass(episode):
        raise TypeError("the episode lacks camera metadata and is not a dataclass that can be copied")
    return dataclasses.replace(episode, extra=extra)


def attempt_records_v11(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per attempt of the segments (SPEC_V1_1 4): the whole span from the attempt's first phase to
    its last, ``outcome`` = the attempt's ``attempt_outcome`` (derived by the old rule where a segment has
    none), the failure type of its failed (or aborted) phase, and ``evident_frame``, the last frame of that
    phase (None for a successful attempt)."""
    from .schema_v7 import fill_attempt_outcome

    rank = {"failed": 2, "aborted": 1, "success": 0}
    groups: dict[int, list[dict[str, Any]]] = {}
    for s in fill_attempt_outcome(segments):
        groups.setdefault(int(s.get("attempt_idx") or 1), []).append(s)
    out = []
    for idx, ss in groups.items():
        outcome = max((str(s.get("attempt_outcome") or "success") for s in ss), key=lambda o: rank.get(o, 0))
        bad = [s for s in ss if s.get("outcome") in ("failed", "aborted")]
        out.append({"attempt_idx": idx, "start": min(int(s["start_frame"]) for s in ss),
                    "end": max(int(s["end_frame"]) for s in ss), "outcome": outcome,
                    "failure_type": str(bad[0].get("failure_type") or "other") if bad else "none",
                    "evident_frame": int(bad[0]["end_frame"]) if bad else None, "source": "vlm"})
    return out


def signal_attempt_records(l1: dict[str, Any]) -> list[dict[str, Any]]:
    """L1's attempts in the shape of the view's attempt records (``source: signal``), with the values the v7
    attempt rows give them: the span from the closing onset to the attempt's end, L1's own outcome (hold,
    empty, slip, released, aborted, unknown) and failure type, and the event frame as ``evident_frame``."""
    return [{"attempt_idx": int(a["attempt_idx"]), "start": int(a["closing_onset"]), "end": int(a["end_frame"]),
             "outcome": a["outcome"], "failure_type": a["failure_type"], "evident_frame": int(a["event_frame"]),
             "source": "signal"} for a in l1.get("attempts") or []]


def _signal_keyframes(l1: dict[str, Any], num_frames: int, max_frames: int) -> list[int]:
    """The v1.1 keyframe plan over every L1 event (Q154) when the gripper source is on; the record's own
    ``keyframes`` when that plan cannot be made."""
    try:
        from .layers.signal import Calibration, keyframe_plan_every_event

        cal = Calibration(**dict(l1["calibration"]))
        return [int(f) for f in keyframe_plan_every_event(l1, int(num_frames), cal, max_frames=max_frames)]
    except (ImportError, KeyError, TypeError, ValueError):
        last = int(num_frames) - 1
        return sorted({min(max(int(f), 0), last) for f in (l1.get("keyframes") or [0, last])})[:max_frames]


def _crawl_step_status(results: list[Any]) -> str:
    if not results:
        return "none"
    bad = [str(getattr(r, "status", "")) for r in results if getattr(r, "status", None) != "ok"]
    return bad[0] if bad else "ok"


def run_episode_v11(episode: Any, *, camera: str, caller: Any, crawl_caller: Any = None, event_source: str = "none",
                    l1: dict[str, Any] | None = None, coarse_mode: str = "frames", video: Any = None,
                    context: dict[str, Any], crawl_context: dict[str, Any] | None = None,
                    reasoning: dict[str, Any] | None = None, image_tokens_per_image: float = 1500.0,
                    crawl_image_tokens: float = 1500.0, start_mode: str = "json_schema_strict", crawl: bool = True,
                    scene_max_frames: int = 8) -> dict[str, Any]:
    """The full v1.1 pipeline on one episode and one camera (SPEC_V1_1 3.4), with
    ``caller.call(CallRequest) -> CallResult`` for every model step.

    Order: L2 inventory (up to ``scene_max_frames`` evenly spaced frames, first and last included), the coarse
    pass (frames or native video, with the inventory IDs when the inventory lists objects, else plain words),
    the crawl (``crawl_caller``, or ``caller`` when it is None: the same model; ``crawl=False`` turns it off),
    L2 facts on keyframes from the refined boundaries, the L4 goal (prompt v8), then the L5 checks of v1.1.

    ``event_source``: ``none`` (video alone), ``motion`` (pixel candidates) or ``gripper`` (the gripper_on
    extra: ``l1`` is required, its events go in as candidates, close/open boundaries tied to them take the L1
    frame and are not crawled, D1a decides robot items, rules 1, 2, 6, 7 and 8 run, and the facts keyframes
    follow the L1 plan over every event). ``l1`` with any other source raises ValueError: a hidden signal never
    reaches a prompt or a decision. So does an ``l1`` whose ``num_frames`` is not the episode's frame count (its
    events would not line up with the frames). The view's ``attempts`` and the rows' attempt records with
    ``attempt_source: vlm`` come from the segments in every run; with the gripper source, L1's attempts are
    kept beside them (the view's ``attempts_signal``, the rows' ``attempt_source: signal``).

    A call that is not valid does not end the episode: an invalid inventory leaves targets in plain words, an
    invalid coarse call gives the missing-output segment (no crawl), invalid facts leave the goal without facts,
    an invalid goal leaves no goal record. A ``stopped`` call (the spend guard) ends the model calls of the
    episode. Every such case is in ``repairs``, and the failed calls go to the checks, which route the episode
    with risk 1.0 (D1b). An empty inventory skips the facts call.

    Returns ``view``, ``rows``, ``calls``, ``repairs``, ``segments_coarse``, ``segments``, ``crawl_log``,
    ``goal``, ``objects`` and ``facts``, plus ``checks``, ``events``, ``coarse_info``, ``step_status``,
    ``failed_calls``, ``stopped``, ``cost_usd`` and ``episode_wall_s``.
    """
    from .layers.check import run_checks_v11
    from .layers.goal import goal_request_v11, postprocess_goal_v11, raw_goal_refs
    from .layers.scene import (
        facts_keyframes_v11,
        facts_request_v11,
        inventory_frames_v11,
        inventory_request_v11,
        parse_facts,
        parse_inventory,
    )
    from .layers.segment import coarse_subtasks
    from .schema_v7 import add_v11_fields, episode_rows
    from .vlite import build_view

    t0 = time.perf_counter()
    signal = event_source == "gripper"
    if signal and l1 is None:
        raise ValueError("the gripper source needs the episode's L1 record (l1=...)")
    if not signal and l1 is not None:
        raise ValueError(f"l1 was given with event source {event_source!r}: pass l1 only with the gripper source, "
                         "so a hidden signal never reaches a prompt or a decision")
    ep = with_camera(episode, camera)
    n = int(ep.num_frames)
    if signal and int((l1 or {}).get("num_frames") or 0) != n:
        raise ValueError(f"the L1 record has {(l1 or {}).get('num_frames')} frames and the episode {n}: its events "
                         "would not line up with the frames the model sees")
    ctx = dict(context)
    cctx = dict(crawl_context) if crawl_context is not None else ctx
    events = get_source(event_source).events(ep, camera=camera, l1=l1)
    candidates = list(events) if event_source != "none" else None
    repairs: list[str] = []
    calls: list[Any] = []
    log: list[tuple[str, Any]] = []
    step_status: dict[str, str] = {}
    state = {"stopped": False}

    def do(step: str, req: Any) -> Any:
        if state["stopped"]:
            step_status[step] = "not run"
            return None
        req.image_tokens_per_image = image_tokens_per_image
        req.start_mode = start_mode
        res = caller.call(req)
        calls.append(res)
        log.append((step, res))
        step_status[step] = str(res.status)
        if res.status == "stopped":
            state["stopped"] = True
        return res

    def status_of(res: Any) -> str:
        return str(res.status) if res is not None else "not run"

    # L2 inventory
    req, _ = inventory_request_v11(ep, camera=camera, context=ctx, reasoning=reasoning, max_frames=scene_max_frames)
    inventory_frames = inventory_frames_v11(n, scene_max_frames)
    res = do("scene_inventory", req)
    objects: list[dict[str, Any]] = []
    if res is not None and res.valid:
        objects = parse_inventory(res.data, ep, repairs)
    else:
        repairs.append(f"scene_inventory {status_of(res)}: no inventory, targets in plain words")
    have_inventory = bool(objects)
    # coarse pass
    req, info = coarse_request(ep, camera=camera, mode=coarse_mode, context=ctx, reasoning=reasoning,
                               candidates=candidates, objects=objects or None, video=video)
    res = do("coarse", req)
    coarse_status = status_of(res)
    coarse_ok = bool(res is not None and res.valid and res.data is not None)
    raw_refs: list[str] = []
    if coarse_ok:
        raw = res.data.get("segments") if isinstance(res.data, dict) else None
        for s in raw if isinstance(raw, list) else []:
            if isinstance(s, dict):
                raw_refs += [str(s.get("target", "")), str(s.get("destination", ""))]
        segments_coarse = postprocess_coarse(res.data, ep, mode=coarse_mode, info=info, objects=objects or None,
                                             candidates=candidates, repairs=repairs)
    else:
        err = getattr(res, "error", None) if res is not None else None
        repairs.append(f"coarse {coarse_status}: missing-output segment (spec 4.0)" + (f": {err}" if err else ""))
        segments_coarse = missing_output_segments(n)
    no_output = not coarse_ok or segments_coarse == missing_output_segments(n)
    # the crawl
    crawl_enabled = bool(crawl)
    crawl_by = crawl_caller if crawl_caller is not None else caller
    crawl_model = str(getattr(crawl_by, "model", "unknown")) if crawl_enabled else None
    crawl_log: list[dict[str, Any]] = []
    if crawl_enabled and coarse_ok and not state["stopped"]:
        segments, crawl_log, crawl_results = crawl_boundaries(
            ep, segments_coarse, crawl_by, camera=camera, context=cctx, reasoning=reasoning,
            image_tokens_per_image=crawl_image_tokens, start_mode=start_mode, objects=objects or None)
        calls.extend(crawl_results)
        log.extend(("crawl", r) for r in crawl_results)
        step_status["crawl"] = _crawl_step_status(crawl_results)
        if any(getattr(r, "status", None) == "stopped" for r in crawl_results):
            state["stopped"] = True
    else:
        segments = copy.deepcopy(segments_coarse)
        step_status["crawl"] = "off" if not crawl_enabled else "not run"
    # L2 facts on keyframes from the refined boundaries (or the L1 plan with the gripper source)
    if signal:
        keyframes, keyframe_source = _signal_keyframes(l1 or {}, n, scene_max_frames), "signal"
    else:
        keyframes, keyframe_source = facts_keyframes_v11(segments, n, scene_max_frames), "boundaries"
    facts: list[dict[str, Any]] = []
    if objects:
        req, manifest = facts_request_v11(ep, camera=camera, keyframes=keyframes, objects=objects, context=ctx,
                                          reasoning=reasoning)
        res = do("scene_facts", req)
        if res is not None and res.valid:
            facts = parse_facts(res.data, manifest, ep, objects, repairs)
        else:
            repairs.append(f"scene_facts {status_of(res)}: no scene facts")
    elif state["stopped"]:
        step_status["scene_facts"] = "not run"
        repairs.append("scene_facts not run: an earlier call was stopped")
    else:
        step_status["scene_facts"] = "skipped"
        repairs.append("scene_facts skipped: the inventory lists no object")
    have_facts = bool(facts)
    # L4 goal
    goal = None
    req = goal_request_v11(ep, camera=camera, objects=objects, facts=facts, context=ctx, reasoning=reasoning,
                           l1=l1 if signal else None, signal=signal)
    res = do("goal", req[0] if isinstance(req, tuple) else req)  # a (request, manifest) pair or a request
    if res is not None and res.valid and isinstance(res.data, dict):
        raw_refs += raw_goal_refs(res.data)
        goal = postprocess_goal_v11(res.data, ep, l1=l1 if signal else None, objects=objects, segments=segments,
                                    repairs=repairs, signal=signal)
    else:
        repairs.append(f"goal {status_of(res)}: no goal record")
    # L5
    failed_calls = [{"step": step, "status": str(r.status), "error": getattr(r, "error", None)}
                    for step, r in log if r.status in BAD_STATUSES]
    failed_calls += [{"step": step, "status": "not run", "error": "an earlier call was stopped"}
                     for step in FULL_STEPS if step_status.get(step) == "not run" and state["stopped"]]
    coarse = coarse_subtasks(segments, objects)
    checks = run_checks_v11(segments, coarse, goal, l1=l1 if signal else None, objects=objects, facts=facts,
                            raw_refs=raw_refs, crawl_log=crawl_log, have_inventory=have_inventory,
                            have_facts=have_facts, failed_calls=failed_calls, no_output=no_output, signal=signal)
    cost = round(sum(float(getattr(c, "usd", 0.0) or 0.0) for c in calls), 8)
    wall = round(sum(float(getattr(c, "wall_s", 0.0) or 0.0) for c in calls), 3)
    valid = bool(calls) and all(bool(getattr(c, "valid", False)) for c in calls) and not failed_calls
    arm = str(ctx.get("arm", "v1.1"))
    names = {o["object_id"]: o["name"] for o in objects}
    view = build_view(arm=arm, episode=ep, objects=objects, segments=segments, coarse=coarse, goal=goal,
                      checks=checks, cost=cost, calls=calls, wall=wall, valid=valid, repairs=repairs,
                      no_output=no_output, names=names)
    for vs, s in zip(view["segments"], segments, strict=True):
        vs.update({"end_event": s.get("end_event"), "coarse_end_frame": s.get("coarse_end_frame"),
                   "crawl_calls": int(s.get("crawl_calls") or 0), "attempt_outcome": s.get("attempt_outcome")})
    coarse_fps = info.get("coarse_fps") if isinstance(info, dict) else None
    has_end_state = goal.get("has_end_state") if goal else None
    goal_command = goal.get("goal_command") if goal else None
    view.update({
        "attempts": attempt_records_v11(segments),
        "attempts_signal": signal_attempt_records(l1 or {}) if signal else None,
        "has_end_state": has_end_state, "goal_command": goal_command, "event_sources": [event_source],
        "coarse_mode": coarse_mode, "coarse_fps": coarse_fps, "crawl_enabled": crawl_enabled,
        "crawl_model": crawl_model, "crawl_calls": sum(int(e.get("calls", 0)) for e in crawl_log),
        "crawl_log": crawl_log, "events": events, "coarse_status": coarse_status, "step_status": dict(step_status),
        "failed_calls": failed_calls, "camera": camera, "inventory_frames": inventory_frames,
        "keyframes": keyframes, "keyframe_source": keyframe_source, "pipeline_version": PIPELINE_VERSION_V11,
        "pipeline_code": PIPELINE_CODE_V11, "prompt_version": PROMPT_VERSION, "prompt_hashes": prompt_hashes_v11(),
    })
    model = str(getattr(caller, "model", "unknown"))
    rows = episode_rows(arm=arm, episode=ep, provider=str(getattr(caller, "name", "unknown")), model=model,
                        pipeline_version=PIPELINE_VERSION_V11, objects=objects, facts=facts, segments=segments,
                        coarse=coarse, goal=goal, l1=l1 if signal else {}, checks=checks, cost_usd=cost,
                        source_model=model)
    # The attempt records of the view and of the rows are the same (SPEC_QUESTIONS Q189): the attempts of the
    # segments (attempt_source vlm), and with the gripper source also L1's own (the v7 rows, attempt_source
    # signal; the view's attempts_signal).
    base = {k: rows[0][k] for k in ("schema_version", "source", "episode_id", "task", "num_frames", "fps",
                                    "provider", "model", "strategy", "arm")}
    for a in view["attempts"]:
        rows.append({**base, "record_type": "attempt", "attempt_idx": a["attempt_idx"], "object_id": None,
                     "start_frame": a["start"], "end_frame": a["end"], "outcome": a["outcome"],
                     "failure_type": a["failure_type"], "evident_frame": a["evident_frame"],
                     "attempt_source": "vlm", "confidence": 0.5})
    rows = add_v11_fields(rows, segments, event_sources=[event_source], episode_fields={
        "has_end_state": has_end_state, "goal_command": goal_command, "coarse_mode": coarse_mode,
        "coarse_fps": coarse_fps, "crawl_enabled": crawl_enabled, "crawl_model": crawl_model})
    layers = {"L1": "signal" if signal else "none", "L2": model, "L3": model, "crawl": crawl_model or "none",
              "L4": model, "L5": "rules"}
    for r in rows:
        r["strategy"] = "v1.1"
        if r.get("record_type") == "episode_metadata":
            r["pipeline_code"] = PIPELINE_CODE_V11
            r["layer_models_json"] = json.dumps(layers, sort_keys=True, separators=(",", ":"))
    return {"view": view, "rows": rows, "calls": calls, "repairs": repairs, "segments_coarse": segments_coarse,
            "segments": segments, "crawl_log": crawl_log, "goal": goal, "objects": objects, "facts": facts,
            "checks": checks, "events": events, "coarse_info": info, "step_status": dict(step_status),
            "failed_calls": failed_calls, "stopped": state["stopped"], "cost_usd": cost,
            "episode_wall_s": round(time.perf_counter() - t0, 3)}
