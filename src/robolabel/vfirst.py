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
"""

from __future__ import annotations

import copy
import time
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
