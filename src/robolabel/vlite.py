"""V-lite: the redesign's vertical slice for one episode (V_LITE): L1 signal, L2 scene, L3 segment,
L4 goal, L5 checks, one model for every model step, schema v7 rows and the view record.

At most 4 calls per episode, in order: ``scene_inventory``, ``scene_facts``, ``segments``, ``goal``.
A call that is still invalid after its repair retry does not end the episode (V_LITE "When one call of
an episode fails"): an invalid inventory leaves targets unsure; invalid facts leave the goal call without
facts; invalid segments give the missing-output segment of spec 4.0 (the goal call still runs); an
invalid goal leaves the episode without a goal record. Every such case is recorded in ``repairs``.

The view record and the episode's ``episode_metadata`` row carry ``pipeline_code``, a hash of the code
that wrote them (the layers, this module, the v7 schema and the v7 prompts), next to
``pipeline_version``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from .layers.check import run_checks
from .layers.goal import goal_request, postprocess_goal, raw_goal_refs, requirement_text
from .layers.scene import facts_request, inventory_request, parse_facts, parse_inventory
from .layers.segment import coarse_subtasks, missing_output_segments, postprocess_segments, segments_request
from .prompts.v7 import VERSION as PROMPT_VERSION
from .providers.base import CallResult
from .schema_v7 import episode_rows

PIPELINE_VERSION = f"v-lite {PROMPT_VERSION}"
STEPS = ("scene_inventory", "scene_facts", "segments", "goal")


def pipeline_code(root: Path | None = None) -> str:
    """First 12 hex of SHA-256 over the code that writes views and rows: ``layers/*.py``, ``vlite.py``,
    ``schema_v7.py`` and ``prompts/v7/*`` (files only), in sorted path order, each as its path relative
    to the package and its bytes with line endings normalized to LF."""
    pkg = root or Path(__file__).resolve().parent
    files = [*pkg.glob("layers/*.py"), pkg / "vlite.py", pkg / "schema_v7.py",
             *(p for p in (pkg / "prompts" / "v7").glob("*") if p.is_file() and p.suffix != ".pyc")]
    h = hashlib.sha256()
    for rel, path in sorted((p.relative_to(pkg).as_posix(), p) for p in files):
        h.update(rel.encode("utf-8") + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


PIPELINE_CODE = pipeline_code()


def _names(objects: list[dict[str, Any]]) -> dict[str, str]:
    return {o["object_id"]: o["name"] for o in objects}


def _name(ref: Any, names: dict[str, str]) -> str:
    if ref in (None, ""):
        return "none"
    return names.get(ref, str(ref))


def run_episode(episode: Any, l1: dict[str, Any], caller: Any, *, arm: str, model_key: str, bucket: str = "sweep",
                reasoning: dict[str, Any] | None = None, image_tokens_per_image: float = 1500.0,
                start_mode: str = "json_schema_strict") -> dict[str, Any]:
    """Run V-lite on one episode with ``caller.call(CallRequest) -> CallResult``. Returns view, rows, calls."""
    t0 = time.perf_counter()
    context = {"arm": arm, "episode_key": episode.episode_id, "bucket": bucket, "model_key": model_key,
               "media_resolution": "long side 448 px, JPEG 85"}
    repairs: list[str] = []
    calls: list[CallResult] = []
    stopped = False

    def do(req) -> CallResult | None:
        nonlocal stopped
        if stopped:
            return None
        req.image_tokens_per_image = image_tokens_per_image
        req.start_mode = start_mode
        res = caller.call(req)
        calls.append(res)
        if res.status == "stopped":
            stopped = True
        return res

    # L2 inventory
    objects: list[dict[str, Any]] = []
    req, _ = inventory_request(episode, context, reasoning)
    res = do(req)
    have_inventory = bool(res and res.valid)
    if have_inventory:
        objects = parse_inventory(res.data, episode, repairs)
    else:
        repairs.append(f"scene_inventory {res.status if res else 'not run'}: empty inventory, targets unsure")
    # L2 facts
    facts: list[dict[str, Any]] = []
    req, manifest = facts_request(episode, l1, objects, context, reasoning)
    res = do(req)
    have_facts = bool(res and res.valid)
    if have_facts:
        facts = parse_facts(res.data, manifest, episode, objects, repairs)
        have_facts = bool(facts)
    else:
        repairs.append(f"scene_facts {res.status if res else 'not run'}: no scene facts")
    # L3 segments
    req, _ = segments_request(episode, l1, objects, context, reasoning)
    res = do(req)
    raw_refs: list[str] = []
    verdicts: list[dict[str, Any]] = []
    seg_ok = bool(res and res.valid)
    if seg_ok:
        for s in (res.data.get("segments", []) if isinstance(res.data, dict) else []):
            if isinstance(s, dict):
                raw_refs += [str(s.get("target", "")), str(s.get("destination", ""))]
        segments, verdicts = postprocess_segments(res.data, episode.num_frames, l1, objects, repairs)
    else:
        repairs.append(f"segments {res.status if res else 'not run'}: missing-output segment (spec 4.0)")
        segments = missing_output_segments(episode.num_frames)
    coarse = coarse_subtasks(segments, objects)
    # L4 goal
    goal = None
    req, _ = goal_request(episode, l1, objects, facts, context, reasoning)
    res = do(req)
    if res and res.valid and isinstance(res.data, dict):
        raw_refs += raw_goal_refs(res.data)
        goal = postprocess_goal(res.data, episode, l1, objects, segments, repairs)
    else:
        repairs.append(f"goal {res.status if res else 'not run'}: no goal record")
    # L5
    no_output = not seg_ok and goal is None
    checks = run_checks(segments, coarse, goal, l1, objects, facts, raw_refs, have_inventory=have_inventory,
                        have_facts=have_facts)
    cost = round(sum(c.usd for c in calls), 8)
    wall = round(sum(c.wall_s for c in calls), 3)
    valid = len(calls) == 4 and all(c.valid for c in calls)
    names = _names(objects)
    view = build_view(arm=arm, episode=episode, objects=objects, segments=segments, coarse=coarse, goal=goal,
                      checks=checks, cost=cost, calls=calls, wall=wall, valid=valid, repairs=repairs,
                      no_output=no_output, names=names)
    view["candidate_verdicts"] = verdicts
    view["step_status"] = {s: (c.status if c else "not run") for s, c in
                           zip(STEPS, calls + [None] * (4 - len(calls)), strict=True)}
    rows = episode_rows(arm=arm, episode=episode, provider=getattr(caller, "name", "unknown"),
                        model=getattr(caller, "model", "unknown"), pipeline_version=PIPELINE_VERSION, objects=objects,
                        facts=facts, segments=segments, coarse=coarse, goal=goal, l1=l1, checks=checks,
                        cost_usd=cost, source_model=getattr(caller, "model", "unknown"))
    for r in rows:
        if r.get("record_type") == "episode_metadata":
            r["pipeline_code"] = PIPELINE_CODE
    return {"view": view, "rows": rows, "calls": calls, "repairs": repairs, "stopped": stopped,
            "episode_wall_s": round(time.perf_counter() - t0, 3), "objects": objects, "facts": facts,
            "segments": segments, "goal": goal}


def build_view(*, arm: str, episode: Any, objects: list[dict[str, Any]], segments: list[dict[str, Any]],
               coarse: list[dict[str, Any]], goal: dict[str, Any] | None, checks: dict[str, Any], cost: float,
               calls: list[Any], wall: float, valid: bool, repairs: list[str], no_output: bool,
               names: dict[str, str]) -> dict[str, Any]:
    """The view record (V_LITE "Output"): the one format the review site, the import tool and metrics read."""
    fam = episode.extra.get("family") or episode.episode_id.split("/")[0]
    by_attempt: dict[int, list[dict[str, Any]]] = {}
    for s in segments:
        by_attempt.setdefault(int(s.get("attempt_idx") or 1), []).append(s)
    attempts = []
    for idx in sorted(by_attempt):
        ss = by_attempt[idx]
        failed = [s for s in ss if s.get("outcome") == "failed"]
        attempts.append({"start": min(s["start_frame"] for s in ss), "end": max(s["end_frame"] for s in ss),
                         "outcome": "failed" if failed else "success",
                         "failure_type": failed[0].get("failure_type", "other") if failed else "none", "source": "vlm"})
    view_goal = None
    if goal is not None:
        view_goal = {"objective": goal["objective_text"], "primary_target_name": _name(goal["primary_target"], names),
                     "primary_destination_name": _name(goal["primary_destination"], names),
                     "requirements": [{"text": requirement_text(r, names), "kind": r["kind"],
                                       "predicate": r["predicate"], "object_name": _name(r["object"], names),
                                       "ref_name": _name(r["ref_object"], names), "value": r["value"],
                                       "status": r["status"], "unsure_kind": r.get("unsure_kind") or "none",
                                       "basis": r["basis"], "achieved": r["achieved"],
                                       "added_by": r.get("added_by", "model")} for r in goal["requirements"]]}
    return {
        "arm": arm, "episode_key": episode.episode_id, "family": fam, "fps": float(episode.fps),
        "num_frames": int(episode.num_frames), "cameras": list(episode.extra.get("camera_order", [])),
        "camera_sizes": episode.extra.get("camera_sizes", {}), "task": episode.task,
        "objects": [{"object_id": o["object_id"], "name": o["name"], "category": o["category"],
                     "points": [{"camera": v["camera"], "x": v["x"], "y": v["y"]} for v in o["views"] if v["visible"]],
                     "boxes": [{"camera": v["camera"], "x0": v["box"][0], "y0": v["box"][1], "x1": v["box"][2],
                                "y1": v["box"][3]} for v in o["views"] if v["visible"]]} for o in objects],
        "segments": [{"start": s["start_frame"], "end": s["end_frame"], "phase_class": s["phase_class"],
                      "phase_text": s["phase_text"], "target_name": _name(s["target"], names),
                      "destination_name": _name(s["destination"], names), "attempt_idx": s["attempt_idx"],
                      "outcome": s["outcome"], "failure_type": s["failure_type"], "mistake": bool(s["mistake"]),
                      "boundary_source": s["boundary_source"]} for s in segments],
        "coarse": [{"start": c["start_frame"], "end": c["end_frame"], "text": c["text"], "mistake": bool(c["mistake"])}
                   for c in coarse],
        "attempts": attempts,
        "goal": view_goal,
        "episode_outcome": goal["episode_outcome"] if goal else "unknown",
        "checks": checks["checks"], "risk": checks["risk"], "routed": checks["routed"],
        "route_reasons": checks.get("route_reasons", []),
        "cost_usd": cost, "calls": len(calls), "wall_s": wall, "valid": valid, "repairs": list(repairs),
        "no_output": no_output, "cache_hits": sum(1 for c in calls if getattr(c, "cache_hit", False)),
        "pipeline_version": PIPELINE_VERSION, "pipeline_code": PIPELINE_CODE,
    }
