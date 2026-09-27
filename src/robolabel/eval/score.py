"""Score view records against gold v2 episodes (MEASUREMENT_SPEC 4.0 to 4.3), and aggregate.

:func:`score_view` scores one view record (V_LITE "Output") against one gold v2 episode and returns
per-episode counts. Every metric is a ratio of sums, ``{"numerator", "denominator", "pending"}``, so
:func:`aggregate` (micro within the given episodes) and ``robolabel.eval.stats`` (cluster bootstrap)
recompute every metric from counts. Metric keys:

- ``T1-P@tau``, ``T1-R@tau``, ``T1-F1@tau`` for tau 3, 5, 10 (F1 = 2M / (m + n)); ``T2-MAE`` (frames,
  pairs at tau 10); ``T3`` (per-episode score, denominator 1, so the micro value is the mean);
  ``T4`` (degenerate); ``T5`` (#pred - #gold), ``T5-abs``, ``T5-ge2``; ``T6-*@tau`` only when the
  gold has coarse subtasks.
- ``S1``, ``S1-seg``, ``S1-unmapped``, ``S2``, ``S2-dest``, ``S4-P``, ``S4-R``, ``S4-F1``,
  ``S4-ep-P``, ``S4-ep-R``, ``S4-type``.
- ``G1a``, ``G1b``, ``G1c``, ``G2``, ``G2-i``, ``G2-ii``, ``G3`` (wrong-target rate), ``G4``,
  ``G4-h`` (mean count), ``G4-h-any``, ``G5-P``, ``G5-R``, ``G5-P-<kind>``, ``G5-R-<kind>``,
  ``G5-dropped``, ``G6``, ``G7`` (achieved), ``G7-outcome``.

A metric that depends on an unanswered judge item (object resolution rule 3, G2 part ii) has
``pending`` > 0 and reads ``"pending"`` in ``values`` and in :func:`aggregate` (spec 10.2). A view
with ``no_output`` true or without usable segments is scored by the missing-output rule of spec 4.0
(one segment over the episode, no targets, no goal, no failed attempts, no outcome). A view whose
goal is null is scored as having no goal (G1 fails, G3 wrong, G6 false); the harness does not
compile a goal here. :func:`score_legacy_boundaries` gives the continuity numbers against legacy
gold segments. Pure functions; floats are rounded to 6 decimals.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .goal import G5_CLASSES, g2_episode, g6_episode, goal_episode, outcome_episode
from .heldout import HeldoutRefused
from .semantic import (
    PENDING,
    ObjectResolver,
    counts,
    s1_episode,
    s2_episode,
    s4_episode,
)
from .temporal import (
    as_span,
    boundaries,
    missing_output_segments,
    seg_end,
    seg_start,
    t1_episode,
    t3_episode,
    t4_episode,
    t5_episode,
    t6_episode,
)

SCHEMA = "robolabel/episode-scores/v1"
DECIMALS = 6
TAUS = (3, 5, 10)
T2_TAU = 10
CONTINUITY_LABEL = "continuity only"
HARD_TAGS = ("distractor", "failed_attempt", "hidden_ending", "long_multistep")
ORDINARY = "ordinary"


def _r6(value: float | None) -> float | None:
    return None if value is None else round(float(value), DECIMALS)


def _family_of(key: str | None) -> str | None:
    return None if not key or "/" not in str(key) else str(key).split("/", 1)[0]


def _episode_sort_key(ep: Mapping[str, Any]) -> tuple[str, int, str]:
    key = str(ep.get("episode_key") or "")
    family, _, index = key.partition("/")
    return (str(ep.get("family") or family), int(index) if index.isdigit() else -1, key)


def _value(c: Mapping[str, Any]) -> Any:
    if c.get("pending"):
        return PENDING
    den = c.get("denominator") or 0
    return _r6(c["numerator"] / den) if den else None


def _sorted_segments(segments: Iterable[Any]) -> list[Mapping[str, Any]]:
    usable = [(k, s) for k, s in enumerate(segments)
              if isinstance(s, Mapping) and seg_start(s) is not None and seg_end(s) is not None]
    usable.sort(key=lambda ks: (seg_start(ks[1]), seg_end(ks[1]), ks[0]))
    return [s for _, s in usable]


def predicted_segments(view: Mapping[str, Any], num_frames: int) -> tuple[list[Mapping[str, Any]], bool, str | None]:
    """(segments sorted by start, missing output, reason). Missing output gives the spec 4.0 segment."""
    no_output = view.get("no_output")
    if no_output is True or (isinstance(no_output, str) and no_output.strip().lower() == "true"):
        return missing_output_segments(num_frames), True, "no_output"
    segs = view.get("segments")
    if not isinstance(segs, list) or not segs:
        return missing_output_segments(num_frames), True, "no segments"
    ordered = _sorted_segments(segs)
    if not ordered:
        return missing_output_segments(num_frames), True, "no usable segments"
    return ordered, False, None


# --------------------------------------------------------------------------- #
# Judge answers
# --------------------------------------------------------------------------- #
def _answers_by_id(*sources: Any) -> dict[str, Any]:
    """Judge answers as {judge_id: answer} from mappings or from judge_answers.jsonl records."""
    out: dict[str, Any] = {}
    for src in sources:
        if isinstance(src, Mapping):
            for key in sorted(src, key=str):
                out[str(key)] = src[key]
        elif isinstance(src, Sequence) and not isinstance(src, str):
            for rec in src:
                if isinstance(rec, Mapping) and rec.get("judge_id") is not None:
                    out[str(rec["judge_id"])] = rec.get("answer")
    return out


def _object_answers(answers: Mapping[str, Any], episode_key: str) -> dict[str, Any]:
    """{normalized name: answer} for this episode's object-resolution judge items.

    Keys ``object|<episode_key>|<name>`` are taken for this episode; keys without ``|`` are taken
    as names (a mapping made for this one episode).
    """
    prefix = f"object|{episode_key}|"
    out: dict[str, Any] = {}
    for key, value in answers.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = value
        elif "|" not in key:
            out.setdefault(key, value)
    return out


# --------------------------------------------------------------------------- #
# score_view
# --------------------------------------------------------------------------- #
def _temporal_counts(pred: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]], episode_key: str,
                     out: dict[str, Any], details: dict[str, Any]) -> None:
    pred_b, gold_b = boundaries(pred), boundaries(gold)
    details["pred_boundaries"] = pred_b
    details["gold_boundaries"] = gold_b
    details["T1"] = {}
    for tau in TAUS:
        t1 = t1_episode(pred_b, gold_b, tau, episode_key)
        out[f"T1-P@{tau}"] = counts(t1["matched"], t1["n_pred"])
        out[f"T1-R@{tau}"] = counts(t1["matched"], t1["n_gold"])
        out[f"T1-F1@{tau}"] = counts(2 * t1["matched"], t1["n_pred"] + t1["n_gold"])
        details["T1"][str(tau)] = {k: t1[k] for k in ("matched", "n_pred", "n_gold", "greedy_matched",
                                                       "greedy_disagrees", "pairs", "abs_errors")}
        if tau == T2_TAU:
            out["T2-MAE"] = counts(sum(t1["abs_errors"]), len(t1["abs_errors"]))
    t3 = t3_episode(pred, gold)
    t4 = t4_episode(pred, len(gold))
    t5 = t5_episode(len(pred), len(gold))
    out["T3"] = counts(t3, 1)
    out["T4"] = counts(int(t4["degenerate"]), 1)
    out["T5"] = counts(t5, 1)
    out["T5-abs"] = counts(abs(t5), 1)
    out["T5-ge2"] = counts(int(abs(t5) >= 2), 1)
    details["T4"] = t4


def score_view(view: Mapping[str, Any], gold_episode: Mapping[str, Any], judge_answers: Any = None,
               gold_objects_resolution_judge: Any = None, *, allow_heldout: bool = False) -> dict[str, Any]:
    """Score one view record against one gold v2 episode.

    ``judge_answers`` (or ``gold_objects_resolution_judge``, the same thing under the name the
    import tool uses) maps judge IDs (``object|<episode_key>|<name>``, ``g2|<episode_key>|<text>``)
    to answers, or is a list of ``{"judge_id", "answer"}`` records; names alone are accepted for a
    mapping made for this episode. A gold episode with ``split: heldout`` is refused unless
    ``allow_heldout`` (the caller has passed the held-out guard).

    Returns ``{"schema", "arm", "episode_key", "family", "hard_tags", "no_output",
    "missing_output_reason", "goal_missing", "counts", "values", "pending", "details",
    "judge_queue"}``.
    """
    if not isinstance(view, Mapping) or not isinstance(gold_episode, Mapping):
        raise TypeError("score_view takes a view record and a gold v2 episode, both dicts")
    episode_key = str(gold_episode.get("episode_key") or view.get("episode_key") or "")
    if gold_episode.get("split") == "heldout" and not allow_heldout:
        raise HeldoutRefused(f"{episode_key} is a held-out episode; scoring it needs the held-out guard")
    if view.get("episode_key") is not None and str(view["episode_key"]) != episode_key:
        raise ValueError(f"view is for {view['episode_key']!r} but the gold episode is {episode_key!r}")
    num_frames = int(gold_episode["num_frames"])
    family = _family_of(episode_key) or view.get("family")

    answers = _answers_by_id(judge_answers, gold_objects_resolution_judge)
    pred_segments, no_output, reason = predicted_segments(view, num_frames)
    goal_raw = None if no_output else view.get("goal")
    view_goal = goal_raw if isinstance(goal_raw, Mapping) and goal_raw else None
    coarse = [] if no_output else [c for c in (view.get("coarse") or []) if isinstance(c, Mapping)]
    pred_outcome = None if no_output else view.get("episode_outcome")
    view_objects = [] if no_output else (view.get("objects") or [])

    gold_objects = [o for o in gold_episode.get("objects") or [] if isinstance(o, Mapping)]
    gold_ids = [str(o["object_id"]) for o in gold_objects if o.get("object_id")]
    gold_segments = _sorted_segments(gold_episode.get("segments") or [])
    gold_failed = [fa for fa in gold_episode.get("failed_attempts") or [] if isinstance(fa, Mapping)]
    resolver = ObjectResolver(gold_objects, view_objects, view.get("camera_sizes"),
                              _object_answers(answers, episode_key), episode_key)

    out: dict[str, Any] = {}
    details: dict[str, Any] = {}
    _temporal_counts(pred_segments, gold_segments, episode_key, out, details)
    gold_coarse = _sorted_segments(gold_episode.get("coarse_subtasks") or [])
    if gold_coarse:
        pred_coarse = _sorted_segments(coarse) or [{"start_frame": 0, "end_frame": num_frames - 1}]
        for tau in TAUS:
            t6 = t6_episode(pred_coarse, gold_coarse, tau, episode_key)
            out[f"T6-P@{tau}"] = counts(t6["matched"], t6["n_pred"])
            out[f"T6-R@{tau}"] = counts(t6["matched"], t6["n_gold"])
            out[f"T6-F1@{tau}"] = counts(2 * t6["matched"], t6["n_pred"] + t6["n_gold"])

    s1 = s1_episode(pred_segments, gold_segments, num_frames)
    for key in ("S1", "S1-seg", "S1-unmapped"):
        out[key] = s1[key]
    details["S1"] = {k: s1[k] for k in ("frames", "uncovered_frames", "gold_unlabeled_frames", "seg_pairs")}
    s2 = s2_episode(pred_segments, gold_segments, resolver, gold_ids)
    s2d = s2_episode(pred_segments, gold_segments, resolver, gold_ids, destination=True)
    out["S2"], out["S2-dest"] = s2["counts"], s2d["counts"]
    details["S2"], details["S2-dest"] = s2["items"], s2d["items"]
    s4 = s4_episode(pred_segments, gold_failed)
    for key in ("S4-P", "S4-R", "S4-F1", "S4-ep-P", "S4-ep-R", "S4-type"):
        out[key] = s4[key]
    details["S4"] = {k: s4[k] for k in ("pred_spans", "gold_spans", "pairs")}

    goal = goal_episode(view_goal, gold_episode, resolver)
    scores = goal["scores"]
    out.update(scores["counts"])
    out["G3"] = goal["g3"]["counts"]
    g2 = g2_episode(pred_segments, coarse, gold_failed, goal["raw_requirements"], episode_key=episode_key,
                    judge_answers=answers)
    out.update(g2["counts"])
    g6 = g6_episode({"G1_all_ending_stated": scores["g1_ok"], "G2_not_copied": g2["ok"],
                     "G3_target_right": goal["g3"]["ok"], "G4_none_incidental_or_hallucinated": scores["g4_ok"]})
    out["G6"] = g6["counts"]
    out["G7-outcome"] = outcome_episode(pred_outcome, gold_episode.get("episode_outcome"))
    details["goal"] = {
        "pairs": scores["pairs"],
        "pred": goal["pred"],
        "gold_ending": scores["gold_ending"],
        "gold_ending_stated": scores["gold_ending_stated"],
        "incidental_as_required": scores["incidental_as_required"],
        "hallucinated_required": scores["hallucinated_required"],
        "gold_unsure_dropped": scores["gold_unsure_dropped"],
        "g5_confusion": scores["g5_confusion"],
        "g5_pred_unsure_without_kind": scores["g5_pred_unsure_without_kind"],
        "G2": {"applicable": g2["applicable"], "part_i_hits": g2["part_i_hits"]},
        "G3": {"reason": goal["g3"]["reason"], "resolution": goal["g3"]["resolution"]},
        "G6": g6["conditions"],
        "pred_outcome": pred_outcome,
        "gold_outcome": gold_episode.get("episode_outcome"),
    }

    queue = {item["judge_id"]: item for item in [*resolver.judge_queue, *g2["judge_items"]]}
    ordered = dict(sorted(out.items()))
    return {
        "schema": SCHEMA,
        "arm": view.get("arm"),
        "episode_key": episode_key,
        "family": family,
        "hard_tags": sorted({str(t) for t in gold_episode.get("hard_tags") or []}),
        "no_output": no_output,
        "missing_output_reason": reason,
        "goal_missing": view_goal is None,
        "counts": ordered,
        "values": {k: _value(c) for k, c in ordered.items()},
        "pending": [k for k, c in ordered.items() if c["pending"]],
        "details": details,
        "judge_queue": [queue[k] for k in sorted(queue)],
    }


# --------------------------------------------------------------------------- #
# Continuity against legacy gold
# --------------------------------------------------------------------------- #
def _legacy_gold(legacy_segments: Iterable[Any]) -> list[dict[str, int]]:
    spans = [as_span(s) for s in legacy_segments]
    return sorted(({"start_frame": s[0], "end_frame": s[1]} for s in spans if s is not None),
                  key=lambda s: (s["start_frame"], s["end_frame"]))


def score_legacy_boundaries(view: Mapping[str, Any], legacy_segments: Iterable[Any],
                            num_frames: int | None = None) -> dict[str, Any]:
    """T1 (tau 3, 5, 10), T3 and T5 against legacy gold segments, labeled "continuity only".

    ``legacy_segments`` are dicts with ``start_frame``/``end_frame`` (or ``start``/``end``) or
    ``[start, end]`` pairs. The legacy gold's granularity came from the S0 draft (spec 3.1), so a
    correct extra boundary counts against a system here; never use this for claims.
    """
    gold = _legacy_gold(legacy_segments)
    if not gold:
        raise ValueError("score_legacy_boundaries needs at least one legacy gold segment")
    n = int(num_frames or view.get("num_frames") or gold[-1]["end_frame"] + 1)
    pred, no_output, reason = predicted_segments(view, n)
    episode_key = str(view.get("episode_key") or "")
    out: dict[str, Any] = {}
    t1: dict[str, Any] = {}
    pred_b, gold_b = boundaries(pred), boundaries(gold)
    for tau in TAUS:
        r = t1_episode(pred_b, gold_b, tau, episode_key)
        out[f"T1-P@{tau}"] = counts(r["matched"], r["n_pred"])
        out[f"T1-R@{tau}"] = counts(r["matched"], r["n_gold"])
        out[f"T1-F1@{tau}"] = counts(2 * r["matched"], r["n_pred"] + r["n_gold"])
        t1[str(tau)] = {k: r[k] for k in ("matched", "n_pred", "n_gold", "precision", "recall", "f1", "pairs")}
    diff = t5_episode(len(pred), len(gold))
    out["T3"] = counts(t3_episode(pred, gold), 1)
    out["T5"] = counts(diff, 1)
    ordered = dict(sorted(out.items()))
    return {
        "label": CONTINUITY_LABEL,
        "arm": view.get("arm"),
        "episode_key": episode_key,
        "no_output": no_output,
        "missing_output_reason": reason,
        "pred_boundaries": pred_b,
        "gold_boundaries": gold_b,
        "T1": t1,
        "T3": out["T3"]["numerator"],
        "T5": diff,
        "counts": ordered,
        "values": {k: _value(c) for k, c in ordered.items()},
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def split_metric_key(key: str) -> tuple[str, int | None]:
    """``"T1-F1@5"`` to ``("T1-F1", 5)``; keys without a tau give None."""
    base, _, tau = key.partition("@")
    return (base, int(tau)) if tau.isdigit() else (key, None)


def _sum(values: list[Any]) -> Any:
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return sum(values)
    return _r6(math.fsum(float(v) for v in values))


def micro(per_episode: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Micro metrics over episodes: sum numerators and denominators per metric key.

    ``value`` is numerator / denominator (6 decimals), None when the denominator is 0, and
    ``"pending"`` when any episode's count waits for the judge (``pending_items`` > 0).
    """
    keys = sorted({k for e in per_episode for k in (e.get("counts") or {})})
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        rows = [e["counts"][key] for e in per_episode if key in (e.get("counts") or {})]
        num = _sum([r["numerator"] for r in rows])
        den = sum(int(r["denominator"]) for r in rows)
        pending = sum(int(r.get("pending") or 0) for r in rows)
        base, tau = split_metric_key(key)
        if pending:
            value: Any = PENDING
            status = PENDING
        elif den:
            value, status = _r6(num / den), "ok"
        else:
            value, status = None, "no_denominator"
        out[key] = {"metric_id": base, "tau": tau, "value": value, "numerator": num, "denominator": den,
                    "n_episodes": len(rows), "pending_items": pending, "status": status}
    return out


def _macro(by_family: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for fam in by_family.values() for k in fam["metrics"]})
    out: dict[str, Any] = {}
    for key in keys:
        vals = [by_family[f]["metrics"][key]["value"] for f in sorted(by_family) if key in by_family[f]["metrics"]]
        if any(v == PENDING for v in vals):
            out[key] = PENDING
            continue
        nums = [float(v) for v in vals if v is not None]
        out[key] = _r6(math.fsum(nums) / len(nums)) if nums else None
    return out


def _in_subset(ep: Mapping[str, Any], subset: str | None) -> bool:
    tags = set(ep.get("hard_tags") or []) & set(HARD_TAGS)
    if subset in (None, "all"):
        return True
    if subset == ORDINARY:
        return not tags
    return subset in tags


def _confusion(per_episode: Sequence[Mapping[str, Any]]) -> Any:
    total = {g: {p: 0 for p in G5_CLASSES} for g in G5_CLASSES}
    for ep in per_episode:
        if (ep.get("counts") or {}).get("G5-P", {}).get("pending"):
            return PENDING
        conf = ((ep.get("details") or {}).get("goal") or {}).get("g5_confusion") or {}
        for g in G5_CLASSES:
            for p in G5_CLASSES:
                total[g][p] += int((conf.get(g) or {}).get(p) or 0)
    return total


def _block(eps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[Mapping[str, Any]]] = {}
    for ep in eps:
        families.setdefault(str(ep.get("family") or _family_of(ep.get("episode_key")) or ""), []).append(ep)
    by_family = {f: {"n_episodes": len(families[f]), "metrics": micro(families[f])} for f in sorted(families)}
    return {
        "n_episodes": len(eps),
        "n_no_output": sum(1 for e in eps if e.get("no_output")),
        "n_goal_missing": sum(1 for e in eps if e.get("goal_missing")),
        "families": {f: len(families[f]) for f in sorted(families)},
        "metrics": micro(eps),
        "by_family": by_family,
        "macro": _macro(by_family),
        "g5_confusion": _confusion(eps),
    }


def aggregate(per_episode: Sequence[Mapping[str, Any]], subset: str | None = None) -> dict[str, Any]:
    """Micro metrics of one arm over ``score_view`` results, with n.

    ``metrics`` is micro over all given episodes; ``by_family`` is micro within each family and
    ``macro`` averages the family values (the spec 4.0 headline across families). ``subset`` keeps
    ``ordinary`` episodes (no hard tag) or one hard tag; without it, ``by_subset`` repeats the
    block for ``ordinary`` and each hard tag present, so the pooled number never stands alone
    (spec 6.6). ``judge_queue`` merges the episodes' judge items; ``pending_metrics`` lists the
    keys that read ``pending``.
    """
    eps = [e for e in per_episode if isinstance(e, Mapping)]
    arms = sorted({str(e.get("arm")) for e in eps})
    if len(arms) > 1:
        raise ValueError(f"aggregate takes one arm at a time, got {arms}")
    keys = [str(e.get("episode_key")) for e in eps]
    if len(set(keys)) != len(keys):
        raise ValueError("aggregate got the same episode twice")
    eps = sorted((e for e in eps if _in_subset(e, subset)), key=_episode_sort_key)
    out: dict[str, Any] = {"arm": arms[0] if arms else None, "subset": subset or "all", **_block(eps)}
    out["pending_metrics"] = sorted(k for k, m in out["metrics"].items() if m["status"] == PENDING)
    queue = {item["judge_id"]: item for e in eps for item in e.get("judge_queue") or []}
    out["judge_queue"] = [queue[k] for k in sorted(queue)]
    if subset in (None, "all"):
        present = sorted({t for e in eps for t in e.get("hard_tags") or [] if t in HARD_TAGS})
        out["by_subset"] = {}
        for name in [ORDINARY, *present]:
            sub = [e for e in eps if _in_subset(e, name)]
            if sub:
                block = _block(sub)
                out["by_subset"][name] = {k: block[k] for k in ("n_episodes", "families", "metrics", "macro")}
    return out


def to_records(agg: Mapping[str, Any], system: str) -> list[dict[str, Any]]:
    """Spec 10.2 ``metrics.json`` records from :func:`aggregate`: one per family and metric, plus
    the macro pool (family ``macro``, no numerator or denominator). ``interval`` is left None; the
    bootstrap in ``robolabel.eval.stats`` fills it."""
    subset = agg.get("subset") or "all"
    records: list[dict[str, Any]] = []
    for family in sorted(agg.get("by_family") or {}):
        for _key, m in sorted(agg["by_family"][family]["metrics"].items()):
            records.append({"system": system, "family": family, "subset": subset, "metric_id": m["metric_id"],
                            "tau": m["tau"], "value": m["value"], "numerator": m["numerator"],
                            "denominator": m["denominator"], "n_episodes": m["n_episodes"], "interval": None})
    for key, value in sorted((agg.get("macro") or {}).items()):
        base, tau = split_metric_key(key)
        records.append({"system": system, "family": "macro", "subset": subset, "metric_id": base, "tau": tau,
                        "value": value, "numerator": None, "denominator": None,
                        "n_episodes": agg.get("n_episodes"), "interval": None})
    return records
