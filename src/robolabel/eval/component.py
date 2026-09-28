"""Component metrics X1 to X3 (MEASUREMENT_SPEC 4.8) for the signal layer's intermediate outputs.

- X1, signal candidate recall: T1 recall at tau = 5 of the candidate boundary frames against the
  gold boundaries whose transition is approach -> grasp, transport -> release or
  release -> retract (judged from the phase class of the gold segments on each side). Precision
  against all gold boundaries is reported too, labeled not constrained.
- X2, robot end-state accuracy: signal end-state items against gold ``robot_end_state``
  requirements (holding, gripper_open, gripper_closed, withdrawn, at_home_pose, near_object) whose
  gold visibility has at least one camera ``visible``. A signal gripper_open item answers a gold
  gripper_closed item negated, and the reverse, since L1 states the gripper with one of the two.
- X3, failed-grasp detection from signals: S4-style span F1 at IoU >= 0.3, restricted to gold
  failed attempts of type missed_grasp, slip or drop, using only signal-layer spans.

Per-episode functions return counts; the ``*_summary`` functions micro-average them and return
``numerator``, ``denominator`` and ``value`` (0 / 0 gives 0.0 and ``zero_denominator: true``, as
in ``temporal.t1_micro``). ``x1_early_read`` scores legacy gold, which has no phase classes: the
caller passes ``phase_of(segment)`` that maps the legacy text with the phase lexicon. Legacy gold
has no requirements and no failed attempts, so X2 and X3 have no early read. Pure functions.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .temporal import as_span, match_boundaries, match_spans, seg_end

logger = logging.getLogger(__name__)

DECIMALS = 6
X1_TAU = 5
X1_TRANSITIONS = frozenset({("approach", "grasp"), ("transport", "release"), ("release", "retract")})
X2_PREDICATES = ("holding", "gripper_open", "gripper_closed", "withdrawn", "at_home_pose", "near_object")
# L1 states the gripper with one of these two items (value true), so each answers the other negated.
GRIPPER_COMPLEMENT = {"gripper_open": "gripper_closed", "gripper_closed": "gripper_open"}
X3_FAILURE_TYPES = frozenset({"missed_grasp", "slip", "drop"})
X3_MIN_IOU = 0.3
# Attempt outcomes that mark a signal-layer attempt as failed (V_LITE L1 names plus the view names).
SIGNAL_FAILED_OUTCOMES = frozenset({"empty", "slip", "drop", "failed", "missed_grasp"})
EARLY_READ_LABEL = "early read on legacy gold, not acceptance"
PRECISION_LABEL = "not constrained"

PhaseOf = Callable[[Mapping[str, Any]], Any]


def _r6(value: float | None) -> float | None:
    return None if value is None else round(float(value), DECIMALS)


def _norm(value: Any) -> str | None:
    """Lowercase stripped string; None, NaN, "", "none" and "null" become None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    return None if text in ("", "none", "null") else text


def _bool_like(value: Any) -> Any:
    """True and False from bools or their strings; anything else unchanged (string states)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return value


def _ratio(num: int, den: int) -> dict[str, Any]:
    return {
        "numerator": num,
        "denominator": den,
        "value": _r6(num / den) if den else 0.0,
        "zero_denominator": den == 0,
    }


def _phase_class(segment: Mapping[str, Any]) -> Any:
    return segment.get("phase_class")


def _frame(candidate: Any) -> int | None:
    if isinstance(candidate, Mapping):
        candidate = candidate.get("frame")
    if candidate is None or (isinstance(candidate, float) and math.isnan(candidate)):
        return None
    return int(candidate)


# --------------------------------------------------------------------------- #
# X1: signal candidate recall
# --------------------------------------------------------------------------- #
def x1_gold_boundaries(gold_segments: Sequence[Mapping[str, Any]],
                       phase_of: PhaseOf | None = None) -> tuple[list[int], list[int]]:
    """(all gold boundaries, the X1-constrained ones), each in segment order."""
    phase = phase_of or _phase_class
    segs = list(gold_segments)
    all_bounds: list[int] = []
    constrained: list[int] = []
    for k in range(len(segs) - 1):
        end = seg_end(segs[k])
        if end is None:
            continue
        all_bounds.append(end)
        if (_norm(phase(segs[k])), _norm(phase(segs[k + 1]))) in X1_TRANSITIONS:
            constrained.append(end)
    return all_bounds, constrained


def x1_episode(candidate_frames: Iterable[Any], gold_segments: Sequence[Mapping[str, Any]],
               tau: int = X1_TAU, phase_of: PhaseOf | None = None) -> dict[str, Any]:
    """X1 counts for one episode.

    ``candidate_frames`` are ints or dicts with ``frame`` (the L1 candidates); they are sorted
    before matching. Recall matches them (T1 optimal matching) against the constrained gold
    boundaries only; precision matches them against all gold boundaries.
    """
    cands = sorted(f for f in (_frame(c) for c in candidate_frames) if f is not None)
    all_bounds, constrained = x1_gold_boundaries(gold_segments, phase_of)
    return {
        "tau": tau,
        "n_candidates": len(cands),
        "n_gold_constrained": len(constrained),
        "n_gold_all": len(all_bounds),
        "recall_matched": len(match_boundaries(cands, constrained, tau)),
        "precision_matched": len(match_boundaries(cands, all_bounds, tau)),
    }


def x1_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Micro X1: recall = matched / constrained gold boundaries; precision (not constrained)."""
    episodes = list(episodes)  # read several times below; a generator would give zeros
    recall = _ratio(sum(int(e["recall_matched"]) for e in episodes),
                    sum(int(e["n_gold_constrained"]) for e in episodes))
    precision = _ratio(sum(int(e["precision_matched"]) for e in episodes),
                       sum(int(e["n_candidates"]) for e in episodes))
    taus = sorted({int(e["tau"]) for e in episodes if e.get("tau") is not None})
    if len(taus) > 1:
        raise ValueError(f"x1_summary got episodes with different tau values: {taus}")
    return {
        "metric_id": "X1",
        "tau": taus[0] if taus else None,
        "n_episodes": len(episodes),
        **recall,
        "precision": {**precision, "label": PRECISION_LABEL},
    }


def x1_early_read(episodes: Mapping[str, tuple[Iterable[Any], Sequence[Mapping[str, Any]]]],
                  phase_of: PhaseOf, tau: int = X1_TAU) -> dict[str, Any]:
    """X1 on legacy gold: ``episodes`` maps episode key to (candidate frames, legacy gold segments).

    Legacy segments carry text, not phase classes; ``phase_of`` maps a segment to its class (for
    example the phase lexicon applied to ``subtask_text``). The result is labeled as an early read.
    """
    per_episode = {
        key: x1_episode(cands, segs, tau, phase_of) for key, (cands, segs) in sorted(episodes.items())
    }
    return {"label": EARLY_READ_LABEL, **x1_summary(list(per_episode.values())), "per_episode": per_episode}


# --------------------------------------------------------------------------- #
# X2: robot end-state accuracy
# --------------------------------------------------------------------------- #
def _has_visible_camera(visibility: Any) -> bool:
    """True when some camera is ``visible`` (dict camera -> class, or list of {camera, class})."""
    if isinstance(visibility, Mapping):
        return any(_norm(v) == "visible" for v in visibility.values())
    if isinstance(visibility, Sequence) and not isinstance(visibility, str):
        return any(isinstance(v, Mapping) and _norm(v.get("class")) == "visible" for v in visibility)
    return False


def _ref_object(item: Mapping[str, Any]) -> str | None:
    value = item.get("ref_object") if "ref_object" in item else item.get("ref_object_id")
    return _norm(value)


def gold_end_fact(requirement: Mapping[str, Any], use_achieved: bool = True) -> tuple[bool, Any]:
    """(known, value) of the robot's actual end state from a gold requirement.

    A null ``value`` is unknown. With ``use_achieved``: ``achieved`` false flips a boolean
    ``value`` (the required state did not hold); ``achieved`` "unknown" makes the fact unknown;
    anything else keeps ``value``. ``holding`` of a named object that was not achieved is unknown
    too: the robot may hold nothing or another object, which the canonical forms of spec 3.2
    (``holding``, null, true or false) cannot tell apart.
    """
    value = _bool_like(requirement.get("value"))
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False, None
    if not use_achieved:
        return True, value
    achieved = _bool_like(requirement.get("achieved"))
    if achieved is False:
        if _norm(requirement.get("predicate")) == "holding" and _ref_object(requirement) is not None:
            return False, None
        return (True, not value) if isinstance(value, bool) else (False, None)
    if isinstance(achieved, str) and _norm(achieved) == "unknown":
        return False, None
    return True, value


def _signal_fact(signal_by_predicate: Mapping[str, Mapping[str, Any]],
                 predicate: str) -> tuple[str | None, str | None, Any]:
    """(signal predicate used, ref_object, value) answering ``predicate``; all None when absent.

    The first signal item of the same predicate is used. For ``gripper_open`` and
    ``gripper_closed`` a signal item of the other one answers with its value negated, because L1
    states the gripper with exactly one of the two.
    """
    item = signal_by_predicate.get(predicate)
    if item is not None:
        return predicate, _ref_object(item), _bool_like(item.get("value"))
    other = GRIPPER_COMPLEMENT.get(predicate)
    item = signal_by_predicate.get(other) if other is not None else None
    if item is not None:
        value = _bool_like(item.get("value"))
        return other, _ref_object(item), (not value) if isinstance(value, bool) else None
    return None, None, None


def x2_episode(signal_items: Iterable[Mapping[str, Any]], gold_requirements: Iterable[Mapping[str, Any]],
               use_achieved: bool = True) -> dict[str, Any]:
    """X2 counts for one episode.

    Eligible gold items: kind ``robot_end_state``, predicate in ``X2_PREDICATES``, at least one
    camera ``visible``, and a known end-state fact (``gold_end_fact``). Each is compared with the
    signal's answer for that predicate (``_signal_fact``): correct when the ``ref_object`` (null
    and "none" are the same) and the value are equal. A gold item the signal does not answer is
    wrong and counted in ``missing_signal_item``.
    """
    signal_by_predicate: dict[str, Mapping[str, Any]] = {}
    for item in signal_items:
        predicate = _norm(item.get("predicate"))
        if predicate is not None:
            signal_by_predicate.setdefault(predicate, item)
    correct = total = missing = not_visible = unknown = 0
    items: list[dict[str, Any]] = []
    for req in gold_requirements:
        predicate = _norm(req.get("predicate"))
        if _norm(req.get("kind")) != "robot_end_state" or predicate not in X2_PREDICATES:
            continue
        if not _has_visible_camera(req.get("visibility")):
            not_visible += 1
            continue
        known, gold_value = gold_end_fact(req, use_achieved)
        if not known:
            unknown += 1
            continue
        signal_predicate, signal_ref, signal_value = _signal_fact(signal_by_predicate, predicate)
        if signal_predicate is None:
            missing += 1
            ok = False
        else:
            ok = signal_ref == _ref_object(req) and signal_value == gold_value
        total += 1
        correct += int(ok)
        items.append({
            "req_id": req.get("req_id"),
            "predicate": predicate,
            "gold_ref_object": _ref_object(req),
            "gold_value": gold_value,
            "signal_predicate": signal_predicate,
            "signal_ref_object": signal_ref,
            "signal_value": signal_value,
            "correct": ok,
        })
    return {
        "correct": correct,
        "total": total,
        "missing_signal_item": missing,
        "excluded_not_visible": not_visible,
        "excluded_unknown": unknown,
        "items": items,
    }


def x2_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Micro X2: correct items / eligible gold items."""
    episodes = list(episodes)
    return {
        "metric_id": "X2",
        "n_episodes": len(episodes),
        **_ratio(sum(int(e["correct"]) for e in episodes), sum(int(e["total"]) for e in episodes)),
        "missing_signal_item": sum(int(e["missing_signal_item"]) for e in episodes),
        "excluded_not_visible": sum(int(e["excluded_not_visible"]) for e in episodes),
        "excluded_unknown": sum(int(e["excluded_unknown"]) for e in episodes),
    }


# --------------------------------------------------------------------------- #
# X3: failed-grasp detection from signals
# --------------------------------------------------------------------------- #
def _attempt_span(attempt: Mapping[str, Any]) -> tuple[int, int] | None:
    """Span of an attempt: ``start``/``end`` (view records) or ``span``; else the raw L1 fields.

    A raw L1 attempt has no start or end key. Its span is ``closing_onset`` (the first frame of the
    grasp phase) to ``event_frame`` (where the miss or slip becomes evident, like the gold span of
    spec 3.4.5), with ``end_frame`` when there is no event frame.
    """
    span = as_span(attempt)
    if span is not None:
        return span
    start = attempt.get("closing_onset")
    end = attempt.get("event_frame")
    if end is None:
        end = attempt.get("end_frame")
    return as_span([start, end]) if start is not None and end is not None else None


def signal_failed_spans(attempts: Iterable[Mapping[str, Any]]) -> list[list[int]]:
    """Spans of failed signal-layer attempts (``source`` absent or "signal"), in the given order.

    Failed means ``outcome`` in ``SIGNAL_FAILED_OUTCOMES`` or ``failure_type`` in the X3 types.
    Accepts view-record attempts and raw L1 attempts (see ``_attempt_span``). A failed attempt
    with no usable span is logged as a warning, never dropped silently.
    """
    spans: list[list[int]] = []
    for att in attempts:
        source = _norm(att.get("source"))
        if source not in (None, "signal"):
            continue
        failed = (_norm(att.get("outcome")) in SIGNAL_FAILED_OUTCOMES
                  or _norm(att.get("failure_type")) in X3_FAILURE_TYPES)
        if not failed:
            continue
        span = _attempt_span(att)
        if span is None:
            logger.warning("X3: failed signal attempt without a usable span was skipped "
                           "(attempt_idx=%r, outcome=%r)", att.get("attempt_idx"), att.get("outcome"))
            continue
        spans.append([span[0], span[1]])
    return spans


def x3_episode(signal_spans: Sequence[Any], gold_failed_attempts: Sequence[Mapping[str, Any]],
               min_iou: float = X3_MIN_IOU) -> dict[str, Any]:
    """X3 counts for one episode.

    ``signal_spans`` are the signal layer's failed-attempt spans only (see
    ``signal_failed_spans``). Gold is restricted to missed_grasp, slip and drop. Every signal span
    counts in the precision denominator; ``pred_matching_excluded_types`` counts unmatched signal
    spans that would match a gold attempt of another type, so that effect stays visible.
    """
    pred = [s for s in (as_span(x) for x in signal_spans) if s is not None]
    kept = [fa for fa in gold_failed_attempts if _norm(fa.get("failure_type")) in X3_FAILURE_TYPES]
    other = [fa for fa in gold_failed_attempts if _norm(fa.get("failure_type")) not in X3_FAILURE_TYPES]
    pairs = match_spans(pred, kept, min_iou)
    matched_pred = {i for i, _, _ in pairs}
    unmatched = [p for i, p in enumerate(pred) if i not in matched_pred]
    return {
        "min_iou": min_iou,
        "matched": len(pairs),
        "n_pred": len(pred),
        "n_gold": len(kept),
        "n_gold_excluded_types": len(other),
        "pred_matching_excluded_types": len(match_spans(unmatched, other, min_iou)),
        "pairs": [[i, j, iou] for i, j, iou in pairs],
    }


def x3_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Micro X3: span F1 = 2M / (m + n) (equal to 2PR / (P + R)), with precision and recall.

    As in ``t1_micro``, F1 is flagged ``zero_denominator`` when precision or recall is 0 / 0.
    """
    episodes = list(episodes)
    matched = sum(int(e["matched"]) for e in episodes)
    n_pred = sum(int(e["n_pred"]) for e in episodes)
    n_gold = sum(int(e["n_gold"]) for e in episodes)
    f1 = _ratio(2 * matched, n_pred + n_gold)
    f1["zero_denominator"] = n_pred == 0 or n_gold == 0
    return {
        "metric_id": "X3",
        "n_episodes": len(episodes),
        **f1,
        "precision": _ratio(matched, n_pred),
        "recall": _ratio(matched, n_gold),
        "pred_matching_excluded_types": sum(int(e["pred_matching_excluded_types"]) for e in episodes),
    }
