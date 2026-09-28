"""Temporal metrics T1 to T6 (MEASUREMENT_SPEC 4.1) and the span matching used by S4 and X3.

A segment is a dict with inclusive, 0-based, episode-relative ``start_frame`` and ``end_frame``
(``start`` and ``end`` are accepted as aliases, as the view records use them). A boundary is the
end frame of every segment except the last (spec section 1). A span is ``[start, end]`` inclusive.

T1 uses the exact optimal matching of the spec (most pairs, then the smallest summed error, then
the lowest gold index, then the lowest predicted index) and also counts the legacy greedy matching
of ``robolabel.metrics.boundary_pr_mae``; a disagreement between the two is logged as a warning.
T3 is the optimal one-to-one IoU assignment divided by the larger segment count; T3-legacy keeps
the index-aligned mean of ``reliability.py`` for continuity only.

Every function is pure and deterministic. Floats in returned values are rounded to 6 decimals.
scipy is imported inside the functions that need it, so the package imports without it.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

T2_TAU = 10  # T2-MAE matches at plus or minus 10 frames (spec 4.1)
UNIFORM_CV_THRESHOLD = 0.12  # legacy gate.is_uniform_split threshold
UNIFORM_MIN_SEGMENTS = 3
S4_MIN_IOU = 0.3
DECIMALS = 6


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _r6(value: float | None) -> float | None:
    """Round a float for JSON output (None stays None)."""
    return None if value is None else round(float(value), DECIMALS)


def _as_int(value: Any) -> int | None:
    """Frame value as int; None and NaN become None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def _first_int(seg: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _as_int(seg.get(key))
        if value is not None:
            return value
    return None


def seg_start(seg: Mapping[str, Any]) -> int | None:
    """First frame of a segment (``start_frame``, else ``start``)."""
    return _first_int(seg, ("start_frame", "start"))


def seg_end(seg: Mapping[str, Any]) -> int | None:
    """Last frame of a segment, inclusive (``end_frame``, else ``end``)."""
    return _first_int(seg, ("end_frame", "end"))


def _is_pair(item: Any) -> bool:
    """True for a list, tuple or numpy array of length 2 (pyarrow and pandas give arrays)."""
    if isinstance(item, (str, bytes, Mapping)):
        return False
    try:
        return len(item) == 2 and hasattr(item, "__getitem__")
    except TypeError:
        return False


def as_span(item: Any) -> tuple[int, int] | None:
    """A ``(start, end)`` span from a 2-sequence or array, a dict with ``span``, or a segment-like dict."""
    if isinstance(item, Mapping):
        if item.get("span") is not None:
            return as_span(item["span"])
        start, end = seg_start(item), seg_end(item)
        return None if start is None or end is None else (start, end)
    if _is_pair(item):
        start, end = _as_int(item[0]), _as_int(item[1])
        return None if start is None or end is None else (start, end)
    return None


def _flag(value: Any) -> bool:
    """True for True, 1 or the string "true"; False for None, NaN and everything else falsy."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _check_tau(tau: Any) -> int:
    if int(tau) != tau or tau < 0:
        raise ValueError(f"tau must be a non-negative integer number of frames, got {tau!r}")
    return int(tau)


def _frames(values: Sequence[Any]) -> list[int]:
    return [int(v) for v in values]


def _percentile(values: Sequence[float], q: float) -> float | None:
    import numpy as np

    if not values:
        return None
    return _r6(float(np.percentile(np.asarray(values, dtype=float), q, method="linear")))


# --------------------------------------------------------------------------- #
# Boundaries and T1 matching
# --------------------------------------------------------------------------- #
def boundaries(segments: Sequence[Mapping[str, Any]]) -> list[int]:
    """End frame of every segment but the last, in the given order (legacy convention).

    Segments whose end frame is missing are skipped, as the legacy gold reader does.
    """
    segs = list(segments)
    out: list[int] = []
    for seg in segs[:-1]:
        end = seg_end(seg)
        if end is not None:
            out.append(end)
    return out


def _optimal_count_sum(pred: Sequence[int], gold: Sequence[int], tau: int) -> tuple[int, int]:
    """(most pairs, smallest summed |p - g| among those) over matchings within tau.

    Solved by ``linear_sum_assignment`` with cost ``|p - g|`` inside tau and
    ``BIG = tau * (m + n) + 1`` outside; BIG pairs are dropped. BIG exceeds any achievable sum
    of real costs, so the solver first maximizes the pair count, then minimizes the sum.
    """
    # Values with no partner within tau can never be matched; dropping them shrinks the problem.
    p = [x for x in pred if any(abs(x - y) <= tau for y in gold)]
    g = [y for y in gold if any(abs(x - y) <= tau for x in p)]
    if not p or not g:
        return 0, 0
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    dist = np.abs(np.subtract.outer(np.asarray(p, dtype=np.int64), np.asarray(g, dtype=np.int64)))
    big = tau * (len(p) + len(g)) + 1
    cost = np.where(dist <= tau, dist, big)
    rows, cols = linear_sum_assignment(cost)
    count = total = 0
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        d = int(dist[r, c])
        if d <= tau:
            count += 1
            total += d
    return count, total


def match_boundaries(pred: Sequence[int], gold: Sequence[int], tau: int) -> list[tuple[int, int]]:
    """Optimal one-to-one boundary matching of spec 4.1, as ``(pred_idx, gold_idx)`` pairs.

    Rule: the matching with the most pairs within tau; among those the smallest sum of
    ``|p - g|``; remaining ties go to the lowest gold index, then the lowest predicted index.
    The solver gives the optimal (count, sum). The tie rules are then applied explicitly: gold
    indices are walked in ascending order, and gold ``j`` takes the lowest unused predicted index
    ``i`` for which the residual problem (unused predictions, gold after ``j``) still reaches the
    optimal count and sum; if no ``i`` works, gold ``j`` stays unmatched. This is the
    lexicographically smallest matching when each gold index is read as the predicted index it
    takes (unmatched last). Pairs come back in ascending gold order.
    """
    tau = _check_tau(tau)
    p, g = _frames(pred), _frames(gold)
    best_count, best_sum = _optimal_count_sum(p, g, tau)
    pairs: list[tuple[int, int]] = []
    if best_count == 0:
        return pairs
    used: set[int] = set()
    count = total = 0
    for j, gj in enumerate(g):
        if count == best_count:
            break  # the optimum is reached; the remaining gold stays unmatched
        for i, pi in enumerate(p):
            d = abs(pi - gj)
            if i in used or d > tau:
                continue
            rest_p = [p[k] for k in range(len(p)) if k not in used and k != i]
            rc, rs = _optimal_count_sum(rest_p, g[j + 1:], tau)
            if count + 1 + rc == best_count and total + d + rs == best_sum:
                pairs.append((i, j))
                used.add(i)
                count += 1
                total += d
                break
    if (count, total) != (best_count, best_sum):  # cannot happen; guards the invariant
        raise RuntimeError("boundary matching lost the optimum while applying the tie rules")
    return pairs


def greedy_match_count(pred: Sequence[int], gold: Sequence[int], tau: int) -> int:
    """Legacy greedy count, exactly as ``robolabel.metrics.boundary_pr_mae``.

    Gold boundaries in ascending order; each takes the nearest unused prediction within tau.
    On equal distance the earlier prediction in list order wins (strict ``<``).
    """
    p = _frames(pred)
    used = [False] * len(p)
    matched = 0
    for gb in sorted(_frames(gold)):
        best, bd = -1, tau + 1
        for j, pb in enumerate(p):
            if used[j]:
                continue
            d = abs(pb - gb)
            if d <= tau and d < bd:
                best, bd = j, d
        if best >= 0:
            used[best] = True
            matched += 1
    return matched


def _episode_prf(matched: int, m: int, n: int) -> tuple[float, float, float]:
    """Per-episode precision, recall, F1 (distributions only).

    F1 is 1 when m = n = 0 and 0 when exactly one of them is 0 (spec 4.1). Precision and recall
    follow the same convention when their denominator is 0.
    """
    if m == 0 and n == 0:
        return 1.0, 1.0, 1.0
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0
    # 2PR / (P + R) with P = M / m and R = M / n is exactly 2M / (m + n).
    return matched / m, matched / n, 2 * matched / (m + n)


def t1_episode(pred_bounds: Sequence[int], gold_bounds: Sequence[int], tau: int,
               episode_key: str | None = None) -> dict[str, Any]:
    """T1 for one episode: optimal matched count, the legacy greedy count, per-episode P, R, F1."""
    tau = _check_tau(tau)
    pred, gold = _frames(pred_bounds), _frames(gold_bounds)
    pairs = match_boundaries(pred, gold, tau)
    matched, m, n = len(pairs), len(pred), len(gold)
    greedy = greedy_match_count(pred, gold, tau)
    disagrees = greedy != matched
    if disagrees:
        logger.warning(
            "T1 greedy/optimal disagreement%s at tau=%d: optimal %d, greedy %d (pred=%s, gold=%s)",
            f" on {episode_key}" if episode_key else "", tau, matched, greedy, pred, gold,
        )
    precision, recall, f1 = _episode_prf(matched, m, n)
    return {
        "episode_key": episode_key,
        "tau": tau,
        "matched": matched,
        "n_pred": m,
        "n_gold": n,
        "greedy_matched": greedy,
        "greedy_disagrees": disagrees,
        "precision": _r6(precision),
        "recall": _r6(recall),
        "f1": _r6(f1),
        "pairs": [[i, j] for i, j in pairs],
        "abs_errors": [abs(pred[i] - gold[j]) for i, j in pairs],
    }


def _ratio(num: int, den: int, name: str, zero: list[str]) -> float:
    """num / den, with 0 / 0 defined as 0.0 and recorded in ``zero``."""
    if den == 0:
        zero.append(name)
        return 0.0
    return num / den


def t1_micro(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Micro T1 over ``t1_episode`` dicts: P = sum M / sum m, R = sum M / sum n, F1 = 2PR / (P + R).

    A zero denominator gives 0.0 and its name is listed in ``zero_denominators``; F1 is listed
    there when it depends on such a value. All episodes must share one tau.
    """
    episodes = list(episodes)  # read several times below; a generator would give zeros
    taus = sorted({int(e["tau"]) for e in episodes if e.get("tau") is not None})
    if len(taus) > 1:
        raise ValueError(f"t1_micro got episodes with different tau values: {taus}")
    matched = sum(int(e["matched"]) for e in episodes)
    n_pred = sum(int(e["n_pred"]) for e in episodes)
    n_gold = sum(int(e["n_gold"]) for e in episodes)
    zero: list[str] = []
    precision = _ratio(matched, n_pred, "precision", zero)
    recall = _ratio(matched, n_gold, "recall", zero)
    if n_pred == 0 or n_gold == 0:
        f1 = 0.0
        zero.append("f1")
    else:
        f1 = 2 * matched / (n_pred + n_gold)  # equals 2PR / (P + R)
    return {
        "tau": taus[0] if taus else None,
        "precision": _r6(precision),
        "recall": _r6(recall),
        "f1": _r6(f1),
        "matched": matched,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "n_episodes": len(episodes),
        "greedy_matched": sum(int(e.get("greedy_matched", 0)) for e in episodes),
        "greedy_disagreements": sum(1 for e in episodes if e.get("greedy_disagrees")),
        "zero_denominators": zero,
    }


# --------------------------------------------------------------------------- #
# T2: boundary error
# --------------------------------------------------------------------------- #
def t2(pred_bounds: Sequence[int], gold_bounds: Sequence[int], fps: float,
       tau: int = T2_TAU) -> dict[str, Any]:
    """T2-MAE over optimal pairs at tau (default 10) and T2-near per gold boundary.

    T2-near: for each gold boundary, the distance to the nearest predicted boundary, capped at
    1.0 s (``round(fps)`` native frames); with no predicted boundary every gold boundary gets the
    cap. ``near_frames`` is in gold order so callers can pool the median across episodes.
    """
    if fps is None or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    tau = _check_tau(tau)
    pred, gold = _frames(pred_bounds), _frames(gold_bounds)
    pairs = match_boundaries(pred, gold, tau)
    errors = [abs(pred[i] - gold[j]) for i, j in pairs]
    mae = sum(errors) / len(errors) if errors else None
    cap = int(round(fps))
    near = [min(cap, min(abs(p - g) for p in pred)) if pred else cap for g in gold]
    near_median = statistics.median(near) if near else None
    return {
        "tau": tau,
        "fps": _r6(fps),  # a plain float, so numpy fps values still serialize to JSON
        "n_pairs": len(errors),
        "abs_errors": errors,
        "sum_abs_error": sum(errors),
        "mae_frames": _r6(mae),
        "mae_seconds": _r6(None if mae is None else mae / fps),
        "near_cap_frames": cap,
        "near_frames": near,
        "near_seconds": [_r6(d / fps) for d in near],
        "near_median_frames": _r6(near_median),
        "near_median_seconds": _r6(None if near_median is None else near_median / fps),
    }


def t2_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool ``t2`` dicts: MAE over all matched pairs, median of T2-near over all gold boundaries."""
    episodes = list(episodes)
    errors_f: list[int] = []
    errors_s: list[float] = []
    near_f: list[int] = []
    near_s: list[float] = []
    for e in episodes:
        fps = float(e["fps"])
        errors_f.extend(int(x) for x in e["abs_errors"])
        errors_s.extend(int(x) / fps for x in e["abs_errors"])
        near_f.extend(int(x) for x in e["near_frames"])
        near_s.extend(int(x) / fps for x in e["near_frames"])
    return {
        "n_episodes": len(episodes),
        "n_pairs": len(errors_f),
        "mae_frames": _r6(sum(errors_f) / len(errors_f)) if errors_f else None,
        "mae_seconds": _r6(math.fsum(errors_s) / len(errors_s)) if errors_s else None,
        "n_gold_boundaries": len(near_f),
        "near_median_frames": _r6(statistics.median(near_f)) if near_f else None,
        "near_median_seconds": _r6(statistics.median(near_s)) if near_s else None,
    }


# --------------------------------------------------------------------------- #
# T3: segment IoU
# --------------------------------------------------------------------------- #
def span_iou(a: tuple[int, int] | None, b: tuple[int, int] | None) -> float:
    """Temporal IoU of two inclusive frame spans (0.0 when either is missing or empty)."""
    if a is None or b is None:
        return 0.0
    len_a = max(0, a[1] - a[0] + 1)
    len_b = max(0, b[1] - b[0] + 1)
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1) if len_a and len_b else 0
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(pred: Sequence[tuple[int, int] | None],
                gold: Sequence[tuple[int, int] | None]) -> list[list[float]]:
    return [[span_iou(p, g) for g in gold] for p in pred]


def _t3_raw_pairs(pred_segments: Sequence[Mapping[str, Any]],
                  gold_segments: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, float]]:
    pred = [as_span(s) for s in pred_segments]
    gold = [as_span(s) for s in gold_segments]
    if not pred or not gold:
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    ious = np.asarray(_iou_matrix(pred, gold), dtype=float)
    rows, cols = linear_sum_assignment(-ious)
    pairs = [(int(r), int(c), float(ious[r, c])) for r, c in zip(rows.tolist(), cols.tolist(), strict=True)
             if ious[r, c] > 0]
    return sorted(pairs, key=lambda x: (x[1], x[0]))


def t3_pairs(pred_segments: Sequence[Mapping[str, Any]],
             gold_segments: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, float]]:
    """Optimal one-to-one (pred_idx, gold_idx, iou) assignment maximizing total IoU.

    Solved with ``linear_sum_assignment`` on ``-IoU`` over inclusive frame ranges. Pairs with
    IoU 0 are dropped (they add nothing). Sorted by gold index. S1-seg can read these.
    """
    return [(i, j, _r6(iou)) for i, j, iou in _t3_raw_pairs(pred_segments, gold_segments)]


def t3_episode(pred_segments: Sequence[Mapping[str, Any]],
               gold_segments: Sequence[Mapping[str, Any]]) -> float:
    """T3: sum of matched IoU divided by max(#gold, #pred); unmatched segments count as 0."""
    denom = max(len(pred_segments), len(gold_segments))
    if denom == 0:
        return 0.0
    total = math.fsum(iou for _, _, iou in _t3_raw_pairs(pred_segments, gold_segments))
    return _r6(total / denom)


def t3_summary(scores: Sequence[float]) -> dict[str, Any]:
    """Mean, median and 10th percentile (numpy, linear interpolation) of per-episode T3 scores."""
    vals = [float(s) for s in scores]
    return {
        "n_episodes": len(vals),
        "mean": _r6(math.fsum(vals) / len(vals)) if vals else None,
        "median": _percentile(vals, 50),
        "p10": _percentile(vals, 10),
    }


def _legacy_iou(a: Mapping[str, Any], g: Mapping[str, Any]) -> float | None:
    """Legacy inclusive-frame IoU of ``reliability.py`` (hull as union; None when a frame is missing)."""
    a0, a1, g0, g1 = seg_start(a), seg_end(a), seg_start(g), seg_end(g)
    if None in (a0, a1, g0, g1):
        return None
    inter = max(0, min(a1, g1) - max(a0, g0) + 1)
    union = max(a1, g1) - min(a0, g0) + 1
    return inter / union if union > 0 else None


def t3_legacy_ious(pred_segments: Sequence[Mapping[str, Any]],
                   gold_segments: Sequence[Mapping[str, Any]]) -> list[float]:
    """Index-aligned IoUs, ``zip(..., strict=False)`` as in ``reliability.py`` (extra segments ignored)."""
    out: list[float] = []
    for a, g in zip(pred_segments, gold_segments, strict=False):
        iou = _legacy_iou(a, g)
        if iou is not None:
            out.append(iou)
    return out


def t3_legacy_episode(pred_segments: Sequence[Mapping[str, Any]],
                      gold_segments: Sequence[Mapping[str, Any]]) -> float | None:
    """T3-legacy for one episode: mean index-aligned IoU (None when nothing aligns). Continuity only."""
    ious = t3_legacy_ious(pred_segments, gold_segments)
    return _r6(statistics.mean(ious)) if ious else None


def t3_legacy_flat_mean(per_episode_ious: Sequence[Sequence[float]]) -> float | None:
    """Pooled per-segment mean over episodes: the legacy headline IoU (``reliability_report``)."""
    pooled = [float(x) for ious in per_episode_ious for x in ious]
    return _r6(statistics.mean(pooled)) if pooled else None


# --------------------------------------------------------------------------- #
# T4 to T6
# --------------------------------------------------------------------------- #
def _lengths(segments: Sequence[Mapping[str, Any]]) -> list[int]:
    out: list[int] = []
    for seg in segments:
        start, end = seg_start(seg), seg_end(seg)
        if start is not None and end is not None:
            out.append(end - start + 1)
    return out


def t4_episode(pred_segments: Sequence[Mapping[str, Any]], n_gold_segments: int) -> dict[str, Any]:
    """T4 flags for one episode.

    ``single_segment``: exactly one predicted segment while the gold has at least two.
    ``uniform``: at least 3 predicted segment lengths (end - start + 1) with coefficient of
    variation (``statistics.pstdev`` / mean) below 0.12, the legacy ``gate.is_uniform_split`` rule.
    """
    n_pred = len(pred_segments)
    single = n_pred == 1 and int(n_gold_segments) >= 2
    lengths = _lengths(pred_segments)
    cv: float | None = None
    if len(lengths) >= UNIFORM_MIN_SEGMENTS:
        mean = statistics.mean(lengths)
        if mean > 0:
            cv = statistics.pstdev(lengths) / mean
    uniform = cv is not None and cv < UNIFORM_CV_THRESHOLD
    return {
        "degenerate": single or uniform,
        "single_segment": single,
        "uniform": uniform,
        "n_pred": n_pred,
        "n_gold": int(n_gold_segments),
        "cv": _r6(cv),
    }


def t4_rate(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Degenerate rate = degenerate episodes / episodes, with the two causes separately."""
    episodes = list(episodes)
    n = len(episodes)
    degenerate = sum(1 for e in episodes if e.get("degenerate"))
    single = sum(1 for e in episodes if e.get("single_segment"))
    uniform = sum(1 for e in episodes if e.get("uniform"))
    return {
        "n_episodes": n,
        "degenerate": degenerate,
        "single_segment": single,
        "uniform": uniform,
        "rate": _r6(degenerate / n) if n else None,
        "single_segment_rate": _r6(single / n) if n else None,
        "uniform_rate": _r6(uniform / n) if n else None,
    }


def t5_episode(n_pred: int, n_gold: int) -> int:
    """T5 granularity error: #predicted segments - #gold segments."""
    return int(n_pred) - int(n_gold)


def t5_summary(diffs: Sequence[int]) -> dict[str, Any]:
    """Mean, mean absolute value, and share of episodes with an absolute difference of 2 or more."""
    vals = [int(d) for d in diffs]
    n = len(vals)
    return {
        "n_episodes": n,
        "mean": _r6(sum(vals) / n) if n else None,
        "mean_abs": _r6(sum(abs(v) for v in vals) / n) if n else None,
        "share_abs_ge_2": _r6(sum(1 for v in vals if abs(v) >= 2) / n) if n else None,
    }


def t6_episode(pred_coarse_segments: Sequence[Mapping[str, Any]],
               gold_coarse_segments: Sequence[Mapping[str, Any]],
               tau: int, episode_key: str | None = None) -> dict[str, Any]:
    """T6: T1 applied to coarse-subtask boundaries (aggregate with ``t1_micro``)."""
    return t1_episode(boundaries(pred_coarse_segments), boundaries(gold_coarse_segments), tau, episode_key)


def missing_output_segments(num_frames: int) -> list[dict[str, Any]]:
    """The spec 4.0 missing-output prediction: one segment over the episode, no target, confidence 0.5."""
    n = int(num_frames)
    if n < 1:
        raise ValueError(f"num_frames must be at least 1, got {num_frames!r}")
    return [{
        "start_frame": 0,
        "end_frame": n - 1,
        "phase_class": None,
        "target": None,
        "destination": None,
        "outcome": None,
        "mistake": False,
        "confidence": 0.5,
        "missing_output": True,
    }]


# --------------------------------------------------------------------------- #
# Span matching (S4, X3)
# --------------------------------------------------------------------------- #
def match_spans(pred_spans: Sequence[Any], gold_spans: Sequence[Any],
                min_iou: float = S4_MIN_IOU) -> list[tuple[int, int, float]]:
    """One-to-one (pred_idx, gold_idx, iou) matching maximizing total IoU over pairs with IoU >= min_iou.

    Spans are ``[start, end]`` inclusive (2-sequences, dicts with ``span``, or segment-like dicts).
    Pairs below ``min_iou`` or with no overlap are not allowed. Sorted by gold index.
    """
    pred = [as_span(s) for s in pred_spans]
    gold = [as_span(s) for s in gold_spans]
    if not pred or not gold:
        return []
    ious = _iou_matrix(pred, gold)
    allowed = [[iou > 0 and iou >= min_iou for iou in row] for row in ious]
    if not any(any(row) for row in allowed):
        return []
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # Disallowed pairs weigh 0, so the best assignment is the best matching of allowed pairs.
    weights = np.asarray([[iou if ok else 0.0 for iou, ok in zip(r, a, strict=True)]
                          for r, a in zip(ious, allowed, strict=True)], dtype=float)
    rows, cols = linear_sum_assignment(-weights)
    pairs = [(int(r), int(c), _r6(ious[r][c])) for r, c in zip(rows.tolist(), cols.tolist(), strict=True)
             if allowed[r][c]]
    return sorted(pairs, key=lambda x: (x[1], x[0]))


def failed_spans_from_segments(segments: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Merge consecutive segments with ``outcome == "failed"`` or ``mistake`` true into spans.

    Segments are ordered by start frame first. Returns ``[start, end]`` inclusive spans.
    """
    ordered = sorted(
        (s for s in segments if seg_start(s) is not None and seg_end(s) is not None),
        key=lambda s: (seg_start(s), seg_end(s)),
    )
    spans: list[list[int]] = []
    open_span: list[int] | None = None
    for seg in ordered:
        failed = str(seg.get("outcome") or "").strip().lower() == "failed" or _flag(seg.get("mistake"))
        if failed:
            if open_span is None:
                open_span = [seg_start(seg), seg_end(seg)]
            else:
                open_span[1] = max(open_span[1], seg_end(seg))
        elif open_span is not None:
            spans.append(open_span)
            open_span = None
    if open_span is not None:
        spans.append(open_span)
    return spans
