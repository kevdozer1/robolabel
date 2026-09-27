"""Statistics of MEASUREMENT_SPEC section 6: cluster bootstrap, outcome words, MEI, Holm, ECE.

- The episode is the unit (6.1). Each bootstrap replicate resamples every family's episodes with
  replacement to that family's size, recomputes the metric from counts (micro within a family,
  macro across families) and, for a contrast, takes the paired difference of two systems on the
  identical episode set (6.2). B = 10,000 by default, generator
  ``numpy.random.default_rng(20261001 + contrast_index)``, 95 percent percentile interval, point
  estimate from the original sample.
- ``p_boot`` is the share of replicates on the other side of 0 (a replicate equal to 0 counts as
  the other side), times 2, capped at 1 (6.5). :func:`holm` adjusts a list of them.
- A paired difference uses only the families where both systems have a nonzero denominator, in
  the point estimate and in every replicate.
- :func:`outcome_word` gives "clear difference", "no clear difference" or "inconclusive" (6.3);
  ``MEI`` is the 6.3 table.
- :func:`ece` is the C2 calibration error with equal-count bins (4.5, Appendix F item 8).

Episodes are ``score_view`` results or any dicts with ``episode_key``, ``family`` and
``counts: {metric_key: {"numerator", "denominator", "pending"}}``. A family whose denominator is 0
in a replicate is left out of that replicate's macro. Floats are rounded to 6 decimals, and the
output is identical for identical input and seed (10.6).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

DECIMALS = 6
BASE_SEED = 20261001
N_BOOT = 10_000
CONFIDENCE = 0.95
PENDING = "pending"
CLEAR = "clear difference"
NO_CLEAR = "no clear difference"
INCONCLUSIVE = "inconclusive"
ECE_BINS = 15

# Minimum effect of interest per metric (spec 6.3), absolute. T2 values are frames. C3 is a share of
# the reference value (see ``MEI_RELATIVE``). Subset rows are keyed "<metric>:<subset>".
MEI: dict[str, float] = {
    "T1-P": 0.05, "T1-R": 0.05, "T1-F1": 0.05,
    "T2-MAE": 1.0, "T2-near": 1.0,
    "T3": 0.05,
    "T4": 0.05,
    "S1": 0.05, "S2": 0.05,
    "S2:distractor": 0.10, "G3:distractor": 0.10,
    "S3": 0.03,
    "S4-F1": 0.10, "G1": 0.10, "G5": 0.10, "G4": 0.10,
    "G2": 0.05, "G3": 0.05,
    "G6": 0.10,
    "V1": 0.05,
    "V2": 0.05,
    "C1-R@10": 0.10,
    "C2": 0.03,
    "C3": 0.20,
    "D1": 0.10,
    "D2": 0.05,
}
MEI_RELATIVE = frozenset({"C3"})
MEI_UNITS = {"T2-MAE": "frames", "T2-near": "frames", "C3": "share of the reference value"}
# Sub-metrics that the 6.3 rows cover (G1 parts a to c, G5 precision and recall).
_MEI_ALIASES = {"G1a": "G1", "G1b": "G1", "G1c": "G1", "G5-P": "G5", "G5-R": "G5"}


def _r6(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    out = round(float(value), DECIMALS)
    return 0.0 if out == 0 else out


def mei_for(metric_key: str, subset: str = "all", reference: float | None = None) -> float | None:
    """The MEI of a metric key such as ``T1-F1@5`` or ``G1b``, or None when 6.3 declares none.

    ``subset`` picks the distractor rows of S2 and G3. C3 needs the ``reference`` value.
    """
    base = metric_key.split("@", 1)[0]
    base = _MEI_ALIASES.get(base, base)
    if base.startswith("G5-") and base.split("-")[1] in ("P", "R"):
        base = "G5"
    if subset and subset != "all" and f"{base}:{subset}" in MEI:
        return MEI[f"{base}:{subset}"]
    if metric_key in MEI and metric_key not in MEI_RELATIVE:
        return MEI[metric_key]
    if base in MEI_RELATIVE:
        return None if reference is None else _r6(MEI[base] * abs(float(reference)))
    return MEI.get(base)


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #
def _episode_sort_key(ep: Mapping[str, Any]) -> tuple[str, int, str]:
    key = str(ep.get("episode_key") or "")
    family, _, index = key.partition("/")
    return (_family(ep), int(index) if index.isdigit() else -1, key)


def _family(ep: Mapping[str, Any]) -> str:
    fam = ep.get("family")
    if fam:
        return str(fam)
    key = str(ep.get("episode_key") or "")
    return key.split("/", 1)[0] if "/" in key else ""


def episode_counts(ep: Mapping[str, Any], metric_key: str) -> tuple[float, float, int]:
    """(numerator, denominator, pending) of one episode; a missing metric is (0, 0, 0)."""
    c = (ep.get("counts") or {}).get(metric_key)
    if c is None:
        return 0.0, 0.0, 0
    if isinstance(c, Mapping):
        num = c.get("numerator", c.get("num", 0))
        den = c.get("denominator", c.get("den", 0))
        return float(num or 0), float(den or 0), int(c.get("pending") or 0)
    num, den = c
    return float(num), float(den), 0


def _family_arrays(episodes: Iterable[Mapping[str, Any]], metric_key: str,
                   families: Iterable[str] | None = None) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int,
                                                                   list[str]]:
    keep = None if families is None else {str(f) for f in families}
    rows: dict[str, list[tuple[float, float]]] = {}
    keys: list[str] = []
    pending = 0
    for ep in sorted(episodes, key=_episode_sort_key):
        fam = _family(ep)
        if keep is not None and fam not in keep:
            continue
        num, den, pend = episode_counts(ep, metric_key)
        pending += pend
        rows.setdefault(fam, []).append((num, den))
        keys.append(str(ep.get("episode_key")))
    arrays = {f: (np.asarray([r[0] for r in rows[f]], dtype=float), np.asarray([r[1] for r in rows[f]], dtype=float))
              for f in sorted(rows)}
    return arrays, pending, keys


def _family_value(num: float, den: float) -> float:
    return num / den if den else math.nan


def _macro(values: Sequence[float]) -> float:
    ok = [v for v in values if not math.isnan(v)]
    return math.fsum(ok) / len(ok) if ok else math.nan


def _point(arrays: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, dict[str, float]]:
    by_family = {f: _family_value(math.fsum(n.tolist()), math.fsum(d.tolist())) for f, (n, d) in arrays.items()}
    return _macro(list(by_family.values())), by_family


def metric_value(episodes: Iterable[Mapping[str, Any]], metric_key: str,
                 families: Iterable[str] | None = None) -> dict[str, Any]:
    """Point value from counts: micro within each family, macro across families."""
    arrays, pending, keys = _family_arrays(episodes, metric_key, families)
    if pending:
        return {"metric_key": metric_key, "value": PENDING, "by_family": {}, "n_episodes": len(keys),
                "pending_items": pending}
    macro, by_family = _point(arrays)
    return {
        "metric_key": metric_key,
        "value": _r6(macro),
        "by_family": {f: _r6(v) for f, v in by_family.items()},
        "n_episodes": len(keys),
        "n_by_family": {f: int(len(n)) for f, (n, _) in arrays.items()},
        "pending_items": 0,
    }


def _point_paired(arrays_a: Mapping[str, tuple[np.ndarray, np.ndarray]],
                  arrays_b: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, list[str]]:
    """Macro values of two systems over the families where both are defined (paired, 6.1)."""
    _, fam_a = _point(arrays_a)
    _, fam_b = _point(arrays_b)
    both = [f for f in sorted(fam_a) if not math.isnan(fam_a[f]) and not math.isnan(fam_b.get(f, math.nan))]
    return _macro([fam_a[f] for f in both]), _macro([fam_b[f] for f in both]), both


def _replicates(systems: Sequence[Mapping[str, tuple[np.ndarray, np.ndarray]]], rng: np.random.Generator,
                n_boot: int) -> list[np.ndarray]:
    """Macro value per replicate for each system, resampling the same episodes for every system.

    With several systems a family enters a replicate's macro only where every system has a
    nonzero denominator, so the systems are compared on the same families.
    """
    fams = sorted(systems[0])
    per_system: list[list[np.ndarray]] = [[] for _ in systems]
    for fam in fams:
        n_f = len(systems[0][fam][0])
        idx = rng.integers(0, n_f, size=(n_boot, n_f))
        for s, arrays in enumerate(systems):
            num, den = arrays[fam]
            nums, dens = num[idx].sum(axis=1), den[idx].sum(axis=1)
            vals = np.full(n_boot, np.nan)
            np.divide(nums, dens, out=vals, where=dens > 0)
            per_system[s].append(vals)
    common = None
    if len(systems) > 1 and fams:
        common = np.logical_and.reduce([~np.isnan(np.vstack(vals)) for vals in per_system])
    out: list[np.ndarray] = []
    for vals in per_system:
        stack = np.vstack(vals) if vals else np.full((1, n_boot), np.nan)
        valid = ~np.isnan(stack) if common is None else common
        total = np.where(valid, stack, 0.0).sum(axis=0)
        n_valid = valid.sum(axis=0)
        macro = np.full(n_boot, np.nan)
        np.divide(total, n_valid, out=macro, where=n_valid > 0)
        out.append(macro)
    return out


def _interval(values: np.ndarray, confidence: float = CONFIDENCE) -> list[float] | None:
    ok = values[~np.isnan(values)]
    if ok.size == 0:
        return None
    tail = (1.0 - confidence) / 2.0 * 100.0
    lo, hi = np.percentile(ok, [tail, 100.0 - tail], method="linear")
    return [_r6(float(lo)), _r6(float(hi))]


def p_boot(replicate_diffs: Iterable[float], point_diff: float) -> float:
    """Share of replicates on the other side of 0 from the point estimate, times 2, capped at 1.

    A replicate equal to 0 counts as the other side. A point estimate of 0 gives 1.
    """
    diffs = np.asarray(list(replicate_diffs), dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0 or point_diff == 0 or math.isnan(point_diff):
        return 1.0
    other = diffs <= 0 if point_diff > 0 else diffs >= 0
    return _r6(min(1.0, 2.0 * float(other.mean())))


def bootstrap_metric(episodes: Iterable[Mapping[str, Any]], metric_key: str, *, contrast_index: int = 0,
                     n_boot: int = N_BOOT, families: Iterable[str] | None = None) -> dict[str, Any]:
    """Point value and 95 percent cluster-bootstrap interval of one system's metric."""
    episodes = list(episodes)
    point = metric_value(episodes, metric_key, families)
    seed = BASE_SEED + int(contrast_index)
    if point["value"] == PENDING:
        return {**point, "interval": None, "B": n_boot, "seed": seed}
    arrays, _, _ = _family_arrays(episodes, metric_key, families)
    if not arrays:
        return {**point, "interval": None, "B": n_boot, "seed": seed, "nan_replicates": n_boot}
    reps = _replicates([arrays], np.random.default_rng(seed), n_boot)[0]
    return {**point, "interval": _interval(reps), "B": n_boot, "seed": seed,
            "nan_replicates": int(np.isnan(reps).sum())}


def paired_difference(episodes_a: Iterable[Mapping[str, Any]], episodes_b: Iterable[Mapping[str, Any]],
                      metric_key: str, *, contrast_index: int, n_boot: int = N_BOOT, mei: float | None = None,
                      families: Iterable[str] | None = None, n_ok: bool = True, min_n: int | None = None,
                      planned_n: int | None = None) -> dict[str, Any]:
    """Paired difference A - B of a metric on the identical episode set, with the outcome word.

    Both lists must hold the same episode keys (and families). ``families`` restricts to some
    families (a per-family result resamples within that family only). The outcome is
    "inconclusive" when ``n_ok`` is false, when n is below ``min_n`` or ``planned_n``, or when the
    interval includes 0 and its half-width exceeds ``mei``.
    """
    a, b = list(episodes_a), list(episodes_b)
    keys_a = {str(e.get("episode_key")): _family(e) for e in a}
    keys_b = {str(e.get("episode_key")): _family(e) for e in b}
    if len(keys_a) != len(a) or len(keys_b) != len(b):
        raise ValueError("paired_difference got the same episode twice")
    if keys_a != keys_b:
        raise ValueError("paired_difference needs both systems on the identical episode set "
                         f"(only in A: {sorted(set(keys_a) - set(keys_b))}, "
                         f"only in B: {sorted(set(keys_b) - set(keys_a))})")
    seed = BASE_SEED + int(contrast_index)
    va, vb = metric_value(a, metric_key, families), metric_value(b, metric_key, families)
    base = {"metric_key": metric_key, "contrast_index": int(contrast_index), "B": n_boot, "seed": seed,
            "mei": mei, "n_episodes": va["n_episodes"], "value_a": va["value"], "value_b": vb["value"]}
    n = va["n_episodes"]
    enough = n_ok and (min_n is None or n >= min_n) and (planned_n is None or n >= planned_n)
    if PENDING in (va["value"], vb["value"]):
        return {**base, "diff": PENDING, "interval": None, "p_boot": None, "outcome_word": PENDING,
                "nan_replicates": None, "n_ok": enough}
    arrays_a, _, _ = _family_arrays(a, metric_key, families)
    arrays_b, _, _ = _family_arrays(b, metric_key, families)
    point_a, point_b, compared = _point_paired(arrays_a, arrays_b) if arrays_a else (math.nan, math.nan, [])
    base["families_compared"] = compared
    if math.isnan(point_a) or math.isnan(point_b):
        return {**base, "diff": None, "interval": None, "p_boot": None, "outcome_word": INCONCLUSIVE,
                "nan_replicates": n_boot, "n_ok": enough}
    # A family where one system's denominator is 0 is left out for both (value_a and value_b keep it).
    diff = point_a - point_b
    rep_a, rep_b = _replicates([arrays_a, arrays_b], np.random.default_rng(seed), n_boot)
    diffs = rep_a - rep_b
    interval = _interval(diffs)
    return {
        **base,
        "diff": _r6(diff),
        "interval": interval,
        "p_boot": p_boot(diffs, diff),
        "outcome_word": outcome_word(interval, mei, n_ok=enough),
        "nan_replicates": int(np.isnan(diffs).sum()),
        "n_ok": enough,
    }


# --------------------------------------------------------------------------- #
# Outcome words and multiplicity
# --------------------------------------------------------------------------- #
def outcome_word(diff_interval: Sequence[float] | None, mei: float | None, n_ok: bool = True) -> str:
    """Spec 6.3: "clear difference" when the interval excludes 0; "no clear difference" when it
    includes 0 and its half-width is at most the MEI; else "inconclusive". Also "inconclusive"
    when ``n_ok`` is false (below the minimum n, or the planned n was not reached), when there is
    no interval, and when no MEI was declared and the interval includes 0."""
    if not n_ok or diff_interval is None or any(v is None for v in diff_interval):
        return INCONCLUSIVE
    lo, hi = float(diff_interval[0]), float(diff_interval[1])
    if lo > 0 or hi < 0:
        return CLEAR
    if mei is not None and (hi - lo) / 2.0 <= float(mei) + 1e-12:
        return NO_CLEAR
    return INCONCLUSIVE


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order (ties keep the input order)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (float(p_values[i]), i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p_values[i])))
        adjusted[i] = _r6(running)
    return adjusted


def holm_reject(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Which hypotheses Holm rejects at family-wise ``alpha``."""
    return [p <= alpha for p in holm(p_values)]


# --------------------------------------------------------------------------- #
# Calibration (C2)
# --------------------------------------------------------------------------- #
def _correct(value: Any) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in ("true", "1", "yes") else 0.0
    return 1.0 if value else 0.0


def _bins(confidences: Sequence[float], correct: Sequence[Any], bins: int) -> list[tuple[list[float], list[float]]]:
    """Equal-count bins by confidence (ties by input order) as (confidences, correctness) lists."""
    conf = [float(c) for c in confidences]
    corr = [_correct(c) for c in correct]
    if len(conf) != len(corr):
        raise ValueError(f"{len(conf)} confidences but {len(corr)} correctness values")
    n = len(conf)
    if n == 0:
        return []
    n_bins = int(bins) if n >= bins else max(1, n // 2)
    order = sorted(range(n), key=lambda i: (conf[i], i))
    parts = np.array_split(np.asarray(order, dtype=int), n_bins)
    return [([conf[i] for i in part.tolist()], [corr[i] for i in part.tolist()]) for part in parts if part.size]


def reliability_table(confidences: Sequence[float], correct: Sequence[Any], bins: int = ECE_BINS) -> list[dict[str, Any]]:
    """Equal-count bins by confidence: n, mean confidence and accuracy per bin.

    With fewer items than ``bins``, uses as many bins as allow at least 2 items per bin.
    """
    return [{"n": len(c), "mean_confidence": _r6(math.fsum(c) / len(c)), "accuracy": _r6(math.fsum(k) / len(k))}
            for c, k in _bins(confidences, correct, bins)]


def ece(confidences: Sequence[float], correct: Sequence[Any], bins: int = ECE_BINS) -> float | None:
    """Expected calibration error, ``sum_b (n_b / N) * |accuracy_b - mean_confidence_b|``, with
    equal-count bins (see :func:`reliability_table`). None for no items."""
    parts = _bins(confidences, correct, bins)
    if not parts:
        return None
    total = sum(len(c) for c, _ in parts)
    return _r6(math.fsum(len(c) / total * abs(math.fsum(k) / len(k) - math.fsum(c) / len(c)) for c, k in parts))
