"""The failure convention of SPEC_V1_1 section 4 for readers and metrics (S4, G2, gold v2, baselines).

v1.1 labels a failed attempt phase by phase:

- ``outcome`` is the phase's own result: in a missed grasp the approach that reached the object is
  ``success`` and the grasp is ``failed`` (``failure_type: missed_grasp``);
- ``attempt_outcome`` (``success``, ``failed``, ``aborted``) is the result of the whole attempt, copied
  onto every phase of it, so a consumer can still drop whole failed attempts;
- ``mistake`` is true only on the phase that failed;
- the attempt record keeps the whole span, from the first phase of the attempt to its last.

v7 outputs (and v1 to v6 ones) have no ``attempt_outcome``. Under the v7 rule every phase of a failed
attempt had ``outcome: failed``, so :func:`derive_attempt_outcome` fills a missing ``attempt_outcome``
from that rule and leaves present values alone.

S4 and G2 pick their rule per prediction (:func:`resolve_convention`): ``v11`` when a predicted segment
carries ``attempt_outcome``, else ``v7``, which reproduces the numbers the harness gave before v1.1.
Pure functions; inputs are never modified.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .temporal import as_span, seg_end, seg_start

ATTEMPT_OUTCOMES = ("success", "failed", "aborted")
CONVENTIONS = ("auto", "v7", "v11")
# attempt outcomes that make a predicted failed-attempt span for S4 (SPEC_V1_1 4: "attempt_outcome: failed")
S4_FAILED_OUTCOMES = frozenset({"failed"})
# attempt outcomes and phase outcomes that mark a failure for G2 part (i)
MARKED_OUTCOMES = frozenset({"failed", "aborted"})


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _missing(value: Any) -> bool:
    """None, NaN, pandas NA or NaT (parquet readers give them), and blank strings."""
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _low(value: Any) -> str | None:
    return None if _missing(value) else str(value).strip().lower()


def _flag(value: Any) -> bool:
    if _missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def attempt_key(seg: Mapping[str, Any]) -> int | None:
    """The segment's ``attempt_idx`` as an int (a digit string counts); None when it has none."""
    value = seg.get("attempt_idx")
    if _missing(value) or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def has_attempt_outcome(seg: Mapping[str, Any]) -> bool:
    """True when the segment carries an ``attempt_outcome`` (None, NaN and "" count as absent)."""
    return isinstance(seg, Mapping) and not _missing(seg.get("attempt_outcome"))


def _old_mark(seg: Mapping[str, Any]) -> str:
    """A phase's mark under the v7 rule: failed (outcome failed, or mistake true), aborted, or success."""
    outcome = _low(seg.get("outcome"))
    if outcome == "failed" or _flag(seg.get("mistake")):
        return "failed"
    return "aborted" if outcome == "aborted" else "success"


def _ordered(segments: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    usable = [(k, s) for k, s in enumerate(segments)
              if isinstance(s, Mapping) and seg_start(s) is not None and seg_end(s) is not None]
    usable.sort(key=lambda ks: (seg_start(ks[1]), seg_end(ks[1]), ks[0]))
    return [s for _, s in usable]


# --------------------------------------------------------------------------- #
# Deriving attempt_outcome
# --------------------------------------------------------------------------- #
def derive_attempt_outcome(segments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copies of ``segments`` with ``attempt_outcome`` filled where it is absent (SPEC_V1_1 4).

    This is the one rule of the package: ``robolabel.schema_v7.fill_attempt_outcome`` calls it. The v7 rule:
    the phases that share an ``attempt_idx`` form one attempt, which is ``failed`` when any of its phases
    failed, else ``aborted`` when any was aborted, else ``success``. Two readings extend it for outputs older
    than v7: a phase with ``mistake: true`` counts as failed (the old S4 rule counted it), and a segment
    without an ``attempt_idx`` is an attempt of its own. Present values (any non-empty value) are kept as
    they are. Non-mapping items are dropped.
    """
    segs = [dict(s) for s in segments if isinstance(s, Mapping)]
    marks: dict[int, set[str]] = {}
    for s in segs:
        key = attempt_key(s)
        if key is not None:
            marks.setdefault(key, set()).add(_old_mark(s))
    for s in segs:
        if has_attempt_outcome(s):
            continue
        key = attempt_key(s)
        found = marks[key] if key is not None else {_old_mark(s)}
        s["attempt_outcome"] = "failed" if "failed" in found else ("aborted" if "aborted" in found else "success")
    return segs


def attempt_outcome_of(seg: Mapping[str, Any]) -> str | None:
    """The segment's ``attempt_outcome``, lowercased (None when absent)."""
    return _low(seg.get("attempt_outcome"))


# --------------------------------------------------------------------------- #
# Which rule S4 and G2 use
# --------------------------------------------------------------------------- #
def resolve_convention(segments: Iterable[Any], convention: str = "auto") -> str:
    """``v11`` or ``v7`` for a prediction. ``auto`` is ``v11`` when any predicted segment carries an
    ``attempt_outcome`` (a v1.1 output), else ``v7`` (an older output, scored as before)."""
    if convention not in CONVENTIONS:
        raise ValueError(f"failure convention must be one of {CONVENTIONS}, not {convention!r}")
    if convention != "auto":
        return convention
    return "v11" if any(has_attempt_outcome(s) for s in segments if isinstance(s, Mapping)) else "v7"


# --------------------------------------------------------------------------- #
# G2 part (i): is a segment marked as part of a failure?
# --------------------------------------------------------------------------- #
def failure_marked(seg: Mapping[str, Any]) -> bool:
    """True when the segment marks a failure: its ``attempt_outcome`` or its own ``outcome`` is failed or
    aborted, or ``mistake`` is true. Read on segments that went through :func:`derive_attempt_outcome`."""
    return (attempt_outcome_of(seg) in MARKED_OUTCOMES or _low(seg.get("outcome")) in MARKED_OUTCOMES
            or _flag(seg.get("mistake")))


def inside_share(seg: Mapping[str, Any], span: Sequence[int] | None) -> float:
    """Share of the segment's frames that lie inside ``span`` (inclusive frames; 0.0 when either is missing)."""
    sp = as_span(seg)
    if sp is None or span is None:
        return 0.0
    length = sp[1] - sp[0] + 1
    if length <= 0:
        return 0.0
    inter = max(0, min(sp[1], span[1]) - max(sp[0], span[0]) + 1)
    return inter / length


# --------------------------------------------------------------------------- #
# S4: predicted failed-attempt spans
# --------------------------------------------------------------------------- #
def usable_attempt_records(attempts: Any) -> list[Mapping[str, Any]]:
    """The attempt records of a prediction that have a span (``start``/``end``, ``start_frame``/``end_frame``
    or ``span``); an empty list when there are none."""
    if not isinstance(attempts, (list, tuple)):
        return []
    return [a for a in attempts if isinstance(a, Mapping) and as_span(a) is not None]


def record_outcome(record: Mapping[str, Any]) -> str | None:
    """An attempt record's outcome (``outcome``, else ``attempt_outcome``), lowercased."""
    value = record.get("outcome")
    if _missing(value):
        value = record.get("attempt_outcome")
    return _low(value)


def failed_spans_from_attempts(attempts: Any) -> list[list[int]]:
    """``[start, end]`` of every attempt record whose outcome is failed, in time order."""
    spans = [list(as_span(a)) for a in usable_attempt_records(attempts) if record_outcome(a) in S4_FAILED_OUTCOMES]
    return sorted(spans)


def failed_spans_by_attempt(segments: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    """Merge consecutive phases (in time order) with the same ``attempt_idx`` and ``attempt_outcome`` failed.

    ``attempt_outcome`` is derived where it is absent (:func:`derive_attempt_outcome`). Two failed attempts
    in a row give two spans.
    """
    spans: list[list[int]] = []
    prev_failed, prev_key = False, None
    for seg in _ordered(derive_attempt_outcome(segments)):
        failed = attempt_outcome_of(seg) in S4_FAILED_OUTCOMES
        key = attempt_key(seg)
        if failed and prev_failed and key == prev_key:
            spans[-1][1] = max(spans[-1][1], seg_end(seg))
        elif failed:
            spans.append([seg_start(seg), seg_end(seg)])
        prev_failed, prev_key = failed, key
    return spans


def predicted_failed_spans_v11(segments: Sequence[Mapping[str, Any]], attempts: Any = None) -> tuple[list[list[int]], str]:
    """(spans, source) of S4 under SPEC_V1_1 4: from the attempt records when the prediction has any
    (source ``attempts``), else from the phases (source ``segments``, :func:`failed_spans_by_attempt`)."""
    if usable_attempt_records(attempts):
        return failed_spans_from_attempts(attempts), "attempts"
    return failed_spans_by_attempt(segments), "segments"


def attempt_record_for_span(attempts: Any, span: Sequence[int]) -> Mapping[str, Any] | None:
    """The failed attempt record whose span is ``span`` (the first one in time order), else None."""
    for a in sorted(usable_attempt_records(attempts), key=lambda r: as_span(r)):
        if record_outcome(a) in S4_FAILED_OUTCOMES and list(as_span(a)) == list(span):
            return a
    return None


# --------------------------------------------------------------------------- #
# Consistency (gold warnings)
# --------------------------------------------------------------------------- #
def failure_convention_warnings(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    """Where segments break the per-phase rule of SPEC_V1_1 4 (non-blocking, in a fixed order).

    - an attempt with more than one phase whose own outcome is failed (mark only the phase that failed);
    - a phase with outcome success that carries a failure type;
    - the phases of one attempt disagree on ``attempt_outcome``;
    - an ``attempt_outcome`` that is not one of success, failed, aborted;
    - an ``attempt_outcome`` that contradicts the phases (success while a phase failed or was aborted, or
      failed or aborted while no phase did).

    Segments are numbered by their position in the list.
    """
    segs = [s for s in segments if isinstance(s, Mapping)]
    warns: list[str] = []
    by: dict[int, list[int]] = {}
    for k, s in enumerate(segs):
        key = attempt_key(s)
        if key is not None:
            by.setdefault(key, []).append(k)
    for key in sorted(by):
        failed = [k for k in by[key] if _low(segs[k].get("outcome")) == "failed"]
        if len(failed) > 1:
            warns.append(f"attempt {key}: {len(failed)} phases have outcome failed (segments "
                         f"{', '.join(str(k) for k in failed)}); only the phase that failed gets outcome failed, "
                         "the others keep their own result (SPEC_V1_1 4)")
    for k, s in enumerate(segs):
        ftype = _low(s.get("failure_type"))
        if _low(s.get("outcome")) in (None, "success") and ftype not in (None, "none"):
            warns.append(f"segment {k}: failure_type {ftype} on a phase whose own outcome is success")
    derived = derive_attempt_outcome([{k: v for k, v in s.items() if k != "attempt_outcome"} for s in segs])
    for key in sorted(by):
        given = sorted({attempt_outcome_of(segs[k]) for k in by[key] if has_attempt_outcome(segs[k])})
        if not given:
            continue
        bad = [g for g in given if g not in ATTEMPT_OUTCOMES]
        if bad:
            warns.append(f"attempt {key}: attempt_outcome {', '.join(bad)} is not success, failed or aborted")
        if len(given) > 1:
            warns.append(f"attempt {key}: its phases disagree on attempt_outcome ({', '.join(given)}); "
                         "copy one value onto every phase of the attempt")
        from_phases = attempt_outcome_of(derived[by[key][0]])
        if len(given) == 1 and not bad and given[0] != from_phases:
            warns.append(f"attempt {key}: attempt_outcome {given[0]} but its phases say {from_phases}")
    return warns
