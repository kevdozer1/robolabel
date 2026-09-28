"""Object resolution (MEASUREMENT_SPEC 4.0) and the semantic metrics S1, S2 and S4 (spec 4.2).

Object resolution turns a predicted object reference (a name from a view record, with the
first-frame points and boxes of the arm's own inventory) into a gold object ID, by the first rule
that applies:

1. Points and boxes: a gold object matches when its first-frame point lies inside a predicted box,
   or within 5 percent of the image diagonal of a predicted point, in the same camera. Distances are
   in pixels, from ``camera_sizes`` (the unit square when a size is missing). Exactly one match
   resolves. Zero or several matches fall through to rule 2 (``RULE1_MISS_NEXT_RULE``).
2. String match: lowercase, strip articles and punctuation. A string equal to a name or alias of
   exactly one gold object resolves. Else a string whose content words are a subset of the words of
   exactly one gold object's names and aliases resolves.
3. Blind judge: everything else becomes a judge item and reads ``pending`` until an answer exists.
   The answer ``ambiguous``, or an ID that is not a gold object, counts as wrong.

A reference that names no object ("none", "nothing", empty) resolves to None with rule None. An
``unsure`` reference also resolves to None and so never equals a gold object.

S1, S2 and S4 return per-episode counts as ``{"numerator", "denominator", "pending"}``, where
``pending`` is the number of items still waiting for the judge; ``robolabel.eval.score`` sums them.
Pure functions; floats are rounded to 6 decimals.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .failure import attempt_record_for_span, predicted_failed_spans_v11, resolve_convention
from .lexicon import UNMAPPED, map_phase, normalize_name, words
from .temporal import as_span, failed_spans_from_segments, match_spans, seg_end, seg_start, span_iou, t3_pairs

DECIMALS = 6
RULE1_DIAGONAL_SHARE = 0.05
# Where zero or several rule-1 matches go: 2 (string match, then the judge) or 3 (the judge directly).
RULE1_MISS_NEXT_RULE = 2
S1_SEG_MIN_IOU = 0.5
S2_MIN_IOU = 0.2
S4_MIN_IOU = 0.3
RESOLVED = "resolved"
PENDING = "pending"
AMBIGUOUS = "ambiguous"
JUDGE_OBJECT = "object"
_EPS = 1e-9
_UNCOVERED = object()  # frame class placeholder for frames no segment covers

# References that name no object ("holding nothing" has ref_object none). Compared after
# lowercasing and joining words with single spaces, so "n/a" reads "n a".
NONE_WORDS = frozenset({"", "none", "null", "nothing", "no object", "n a", "na"})
# References that name an object without saying which one; they never resolve.
UNSURE_WORDS = frozenset({"unsure", "unknown", "not sure", "unclear"})
# Function words dropped before the content-word subset test of rule 2 (articles are gone already).
FUNCTION_WORDS = frozenset({
    "of", "and", "or", "to", "with", "in", "on", "at", "by", "for", "from", "into", "onto", "is",
    "it", "its", "this", "that", "these", "those", "their",
})

Resolve = Callable[[Any], dict[str, Any]]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _r6(value: float | None) -> float | None:
    return None if value is None else round(float(value), DECIMALS)


def counts(numerator: float, denominator: int, pending: int = 0) -> dict[str, Any]:
    """The per-episode count record every metric returns."""
    return {"numerator": numerator, "denominator": int(denominator), "pending": int(pending)}


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _plain(text: Any) -> str:
    """Lowercase words joined by single spaces (articles kept)."""
    return " ".join(words(str(text)))


def _low(value: Any) -> str | None:
    """Lowercase stripped string; None, NaN, "", "none" and "null" become None."""
    if value is None or _is_nan(value):
        return None
    text = str(value).strip().lower()
    return None if text in ("", "none", "null") else text


def _flag(value: Any) -> bool:
    if value is None or _is_nan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _get(item: Mapping[str, Any], *keys: str) -> Any:
    """Value of the first key present in ``item`` (its value may be None)."""
    for key in keys:
        if key in item:
            return item[key]
    return None


def is_no_reference(name: Any) -> bool:
    """True for None, NaN and names that refer to no object ("none", "nothing", "n/a", "the")."""
    if name is None or _is_nan(name) or isinstance(name, bool):
        return True
    return _plain(name) in NONE_WORDS or normalize_name(str(name)) == ""


def is_unsure_reference(name: Any) -> bool:
    """True for "unsure", "unknown" and the like."""
    return not is_no_reference(name) and _plain(name) in UNSURE_WORDS


def judge_id(kind: str, episode_key: str | None, text: str) -> str:
    """Stable, arm-free ID of a judge item: ``<kind>|<episode_key>|<normalized text>``."""
    return f"{kind}|{episode_key or ''}|{text}"


# --------------------------------------------------------------------------- #
# Object resolution (spec 4.0)
# --------------------------------------------------------------------------- #
def gold_names(obj: Mapping[str, Any]) -> list[str]:
    """A gold object's name and aliases (non-empty strings)."""
    aliases = obj.get("aliases")
    out = [obj.get("name"), *(aliases if isinstance(aliases, list) else [])]
    return [str(n) for n in out if isinstance(n, str) and n.strip()]


def _same_camera(a: Any, b: Any) -> bool:
    """Same camera: equal names, or equal last dotted parts ("up" and "observation.images.up")."""
    if a is None or b is None:
        return False
    a, b = str(a), str(b)
    return a == b or a.rsplit(".", 1)[-1] == b.rsplit(".", 1)[-1]


def _camera_size(camera_sizes: Any, camera: str) -> tuple[float, float]:
    """(width, height) in pixels for ``camera``; (1, 1) when unknown."""
    if isinstance(camera_sizes, Mapping):
        size = camera_sizes.get(camera)
        if size is None:
            for key in sorted(camera_sizes, key=str):
                if _same_camera(key, camera):
                    size = camera_sizes[key]
                    break
        if isinstance(size, Sequence) and not isinstance(size, str) and len(size) == 2:
            try:
                w, h = float(size[0]), float(size[1])
            except (TypeError, ValueError):
                return 1.0, 1.0
            if w > 0 and h > 0:
                return w, h
    return 1.0, 1.0


def _xy(item: Mapping[str, Any]) -> tuple[float, float] | None:
    xy = item.get("xy")
    if isinstance(xy, Sequence) and not isinstance(xy, str) and len(xy) == 2:
        x, y = xy
    else:
        x, y = item.get("x"), item.get("y")
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _gold_point(obj: Mapping[str, Any]) -> tuple[str, float, float] | None:
    point = obj.get("first_frame_point")
    if not isinstance(point, Mapping) or point.get("camera") is None:
        return None
    xy = _xy(point)
    return None if xy is None else (str(point["camera"]), xy[0], xy[1])


def _pred_points(points: Any) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for p in points or []:
        if isinstance(p, Mapping) and p.get("camera") is not None:
            xy = _xy(p)
            if xy is not None:
                out.append((str(p["camera"]), xy[0], xy[1]))
    return out


def _pred_boxes(boxes: Any) -> list[tuple[str, float, float, float, float]]:
    out: list[tuple[str, float, float, float, float]] = []
    for b in boxes or []:
        if not isinstance(b, Mapping) or b.get("camera") is None:
            continue
        vals = b.get("box") if b.get("box") is not None else [b.get(k) for k in ("x0", "y0", "x1", "y1")]
        try:
            x0, y0, x1, y1 = (float(v) for v in vals)
        except (TypeError, ValueError):
            continue
        out.append((str(b["camera"]), min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return out


def rule1_candidates(pred_points: Any, pred_boxes: Any, gold_objects: Iterable[Mapping[str, Any]],
                     camera_sizes: Any) -> list[str]:
    """Gold object IDs whose first-frame point is inside a predicted box or near a predicted point.

    Near means within ``RULE1_DIAGONAL_SHARE`` of the image diagonal, measured in pixels. Only
    points and boxes in the gold point's camera count. IDs come back in gold order.
    """
    points, boxes = _pred_points(pred_points), _pred_boxes(pred_boxes)
    out: list[str] = []
    for obj in gold_objects:
        oid, gp = obj.get("object_id"), _gold_point(obj)
        if not oid or gp is None or oid in out:
            continue
        cam, gx, gy = gp
        hit = any(_same_camera(c, cam) and x0 - _EPS <= gx <= x1 + _EPS and y0 - _EPS <= gy <= y1 + _EPS
                  for c, x0, y0, x1, y1 in boxes)
        if not hit:
            w, h = _camera_size(camera_sizes, cam)
            limit = RULE1_DIAGONAL_SHARE * math.hypot(w, h)
            hit = any(_same_camera(c, cam) and math.hypot((gx - x) * w, (gy - y) * h) <= limit + _EPS
                      for c, x, y in points)
        if hit:
            out.append(str(oid))
    return out


def rule2_candidates(pred_name: Any, gold_objects: Iterable[Mapping[str, Any]]) -> list[str]:
    """Gold object IDs that rule 2 finds for ``pred_name``, in gold order.

    First the objects with a name or alias equal to the normalized string; if that is not exactly
    one, the objects whose name and alias words contain every content word of the string.
    """
    key = normalize_name(str(pred_name)) if pred_name is not None else ""
    if not key:
        return []
    objs = [o for o in gold_objects if o.get("object_id")]
    exact = [str(o["object_id"]) for o in objs if key in {normalize_name(n) for n in gold_names(o)}]
    if len(exact) == 1:
        return exact
    content = set(key.split()) - FUNCTION_WORDS
    if not content:
        return exact
    return [str(o["object_id"]) for o in objs
            if content <= {w for n in gold_names(o) for w in words(n)}]


def _judge_answer(judge_answers: Any, raw: Any, key: str) -> Any:
    if not isinstance(judge_answers, Mapping):
        return None
    for candidate in (key, str(raw), str(raw).strip().lower()):
        if candidate in judge_answers and judge_answers[candidate] is not None:
            return judge_answers[candidate]
    return None


def resolve_object(pred_name: Any, pred_points: Any, pred_boxes: Any,
                   gold_objects: Iterable[Mapping[str, Any]] | None, camera_sizes: Any,
                   judge_answers: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve one predicted object reference to a gold object ID (spec 4.0 rules 1 to 3).

    Returns ``{"object_id", "rule", "status", "candidates", "judge_key", "reason"}``. ``rule`` is 1,
    2 or 3 (None for a reference that names no object). ``status`` is ``resolved`` or ``pending``
    (rule 3 without a judge answer). ``judge_answers`` maps the normalized name (``judge_key``) to
    one gold object ID or ``ambiguous``; ``ambiguous`` resolves to None, which counts as wrong.
    """
    golds = [o for o in (gold_objects or []) if isinstance(o, Mapping) and o.get("object_id")]
    out: dict[str, Any] = {"object_id": None, "rule": None, "status": RESOLVED, "candidates": [],
                           "judge_key": None, "reason": ""}
    if is_no_reference(pred_name):
        out["reason"] = "no reference"
        return out
    if is_unsure_reference(pred_name):
        out["reason"] = "unsure reference"
        return out
    key = normalize_name(str(pred_name))
    found: list[str] = []
    go_rule2 = True
    if _pred_points(pred_points) or _pred_boxes(pred_boxes):
        found = rule1_candidates(pred_points, pred_boxes, golds, camera_sizes)
        if len(found) == 1:
            out.update(object_id=found[0], rule=1, candidates=found, reason="point or box")
            return out
        go_rule2 = RULE1_MISS_NEXT_RULE == 2
    if go_rule2:
        by_name = rule2_candidates(pred_name, golds)
        if len(by_name) == 1:
            out.update(object_id=by_name[0], rule=2, candidates=by_name, reason="name")
            return out
        found = [str(o["object_id"]) for o in golds if o["object_id"] in set(found) | set(by_name)]
    out.update(rule=3, candidates=found, judge_key=key)
    answer = _judge_answer(judge_answers, pred_name, key)
    if answer is None:
        out.update(status=PENDING, reason="waiting for the blind judge")
        return out
    text = str(answer).strip()
    if text in {str(o["object_id"]) for o in golds}:
        out.update(object_id=text, reason="judge")
    elif text.lower() == AMBIGUOUS:
        out["reason"] = "judge: ambiguous"
    else:
        out["reason"] = f"judge answer {text!r} is not a gold object"
    return out


class ObjectResolver:
    """Resolves the object names of one view record against one gold episode.

    A name is looked up in the view's own inventory (by name, or by its object_id) to find the
    points and boxes for rule 1. Results are cached per name. Every rule-3 item is kept for the
    judge queue, keyed by :func:`judge_id` so that the same name in two arms is one judge item.
    """

    def __init__(self, gold_objects: Iterable[Mapping[str, Any]] | None,
                 view_objects: Iterable[Mapping[str, Any]] | None = None, camera_sizes: Any = None,
                 judge_answers: Mapping[str, Any] | None = None, episode_key: str | None = None):
        self.gold_objects = [o for o in (gold_objects or []) if isinstance(o, Mapping) and o.get("object_id")]
        self.gold_ids = [str(o["object_id"]) for o in self.gold_objects]
        self.camera_sizes = camera_sizes if isinstance(camera_sizes, Mapping) else {}
        self.judge_answers = judge_answers if isinstance(judge_answers, Mapping) else {}
        self.episode_key = episode_key
        self._by_name: dict[str, Mapping[str, Any]] = {}
        self._by_id: dict[str, Mapping[str, Any]] = {}
        for obj in view_objects or []:
            if not isinstance(obj, Mapping):
                continue
            name_key = normalize_name(str(obj.get("name") or ""))
            if name_key:
                self._by_name.setdefault(name_key, obj)
            if obj.get("object_id"):
                self._by_id.setdefault(str(obj["object_id"]), obj)
        self._cache: dict[str, dict[str, Any]] = {}
        self._queue: dict[str, dict[str, Any]] = {}

    def view_object(self, name: Any) -> Mapping[str, Any] | None:
        """The view inventory object for ``name`` (its object_id or its normalized name), else None."""
        if name is None or is_no_reference(name):
            return None
        text = str(name).strip()
        return self._by_id.get(text) or self._by_name.get(normalize_name(text))

    @staticmethod
    def _plain_name(obj: Mapping[str, Any]) -> str:
        """The inventory name without the " (oN)" suffix the scene layer adds to duplicate names.

        The suffix is the arm's own inventory ID, not part of the predicted string; left in, it
        would block rule 2 and show the judge an ID that looks like a gold ID.
        """
        name = str(obj.get("name") or "").strip()
        suffix = f"({obj.get('object_id')})"
        if obj.get("object_id") and name.endswith(suffix) and name[:-len(suffix)].strip():
            return name[:-len(suffix)].strip()
        return name

    def __call__(self, name: Any) -> dict[str, Any]:
        cache_key = "" if name is None or _is_nan(name) else str(name)
        if cache_key in self._cache:
            return dict(self._cache[cache_key])
        obj = self.view_object(name)
        text = self._plain_name(obj) if obj is not None and obj.get("name") else name
        points = obj.get("points") if obj is not None else None
        boxes = obj.get("boxes") if obj is not None else None
        res = resolve_object(text, points, boxes, self.gold_objects, self.camera_sizes, self.judge_answers)
        if res["rule"] == 3:
            jid = judge_id(JUDGE_OBJECT, self.episode_key, res["judge_key"])
            self._queue.setdefault(jid, {
                "judge_id": jid,
                "type": "object_resolution",
                "episode_key": self.episode_key,
                "text": str(text),
                "gold_object_ids": list(self.gold_ids),
                "question": "Which gold object does this name refer to? Answer one object_id or ambiguous.",
                "answered": res["status"] == RESOLVED,
            })
        self._cache[cache_key] = res
        return dict(res)

    @property
    def judge_queue(self) -> list[dict[str, Any]]:
        """Rule-3 items seen so far, sorted by judge_id."""
        return [dict(self._queue[k]) for k in sorted(self._queue)]


# --------------------------------------------------------------------------- #
# S1: phase accuracy
# --------------------------------------------------------------------------- #
def _missing_phase(value: Any) -> bool:
    return _low(value) in (None, "unsure", "unknown")


def predicted_phase_class(seg: Mapping[str, Any]) -> str:
    """A predicted segment's class: its ``phase_class`` mapped by the lexicon, or its
    ``phase_text`` mapped when the class is missing (``unmapped`` when neither maps)."""
    phase = _get(seg, "phase_class", "phase")
    if not _missing_phase(phase):
        return map_phase(str(phase))
    return map_phase(_get(seg, "phase_text", "text"))


def gold_phase_class(seg: Mapping[str, Any]) -> str | None:
    """A gold segment's class, or None when the annotator left it empty."""
    phase = seg.get("phase_class")
    if _missing_phase(phase):
        return None
    cls = map_phase(str(phase))
    return None if cls == UNMAPPED else cls


def _frame_classes(segments: Sequence[Mapping[str, Any]], num_frames: int,
                   class_of: Callable[[Mapping[str, Any]], str | None]) -> list[Any]:
    """Class per frame; the first segment (by start) that covers a frame wins."""
    out: list[Any] = [_UNCOVERED] * num_frames
    ordered = sorted((s for s in segments if seg_start(s) is not None and seg_end(s) is not None),
                     key=lambda s: (seg_start(s), seg_end(s)))
    for seg in ordered:
        cls = class_of(seg)
        for f in range(max(0, seg_start(seg)), min(num_frames - 1, seg_end(seg)) + 1):
            if out[f] is _UNCOVERED:
                out[f] = cls
    return out


def s1_episode(pred_segments: Sequence[Mapping[str, Any]], gold_segments: Sequence[Mapping[str, Any]],
               num_frames: int) -> dict[str, Any]:
    """S1 counts for one episode.

    ``S1``: frames whose predicted class equals the gold class, over frames with a gold class.
    Frames with an unmapped predicted phase, or not covered by any predicted segment, are wrong.
    ``S1-seg``: T3 matched pairs (``temporal.t3_pairs``) with IoU at least 0.5 and a gold class,
    and among them the pairs with equal class. ``S1-unmapped``: covered frames whose predicted
    phase does not map, over covered frames.
    """
    n = int(num_frames)
    pred = _frame_classes(pred_segments, n, predicted_phase_class)
    gold = _frame_classes(gold_segments, n, gold_phase_class)
    correct = scored = unmapped = covered = gold_unlabeled = 0
    for p, g in zip(pred, gold, strict=True):
        if p is not _UNCOVERED:
            covered += 1
            unmapped += int(p == UNMAPPED)
        if g is _UNCOVERED or g is None:
            gold_unlabeled += 1
            continue
        scored += 1
        correct += int(p is not _UNCOVERED and p != UNMAPPED and p == g)
    pairs = t3_pairs(pred_segments, gold_segments)
    seg_num = seg_den = 0
    seg_pairs: list[dict[str, Any]] = []
    for i, j, iou in pairs:
        if iou < S1_SEG_MIN_IOU:
            continue
        g_cls = gold_phase_class(gold_segments[j])
        if g_cls is None:
            continue
        p_cls = predicted_phase_class(pred_segments[i])
        seg_den += 1
        seg_num += int(p_cls == g_cls)
        seg_pairs.append({"pred_idx": i, "gold_idx": j, "iou": iou, "pred_class": p_cls, "gold_class": g_cls})
    return {
        "S1": counts(correct, scored),
        "S1-seg": counts(seg_num, seg_den),
        "S1-unmapped": counts(unmapped, covered),
        "frames": n,
        "uncovered_frames": n - covered,
        "gold_unlabeled_frames": gold_unlabeled,
        "seg_pairs": seg_pairs,
    }


# --------------------------------------------------------------------------- #
# S2: target and destination accuracy
# --------------------------------------------------------------------------- #
def best_counterpart(pred_segments: Sequence[Mapping[str, Any]],
                     gold_segment: Mapping[str, Any]) -> tuple[int | None, float]:
    """(index, IoU) of the predicted segment with the largest temporal IoU; ties go to the lowest index."""
    gold_span = as_span(gold_segment)
    best_i: int | None = None
    best = 0.0
    for i, seg in enumerate(pred_segments):
        iou = span_iou(as_span(seg), gold_span)
        if iou > best:
            best_i, best = i, iou
    return best_i, best


def compare_reference(pred_name: Any, gold_ref: Any, resolve: Resolve,
                      gold_ids: Iterable[str]) -> tuple[bool | None, dict[str, Any]]:
    """(correct or None when pending, resolution) for one predicted reference against a gold one.

    A gold reference that is an object ID is compared with the resolved ID. A gold reference that
    is a region name (destinations and ``ref_object`` may name a region) is compared as a
    normalized string, without the judge.
    """
    if str(gold_ref) in set(gold_ids):
        res = resolve(pred_name)
        if res["status"] == PENDING:
            return None, res
        return res["object_id"] == str(gold_ref), res
    same = not is_no_reference(pred_name) and normalize_name(str(pred_name)) == normalize_name(str(gold_ref))
    return same, {"object_id": None, "rule": "region", "status": RESOLVED, "candidates": [],
                  "judge_key": None, "reason": "gold names a region; compared as text"}


def s2_episode(pred_segments: Sequence[Mapping[str, Any]], gold_segments: Sequence[Mapping[str, Any]],
               resolve: Resolve, gold_ids: Iterable[str], *, destination: bool = False) -> dict[str, Any]:
    """S2 (targets) or S2-dest (destinations on release-class gold segments) for one episode.

    For each gold segment with a non-null target (destination), the predicted segment with the
    largest temporal IoU is its counterpart. IoU below 0.2 is wrong ("no counterpart"). Else the
    predicted ``target_name`` (``destination_name``) is resolved and compared. Items whose
    resolution waits for the judge are counted in ``pending``.
    """
    ids = [str(i) for i in gold_ids]
    gold_field = "destination" if destination else "target"
    pred_fields = ("destination_name", "destination") if destination else ("target_name", "target")
    items: list[dict[str, Any]] = []
    num = den = pending = 0
    for j, gseg in enumerate(gold_segments):
        gold_ref = gseg.get(gold_field)
        if gold_ref is None or is_no_reference(gold_ref):
            continue
        if destination and gold_phase_class(gseg) != "release":
            continue
        den += 1
        i, iou = best_counterpart(pred_segments, gseg)
        item: dict[str, Any] = {"gold_idx": j, "gold": gold_ref, "pred_idx": i, "iou": _r6(iou)}
        if i is None or iou < S2_MIN_IOU:
            item.update(pred_name=None, resolved=None, rule=None, correct=False, reason="no counterpart")
        else:
            name = _get(pred_segments[i], *pred_fields)
            ok, res = compare_reference(name, gold_ref, resolve, ids)
            item.update(pred_name=name, resolved=res["object_id"], rule=res["rule"], correct=ok,
                        reason=res["reason"])
            if ok is None:
                pending += 1
            else:
                num += int(ok)
        items.append(item)
    return {"counts": counts(num, den, pending), "items": items}


# --------------------------------------------------------------------------- #
# S4: failed-attempt detection
# --------------------------------------------------------------------------- #
def _marked_failed(seg: Mapping[str, Any]) -> bool:
    return _low(seg.get("outcome")) == "failed" or _flag(seg.get("mistake"))


def predicted_failure_type(pred_segments: Sequence[Mapping[str, Any]], span: Sequence[int]) -> str | None:
    """First failure type (not none) among the failed segments inside ``span``, in time order."""
    ordered = sorted((s for s in pred_segments if seg_start(s) is not None and seg_end(s) is not None),
                     key=lambda s: (seg_start(s), seg_end(s)))
    for seg in ordered:
        if seg_start(seg) >= span[0] and seg_end(seg) <= span[1] and _marked_failed(seg):
            ftype = _low(seg.get("failure_type"))
            if ftype is not None:
                return ftype
    return None


def _record_failure_type(record: Mapping[str, Any] | None) -> str | None:
    ftype = _low(record.get("failure_type")) if record is not None else None
    return None if ftype in (None, "none") else ftype


def s4_episode(pred_segments: Sequence[Mapping[str, Any]],
               gold_failed_attempts: Sequence[Mapping[str, Any]], min_iou: float = S4_MIN_IOU, *,
               pred_attempts: Any = None, convention: str = "auto") -> dict[str, Any]:
    """S4 counts for one episode.

    Predicted failed spans, by the failure convention of the prediction (``failure.resolve_convention``:
    ``auto`` reads ``v11`` when a predicted segment carries ``attempt_outcome``, else ``v7``):

    - ``v7`` (outputs older than v1.1, scored as before): merge consecutive segments with ``outcome``
      failed or ``mistake`` true (``temporal.failed_spans_from_segments``); ``pred_attempts`` is not read.
    - ``v11`` (SPEC_V1_1 4): the failed attempt records of ``pred_attempts`` (the view's ``attempts``, each
      the whole attempt), or else consecutive phases with the same ``attempt_idx`` and ``attempt_outcome``
      failed (derived from the v7 rule where absent).

    They are matched one to one to the gold ``failed_attempts`` spans at IoU >= 0.3, maximizing total
    IoU (``temporal.match_spans``). Span F1 is ``2M / (m + n)``. Episode level: "has at least one failed
    attempt", precision over episodes that predict one, recall over episodes whose gold has one.
    S4-type compares the failure type on matched spans whose gold type is set; a v11 attempt record gives
    its own ``failure_type``, else the first failed phase inside the span does.
    """
    rule = resolve_convention(pred_segments, convention)
    if rule == "v7":
        pred_spans, source = failed_spans_from_segments(pred_segments), "segments"
    else:
        pred_spans, source = predicted_failed_spans_v11(pred_segments, pred_attempts)
    gold = [fa for fa in gold_failed_attempts if isinstance(fa, Mapping)]
    gold_spans = [as_span(fa) for fa in gold]
    pairs = match_spans(pred_spans, gold_spans, min_iou)
    type_num = type_den = 0
    typed: list[dict[str, Any]] = []
    for i, j, iou in pairs:
        g_type = _low(gold[j].get("failure_type"))
        p_type = None
        if source == "attempts":
            p_type = _record_failure_type(attempt_record_for_span(pred_attempts, pred_spans[i]))
        if p_type is None:
            p_type = predicted_failure_type(pred_segments, pred_spans[i])
        typed.append({"pred_idx": i, "gold_idx": j, "iou": iou, "pred_type": p_type, "gold_type": g_type})
        if g_type is None:
            continue
        type_den += 1
        type_num += int(p_type == g_type)
    m, n, matched = len(pred_spans), len(gold_spans), len(pairs)
    both = int(m > 0 and n > 0)
    return {
        "S4-P": counts(matched, m),
        "S4-R": counts(matched, n),
        "S4-F1": counts(2 * matched, m + n),
        "S4-ep-P": counts(both, int(m > 0)),
        "S4-ep-R": counts(both, int(n > 0)),
        "S4-type": counts(type_num, type_den),
        "pred_spans": [list(s) for s in pred_spans],
        "gold_spans": [None if s is None else list(s) for s in gold_spans],
        "pairs": typed,
        "convention": rule,
        "pred_span_source": source,
    }
