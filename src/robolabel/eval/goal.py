"""Goal metrics G1 to G7 and requirement matching (MEASUREMENT_SPEC 4.3).

A predicted requirement (view record, object names) matches a gold requirement (gold v2, object
IDs) when kind, resolved object and ``ref_object`` IDs, predicate class (Appendix B, through
``map_predicate``) and value are all equal. An ``other`` predicate also needs the same normalized
text. The canonical holding form compares ``ref_object`` including null: "none" and null are the
same, so "holding nothing" is ``holding, none, false`` on both sides. Each gold item matches at most
one predicted item: a first pass pairs items with equal status (gold items in order, each taking the
first free predicted item), a second pass pairs the rest by order.

Names go through :class:`robolabel.eval.semantic.ObjectResolver`. A name that waits for the blind
judge makes every matching-based count of the episode ``pending`` (spec 10.2: never computed on
partial answers). G6 is still false when a condition is already false without the judge: a wrong
or missing primary target, a copied failed attempt (G2 part i), a required predicted item that
cannot match any gold item, or a gold ending item that no predicted item could match.

G2 part (ii) (a requirement about something that exists only inside a gold failed span) needs the
blind judge. Each predicted requirement of an episode with a gold failed attempt becomes a judge
item ``g2|<episode_key>|<normalized requirement text>``; the answer is yes (copied) or no.

Every metric returns ``{"numerator", "denominator", "pending"}``. Pure functions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .lexicon import OTHER, map_predicate, normalize_name, words
from .semantic import (
    PENDING,
    Resolve,
    _flag,
    _get,
    _is_nan,
    _low,
    counts,
    is_no_reference,
    is_unsure_reference,
    judge_id,
)
from .temporal import as_span, span_iou

REQUIRED, INCIDENTAL, UNSURE = "required", "incidental", "unsure"
UNSURE_KINDS = ("perception", "intent")
G5_CLASSES = ("none", "perception", "intent")
ENDING_KINDS = frozenset({"object_end_state", "robot_end_state"})
PENDING_REF = "<pending>"
UNRESOLVED_REF = "<unresolved>"
REGION_PREFIX = "region:"
G2_SEGMENT_MIN_IOU = 0.5
JUDGE_G2 = "g2"
# Names a robot end-state item may give its subject; they mean "no object" there. A subject made
# only of ROBOT_PART_WORDS ("robot's gripper", "gripper fingers") counts too.
ROBOT_WORDS = frozenset({"robot", "gripper", "arm", "robot arm", "robot gripper", "end effector", "hand"})
ROBOT_PART_WORDS = frozenset({
    "robot", "robots", "gripper", "grippers", "arm", "arms", "hand", "end", "effector", "finger", "fingers",
    "jaw", "jaws", "claw", "s", "its", "own",
})
YES_ANSWERS = frozenset({"yes", "true", "copied", "1"})
NO_ANSWERS = frozenset({"no", "false", "not copied", "0"})
# Words that make a coarse text narrate an attempt or its failure (spec 4.3 G2 part i and 3.4.3:
# coarse text states the intended instruction, never the failure). "drop" is left out on purpose,
# because "drop the cube in the box" is a placement.
FAILURE_NARRATIVE_WORDS = frozenset({
    "try", "tries", "tried", "trying", "attempt", "attempts", "attempted", "attempting",
    "again", "twice", "retry", "retries", "retried", "retrying", "re", "regrasp", "regrasps",
    "regrasped", "regrasping", "miss", "misses", "missed", "missing", "fail", "fails", "failed",
    "failing", "failure", "slip", "slips", "slipped", "slipping", "lose", "loses", "lost", "losing",
    "unsuccessful", "unsuccessfully", "mistake",
})


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def norm_value(value: Any) -> Any:
    """True, False, None (unknown) or a normalized string (for ``state``)."""
    if isinstance(value, bool):
        return value
    if value is None or _is_nan(value):
        return None
    text = str(value).strip().lower()
    if text in ("true", "yes"):
        return True
    if text in ("false", "no"):
        return False
    if text in ("", "none", "null", "unknown", "unsure"):
        return None
    return " ".join(words(text))


def norm_achieved(value: Any) -> Any:
    """True, False or "unknown"."""
    value = norm_value(value)
    return value if isinstance(value, bool) else "unknown"


def _unsure_kind(value: Any) -> str | None:
    kind = _low(value)
    return kind if kind in UNSURE_KINDS else None


def gold_ref_key(ref: Any, gold_ids: Iterable[str]) -> str | None:
    """A gold reference as a match key: the object ID, ``region:<text>`` for a region, or None."""
    if ref is None or is_no_reference(ref):
        return None
    text = str(ref).strip()
    if text in set(gold_ids):
        return text
    return REGION_PREFIX + normalize_name(text)


def pred_ref_key(name: Any, resolve: Resolve, gold_regions: Iterable[str],
                 *, robot_subject: bool = False) -> tuple[str | None, dict[str, Any] | None]:
    """A predicted reference as a match key, with its resolution.

    None for "none"; ``region:<text>`` when the name equals a gold region; the resolved object ID;
    ``<pending>`` while the judge has not answered; ``<unresolved>`` for "unsure", "ambiguous" and
    other names that resolve to no gold object (they never match).
    """
    if is_no_reference(name):
        return None, None
    plain = normalize_name(str(name))
    if robot_subject and (plain in ROBOT_WORDS or set(plain.split()) <= ROBOT_PART_WORDS):
        return None, None
    if is_unsure_reference(name):
        return UNRESOLVED_REF, None
    if plain in set(gold_regions):
        return REGION_PREFIX + plain, None
    res = resolve(name)
    if res["status"] == PENDING:
        return PENDING_REF, res
    if res["object_id"] is None:
        return UNRESOLVED_REF, res
    return str(res["object_id"]), res


def gold_requirements(requirements: Sequence[Any], gold_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Gold v2 requirements as match items (in gold order)."""
    ids = [str(i) for i in gold_ids]
    out: list[dict[str, Any]] = []
    for k, req in enumerate(requirements):
        if not isinstance(req, Mapping):
            continue
        predicate = map_predicate(req.get("predicate"))
        out.append({
            "position": k,
            "req_id": req.get("req_id"),
            "kind": _low(req.get("kind")),
            "object": gold_ref_key(req.get("object"), ids),
            "ref": gold_ref_key(req.get("ref_object"), ids),
            "predicate": predicate,
            "value": norm_value(req.get("value")),
            "other_text": normalize_name(str(req.get("text") or "")) if predicate == OTHER else "",
            "status": _low(req.get("status")),
            "unsure_kind": _unsure_kind(req.get("unsure_kind")),
            "achieved": norm_achieved(req.get("achieved")),
        })
    return out


def requirement_text(req: Mapping[str, Any]) -> str:
    """Normalized text of a predicted requirement: its ``text``, else predicate, names and value."""
    text = req.get("text")
    if isinstance(text, str) and normalize_name(text):
        return normalize_name(text)
    parts = [_get(req, "predicate"), _get(req, "object_name", "object"), _get(req, "ref_name", "ref_object"),
             _get(req, "value")]
    return normalize_name(" ".join(str(p) for p in parts if p is not None))


def pred_requirements(requirements: Sequence[Any], resolve: Resolve,
                      gold_regions: Iterable[str]) -> list[dict[str, Any]]:
    """View-record requirements as match items (in predicted order)."""
    regions = sorted(set(gold_regions))
    out: list[dict[str, Any]] = []
    for k, req in enumerate(requirements):
        if not isinstance(req, Mapping):
            continue
        kind = _low(req.get("kind"))
        obj, obj_res = pred_ref_key(_get(req, "object_name", "object"), resolve, regions,
                                    robot_subject=kind == "robot_end_state")
        ref, ref_res = pred_ref_key(_get(req, "ref_name", "ref_object"), resolve, regions)
        predicate = map_predicate(req.get("predicate"))
        out.append({
            "position": k,
            "kind": kind,
            "object": obj,
            "ref": ref,
            "predicate": predicate,
            "value": norm_value(req.get("value")),
            "other_text": normalize_name(str(req.get("text") or "")) if predicate == OTHER else "",
            "status": _low(req.get("status")),
            "unsure_kind": _unsure_kind(req.get("unsure_kind")),
            "achieved": norm_achieved(req.get("achieved")),
            "text": requirement_text(req),
            "object_rule": None if obj_res is None else obj_res["rule"],
            "ref_rule": None if ref_res is None else ref_res["rule"],
        })
    return out


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
_KEY_FIELDS = ("kind", "object", "ref", "predicate", "value", "other_text")


def is_pending_item(item: Mapping[str, Any]) -> bool:
    return PENDING_REF in (item.get("object"), item.get("ref"))


def items_equal(pred: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    """Kind, object, ref, predicate, value (and ``other`` text) equal; never true while pending."""
    if is_pending_item(pred):
        return False
    return all(pred.get(f) == gold.get(f) for f in _KEY_FIELDS)


def could_match(pred: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    """Whether ``pred`` could equal ``gold`` once its pending names are resolved.

    A pending name resolves to a gold object ID or to nothing, so it can only equal a gold object ID.
    """
    for field in _KEY_FIELDS:
        p, g = pred.get(field), gold.get(field)
        if p == PENDING_REF and field in ("object", "ref"):
            if g is None or str(g).startswith(REGION_PREFIX):
                return False
            continue
        if p != g:
            return False
    return True


def match_requirements(pred: Sequence[Mapping[str, Any]],
                       gold: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """One-to-one (pred position, gold position) pairs, sorted by gold position.

    Pass 1: each gold item in order takes the first free predicted item that is equal and has the
    same status. Pass 2: each gold item still free takes the first free equal predicted item.
    """
    taken: set[int] = set()
    by_gold: dict[int, int] = {}
    for same_status in (True, False):
        for gi, g in enumerate(gold):
            if gi in by_gold:
                continue
            for pi, p in enumerate(pred):
                if pi in taken or not items_equal(p, g):
                    continue
                if same_status and p.get("status") != g.get("status"):
                    continue
                by_gold[gi] = pi
                taken.add(pi)
                break
    return [(by_gold[gi], gi) for gi in sorted(by_gold)]


# --------------------------------------------------------------------------- #
# G1, G4, G5, G7
# --------------------------------------------------------------------------- #
def _g5_class(item: Mapping[str, Any]) -> str:
    if item.get("status") == UNSURE and item.get("unsure_kind") in UNSURE_KINDS:
        return str(item["unsure_kind"])
    return "none"


def requirement_scores(pred: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]],
                       has_goal: bool) -> dict[str, Any]:
    """G1 (a, b, c), G4, G4-h, G5 and G7 (achieved) counts for one episode.

    ``has_goal`` false (the arm produced no goal) fails G1 even when the gold has no ending item.
    With pending predicted items every count carries ``pending`` = the number of pending items, and
    ``g1_ok`` and ``g4_ok`` are ``pending`` unless already false.
    """
    n_pending = sum(1 for p in pred if is_pending_item(p))
    pairs = match_requirements(pred, gold)
    g_to_p = {gi: pi for pi, gi in pairs}
    p_to_g = {pi: gi for pi, gi in pairs}

    # G1: gold ending items stated
    ending = [gi for gi, g in enumerate(gold) if g["status"] == REQUIRED and g["kind"] in ENDING_KINDS]
    stated = [gi for gi in ending if gi in g_to_p and pred[g_to_p[gi]]["status"] in (REQUIRED, UNSURE)]
    stated_req = [gi for gi in ending if gi in g_to_p and pred[g_to_p[gi]]["status"] == REQUIRED]
    all_stated = has_goal and len(stated) == len(ending)
    all_stated_req = has_goal and len(stated_req) == len(ending)
    never_stated = [gi for gi in ending if not any(
        p["status"] in (REQUIRED, UNSURE) and could_match(p, gold[gi]) for p in pred)]
    if not n_pending:
        g1_ok: Any = all_stated
    else:
        g1_ok = False if never_stated else PENDING

    # G4 and G4-h: incidental tagged as required, hallucinated required
    req_pos = [pi for pi, p in enumerate(pred) if p["status"] == REQUIRED]
    matched_req = [pi for pi in req_pos if pi in p_to_g]
    incidental = [pi for pi in matched_req if gold[p_to_g[pi]]["status"] == INCIDENTAL]
    halluc = [pi for pi in req_pos if pi not in p_to_g]
    sure_halluc = [pi for pi in req_pos if not any(could_match(pred[pi], g) for g in gold)]
    if not n_pending:
        g4_ok: Any = not incidental and not halluc
    else:
        g4_ok = False if sure_halluc else PENDING

    # G5: unsure flags on matched pairs
    confusion = {g: {p: 0 for p in G5_CLASSES} for g in G5_CLASSES}
    both = pred_pos = gold_pos = 0
    kind_counts = {k: {"both": 0, "pred": 0, "gold": 0} for k in UNSURE_KINDS}
    unsure_without_kind = 0
    for pi, gi in pairs:
        p, g = pred[pi], gold[gi]
        p_cls, g_cls = _g5_class(p), _g5_class(g)
        confusion[g_cls][p_cls] += 1
        p_uns, g_uns = p["status"] == UNSURE, g["status"] == UNSURE
        pred_pos += int(p_uns)
        gold_pos += int(g_uns)
        both += int(p_uns and g_uns)
        unsure_without_kind += int(p_uns and p_cls == "none")
        for k in UNSURE_KINDS:
            kind_counts[k]["pred"] += int(p_cls == k)
            kind_counts[k]["gold"] += int(g_cls == k)
            kind_counts[k]["both"] += int(p_cls == k and g_cls == k)
    gold_unsure = [gi for gi, g in enumerate(gold) if g["status"] == UNSURE]
    dropped = [gi for gi in gold_unsure if gi not in g_to_p]

    # G7: achieved on matched pairs with gold status required
    req_pairs = [(pi, gi) for pi, gi in pairs if gold[gi]["status"] == REQUIRED]
    achieved_ok = sum(1 for pi, gi in req_pairs if pred[pi]["achieved"] == gold[gi]["achieved"])

    c = n_pending
    result: dict[str, Any] = {
        "G1a": counts(len(stated), len(ending), c),
        "G1b": counts(int(all_stated), 1, c),
        "G1c": counts(int(all_stated_req), 1, c),
        "G4": counts(len(incidental), len(matched_req), c),
        "G4-h": counts(len(halluc), 1, c),
        "G4-h-any": counts(int(bool(halluc)), 1, c),
        "G5-P": counts(both, pred_pos, c),
        "G5-R": counts(both, gold_pos, c),
        "G5-dropped": counts(len(dropped), len(gold_unsure), c),
        "G7": counts(achieved_ok, len(req_pairs), c),
    }
    for k in UNSURE_KINDS:
        result[f"G5-P-{k}"] = counts(kind_counts[k]["both"], kind_counts[k]["pred"], c)
        result[f"G5-R-{k}"] = counts(kind_counts[k]["both"], kind_counts[k]["gold"], c)
    return {
        "counts": result,
        "g1_ok": g1_ok,
        "g4_ok": g4_ok,
        "pairs": [[pi, gi] for pi, gi in pairs],
        "pending_items": n_pending,
        "gold_ending": ending,
        "gold_ending_stated": stated,
        "incidental_as_required": incidental,
        "hallucinated_required": halluc,
        "g5_confusion": confusion,
        "g5_pred_unsure_without_kind": unsure_without_kind,
        "gold_unsure_dropped": dropped,
    }


# --------------------------------------------------------------------------- #
# G2: failed attempt copied
# --------------------------------------------------------------------------- #
def _marked_not_normal(seg: Mapping[str, Any]) -> bool:
    return _low(seg.get("outcome")) in ("failed", "aborted") or _flag(seg.get("mistake"))


def narrates_failure(text: Any) -> bool:
    """True when a coarse text describes an attempt or its failure ("try to grasp", "grasp twice")."""
    return bool(set(words(str(text or ""))) & FAILURE_NARRATIVE_WORDS)


def _judge_yes_no(answer: Any) -> bool | None:
    if isinstance(answer, bool):
        return answer
    if answer is None:
        return None
    text = str(answer).strip().lower()
    if text in YES_ANSWERS:
        return True
    if text in NO_ANSWERS:
        return False
    return None


def g2_episode(pred_segments: Sequence[Mapping[str, Any]], pred_coarse: Sequence[Mapping[str, Any]],
               gold_failed_attempts: Sequence[Mapping[str, Any]], pred_requirements_raw: Sequence[Any],
               *, episode_key: str | None = None,
               judge_answers: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """G2 for one episode; the denominator is 1 only when the gold has a failed attempt.

    Part (i): a predicted segment with IoU >= 0.5 to a gold failed span that is not marked failed,
    aborted or mistake; or a coarse subtask overlapping a gold failed span whose text narrates the
    attempt or its failure (:func:`narrates_failure`). Part (ii): the blind judge says a predicted
    requirement refers to something that exists only inside a gold failed span. ``judge_answers``
    maps ``g2|<episode_key>|<text>`` (or the normalized text alone) to yes or no.
    """
    spans = [s for s in (as_span(fa) for fa in gold_failed_attempts if isinstance(fa, Mapping)) if s]
    empty = counts(0, 0)
    if not spans:
        return {"counts": {"G2": empty, "G2-i": dict(empty), "G2-ii": dict(empty)}, "ok": True,
                "applicable": False, "part_i_hits": [], "judge_items": []}
    hits: list[dict[str, Any]] = []
    for k, seg in enumerate(pred_segments):
        sp = as_span(seg)
        for gs in spans:
            iou = span_iou(sp, gs)
            if iou >= G2_SEGMENT_MIN_IOU and not _marked_not_normal(seg):
                hits.append({"source": "segment", "index": k, "gold_span": list(gs), "iou": round(iou, 6)})
    for k, item in enumerate(pred_coarse):
        sp = as_span(item)
        if sp is None or not narrates_failure(item.get("text")):
            continue
        for gs in spans:
            if min(sp[1], gs[1]) >= max(sp[0], gs[0]):
                hits.append({"source": "coarse", "index": k, "gold_span": list(gs), "text": item.get("text")})
    part_i = bool(hits)

    answers = judge_answers if isinstance(judge_answers, Mapping) else {}
    items: list[dict[str, Any]] = []
    verdicts: list[bool | None] = []
    seen: set[str] = set()
    for req in pred_requirements_raw:
        if not isinstance(req, Mapping):
            continue
        text = requirement_text(req)
        jid = judge_id(JUDGE_G2, episode_key, text)
        if jid in seen:
            continue
        seen.add(jid)
        verdict = _judge_yes_no(answers.get(jid, answers.get(text)))
        verdicts.append(verdict)
        items.append({
            "judge_id": jid,
            "type": "g2_requirement",
            "episode_key": episode_key,
            "text": text,
            "failed_spans": [list(s) for s in spans],
            "question": "Does this requirement refer to an action, state or object that exists only "
                        "within a failed span? Answer yes or no.",
            "answered": verdict is not None,
        })
    unanswered = sum(1 for v in verdicts if v is None)
    if any(v is True for v in verdicts):
        part_ii: Any = True
    elif unanswered:
        part_ii = PENDING
    else:
        part_ii = False
    copied = part_i or part_ii is True
    pending = not copied and part_ii == PENDING
    return {
        "counts": {
            "G2": counts(int(copied), 1, unanswered if pending else 0),
            "G2-i": counts(int(part_i), 1),
            "G2-ii": counts(int(part_ii is True), 1, unanswered if part_ii == PENDING else 0),
        },
        "ok": False if copied else (PENDING if pending else True),
        "applicable": True,
        "part_i_hits": hits,
        "judge_items": items,
    }


# --------------------------------------------------------------------------- #
# G3, G6, outcome
# --------------------------------------------------------------------------- #
def g3_episode(view_goal: Mapping[str, Any] | None, gold_episode: Mapping[str, Any],
               resolve: Resolve) -> dict[str, Any]:
    """G3 (wrong target) for one episode: numerator 1 when the predicted primary target does not
    resolve to the gold one, or is missing. A gold episode without a primary target is not scored."""
    gold_pt = gold_episode.get("primary_target")
    if gold_pt is None:
        return {"counts": counts(0, 0), "ok": True, "reason": "gold has no primary_target", "resolution": None}
    name = _get(view_goal, "primary_target_name", "primary_target") if view_goal else None
    if not view_goal or is_no_reference(name):
        reason = "no goal" if not view_goal else "no primary target"
        return {"counts": counts(1, 1), "ok": False, "reason": reason, "resolution": None}
    res = resolve(name)
    if res["status"] == PENDING:
        return {"counts": counts(0, 1, 1), "ok": PENDING, "reason": res["reason"], "resolution": res}
    wrong = res["object_id"] != str(gold_pt)
    return {"counts": counts(int(wrong), 1), "ok": not wrong, "reason": res["reason"], "resolution": res}


def g6_episode(conditions: Mapping[str, Any]) -> dict[str, Any]:
    """G6 from its conditions (True, False or ``pending``): false if any is false, else pending if
    any is pending, else true."""
    values = [conditions[k] for k in sorted(conditions)]
    if any(v is False for v in values):
        correct: Any = False
    elif any(v == PENDING for v in values):
        correct = PENDING
    else:
        correct = True
    return {"counts": counts(int(correct is True), 1, int(correct == PENDING)), "correct": correct,
            "conditions": {k: conditions[k] for k in sorted(conditions)}}


def outcome_episode(pred_outcome: Any, gold_outcome: Any) -> dict[str, Any]:
    """Episode-outcome accuracy (G7): 1 when the predicted outcome equals the gold one."""
    p, g = _low(pred_outcome), _low(gold_outcome)
    return counts(int(p is not None and p == g), 1)


def goal_episode(view_goal: Mapping[str, Any] | None, gold_episode: Mapping[str, Any],
                 resolve: Resolve) -> dict[str, Any]:
    """Requirement matching and G1, G3, G4, G5, G7 for one episode (G2 and G6 are separate)."""
    gold_ids = [str(o["object_id"]) for o in gold_episode.get("objects") or []
                if isinstance(o, Mapping) and o.get("object_id")]
    gold_goal = gold_episode.get("goal") if isinstance(gold_episode.get("goal"), Mapping) else {}
    gold = gold_requirements(gold_goal.get("requirements") or [], gold_ids)
    regions = sorted({str(g[f])[len(REGION_PREFIX):] for g in gold for f in ("object", "ref")
                      if g[f] is not None and str(g[f]).startswith(REGION_PREFIX)})
    has_goal = isinstance(view_goal, Mapping) and bool(view_goal)
    raw = (view_goal.get("requirements") or []) if has_goal else []
    pred = pred_requirements(raw, resolve, regions)
    scores = requirement_scores(pred, gold, has_goal)
    g3 = g3_episode(view_goal if has_goal else None, gold_episode, resolve)
    return {"pred": pred, "gold": gold, "scores": scores, "g3": g3, "has_goal": has_goal,
            "raw_requirements": list(raw)}
