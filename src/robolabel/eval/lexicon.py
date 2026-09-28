"""Phase and predicate lexicons (MEASUREMENT_SPEC Appendices A and B), coarse grouping, compiled goal.

The tables live in ``src/robolabel/lexicon/*.yaml`` (drafts until they are frozen before a held-out
run); this module only applies them:

- :func:`map_phase` maps a free-text phase name to a phase class, or ``"unmapped"``.
- :func:`map_predicate` maps a free-text relation to a predicate, or ``"other"``.
- :func:`coarse_groups` derives coarse subtasks from phases with the Appendix A grouping rule and
  the fixed V_LITE L3 text templates.
- :func:`compiled_goal` builds the goal record that spec 4.0 gives to systems without a goal output.

Everything is deterministic and makes no network call.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import yaml

PHASE_LEXICON_FILE = "phase_lexicon_v1.yaml"
PREDICATE_LEXICON_FILE = "predicate_lexicon_v1.yaml"
UNMAPPED = "unmapped"
OTHER = "other"
COMPILED_BASIS = "compiled"

# Letters and digits in any script; underscores and punctuation split words.
_WORD_RE = re.compile(r"[^\W_]+")
_ARTICLES = frozenset({"the", "a", "an"})
# A name that already starts with one of these gets no "the" in coarse text.
_DETERMINERS = ("the ", "a ", "an ", "this ", "that ", "these ", "those ", "its ", "their ", "my ", "your ")
_PLACE_CLASSES = ("release", "insert", "pour")
_RELATION_TEXT = {"inside": "inside", "on_top_of": "on top of", "at_location": "at"}
# Target and destination values that name no object (V_LITE L3 allows "unsure" and "none").
_PLACEHOLDER_REFS = frozenset({"", "none", "null", "unsure", "unknown", "n/a"})
# Words that end the head of an object name ("bowl between the plate and the ramekin" -> "bowl").
_NAME_HEAD_ENDS = frozenset({
    "above", "at", "behind", "below", "beside", "between", "by", "from", "in", "inside", "into", "near",
    "next", "of", "on", "onto", "over", "that", "to", "under", "where", "which", "with",
})

# Resolves a reference (object id or name) to a display name or a category; None when unknown.
NameOf = Callable[[Any], Any]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@cache
def _read_lexicon(filename: str) -> dict[str, Any]:
    text = resources.files("robolabel").joinpath("lexicon").joinpath(filename).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError(f"lexicon file {filename} must be a mapping with a version")
    return data


def load_phase_lexicon() -> dict[str, Any]:
    """Return the phase lexicon (a copy of the cached parse of ``phase_lexicon_v1.yaml``)."""
    return copy.deepcopy(_read_lexicon(PHASE_LEXICON_FILE))


def load_predicate_lexicon() -> dict[str, Any]:
    """Return the predicate lexicon (a copy of the cached parse of ``predicate_lexicon_v1.yaml``)."""
    return copy.deepcopy(_read_lexicon(PREDICATE_LEXICON_FILE))


# --------------------------------------------------------------------------- #
# Words and phrases
# --------------------------------------------------------------------------- #
def words(text: str) -> tuple[str, ...]:
    """Lowercase ``text`` and split it into words (runs of letters and digits)."""
    return tuple(_WORD_RE.findall(str(text).lower()))


def _inflections(word: str) -> frozenset[str]:
    """``word`` plus its simple inflections: suffix s, es, ed, ing with regular spelling changes."""
    forms = {word, word + "s", word + "es", word + "ed", word + "ing"}
    if word.endswith("e"):
        forms.update({word + "d", word[:-1] + "ing"})
    if len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        forms.update({word[:-1] + "ies", word[:-1] + "ied"})
    if len(word) >= 3 and word[-1] not in "aeiouwxy" and word[-2] in "aeiou" and word[-3] not in "aeiou":
        forms.update({word + word[-1] + "ed", word + word[-1] + "ing"})
    return frozenset(forms)


@dataclass(frozen=True)
class _Phrase:
    """A keyword or pattern: accepted forms of its first word, then the exact remaining words."""

    first: frozenset[str]
    rest: tuple[str, ...]

    def found_in(self, tokens: Sequence[str]) -> bool:
        width = 1 + len(self.rest)
        for i in range(len(tokens) - width + 1):
            if tokens[i] in self.first and tuple(tokens[i + 1:i + width]) == self.rest:
                return True
        return False


def _phrase(text: str, *, inflect: bool) -> _Phrase:
    toks = words(text)
    if not toks:
        raise ValueError(f"empty lexicon phrase: {text!r}")
    first = _inflections(toks[0]) if inflect else frozenset({toks[0]})
    return _Phrase(first=first, rest=toks[1:])


# --------------------------------------------------------------------------- #
# Phase mapping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PhaseRules:
    classes: tuple[str, ...]
    legacy: dict[str, str]
    patterns: tuple[tuple[_Phrase, str], ...]
    keywords: tuple[tuple[str, tuple[_Phrase, ...]], ...]


@cache
def _phase_rules() -> _PhaseRules:
    lex = _read_lexicon(PHASE_LEXICON_FILE)
    classes = tuple(str(c) for c in lex["classes"])
    legacy = {" ".join(words(k)): str(v) for k, v in lex.get("legacy_mapping", {}).items()}
    patterns = tuple(
        (_phrase(p["pattern"], inflect=True), str(p["class"])) for p in lex["multi_word_patterns"]
    )
    rows = sorted(lex["keyword_priority"], key=lambda r: int(r["priority"]))
    keywords = tuple(
        (str(r["class"]), tuple(_phrase(k, inflect=True) for k in r["keywords"])) for r in rows
    )
    for cls in [*legacy.values(), *(c for _, c in patterns), *(c for c, _ in keywords)]:
        if cls not in classes:
            raise ValueError(f"phase lexicon maps to unknown class {cls!r}")
    return _PhaseRules(classes=classes, legacy=legacy, patterns=patterns, keywords=keywords)


def phase_classes() -> tuple[str, ...]:
    """The canonical phase classes, in lexicon order."""
    return _phase_rules().classes


def map_phase(text: str | None) -> str:
    """Map a free-text phase name to a phase class, or ``"unmapped"`` (Appendix A).

    Order: an exact class name maps to itself; a legacy closed-vocabulary name maps by the legacy
    table (``release-place`` to ``release``); then the multi-word patterns in lexicon order; then
    the keywords by priority, first hit wins. Patterns and keywords match whole words and whole
    phrases only ("open" does not match "reopen"); the first word of each also matches its simple
    inflections with suffix s, es, ed or ing ("picks", "placing", "dropped", "opening the gripper").
    """
    if text is None:
        return UNMAPPED
    tokens = words(text)
    if not tokens:
        return UNMAPPED
    rules = _phase_rules()
    raw = str(text).strip().lower()
    if raw in rules.classes:
        return raw
    joined = " ".join(tokens)
    if joined in rules.classes:
        return joined
    if joined in rules.legacy:
        return rules.legacy[joined]
    for phrase, cls in rules.patterns:
        if phrase.found_in(tokens):
            return cls
    for cls, phrases in rules.keywords:
        if any(p.found_in(tokens) for p in phrases):
            return cls
    return UNMAPPED


# --------------------------------------------------------------------------- #
# Predicate mapping and relations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PredicateRules:
    names: tuple[str, ...]
    value_types: dict[str, str]
    aliases: tuple[tuple[_Phrase, str], ...]
    relation_by_category: dict[str, str]
    default_relation: str
    relation_words: dict[str, str]


@cache
def _predicate_rules() -> _PredicateRules:
    lex = _read_lexicon(PREDICATE_LEXICON_FILE)
    names = tuple(str(p["name"]) for p in lex["predicates"])
    value_types = {str(p["name"]): str(p["value"]) for p in lex["predicates"]}
    ranked: list[tuple[int, int, int, str, str]] = []
    order = 0
    for pred in lex["predicates"]:
        name = str(pred["name"])
        for alias in [name.replace("_", " "), *pred.get("aliases", [])]:
            alias_text = " ".join(words(alias))
            ranked.append((-len(alias_text.split()), -len(alias_text), order, alias_text, name))
            order += 1
    # Longest alias first (by words, then characters); ties go to lexicon order.
    ranked.sort()
    seen: set[str] = set()
    aliases: list[tuple[_Phrase, str]] = []
    for _, _, _, alias_text, name in ranked:
        if alias_text in seen:
            continue
        seen.add(alias_text)
        aliases.append((_phrase(alias_text, inflect=False), name))
    default = str(lex.get("default_relation", "at_location"))
    return _PredicateRules(
        names=names,
        value_types=value_types,
        aliases=tuple(aliases),
        relation_by_category={
            str(k).lower(): str(v) for k, v in lex["destination_relation_by_category"].items()
        },
        default_relation=default,
        relation_words={str(k): str(v) for k, v in lex["coarse_relation_words"].items()},
    )


def predicate_names() -> tuple[str, ...]:
    """The lexicon predicates, in lexicon order (``other`` is not among them)."""
    return _predicate_rules().names


def predicate_value_type(predicate: str) -> str | None:
    """``"bool"`` or ``"string"`` for a lexicon predicate, None for ``other`` or an unknown name."""
    return _predicate_rules().value_types.get(predicate)


def map_predicate(text: str | None) -> str:
    """Map free text to a predicate name, or ``"other"`` (Appendix B).

    An exact predicate name wins (``on_top_of`` or ``on top of``). Then the aliases, each
    predicate's own name included, are matched as whole words or phrases, longest alias first
    ("gripper open" before "open", "in the gripper" before "in"). No inflection is applied.
    """
    if text is None:
        return OTHER
    tokens = words(text)
    if not tokens:
        return OTHER
    rules = _predicate_rules()
    underscored = "_".join(tokens)
    if underscored in rules.names:
        return underscored
    for phrase, name in rules.aliases:
        if phrase.found_in(tokens):
            return name
    return OTHER


def relation_for_category(category: str | None) -> str:
    """Placement predicate for a destination category: ``inside``, ``on_top_of`` or ``at_location``."""
    rules = _predicate_rules()
    if not category:
        return rules.default_relation
    return rules.relation_by_category.get(str(category).strip().lower(), rules.default_relation)


def relation_words(predicate: str | None) -> str:
    """Word for a placement relation in coarse text: ``in``, ``on`` or ``at`` (``at`` if unknown)."""
    rules = _predicate_rules()
    fallback = rules.relation_words.get(rules.default_relation, "at")
    return rules.relation_words.get(str(predicate), fallback)


def _relation_from_name(name: str | None) -> str | None:
    """Relation from the last category word in the head of a name ("the wooden tray" -> tray).

    The head ends at the first preposition or relative word after the first word, so "bowl
    between the plate and the ramekin" is a bowl, not a plate.
    """
    if not name:
        return None
    table = _predicate_rules().relation_by_category
    toks = words(name)
    head = next((toks[:i] for i, w in enumerate(toks) if i > 0 and w in _NAME_HEAD_ENDS), toks)
    for word in reversed(head):
        if word in table:
            return table[word]
    return None


# --------------------------------------------------------------------------- #
# Objects
# --------------------------------------------------------------------------- #
def normalize_name(text: str) -> str:
    """Lowercase, drop punctuation and articles (spec 4.0 object resolution rule 2)."""
    return " ".join(w for w in words(text) if w not in _ARTICLES)


def _find_object(ref: Any, objects: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The object whose ``object_id`` equals ``ref``, else the one whose name or alias equals it."""
    if ref is None:
        return None
    objs = [o for o in objects if isinstance(o, Mapping)]
    for obj in objs:
        if obj.get("object_id") == ref:
            return obj
    key = normalize_name(str(ref))
    if not key:
        return None
    hits = [o for o in objs if key in {normalize_name(str(n)) for n in _names(o)}]
    return hits[0] if len(hits) == 1 else None


def _names(obj: Mapping[str, Any]) -> list[Any]:
    """An object's name and aliases."""
    aliases = obj.get("aliases")
    return [obj.get("name") or "", *(aliases if isinstance(aliases, list) else [])]


def object_namer(objects: Iterable[Mapping[str, Any]] | None) -> NameOf:
    """A ``name_of`` for :func:`coarse_groups`: object id to its name; other strings pass through."""
    objs = list(objects or [])

    def name_of(ref: Any) -> str | None:
        if ref is None:
            return None
        obj = _find_object(ref, objs)
        if obj is not None and obj.get("name"):
            return str(obj["name"])
        return str(ref)

    return name_of


def object_categorizer(objects: Iterable[Mapping[str, Any]] | None) -> NameOf:
    """A ``category_of`` for :func:`coarse_groups`: object id or name to its category, else None."""
    objs = list(objects or [])

    def category_of(ref: Any) -> str | None:
        obj = _find_object(ref, objs)
        if obj is None or not obj.get("category"):
            return None
        return str(obj["category"])

    return category_of


def _placement_relation(ref: Any, name_of: NameOf, category_of: NameOf | None) -> str:
    """Relation for a destination: its category in the lexicon, else a category word in its name."""
    rules = _predicate_rules()
    if ref is None:
        return rules.default_relation
    category = category_of(ref) if category_of is not None else None
    if category and str(category).strip().lower() in rules.relation_by_category:
        return relation_for_category(category)
    return _relation_from_name(name_of(ref)) or rules.default_relation


def _noun(ref: Any, name_of: NameOf) -> str:
    """Name an object for coarse text: "the <name>", or "the object" when the name is unknown."""
    unknown = str(_read_lexicon(PHASE_LEXICON_FILE)["coarse_templates"]["unknown_object"])
    if ref is None:
        return unknown
    name = name_of(ref)
    if not name or str(name).strip().lower() in _PLACEHOLDER_REFS:
        return unknown
    name = " ".join(str(name).split())
    lowered = name.lower()
    if lowered in ("it", "them") or lowered.startswith(_DETERMINERS):
        return name
    return f"the {name}"


# --------------------------------------------------------------------------- #
# Segment accessors (gold segments and view records)
# --------------------------------------------------------------------------- #
def _get(seg: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in seg and seg[key] is not None:
            return seg[key]
    return None


def seg_start(seg: Mapping[str, Any]) -> int:
    """First frame of a segment (``start_frame``, or the view-record alias ``start``)."""
    return int(_get(seg, "start_frame", "start"))


def seg_end(seg: Mapping[str, Any]) -> int:
    """Last frame of a segment, inclusive (``end_frame``, or the view-record alias ``end``)."""
    return int(_get(seg, "end_frame", "end"))


def _get_ref(seg: Mapping[str, Any], *keys: str) -> Any:
    """First key holding a real reference; placeholders such as "none" or "unsure" count as missing."""
    for key in keys:
        value = seg.get(key)
        if value is None or (isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_REFS):
            continue
        return value
    return None


def _seg_target(seg: Mapping[str, Any]) -> Any:
    return _get_ref(seg, "target", "target_name", "target_object_id")


def _seg_destination(seg: Mapping[str, Any]) -> Any:
    return _get_ref(seg, "destination", "destination_name", "destination_object_id")


def seg_class(seg: Mapping[str, Any]) -> str:
    """Phase class of a segment: its ``phase_class`` (or ``phase``) mapped by :func:`map_phase`."""
    return map_phase(_get(seg, "phase_class", "phase"))


def _seg_failed(seg: Mapping[str, Any]) -> bool:
    """A segment of a failed attempt: outcome failed or aborted, or marked as a mistake."""
    return seg.get("outcome") in ("failed", "aborted") or seg.get("mistake") is True


def _ordered(segments: Sequence[Mapping[str, Any]]) -> list[int]:
    """Input positions sorted by (start, end, position)."""
    return sorted(range(len(segments)), key=lambda i: (seg_start(segments[i]), seg_end(segments[i]), i))


# --------------------------------------------------------------------------- #
# Coarse grouping
# --------------------------------------------------------------------------- #
def _group_members(segments: Sequence[Mapping[str, Any]], order: list[int]) -> list[tuple[list[int], bool]]:
    """Walk the phases in order and return (input positions, mistake) per coarse group."""
    classes = [seg_class(segments[i]) for i in order]
    failed = [_seg_failed(segments[i]) for i in order]
    attempts = [segments[i].get("attempt_idx") for i in order]
    groups: list[tuple[list[int], bool]] = []
    pending: list[int] = []   # approaches (and a transport before a failed placement) waiting for a group
    k, n = 0, len(order)
    while k < n:
        if failed[k]:
            # Consecutive failed phases of the same attempt form one group, with its approaches.
            j = k + 1
            while j < n and failed[j] and attempts[j] == attempts[k]:
                j += 1
            groups.append((pending + [order[m] for m in range(k, j)], True))
            pending, k = [], j
            continue
        if classes[k] == "approach":
            pending.append(order[k])
            k += 1
            continue
        if classes[k] == "transport" and k + 1 < n and failed[k + 1] and classes[k + 1] in _PLACE_CLASSES:
            # The transport of a failed placement (misplace) belongs to that failed group.
            pending.append(order[k])
            k += 1
            continue
        members = [k]
        k += 1
        if classes[members[0]] == "transport" and k < n and not failed[k] and classes[k] in _PLACE_CLASSES:
            members.append(k)
            k += 1
        if classes[members[-1]] in ("insert", "pour") and k < n and not failed[k] and classes[k] == "release":
            members.append(k)
            k += 1
        groups.append((pending + [order[m] for m in members], False))
        pending = []
    if pending:
        groups.append((pending, False))
    return groups


def _first(values: Iterable[Any]) -> Any:
    return next((v for v in values if v is not None), None)


def _group_text(
    segments: Sequence[Mapping[str, Any]],
    members: list[int],
    mistake: bool,
    later: list[int],
    name_of: NameOf,
    category_of: NameOf | None,
) -> tuple[str, Any, Any]:
    """Text, target and destination of one coarse group (the intended instruction if failed)."""
    templates = _read_lexicon(PHASE_LEXICON_FILE)["coarse_templates"]
    classes = [seg_class(segments[i]) for i in members]
    main = [i for i, c in zip(members, classes, strict=True) if c != "approach"] or list(members)
    if mistake:
        # Moving the arm away is never the intended instruction of a failed attempt.
        approaches = [i for i, c in zip(members, classes, strict=True) if c == "approach"]
        main = [i for i in main if seg_class(segments[i]) != "retract"] or approaches or main
    main_classes = {seg_class(segments[i]) for i in main}
    target = _first(_seg_target(segments[i]) for i in main)
    if target is None:
        target = _first(_seg_target(segments[i]) for i in members)

    # A failed attempt that got as far as transport was meant to end in a placement.
    if main_classes & set(_PLACE_CLASSES) or (mistake and "transport" in main_classes):
        place = [i for i in main if seg_class(segments[i]) in _PLACE_CLASSES]
        carry = [i for i in main if seg_class(segments[i]) == "transport"]
        destination = _first(_seg_destination(segments[i]) for i in place + carry)
        if destination is None and carry and place:
            # Legacy convention: a release segment's target is the destination (spec 4.0).
            moved = _first(_seg_target(segments[i]) for i in carry)
            placed_on = _first(_seg_target(segments[i]) for i in place)
            if placed_on is not None and placed_on != moved:
                target, destination = moved, placed_on
        if destination is None and mistake:
            # The intended placement of a failed attempt: the next placement in the episode.
            destination = _first(
                _seg_destination(segments[i]) for i in later if seg_class(segments[i]) in _PLACE_CLASSES
            )
        fields = {"target": _noun(target, name_of), "destination": _noun(destination, name_of)}
        if "pour" in main_classes:
            return templates["pour"].format(**fields), target, destination
        if "insert" in main_classes:
            return templates["insert"].format(**fields), target, destination
        relation = _placement_relation(destination, name_of, category_of)
        fields["relation"] = relation_words(relation)
        return templates["put"].format(**fields), target, destination
    if "press" in main_classes:
        return templates["press"].format(target=_noun(target, name_of)), target, None
    if "grasp" in main_classes or (mistake and main_classes == {"approach"}):
        return templates["pick_up"].format(target=_noun(target, name_of)), target, None
    if "retract" in main_classes:
        return templates["retract"], None, None
    cls = seg_class(segments[main[0]])
    cls = OTHER if cls == UNMAPPED else cls
    destination = _first(_seg_destination(segments[i]) for i in members)
    text = templates["other_class"].format(phase_class=cls, target=_noun(target, name_of))
    return text, target, destination


def coarse_groups(
    segments: Sequence[Mapping[str, Any]],
    name_of: NameOf | None = None,
    *,
    category_of: NameOf | None = None,
) -> list[dict[str, Any]]:
    """Coarse subtasks from phases (Appendix A grouping rule, V_LITE L3 templates).

    Segments carry ``start_frame``/``end_frame`` (or ``start``/``end``), ``phase_class`` (a class or
    free text, mapped by :func:`map_phase`), ``target`` and ``destination`` (ids or names; view
    records may use ``target_name``/``destination_name``), ``outcome`` and optionally
    ``attempt_idx``. ``name_of`` turns a reference into a display name (default: the reference
    itself); ``category_of`` gives a destination's category for the put relation (see
    :func:`object_namer` and :func:`object_categorizer`). Without a known category the relation
    comes from a category word in the destination's name, else ``at_location``.

    Rules: an approach joins the next non-approach phase; grasp is "pick up <target>"; transport
    followed by release, insert or pour (and the release after an insert or pour) is "put <target>
    <relation> <destination>", "insert <target> into <destination>" or "pour <target> into
    <destination>" (a placement without a transport reads the same way); retract is "move the arm
    away"; press is "press <control>"; any other class is "<class> <target>". Consecutive phases
    with outcome failed or aborted (or ``mistake: true``) of one attempt form their own group with
    ``mistake: true``, together with the approaches before them and, for a failed placement, the
    transport just before it. The text is the intended instruction: "pick up <target>" for a failed
    approach or grasp (a failed retract never sets the intent), the put (pour, insert) text once the
    attempt reached transport or a placement (its destination, if the failed phases name none, is
    that of the next placement). Targets and destinations "none", "unsure", "unknown" or empty
    count as unknown and read "the object".

    Returns dicts with ``coarse_idx``, ``start_frame``, ``end_frame``, ``text``, ``target``,
    ``destination`` (raw references), ``mistake``, ``phase_classes`` and ``segment_indices``
    (positions in the input list).
    """
    namer: NameOf = name_of if name_of is not None else (lambda ref: None if ref is None else str(ref))
    order = _ordered(segments)
    out: list[dict[str, Any]] = []
    for members, mistake in _group_members(segments, order):
        last_pos = order.index(members[-1])
        text, target, destination = _group_text(
            segments, members, mistake, order[last_pos + 1:], namer, category_of
        )
        out.append({
            "coarse_idx": len(out),
            "start_frame": min(seg_start(segments[i]) for i in members),
            "end_frame": max(seg_end(segments[i]) for i in members),
            "text": text,
            "target": target,
            "destination": destination,
            "mistake": mistake,
            "phase_classes": [seg_class(segments[i]) for i in members],
            "segment_indices": list(members),
        })
    return out


# --------------------------------------------------------------------------- #
# Compiled goal (spec 4.0)
# --------------------------------------------------------------------------- #
def compiled_goal(
    segments: Sequence[Mapping[str, Any]],
    objects: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The goal record the harness compiles for a system without one (spec 4.0).

    ``primary_target`` is the target of the last grasp-class segment. The destination is the
    ``destination`` of the last release-class segment, or its ``target`` when it has none (the
    legacy convention). With a destination there is one ``required`` requirement ``inside``,
    ``on_top_of`` or ``at_location``, chosen by the destination's category in the lexicon (or a
    category word in its name), ``at_location`` if unknown. If the last segment's class is retract,
    one ``required`` requirement ``withdrawn``. No unsure items; every item has basis
    ``compiled``. ``achieved`` is what the segments imply: true, or false when the placing
    segment failed.
    """
    objs = [o for o in (objects or []) if isinstance(o, Mapping)]
    name_of = object_namer(objs)
    order = _ordered(segments)
    grasps = [i for i in order if seg_class(segments[i]) == "grasp"]
    releases = [i for i in order if seg_class(segments[i]) == "release"]
    primary_target = _seg_target(segments[grasps[-1]]) if grasps else None
    destination = None
    placed_ok = True
    if releases:
        last_release = segments[releases[-1]]
        destination = _seg_destination(last_release)
        if destination is None:
            destination = _seg_target(last_release)
        placed_ok = not _seg_failed(last_release)
    last_frame = seg_end(segments[order[-1]]) if order else None

    requirements: list[dict[str, Any]] = []
    clauses: list[str] = []
    if destination is not None:
        relation = _placement_relation(destination, name_of, object_categorizer(objs))
        requirements.append(_compiled_requirement(
            len(requirements) + 1, "object_end_state", primary_target, relation, destination, placed_ok,
            last_frame,
        ))
        clauses.append(f"{_noun(primary_target, name_of)} is {_RELATION_TEXT.get(relation, relation)} "
                       f"{_noun(destination, name_of)}")
    if order and seg_class(segments[order[-1]]) == "retract":
        requirements.append(_compiled_requirement(
            len(requirements) + 1, "robot_end_state", None, "withdrawn", None, True, last_frame,
        ))
        clauses.append("the arm is withdrawn")
    return {
        "objective_text": " and ".join(clauses),
        "primary_target": primary_target,
        "primary_destination": destination,
        "requirements": requirements,
        "basis": COMPILED_BASIS,
    }


def _compiled_requirement(
    number: int, kind: str, obj: Any, predicate: str, ref_object: Any, achieved: bool, frame: int | None,
) -> dict[str, Any]:
    return {
        "req_id": f"r{number}",
        "kind": kind,
        "object": obj,
        "predicate": predicate,
        "ref_object": ref_object,
        "value": True,
        "status": "required",
        "unsure_kind": None,
        "basis": COMPILED_BASIS,
        "achieved": achieved,
        "deciding_frame": frame,
        "deciding_camera": None,
        "visibility": {},
        "reason": "compiled from segments (MEASUREMENT_SPEC 4.0)",
    }
