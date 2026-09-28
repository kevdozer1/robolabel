"""L4 multi-episode consensus (PLAN 3.7): the goal from what the successful demonstrations share.

Per episode, an end-state record is built from facts only (the L1 robot end state plus last-frame scene
facts). Success comes from those facts, never from a human label. Each canonical predicate is then
counted over the successful episodes; with p the share showing it and L the lower end of its Wilson 95
percent interval:

* stated by the task string: ``required`` (basis ``task_string``) regardless of p;
* p >= 0.9 and L >= 0.75: ``required`` (basis ``consensus``);
* 0.3 < p < 0.9: ``unsure`` with kind ``intent`` (rendered "if possible");
* p <= 0.3: ``incidental``.

One strong call per group may write the objective sentence and point out predicates the table missed,
but it cannot promote an item above its consensus status. The per-episode goal is the group spec plus
that episode's own ``achieved`` values. Deterministic apart from the optional group call.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

Z95 = 1.959963984540054


def wilson_lower(k: int, n: int, z: float = Z95) -> float:
    """Lower end of the Wilson score interval for k successes in n trials (0 when n is 0)."""
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / den)


def status_for(p: float, lower: float, stated: bool) -> tuple[str, str | None, str]:
    """(status, unsure_kind, basis) by the PLAN 3.7 thresholds."""
    if stated:
        return "required", None, "task_string"
    if p >= 0.9 and lower >= 0.75:
        return "required", None, "consensus"
    if p > 0.3:
        return "unsure", "intent", "consensus"
    return "incidental", None, "consensus"


def canonical_key(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """(kind, role or object, predicate, ref) with objects named by role, not by per-episode ID."""
    return (str(item.get("kind", "")), str(item.get("object", "none")), str(item.get("predicate", "")),
            str(item.get("ref_object", "none")))


def count_predicates(records: Sequence[Mapping[str, Any]], stated: Iterable[tuple[str, str, str, str]] = ()
                     ) -> list[dict[str, Any]]:
    """Count canonical predicates over the successful episodes.

    ``records`` are per-episode end-state records ``{"episode_key", "successful": bool, "facts": [{kind,
    object, predicate, ref_object, value}]}`` with objects named by role (target, destination, robot).
    Only facts with value true count as present; unknown values count as absent. Returns one row per
    predicate seen in any successful episode, sorted, with p, the Wilson lower bound and the status.
    """
    ok = [r for r in records if r.get("successful")]
    n = len(ok)
    stated_set = {tuple(s) for s in stated}
    keys: set[tuple[str, str, str, str]] = set(stated_set)
    for r in ok:
        for f in r.get("facts", []):
            if f.get("value") is True:
                keys.add(canonical_key(f))
    rows = []
    for key in sorted(keys):
        k = sum(1 for r in ok if any(canonical_key(f) == key and f.get("value") is True for f in r.get("facts", [])))
        p = k / n if n else 0.0
        lower = wilson_lower(k, n)
        status, uk, basis = status_for(p, lower, key in stated_set)
        rows.append({"kind": key[0], "object": key[1], "predicate": key[2], "ref_object": key[3], "present_in": k,
                     "of_successful": n, "p": round(p, 4), "wilson_lower": round(lower, 4), "status": status,
                     "unsure_kind": uk, "basis": basis})
    return rows


def episode_goal(spec_rows: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The group spec with this episode's own achieved values (true, false, or unknown when not observed)."""
    facts = {canonical_key(f): f.get("value") for f in record.get("facts", [])}
    out = []
    for i, row in enumerate(spec_rows, 1):
        key = (row["kind"], row["object"], row["predicate"], row["ref_object"])
        v = facts.get(key)
        out.append({"req_id": f"r{i}", "kind": row["kind"], "object": row["object"], "predicate": row["predicate"],
                    "ref_object": row["ref_object"], "value": True, "status": row["status"],
                    "unsure_kind": row["unsure_kind"], "basis": row["basis"],
                    "achieved": v if isinstance(v, bool) else "unknown",
                    "consensus_present": row["present_in"], "consensus_of": row["of_successful"]})
    return out
