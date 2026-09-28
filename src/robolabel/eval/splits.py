"""Episode selection rules of MEASUREMENT_SPEC 2.2, as pure functions (no file or network access).

Every shuffle uses a fresh ``numpy.random.default_rng(SEED)`` created immediately before it, and
``rng.permutation`` over the sorted list (SPEC_QUESTIONS Q1). Items are then taken in shuffled order.
The caller reads the dataset metadata and writes the split files; this module only decides.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

SEED = 20261001

F3_STRATA: tuple[tuple[str, int], ...] = (
    ("teleoperated_successful", 8),
    ("policy_failure", 14),
    ("policy_successful", 8),
)


def seeded_order(items: Iterable[Any], seed: int = SEED) -> list[Any]:
    """Sort ``items``, then return them in the order of a fresh seeded permutation."""
    ordered = sorted(items)
    if not ordered:
        return []
    rng = np.random.default_rng(seed)
    return [ordered[int(i)] for i in rng.permutation(len(ordered))]


def normalize_libero_name(name: str) -> str:
    """Official LIBERO task name to the dataset's task string form.

    Strips a scene prefix such as ``LIVING_ROOM_SCENE2_`` and replaces underscores with spaces.
    """
    stripped = re.sub(r"^[A-Z0-9_]*SCENE\d+_", "", name.strip())
    return re.sub(r"\s+", " ", stripped.replace("_", " ")).strip().lower()


def libero_suites(task_strings: Sequence[str], suite_map: dict[str, Sequence[str]]) -> dict[str, list[str]]:
    """Group the dataset's task strings by LIBERO suite. Unmatched strings go under ``"unmatched"``."""
    lookup: dict[str, str] = {}
    for suite, names in suite_map.items():
        for name in names:
            lookup.setdefault(normalize_libero_name(name), suite)
    out: dict[str, list[str]] = {}
    for task in task_strings:
        suite = lookup.get(re.sub(r"\s+", " ", task.strip().lower()), "unmatched")
        out.setdefault(suite, []).append(task)
    return {k: sorted(v) for k, v in sorted(out.items())}


def f2_split(
    suite_tasks: dict[str, Sequence[str]],
    episodes_by_task: dict[str, Sequence[int]],
    plan: Sequence[tuple[str, int, int]] = (("libero_spatial", 2, 2), ("libero_10", 1, 1)),
    gold_per_task: int = 8,
) -> dict[str, Any]:
    """Spec 2.2 F2: per suite, n_dev then n_heldout tasks in shuffled order; 8 gold episodes per task.

    Returns ``{"tasks": {task: {"suite", "split", "order", "gold_episodes", "all_episodes"}}}`` where
    ``order`` is the task's position in its suite's shuffled order.
    """
    tasks: dict[str, Any] = {}
    for suite, n_dev, n_held in plan:
        order = seeded_order(suite_tasks.get(suite, []))
        for pos, task in enumerate(order[: n_dev + n_held]):
            eps = seeded_order(int(e) for e in episodes_by_task.get(task, []))
            tasks[task] = {
                "suite": suite,
                "split": "dev" if pos < n_dev else "heldout",
                "order": pos,
                "gold_episodes": eps[:gold_per_task],
                "all_episodes": sorted(int(e) for e in episodes_by_task.get(task, [])),
            }
    return {"tasks": tasks}


def f3_stratum(policy_type: str, success_class: str) -> str | None:
    """The spec 2.2 F3 stratum of one episode, or None when it belongs to no stratum."""
    teleop = str(policy_type).strip().lower() == "teleoperated"
    cls = str(success_class).strip().lower()
    if teleop:
        return "teleoperated_successful" if cls == "successful" else None
    if cls == "failure":
        return "policy_failure"
    if cls == "successful":
        return "policy_successful"
    if cls == "suboptimal":
        return "policy_suboptimal"
    return None


def f3_split(episodes: Sequence[dict[str, Any]], tasks: Sequence[str]) -> dict[str, Any]:
    """Spec 2.2 F3 per task: 8 teleoperated successful, 14 policy failure, 8 policy successful
    (suboptimal fills a short successful stratum, itself in shuffled order); dev = the first half of
    each stratum's selection in shuffled order (SPEC_QUESTIONS Q2).

    ``episodes`` items need ``episode_index``, ``task``, ``policy_type`` and ``success_class``.
    """
    out: dict[str, Any] = {}
    for task in tasks:
        rows = [e for e in episodes if e["task"] == task]
        by_stratum: dict[str, list[int]] = {}
        for e in rows:
            s = f3_stratum(e["policy_type"], e["success_class"])
            if s is not None:
                by_stratum.setdefault(s, []).append(int(e["episode_index"]))
        strata: dict[str, Any] = {}
        for name, n in F3_STRATA:
            order = seeded_order(by_stratum.get(name, []))
            picked = order[:n]
            filled_from_suboptimal: list[int] = []
            if name == "policy_successful" and len(picked) < n:
                sub = seeded_order(by_stratum.get("policy_suboptimal", []))
                filled_from_suboptimal = sub[: n - len(picked)]
                picked = picked + filled_from_suboptimal
            half = len(picked) // 2 if len(picked) % 2 == 0 else (len(picked) + 1) // 2
            strata[name] = {
                "available": len(by_stratum.get(name, [])),
                "selected_in_order": picked,
                "filled_from_suboptimal": filled_from_suboptimal,
                "dev": picked[:half],
                "heldout": picked[half:],
            }
        out[task] = {"strata": strata,
                     "dev": [i for s in strata.values() for i in s["dev"]],
                     "heldout": [i for s in strata.values() for i in s["heldout"]]}
    return out
