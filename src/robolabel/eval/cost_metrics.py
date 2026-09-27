"""Cost and latency metrics K1 to K3 (MEASUREMENT_SPEC 4.6), computed from receipt dicts only.

A receipt here is any dict with the spec 9.1 fields these metrics read: ``usd``,
``usd_batch_eq`` (optional), ``usage`` (``input_text_tokens``, ``input_image_tokens``,
``input_video_tokens``, ``input_audio_tokens``, ``cached_tokens``, ``output_tokens``,
``reasoning_tokens``), ``latency_s``, ``wall_s``, ``episode_key`` (top level, else
``inputs.episode_key`` as ``eval.receipts`` writes it), ``step`` and ``cache_hit``.

A cache hit carries the cost, tokens and latency of the call that produced it and counts with
them by default; pass ``include_cache_hits=False`` to leave cache hits out. Percentiles are numpy
percentiles with linear interpolation. Dollar sums use ``math.fsum``, so they do not depend on
receipt order. Floats are rounded to 6 decimals. Pure functions, no file I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

DECIMALS = 6
TOKEN_KINDS = ("input_text", "input_image", "input_video", "input_audio", "cached", "output", "reasoning")
USAGE_KEYS = {
    "input_text": "input_text_tokens",
    "input_image": "input_image_tokens",
    "input_video": "input_video_tokens",
    "input_audio": "input_audio_tokens",
    "cached": "cached_tokens",
    "output": "output_tokens",
    "reasoning": "reasoning_tokens",
}
UNKNOWN_STEP = "unknown"


def _r6(value: float | None) -> float | None:
    return None if value is None else round(float(value), DECIMALS)


def _num(value: Any) -> float | None:
    """A finite float, or None for None, NaN and non-numbers."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _flag(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(value)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return _r6(float(np.percentile(np.asarray(values, dtype=float), q, method="linear")))


def episode_sort_key(key: str) -> tuple[str, int, int, str]:
    """Sort "F1/2" before "F1/10": family, then numeric episode index."""
    family, _, rest = str(key).partition("/")
    if rest.isdigit():
        return family, 0, int(rest), ""
    return family, 1, 0, rest


def _episode_list(receipts: Iterable[Mapping[str, Any]], episode_keys: Iterable[str] | None) -> list[str]:
    """The episode set: the given keys, else every episode key seen in the receipts."""
    if episode_keys is not None:
        keys = {str(k) for k in episode_keys}
    else:
        keys = {key for key in (_episode_key(r) for r in receipts) if key is not None}
    return sorted(keys, key=episode_sort_key)


def _episode_key(receipt: Mapping[str, Any]) -> str | None:
    """The receipt's episode key: top-level ``episode_key``, else ``inputs.episode_key``."""
    key = receipt.get("episode_key")
    if key is None and isinstance(receipt.get("inputs"), Mapping):
        key = receipt["inputs"].get("episode_key")
    return None if key is None else str(key)


def _select(receipts: Iterable[Mapping[str, Any]], include_cache_hits: bool) -> list[Mapping[str, Any]]:
    return [r for r in receipts if include_cache_hits or not _flag(r.get("cache_hit"))]


def _step(receipt: Mapping[str, Any]) -> str:
    step = receipt.get("step")
    return UNKNOWN_STEP if step is None else str(step)


# --------------------------------------------------------------------------- #
# K1: dollars
# --------------------------------------------------------------------------- #
def k1(receipts: Sequence[Mapping[str, Any]], include_cache_hits: bool = True,
       episode_keys: Iterable[str] | None = None) -> dict[str, Any]:
    """K1: dollars per episode (list, p50, mean) and per 1,000 episodes, actual and batch-equivalent.

    The episode set is ``episode_keys`` when given, else every episode key in the receipts
    (before cache hits are dropped, so a fully cached episode counts as $0 when they are
    excluded; ``n_cache_hits`` counts them either way). Receipts with no episode key, or one
    outside the set, go to ``unattributed_usd``. Episode keys are read with ``_episode_key``.
    A receipt without ``usd_batch_eq`` counts its actual ``usd`` in the batch-equivalent sum and
    is counted in ``n_receipts_without_batch_eq``. A null ``usd`` counts as 0 and is counted in
    ``n_receipts_missing_usd`` (spec 9.3 says it should never happen). Also broken down by step.
    """
    receipts = list(receipts)  # read several times below; a generator would give zeros
    episodes = _episode_list(receipts, episode_keys)
    chosen = _select(receipts, include_cache_hits)
    usd: dict[str, list[float]] = {ep: [] for ep in episodes}
    batch: dict[str, list[float]] = {ep: [] for ep in episodes}
    step_usd: dict[str, list[float]] = {}
    step_batch: dict[str, list[float]] = {}
    step_calls: dict[str, int] = {}
    unattributed: list[float] = []
    unattributed_batch: list[float] = []
    missing_usd = without_batch = 0
    for r in chosen:
        cost = _num(r.get("usd"))
        if cost is None:
            missing_usd += 1
            cost = 0.0
        cost_batch = _num(r.get("usd_batch_eq"))
        if cost_batch is None:
            without_batch += 1
            cost_batch = cost
        key = _episode_key(r)
        if key is None or key not in usd:
            unattributed.append(cost)
            unattributed_batch.append(cost_batch)
            continue
        usd[key].append(cost)
        batch[key].append(cost_batch)
        step = _step(r)
        step_usd.setdefault(step, []).append(cost)
        step_batch.setdefault(step, []).append(cost_batch)
        step_calls[step] = step_calls.get(step, 0) + 1

    n = len(episodes)
    per_ep = [math.fsum(usd[ep]) for ep in episodes]
    per_ep_batch = [math.fsum(batch[ep]) for ep in episodes]
    mean = math.fsum(per_ep) / n if n else None
    mean_batch = math.fsum(per_ep_batch) / n if n else None
    by_step: dict[str, dict[str, Any]] = {}
    for step in sorted(step_usd):
        total = math.fsum(step_usd[step])
        total_batch = math.fsum(step_batch[step])
        by_step[step] = {
            "n_calls": step_calls[step],
            "total_usd": _r6(total),
            "total_usd_batch_eq": _r6(total_batch),
            "mean_usd_per_episode": _r6(total / n) if n else None,
            "usd_per_1000_episodes": _r6(total / n * 1000) if n else None,
            "usd_batch_eq_per_1000_episodes": _r6(total_batch / n * 1000) if n else None,
        }
    return {
        "n_episodes": n,
        "n_receipts": len(chosen),
        "n_cache_hits": sum(1 for r in receipts if _flag(r.get("cache_hit"))),
        "cache_hits_included": include_cache_hits,
        "total_usd": _r6(math.fsum(per_ep)),
        "total_usd_batch_eq": _r6(math.fsum(per_ep_batch)),
        "per_episode": [
            {"episode_key": ep, "usd": _r6(a), "usd_batch_eq": _r6(b)}
            for ep, a, b in zip(episodes, per_ep, per_ep_batch, strict=True)
        ],
        "p50_usd_per_episode": _percentile(per_ep, 50),
        "mean_usd_per_episode": _r6(mean),
        "usd_per_1000_episodes": _r6(None if mean is None else mean * 1000),
        "p50_usd_batch_eq_per_episode": _percentile(per_ep_batch, 50),
        "mean_usd_batch_eq_per_episode": _r6(mean_batch),
        "usd_batch_eq_per_1000_episodes": _r6(None if mean_batch is None else mean_batch * 1000),
        "by_step": by_step,
        "unattributed_usd": _r6(math.fsum(unattributed)),
        "unattributed_usd_batch_eq": _r6(math.fsum(unattributed_batch)),
        "n_unattributed_receipts": len(unattributed),
        "n_receipts_missing_usd": missing_usd,
        "n_receipts_without_batch_eq": without_batch,
    }


# --------------------------------------------------------------------------- #
# K2: tokens
# --------------------------------------------------------------------------- #
def _tokens(receipt: Mapping[str, Any]) -> dict[str, int | None]:
    """Token counts by kind; None when the receipt does not know the count (receipts.py stores null)."""
    usage = receipt.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    out: dict[str, int | None] = {}
    for kind in TOKEN_KINDS:
        value = _num(usage.get(USAGE_KEYS[kind]))
        out[kind] = None if value is None else int(value)
    return out


def k2(receipts: Sequence[Mapping[str, Any]], include_cache_hits: bool = True,
       episode_keys: Iterable[str] | None = None) -> dict[str, Any]:
    """K2: tokens per episode split into input text, image, video, audio, cached, output, reasoning.

    Unknown (null or missing) usage fields count as 0 in the sums; ``n_receipts_unknown`` gives,
    per kind, how many of the counted receipts did not know that count, so a 0 is not mistaken for
    a measured zero. The episode set follows the same rule as ``k1``.
    """
    receipts = list(receipts)
    episodes = _episode_list(receipts, episode_keys)
    chosen = _select(receipts, include_cache_hits)
    per_ep: dict[str, dict[str, int]] = {ep: dict.fromkeys(TOKEN_KINDS, 0) for ep in episodes}
    unattributed = dict.fromkeys(TOKEN_KINDS, 0)
    unknown = dict.fromkeys(TOKEN_KINDS, 0)
    for r in chosen:
        key = _episode_key(r)
        bucket = per_ep.get(key) if key is not None else None
        if bucket is None:
            bucket = unattributed
        for kind, value in _tokens(r).items():
            if value is None:
                unknown[kind] += 1
            else:
                bucket[kind] += value
    n = len(episodes)
    totals = {kind: sum(per_ep[ep][kind] for ep in episodes) for kind in TOKEN_KINDS}
    return {
        "n_episodes": n,
        "n_receipts": len(chosen),
        "cache_hits_included": include_cache_hits,
        "kinds": list(TOKEN_KINDS),
        "per_episode": [{"episode_key": ep, **per_ep[ep]} for ep in episodes],
        "total": totals,
        "mean_per_episode": {kind: (_r6(totals[kind] / n) if n else None) for kind in TOKEN_KINDS},
        "p50_per_episode": {
            kind: _percentile([per_ep[ep][kind] for ep in episodes], 50) for kind in TOKEN_KINDS
        },
        "unattributed": unattributed,
        "n_receipts_unknown": unknown,
    }


# --------------------------------------------------------------------------- #
# K3: latency
# --------------------------------------------------------------------------- #
def k3(receipts: Sequence[Mapping[str, Any]], episode_wall_s: Mapping[str, float] | None,
       include_cache_hits: bool = True) -> dict[str, Any]:
    """K3: per call p50 and p95 of ``latency_s`` (and of ``wall_s`` when receipts carry it), per step
    too; per episode p50 and p95 of the sync wall clock given in ``episode_wall_s`` (episode key to
    seconds from first call to last result). ``include_cache_hits`` applies to the per-call numbers.
    """
    receipts = list(receipts)
    chosen = _select(receipts, include_cache_hits)
    latency: list[float] = []
    call_wall: list[float] = []
    step_latency: dict[str, list[float]] = {}
    for r in chosen:
        lat = _num(r.get("latency_s"))
        if lat is not None:
            latency.append(lat)
            step_latency.setdefault(_step(r), []).append(lat)
        wall = _num(r.get("wall_s"))
        if wall is not None:
            call_wall.append(wall)
    walls: list[float] = []
    if episode_wall_s:
        for key in sorted(episode_wall_s, key=episode_sort_key):
            value = _num(episode_wall_s[key])
            if value is not None:
                walls.append(value)
    return {
        "cache_hits_included": include_cache_hits,
        "n_calls_with_latency": len(latency),
        "latency_p50_s": _percentile(latency, 50),
        "latency_p95_s": _percentile(latency, 95),
        "n_calls_with_wall": len(call_wall),
        "call_wall_p50_s": _percentile(call_wall, 50),
        "call_wall_p95_s": _percentile(call_wall, 95),
        "by_step": {
            step: {"n_calls": len(vals), "latency_p50_s": _percentile(vals, 50),
                   "latency_p95_s": _percentile(vals, 95)}
            for step, vals in sorted(step_latency.items())
        },
        "n_episodes_wall": len(walls),
        "episode_wall_p50_s": _percentile(walls, 50),
        "episode_wall_p95_s": _percentile(walls, 95),
    }
