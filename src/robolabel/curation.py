"""Curation value + value-tiered overlay (optional, deterministic, honest-status).

A single per-episode ``value = f(quality, novelty)`` (min-max normalized, weights from the run
config). Curation then either selects a top-cut (``keep``/``cut``) or, if ``compress`` is on,
assigns a fidelity **tier** (``full`` / ``reduced`` / ``minimal``) — a value-tiered *overlay*
that a downstream loader could honour by storing/decoding low-value episodes at lower fidelity.
It is an annotation, **never** a deletion or an in-place re-encode: robolabel writes the tier,
nothing else.

Precedents (docs): the Smart Black Box (value-driven tiered storage/compression of recordings)
and the "train on the most valuable ~20% of data" line of dataset-curation work. The machinery
is sound and precedented; the downstream training utility is UNVALIDATED here — see CLAIMS.
"""
from __future__ import annotations

import numpy as np


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 0.5 for k in d}        # no spread -> neutral 0.5 (honest: nothing to rank on)
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def curation_values(quality_by_ep: dict[str, float | None], novelty_by_ep: dict[str, float],
                    w_quality: float = 0.5, w_novelty: float = 0.5) -> dict[str, float]:
    """value in [0,1] = weighted blend of min-max-normalized quality and novelty."""
    qn = _minmax({e: float(q) for e, q in quality_by_ep.items() if q is not None})
    nn = _minmax(novelty_by_ep)
    s = (w_quality + w_novelty) or 1.0
    out = {}
    for e in quality_by_ep:
        out[e] = round((w_quality * qn.get(e, 0.5) + w_novelty * nn.get(e, 0.5)) / s, 4)
    return out


def assign_tiers(values: dict[str, float], *, compress: bool = False,
                 top_cut: float | None = None) -> dict[str, str | None]:
    """compress -> full/reduced/minimal by value tercile; else top_cut -> keep/cut; else None."""
    if not values:
        return {}
    vals = list(values.values())
    if compress:
        if len(set(vals)) < 2:
            return {e: "full" for e in values}
        lo, hi = float(np.quantile(vals, 1 / 3)), float(np.quantile(vals, 2 / 3))
        return {e: ("minimal" if v <= lo else "full" if v >= hi else "reduced") for e, v in values.items()}
    if top_cut is not None:
        thr = float(np.quantile(vals, max(0.0, 1.0 - top_cut)))
        return {e: ("keep" if v >= thr else "cut") for e, v in values.items()}
    return {e: None for e in values}
