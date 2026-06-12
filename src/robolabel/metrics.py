"""Boundary-quality metrics shared by the eval harness and the inspect viewer.

A *boundary* is the end-frame of every segment except the last (the internal
transition points). All functions take segments as plain ``{"start","end"}`` dicts
so they work on gold, VLM, and baseline tracks alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean


def temporal_iou(a: dict, g: dict) -> float | None:
    """Inclusive-frame temporal IoU of two ``{start,end}`` segments."""
    a0, a1, g0, g1 = a.get("start"), a.get("end"), g.get("start"), g.get("end")
    if None in (a0, a1, g0, g1):
        return None
    inter = max(0, min(a1, g1) - max(a0, g0) + 1)
    union = max(a1, g1) - min(a0, g0) + 1
    return inter / union if union > 0 else None


def episode_iou(auto: Sequence[dict], gold: Sequence[dict]) -> float | None:
    """Mean index-aligned per-segment IoU (the reliability-report definition)."""
    ious = [iou for a, g in zip(auto, gold, strict=False) if (iou := temporal_iou(a, g)) is not None]
    return mean(ious) if ious else None


def boundaries(segs: Sequence[dict]) -> list[int]:
    """Internal transition frames = end-frame of every segment but the last."""
    return [int(s["end"]) for s in list(segs)[:-1] if s.get("end") is not None]


def boundary_pr_mae(pred: Sequence[int], gold: Sequence[int], tol: int = 5) -> dict:
    """Greedy-match predicted boundaries to gold within ``tol`` frames.

    Returns precision, recall, mean-abs-frame-error on matches, and counts.
    """
    pred = list(pred)
    used = [False] * len(pred)
    errs: list[int] = []
    for gb in sorted(gold):
        best, bd = -1, tol + 1
        for j, pb in enumerate(pred):
            if used[j]:
                continue
            d = abs(pb - gb)
            if d <= tol and d < bd:
                best, bd = j, d
        if best >= 0:
            used[best] = True
            errs.append(bd)
    matched = len(errs)
    return {
        "matched": matched,
        "n_pred": len(pred),
        "n_gold": len(list(gold)),
        "precision": matched / len(pred) if pred else None,
        "recall": matched / len(list(gold)) if gold else None,
        "mae": mean(errs) if errs else None,
    }
