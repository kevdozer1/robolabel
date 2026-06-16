"""Deterministic per-episode `novelty` (no VLM) — a diversity/coverage signal for curation.

Embed each episode (reusing the retrieval frame embedding — a cheap 12x12 grayscale vector,
averaged over a few evenly-spaced frames), then score novelty as the mean distance to the
episode's k nearest neighbours in the set. Isolated episodes score high (novel/rare); episodes
in a dense cluster score low (redundant). No model, no training-utility claim — see CLAIMS.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .retrieve import _embedding


def episode_embedding(frame_getter: Callable[[int], np.ndarray], num_frames: int,
                      n_samples: int = 5) -> np.ndarray:
    """Average frame embedding over ``n_samples`` evenly-spaced frames of the episode."""
    if num_frames <= 0:
        return np.zeros(144, dtype="float32")
    idxs = np.unique(np.linspace(0, num_frames - 1, n_samples).astype(int))
    vecs = []
    for i in idxs:
        try:
            vecs.append(_embedding(frame_getter(int(i))))
        except Exception:  # noqa: BLE001 - a missing frame just drops out of the average
            continue
    return np.mean(vecs, axis=0) if vecs else np.zeros(144, dtype="float32")


def novelty_scores(emb_by_ep: dict[str, np.ndarray], k: int = 5) -> dict[str, float]:
    """Mean distance to the k nearest other episodes (higher = more novel)."""
    eps = list(emb_by_ep)
    if len(eps) < 2:
        return {e: 0.0 for e in eps}
    dim = max(len(v) for v in emb_by_ep.values())
    M = np.stack([np.resize(emb_by_ep[e], dim) for e in eps])
    out = {}
    kk = min(k, len(eps) - 1)
    for i, e in enumerate(eps):
        d = np.linalg.norm(M - M[i], axis=1)
        d[i] = np.inf
        out[e] = round(float(np.sort(d)[:kk].mean()), 5)
    return out
