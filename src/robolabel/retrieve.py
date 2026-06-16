"""Retrieval subgoal: a same-phase end-of-segment keyframe from a DIFFERENT episode.

The real same-episode end-of-sub-step keyframe is the ground-truth subgoal (kept untouched).
This adds an OPTIONAL *retrieved* subgoal — for each grounded segment, the end frame of a
segment with the SAME phase label from another episode — stored alongside the real keyframe
(never replacing it). It exists so policy training/eval can be fed a goal image that the model
did **not** see in this episode, avoiding the "copy the last frame" shortcut. robolabel does
**not** generate subgoal images; this only *selects* a real frame from elsewhere in the dataset.

Selection: nearest same-phase candidate by a cheap frame embedding (downsampled grayscale) when
a frame source is available, else a seeded random same-phase pick (deterministic). When no
same-phase candidate exists in any other episode, the retrieved subgoal is left null (honest).
"""
from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np


def _phase_index(df) -> dict[str, list[tuple[str, int]]]:
    """phase (lowercased) -> [(episode_id, end_frame), ...] across all episodes."""
    idx: dict[str, list[tuple[str, int]]] = {}
    sub = df[df["record_type"] == "subtask"]
    for _, r in sub.iterrows():
        phase = str(r.get("phase") or "").strip().lower()
        if not phase:
            continue
        ef = r.get("end_frame")
        if ef is None or (isinstance(ef, float) and np.isnan(ef)):
            continue
        idx.setdefault(phase, []).append((str(r["episode_id"]), int(ef)))
    return idx


def _seg_phase(df) -> dict[tuple[str, int], str]:
    """(episode_id, segment_idx) -> phase, for subtask rows."""
    out: dict[tuple[str, int], str] = {}
    sub = df[df["record_type"] == "subtask"]
    for _, r in sub.iterrows():
        out[(str(r["episode_id"]), int(r["segment_idx"]))] = str(r.get("phase") or "").strip().lower()
    return out


def _embedding(arr: np.ndarray) -> np.ndarray:
    """Cheap, deterministic frame embedding: 12x12 grayscale, mean-removed, L2-normalized."""
    a = np.asarray(arr).astype("float32")
    if a.ndim == 3:
        a = a.mean(axis=2)
    h, w = a.shape[:2]
    gh, gw = max(1, h // 12), max(1, w // 12)
    small = a[: gh * 12: gh, : gw * 12: gw][:12, :12].reshape(-1)
    small = small - small.mean()
    n = np.linalg.norm(small)
    return small / n if n > 1e-6 else small


def retrieve_subgoals(df, frame_getter: Callable[[str, int], np.ndarray] | None = None,
                      method: str = "random", seed: int = 0,
                      allowed_sources: set[str] | None = None):
    """Write retrieved_subgoal_episode_id/frame_idx into a copy of ``df``.

    ``frame_getter(episode_id, frame_idx) -> ndarray`` enables embedding selection; without it
    (or with ``method="random"``) selection is a seeded random same-phase pick.
    ``allowed_sources`` restricts which episodes a subgoal may be *retrieved from* — pass the
    gate-passed set so a failure-band episode can't poison another episode's subgoal.
    """
    df = df.copy()
    for col in ("retrieved_subgoal_episode_id", "retrieved_subgoal_frame_idx"):
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype("object")
    phase_idx = _phase_index(df)
    seg_phase = _seg_phase(df)
    use_embed = method == "embedding" and frame_getter is not None
    emb_cache: dict[tuple[str, int], np.ndarray] = {}

    def emb(ep: str, fr: int) -> np.ndarray | None:
        key = (ep, fr)
        if key not in emb_cache:
            try:
                emb_cache[key] = _embedding(frame_getter(ep, fr))
            except Exception:  # noqa: BLE001 - missing frame -> treat as no embedding
                emb_cache[key] = None  # type: ignore[assignment]
        return emb_cache[key]

    sg_mask = df["record_type"] == "subgoal"
    for idx in df[sg_mask].index:
        ep = str(df.at[idx, "episode_id"])
        seg = int(df.at[idx, "segment_idx"])
        phase = seg_phase.get((ep, seg), "")
        cands = [(e, f) for (e, f) in phase_idx.get(phase, [])
                 if e != ep and (allowed_sources is None or e in allowed_sources)]
        if not cands:
            continue
        if use_embed:
            q = emb(ep, int(df.at[idx, "subgoal_frame_idx"]))
            scored = [(e, f, emb(e, f)) for (e, f) in cands]
            scored = [(e, f, v) for (e, f, v) in scored if v is not None]
            if q is not None and scored:
                e, f, _ = min(scored, key=lambda t: float(np.linalg.norm(q - t[2])))
            else:
                e, f = random.Random(f"{seed}:{ep}:{seg}").choice(cands)
        else:
            e, f = random.Random(f"{seed}:{ep}:{seg}").choice(cands)
        df.at[idx, "retrieved_subgoal_episode_id"] = e
        df.at[idx, "retrieved_subgoal_frame_idx"] = int(f)
    return df
