"""``robolabel query`` — the usefulness path: do something with the annotations.

Two canned queries that prove the metadata is actually queryable:

* ``phase_contact_sheet`` — retrieve every segment with a given phase (e.g. every
  ``grasp`` across all episodes) and tile a representative frame from each into a
  contact-sheet PNG. The visceral "the labels mean something" proof.
* ``needs_review_episodes`` — every episode the gate flags ``needs_review`` (a
  hallucinated low quality score), worst first. The "the gate is a filter you can
  act on" proof.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .gate import run_gate
from .schema import episode_records, list_episode_ids, read_annotations


def _clean_target(v: object) -> str | None:
    """None for missing/empty/NaN targets (empty parquet cells read back as float NaN)."""
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none") else None


def find_phase_segments(annotations_dir: str | Path, phase: str) -> list[dict[str, Any]]:
    """All subtask segments whose phase matches ``phase`` (case-insensitive), across episodes."""
    df = read_annotations(annotations_dir)
    want = phase.strip().lower()
    hits = []
    for eid in list_episode_ids(df):
        rec = episode_records(df, eid)
        for s in rec["subtasks"]:
            if str(s.get("phase") or "").strip().lower() == want:
                start, end = int(s.get("start_frame") or 0), int(s.get("end_frame") or 0)
                hits.append({"episode_id": eid, "segment_idx": int(s.get("segment_idx") or 0),
                             "start_frame": start, "end_frame": end, "mid_frame": (start + end) // 2,
                             "text": str(s.get("subtask_text") or ""),
                             "target": _clean_target(s.get("target")),
                             "evidence": s.get("boundary_evidence")})
    return hits


def phase_contact_sheet(annotations_dir: str | Path, phase: str, *, source=None,
                        out: str | Path | None = None, limit: int = 24, thumb: int = 200) -> dict[str, Any]:
    """Tile the representative frame of each phase-matching segment into a contact sheet."""
    hits = find_phase_segments(annotations_dir, phase)
    result: dict[str, Any] = {"phase": phase, "n_segments": len(hits),
                              "episodes": sorted({h["episode_id"] for h in hits})}
    if source is None or out is None:
        result["note"] = "pass --source/--target and --out to render the montage PNG"
        return result
    episodes = {ep.episode_id: ep for ep in source}
    shown = hits[:limit]
    tiles: list[Image.Image] = []
    for h in shown:
        ep = episodes.get(h["episode_id"])
        if ep is None:
            continue
        arr = ep.frame(h["mid_frame"])
        img = Image.fromarray(np.asarray(arr).astype("uint8")).convert("RGB")
        ratio = thumb / img.width
        img = img.resize((thumb, max(1, int(img.height * ratio))))
        canvas = Image.new("RGB", (img.width, img.height + 18), "white")
        canvas.paste(img, (0, 18))
        cap = f"ep{h['episode_id']} f{h['mid_frame']}" + (f" → {h['target']}" if h.get("target") else "")
        ImageDraw.Draw(canvas).text((4, 4), cap, fill=(0, 0, 0))
        tiles.append(canvas)
    if not tiles:
        result["note"] = "no frames rendered (no matching segments or no source frames)"
        return result
    cols = min(6, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].width, tiles[0].height
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    result["out"] = str(out)
    result["rendered"] = len(tiles)
    return result


def needs_review_episodes(annotations_dir: str | Path) -> list[dict[str, Any]]:
    """Episodes the gate flags needs_review (quality outliers), worst-first."""
    report = run_gate(annotations_dir)
    df = read_annotations(annotations_dir)
    out = []
    for issue in report.issues:
        if issue.check != "quality_outlier_needs_review":
            continue
        rec = episode_records(df, issue.episode_id)
        q = rec["metadata"].get("quality")
        out.append({"episode_id": issue.episode_id, "quality": _int(q),
                    "task": rec["task"], "detail": issue.detail})
    out.sort(key=lambda r: (r["quality"] if r["quality"] is not None else 99))
    return out


def _int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
