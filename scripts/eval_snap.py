"""Validate proprioception-fused grasp/release snapping against the SO-101 human gold.

Deterministic, zero-API: reads the existing pick-place grounded annotations + the dataset's
observation.state (gripper) + the human gold, and reports grasp/release boundary recall@±5 and
MAE, with vs without the snap. Decision input for whether to keep the snap on by default.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from robolabel.control import load_states  # noqa: E402
from robolabel.rubric import load_rubric  # noqa: E402
from robolabel.schema import SubtaskSegment, episode_records, list_episode_ids, read_annotations  # noqa: E402
from robolabel.snap import _kind, gripper_dim, snap_contact_boundaries  # noqa: E402

GOLD = "../robovid_work/so101_gemini/gold.json"
RUN = "run_out/full_pp"
REPO = "lerobot/svla_so101_pickplace"
TOL = 5


def gold_bounds(gold, eid):
    e = gold.get(eid)
    if not e:
        return None
    b = [int(s["end_frame"]) for s in e.get("gold", {}).get("subtasks", []) if s.get("end_frame") is not None]
    return b[:-1] if len(b) > 1 else b


def segs_of(df, eid):
    return [SubtaskSegment(int(s["segment_idx"]), int(s["start_frame"]), int(s["end_frame"]),
                           str(s.get("subtask_text") or ""), phase=s.get("phase"))
            for s in sorted(episode_records(df, eid)["subtasks"], key=lambda r: int(r["segment_idx"]))]


def contact_boundaries(segs):
    """Frames of the grasp-onset / release-onset boundaries (the ones snap targets)."""
    out = []
    for i in range(len(segs) - 1):
        ka, kb = _kind(segs[i].phase), _kind(segs[i + 1].phase)
        if (kb == "grasp" and ka != "grasp") or (kb == "release" and ka != "release"):
            out.append(segs[i].end_frame)
    return out


def score(bounds, gold_b):
    matched, errs = 0, []
    for bd in bounds:
        if gold_b:
            d = min(abs(g - bd) for g in gold_b)
            if d <= TOL:
                matched += 1
                errs.append(d)
    return matched, len(bounds), errs


def main() -> int:
    gold = {str(e["episode_id"]): e for e in json.load(open(GOLD, encoding="utf-8"))["episodes"]}
    states, snames = load_states(REPO)
    gd = gripper_dim(snames, next(iter(states.values())).shape[1] if states else 6)
    gb = load_rubric().gripper_baseline
    window = load_rubric().snap_window
    df = read_annotations(RUN)
    shared = [e for e in list_episode_ids(df) if gold_bounds(gold, e) and e in states]

    m_b = n_b = m_s = n_s = total_snapped = 0
    err_b, err_s = [], []
    for eid in shared:
        gold_b = gold_bounds(gold, eid)
        base = segs_of(df, eid)
        mb, nb, eb = score(contact_boundaries(base), gold_b)
        m_b += mb
        n_b += nb
        err_b += eb
        grip = states[eid][:, gd]
        snapped_segs, ns = snap_contact_boundaries(copy.deepcopy(base), grip,
                                                   window=window, threshold=gb.get("gripper_norm_threshold", 0.5),
                                                   min_spacing=gb.get("min_transition_frames", 8))
        total_snapped += ns
        ms, nsb, es = score(contact_boundaries(snapped_segs), gold_b)
        m_s += ms
        n_s += nsb
        err_s += es

    def fmt(m, n, errs):
        rec = m / n if n else 0.0
        mae = float(np.mean(errs)) if errs else None
        return f"recall@±{TOL}={rec:.3f} ({m}/{n})  MAE={mae:.2f}" if mae is not None else f"recall@±{TOL}={rec:.3f} ({m}/{n})  MAE=–"

    print(f"pick-place grasp/release boundaries vs gold ({len(shared)} eps, window={window}):")
    print(f"  WITHOUT snap: {fmt(m_b, n_b, err_b)}")
    print(f"  WITH snap:    {fmt(m_s, n_s, err_s)}   ({total_snapped} boundaries snapped)")
    improved = (m_s / n_s if n_s else 0) > (m_b / n_b if n_b else 0)
    print(f"  -> recall@±{TOL} {'IMPROVES' if improved else 'does NOT improve'} with snap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
