"""Build the ``inspect_data.json`` the verification viewer renders (zero-API).

Two modes:
  * ``--from-eval``: reconstruct the SO-101 tracks (gold, S0-Flash, grounded=S2-Pro,
    S_grip, uniform-fifths) from the sweep's cached receipts + the frozen gold.
  * ``--from-annotations``: build tracks from one or more fresh annotation parquets
    (used for the Part-2 blind trial; pass ``--blind`` to hide track identity).

The network call is monkeypatched to raise, so reconstruction can only read cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from robolabel.episode import Episode  # noqa: E402
from robolabel.gate import is_degenerate_single_segment, is_uniform_split  # noqa: E402
from robolabel.inspect_data import assemble, build_episode, segments_from_records  # noqa: E402
from robolabel.labelers.gripper_baseline import segment_from_state  # noqa: E402
from robolabel.labelers.segmentation import segment_episode  # noqa: E402
from robolabel.rubric import load_rubric  # noqa: E402
from robolabel.schema import episode_records, list_episode_ids, read_annotations  # noqa: E402
from robolabel.strategy import load_strategy  # noqa: E402

CANON = ["approach", "grasp", "transport", "release-place", "retract"]


def _zero_api():
    import robolabel.providers.gemini as g

    def _blocked(*a, **k):
        raise RuntimeError("zero-API guard: cache miss")
    g.requests.post = _blocked


def _dummy_ep(eid, num_frames, task):
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    return Episode(episode_id=eid, num_frames=int(num_frames), fps=30.0, task=task, get_frame=lambda i: arr)


def _segs(segments) -> list[dict]:
    return [{"start": s.start_frame, "end": s.end_frame, "phase": s.phase,
             "target": getattr(s, "target", None),
             "text": s.subtask_text, "evidence": s.evidence} for s in segments]


def _uniform5(n):
    last = max(0, n - 1)
    edges = [round(i * last / 5) for i in range(6)]
    out = []
    for i in range(5):
        start = 0 if i == 0 else edges[i] + 1
        end = last if i == 4 else edges[i + 1]
        out.append({"start": start, "end": min(max(start, end), last), "phase": CANON[i],
                    "text": f"{CANON[i]} (uniform)", "evidence": None})
    out[-1]["end"] = last
    return out


def _reconstruct(eval_out, model_dir, model_name, strat, eps, rubric):
    from robolabel.providers.base import build_provider
    prov = build_provider("gemini", model_name)
    cfg = replace(load_strategy(strat), min_granularity_policy="reject", require_target=False)
    out = {}
    for eid, ep in eps.items():
        rdir = Path(eval_out) / model_dir / strat / "raw_receipts" / eid
        if not rdir.exists():
            continue
        try:
            out[eid] = segment_episode(ep, prov, rubric, cfg, rdir).segments
        except Exception:  # noqa: BLE001
            continue
    return out


def _auto_quality(eval_out, model_dir, eid):
    from robolabel.labelers.metadata import _parse_metadata
    from robolabel.providers.base import try_extract_json
    p = Path(eval_out) / model_dir / "metadata" / eid / "metadata_label.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    txt = ""
    for c in d.get("response_json", {}).get("candidates", []):
        for part in c.get("content", {}).get("parts", []):
            txt += part.get("text", "")
    return _parse_metadata(try_extract_json(txt)).quality


def from_eval(args):
    _zero_api()
    rubric = load_rubric()
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    ids = [str(e["episode_id"]) for e in gold["episodes"]]
    eps = {str(e["episode_id"]): _dummy_ep(str(e["episode_id"]), e["num_frames"], e.get("task")) for e in gold["episodes"]}
    states = _load_states(args.dataset)

    s0 = _reconstruct(args.eval_out, "gemini__gemini-2.5-flash", "gemini-2.5-flash", "S0", eps, rubric)
    grounded = _reconstruct(args.eval_out, "gemini__gemini-2.5-pro", "gemini-2.5-pro", "S2", eps, rubric)
    q_auto = {e: _auto_quality(args.eval_out, "gemini__gemini-2.5-pro", e) for e in ids}
    qvals = [v for v in q_auto.values() if v is not None]
    med = sorted(qvals)[len(qvals) // 2] if qvals else 5

    episodes = []
    for e in gold["episodes"]:
        eid = str(e["episode_id"])
        nf = int(e["num_frames"])
        gold_segs = [{"start": s.get("start_frame"), "end": s.get("end_frame"), "phase": None,
                      "text": s.get("subtask_text"), "evidence": None}
                     for s in e["gold"]["subtasks"] if s.get("end_frame") is not None]
        tracks = {"gold": {"segments": gold_segs}}
        if eid in s0:
            tracks["S0-Flash"] = {"segments": _segs(s0[eid])}
        if eid in grounded:
            tracks["grounded"] = {"segments": _segs(grounded[eid])}
        if eid in states:
            tracks["S_grip"] = {"segments": _segs(segment_from_state(states[eid], rubric.gripper_baseline, rubric.phase_vocabulary))}
        tracks["uniform-fifths"] = {"segments": _uniform5(nf)}
        # gate flags on the grounded track
        g_segs = tracks.get("grounded", {}).get("segments", [])
        st = [{"start_frame": s["start"], "end_frame": s["end"]} for s in g_segs]
        flags = []
        if st and is_degenerate_single_segment(st):
            flags.append("degenerate")
        elif st and is_uniform_split(st, 0.12, 3):
            flags.append("uniform_split")
        qa = q_auto.get(eid)
        qg = (e["gold"].get("metadata") or {}).get("quality")
        if qa is not None and med - qa >= 2:
            flags.append("quality_outlier")
        episodes.append(build_episode(eid, nf, 30.0, e.get("task", ""), tracks,
                                      quality={"auto": qa, "gold": qg}, gate_flags=flags))
    payload = assemble(args.dataset, "lerobot",
                       ["gold", "S0-Flash", "grounded", "S_grip", "uniform-fifths"], episodes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(episodes)} episodes, tracks={payload['track_order']}")


def _bands_for(segs):
    st = [{"start_frame": s["start"], "end_frame": s["end"]} for s in segs]
    if st and is_degenerate_single_segment(st):
        return ["degenerate"]
    if st and is_uniform_split(st, 0.12, 3):
        return ["uniform_split"]
    return []


def from_annotations(args):
    rubric = load_rubric()
    states = _load_states(args.dataset) if args.dataset else {}
    track_specs = [s.split("=", 1) for s in args.track]  # name=annotations_dir
    dfs = {name: read_annotations(d) for name, d in track_specs}

    if args.blind:
        # One BLIND ITEM per (episode, strategy): strategy identity hidden, shuffled.
        items = []
        for name, df in dfs.items():
            for eid in list_episode_ids(df):
                rec = episode_records(df, eid)
                segs = segments_from_records(rec["subtasks"])
                items.append({"frame_ep": eid, "_real_ep": eid, "_strategy": name,
                              "num_frames": int(rec["num_frames"]), "fps": 30.0,
                              "task": rec["task"] or "", "tracks": {"model": {"segments": segs}},
                              "gate_flags": _bands_for(segs), "_bands": _bands_for(segs), "sort_iou": 1.0})
        _seeded(args.seed).shuffle(items)
        unblind = {"__dataset__": args.dataset or ""}
        for i, it in enumerate(items):
            iid = f"T{i:03d}"
            it["episode_id"] = iid
            it["n_flags"] = len(it["gate_flags"])
            # denominators for the mark-failures-only tally: the grade panel asks one
            # boundary/phase/evidence question per non-final segment.
            graded = it["tracks"]["model"]["segments"][:-1]
            denom = {"b": len(graded), "p": len(graded),
                     "e": sum(1 for s in graded if s.get("evidence"))}
            unblind[iid] = {"episode_id": it.pop("_real_ep"), "strategy": it.pop("_strategy"),
                            "bands": it.pop("_bands"), "denom": denom}
        payload = assemble(args.dataset or "", "lerobot", ["model"], items, blind=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ub = out.with_suffix(".unblind.json")
        ub.write_text(json.dumps(unblind, indent=2), encoding="utf-8")
        print(f"wrote {out}: {len(items)} blind items ({len(dfs)} strategies × episodes); unblind -> {ub}")
        return

    # non-blind: every strategy as a parallel track per episode (primary = pseudo-gold for sort).
    primary = dfs[track_specs[0][0]]
    track_order = [name for name, _ in track_specs]
    if "S_grip" in args.add_baselines and states:
        track_order.append("S_grip")
    if "uniform-fifths" in args.add_baselines:
        track_order.append("uniform-fifths")
    episodes = []
    for eid in list_episode_ids(primary):
        rec0 = episode_records(primary, eid)
        nf = int(rec0["num_frames"])
        tracks = {name: {"segments": segments_from_records(episode_records(df, eid)["subtasks"])}
                  for name, df in dfs.items()}
        if "S_grip" in track_order and eid in states:
            tracks["S_grip"] = {"segments": _segs(segment_from_state(states[eid], rubric.gripper_baseline, rubric.phase_vocabulary))}
        if "uniform-fifths" in track_order:
            tracks["uniform-fifths"] = {"segments": _uniform5(nf)}
        gname = track_specs[0][0]
        ep = build_episode(eid, nf, 30.0, rec0["task"] or "", {**tracks, "gold": tracks[gname]},
                           gate_flags=_bands_for(tracks[gname]["segments"]))
        ep["tracks"].pop("gold", None)
        episodes.append(ep)
    payload = assemble(args.dataset or "", "lerobot", track_order, episodes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(episodes)} episodes, tracks={track_order}")


def _seeded(seed):
    import random
    return random.Random(seed)


def _load_states(dataset):
    if not dataset:
        return {}
    import glob

    import pandas as pd
    root = Path(os.path.expanduser(f"~/.cache/huggingface/lerobot/{dataset}"))
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        return {}
    frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = {}
    for ep_idx, grp in frames.groupby("episode_index"):
        out[str(int(ep_idx))] = np.stack(grp.sort_values("frame_index")["observation.state"].to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("from-eval")
    e.add_argument("--gold", required=True)
    e.add_argument("--eval-out", default="eval_out")
    e.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    e.add_argument("--out", default="inspect_data/so101.json")
    e.set_defaults(func=from_eval)
    a = sub.add_parser("from-annotations")
    a.add_argument("--track", nargs="+", required=True, help="name=annotations_dir (first is primary)")
    a.add_argument("--dataset", default=None)
    a.add_argument("--add-baselines", nargs="*", default=[])
    a.add_argument("--blind", action="store_true")
    a.add_argument("--seed", type=int, default=20260612)
    a.add_argument("--out", required=True)
    a.set_defaults(func=from_annotations)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
