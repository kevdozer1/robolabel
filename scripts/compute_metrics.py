"""Zero-API supplementary metrics over the completed ablation.

Reconstructs every cell's segments from the **cached receipts** the sweep already
wrote (the network call is monkeypatched to raise, so this can only read cache — a
hard zero-API guarantee), then adds metrics the first pass didn't compute:

* the **uniform-fifths trivial baseline** (5 equal segments, canonical phases),
* **distributional IoU** (mean / median / p10) and a **per-band IoU breakdown**,
* **boundary placement**: precision / recall of predicted boundaries within ±5
  frames of a gold boundary (greedy match) + mean absolute frame error on matches.

Frames are dummies (cache-hits ignore image content); `num_frames` comes from the
gold file. Existing report numbers are never altered — this only writes
`eval_metrics/metrics.json` for the report's *added* sections.

    python scripts/compute_metrics.py --gold ../robovid_work/so101_gemini/gold.json \
        --split eval/so101_split.json --eval-out eval_out --out eval_metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import eval_strategies as ev  # noqa: E402

from robovid_conditioner.episode import Episode  # noqa: E402
from robovid_conditioner.gate import is_degenerate_single_segment, is_uniform_split  # noqa: E402
from robovid_conditioner.labelers.gripper_baseline import segment_from_state  # noqa: E402
from robovid_conditioner.labelers.segmentation import segment_episode  # noqa: E402
from robovid_conditioner.rubric import load_rubric  # noqa: E402
from robovid_conditioner.schema import SubtaskSegment  # noqa: E402
from robovid_conditioner.strategy import load_strategy  # noqa: E402

CANON = ["approach", "grasp", "transport", "release-place", "retract"]


def _enforce_zero_api():
    """Make any un-cached Gemini call raise, so reconstruction can only read cache."""
    import robovid_conditioner.providers.gemini as g

    def _blocked(*a, **k):
        raise RuntimeError("zero-API guard: cache miss (no network)")
    g.requests.post = _blocked


def _dummy_episode(eid: str, num_frames: int, task: str | None) -> Episode:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    return Episode(episode_id=eid, num_frames=int(num_frames), fps=30.0, task=task,
                   get_frame=lambda i: arr)


def uniform_fifths(num_frames: int) -> list[SubtaskSegment]:
    last = max(0, num_frames - 1)
    edges = [round(i * last / 5) for i in range(6)]
    segs = []
    for i in range(5):
        start = 0 if i == 0 else edges[i] + 1
        end = last if i == 4 else edges[i + 1]
        end = min(max(start, end), last)
        segs.append(SubtaskSegment(i, start, end, f"{CANON[i]} (uniform fifths)", phase=CANON[i]))
    segs[-1].end_frame = last
    return segs


# --------------------------------------------------------------------------- #
# Boundary placement metrics
# --------------------------------------------------------------------------- #
def _boundaries(segs) -> list[int]:
    return [int(s.end_frame if hasattr(s, "end_frame") else s["end_frame"]) for s in segs[:-1]]


def boundary_pr_mae(pred: list[int], gold: list[int], tol: int = 5) -> tuple[int, int, int, list[int]]:
    """Greedy match each gold boundary to the nearest unused pred within tol."""
    used = [False] * len(pred)
    errs: list[int] = []
    for g in sorted(gold):
        best, bd = -1, tol + 1
        for j, p in enumerate(pred):
            if used[j]:
                continue
            d = abs(p - g)
            if d <= tol and d < bd:
                best, bd = j, d
        if best >= 0:
            used[best] = True
            errs.append(bd)
    matched = len(errs)
    return matched, len(pred), len(gold), errs


def _agg_boundary(per_ep_bounds: list[tuple[list[int], list[int]]], tol: int = 5) -> dict:
    tp = fp_denom = fn_denom = 0
    all_errs: list[int] = []
    for pred, gold in per_ep_bounds:
        m, npred, ngold, errs = boundary_pr_mae(pred, gold, tol)
        tp += m
        fp_denom += npred
        fn_denom += ngold
        all_errs += errs
    prec = tp / fp_denom if fp_denom else None
    rec = tp / fn_denom if fn_denom else None
    return {"precision_pm5": _r(prec), "recall_pm5": _r(rec),
            "mae_frames": _r(mean(all_errs)) if all_errs else None, "n_matched": tp}


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_vlm(eval_out: Path, model_dir: str, model_name: str, strategy: str,
                    episodes: dict, rubric) -> dict[str, list]:
    from robovid_conditioner.providers.base import build_provider
    provider = build_provider("gemini", model_name)
    cfg = load_strategy(strategy)
    segs: dict[str, list] = {}
    for eid, ep in episodes.items():
        rdir = eval_out / model_dir / strategy / "raw_receipts" / eid
        if not rdir.exists():
            continue
        try:
            res = segment_episode(ep, provider, rubric, cfg, rdir)
            segs[eid] = res.segments
        except Exception:  # noqa: BLE001 - cache miss / dropped episode -> skip
            continue
    return segs


def metrics_for(segs_by_ep: dict[str, list], gold: dict, ids: list[str], rubric) -> dict:
    ids = [e for e in ids if e in segs_by_ep]
    if not ids:
        return {"n": 0}
    auto = {e: ev.segments_to_auto(segs_by_ep[e], _numframes(gold, e)) for e in ids}
    rep = ev.score_against_gold(ev.build_eval_gold(gold, auto, {e: None for e in ids}, ids))
    per = {p["episode_id"]: p["boundary_iou_mean"] for p in rep["per_episode"] if p["boundary_iou_mean"] is not None}
    ious = list(per.values())
    # per-band IoU
    bands = {"degenerate": [], "uniform_split": [], "drifted_or_ok": []}
    bounds: list[tuple[list[int], list[int]]] = []
    for e in ids:
        st = [{"start_frame": s.start_frame, "end_frame": s.end_frame} for s in segs_by_ep[e]]
        band = ("degenerate" if is_degenerate_single_segment(st)
                else "uniform_split" if is_uniform_split(st, 0.12, 3) else "drifted_or_ok")
        if e in per:
            bands[band].append(per[e])
        bounds.append((_boundaries(segs_by_ep[e]), _gold_bounds(gold, e)))
    return {
        "n": len(ids),
        # Flat per-segment mean — matches the headline IoU in STRATEGY_REPORT.md
        # (reliability_report's definition); used to verify reconstruction fidelity.
        "iou_flat_mean": _r(rep["subtask_boundary_temporal_iou_mean"]),
        # Per-episode IoU distribution (mean/median/p10 over episodes).
        "iou_mean": _r(mean(ious)) if ious else None,
        "iou_median": _r(median(ious)) if ious else None,
        "iou_p10": _r(float(np.percentile(ious, 10))) if ious else None,
        "per_band_iou": {k: {"n": len(v), "iou_mean": _r(mean(v)) if v else None} for k, v in bands.items()},
        "boundary": _agg_boundary(bounds),
    }


def _numframes(gold, eid):
    for e in gold["episodes"]:
        if str(e["episode_id"]) == str(eid):
            return int(e["num_frames"])
    return 0


def _gold_bounds(gold, eid):
    for e in gold["episodes"]:
        if str(e["episode_id"]) == str(eid):
            gl = e["gold"]["subtasks"]
            return [int(s["end_frame"]) for s in gl[:-1] if s.get("end_frame") is not None]
    return []


def _r(x):
    return None if x is None else round(float(x), 4)


def run(args):
    _enforce_zero_api()
    rubric = load_rubric()
    gold = json.load(open(args.gold, encoding="utf-8"))
    split = json.load(open(args.split, encoding="utf-8"))
    eval_out = Path(args.eval_out)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # dummy episodes by id (num_frames from gold)
    eps_all = {str(e["episode_id"]): _dummy_episode(str(e["episode_id"]), e["num_frames"], e.get("task"))
               for e in gold["episodes"]}
    states = _load_states(args.dataset) if args.dataset else {}

    MODELS = [("gemini__gemini-2.5-flash", "gemini-2.5-flash", "Flash"),
              ("gemini__gemini-2.5-pro", "gemini-2.5-pro", "Pro")]
    results = {"tune": [], "test": []}
    # Which cells exist per phase: all VLM cells on tune; the test cells run = {Flash S0, Pro S2}.
    TEST_CELLS = {("gemini__gemini-2.5-flash", "S0"), ("gemini__gemini-2.5-pro", "S2")}

    for phase in ("tune", "test"):
        ids = [str(x) for x in split[phase]]
        eps = {e: eps_all[e] for e in ids}
        for md, mname, label in MODELS:
            for strat in ["S0", "S1", "S2", "S3", "S4"]:
                if phase == "test" and (md, strat) not in TEST_CELLS:
                    continue
                segs = reconstruct_vlm(eval_out, md, mname, strat, eps, rubric)
                m = metrics_for(segs, gold, ids, rubric)
                m.update({"model": f"gemini/{mname}", "strategy": strat, "phase": phase})
                results[phase].append(m)
                print(f"  [{phase}] {label} {strat}: n={m.get('n')} flat={m.get('iou_flat_mean')} "
                      f"epMean={m.get('iou_mean')} med={m.get('iou_median')} p10={m.get('iou_p10')} "
                      f"bP={m['boundary']['precision_pm5']} bR={m['boundary']['recall_pm5']} "
                      f"MAE={m['boundary']['mae_frames']}", file=sys.stderr)
        # baselines
        for name, segfn in [("uniform5", lambda e: uniform_fifths(_numframes(gold, e))),
                            ("S_grip", lambda e: segment_from_state(states[e], rubric.gripper_baseline,
                                                                    rubric.phase_vocabulary) if e in states else None)]:
            segs = {e: segfn(e) for e in ids if segfn(e) is not None}
            m = metrics_for(segs, gold, ids, rubric)
            m.update({"model": "baseline", "strategy": name, "phase": phase})
            results[phase].append(m)
            print(f"  [{phase}] {name}: n={m.get('n')} IoU={m.get('iou_mean')} med={m.get('iou_median')} "
                  f"p10={m.get('iou_p10')} bP={m['boundary']['precision_pm5']} "
                  f"bR={m['boundary']['recall_pm5']} MAE={m['boundary']['mae_frames']}", file=sys.stderr)

    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/metrics.json")


def _load_states(dataset: str) -> dict:
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
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split", default="eval/so101_split.json")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--eval-out", default="eval_out")
    ap.add_argument("--out", default="eval_metrics")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
