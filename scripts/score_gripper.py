"""Score the S_grip proprioceptive baseline against the human gold set.

Zero-API: reads the robot's own ``observation.state`` from the dataset's data parquet
(no VLM, no video decode), runs the gripper-event segmenter, and scores it with the
**same** reliability code as the VLM strategies. Writes to ``eval_out_grip/`` so it
never touches the running VLM sweep's ``eval_out/``.

Per the run rules: tune-only by default; the single test-set run is deferred until the
VLM sweep finishes (pass ``--phase test`` then, alongside the chosen VLM cell).

    python scripts/score_gripper.py --gold ../robovid_work/so101_gemini/gold.json \
        --split eval/so101_split.json --dataset lerobot/svla_so101_pickplace \
        --phase tune --out eval_out_grip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_strategies as ev  # noqa: E402
import numpy as np  # noqa: E402

from robolabel.labelers.gripper_baseline import segment_from_state  # noqa: E402
from robolabel.rubric import load_rubric  # noqa: E402


def _load_states(dataset: str, episode_ids: list[str]) -> dict[str, np.ndarray]:
    """Read per-episode ``observation.state`` arrays straight from the data parquet."""
    import glob

    import pandas as pd
    root = Path(os.path.expanduser(f"~/.cache/huggingface/lerobot/{dataset}"))
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No data parquet under {root}/data (is the dataset cached?)")
    frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    wanted = {int(e) for e in episode_ids}
    out: dict[str, np.ndarray] = {}
    for ep_idx, grp in frames[frames["episode_index"].isin(wanted)].groupby("episode_index"):
        grp = grp.sort_values("frame_index")
        out[str(int(ep_idx))] = np.stack(grp["observation.state"].to_numpy())
    return {str(e): out[str(e)] for e in episode_ids if str(e) in out}


def run(args: argparse.Namespace) -> None:
    rubric = load_rubric(args.rubric)
    cfg = rubric.gripper_baseline
    vocab = rubric.phase_vocabulary
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    episode_ids = list(split[args.phase])
    if args.limit:
        episode_ids = episode_ids[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    states = _load_states(args.dataset, episode_ids)
    segs_by_ep = {eid: segment_from_state(st, cfg, vocab) for eid, st in states.items()}
    nseg = [len(s) for s in segs_by_ep.values()]
    auto_by_ep = {eid: ev.segments_to_auto(segs, len(states[eid])) for eid, segs in segs_by_ep.items()}
    # S_grip is segmentation-only: no quality judgment (quality columns are n/a).
    quality_by_ep = {eid: None for eid in segs_by_ep}
    rep = ev.score_against_gold(ev.build_eval_gold(gold, auto_by_ep, quality_by_ep, list(segs_by_ep)))
    row = {
        "model": "proprioceptive/none", "strategy": "S_grip", "phase": args.phase,
        "n": len(episode_ids), "n_scored": len(segs_by_ep), "episodes_failed": [],
        "reportable": len(segs_by_ep) >= min(25, len(episode_ids)),
        "boundary_iou": rep["subtask_boundary_temporal_iou_mean"],
        "quality_exact": None, "quality_within_one": None,
        "subgoal_agreement": rep["subgoal_frame_agreement"],
        "cost_per_episode_usd": 0.0,
        "mean_segments": round(mean(nseg), 2) if nseg else 0,
        "bands": ev.count_bands(segs_by_ep, rubric),
    }
    (out / f"results_{args.phase}.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    # Per-episode segments for the qualitative exhibits in the writeup.
    ev._checkpoint_segments(out / "segments.json", segs_by_ep)
    print(json.dumps(row, indent=2))
    print(f"\nWrote {out}/results_{args.phase}.json (S_grip, zero-API)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split", default="eval/so101_split.json")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--phase", choices=["tune", "test"], default="tune")
    ap.add_argument("--rubric", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="eval_out_grip")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
