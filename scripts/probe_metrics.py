"""Gold-free metrics for the cross-task probe (Phase B): failure-band rate, segment-count
distribution, target-present rate, and phase-vocabulary fit (closed-S2 "other"-coercion
rate vs open-vocab free-text phases). Zero-API; reads the probe annotation parquets.

    python scripts/probe_metrics.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robolabel.gate import is_degenerate_single_segment, is_uniform_split  # noqa: E402
from robolabel.inspect_data import segments_from_records  # noqa: E402
from robolabel.rubric import load_rubric  # noqa: E402
from robolabel.schema import episode_records, list_episode_ids, read_annotations  # noqa: E402

VOCAB = set(load_rubric().phase_vocabulary)
TASKS = ["pour", "fold"]
CONDS = [("s2_closed", "closed-vocab S2"), ("s2_open", "open-vocab S2-open")]


def analyze(d: Path) -> dict:
    df = read_annotations(d)
    ids = sorted(list_episode_ids(df), key=lambda x: int(x))
    degen = usplit = 0
    seg_counts, n_target_req, n_target_have, other_phases = [], 0, 0, 0
    phase_samples = []
    for eid in ids:
        segs = segments_from_records(episode_records(df, eid)["subtasks"])
        st = [{"start_frame": s["start"], "end_frame": s["end"]} for s in segs]
        d1 = bool(st and is_degenerate_single_segment(st))
        d2 = bool(st and not d1 and is_uniform_split(st, 0.12, 3))
        degen += d1
        usplit += d2
        seg_counts.append(len(segs))
        for s in segs:
            ph = (s.get("phase") or "").lower()
            if ph not in ("retract",) and "withdraw" not in ph and "retreat" not in ph:
                n_target_req += 1
                if s.get("target"):
                    n_target_have += 1
            if ph in VOCAB and ph not in ("other",):
                pass
            if ph == "other":
                other_phases += 1
            phase_samples.append(s.get("phase") or "")
    n_seg_total = sum(seg_counts)
    return {
        "episodes": len(ids),
        "failure_band": degen + usplit,
        "degenerate": degen,
        "uniform_split": usplit,
        "seg_counts": seg_counts,
        "seg_mean": round(mean(seg_counts), 2) if seg_counts else 0,
        "target_req": n_target_req,
        "target_have": n_target_have,
        "target_rate": round(n_target_have / n_target_req, 3) if n_target_req else None,
        "n_seg_total": n_seg_total,
        "other_phase_count": other_phases,
        "other_phase_rate": round(other_phases / n_seg_total, 3) if n_seg_total else None,
        "phase_vocab": [p for p, _ in Counter(phase_samples).most_common(14)],
    }


def main() -> int:
    out = {}
    for task in TASKS:
        out[task] = {}
        for cond, label in CONDS:
            d = Path(f"probe_{task}") / cond
            if not (d / "annotations.parquet").exists():
                print(f"[skip] {d} (no parquet)", file=sys.stderr)
                continue
            m = analyze(d)
            out[task][cond] = m
            print(f"\n## {task} / {label}")
            print(f"  episodes={m['episodes']}  failure-band={m['failure_band']}/{m['episodes']} "
                  f"(degen={m['degenerate']}, uniform={m['uniform_split']})")
            print(f"  seg-count mean={m['seg_mean']} dist={m['seg_counts']}")
            print(f"  target-present={m['target_have']}/{m['target_req']} ({m['target_rate']})")
            print(f"  phase 'other'-coercion={m['other_phase_count']}/{m['n_seg_total']} ({m['other_phase_rate']})")
            print(f"  phases seen: {m['phase_vocab']}")
    Path("probe_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote probe_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
