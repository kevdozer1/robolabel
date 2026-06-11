"""Strategy ablation harness.

Runs each annotation strategy (S0..S4) and model over the frozen tune/test split
and scores every (strategy, model) cell **with the existing reliability code**
(``reliability_report``) against the human gold file. Produces the table in
``STRATEGY_REPORT.md`` plus a machine-readable results JSON.

Evaluation hygiene:
  * The split (``eval/so101_split.json``) is frozen and seeded. Iterate strategies
    on ``--phase tune`` only; score the chosen strategy on ``--phase test`` once.
  * Quality metadata is strategy-independent, so it is labeled once per (model,
    episode) and reused across strategies — strategies only move boundaries.
  * Cost per episode = strategy segmentation calls + the shared metadata calls.

Resumable + cost-safe: the Gemini provider caches receipts per path, so re-running
reuses completed calls for free; segments are checkpointed per (model, strategy).

Examples
--------
Offline plumbing check (no API, synthetic frames):
    python scripts/eval_strategies.py --self-test

Live ablation on the SO-101 tune split:
    python scripts/eval_strategies.py \
        --gold robovid_work/so101_gemini/gold.json \
        --split eval/so101_split.json \
        --dataset lerobot/svla_so101_pickplace \
        --camera-key observation.images.side \
        --phase tune --strategies S0 S1 S2 S3 S4 \
        --models gemini/gemini-2.5-flash gemini/gemini-2.5-pro \
        --out eval_out --report STRATEGY_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robovid_conditioner.gate import is_degenerate_single_segment, is_uniform_split  # noqa: E402
from robovid_conditioner.labelers.metadata import label_metadata  # noqa: E402
from robovid_conditioner.labelers.segmentation import segment_episode  # noqa: E402
from robovid_conditioner.labelers.subgoals import derive_subgoals  # noqa: E402
from robovid_conditioner.reliability import reliability_report  # noqa: E402
from robovid_conditioner.rubric import Rubric, load_rubric  # noqa: E402
from robovid_conditioner.schema import SubtaskSegment  # noqa: E402
from robovid_conditioner.strategy import load_strategy  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure scoring helpers (unit-tested offline)
# --------------------------------------------------------------------------- #
def segments_to_auto(segments: list[SubtaskSegment], num_frames: int) -> dict:
    """Render strategy segments into a gold-file ``auto`` block (subtasks+subgoals)."""
    subgoals = derive_subgoals(segments, num_frames)
    return {
        "subtasks": [
            {"segment_idx": s.segment_idx, "start_frame": s.start_frame,
             "end_frame": s.end_frame, "subtask_text": s.subtask_text}
            for s in segments
        ],
        "subgoals": [{"segment_idx": sg.segment_idx, "frame_idx": sg.frame_idx} for sg in subgoals],
    }


def build_eval_gold(real_gold: dict, auto_by_ep: dict[str, dict],
                    quality_by_ep: dict[str, int | None], episode_ids: list[str]) -> dict:
    """Splice strategy ``auto`` blocks onto the human ``gold`` blocks for scoring.

    The human gold frames are absolute, so they validly score any strategy's
    boundaries regardless of which ``auto`` they were reviewed against.
    """
    by_id = {str(e["episode_id"]): e for e in real_gold["episodes"]}
    episodes = []
    for eid in episode_ids:
        src = by_id[eid]
        auto = dict(auto_by_ep[eid])
        auto["metadata"] = {"quality": quality_by_ep.get(eid)}
        episodes.append({
            "episode_id": eid, "task": src.get("task"), "num_frames": src.get("num_frames"),
            "auto": auto, "gold": src["gold"], "review_notes": "",
        })
    return {"schema_version": "robovid_conditioner/gold/v1", "episodes": episodes}


def score_against_gold(eval_gold: dict) -> dict:
    """Run the existing reliability report over a spliced eval-gold structure."""
    fd, name = tempfile.mkstemp(suffix=".json")
    os.close(fd)  # Windows: cannot unlink a file with an open handle
    tmp = Path(name)
    try:
        tmp.write_text(json.dumps(eval_gold), encoding="utf-8")
        return reliability_report(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def count_bands(segments_by_ep: dict[str, list[SubtaskSegment]], rubric: Rubric) -> dict[str, int]:
    """Count failure-band membership using the gate detectors."""
    gate = rubric.gate
    cv = float(gate.get("uniform_split_cv_threshold", 0.12))
    min_seg = int(gate.get("min_segments_for_uniform_check", 3))
    degenerate = uniform = 0
    for segs in segments_by_ep.values():
        st = [{"start_frame": s.start_frame, "end_frame": s.end_frame} for s in segs]
        if is_degenerate_single_segment(st):
            degenerate += 1
        elif is_uniform_split(st, cv, min_seg):
            uniform += 1
    total = len(segments_by_ep)
    return {"degenerate": degenerate, "uniform_split": uniform,
            "drifted_or_ok": total - degenerate - uniform}


# --------------------------------------------------------------------------- #
# Live run
# --------------------------------------------------------------------------- #
def _safe(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def _load_episodes(dataset: str, camera_key: str | None, episode_ids: list[str]) -> dict:
    """Load the requested episodes by id.

    We load the FULL dataset (no ``episodes=`` subset) and filter, rather than
    passing a subset: the LeRobot adapter indexes frames by the episode's *global*
    ``dataset_from_index``, which only matches the loaded rows when the dataset is
    full (or a 0-based contiguous prefix). A non-contiguous subset like
    ``[2, 6, 10, ...]`` re-indexes the rows 0..N and the global index runs off the
    end ("Invalid key: 569 out of bounds for size 454"). The dataset is already
    cached locally, so loading it whole costs nothing and decoding stays lazy.
    """
    from robovid_conditioner.adapters import build_source
    wanted = {str(e) for e in episode_ids}
    kwargs: dict = {}
    if camera_key:
        kwargs["camera_key"] = camera_key
    source = build_source("lerobot", dataset, **kwargs)
    found = {ep.episode_id: ep for ep in source if ep.episode_id in wanted}
    # Preserve the requested order.
    return {str(e): found[str(e)] for e in episode_ids if str(e) in found}


def _checkpoint_segments(path: Path, segs_by_ep: dict[str, list[SubtaskSegment]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        eid: [{"segment_idx": s.segment_idx, "start_frame": s.start_frame, "end_frame": s.end_frame,
               "subtask_text": s.subtask_text, "phase": s.phase, "evidence": s.evidence} for s in segs]
        for eid, segs in segs_by_ep.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def label_metadata_for(provider, model_id: str, rubric: Rubric, episodes: dict, out: Path,
                       max_retries: int = 3) -> tuple[dict, dict]:
    """Label episode quality metadata once per (model, episode). Reused across strategies.

    Returns ``(quality_by_ep, meta_cost_by_ep)``. Per-episode failures (after retries)
    leave that episode's quality as None; it simply does not contribute to the
    quality-agreement metric.
    """
    quality_by_ep: dict[str, int | None] = {}
    meta_cost_by_ep: dict[str, float] = {}
    for eid, ep in episodes.items():
        rdir = out / _safe(model_id) / "metadata" / eid
        mres = _with_retries(
            lambda ep=ep, rdir=rdir: label_metadata(ep, provider, rubric, rdir), max_retries)
        if mres is None:
            quality_by_ep[eid] = None
            meta_cost_by_ep[eid] = 0.0
            continue
        quality_by_ep[eid] = mres.metadata.quality
        meta_cost_by_ep[eid] = sum(c.estimated_cost_usd or 0.0 for c in mres.calls)
    return quality_by_ep, meta_cost_by_ep


def run_cell(provider, model_id: str, rubric: Rubric, gold: dict, episodes: dict,
             quality_by_ep: dict, meta_cost_by_ep: dict, strategy_name: str,
             out: Path, phase: str, *, max_retries: int = 3,
             min_reportable: int = 25) -> dict:
    """Run one (strategy, model) cell over ``episodes`` and score it with reliability.

    Resilient: each episode's segmentation is retried with exponential backoff up to
    ``max_retries``; an episode that still fails is recorded in ``episodes_failed``
    and excluded from scoring (never silently counted as success). A cell is marked
    ``reportable`` only if at least ``min_reportable`` episodes were scored.
    """
    cfg = load_strategy(strategy_name)
    seg_dir = out / _safe(model_id) / strategy_name
    segs_by_ep: dict[str, list[SubtaskSegment]] = {}
    cost_by_ep: dict[str, float] = {}
    latencies: list[float] = []
    failed: list[str] = []
    for i, (eid, ep) in enumerate(episodes.items(), 1):
        rdir = seg_dir / "raw_receipts" / eid
        sres = _with_retries(
            lambda ep=ep, rdir=rdir: segment_episode(ep, provider, rubric, cfg, rdir), max_retries)
        if sres is None:
            failed.append(eid)
            print(f"  [{model_id} {strategy_name}] {i}/{len(episodes)} ep {eid}: FAILED", file=sys.stderr)
            continue
        segs_by_ep[eid] = sres.segments
        cost_by_ep[eid] = sum(c.estimated_cost_usd or 0.0 for c in sres.calls) + meta_cost_by_ep.get(eid, 0.0)
        latencies.extend(c.elapsed_seconds for c in sres.calls if c.elapsed_seconds)
        _checkpoint_segments(seg_dir / "segments.json", segs_by_ep)
        print(f"  [{model_id} {strategy_name}] {i}/{len(episodes)} ep {eid}: "
              f"{len(sres.segments)} segs, ${cost_by_ep[eid]:.4f}", file=sys.stderr)

    scored_ids = list(segs_by_ep)
    auto_by_ep = {eid: segments_to_auto(segs_by_ep[eid], episodes[eid].num_frames) for eid in scored_ids}
    rep = score_against_gold(build_eval_gold(gold, auto_by_ep, quality_by_ep, scored_ids)) if scored_ids else {}
    return {
        "model": model_id, "strategy": strategy_name, "phase": phase,
        "n": len(episodes), "n_scored": len(scored_ids), "episodes_failed": failed,
        "reportable": len(scored_ids) >= min_reportable,
        "boundary_iou": rep.get("subtask_boundary_temporal_iou_mean"),
        "quality_exact": rep.get("quality_exact_agreement"),
        "quality_within_one": rep.get("quality_within_one_agreement"),
        "subgoal_agreement": rep.get("subgoal_frame_agreement"),
        "cost_per_episode_usd": mean(cost_by_ep.values()) if cost_by_ep else None,
        "mean_latency_s": round(mean(latencies), 2) if latencies else None,
        "bands": count_bands(segs_by_ep, rubric),
        "needs_review": _needs_review_ids(quality_by_ep, scored_ids, rubric),
    }


def _needs_review_ids(quality_by_ep: dict, scored_ids: list[str], rubric: Rubric) -> list[str]:
    """Episode ids whose quality is >= margin below the scored-set median (gate policy)."""
    import statistics
    margin = int(rubric.gate.get("quality_outlier_margin", 2))
    vals = [quality_by_ep[e] for e in scored_ids if quality_by_ep.get(e) is not None]
    if len(vals) < 3:
        return []
    med = statistics.median(vals)
    return [e for e in scored_ids if quality_by_ep.get(e) is not None and med - quality_by_ep[e] >= margin]


def _with_retries(thunk, max_retries: int):
    """Call ``thunk`` with exponential backoff; return its result or None after exhaustion."""
    import time
    for attempt in range(max_retries + 1):
        try:
            return thunk()
        except Exception as exc:  # noqa: BLE001 - one bad episode must not sink the sweep
            if attempt >= max_retries:
                print(f"    retry exhausted ({attempt}): {str(exc)[:160]}", file=sys.stderr)
                return None
            time.sleep(min(2.0 ** attempt, 30.0))
    return None


def spend_from_receipts(out: str | Path) -> float:
    """True $ spent so far = sum of estimated cost over every unique on-disk receipt.

    Each Gemini call writes exactly one receipt file; cache hits reuse the file and
    do not re-spend, so summing files is the authoritative spend (not per-cell costs,
    which double-count the shared metadata calls).
    """
    from robovid_conditioner.providers.gemini import _estimate_cost
    total = 0.0
    root = Path(out)
    if not root.exists():
        return 0.0
    for f in root.rglob("*.json"):
        if f.name in {"segments.json", "strategy.json"} or f.name.startswith("results_"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rj, model = data.get("response_json"), data.get("model")
        if isinstance(rj, dict) and model:
            total += _estimate_cost(rj, model) or 0.0
    return round(total, 6)


def run(args: argparse.Namespace) -> None:
    """Simple sweep: every (model, strategy) on one phase, no budget gating.

    The budget-enforced, priority-ordered, mechanically-selecting autonomous run is
    ``scripts/run_ablation.py``; this remains for ad-hoc single-cell or full-grid runs.
    """
    rubric = load_rubric(args.rubric)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    episode_ids = list(split[args.phase])
    if args.limit:
        episode_ids = episode_ids[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from robovid_conditioner.providers.base import build_provider

    results: list[dict] = []
    for spec in args.models:
        provider_name, _, model = spec.partition("/")
        provider = build_provider(provider_name, model or None)
        model_id = f"{provider_name}/{provider.model}"
        print(f"\n=== model {model_id} | {len(episode_ids)} episodes ({args.phase}) ===", file=sys.stderr)
        episodes = _load_episodes(args.dataset, args.camera_key, episode_ids)
        quality_by_ep, meta_cost_by_ep = label_metadata_for(provider, model_id, rubric, episodes, out)
        for strat_name in args.strategies:
            row = run_cell(provider, model_id, rubric, gold, episodes,
                           quality_by_ep, meta_cost_by_ep, strat_name, out, args.phase)
            row["spend_so_far_usd"] = spend_from_receipts(out)
            results.append(row)
            _write_results(out, args.phase, results)
            print(json.dumps(row), file=sys.stderr)

    if args.report:
        write_report(out, args.phase, args.report)


def _write_results(out: Path, phase: str, results: list[dict]) -> None:
    (out / f"results_{phase}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def write_report(out: Path, phase: str, report_path: str) -> None:
    """(Re)generate the markdown tables from whatever results JSONs exist."""
    rows: list[dict] = []
    for phase_file in sorted(out.glob("results_*.json")):
        rows.extend(json.loads(phase_file.read_text(encoding="utf-8")))
    Path(report_path).write_text(render_tables(rows), encoding="utf-8")
    print(f"wrote {report_path}")


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render_tables(rows: list[dict]) -> str:
    lines = ["<!-- generated by scripts/eval_strategies.py -->", ""]
    for phase in ("tune", "test"):
        prows = [r for r in rows if r.get("phase") == phase]
        if not prows:
            continue
        lines.append(f"### {phase} results\n")
        lines.append("| model | strategy | boundary IoU | quality exact | quality ±1 | subgoal | $/episode | degenerate | uniform |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(prows, key=lambda r: (r["model"], r["strategy"])):
            b = r.get("bands", {})
            cost = r.get("cost_per_episode_usd")
            lines.append(
                f"| {r['model']} | {r['strategy']} | {_fmt(r['boundary_iou'])} | "
                f"{_fmt(r['quality_exact'])} | {_fmt(r['quality_within_one'])} | "
                f"{_fmt(r['subgoal_agreement'])} | "
                f"{'n/a' if cost is None else f'${cost:.4f}'} | "
                f"{b.get('degenerate', 0)} | {b.get('uniform_split', 0)} |"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Offline self-test (no API): proves the scoring plumbing end to end with mock.
# --------------------------------------------------------------------------- #
def self_test() -> None:
    import tempfile as _tf

    from robovid_conditioner.demo import synthetic_episode
    from robovid_conditioner.providers import build_provider

    rubric = load_rubric()
    provider = build_provider("mock")
    eids = ["0", "1", "2"]
    episodes = {e: synthetic_episode(int(e)) for e in eids}
    # A minimal human gold with absolute frames.
    gold = {"schema_version": "robovid_conditioner/gold/v1", "episodes": [
        {"episode_id": e, "task": "t", "num_frames": episodes[e].num_frames,
         "auto": {"subtasks": [], "metadata": {}, "subgoals": []},
         "gold": {"metadata": {"quality": 4}, "subtasks": [
             {"segment_idx": 0, "start_frame": 0, "end_frame": episodes[e].num_frames // 2},
             {"segment_idx": 1, "start_frame": episodes[e].num_frames // 2 + 1,
              "end_frame": episodes[e].num_frames - 1}],
             "subgoals": []}} for e in eids]}
    for strat in ("S0", "S1", "S2", "S3", "S4"):
        cfg = load_strategy(strat)
        segs_by_ep = {}
        for e, ep in episodes.items():
            with _tf.TemporaryDirectory() as d:
                segs_by_ep[e] = segment_episode(ep, provider, rubric, cfg, Path(d)).segments
        auto = {e: segments_to_auto(segs, episodes[e].num_frames) for e, segs in segs_by_ep.items()}
        rep = score_against_gold(build_eval_gold(gold, auto, {e: 4 for e in eids}, eids))
        bands = count_bands(segs_by_ep, rubric)
        print(f"{strat}: boundary_iou={_fmt(rep['subtask_boundary_temporal_iou_mean'])} "
              f"quality_exact={_fmt(rep['quality_exact_agreement'])} bands={bands}")
    print("self-test OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Offline plumbing check (no API).")
    ap.add_argument("--gold")
    ap.add_argument("--split", default="eval/so101_split.json")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--camera-key", default=None)
    ap.add_argument("--phase", choices=["tune", "test"], default="tune")
    ap.add_argument("--strategies", nargs="+", default=["S0", "S1", "S2", "S3", "S4"])
    ap.add_argument("--models", nargs="+", default=["gemini/gemini-2.5-flash"])
    ap.add_argument("--rubric", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Use at most N episodes (debugging).")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--report", default="eval_out/strategy_tables.md",
                    help="Where to write the auto-generated markdown tables (paste into STRATEGY_REPORT.md).")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.gold:
        ap.error("--gold is required for a live run (or pass --self-test)")
    run(args)


if __name__ == "__main__":
    main()
