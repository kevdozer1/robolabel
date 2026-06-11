"""Autonomous, budget-capped strategy ablation — the guardrailed end-to-end run.

Encodes the operating protocol as code so the whole thing is one resilient command:

  Phase 1 PREFLIGHT  verify key (1 tiny call), integrity, probe S1 on one tune
                     episode, project the full-sweep cost, abort/​degrade if the
                     projection exceeds the budget ceiling.
  Phase 2 TUNE SWEEP priority order: all Flash S0→S4, then Pro S0 (anchor), then
                     Pro on the Flash winner, then remaining Pro — each cell gated
                     by the running spend vs the ceiling. Per-episode retries.
  Phase 3 SELECTION  mechanical: highest boundary IoU; ties within 0.02 → cheaper;
                     then quality-exact; winner must beat S0-Flash by ≥0.05 IoU or
                     S0 is selected ("the layer didn't move the needle").
  Phase 4 TEST       the selected cell, once, on the 20 held-out episodes (+ S0-Flash
                     on test for a before/after if budget remains).

Spend is tracked continuously from the on-disk receipts (the authoritative number),
never from per-cell sums. Everything checkpoints; re-running resumes for free via the
provider receipt cache.

    python scripts/run_ablation.py \
      --gold ../robovid_work/so101_gemini/gold.json --split eval/so101_split.json \
      --dataset lerobot/svla_so101_pickplace --camera-key observation.images.side \
      --budget 30 --out eval_out

Offline plumbing check (no API, synthetic episodes):
    python scripts/run_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling eval_strategies

import eval_strategies as ev  # noqa: E402

from robovid_conditioner.rubric import load_rubric  # noqa: E402

# Expected segmentation calls per episode: observe + k label samples + refinement.
# (S3 ≈ observe+label+3 refine; S4 ≈ observe+3 labels+3 refine on ~4 segments.)
STRATEGY_CALLS = {"S0": 2, "S1": 2, "S2": 2, "S3": 5, "S4": 7}
METADATA_CALLS = 2
PRO_PRICE_RATIO = 4.1  # Gemini 2.5 Pro vs Flash per-token price, ~4×
ALL_STRATEGIES = ["S0", "S1", "S2", "S3", "S4"]


# --------------------------------------------------------------------------- #
# Pure decision functions (unit-tested in tests/test_run_ablation.py)
# --------------------------------------------------------------------------- #
def estimate_cell_cost(strategy: str, n_episodes: int, per_call_flash: float,
                       model_id: str, model_paid_metadata: bool) -> float:
    """Rough $ for one cell, used only for the budget gate (true spend is authoritative)."""
    ratio = PRO_PRICE_RATIO if "pro" in model_id.lower() else 1.0
    seg = STRATEGY_CALLS.get(strategy, 2) * n_episodes
    meta = 0 if model_paid_metadata else METADATA_CALLS * n_episodes
    return (seg + meta) * per_call_flash * ratio


def project_full_sweep(strategies: list[str], models: list[str], n_tune: int,
                       per_call_flash: float) -> tuple[float, list[dict]]:
    """Project the full tune sweep cost, paying metadata once per model."""
    total = 0.0
    rows: list[dict] = []
    for m in models:
        paid = False
        for s in strategies:
            c = estimate_cell_cost(s, n_tune, per_call_flash, m, paid)
            paid = True
            rows.append({"model": m, "strategy": s, "est_usd": round(c, 4)})
            total += c
    return round(total, 4), rows


def priority_order(strategies: list[str], flash_model: str, pro_model: str | None,
                   flash_winner_strategy: str | None) -> list[tuple[str, str]]:
    """Cells in graceful-degradation order: all Flash, Pro anchor, Pro winner, rest."""
    order: list[tuple[str, str]] = [(flash_model, s) for s in strategies]
    if pro_model:
        order.append((pro_model, "S0"))
        if flash_winner_strategy and flash_winner_strategy != "S0":
            order.append((pro_model, flash_winner_strategy))
        for s in strategies:
            if (pro_model, s) not in order:
                order.append((pro_model, s))
    return order


def select_winner(rows: list[dict], baseline_row: dict | None,
                  min_gain: float = 0.05, near: float = 0.02) -> tuple[dict | None, str]:
    """Mechanical winner selection. Returns (winner_row, rationale)."""
    reportable = [r for r in rows if r.get("reportable") and r.get("boundary_iou") is not None]
    if not reportable:
        return baseline_row, "no reportable cells; defaulting to S0 baseline"
    leader = max(reportable, key=lambda r: r["boundary_iou"])
    cluster = [r for r in reportable if leader["boundary_iou"] - r["boundary_iou"] <= near]
    pick = min(cluster, key=lambda r: ((r["cost_per_episode_usd"] or float("inf")),
                                       -(r["quality_exact"] or 0.0)))
    cluster_names = ", ".join(f"{r['strategy']}@{r['model']}" for r in cluster)
    desc = (f"leader={leader['strategy']}@{leader['model']} IoU={leader['boundary_iou']:.3f}; "
            f"within-{near} cluster=[{cluster_names}]; "
            f"cheapest pick={pick['strategy']}@{pick['model']} IoU={pick['boundary_iou']:.3f}")
    base_iou = baseline_row.get("boundary_iou") if baseline_row else None
    if base_iou is not None and (pick["boundary_iou"] - base_iou) < min_gain:
        return baseline_row, desc + (f"; did NOT beat S0-Flash ({base_iou:.3f}) by >={min_gain} "
                                     "-> select S0 baseline for test")
    bar = "n/a (no S0-Flash baseline)" if base_iou is None else f"clears +{min_gain} over S0-Flash ({base_iou:.3f})"
    return pick, desc + f"; {bar} -> WINNER"


def budget_allows(spent: float, est_next: float, ceiling: float) -> bool:
    return (spent + est_next) <= ceiling


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _log(run_log: dict, out: Path, msg: str) -> None:
    run_log.setdefault("events", []).append(msg)
    (out / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(msg, file=sys.stderr)


def _probe_per_call(out: Path, model_dir: str, probe_id: str) -> float:
    """Average $/call over the probe episode's own (cache-bypassed) receipts."""
    from robovid_conditioner.providers.gemini import _estimate_cost
    costs: list[float] = []
    for d in (out / model_dir / "metadata" / probe_id, out / model_dir / "S1" / "raw_receipts" / probe_id):
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rj, model = data.get("response_json"), data.get("model")
            if isinstance(rj, dict) and model:
                c = _estimate_cost(rj, model)
                if c:
                    costs.append(c)
    return sum(costs) / len(costs) if costs else 0.0


def _price_table_estimate(model: str) -> float:
    """Conservative per-call $ from the gemini price table (a typical grounded call)."""
    from robovid_conditioner.providers.gemini import _PRICES
    low = model.lower()
    key = "2.5-pro" if "2.5-pro" in low else "2.5-flash" if "2.5-flash" in low else "flash-lite"
    prices = _PRICES.get(key)
    if not prices:
        return 0.01
    # ~1.5k input tokens (12-frame contact sheet + prompt) + ~0.3k output per call.
    return 1500 / 1e6 * prices["input"] + 300 / 1e6 * prices["output"]


def run(args: argparse.Namespace) -> int:
    rubric = load_rubric(args.rubric)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_log: dict = {"budget_usd": args.budget, "events": []}

    from robovid_conditioner.providers.base import MissingCredentialError, build_provider

    flash_model, pro_model = args.flash, args.pro
    tune_ids = list(split["tune"])
    test_ids = list(split["test"])
    if args.limit:
        tune_ids, test_ids = tune_ids[: args.limit], test_ids[: args.limit]

    # ---- Phase 1: preflight ------------------------------------------------ #
    try:
        flash = build_provider(*_spec(flash_model))
    except MissingCredentialError as exc:
        _log(run_log, out, f"PREFLIGHT ABORT: {exc}")
        return 3
    flash_id = f"{flash.name}/{flash.model}"

    tune_eps = ev._load_episodes(args.dataset, args.camera_key, tune_ids)
    probe_id = next(iter(tune_eps))
    # One real S1 episode (also satisfies the "verify a parsed schema-valid response,
    # captions out, frame-indexed boundaries + evidence back" preflight check).
    # IMPORTANT: bypass the receipt cache for the probe so per-call cost is *measured*,
    # not read as $0 off a cached receipt (the bug from the first run, where the probe
    # hit the smoke's cache and the budget gate went inert).
    prev_cache = getattr(flash, "use_cache", None)
    if prev_cache is not None:
        flash.use_cache = False
    q1, mc1 = ev.label_metadata_for(flash, flash_id, rubric, {probe_id: tune_eps[probe_id]}, out)
    probe_row = ev.run_cell(flash, flash_id, rubric, gold, {probe_id: tune_eps[probe_id]},
                            q1, mc1, "S1", out, "tune", min_reportable=1)
    if prev_cache is not None:
        flash.use_cache = prev_cache
    per_call = _probe_per_call(out, ev._safe(flash_id), probe_id)
    if per_call <= 0:  # final fallback: a conservative price-table estimate
        per_call = _price_table_estimate(flash.model)
    projected, proj_rows = project_full_sweep(ALL_STRATEGIES,
                                              [flash_model] + ([pro_model] if pro_model else []),
                                              len(tune_ids), per_call)
    run_log["preflight"] = {
        "probe": {k: probe_row.get(k) for k in ("strategy", "boundary_iou", "n_scored", "bands")},
        "per_call_usd": round(per_call, 5),
        "projected_full_sweep_usd": projected, "projection_rows": proj_rows,
        "exceeds_ceiling": projected > args.budget,
    }
    _log(run_log, out, f"PREFLIGHT ok: per_call≈${per_call:.4f}, projected full sweep "
                       f"${projected:.2f} vs ceiling ${args.budget:.2f} "
                       f"({'DEGRADE via priority order' if projected > args.budget else 'fits'})")

    # ---- Phase 2: tune sweep (priority order, budget-gated) --------------- #
    tune_rows: list[dict] = []
    meta_cache: dict[str, tuple[dict, dict]] = {}
    pro_dead = False

    def run_one(model_spec: str, strat: str) -> dict | None:
        nonlocal pro_dead
        provider = flash if model_spec == flash_model else build_provider(*_spec(model_spec))
        model_id = f"{provider.name}/{provider.model}"
        spent = ev.spend_from_receipts(out)
        est = estimate_cell_cost(strat, len(tune_ids), per_call, model_id, model_id in meta_cache)
        if not budget_allows(spent, est, args.budget):
            _log(run_log, out, f"SKIP {strat}@{model_id}: est ${est:.2f} + spent ${spent:.2f} > ${args.budget}")
            return None
        if model_id not in meta_cache:
            meta_cache[model_id] = ev.label_metadata_for(provider, model_id, rubric, tune_eps, out)
        q, mc = meta_cache[model_id]
        row = ev.run_cell(provider, model_id, rubric, gold, tune_eps, q, mc, strat, out, "tune")
        row["spend_so_far_usd"] = ev.spend_from_receipts(out)
        tune_rows.append(row)
        ev._write_results(out, "tune", tune_rows)
        _log(run_log, out, f"CELL {strat}@{model_id}: IoU={_f(row['boundary_iou'])} "
                           f"scored={row['n_scored']}/{row['n']} ${row.get('spend_so_far_usd'):.2f} spent")
        if "pro" in model_id.lower() and row["n_scored"] == 0:
            pro_dead = True
            _log(run_log, out, f"Pro model {model_id} produced 0 scored episodes -> marking Pro unavailable")
        return row

    # Flash sweep first.
    for s in ALL_STRATEGIES:
        run_one(flash_model, s)
    flash_rows = [r for r in tune_rows if r["model"] == flash_id]
    flash_reportable = [r for r in flash_rows if r.get("reportable") and r.get("boundary_iou") is not None]
    flash_winner = (max(flash_reportable, key=lambda r: r["boundary_iou"])["strategy"]
                    if flash_reportable else None)
    _log(run_log, out, f"Flash sweep done. Flash winner (by IoU): {flash_winner}")

    # Pro cells in priority order.
    if pro_model:
        for model_spec, strat in priority_order(ALL_STRATEGIES, flash_model, pro_model, flash_winner):
            if model_spec == flash_model:
                continue
            if pro_dead:
                _log(run_log, out, f"SKIP {strat}@{pro_model}: Pro marked unavailable")
                continue
            run_one(model_spec, strat)

    # ---- Phase 3: selection (mechanical) ---------------------------------- #
    baseline_row = next((r for r in tune_rows if r["model"] == flash_id and r["strategy"] == "S0"), None)
    winner, rationale = select_winner(tune_rows, baseline_row)
    run_log["selection"] = {"winner": None if not winner else {"model": winner["model"], "strategy": winner["strategy"]},
                            "rationale": rationale}
    _log(run_log, out, f"SELECTION: {rationale}")

    # ---- Phase 4: test (once) --------------------------------------------- #
    if winner:
        _run_test_cell(args, rubric, gold, test_ids, winner["model"], winner["strategy"],
                       out, run_log, per_call, flash, flash_model, flash_id, build_provider)
        # before/after: S0-Flash on test too, if budget remains and winner wasn't already it.
        if not (winner["model"] == flash_id and winner["strategy"] == "S0"):
            _run_test_cell(args, rubric, gold, test_ids, flash_id, "S0",
                           out, run_log, per_call, flash, flash_model, flash_id, build_provider,
                           only_if_budget=True)

    ev.write_report(out, "tune", args.report)
    _log(run_log, out, f"DONE. Total spend ${ev.spend_from_receipts(out):.2f} of ${args.budget:.2f} ceiling.")
    return 0


def _run_test_cell(args, rubric, gold, test_ids, model_id, strat, out, run_log, per_call,
                   flash, flash_model, flash_id, build_provider, only_if_budget=False) -> None:
    from robovid_conditioner.providers.base import build_provider as _bp  # noqa: F811
    provider = flash if model_id == flash_id else _bp(*_spec(_model_spec_from_id(model_id, args)))
    test_eps = ev._load_episodes(args.dataset, args.camera_key, test_ids)
    spent = ev.spend_from_receipts(out)
    est = estimate_cell_cost(strat, len(test_ids), per_call, model_id, False)
    if only_if_budget and not budget_allows(spent, est, args.budget):
        _log(run_log, out, f"SKIP test {strat}@{model_id}: est ${est:.2f} + ${spent:.2f} > ${args.budget}")
        return
    q, mc = ev.label_metadata_for(provider, model_id, rubric, test_eps, out)
    test_rows_path = out / "results_test.json"
    existing = json.loads(test_rows_path.read_text()) if test_rows_path.exists() else []
    row = ev.run_cell(provider, model_id, rubric, gold, test_eps, q, mc, strat, out, "test")
    row["spend_so_far_usd"] = ev.spend_from_receipts(out)
    existing.append(row)
    test_rows_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    _log(run_log, out, f"TEST {strat}@{model_id}: IoU={_f(row['boundary_iou'])} scored={row['n_scored']}/{row['n']}")


def _model_spec_from_id(model_id: str, args) -> str:
    return args.pro if model_id == f"{args.pro.split('/')[0]}/{args.pro.split('/')[-1]}" or "pro" in model_id else args.flash


def _spec(model_spec: str) -> tuple[str, str | None]:
    name, _, model = model_spec.partition("/")
    return name, (model or None)


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _dry_run() -> int:
    """Exercise the full orchestration offline with mock + synthetic episodes."""
    import tempfile

    from robovid_conditioner.demo import synthetic_episode
    rubric = load_rubric()
    ids = [str(i) for i in range(5)]
    eps = {i: synthetic_episode(int(i)) for i in ids}
    gold = {"schema_version": "robovid_conditioner/gold/v1", "episodes": [
        {"episode_id": i, "task": "t", "num_frames": eps[i].num_frames,
         "auto": {"subtasks": [], "metadata": {}, "subgoals": []},
         "gold": {"metadata": {"quality": 4},
                  "subtasks": [{"segment_idx": 0, "start_frame": 0, "end_frame": eps[i].num_frames // 2},
                               {"segment_idx": 1, "start_frame": eps[i].num_frames // 2 + 1,
                                "end_frame": eps[i].num_frames - 1}], "subgoals": []}} for i in ids]}
    from robovid_conditioner.providers import build_provider
    prov = build_provider("mock")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        rows = []
        q, mc = ev.label_metadata_for(prov, "mock/mock", rubric, eps, out)
        for s in ALL_STRATEGIES:
            rows.append(ev.run_cell(prov, "mock/mock", rubric, gold, eps, q, mc, s, out, "tune", min_reportable=1))
        baseline = next(r for r in rows if r["strategy"] == "S0")
        winner, rationale = select_winner(rows, baseline)
        order = priority_order(ALL_STRATEGIES, "gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro", "S3")
        proj, _ = project_full_sweep(ALL_STRATEGIES, ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"], 30, 0.012)
        print("dry-run cells:", [(r["strategy"], _f(r["boundary_iou"]), r["n_scored"]) for r in rows])
        print("priority order:", order)
        print(f"projected (per_call=$0.012): ${proj}")
        print("selection:", None if not winner else f"{winner['strategy']}@{winner['model']}", "|", rationale)
    print("dry-run OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Offline orchestration check (no API).")
    ap.add_argument("--gold")
    ap.add_argument("--split", default="eval/so101_split.json")
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--camera-key", default="observation.images.side")
    ap.add_argument("--flash", default="gemini/gemini-2.5-flash")
    ap.add_argument("--pro", default="gemini/gemini-2.5-pro", help="Stronger model spec, or '' to skip Pro.")
    ap.add_argument("--budget", type=float, default=30.0, help="Hard $ ceiling for the whole run.")
    ap.add_argument("--rubric", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--report", default="eval_out/strategy_tables.md")
    args = ap.parse_args()
    if args.dry_run:
        return _dry_run()
    if not args.gold:
        ap.error("--gold is required (or pass --dry-run)")
    if args.pro.strip() == "":
        args.pro = None
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
