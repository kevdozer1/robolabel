"""Unit tests for the autonomous ablation orchestrator's decision functions.

These encode the operating protocol (budget projection, priority order, mechanical
selection) and must be exactly right — they run unattended and spend real money.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RA = _load()


# --------------------------------------------------------------------------- #
# Cost projection
# --------------------------------------------------------------------------- #
def test_estimate_cell_cost_pro_is_pricier_and_metadata_once():
    flash_s0 = RA.estimate_cell_cost("S0", 30, 0.01, "gemini/gemini-2.5-flash", model_paid_metadata=False)
    pro_s0 = RA.estimate_cell_cost("S0", 30, 0.01, "gemini/gemini-2.5-pro", model_paid_metadata=False)
    assert pro_s0 > flash_s0  # Pro price ratio applies
    # Once metadata is paid, the next cell for that model is cheaper.
    flash_s0_paid = RA.estimate_cell_cost("S0", 30, 0.01, "gemini/gemini-2.5-flash", model_paid_metadata=True)
    assert flash_s0_paid < flash_s0
    # S4 (7 calls/ep) costs more than S0 (2 calls/ep).
    assert RA.estimate_cell_cost("S4", 30, 0.01, "gemini/gemini-2.5-flash", True) > flash_s0_paid


def test_project_full_sweep_pays_metadata_once_per_model():
    total, rows = RA.project_full_sweep(["S0", "S1"], ["gemini/gemini-2.5-flash"], 30, 0.01)
    assert len(rows) == 2
    # S0 includes metadata (60+60 calls), S1 reuses it (60 calls) -> S0 row dearer.
    assert rows[0]["est_usd"] > rows[1]["est_usd"]
    assert abs(total - sum(r["est_usd"] for r in rows)) < 1e-6


# --------------------------------------------------------------------------- #
# Priority order
# --------------------------------------------------------------------------- #
def test_priority_order_flash_then_pro_anchor_then_winner_then_rest():
    order = RA.priority_order(["S0", "S1", "S2", "S3", "S4"],
                              "gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro", "S3")
    flash = [c for c in order if c[0].endswith("flash")]
    pro = [c for c in order if c[0].endswith("pro")]
    assert [c[1] for c in flash] == ["S0", "S1", "S2", "S3", "S4"]   # all Flash first
    assert pro[0][1] == "S0"                                          # Pro anchor
    assert pro[1][1] == "S3"                                          # Pro on Flash winner next
    assert set(c[1] for c in pro) == {"S0", "S1", "S2", "S3", "S4"}   # no duplicates, all covered


def test_priority_order_no_pro_model():
    order = RA.priority_order(["S0", "S1"], "gemini/gemini-2.5-flash", None, None)
    assert all(c[0].endswith("flash") for c in order)


def test_priority_order_flash_winner_is_s0_no_duplicate():
    order = RA.priority_order(["S0", "S1"], "f", "p", "S0")
    pro = [c for c in order if c[0] == "p"]
    assert [c[1] for c in pro] == ["S0", "S1"]  # S0 not duplicated


# --------------------------------------------------------------------------- #
# Mechanical selection
# --------------------------------------------------------------------------- #
def _row(model, strat, iou, cost, qexact=0.5, reportable=True):
    return {"model": model, "strategy": strat, "boundary_iou": iou,
            "cost_per_episode_usd": cost, "quality_exact": qexact, "reportable": reportable}


def test_select_winner_highest_iou_clearing_bar():
    base = _row("f", "S0", 0.45, 0.01)
    rows = [base, _row("f", "S3", 0.60, 0.05)]
    winner, why = RA.select_winner(rows, base)
    assert (winner["model"], winner["strategy"]) == ("f", "S3")
    assert "WINNER" in why


def test_select_winner_prefers_cheaper_within_002():
    base = _row("f", "S0", 0.45, 0.01)
    # S3 leads at 0.61, S2 is 0.60 (within 0.02) and cheaper -> S2 wins.
    rows = [base, _row("p", "S3", 0.61, 0.20), _row("f", "S2", 0.60, 0.03)]
    winner, _ = RA.select_winner(rows, base)
    assert (winner["model"], winner["strategy"]) == ("f", "S2")


def test_select_winner_quality_breaks_cost_ties():
    base = _row("f", "S0", 0.45, 0.01)
    rows = [base, _row("f", "S2", 0.60, 0.05, qexact=0.7), _row("f", "S3", 0.61, 0.05, qexact=0.9)]
    winner, _ = RA.select_winner(rows, base)
    assert winner["strategy"] == "S3"  # same cost, higher quality_exact


def test_select_winner_fails_bar_falls_back_to_s0():
    base = _row("f", "S0", 0.45, 0.01)
    rows = [base, _row("f", "S3", 0.48, 0.05)]  # only +0.03 over S0, < 0.05 bar
    winner, why = RA.select_winner(rows, base)
    assert (winner["model"], winner["strategy"]) == ("f", "S0")
    assert "did NOT beat" in why


def test_select_winner_ignores_unreportable_cells():
    base = _row("f", "S0", 0.45, 0.01)
    rows = [base, _row("p", "S4", 0.90, 0.30, reportable=False)]  # great but unreportable
    winner, _ = RA.select_winner(rows, base)
    assert winner["strategy"] == "S0"  # the 0.90 cell is ignored


def test_select_winner_no_reportable_defaults_baseline():
    base = _row("f", "S0", 0.45, 0.01, reportable=False)
    winner, why = RA.select_winner([base], base)
    assert winner is base and "defaulting" in why


# --------------------------------------------------------------------------- #
# Budget gate
# --------------------------------------------------------------------------- #
def test_budget_allows():
    assert RA.budget_allows(20.0, 5.0, 30.0)
    assert RA.budget_allows(25.0, 5.0, 30.0)      # exactly at ceiling is allowed
    assert not RA.budget_allows(28.0, 5.0, 30.0)  # would exceed
