"""Tally a blind-trial grades file into FRESH_TRIAL_REPORT.md, split by strategy.

Inputs:
  * grades.json  — ``{item_id: {marks: {b0,p0,e0,...: bool}, verdict: str}}`` written by
    the inspect viewer's blind grading panel.
  * unblind.json — ``{item_id: {episode_id, strategy, bands: [...]}}`` written by
    ``build_inspect_data.py from-annotations --blind`` (the identity map).

Metrics per strategy:
  * boundary acceptance rate  — mean of the ``b*`` marks (boundary within ±5 of truth);
  * phase accuracy            — mean of the ``p*`` marks;
  * evidence factual-accuracy — mean of the ``e*`` marks (OURS; computed prominently);
  * failure-band rate         — fraction of items the gate flagged degenerate/uniform;
  * verdict distribution      — usable / needs touch-up / garbage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rate(flags: list[bool]) -> tuple[float | None, int]:
    vals = [bool(x) for x in flags if x is not None]
    return (sum(vals) / len(vals) if vals else None, len(vals))


def tally(grades: dict[str, Any], unblind: dict[str, Any]) -> dict[str, Any]:
    by_strategy: dict[str, dict[str, list]] = {}
    for item_id, g in grades.items():
        meta = unblind.get(item_id)
        if not meta:
            continue
        strat = meta.get("strategy", "?")
        s = by_strategy.setdefault(strat, {"b": [], "p": [], "e": [], "bands": [], "verdict": []})
        marks = g.get("marks", {})
        for k, v in marks.items():
            if k.startswith("b"):
                s["b"].append(v)
            elif k.startswith("p"):
                s["p"].append(v)
            elif k.startswith("e"):
                s["e"].append(v)
        s["bands"].append(bool(meta.get("bands")))
        if g.get("verdict"):
            s["verdict"].append(g["verdict"])

    out = {}
    for strat, s in by_strategy.items():
        bacc, bn = _rate(s["b"])
        pacc, pn = _rate(s["p"])
        eacc, en = _rate(s["e"])
        n_items = len(s["bands"])
        fb = sum(1 for x in s["bands"] if x)
        verdicts = {v: s["verdict"].count(v) for v in ("usable", "touchup", "garbage")}
        out[strat] = {
            "n_items": n_items,
            "boundary_acceptance": bacc, "n_boundaries_graded": bn,
            "phase_accuracy": pacc, "n_phases_graded": pn,
            "evidence_factual_accuracy": eacc, "n_evidence_graded": en,
            "failure_band_rate": (fb / n_items if n_items else None), "n_failure_band": fb,
            "verdicts": verdicts,
        }
    return out


def write_trial_report(grades_path: str | Path, unblind_path: str | Path, out_path: str | Path) -> str:
    grades = json.loads(Path(grades_path).read_text(encoding="utf-8"))
    unblind = json.loads(Path(unblind_path).read_text(encoding="utf-8"))
    t = tally(grades, unblind)
    dataset = unblind.get("__dataset__", "(fresh dataset)")

    def pct(x):
        return "n/a" if x is None else f"{x:.2f}"

    lines = [
        "# Fresh-dataset blind trial",
        "",
        f"Dataset: **{dataset}** — a dataset never used to build any robolabel number, with",
        "**no S0-anchored gold**. Episodes were graded blind (strategy identity hidden, random",
        "order) by the author against the raw video. Numbers below are unblinded tallies.",
        "",
        f"Items graded: **{sum(v['n_items'] for v in t.values())}** across {len(t)} strategies.",
        "",
        "| strategy | items | boundary acceptance (±5f) | phase accuracy | **evidence factual-accuracy** | failure-band rate | usable / touch-up / garbage |",
        "|---|---|---|---|---|---|---|",
    ]
    for strat in sorted(t):
        v = t[strat]
        ver = v["verdicts"]
        lines.append(
            f"| {strat} | {v['n_items']} | {pct(v['boundary_acceptance'])} ({v['n_boundaries_graded']}) "
            f"| {pct(v['phase_accuracy'])} ({v['n_phases_graded']}) "
            f"| **{pct(v['evidence_factual_accuracy'])}** ({v['n_evidence_graded']}) "
            f"| {pct(v['failure_band_rate'])} "
            f"| {ver['usable']} / {ver['touchup']} / {ver['garbage']} |"
        )
    lines += [
        "",
        "## What these mean",
        "",
        "- **boundary acceptance (±5f)**: fraction of predicted subtask boundaries the author",
        "  judged within 5 frames of the true transition, watching the video.",
        "- **phase accuracy**: fraction of phase labels (approach/grasp/…) judged correct.",
        "- **evidence factual-accuracy** *(robolabel-specific)*: fraction of grounded evidence",
        "  strings (\"gripper contacts brick\") judged factually true of their cited frame. This",
        "  is the metric no other tool reports — it measures whether the model's stated reason",
        "  is real, not just whether the boundary landed.",
        "- **failure-band rate**: fraction of items the gate flagged degenerate or uniform-split.",
        "- **verdict**: the author's overall call per item (usable as conditioning / needs",
        "  touch-up / garbage).",
        "",
        "This is the **generalization claim**: it is measured on a dataset with no S0 anchoring,",
        "so unlike the SO-101 numbers it is not biased toward any one segmentation. Whatever it",
        "says is what the README may claim about generalization.",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)
