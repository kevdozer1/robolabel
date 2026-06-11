# Strategy report: improving subtask-boundary quality on SO-101

**Question.** Baseline Gemini 2.5 Flash on `lerobot/svla_so101_pickplace` segments
episodes at a subtask-boundary temporal IoU of **0.457** against a 50-episode human
gold set. Can a better *annotation strategy* — not a different model — close that
gap, and which part of the strategy carries the weight?

This report is the measured answer. It is produced by `scripts/run_ablation.py`
(orchestration) on top of `scripts/eval_strategies.py` (scoring), which scores
every (strategy, model) cell with the **same** `reliability_report` used
everywhere else in the tool, against the **same** human gold file.

---

## ⏸ Run status: built and verified, awaiting the API credential

**The live ablation has not run yet** — not for any code reason, but because no
`GEMINI_API_KEY` is reachable on the machine: it is absent from `labelkit/.env`
and from the Process/User/Machine environment scopes. (A `$env:` set in an
interactive shell is process-scoped and does not reach the fresh process the run
spawns; that is the most likely cause.) Everything that does **not** cost money is
verified:

- **Preflight, zero-cost:** git repo on a feature branch; `.env` git-ignored and
  untracked; secret scan clean; `eval/so101_split.json` integrity confirmed (30
  tune / 20 test, no overlap, seed 20260607, all ids present in the gold file); the
  LeRobot adapter resolves `lerobot/svla_so101_pickplace` with camera
  `observation.images.side` and decodes real frames (ep 0: 303 frames, 30 fps,
  480×640) **from the local cache** — no re-download needed.
- **Orchestration logic, offline:** `run_ablation.py --dry-run` exercises the full
  Phase 1–4 flow on synthetic episodes (mock provider). The priority order, the
  mechanical selection (within-0.02 cheaper, quality tie-break, +0.05-over-S0
  bar), the budget gate, and the reportability threshold are unit-tested
  (`tests/test_run_ablation.py`).
- **Cost projection (the budget call):** at a placeholder $0.012/call the full
  5×2×30 tune sweep projects to **~$36.7**, which **exceeds the $30 ceiling**. So on
  the real run the budget gate is *expected* to engage: all Flash cells complete
  (cheap), then Pro is added in priority order (S0 anchor → Flash winner → rest)
  until the next cell would cross $30, at which point remaining Pro cells are
  skipped and noted. The held-out test cell is reserved within that budget. The
  real per-call cost is measured in preflight from a single live S1 episode and the
  projection is rewritten before the sweep starts.

**To run it (one command, fully autonomous under the guardrails):**

```bash
# 1) put the key where the loader looks (it is git-ignored):
printf 'GEMINI_API_KEY=YOUR_KEY\n' >> labelkit/.env
# 2) launch — preflight → budget-gated tune sweep → mechanical selection → one test cell:
python scripts/run_ablation.py \
  --gold ../robovid_work/so101_gemini/gold.json --split eval/so101_split.json \
  --dataset lerobot/svla_so101_pickplace --camera-key observation.images.side \
  --budget 30 --out eval_out
```

It checkpoints per episode and resumes for free (receipt cache), so a credit
run-out mid-sweep loses nothing: every completed cell is already scored in
`eval_out/results_tune.json`, and this report can be filled from whatever exists.
The tables below stay marked _pending_ until that run produces numbers.

---

## The three failure bands

The baseline boundaries fail in three distinguishable ways. The eval counts how
many tune episodes fall in each band per strategy, using the gate detectors:

- **(a) degenerate** — a single "complete the task" segment spanning the episode
  (`is_degenerate_single_segment`).
- **(b) uniform split** — the right number of phases, but boundaries at uniform
  fractions of the duration; the model never grounded them to frames
  (`is_uniform_split`, coefficient-of-variation of segment lengths below 0.12).
- **(c) drifted** — plausible, frame-grounded boundaries that are nonetheless
  systematically off. This is the residual band (not degenerate, not uniform); it
  is what the refinement and self-consistency passes target, and it shows up as a
  middling IoU rather than as a detector flag.

---

## Strategies under test (cumulative)

| strategy | adds | what it should fix |
|---|---|---|
| **S0** | baseline (current default): 6 evenly-spaced keyframes, free-text segments | — |
| **S1** | frame-indexed grounding: 12 captioned frames (index + timestamp), boundaries returned as frame indices each with a one-line evidence string, schema-validated | band (b): boundaries become frame-grounded, not fractional |
| **S2** | S1 + closed phase vocabulary (approach / grasp / transport / release-place / retract / other) + minimum granularity (single-segment outputs rejected at the schema level, re-prompted) | band (a): degenerate outputs |
| **S3** | S2 + dense-window refinement: for each boundary, send ±15 frames at full stride and ask for the exact transition frame | band (c): drift |
| **S4** | S3 + self-consistency: k=3 label samples, per-boundary median | band (c): residual variance |

The full resolved config for each strategy is in
`src/robovid_conditioner/strategy.py` (and written to `strategy.json` on every
run). The prompts live in `rubric.yaml` under `strategies:` — config, not code.

---

## Evaluation protocol (hygiene)

- **Frozen split.** `eval/so101_split.json` is seeded (seed 20260607) and committed:
  **30 tune / 20 test**, no overlap. Regenerate identically with
  `scripts/make_split.py`.
- **Tune only.** All strategy iteration and threshold-setting happens on the 30
  tune episodes. The 20 test episodes are scored **once**, with the single chosen
  strategy, and reported in the held-out table below — not iterated against.
- **Metadata is labeled once per (model, episode)** and reused across strategies:
  the strategies only move subtask boundaries, so quality agreement varies by
  *model*, not by strategy. This keeps the quality columns honest and avoids paying
  5× for an identical metadata call.
- **Same metrics, same code.** Boundary temporal IoU, quality exact / within-one
  agreement, and subgoal frame agreement all come from `reliability_report`. Cost
  per episode = strategy segmentation calls + the shared metadata call, from the
  per-call usage receipts.

Reproduce:

```bash
python scripts/eval_strategies.py \
  --gold <gold.json> --split eval/so101_split.json \
  --dataset lerobot/svla_so101_pickplace --camera-key observation.images.side \
  --phase tune  --strategies S0 S1 S2 S3 S4 \
  --models gemini/gemini-2.5-flash gemini/gemini-2.5-pro --out eval_out
# then, once, with the chosen strategy:
python scripts/eval_strategies.py ... --phase test --strategies <CHOSEN>
```

---

## Results — tune (30 episodes)

<!-- Paste the table block from eval_out/strategy_tables.md (tune) here after the run. -->
_Pending the live ablation run. The harness writes this table to
`eval_out/strategy_tables.md` and the raw rows to `eval_out/results_tune.json`._

| model | strategy | boundary IoU | quality exact | quality ±1 | subgoal | $/episode | degenerate | uniform |
|---|---|---|---|---|---|---|---|---|
| gemini/gemini-2.5-flash | S0 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| … | S1–S4 | | | | | | | |
| gemini/gemini-2.5-pro | S0–S4 | | | | | | | |

### Per-failure-band counts (tune)

_Filled from the `degenerate` / `uniform` columns above plus the residual
(drifted/ok) count in `results_tune.json`._

---

## Results — held-out test (20 episodes), chosen strategy, reported once

<!-- Paste the test table after the single held-out run. -->
_Pending. Chosen strategy + model decided on tune, then scored once here._

| model | strategy | boundary IoU | quality exact | quality ±1 | subgoal | $/episode |
|---|---|---|---|---|---|---|
| _chosen_ | _chosen_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

---

## Reading guide (what to conclude)

Once the tables are filled, the honest read is:

- **Does grounding (S1) move IoU off the 0.457 baseline, and does it empty the
  uniform-split band?** If yes, band (b) was the dominant baseline failure.
- **Does S2 empty the degenerate band** without hurting IoU? That isolates band (a).
- **Do S3/S4 add IoU beyond S2**, and is the added cost per episode worth it? If the
  refinement/self-consistency gain is within noise, the honest recommendation is to
  stop at S2 and bank the cost.
- **Flash vs Pro:** if the stronger model mostly helps quality agreement (a metadata
  effect) but not boundary IoU, then boundary quality is a *prompting/strategy*
  problem, not a *model-capability* problem — which is the thesis of this layer.

The point of the tool is unchanged: whatever the numbers say, they are measured on
your gold set, in the same units, and the loser strategies are reported next to the
winner rather than hidden.
