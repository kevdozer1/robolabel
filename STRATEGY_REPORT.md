# Strategy report: improving subtask-boundary quality on SO-101

**Question.** Baseline Gemini 2.5 Flash on `lerobot/svla_so101_pickplace` segments
episodes at a subtask-boundary temporal IoU of **0.457** against a 50-episode human
gold set. Can a better *annotation strategy* — not a different model — close that
gap, and which part of the strategy carries the weight?

**Short answer (honest).** The strategy layer's tune-set advantage **did not
generalize** to the held-out test set on *mean* boundary IoU: the mechanically-chosen
winner (Gemini 2.5 Pro, S2) scored **0.444** on test, while the S0-Flash baseline
scored **0.460** on the same 20 episodes. What the strategy layer *did* deliver,
robustly and on held-out data, is the elimination of the **catastrophic failure
bands**: S0-Flash put **5 of 20** test episodes into a degenerate or uniform-split
failure mode; the grounded strategy put **0 of 20** there. The stronger model also cut
catastrophic quality false-negatives. So grounding buys **robustness and data
hygiene**, not higher mean IoU, on this (easy, near-saturated) dataset — and the
held-out test is exactly what kept us from claiming otherwise.

Produced by `scripts/run_ablation.py` + `scripts/eval_strategies.py`; every cell is
scored with the **same** `reliability_report` used everywhere else in the tool,
against the **same** human gold file. Total spend: **$16.54 / $30 ceiling**.

---

## The three failure bands

The baseline boundaries fail in three distinguishable ways, counted per cell by the
gate detectors:

- **(a) degenerate** — a single "complete the task" segment (`is_degenerate_single_segment`).
- **(b) uniform split** — the right number of phases, but at uniform fractions of the
  duration; never grounded to frames (`is_uniform_split`, CV of segment lengths < 0.12).
- **(c) drifted** — plausible, frame-grounded boundaries that are still systematically
  off. The residual band (a middling IoU rather than a detector flag).

---

## Strategies under test (cumulative)

| strategy | adds | targets |
|---|---|---|
| **S0** | baseline: 6 evenly-spaced keyframes, free-text segments | — |
| **S1** | frame-indexed grounding: 12 captioned frames (index+timestamp), boundaries as frame indices each with an evidence string, schema-validated | band (b) |
| **S2** | S1 + closed phase vocabulary + min granularity (single-segment rejected, re-prompted) | band (a) |
| **S3** | S2 + dense-window (±15 frame) boundary refinement | band (c) |
| **S4** | S3 + self-consistency (k=3 label samples, per-boundary median) | band (c) |
| **S_grip** | **free, zero-API** proprioceptive baseline: boundaries from gripper open/close + end-effector-speed pauses, phases by event order | (reference floor) |

Resolved configs in `src/robovid_conditioner/strategy.py`; grounded prompts in
`rubric.yaml`. (Note: the grounded re-prompt cap is ≤2 re-prompts per label pass,
within the operating guardrail.)

---

## Evaluation protocol (hygiene)

- **Frozen split** `eval/so101_split.json` (seed 20260607): **30 tune / 20 test**, no
  overlap. Regenerate with `scripts/make_split.py`.
- **Tune only** for all iteration/selection. The 20 test episodes were scored **once**,
  with the single chosen cell (+ an S0-Flash before/after), reported below, not iterated.
- **Metadata labeled once per model** and reused across strategies (strategies only move
  boundaries) — so quality numbers vary by *model*, not strategy.
- Same metrics, same `reliability_report`; cost from per-call usage receipts.

---

## Results — tune (30 episodes)

| model | strat | boundary IoU | quality exact | subgoal | $/ep | degen | uniform |
|---|---|---|---|---|---|---|---|
| Flash | S0 | 0.400 | 0.70 | 0.422 | $0.0185 | **3** | **9** |
| Flash | S1 | 0.391 | 0.70 | 0.095 | $0.0180 | 0 | 0 |
| Flash | S2 | 0.415 | 0.70 | 0.095 | $0.0180 | 0 | 0 |
| Flash | S3 | 0.426 | 0.72 | 0.090 | $0.0262 | 0 | 0 |
| Flash | S4 | 0.430 | 0.71 | 0.102 | $0.0403 | 0 | 1 |
| Pro | S0 | 0.355 | 0.87 | 0.069 | $0.0737 | 0 | 0 |
| Pro | S1 | 0.438 | 0.87 | 0.112 | $0.0728 | 0 | 0 |
| **Pro** | **S2** | **0.453** | **0.87** | 0.112 | $0.0684 | 0 | 0 |
| Pro | S3 | 0.430 | 0.87 | 0.112 | $0.1113 | 0 | 0 |
| Pro | S4 | 0.435 | 0.87 | 0.129 | $0.1694 | 0 | 0 |
| S_grip | — | 0.204 | n/a | 0.060 | **$0.00** | 0 | 0 |

Reportable cells (≥25/30 scored): all VLM cells (S3/S4-Flash dropped 1–2 episodes to
the capped re-prompt loop; those count toward no band, they are simply un-scored).

### Per-failure-band movement (tune), the mechanism story

| | S0-Flash | S1-Flash | S2-Flash | grounded (S1+, both models) |
|---|---|---|---|---|
| degenerate | 3 | 0 | 0 | 0 |
| uniform-split | 9 | 0 | 0 | 0 (one S4-Flash relapse) |

**Frame grounding (S1) empties the uniform-split band (9 → 0)** and **the min-granularity
floor (S2) keeps the degenerate band at 0** — the two detectors confirm each strategy
does what it was designed to do. The catch: emptying those bands **did not raise mean
IoU** (S1-Flash 0.391 ≤ S0-Flash 0.400). Grounding replaces a few catastrophic outputs
and a few good ones with uniformly frame-grounded-but-drifted ones; the mean barely
moves. The stronger model is where boundary IoU actually rises (Pro S1–S4 ≈ 0.43–0.45 vs
Flash 0.39–0.43), but even Pro tops out at **0.453**.

---

## Selection (mechanical, no discretion)

Rule: highest tune boundary IoU; ties within 0.02 → cheaper; quality-exact breaks
remaining ties; winner must beat S0-Flash (0.400) by ≥0.05.

> leader = **Pro S2 (0.453)**; within-0.02 cluster = {Pro S4 0.435, Pro S1 0.438, Pro S2
> 0.453}; cheapest in cluster = **Pro S2** ($0.0684/ep); 0.453 − 0.400 = 0.053 ≥ 0.05 →
> **WINNER: Pro S2**.

It cleared the bar by 0.003. That margin is the first hint that the result is fragile —
borne out next.

---

## Results — held-out test (20 episodes), reported once

| model | strat | boundary IoU | quality exact | subgoal | degen | uniform |
|---|---|---|---|---|---|---|
| **Pro** | **S2** (chosen) | **0.444** | 0.95 | 0.122 | **0** | **0** |
| Flash | S0 (baseline) | **0.460** | 0.90 | 0.622 | **2** | **3** |

Reported once, unmodified, no reruns.

**The chosen strategy lost to the baseline on mean boundary IoU on held-out data**
(0.444 < 0.460). The tune-set +0.05 edge was overfit to the 30 tune episodes and did
not transfer — precisely the failure the held-out split exists to catch.

**But the mean hides the distribution.** S0-Flash reaches 0.460 *while putting 5 of 20
test episodes (25%) into a failure band* — 2 degenerate single-segment, 3 uniform-fifths.
Pro-S2 reaches 0.444 with **0 of 20** in any failure band. If your downstream use silently
trusts auto-labels, S0's "higher mean" includes 25% catastrophic episodes; the grounded
strategy trades ~0.016 mean IoU for never emitting one. Which you prefer depends on
whether a bad label is worse than a mediocre one — for training-data curation, it usually is.

---

## Quality: the metric is near-degenerate here

The human gold quality distribution is **49/50 score-5** (tune: 29×5 + 1×4; test: 20×5).
So a model that **always answers "5"** scores **0.97 exact on tune and 1.00 on test** —
**above both Gemini Flash and Pro.** Read the exact-agreement column with that in mind:

| | constant-5 baseline | Flash | Pro |
|---|---|---|---|
| quality exact (tune) | **0.967** | 0.700 | 0.867 |
| quality exact (test) | **1.000** | 0.900 | 0.950 |

The metric that actually matters on data without quality variance is the **catastrophic
false-negative rate** — episodes the model scores **≤2 while the human scored ≥4** (the
silent-filtering hazard):

| | Flash | Pro |
|---|---|---|
| catastrophic FN (tune) | **3 / 30 (0.10)** | **1 / 30 (0.033)** |
| catastrophic FN (test) | 0 / 20 | 0 / 20 |

**This is the real quality story:** the stronger model's value is **fewer catastrophic
false-negatives** (3 → 1 on tune), not higher exact agreement (which the trivial baseline
wins). On test neither model produced a catastrophic miss. Quality-score *discrimination*
cannot be meaningfully evaluated on this dataset — see Limitations.

---

## Subgoal agreement regressed — and why that is partly an artifact

Grounded strategies' subgoal frame agreement craters (S0-Flash 0.42 tune / **0.62 test**
→ grounded ≈ 0.10–0.12). Two things are true at once:

1. It is a **real cost**: subgoals are subtask end-frames, so moving boundaries moves
   subgoals away from the human's exact-frame picks, and "exact frame match" is unforgiving.
2. It is **partly a measurement artifact**: the human gold subgoals were reviewed against
   the *original* S0 segmentation (many were `accept_auto`), so they are anchored to S0's
   end-frames. *Any* re-segmentation — including a better one — diverges from them. Do not
   read the 0.62 → 0.12 drop as "6× worse subgoals"; read it as "subgoal-frame agreement is
   confounded with the segmentation the gold was built on." A clean test needs subgoals
   labeled independently of any one segmentation.

---

## Three exhibits (S0-Flash vs the winner vs human gold)

One S0-Flash failure from each band, the same episode under the chosen strategy (Pro-S2),
and the human boundaries. Numbers are subtask **end-frames**.

**(a) Degenerate — episode 11** (201 frames)
- S0-Flash: `[200]` — one segment, the whole episode. Useless as conditioning.
- Pro-S2: `[73, 109, 145, 164, 200]` (approach/grasp/transport/release-place/retract)
- human: `[70, 87, 130, 200]` — Pro-S2's first boundary (73) lands within 3 frames of the
  human's (70). **Clean win for grounding.**

**(b) Uniform-split — episode 14** (223 frames)
- S0-Flash: `[44, 89, 133, 178, 222]` — five near-identical ~44-frame chunks, ungrounded.
- Pro-S2: `[81, 101, 161, 182, 222]`
- human: `[83, 93, 136, 153, 222]` — Pro-S2's first boundary (81) matches the human (83);
  S0's (44) is off by ~40 frames. **Clean win for grounding.**

**(c) Drifted — episode 7** (203 frames) — *the honest counter-example*
- human: `[202]` — **the human labeled this as a single continuous action.**
- S0-Flash: `[40, 120, 161, 202]` (4 seg) — over-segmented.
- Pro-S2: `[55, 92, 129, 147, 202]` (5 seg) — **also over-segmented, and S2's
  min-granularity floor *forbids* it from ever matching the single-segment ground truth.**
  Here the anti-degenerate constraint actively hurts: not every episode is a 5-phase
  pick-and-place, but S2 is required to produce ≥3 phases.

---

## Gate behavior

The gate flagged, never dropped (`Episodes dropped by the gate: 0` on every run). On the
S0-Flash cells it raised the degenerate / uniform-split flags shown above and
`quality_outlier_needs_review` on the catastrophic-FN episodes (3 on Flash-tune, 1 on
Pro-tune) — i.e. the gate surfaced exactly the episodes a human should re-check, and the
grounded strategies produced none of the band flags.

---

## Cost accounting (vs the $30 ceiling)

Total spend **$16.54 / $30**, tracked continuously from on-disk receipts (the
authoritative number). Per-cell costs are in the tune table; Pro is ~4× Flash per call.

**One honest wrinkle:** the preflight cost *projection* read **$0** because its probe
(S1 on tune-ep-0) hit a cached receipt from the earlier 2-episode smoke, so the per-call
estimate was 0 and the budget *gate* never bound this run. No harm done — actual spend was
always tracked from receipts and landed at $16.54, well under the ceiling — but had Pro
been much pricier, the inert gate would not have caught it. Fix for next time: derive the
per-call estimate from a *fresh* (un-cached) probe, or from the published price table
directly. (The true-spend accounting was correct throughout; only the a-priori projection
was fooled.)

---

## Limitations (stated, not hidden)

- **One dataset.** `svla_so101_pickplace` is short, single-camera, tabletop pick-and-place,
  and **easy**: S0-Flash already clears 0.46 on test and 49/50 episodes are human-rated 5.
  Conclusions may invert on long-horizon, multi-stage, or genuinely variable-quality data.
- **One annotator's gold**, 50 episodes; the held-out test is only 20. The +0.05 selection
  margin was 0.003 — small-sample fragile, as the test result showed.
- **Quality metric near-degenerate.** With 49/50 at score 5, exact agreement is dominated
  by the constant-5 baseline; only the catastrophic-FN rate is informative. Re-run on a
  dataset with real quality spread to evaluate quality *discrimination*.
- **Subgoal agreement is confounded** with the original S0 segmentation the gold was built
  against (above).
- **Tune-set rubric exposure.** The grounded prompts and gate thresholds were authored
  while looking at SO-101 behavior; some tune-set fit is baked into the prompts themselves,
  not just the selection. The held-out test mitigates but does not eliminate this.
- Test boundary IoU **0.444 < 0.65**, so the post-tuning full-dataset re-annotation
  (gated on ≥0.65) was **not** run — the result does not warrant reshipping the demo
  artifact as "the tool's real capability."

---

## Verdict

On this dataset, the strategy layer is **not** a free win on mean boundary IoU — the
held-out test says the S0-Flash baseline is as good or better on the average. Its real,
held-out-confirmed value is **robustness**: it eliminates the degenerate and uniform-split
failure bands entirely (25% → 0% of test episodes), and the stronger model cuts
catastrophic quality false-negatives. Whether that robustness is worth ~4× the cost
(Pro) and a subgoal-agreement regression is a judgment for your pipeline — and the only
reason we can state the trade-off honestly is that everything here was measured against a
human gold set in the same units, winners and losers reported side by side.
