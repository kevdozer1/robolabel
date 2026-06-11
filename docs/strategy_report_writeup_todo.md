# STRATEGY_REPORT.md writeup TODO (apply AFTER the sweep completes)

Do not edit STRATEGY_REPORT.md while the sweep is running. When writing it up, apply
these, in addition to the tune/test tables and failure-band analysis:

## Quality-metric reframing (the gold set is near-degenerate on quality)

The human gold quality distribution is **49/50 score-5** — the dataset is genuinely
easy, so **quality exact-agreement is a near-degenerate metric here**. Report it, but
do not lean on it. Specifically:

1. **Add a constant-5 trivial baseline row** to the quality columns. On this gold set
   a model that always answers "5" scores **~0.98 exact agreement** — above both
   Gemini Flash and Pro. State this plainly: any quality-agreement number near 0.98
   is indistinguishable from the trivial baseline.
2. **Reframe the quality discussion around catastrophic false-negative rate**: the
   fraction of episodes where the model assigns **≤2 to a human-rated 4–5**. This is
   the real hazard — silent filtering of good data. Report this rate per
   model × strategy. (On the original 50-ep run this was 2/50: eps 45, 46 scored 1.)
3. **Known limitation, stated not hidden:** quality-score *discrimination* cannot be
   meaningfully evaluated on a dataset without real quality variance. To assess it,
   re-run on a dataset with a spread of human quality scores. List under Limitations.

## S_grip integration (WS2, deferred)

Merge `eval_out_grip/results_tune.json` (proprioceptive gripper-event baseline) into
the tune table as the **free / zero-API baseline row**, and run its single test cell
alongside the chosen VLM cell. If S_grip matches or beats the VLM boundary IoU on this
dataset, say so plainly in both the report and the README — that honesty is the point.

**Measured (tune, 30 eps, zero-API):** S_grip boundary IoU **0.204**, subgoal agreement
**0.060**, mean 4.27 segments, 0 degenerate / 0 uniform, $0/episode. This is **below**
the VLM strategies (S0-Flash 0.40, S3-Flash 0.43): gripper-close / EE-pause events fire
where the robot *acts*, which is several frames off from where a human draws the
*visual-semantic* subtask boundary. Report it as the honest free baseline that the VLM
beats on this clean dataset — and run S_grip's single TEST cell alongside the chosen VLM
cell during the test phase (`scripts/score_gripper.py --phase test`). Do NOT tune the
`gripper_baseline` thresholds to flatter the number; they stay at the rubric defaults.
