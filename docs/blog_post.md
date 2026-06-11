<!-- DRAFT — for the author to edit before posting. Not published. Voice modeled on
docs/why.md: honest, technical, no marketing. Numbers are final (from STRATEGY_REPORT.md). -->

# DRAFT: A better prompt did not make better boundaries (and what did)

I built a layer of "smarter" annotation strategies for [robolabel](../README.md) — a
tool that uses a VLM to draft subtask boundaries, episode quality, and subgoal frames for
LeRobot datasets, and then measures those drafts against a human gold set instead of
trusting them. The strategies were supposed to fix the boundaries. Then I ran a held-out
test, and the honest answer is: **on this dataset, the strategy layer did not improve the
average boundary, and the average boundary was the wrong thing to measure.**

This is a short writeup of what the measurement actually said. It is in the same spirit as
[*Do these annotations improve training?*](why.md): the point of the tool is to make it
hard to fool yourself, including about the tool's own features.

## The setup

Baseline Gemini 2.5 Flash, asked to "segment this episode into subtasks," scores a
subtask-boundary temporal IoU of **0.457** against my 50-episode human gold set on
`lerobot/svla_so101_pickplace`.[^numbers] Three visible failure modes: a degenerate single
"complete the task" segment, a correct phase count with boundaries at uniform fifths of
the duration (never grounded to the video), and plausible-but-drifted boundaries.

So I built S0→S4: frame-indexed grounding (return boundaries as frame indices with a
one-line visual evidence string), a closed phase vocabulary with a minimum-granularity
floor, a dense-window refinement pass, and self-consistency. I froze a 30-episode
tune / 20-episode test split, tuned only on tune, and scored the chosen cell once on test.

## The result that mattered

The mechanical selection (highest tune IoU, cheapest within 0.02, must beat S0-Flash by
≥0.05) picked **Gemini 2.5 Pro, S2** at tune IoU 0.453 — clearing the bar by 0.003. On the
held-out test it scored **0.444**. The S0-Flash baseline scored **0.460** on the same 20
episodes.

The strategy that won on tune **lost to the baseline on held-out mean IoU.** A 0.003 tune
margin did not survive contact with 20 unseen episodes. If I had reported the tune number
as the result, I would have shipped a regression dressed as a win.

## Why mean IoU was the wrong number

Mean temporal IoU rewards getting a segment's *extent* roughly right; it barely cares
whether you nailed the transition *frame*. When I added metrics that do care, the picture
inverted:

- **Boundary recall within ±5 frames** (did you hit the actual transition?): on the
  held-out test, the grounded winner placed **0.307** of gold boundaries within ±5 frames;
  S0-Flash placed **0.226**. The grounded strategy hits ~36% more exact transitions — and
  wins on test — exactly where mean IoU said it lost.
- **Failure-band rate:** S0-Flash put 5 of 20 test episodes into a degenerate or
  uniform-fifths failure mode. The grounded strategy put **0**. S0's "higher mean" is
  propped up by some good episodes while a quarter of them are silent garbage.
- **Catastrophic quality false-negatives** (the model scores a human-5 episode ≤2, i.e.
  it would be silently filtered out of a training set): Flash did this 3 times on tune,
  Pro once. That is the quality number that matters here — because exact agreement is
  degenerate: 49 of 50 gold episodes are rated 5, so a model that *always says 5* scores
  0.97–1.00, above both Flash and Pro.

So the discriminating signals are failure-band rate, boundary placement, and catastrophic
false-negatives. Mean IoU is a near-wash sitting ~0.10 above a uniform-fifths trivial
baseline (0.36) — there is real signal above trivial, but it is not in the average.

## The example that changed the tool

Episode 7's human gold is a *single* continuous segment. My S2 strategy had a
min-granularity floor that rejected any single-segment output and forced ≥3 phases — so on
ep7 it was structurally incapable of matching the ground truth, and every boundary it drew
was a false positive. Boundary precision caught this where IoU mostly hid it.

The fix is a tool change, not a metric: the min-granularity rule is now a configurable
policy that **defaults to warning** (accept the single segment, flag it a
`single_segment_candidate`) instead of hard-rejecting. Not every episode is a five-phase
pick-and-place, and the tool should not pretend otherwise.

## What I'm actually claiming

Grounding (and the stronger model) buy **robustness and data hygiene** on this dataset —
no degenerate or uniform episodes, more exact transitions, fewer catastrophic quality
misses — at ~4× the cost for Pro and a subgoal-agreement regression that is partly an
artifact of how the gold was built.[^anchor] They do **not** buy a higher average boundary
IoU on held-out data. Whether that trade is worth it is a pipeline decision, and the only
reason I can state it precisely is that every cell, every free baseline, and every metric
was scored against the same human gold, winners and losers side by side.

That is the whole pitch of the tool, applied to itself: measured labels, not asserted ones.

[^numbers]: One canonical baseline number per context: **0.457** is S0-Flash over the full
50-episode calibration set; **0.400 / 0.460** are the same baseline on the 30-episode tune
/ 20-episode test subsets used for the comparisons.

[^anchor]: The human gold was created by *correcting* S0 drafts, so it is anchored toward
S0's boundary placement and granularity — a quiet home-field advantage for S0 on every
boundary metric. The grounded strategy winning boundary placement on held-out data *despite*
that bias is the stronger reading, not the weaker one.
