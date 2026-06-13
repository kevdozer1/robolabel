# Fresh-dataset trial — `lerobot/svla_so100_stacking`

A second dataset, **never used to build any robolabel number**, with **no
S0-anchored gold** — the test of whether the findings generalize off the one dataset
they were tuned on.

**Dataset choice (documented):** `lerobot/svla_so100_stacking` — Apache-2.0 (verified
on the HF page), LeRobot **v3.0**, SO-100 arm, task *"Put the red cube on top of the
blue cube"* (a genuinely different task from SO-101 lego pick-place), `observation.images.top`
+ `observation.images.wrist`, 30 fps. Chosen over the MIT `Tomas0413/so100_screw_lid_v0`
(v2.1 format risk + ~2500-frame episodes) for format safety and shorter clips. 20 episodes
annotated.

## Objective signal — paired, computed now, gold-free

Both strategies ran on **exactly the same 20 episodes** (0–19). No human grading needed;
these come straight from the fresh annotations.

| metric (over the same 20 episodes) | grounded-Flash (S2) | S0-Flash (baseline) |
|---|---|---|
| **failure-band rate** (degenerate or uniform-split) | **0 / 20** | **8 / 20** (3 degenerate + 5 uniform) |
| mean segments / episode | 5.2 | 4.0 |
| per-boundary evidence present | 20 / 20 | — (S0 produces none) |

Total API spend (both strategies, paired 20): **$0.765 of the $10 ceiling**.

**The failure-band elimination result holds — and is stronger here.** On a brand-new task,
paired on the same episodes, the baseline collapses **40% (8/20)** of episodes into a
single blob or uniform fifths, while the grounded strategy collapses **none (0/20)**. On
SO-101 held-out the same gap was 5/20 → 0/20; on this fresh dataset it is **8/20 → 0/20**.
So the claim "frame grounding eliminates the degenerate/uniform failure bands" is now
**verified on two datasets, paired**, with no S0-anchored gold involved. The grounded
evidence also referenced the *new* scene ("red cube lifts off the table", "red cube arrives
over the blue cube"), not parroted lego/brick — it is grounding to the actual video.

**Run note (honest):** the first annotation pass hit a Gemini credit limit (HTTP 429)
mid-run; the resilient `annotate_source` checkpointing saved every completed episode
(grounded 20/22), confirming the resilience works under a real mid-run failure. After a
$10 top-up, S0-Flash was run on exactly those 20 episodes for the paired contrast above.

## Blind grading (subjective) — author's session, all 40 items

The 40 blind items (20 grounded-Flash + 20 S0-Flash, same episodes, identity hidden,
shuffled) were graded blind by the author against the video. **Grading protocol:
mark-failures-only — the grader marked only the failures; every unmarked boundary, phase,
and evidence string counts as a pass over the known denominator** (the number of
boundaries / phases / evidence slots in the graded items, not just the marks entered).

| strategy | items | boundary acceptance (±5f) | phase accuracy | **evidence factual-accuracy** | failure-band | usable / touch-up / garbage |
|---|---|---|---|---|---|---|
| **grounded-Flash** | 20 | **0.86** (n=84) | 0.80 (n=84) | **0.98** (n=84) | **0.00** | **14 / 4 / 2** |
| S0-Flash | 20 | 0.66 (n=61) | 0.89 (n=61) | n/a (no evidence) | 0.40 | 4 / 6 / 9 |

**Read:** grounded wins decisively on boundaries (0.86 vs 0.66), on the verdict (14
"usable" vs 4), and on the metric no other tool reports — **evidence factual-accuracy
0.98**, i.e. when the model says "red cube lifts off the table" it is true of the cited
frame 98% of the time. The one place grounded *loses* is **phase accuracy (0.80 vs S0's
free-text 0.89)** — which is precisely the **label-underspecification** flaw the author's
grading surfaced: the closed phase label ("approach") often doesn't say *which* object.
The structured-label patch (phase → target) targets exactly this number.

**evidence factual-accuracy** is the metric no other tool reports: the fraction of the
model's stated reasons ("gripper contacts brick") that are factually true of the exact
frame they cite. It is the difference between "the boundary happened to land right" and
"the model knew why."

## What this report licenses the README to claim

- **Verified on two datasets:** grounding eliminates the degenerate/uniform failure bands
  (SO-101 held-out 5/20 → 0/20; fresh 0/20).
- **Pending the blind grading above:** boundary acceptance, phase accuracy, and evidence
  factual-accuracy on the fresh task. Until filled, the README treats these as not-yet-shown.
