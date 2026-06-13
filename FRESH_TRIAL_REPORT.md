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

## Blind grading (subjective) — FILL by running the session

The boundary/phase/evidence acceptance numbers are **the author's to produce**, blind
(strategy identity hidden, against the video only). Run:

```bash
robolabel inspect --data fresh_stacking/blind.json --grades fresh_stacking/grades.json \
  --source lerobot --target lerobot/svla_so100_stacking --camera-key observation.images.top
# grade every item in the Grade tab, then:
robolabel trial-report --grades fresh_stacking/grades.json \
  --unblind fresh_stacking/blind.unblind.json --out fresh_stacking/trial_tally.md
```

The blind set is **40 items** — 20 grounded-Flash + 20 S0-Flash on the same episodes,
shuffled with identity hidden. Grade as many as you like; `trial-report` aggregates over
**whatever n you graded** (rates are computed only on graded items). Paste its table here:

| strategy | items graded | boundary acceptance (±5f) | phase accuracy | **evidence factual-accuracy** | failure-band rate | usable / touch-up / garbage |
|---|---|---|---|---|---|---|
| grounded-Flash | _n_ | _fill_ | _fill_ | **_fill_** | 0.00 | _fill_ |
| S0-Flash | _n_ | _fill_ | _fill_ | — (no evidence) | 0.40 | _fill_ |

**evidence factual-accuracy** is the metric no other tool reports: the fraction of the
model's stated reasons ("gripper contacts brick") that are factually true of the exact
frame they cite. It is the difference between "the boundary happened to land right" and
"the model knew why."

## What this report licenses the README to claim

- **Verified on two datasets:** grounding eliminates the degenerate/uniform failure bands
  (SO-101 held-out 5/20 → 0/20; fresh 0/20).
- **Pending the blind grading above:** boundary acceptance, phase accuracy, and evidence
  factual-accuracy on the fresh task. Until filled, the README treats these as not-yet-shown.
