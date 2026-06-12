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

## Objective signal (computed now, gold-free)

These need no human grading — they come straight from the fresh annotations.

| | grounded-Flash (S2) | S0-Flash |
|---|---|---|
| episodes annotated | **20 / 22** | **0** (see note) |
| failure-band rate (degenerate or uniform-split) | **0 / 20** | — |
| mean segments / episode | 5.2 | — |
| episodes with per-boundary evidence | **20 / 20** | — |

Total API spend (both strategies): **$0.38 of the $10 ceiling**.

**The structural generalization result holds:** on a brand-new task, the grounded
strategy produced **zero** degenerate or uniform-fifths episodes, real phase
decompositions (mean 5.2 segments), and a frame-grounded evidence string on every
boundary — and the evidence referenced the *new* scene ("red cube lifts off the table",
"red cube arrives over the blue cube"), not parroted lego/brick. So the failure-band
elimination claim is **verified on two datasets**.

**Note on S0-Flash (honest):** partway through, the Gemini key hit a prepayment/credit
limit (HTTP 429). The pipeline's resilience handled it exactly as designed — 20 grounded
episodes were checkpointed and saved despite 2 late failures; nothing crashed — but the S0
contrast run, which started after credits were exhausted, got 0 episodes. The S0 contrast
on this dataset was "budget permitting" and credits did not permit. It can be filled by
topping up the key and re-running the `s0_flash` annotate command (it resumes).

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

Paste the resulting table here:

| strategy | items | boundary acceptance (±5f) | phase accuracy | **evidence factual-accuracy** | failure-band rate | usable / touch-up / garbage |
|---|---|---|---|---|---|---|
| grounded-Flash | 20 | _fill_ | _fill_ | **_fill_** | 0.00 | _fill_ |

**evidence factual-accuracy** is the metric no other tool reports: the fraction of the
model's stated reasons ("gripper contacts brick") that are factually true of the exact
frame they cite. It is the difference between "the boundary happened to land right" and
"the model knew why."

## What this report licenses the README to claim

- **Verified on two datasets:** grounding eliminates the degenerate/uniform failure bands
  (SO-101 held-out 5/20 → 0/20; fresh 0/20).
- **Pending the blind grading above:** boundary acceptance, phase accuracy, and evidence
  factual-accuracy on the fresh task. Until filled, the README treats these as not-yet-shown.
