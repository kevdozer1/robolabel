# robolabel acceptance review — 90-minute session

A guided session for **you, the author**, to personally verify that robolabel's
annotations are correct, useful, and honestly characterized — and to end by signing off
on each public claim or cutting it. Work top to bottom; tick the boxes. Everything is
zero-API except a fresh annotation you already paid for (~$1, under the $10 ceiling).

**One-time setup** (5 min):

```bash
pip install -e '.[lerobot]'
printf 'GEMINI_API_KEY=...\n' >> .env            # only needed if you re-annotate
# build the SO-101 viewer data from the cached sweep receipts (zero-API):
python scripts/build_inspect_data.py from-eval \
  --gold ../robovid_work/so101_gemini/gold.json --eval-out eval_out --out inspect_data/so101.json
```

---

## 0. Open the viewer (2 min)

```bash
robolabel inspect --data inspect_data/so101.json \
  --source lerobot --target lerobot/svla_so101_pickplace --camera-key observation.images.side
```

- [ ] The browser opens; episodes list on the left, video + a stack of color-coded
  boundary tracks (gold, S0-Flash, grounded, S_grip, uniform-fifths) in the middle, a
  Metrics tab on the right. Press space to play, ←/→ to step.

## 1. The 5 worst grounded episodes (15 min)

Sort dropdown → **"worst grounded IoU first."** For each of the top 5:

- [ ] Watch the clip. Does the **grounded** track's segmentation match what you see —
  even though its IoU is low? (Low IoU can mean "off by a few frames," not "wrong.")
- [ ] Compare to **gold**: is gold actually better, or just different? (Remember gold was
  made by correcting S0, so it is S0-anchored.)
- [ ] Note any episode where the model is genuinely wrong (not just drifted). __________

## 2. The failure-band episodes (15 min)

Filter dropdown → **"gate-flagged only."** These are episodes the gate flagged
(degenerate / uniform-split / quality-outlier).

- [ ] For each flagged episode, confirm the flag is *correct* (the S0 or grounded track
  really is degenerate/uniform, or the quality score really is a hallucinated outlier).
- [ ] Confirm the **grounded** track is *not* in a failure band on these — i.e. grounding
  fixed what S0 got wrong. This is the "25% → 0%" claim; verify it episode by episode.

## 3. Evidence spot-checks (15 min)

Pick ~8 episodes across the list. On each, open the **Evidence** tab.

- [ ] Each grounded evidence string ("gripper contacts brick") sits next to a thumbnail of
  its cited frame. Click the thumbnail to jump there. **Is the claim factually true of
  that frame?** Tally yes/no in your head; this is the evidence-accuracy claim.
- [ ] Flag any evidence string that is confident but false. __________

## 4. The fresh-dataset blind trial (25 min) — the real test

A second dataset (`lerobot/svla_so100_stacking`, apache-2.0) you have never touched, with
**no S0-anchored gold**. Build the blind items and grade them:

```bash
# build blind items (grounded-Flash + S0-Flash per episode, shuffled, identity hidden):
python scripts/build_inspect_data.py from-annotations --blind \
  --track grounded-Flash=fresh_stacking/grounded_flash S0-Flash=fresh_stacking/s0_flash \
  --dataset lerobot/svla_so100_stacking --out fresh_stacking/blind.json
# grade them (records to grades.json):
robolabel inspect --data fresh_stacking/blind.json --grades fresh_stacking/grades.json \
  --source lerobot --target lerobot/svla_so100_stacking --camera-key observation.images.top
```

For every item (the **Grade** tab), against the **video only** (identity is hidden):

- [ ] each boundary: within ±5 frames of the true transition? (yes/no)
- [ ] each phase label: correct? (yes/no)
- [ ] each evidence string: true of its cited frame? (yes/no)
- [ ] overall verdict: usable / needs touch-up / garbage.

When all items are graded:

```bash
robolabel trial-report --grades fresh_stacking/grades.json \
  --unblind fresh_stacking/blind.unblind.json --out fresh_stacking/trial_tally.md
```

- [ ] Open `fresh_stacking/trial_tally.md` and paste its table into the "Blind grading"
  section of `FRESH_TRIAL_REPORT.md` (which already holds the objective stats). **This is
  the generalization claim.** Note the evidence factual-accuracy rate — the metric no other
  tool reports. __________
- [ ] (S0-Flash contrast was lost to a credit limit during annotation; top up the key and
  re-run the `s0_flash` annotate command if you want it — it resumes.) __________

## 5. The two queries (8 min)

```bash
robolabel query --annotations fresh_stacking/grounded_flash --phase grasp \
  --source lerobot --target lerobot/svla_so100_stacking --camera-key observation.images.top \
  --out grasp_montage.png
robolabel query --annotations fresh_stacking/grounded_flash --needs-review
```

- [ ] Open `grasp_montage.png` — every tile should actually be a grasp. (If some are not,
  the phase labels are not query-trustworthy — note it.)
- [ ] The needs-review list: are these really the episodes you'd re-check?

## 6. Claims sign-off (5 min)

Open `CLAIMS.md`. For each row, mark your verdict from this session:

- [ ] **verified** — keep the claim in the README.
- [ ] **verified-on-one-dataset** — keep, with the caveat.
- [ ] **cut** — the session did not support it; strike it from the README.

---

## Go / no-go (fill at the end)

| public claim | survives? (yes/cut) | note |
|---|---|---|
| "drafts subtask boundaries, quality, subgoals for LeRobot" | | |
| failure-band tail 25% → 0% (grounding eliminates degenerate/uniform) | | |
| boundary placement: grounded beats S0 on held-out transitions (±5f) | | |
| evidence strings are mostly factually true of their frame | | |
| stronger model cuts catastrophic quality false-negatives | | |
| exports to the LeRobot subtask convention, indices resolve | | |
| generalizes to a fresh dataset (fresh-trial numbers) | | |
| these annotations improve training | **cut (untested)** | training utility never shown |

When this table is filled, the README's "How well does it work" section must match it
exactly — no claim survives that you did not tick.
