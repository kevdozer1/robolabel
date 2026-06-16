# Run summary (proprioception grasp/release snap — the last refinement)

**Gold result (SO-101 pick-place grasp/release boundaries, deterministic, zero API):** recall@±5
**0.174 (4/23, MAE 3.75) without** the snap → **0.130 (3/23, MAE 3.33) with** it at a small
window; a window sweep (5→12) on both `observation.state` and `action` stays ≤ baseline. It does
**not** improve recall, because the SO-101 gold is S0-anchored, not gripper-aligned (the actual
gripper close is frame ~169 while the gold marks 159/174 and the grounded grasp-onset sits at 82).
**So the snap is OFF by default** (`segmentation.snap_contact`) — kept available for gripper-aligned
ground truth; it no-ops cleanly on pour/fold where the gripper holds. **Grasp/release timing is the
documented precision limit**: two independent refinement attempts (VLM dense-window, deterministic
gripper snap) didn't move it. **New API spend: $0** (validation reused existing annotations + dataset
reads). GIF re-rendered for the requested cosmetics only (caption removed, larger video panels) —
not for the snap. 137 tests pass, ruff clean. `snap.py`, `eval_snap.py`, CLAIMS row 25.

---

# Run summary (gallery-review cleanup)

**Changed:** population stats are now **corpus-relative** — novelty + curation/speed tiers are pooled
across all datasets with global thresholds and a population guard (small/same-y runs emit raw scores +
"insufficient population to tier" instead of fake bands); **speed** became a continuous, phase-agnostic
`active_duration` (motion onset→offset, +`active_fraction`); the open-vocab prompt was de-primed of a
hallucinated terminal retract and the trailing wind-down collapse now merges different-label retracts;
a bounded grasp/release refinement was tried, didn't help (recall@±5 0.211→0.211 vs gold), so it's off
by default with grasp/release timing documented as the precision limit (schema → v6; CLAIMS rows 17–24).
**Pour hallucination:** the forced terminal `retract` dropped from **8/8 → 3/8** episodes (the 3 are
plausibly real post-pour withdrawals).
**Artifacts:** new eval GIF `docs/figures/grounded_annotations.gif` (3 fixed episodes; phase→target,
quality, active_duration, real keyframes "selected — not generated", action coordinate frame) and
`PRICE_EFFICIENCY.md` (deterministic modules are free → full-stack ≈ minimal; Pro's 4× buys only quality;
~$20–28 / 1,000 eps on Flash, ~$6–10 with batch+caching).
**State:** 132 tests pass, ruff clean, new API spend **$0.70 / $3**; `main` pushed.
**One thing left for you:** eyeball the new GIF + `robolabel gallery --config gallery.json` to sign off the
labels read right — and decide whether the grasp/release precision ceiling is worth a future
proprioception-fused (non-VLM) refinement pass, since the VLM dense-window attempt didn't move it.

---

# Run summary (unified evaluation gallery)

A unified evaluation **gallery** now shows every module's output across all tasks in one place,
reusing the existing `robolabel inspect` machinery (dependency-free, offline, zero-API). Landing
= a card grid grouped by task (thumbnail + quality / speed / novelty / curation tier), sortable +
filterable; clicking a card opens the per-episode inspect view with a new **Modules** panel
(honest labels: control = action coordinate frame; subgoals = selected/retrieved pointers, not
generated; curation = value = f(quality, novelty), downstream utility unvalidated) and a run
header of enabled modules.

**Launch:**
```bash
python scripts/make_gallery.py            # builds gallery.json from the run outputs
robolabel gallery --config gallery.json
```

**Data (everything-on, ≤8 eps each):** pick-place 8 ($0.139), fold 8 ($0.171), pour 5 (reused,
$0 new). **New API spend this run: $0.31 of the $3 ceiling.** **128 tests pass** (+3 gallery data
assembly), ruff clean. **Offline validated:** all 3 tasks (21 episodes) render end-to-end with
`HF_HUB_OFFLINE=1` and the VLM network path blocked — zero network calls, frames (incl. retrieved
cross-episode subgoals) served from cache.

**Decisions from the last run, resolved:** (1) curation stays **off by default** with its honest
"downstream utility unvalidated" status — no curation-utility experiment (that's a separate
real-arm training project), no change to the module. (2) `main` **pushed** (consolidation +
gallery). Frozen ablation, eval split, S0, and closed-vocab default untouched.

---

# Run summary (consolidation → config-driven modular pipeline)

robolabel is now an **automated, model-agnostic conditioning-annotation + curation pipeline for
VLA finetuning on LeRobot data**, driven by one run-config with a module registry. Architecture
+ a few small deterministic scorers; the frozen ablation, eval split, S0, and closed-vocab path
are untouched. **125 tests pass, ruff clean. Total new API spend: $0.48 of the $6 ceiling.**

## The config surface

**Minimal** (segmentation + quality only; open-vocab grounded is the default):
```yaml
run:
  dataset: { source: lerobot, target: lerobot/svla_so101_pickplace }   # camera_key: auto
  model:   { provider: gemini, name: gemini-2.5-flash }
  probe:   { max_episodes: 10 }
  out: run_out/pickplace
```
**Everything-on** adds `speed`, `subgoals` (+gate-passed retrieval), `control`, `novelty`,
`curation` — see `configs/run_full.yaml` and [`CONFIG.md`](CONFIG.md). For a LeRobot dataset you
provide nothing else; non-LeRobot inputs need a tiny JSON ([`PORTING.md`](PORTING.md)).
`robolabel run --config run.yaml`.

## Modules — net-new vs refactored vs corrected

| module | status | note |
|---|---|---|
| **run-config spine + module registry** | **net-new** | `run.py`; minimal default = segmentation + quality |
| **auto-detect** (camera/fps/control-space/arm-gripper) | **net-new** | `detect.py`; zero config for LeRobot |
| `speed` | **net-new** | `speed.py`; deterministic pace, binned vs dataset (π0.7 metadata) |
| `novelty` | **net-new** | `novelty.py`; distance to NN in a frame embedding |
| `curation` | **net-new** | `curation.py`; value = f(quality, novelty) + value-tiered overlay (never deletes) |
| `segmentation` | **refactored** | now a module; **open-vocab default** (closed-vocab `S2` still available) |
| `quality` | **refactored** | now a module |
| `subgoals` | **refactored** | both kinds are pointers (no image files); retrieval only from **gate-passed** episodes; off by default |
| `control` | **corrected** | `control_modality` = action **coordinate frame** (joint vs Cartesian), NOT gripper involvement; `active_dof` demoted to optional/off (low-discrimination, mostly `both`) |

## Probe (5–10 eps each; never a large run)

- **Minimal default** on pick-place / pour / fold (5 eps each): no failures, no regression.
  Auto-detect confirmed — cameras `up` / `front` / `overhead`, all `control_space: joint`,
  arm `0–4` / gripper `5`; strategy `S2-open`. Cost **$0.111 / $0.142 / $0.100**.
- **Everything-on** (pour, 5 eps): all 7 modules produced sensible fields end-to-end — quality
  varies 3–5 (informative here), speed `slow/medium/fast`, novelty 0.41–0.63, `curation_value`
  0.03–1.0 with `full/reduced/minimal` tiers, retrieval populated from gate-passed episodes,
  `control_modality: joint`. Cost **$0.130**. Schema v5.

## CLAIMS rows touched

17 (control corrected), 18 (subgoal retrieval = gate-passed pointers), **19 speed**, **20
novelty**, **21 curation** (machinery sound + precedented; downstream utility UNVALIDATED), **22
open-vocab-default**; row 16 supplies the open-vocab quality evidence. Schema → v5 (additive;
v1–v4 still read).

## The one decision left to you

**`curation` is the only module whose downstream value is unproven** — the machinery is sound and
precedented (Smart Black Box value-tiering; "train on the top-value ~20%"), but whether the
value score / compression tiers actually improve a finetune is UNVALIDATED (CLAIMS row 21). It
ships **off by default** with that honest status. Decision: keep it that way, or prioritize a
downstream curation-utility experiment before promoting it. (Routine: nothing pushed/published —
say the word and I'll push `main`.)
