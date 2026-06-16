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
