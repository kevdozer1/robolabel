# Run config

`robolabel run --config run.yaml` drives the whole pipeline from one YAML file. A run-config has
a `run` block (dataset / model / probe) and a `modules` block where **each module is
independently toggleable**. The minimal default runs only **segmentation + quality** with
**open-vocabulary grounded** segmentation. Modules execute in dependency order; dataset-level
modules (novelty, curation, retrieval) run after the per-episode pass.

For a standard LeRobot dataset you provide **nothing** beyond `source`/`target` — camera key,
fps, control space, and arm/gripper dims are auto-detected (see [`PORTING.md`](PORTING.md)).

## Minimal (copy-paste)

```yaml
run:
  dataset: { source: lerobot, target: lerobot/svla_so101_pickplace }   # camera_key: auto
  model:   { provider: gemini, name: gemini-2.5-flash }
  probe:   { max_episodes: 10 }
  out: run_out/pickplace
# modules: omitted -> the default is segmentation + quality only
```

```bash
robolabel run --config run.yaml
```

## Everything on

```yaml
run:
  dataset: { source: lerobot, target: lerobot/svla_so101_pickplace, camera_key: auto }
  model:   { provider: gemini, name: gemini-2.5-flash }
  probe:   { max_episodes: 5 }
  out: run_out/full
  seed: 0
modules:
  segmentation: { enabled: true, strategy: grounded, vocabulary: open }   # vocabulary: closed for S2
  quality:      { enabled: true }
  speed:        { enabled: true, cuts: [0.3333, 0.6667] }
  subgoals:     { enabled: true, retrieval: true, retrieval_method: embedding }
  control:      { enabled: true, active_dof: false }                      # active_dof low-discrimination here
  novelty:      { enabled: true, k: 5 }
  curation:     { enabled: true, compress: true,
                  weights: { quality: 0.5, novelty: 0.5 }, top_cut: null }
```

## Modules

| module | scope | default | requires | does |
|---|---|---|---|---|
| `segmentation` | episode | **on** | — | grounded `phase → target` subtasks. `vocabulary: open` (default) = `S2-open`; `closed` = `S2`; `strategy: baseline` = S0 |
| `quality` | episode | **on** | — | episode quality 1–5 (VLM). Near-degenerate on easy datasets — see `speed` |
| `speed` | episode→dataset | off | — | deterministic pace (mean action velocity) binned `fast`/`medium`/`slow` vs the dataset |
| `subgoals` | episode→dataset | off | `segmentation` | real end-of-sub-step keyframe (pointer); `retrieval: true` adds a same-phase keyframe from another **gate-passed** episode (pointer). No image files written |
| `control` | episode | off | `segmentation` | `control_modality` (joint vs end-effector coordinate frame); `active_dof: true` adds per-segment arm/gripper/both (optional, low-discrimination) |
| `novelty` | dataset | off | — | deterministic per-episode novelty (distance to nearest neighbours in a cheap frame embedding) |
| `curation` | dataset | off | `quality`, `novelty` | `curation_value = f(quality, novelty)`; `compress: true` assigns a fidelity tier (`full`/`reduced`/`minimal`); `top_cut: 0.2` marks `keep`/`cut`. Overlay only — never deletes |

A module whose `requires` are not all enabled raises a clear error at validation. Everything is
additive in the sidecar (schema v5); see [`SCHEMA.md`](SCHEMA.md). All deterministic modules
(`speed`, `control`, `novelty`, `curation`) cost **$0** — only `segmentation`/`quality` call the
VLM, and `robolabel run` reports per-module cost.
