"""``robolabel run`` — one YAML config drives a modular conditioning + curation pipeline.

A run-config has a ``run`` block (dataset / model / probe) and a ``modules`` block where each
module is independently toggleable. The minimal default runs only **segmentation + quality**
(open-vocab grounded). Enabled modules execute in a fixed dependency order; dataset-level
modules (novelty, curation, retrieval) run after the per-episode pass. See ``CONFIG.md``.

Nothing here touches the frozen ablation, the eval split, or S0 — ``run`` is a new spine over
the existing labelers (`segment_episode`, `label_metadata`, `derive_subgoals`) plus the
deterministic scorers (`speed`, `novelty`, `curation`, `control`).
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Module registry: name -> {scope, requires}. Execution order is EPISODE_ORDER then DATASET_ORDER.
MODULES: dict[str, dict] = {
    "segmentation": {"scope": "episode", "requires": ()},
    "quality":      {"scope": "episode", "requires": ()},
    "speed":        {"scope": "episode+dataset", "requires": ()},
    "subgoals":     {"scope": "episode+dataset", "requires": ("segmentation",)},
    "control":      {"scope": "episode", "requires": ("segmentation",)},
    "novelty":      {"scope": "dataset", "requires": ()},
    "curation":     {"scope": "dataset", "requires": ("quality", "novelty")},
}
EPISODE_ORDER = ["segmentation", "quality", "speed", "subgoals", "control"]

DEFAULTS: dict[str, Any] = {
    "run": {
        "dataset": {"source": "lerobot", "target": None, "camera_key": "auto", "directory_config": None},
        "model": {"provider": "gemini", "name": "gemini-2.5-flash"},
        "probe": {"max_episodes": 10},
        "out": "run_out",
        "seed": 0,
    },
    "modules": {
        "segmentation": {"enabled": True, "strategy": "grounded", "vocabulary": "open"},
        "quality":      {"enabled": True},
        "speed":        {"enabled": False, "cuts": [0.3333, 0.6667]},
        "subgoals":     {"enabled": False, "retrieval": False, "retrieval_method": "embedding"},
        "control":      {"enabled": False, "active_dof": False},
        "novelty":      {"enabled": False, "k": 5},
        "curation":     {"enabled": False, "compress": False,
                         "weights": {"quality": 0.5, "novelty": 0.5}, "top_cut": None},
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class RunConfig:
    data: dict

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        return cls(_deep_merge(DEFAULTS, d or {}))

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    @property
    def run(self) -> dict:
        return self.data["run"]

    @property
    def modules(self) -> dict:
        return self.data["modules"]

    def enabled(self) -> list[str]:
        return [m for m in MODULES if self.modules.get(m, {}).get("enabled")]

    def validate(self) -> None:
        on = set(self.enabled())
        for m in on:
            missing = [r for r in MODULES[m]["requires"] if r not in on]
            if missing:
                raise ValueError(f"module '{m}' requires {missing} to be enabled too")


def resolve_strategy(seg_cfg: dict):
    """segmentation config -> a StrategyConfig. open-vocab grounded is the default."""
    from .strategy import load_strategy
    strat = str(seg_cfg.get("strategy", "grounded")).lower()
    if strat in ("baseline", "s0"):
        return load_strategy("S0")
    if strat not in ("grounded", "s2", "s2-open"):
        return load_strategy(seg_cfg["strategy"])         # power user: a direct preset / JSON path
    vocab = str(seg_cfg.get("vocabulary", "open")).lower()
    return load_strategy("S2-open" if vocab == "open" else "S2")


def _gate_passed(df) -> set[str]:
    """Episode ids with NO failure-band flag (degenerate / uniform-split) — safe to retrieve from."""
    from .gate import is_degenerate_single_segment, is_uniform_split
    from .schema import episode_records, list_episode_ids
    ok = set()
    for eid in list_episode_ids(df):
        st = [{"start_frame": s.get("start_frame"), "end_frame": s.get("end_frame")}
              for s in episode_records(df, eid)["subtasks"]]
        if st and not is_degenerate_single_segment(st) and not is_uniform_split(st, 0.12, 3):
            ok.add(str(eid))
    return ok


def run_pipeline(config: RunConfig, *, source=None, provider=None, rubric=None,
                 progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Execute the enabled modules. ``source``/``provider`` may be injected (tests/offline)."""
    import numpy as np

    from .control import load_actions, segment_active_dof
    from .curation import assign_tiers, curation_values
    from .detect import detect_directory, detect_lerobot
    from .labelers.metadata import label_metadata
    from .labelers.segmentation import segment_episode
    from .labelers.subgoals import derive_subgoals
    from .novelty import episode_embedding, novelty_scores
    from .retrieve import retrieve_subgoals
    from .rubric import load_rubric
    from .schema import (
        EpisodeAnnotation,
        EpisodeMetadata,
        episode_records,
        list_episode_ids,
        read_annotations,
        save_dataframe,
        to_dataframe,
        write_annotations,
    )
    from .speed import active_window, bin_speeds, episode_speed_norm

    config.validate()
    rubric = rubric or load_rubric()
    mods = config.modules
    on = set(config.enabled())
    rcfg, dcfg = config.run, config.run["dataset"]
    out = Path(rcfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    limit = int(rcfg["probe"].get("max_episodes") or 0) or None

    # --- source + auto-detect ------------------------------------------------ #
    if source is None:
        from .adapters import build_source
        kwargs = {}
        if dcfg["source"] == "lerobot" and dcfg.get("camera_key") and dcfg["camera_key"] != "auto":
            kwargs["camera_key"] = dcfg["camera_key"]
        # Load only the first `limit` episodes (contiguous prefix) so a probe doesn't download
        # the whole dataset; frame indices stay aligned. For a full-scale run (no limit), load all.
        if dcfg["source"] == "lerobot" and limit:
            kwargs["episodes"] = list(range(limit))
        source = build_source(dcfg["source"], dcfg["target"], **kwargs)
    is_lerobot = getattr(source, "name", "") == "lerobot" or hasattr(source, "meta")
    actions_by_ep: dict = {}
    action_names = None
    if is_lerobot and ({"control", "speed"} & on):
        actions_by_ep, action_names = load_actions(dcfg["target"])
    detected = (detect_lerobot(source, action_names) if is_lerobot
                else detect_directory(dcfg.get("directory_config")))

    if provider is None:
        from .providers.base import build_provider
        provider = build_provider(rcfg["model"]["provider"], rcfg["model"]["name"])
    strategy = resolve_strategy(mods["segmentation"])

    # --- per-episode pass ---------------------------------------------------- #
    anns: list[EpisodeAnnotation] = []
    eps_by_id: dict = {}
    speed_norm: dict[str, float] = {}
    seg_cost = qual_cost = 0.0
    failures: list[dict] = []
    episodes = list(source)
    if limit:
        episodes = episodes[:limit]
    total = len(episodes)
    for i, ep in enumerate(episodes):
        if progress:
            progress(i + 1, total, ep.episode_id)
        try:
            rdir = out / "raw_receipts" / ep.episode_id
            subtasks, metadata = [], None
            if "segmentation" in on:
                res = segment_episode(ep, provider, rubric, strategy, rdir)
                subtasks = res.segments
                seg_cost += sum(c.estimated_cost_usd or 0 for c in res.calls)
            if "quality" in on:
                mres = label_metadata(ep, provider, rubric, rdir)
                metadata = mres.metadata
                qual_cost += sum(c.estimated_cost_usd or 0 for c in mres.calls)
            acts = actions_by_ep.get(ep.episode_id, getattr(ep, "actions", None))
            if "speed" in on:
                metadata = metadata or EpisodeMetadata()
                sn = episode_speed_norm(acts)
                speed_norm[ep.episode_id] = sn
                metadata.speed_norm = round(sn, 6)
                mo = rubric.speed_motion
                aw = active_window(acts, rel_threshold=mo["rel_threshold"], smooth=mo["smooth"])
                metadata.active_frames = aw["active_frames"]
                metadata.active_seconds = round(aw["active_frames"] / max(1e-6, ep.fps), 3)
                metadata.active_fraction = aw["active_fraction"]
            subgoals = derive_subgoals(subtasks, ep.num_frames, rubric.subgoal_source) if ("subgoals" in on and subtasks) else []
            if "control" in on:
                metadata = metadata or EpisodeMetadata()
                metadata.control_modality = detected.control_space
                if mods["control"].get("active_dof") and acts is not None and len(acts) >= 2:
                    er = np.asarray(acts).max(0) - np.asarray(acts).min(0)
                    grip = detected.gripper_dims or [np.asarray(acts).shape[1] - 1]
                    for s in subtasks:
                        s.active_dof = segment_active_dof(np.asarray(acts), s.start_frame,
                                                          min(s.end_frame, len(acts) - 1), grip, er,
                                                          rubric.active_dof_threshold)
            anns.append(EpisodeAnnotation(
                episode_id=ep.episode_id, task=ep.task, num_frames=ep.num_frames, fps=ep.fps,
                provider=provider.name, model=provider.model, metadata=metadata, subtasks=subtasks,
                subgoals=subgoals, receipts=[str(rdir)], strategy=strategy.name,
                cost_usd=None,
            ))
            eps_by_id[ep.episode_id] = ep
        except Exception as exc:  # noqa: BLE001 - resilience: skip a bad episode, keep the rest
            failures.append({"episode_id": ep.episode_id, "error": str(exc)[:200]})

    df = to_dataframe(anns)
    write_annotations(anns, out)

    # --- dataset-level pass -------------------------------------------------- #
    # corpus-relative tier guard: min population to bin against (config override, else rubric).
    min_pop = int(mods["curation"].get("min_population")
                  or mods["speed"].get("min_population") or rubric.curation_min_population)
    if "speed" in on and speed_norm:
        bins = bin_speeds(speed_norm, tuple(mods["speed"].get("cuts", [0.3333, 0.6667])),
                          min_population=min_pop)
        df["speed"] = df["speed"].astype("object")
        for eid, b in bins.items():
            df.loc[(df["episode_id"].astype(str) == eid) & (df["record_type"] == "episode_metadata"), "speed"] = b

    novelty_by_ep: dict[str, float] = {}
    if "novelty" in on:
        embs = {eid: episode_embedding(ep.frame, ep.num_frames) for eid, ep in eps_by_id.items()}
        novelty_by_ep = novelty_scores(embs, int(mods["novelty"].get("k", 5)))
        df["novelty"] = df["novelty"].astype("object")
        for eid, v in novelty_by_ep.items():
            df.loc[(df["episode_id"].astype(str) == eid) & (df["record_type"] == "episode_metadata"), "novelty"] = v

    if "curation" in on:
        quality_by_ep = {}
        for eid in list_episode_ids(df):
            q = episode_records(df, eid)["metadata"].get("quality")
            quality_by_ep[str(eid)] = (None if q is None or (isinstance(q, float) and q != q) else float(q))
        w = mods["curation"].get("weights", {})
        values = curation_values(quality_by_ep, novelty_by_ep, w.get("quality", 0.5), w.get("novelty", 0.5))
        tiers = assign_tiers(values, compress=bool(mods["curation"].get("compress")),
                             top_cut=mods["curation"].get("top_cut"), min_population=min_pop)
        df["curation_value"] = df["curation_value"].astype("object")
        df["curation_tier"] = df["curation_tier"].astype("object")
        for eid, v in values.items():
            mask = (df["episode_id"].astype(str) == eid) & (df["record_type"] == "episode_metadata")
            df.loc[mask, "curation_value"] = v
            df.loc[mask, "curation_tier"] = tiers.get(eid)

    if "subgoals" in on and mods["subgoals"].get("retrieval"):
        allowed = _gate_passed(df)
        method = mods["subgoals"].get("retrieval_method", "embedding")
        getter = (lambda e, f: eps_by_id[e].frame(int(f))) if eps_by_id else None  # noqa: E731
        df = retrieve_subgoals(df, frame_getter=getter, method=method,
                               seed=int(rcfg.get("seed", 0)), allowed_sources=allowed)

    save_dataframe(df, out)
    read_annotations(out)  # sanity: re-read

    return {
        "out": str(out),
        "modules_enabled": sorted(on),
        "episodes": len(anns),
        "failures": failures,
        "detected": detected.summary(),
        "strategy": strategy.name,
        "cost": {"segmentation": round(seg_cost, 6), "quality": round(qual_cost, 6),
                 "speed": 0.0, "subgoals": 0.0, "control": 0.0, "novelty": 0.0, "curation": 0.0,
                 "total": round(seg_cost + qual_cost, 6)},
    }
