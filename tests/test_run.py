"""Tests for the config-driven run orchestrator (offline, mock provider)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robolabel.episode import Episode
from robolabel.providers import build_provider
from robolabel.run import MODULES, RunConfig, resolve_strategy, run_pipeline
from robolabel.schema import episode_records, list_episode_ids, read_annotations


def _episodes(n_eps: int = 5, n_frames: int = 40, dim: int = 6) -> list[Episode]:
    eps = []
    for k in range(n_eps):
        color = np.uint8((k * 40) % 255)
        arr = np.full((48, 64, 3), color, dtype=np.uint8)
        rng = np.random.default_rng(k)
        # different motion scale per episode -> speed + novelty have spread
        actions = np.cumsum(rng.standard_normal((n_frames, dim)) * (0.05 + 0.05 * k), axis=0).astype("float32")
        eps.append(Episode(episode_id=str(k), num_frames=n_frames, fps=30.0, task="do the task",
                           get_frame=lambda i, a=arr: a, actions=actions))
    return eps


def _run(cfg: dict, tmp_path: Path):
    cfg = {**cfg}
    cfg.setdefault("run", {})["out"] = str(tmp_path / "out")
    config = RunConfig.from_dict(cfg)
    return run_pipeline(config, source=_episodes(), provider=build_provider("mock")), config


# --------------------------------------------------------------------------- #
# Config + registry
# --------------------------------------------------------------------------- #
def test_minimal_default_is_segmentation_and_quality():
    c = RunConfig.from_dict({})
    assert c.enabled() == ["segmentation", "quality"]
    # open-vocab grounded is the default segmentation strategy
    assert resolve_strategy(c.modules["segmentation"]).name == "S2-open"


def test_vocabulary_closed_still_available():
    c = RunConfig.from_dict({"modules": {"segmentation": {"vocabulary": "closed"}}})
    assert resolve_strategy(c.modules["segmentation"]).name == "S2"
    assert resolve_strategy({"strategy": "baseline"}).name == "S0"


def test_curation_requires_quality_and_novelty():
    c = RunConfig.from_dict({"modules": {"curation": {"enabled": True}, "novelty": {"enabled": False}}})
    with pytest.raises(ValueError, match="curation"):
        c.validate()


def test_module_registry_shape():
    assert MODULES["curation"]["requires"] == ("quality", "novelty")
    assert MODULES["subgoals"]["requires"] == ("segmentation",)


# --------------------------------------------------------------------------- #
# Orchestrator (offline)
# --------------------------------------------------------------------------- #
def test_minimal_run_offline(tmp_path: Path):
    res, _ = _run({"modules": {}}, tmp_path)
    assert res["episodes"] == 5 and not res["failures"]
    assert set(res["modules_enabled"]) == {"segmentation", "quality"}
    df = read_annotations(res["out"])
    rec = episode_records(df, "0")
    assert rec["subtasks"] and rec["metadata"].get("quality") is not None
    # disabled modules leave their columns null
    assert _allnull(df, "novelty") and _allnull(df, "speed")


def test_everything_on_run_offline(tmp_path: Path):
    cfg = {"modules": {
        "segmentation": {"enabled": True, "vocabulary": "open"},
        "quality": {"enabled": True},
        "speed": {"enabled": True},
        "subgoals": {"enabled": True, "retrieval": True, "retrieval_method": "random"},
        "control": {"enabled": True, "active_dof": True},
        "novelty": {"enabled": True, "k": 3},
        "curation": {"enabled": True, "compress": True, "weights": {"quality": 0.5, "novelty": 0.5}},
    }}
    res, _ = _run(cfg, tmp_path)
    assert res["episodes"] == 5 and not res["failures"]
    df = read_annotations(res["out"])
    metas = [episode_records(df, e)["metadata"] for e in list_episode_ids(df)]
    assert all(m.get("speed") in ("fast", "medium", "slow") for m in metas)
    assert all(m.get("novelty") is not None for m in metas)
    assert all(m.get("curation_value") is not None for m in metas)
    assert all(m.get("curation_tier") in ("full", "reduced", "minimal") for m in metas)
    # retrieval ran and respected the gate: the mock makes uniform-split segments (a failure
    # band), so NO episode is gate-passed and nothing may be retrieved from -> all null.
    # (Positive retrieval is covered in test_retrieve::test_retrieve_only_from_allowed_sources.)
    assert "retrieved_subgoal_episode_id" in df.columns
    sgs = [s for e in list_episode_ids(df) for s in episode_records(df, e)["subgoals"]]
    assert all(_isnull(s.get("retrieved_subgoal_episode_id")) for s in sgs)
    # active_dof present on subtasks
    subs = episode_records(df, "0")["subtasks"]
    assert all(s.get("active_dof") in ("arm", "gripper", "both", "none") for s in subs)


def _allnull(df, col) -> bool:
    s = df[df["record_type"] == "episode_metadata"][col]
    return bool(s.isna().all())


def _isnull(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)
