"""Tests for the deterministic scorer modules: speed, novelty, curation, detect."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robolabel.curation import assign_tiers, curation_values
from robolabel.detect import detect_directory
from robolabel.novelty import novelty_scores
from robolabel.speed import bin_speeds, episode_speed_norm


def test_speed_norm_and_bins():
    assert episode_speed_norm(np.zeros((20, 6), dtype="float32")) == 0.0
    fast = np.cumsum(np.ones((20, 6)) * 0.5, axis=0)
    slow = np.cumsum(np.ones((20, 6)) * 0.01, axis=0)
    assert episode_speed_norm(fast) > episode_speed_norm(slow)
    # with a sufficient population, bin by tercile (higher = faster)
    pop = {f"e{i}": float(i) for i in range(15)}
    bins = bin_speeds(pop, min_population=15)
    assert bins["e0"] == "slow" and bins["e14"] == "fast"
    # corpus-relative GUARD: too small a population -> all None ("insufficient to tier")
    assert set(bin_speeds({"a": 0.01, "b": 0.5, "c": 5.0}).values()) == {None}


def test_active_window_motion_defined():
    from robolabel.speed import active_window
    # still for 10 frames, then a clear burst of motion, then still again
    a = np.zeros((40, 6), dtype="float32")
    a[12:28] = np.cumsum(np.ones((16, 6)) * 0.4, axis=0)
    a[28:] = a[27]
    w = active_window(a)
    assert 0 < w["active_frames"] < 40 and 0 < w["active_fraction"] < 1.0
    assert active_window(np.zeros((30, 6), dtype="float32"))["active_frames"] == 0  # no motion


def test_novelty_isolated_scores_higher():
    # three tight points + one far outlier
    embs = {"a": np.array([0.0, 0.0]), "b": np.array([0.1, 0.0]),
            "c": np.array([0.0, 0.1]), "out": np.array([9.0, 9.0])}
    nov = novelty_scores(embs, k=2)
    assert nov["out"] > max(nov["a"], nov["b"], nov["c"])


def test_curation_value_and_tiers():
    q = {"hi": 5, "mid": 3, "lo": 1}
    nov = {"hi": 0.9, "mid": 0.5, "lo": 0.1}
    vals = curation_values(q, nov, w_quality=0.5, w_novelty=0.5)
    assert vals["hi"] > vals["mid"] > vals["lo"]
    # compress -> fidelity tiers by value tercile (min_population lowered to test the logic)
    tiers = assign_tiers(vals, compress=True, min_population=3)
    assert tiers["hi"] == "full" and tiers["lo"] == "minimal"
    # top_cut -> keep the top fraction, cut the rest
    cut = assign_tiers(vals, top_cut=0.34, min_population=3)
    assert cut["hi"] == "keep" and cut["lo"] == "cut"
    # quality weight only -> ranks by quality
    vq = curation_values(q, nov, w_quality=1.0, w_novelty=0.0)
    assert vq["hi"] > vq["lo"]


def test_curation_tier_guard():
    from robolabel.curation import tierable
    big = {f"e{i}": i / 14 for i in range(15)}             # 15 eps, full spread
    small = {"a": 0.5, "b": 0.51, "c": 0.49}               # tiny, near-homogeneous
    assert tierable(big) and not tierable(small)
    # guard: too small / homogeneous -> all None (the run reports "insufficient population to tier")
    assert set(assign_tiers(small, compress=True).values()) == {None}
    assert assign_tiers(big, compress=True)["e14"] == "full"


def test_detect_directory_config(tmp_path: Path):
    cfg = {"control_space": "joint", "arm_dims": [0, 1, 2], "gripper_dims": [3],
           "phase_vocabulary": ["reach", "wipe"]}
    p = tmp_path / "dir.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    d = detect_directory(p)
    assert d.control_space == "joint" and d.gripper_dims == [3] and d.arm_dims == [0, 1, 2]
    assert d.n_action_dims == 4 and d.phase_vocabulary == ["reach", "wipe"]
    # no config -> empty, no crash
    assert detect_directory(None).control_space is None
