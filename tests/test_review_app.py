from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from robovid_conditioner import review_app
from robovid_conditioner.demo import synthetic_episode


def test_review_app_imports_without_streamlit():
    # Module-level imports must be stdlib-only so the CLI can locate the app
    # without the review extra installed.
    assert hasattr(review_app, "run")


def test_review_gui_renders_and_save_keeps_auto_separate(tmp_path: Path):
    """Full GUI smoke via Streamlit AppTest: render, edit a score, save.

    Runs the real review_app.py the way ``streamlit run`` does (whole module,
    __main__ guard). This is the regression test for the relative-import crash
    that broke the GUI on first open. Skips if the review extra is absent.
    """
    st_testing = pytest.importorskip("streamlit.testing.v1")
    from robovid_conditioner.annotate import annotate_source
    from robovid_conditioner.demo import synthetic_source
    from robovid_conditioner.gold import load_or_sync_gold
    from robovid_conditioner.providers import build_provider

    out = tmp_path / "out"
    annotate_source(synthetic_source(3), out, provider=build_provider("mock"))
    gold = tmp_path / "gold.json"
    load_or_sync_gold(out, gold)

    sys.argv = ["review_app.py", "--annotations", str(out), "--gold", str(gold)]
    at = st_testing.AppTest.from_file(review_app.__file__, default_timeout=60)
    at.run()
    assert not at.exception, at.exception  # GUI must render with no error on first open

    at.slider[0].set_value(2).run()
    save = next(b for b in at.button if "Save" in b.label)
    save.click().run()
    assert not at.exception, at.exception

    g = json.loads(gold.read_text(encoding="utf-8"))
    ep = g["episodes"][0]
    assert ep["gold"]["metadata"]["quality"] == 2          # human edit landed
    assert ep["auto"]["metadata"]["quality"] is not None   # VLM label untouched


def test_linspace_endpoints():
    assert review_app._linspace(0, 10, 1) == [0]
    pts = review_app._linspace(0, 9, 4)
    assert pts[0] == 0 and pts[-1] == 9
    assert len(pts) == 4


def test_keyframe_strip_returns_frames_and_indices():
    ep = synthetic_episode(0, num_frames=20)
    frames, idxs = review_app._keyframe_strip(ep, n=5)
    assert len(frames) == len(idxs) == 5
    assert idxs[0] == 0 and idxs[-1] == 19
    assert frames[0].shape[-1] == 3  # RGB


def test_load_episode_frames_no_source_returns_empty():
    assert review_app._load_episode_frames(None, None) == {}
