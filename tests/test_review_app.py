from __future__ import annotations

from labelkit import review_app
from labelkit.demo import synthetic_episode


def test_review_app_imports_without_streamlit():
    # Module-level imports must be stdlib-only so the CLI can locate the app
    # without the review extra installed.
    assert hasattr(review_app, "run")


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
