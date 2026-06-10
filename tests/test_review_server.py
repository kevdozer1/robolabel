"""Tests for the browser calibration GUI's data + frame server.

The GUI is a stdlib http.server single-page app; these tests exercise the
ReviewSession (state, episode payload, frame JPEG, save) and one real HTTP
round-trip, all offline with a DirectoryAdapter of generated frames.
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from robovid_conditioner.adapters import DirectoryAdapter
from robovid_conditioner.annotate import annotate_source
from robovid_conditioner.gold import load_or_sync_gold
from robovid_conditioner.providers import build_provider
from robovid_conditioner.review_server import ReviewSession, make_handler


def _frame_dataset(root: Path, n_eps: int = 2, n_frames: int = 12) -> DirectoryAdapter:
    for ep in range(n_eps):
        d = root / f"ep_{ep:03d}"
        d.mkdir(parents=True)
        for i in range(n_frames):
            Image.fromarray(np.full((24, 32, 3), (ep * 20 + i * 6) % 255, dtype=np.uint8)).save(d / f"{i:04d}.png")
    (root / "episodes.jsonl").write_text(
        json.dumps({"episode_id": "ep_000", "task": "put cube in box", "fps": 12}) + "\n", encoding="utf-8"
    )
    return DirectoryAdapter(root)


def _session(tmp_path: Path) -> ReviewSession:
    data = tmp_path / "data"
    src = _frame_dataset(data)
    out = tmp_path / "out"
    annotate_source(src, out, provider=build_provider("mock"))
    gold = tmp_path / "gold.json"
    load_or_sync_gold(out, gold)
    # A fresh source for the session (adapters are single-pass iterables).
    return ReviewSession(out, gold, source=DirectoryAdapter(data))


def test_state_and_episode_payload(tmp_path: Path):
    s = _session(tmp_path)
    state = s.state()
    assert state["episode_count"] == 2
    assert state["reviewed_count"] == 0
    assert state["has_frames"] is True
    payload = s.episode_payload("ep_000")
    assert payload["task"] == "put cube in box"
    assert payload["num_frames"] == 12
    assert payload["has_frames"] is True
    assert len(payload["segments"]) >= 1
    assert payload["segments"][0]["color"].startswith("#")


def test_frame_jpeg_served(tmp_path: Path):
    s = _session(tmp_path)
    data = s.frame_jpeg("ep_000", 5)
    assert data is not None and data[:2] == b"\xff\xd8"  # JPEG SOI marker
    # cached on second call
    assert s.frame_jpeg("ep_000", 5) is not None


def test_save_review_writes_gold_keeps_auto(tmp_path: Path):
    s = _session(tmp_path)
    before = s.episode_payload("ep_000")["review"]["auto_score"]
    result = s.save_review({
        "episode_id": "ep_000", "score": 2, "mistake": True, "reason": "wrong object",
        "subtasks": [{"segment_idx": seg["segment_idx"], "accept_auto": True}
                     for seg in s.episode_payload("ep_000")["segments"]],
    })
    assert result["saved"] is True
    gold = json.loads((tmp_path / "gold.json").read_text(encoding="utf-8"))
    entry = next(e for e in gold["episodes"] if e["episode_id"] == "ep_000")
    assert entry["gold"]["metadata"]["quality"] == 2          # human edit landed
    assert entry["auto"]["metadata"]["quality"] == before     # VLM auto untouched
    assert s.state()["reviewed_count"] == 1


def test_http_round_trip(tmp_path: Path):
    s = _session(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(s))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import urllib.request

        base = f"http://127.0.0.1:{server.server_address[1]}"
        html = urllib.request.urlopen(base + "/", timeout=10).read().decode("utf-8")
        assert "robovid_conditioner" in html and "slider" in html  # the SPA with a scrubber
        state = json.loads(urllib.request.urlopen(base + "/api/state", timeout=10).read())
        assert state["episode_count"] == 2
        frame = urllib.request.urlopen(base + "/frame/ep_000/3", timeout=10).read()
        assert frame[:2] == b"\xff\xd8"
    finally:
        server.shutdown()
        server.server_close()
