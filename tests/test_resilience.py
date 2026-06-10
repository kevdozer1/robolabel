"""A transient per-episode provider failure must not sink the whole run.

Regression test for the dogfood finding: a single Gemini 503 near the end of a
50-episode run raised and discarded every completed episode (the sidecar was
only written at the very end). annotate_source now records per-episode failures,
checkpoints after each episode, and resumes on re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

from robovid_conditioner.annotate import annotate_source
from robovid_conditioner.demo import synthetic_source
from robovid_conditioner.providers.base import ProviderResponse
from robovid_conditioner.providers.mock import MockProvider
from robovid_conditioner.schema import list_episode_ids, read_annotations


class _FlakyProvider(MockProvider):
    """Mock provider that raises on the Nth call (to fail one episode)."""

    def __init__(self, fail_on_call: int):
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = 0

    def ask(self, frames, frame_labels, question, receipt_path) -> ProviderResponse:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated 503: model overloaded")
        return super().ask(frames, frame_labels, question, receipt_path)


def test_one_failed_episode_does_not_lose_the_others(tmp_path: Path):
    out = tmp_path / "out"
    # 3 episodes x 4 calls each; fail the first call of episode index 1 (call 5).
    produced = annotate_source(synthetic_source(3), out, provider=_FlakyProvider(fail_on_call=5))

    ids = list_episode_ids(read_annotations(out))
    assert {"synthetic_000", "synthetic_002"}.issubset(set(ids))   # survivors written
    assert "synthetic_001" not in ids                              # the failed one is absent
    assert len(produced) == 2
    failures = json.loads((out / "failures.json").read_text(encoding="utf-8"))
    assert failures["count"] == 1
    assert failures["failed"][0]["episode_id"] == "synthetic_001"


def test_rerun_resumes_only_the_missing_episode(tmp_path: Path):
    out = tmp_path / "out"
    annotate_source(synthetic_source(3), out, provider=_FlakyProvider(fail_on_call=5))
    # Re-run with a healthy provider: episodes 0 and 2 are skipped (already in the
    # sidecar); only episode 1 is annotated.
    produced = annotate_source(synthetic_source(3), out, provider=MockProvider(), resume=True)
    assert [a.episode_id for a in produced] == ["synthetic_001"]
    assert set(list_episode_ids(read_annotations(out))) == {"synthetic_000", "synthetic_001", "synthetic_002"}


def test_resume_disabled_reannotates_all(tmp_path: Path):
    out = tmp_path / "out"
    annotate_source(synthetic_source(2), out, provider=MockProvider())
    produced = annotate_source(synthetic_source(2), out, provider=MockProvider(), resume=False)
    assert len(produced) == 2
