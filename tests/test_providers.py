from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robolabel.providers import available_providers, build_provider
from robolabel.providers.base import MissingCredentialError, extract_json, load_secret


def _frames(n=4):
    return [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(n)]


def test_registry_has_builtin_providers():
    names = available_providers()
    for expected in ("mock", "gemini", "openai"):
        assert expected in names


def test_mock_provider_two_stage_returns_parseable_json(tmp_path: Path):
    provider = build_provider("mock")
    result = provider.observe_then_label(
        _frames(),
        [0, 1, 2, 3],
        "Observe the physical evidence.",
        lambda obs: 'Return the subtask "segments" as JSON.',
        tmp_path / "observe.json",
        tmp_path / "label.json",
    )
    assert (tmp_path / "observe.json").exists()
    assert (tmp_path / "label.json").exists()
    parsed = extract_json(result.label.answer)
    assert "segments" in parsed
    # Mock receipts carry the explicit meaninglessness warning.
    assert "MOCK" in result.label.raw["warning"]


def test_build_provider_unknown_name_lists_options():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_provider("not-a-provider")


def test_load_secret_names_exact_env_var(monkeypatch):
    monkeypatch.delenv("ROBOVID_FAKE_KEY", raising=False)
    monkeypatch.chdir("/")  # avoid picking up a stray .env
    with pytest.raises(MissingCredentialError, match="ROBOVID_FAKE_KEY"):
        load_secret(["ROBOVID_FAKE_KEY"], "Fake")


def test_load_secret_reads_env(monkeypatch):
    monkeypatch.setenv("ROBOVID_FAKE_KEY", "abc123")
    assert load_secret(["ROBOVID_FAKE_KEY"], "Fake") == "abc123"
