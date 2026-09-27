"""The ``none`` event source: no candidates (video alone, SPEC_V1_1 3.1)."""

from __future__ import annotations

from typing import Any

from .base import EventSource


class NoneSource(EventSource):
    """Returns no events and reads nothing from the episode."""

    name = "none"
    version = "none-2026-09-27.1"

    def events(self, episode: Any, *, camera: str | None = None,
               l1: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return []
