"""Dataset adapters: turn a stored dataset into :class:`~robolabel.episode.Episode`s.

* :class:`~robolabel.adapters.lerobot.LeRobotAdapter` — primary (HF hub id or local).
* :class:`~robolabel.adapters.directory.DirectoryAdapter` — escape hatch (videos / frame dirs).
"""

from __future__ import annotations

from pathlib import Path

from ..episode import EpisodeSource
from .directory import DirectoryAdapter
from .lerobot import LeRobotAdapter

__all__ = ["LeRobotAdapter", "DirectoryAdapter", "build_source"]


def build_source(kind: str, target: str, **kwargs) -> EpisodeSource:
    """Construct an adapter by kind: ``lerobot`` or ``directory``."""
    k = kind.strip().lower()
    if k == "lerobot":
        return LeRobotAdapter(repo_id=target, **kwargs)
    if k == "directory":
        return DirectoryAdapter(root=Path(target), **kwargs)
    raise ValueError(f"Unknown source kind {kind!r}; expected 'lerobot' or 'directory'.")
