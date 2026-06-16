"""Directory adapter — the escape hatch for non-LeRobot data.

Points at a folder of episodes, where each episode is either:

* a subdirectory of image frames (``*.png`` / ``*.jpg``, sorted by name), or
* a single ``.mp4`` video file.

An optional ``episodes.jsonl`` (one JSON object per line) supplies ``episode_id``,
``task``, and ``fps`` overrides, keyed by the file/dir stem::

    {"episode_id": "ep_000", "task": "put the cube in the box", "fps": 15}

Frame directories need only Pillow. Reading ``.mp4`` requires ``imageio`` with an
ffmpeg plugin (``pip install 'imageio[ffmpeg]'``); the import is deferred.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from ..episode import Episode, EpisodeSource

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


class DirectoryAdapter(EpisodeSource):
    name = "directory"

    def __init__(self, root: str | Path, fps: float = 10.0, jsonl: str | Path | None = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root}")
        self.default_fps = float(fps)
        self._meta = _load_jsonl(Path(jsonl)) if jsonl else _maybe_load_default_jsonl(self.root)
        self._episodes = self._discover()
        if not self._episodes:
            raise ValueError(
                f"No episodes found under {self.root}. Expected per-episode frame "
                "subdirectories or video files."
            )

    def _discover(self) -> list[tuple[str, Path, str]]:
        """Return ``(episode_id, path, kind)`` tuples sorted by id."""
        found: list[tuple[str, Path, str]] = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir():
                frames = _sorted_frames(child)
                if frames:
                    found.append((child.name, child, "frames"))
            elif child.suffix.lower() in _VIDEO_EXTS:
                found.append((child.stem, child, "video"))
        return found

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[Episode]:
        for episode_id, path, kind in self._episodes:
            meta = self._meta.get(episode_id, {})
            fps = float(meta.get("fps", self.default_fps))
            task = meta.get("task")
            if kind == "frames":
                yield self._frame_dir_episode(episode_id, path, fps, task)
            else:
                yield self._video_episode(episode_id, path, fps, task)

    def _frame_dir_episode(self, episode_id: str, path: Path, fps: float, task: str | None) -> Episode:
        frames = _sorted_frames(path)

        def get_frame(i: int) -> np.ndarray:
            return np.asarray(Image.open(frames[min(i, len(frames) - 1)]).convert("RGB"))

        return Episode(episode_id, len(frames), fps, task, get_frame, extra={"path": str(path)})

    def _video_episode(self, episode_id: str, path: Path, fps: float, task: str | None) -> Episode:
        reader = _VideoReader(path)

        def get_frame(i: int) -> np.ndarray:
            return reader.frame(i)

        return Episode(episode_id, reader.num_frames, fps, task, get_frame, extra={"path": str(path)})


def _sorted_frames(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


class _VideoReader:
    """Lazy mp4 reader backed by imageio; loads the whole clip on first access."""

    def __init__(self, path: Path):
        self.path = path
        self._frames: list[np.ndarray] | None = None

    def _load(self) -> list[np.ndarray]:
        if self._frames is None:
            try:
                import imageio.v3 as iio
            except ImportError as exc:
                raise RuntimeError(
                    f"Reading {self.path.name} needs imageio with a video backend. Install it with "
                    "`pip install 'imageio[ffmpeg]'`, or pre-extract frames into a folder."
                ) from exc
            # Try whichever video backend is installed: FFMPEG (imageio[ffmpeg]), then pyav, then auto.
            last: Exception | None = None
            for plugin in ("FFMPEG", "pyav", None):
                try:
                    kwargs = {"plugin": plugin} if plugin else {}
                    self._frames = [np.asarray(f) for f in iio.imiter(self.path, **kwargs)]
                    break
                except Exception as exc:  # try the next backend
                    last = exc
            if self._frames is None:
                raise RuntimeError(
                    f"Could not read {self.path.name} with any imageio video backend; install "
                    f"`imageio[ffmpeg]` or `imageio[pyav]`. Last error: {last}"
                ) from last
        return self._frames

    @property
    def num_frames(self) -> int:
        return len(self._load())

    def frame(self, i: int) -> np.ndarray:
        frames = self._load()
        return frames[min(i, len(frames) - 1)]


def _load_jsonl(path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    if not path.exists():
        return meta
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        key = str(record.get("episode_id"))
        if key:
            meta[key] = record
    return meta


def _maybe_load_default_jsonl(root: Path) -> dict[str, dict]:
    for name in ("episodes.jsonl", "tasks.jsonl"):
        candidate = root / name
        if candidate.exists():
            return _load_jsonl(candidate)
    return {}
