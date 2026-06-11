"""LeRobot dataset adapter.

Reads a LeRobot dataset (Hugging Face hub id or local path) through the public
``LeRobotDataset`` API and yields :class:`~robolabel.episode.Episode` objects.

Written against **lerobot 0.4.x** (``lerobot.datasets.lerobot_dataset``). The
frame range of an episode comes from ``meta.episodes[ep_idx]`` (``dataset_from_index``
/ ``dataset_to_index``); a frame is fetched as ``dataset[abs_idx][camera_key]``.
``lerobot`` is an optional dependency (``pip install 'robolabel[lerobot]'``);
the import is deferred so this module loads without it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..episode import Episode, EpisodeSource


class LeRobotAdapter(EpisodeSource):
    name = "lerobot"

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        camera_key: str | None = None,
        episodes: list[int] | None = None,
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "The LeRobot adapter needs the 'lerobot' extra. Install it with "
                "`pip install 'robolabel[lerobot]'`."
            ) from exc

        self.repo_id = repo_id
        # download_videos=True is the default; videos are required to read frames.
        self.dataset = LeRobotDataset(repo_id, root=root, episodes=episodes)
        self.meta = self.dataset.meta
        cameras = list(self.meta.camera_keys)
        if not cameras:
            raise ValueError(f"Dataset {repo_id!r} has no image/video (camera) features.")
        if camera_key is not None and camera_key not in cameras:
            raise ValueError(f"camera_key {camera_key!r} not in dataset cameras {cameras}.")
        self.camera_key = camera_key or cameras[0]
        self._ep_indices = list(episodes) if episodes is not None else list(range(self.meta.total_episodes))

    def __len__(self) -> int:
        return len(self._ep_indices)

    def __iter__(self) -> Iterator[Episode]:
        fps = float(self.meta.fps)
        for ep_idx in self._ep_indices:
            ep = self.meta.episodes[ep_idx]
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            num_frames = max(0, end - start)
            task = self._episode_task(ep, start)

            def get_frame(i: int, _start: int = start, _end: int = end) -> np.ndarray:
                item = self.dataset[min(_start + i, _end - 1)]
                return _frame_to_numpy(item[self.camera_key])

            yield Episode(
                episode_id=str(ep_idx),
                num_frames=num_frames,
                fps=fps,
                task=task,
                get_frame=get_frame,
                camera_key=self.camera_key,
                extra={"repo_id": self.repo_id, "lerobot_episode_index": ep_idx},
            )

    def _episode_task(self, ep: object, start: int) -> str | None:
        # Prefer episode metadata; fall back to the decoded first frame's task string.
        for key in ("tasks", "task"):
            try:
                value = ep[key]  # type: ignore[index]
            except (KeyError, TypeError):
                value = None
            if value:
                return value[0] if isinstance(value, (list, tuple)) else str(value)
        try:
            return str(self.dataset[start].get("task")) or None
        except Exception:  # noqa: BLE001 - task is optional; never fail iteration over it
            return None


def _frame_to_numpy(frame: object) -> np.ndarray:
    """Convert a LeRobot camera frame (CHW float tensor) to an HWC array.

    The :class:`Episode` coerces to uint8 RGB; here we only get it into numpy.
    """
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    return np.asarray(frame)
