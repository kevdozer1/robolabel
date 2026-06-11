"""The adapter contract: an :class:`Episode` and an :class:`EpisodeSource`.

Every data adapter (LeRobot, a directory of videos, ...) yields ``Episode``
objects with a uniform interface so the labelers never learn the storage layout.
A frame is an ``(H, W, 3)`` uint8 RGB numpy array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    """One robot demonstration, exposed uniformly to the labelers.

    Args:
        episode_id: Stable identifier, unique within a source (used as the
            primary key in ``annotations.parquet``).
        num_frames: Number of frames in the episode.
        fps: Frames per second (used to convert frame indices to timestamps).
        task: Natural-language task string if the dataset records one, else None.
        get_frame: Callable returning the RGB uint8 frame at an index.
        actions: Optional ``(num_frames, action_dim)`` array. When present and
            the last channel looks like a gripper, it is used to bias keyframe
            selection toward action transitions; never required.
        camera_key: Optional name of the camera/video stream this came from.
    """

    episode_id: str
    num_frames: int
    fps: float
    task: str | None
    get_frame: Callable[[int], np.ndarray]
    actions: np.ndarray | None = None
    camera_key: str | None = None
    extra: dict = field(default_factory=dict)

    def frame(self, idx: int) -> np.ndarray:
        """Return the RGB uint8 frame at ``idx`` (clamped to range)."""
        i = int(min(max(idx, 0), self.num_frames - 1))
        arr = np.asarray(self.get_frame(i))
        return _as_rgb_uint8(arr)

    def frames(self, indices: Sequence[int]) -> list[np.ndarray]:
        """Return frames at the given indices."""
        return [self.frame(i) for i in indices]


class EpisodeSource(ABC):
    """A dataset adapter: an iterable of :class:`Episode` objects.

    Implementations live in :mod:`robolabel.adapters`. ``name`` identifies the
    source format for provenance.
    """

    name: str = "episode_source"

    @abstractmethod
    def __iter__(self) -> Iterator[Episode]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    def episode_ids(self) -> list[str]:
        return [ep.episode_id for ep in self]


def _as_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce a frame to ``(H, W, 3)`` uint8 RGB."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and float(a.max(initial=0.0)) <= 1.0:
            a = (a * 255.0).round()
        a = a.clip(0, 255).astype(np.uint8)
    if a.ndim == 2:  # grayscale
        a = np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
    if a.ndim == 3 and a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    return np.ascontiguousarray(a)
