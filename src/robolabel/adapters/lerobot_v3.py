"""LeRobot v3.0 folder adapter: reads every camera plus ``observation.state`` and ``action`` directly.

No ``lerobot`` install is needed: metadata comes from ``meta/`` (pyarrow), per-frame data from
``data/chunk-*/file-*.parquet`` filtered on ``episode_index``, and video frames from the packed
per-camera mp4 files through PyAV (the ``video`` extra). The legacy ``adapters/lerobot.py`` is
unchanged; this adapter is what the redesign's layers read.

v3.0 packs many episodes into one mp4 per camera. An episode is decoded from its
``from_timestamp`` to its ``to_timestamp``, and each decoded frame maps to the episode-relative index
``round((pts_seconds - from_timestamp) * fps)``. The decode report says how many frames mapped, which
indices were missing (filled with the previous frame) and how many were extra (dropped), so a caller
can apply its own rule when the count does not equal the episode length.

Decoded frames are kept as JPEG bytes (one list per episode and camera, never loose files) at a
bounded size, so a long three-camera episode stays in memory comfortably.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..episode import Episode, EpisodeSource

WRIST_HINTS = ("wrist", "image2", "hand", "gripper")


@dataclass
class DecodeReport:
    """What happened when one episode and camera was decoded."""

    camera: str
    expected: int
    mapped: int
    missing: list[int] = field(default_factory=list)
    extra: int = 0
    decoder: str = ""
    source_size: tuple[int, int] = (0, 0)  # width, height of the stored video
    stored_size: tuple[int, int] = (0, 0)  # width, height kept in memory

    @property
    def exact(self) -> bool:
        return not self.missing and self.mapped == self.expected

    def as_dict(self) -> dict[str, Any]:
        return {"camera": self.camera, "expected": self.expected, "mapped": self.mapped,
                "missing": list(self.missing), "n_missing": len(self.missing), "extra": self.extra,
                "decoder": self.decoder, "source_size": list(self.source_size),
                "stored_size": list(self.stored_size), "exact": self.exact}


def is_wrist_camera(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in WRIST_HINTS)


def resize_long_side(img, max_side: int | None):
    """Downscale a PIL image so its long side is at most ``max_side``; never upscale."""
    from PIL import Image

    if not max_side or max(img.size) <= max_side:
        return img
    scale = max_side / float(max(img.size))
    size = (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def encode_jpeg(arr: np.ndarray, quality: int = 85, max_side: int | None = None) -> bytes:
    """RGB uint8 array to JPEG bytes (deterministic for identical input and settings)."""
    from PIL import Image

    img = resize_long_side(Image.fromarray(np.asarray(arr, dtype=np.uint8)), max_side)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=False, progressive=False)
    return buf.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


class LeRobotV3Folder:
    """One LeRobot v3.0 dataset folder (the layout of the Hub repo)."""

    def __init__(self, root: str | Path, repo_id: str | None = None, revision: str | None = None):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        if str(self.info.get("codebase_version", "")).split(".")[0] != "v3":
            raise ValueError(f"{self.root}: codebase_version {self.info.get('codebase_version')!r} is not v3.x")
        self.repo_id = repo_id
        self.revision = revision
        self.fps = float(self.info["fps"])
        feats = self.info["features"]
        self.cameras = [k for k, v in feats.items() if v.get("dtype") == "video"]
        stats_path = self.root / "meta" / "stats.json"
        self.stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        self._episodes = self._read_episodes()
        self._tasks = self._read_tasks()
        self._data_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ metadata
    def _read_episodes(self):
        import pandas as pd

        files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{self.root}: no meta/episodes parquet files")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return df.set_index("episode_index", drop=False).sort_index()

    def _read_tasks(self) -> dict[int, str]:
        import pandas as pd

        path = self.root / "meta" / "tasks.parquet"
        if not path.is_file():
            return {}
        t = pd.read_parquet(path)
        if "task_index" in t.columns:
            return {int(i): str(name) for name, i in zip(t.index, t["task_index"], strict=True)}
        return {}

    def feature_names(self, key: str) -> list[str]:
        names = self.info["features"].get(key, {}).get("names")
        if isinstance(names, dict):  # some datasets nest names per axis
            names = next(iter(names.values()), None)
        shape = self.info["features"].get(key, {}).get("shape") or [0]
        if not isinstance(names, list) or len(names) != int(shape[0]):
            return [f"{key}[{i}]" for i in range(int(shape[0]))]
        return [str(n) for n in names]

    def episode_indices(self) -> list[int]:
        return [int(i) for i in self._episodes.index]

    def row(self, idx: int) -> dict[str, Any]:
        return self._episodes.loc[int(idx)].to_dict()

    def num_frames(self, idx: int) -> int:
        return int(self.row(idx)["length"])

    # ------------------------------------------------------------------ per-frame data
    def _data_file(self, idx: int) -> Path:
        r = self.row(idx)
        tmpl = self.info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        return self.root / tmpl.format(chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"]))

    def arrays(self, idx: int) -> dict[str, Any]:
        """``state``, ``action`` (float64 arrays), ``frame_index``, ``timestamp`` and ``task_index``."""
        import pandas as pd

        path = self._data_file(idx)
        key = str(path)
        if key not in self._data_cache:
            cols = ["episode_index", "frame_index", "timestamp", "task_index", "observation.state", "action"]
            self._data_cache = {key: pd.read_parquet(path, columns=cols)}  # one file cached at a time
        df = self._data_cache[key]
        ep = df[df["episode_index"] == int(idx)].sort_values("frame_index")
        if ep.empty:
            raise ValueError(f"episode {idx} has no rows in {path.name}")
        return {
            "state": np.stack(ep["observation.state"].to_numpy()).astype(np.float64),
            "action": np.stack(ep["action"].to_numpy()).astype(np.float64),
            "frame_index": ep["frame_index"].to_numpy().astype(np.int64),
            "timestamp": ep["timestamp"].to_numpy().astype(np.float64),
            "task_index": int(ep["task_index"].iloc[0]),
        }

    def task(self, idx: int) -> str | None:
        r = self.row(idx)
        tasks = r.get("tasks")
        if tasks is not None and len(tasks):
            return str(tasks[0])
        try:
            return self._tasks.get(self.arrays(idx)["task_index"])
        except (ValueError, FileNotFoundError):
            return None

    # ------------------------------------------------------------------ video
    def video_path(self, idx: int, camera: str) -> Path:
        r = self.row(idx)
        tmpl = self.info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        return self.root / tmpl.format(video_key=camera, chunk_index=int(r[f"videos/{camera}/chunk_index"]),
                                       file_index=int(r[f"videos/{camera}/file_index"]))

    def decode(self, idx: int, camera: str, *, max_side: int | None = 512,
               jpeg_quality: int = 92) -> tuple[list[bytes], DecodeReport]:
        """Decode one episode's frames for one camera as JPEG bytes, one per episode frame.

        Missing indices are filled with the previous frame (the next one at the start) and listed in
        the report; frames that map outside ``[0, n)`` or duplicate an index are dropped and counted.
        """
        import av

        r = self.row(idx)
        n = int(r["length"])
        t0 = float(r[f"videos/{camera}/from_timestamp"])
        t1 = float(r[f"videos/{camera}/to_timestamp"])
        slots: list[bytes | None] = [None] * n
        extra = 0
        report = DecodeReport(camera=camera, expected=n, mapped=0)
        with av.open(str(self.video_path(idx, camera))) as container:
            stream = container.streams.video[0]
            report.decoder = str(stream.codec_context.name)
            report.source_size = (int(stream.width), int(stream.height))
            tb = float(stream.time_base)
            if t0 > 0:
                container.seek(max(0, int((t0 - 1.0) / tb)), stream=stream, backward=True, any_frame=False)
            half = 0.5 / self.fps
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                ts = float(frame.pts) * tb
                if ts < t0 - half:
                    continue
                if ts >= t1 - half:
                    break
                k = int(round((ts - t0) * self.fps))
                if k < 0 or k >= n or slots[k] is not None:
                    extra += 1
                    continue
                img = resize_long_side(frame.to_image(), max_side)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=int(jpeg_quality))
                slots[k] = buf.getvalue()
                report.stored_size = img.size
        report.mapped = sum(1 for s in slots if s is not None)
        report.extra = extra
        report.missing = [i for i, s in enumerate(slots) if s is None]
        if report.mapped == 0:
            raise RuntimeError(f"episode {idx} camera {camera}: no frames decoded between {t0} and {t1}")
        first = next(s for s in slots if s is not None)
        prev = first
        out: list[bytes] = []
        for s in slots:
            prev = s if s is not None else prev
            out.append(prev)
        return out, report


class CameraFrames:
    """Random access to one decoded camera of one episode (JPEG bytes in memory, decoded on demand)."""

    def __init__(self, jpegs: Sequence[bytes], report: DecodeReport):
        self.jpegs = list(jpegs)
        self.report = report

    def __len__(self) -> int:
        return len(self.jpegs)

    def jpeg(self, i: int) -> bytes:
        return self.jpegs[int(min(max(i, 0), len(self.jpegs) - 1))]

    def frame(self, i: int) -> np.ndarray:
        return decode_jpeg(self.jpeg(i))


class LeRobotV3Source(EpisodeSource):
    """Yield :class:`Episode` objects for an explicit list of episode indices of one v3.0 folder.

    ``episode_id`` is the episode key ``"<family>/<index>"``. ``camera_key`` is the first external
    (non-wrist) camera; every camera is reachable through ``extra["cameras"]`` (name to frame getter).
    ``actions`` stays None unless ``with_actions`` is set, matching the legacy adapter; the arrays are
    in ``extra["state"]`` and ``extra["action"]``. Cameras decode lazily on first access.
    """

    name = "lerobot_v3"

    def __init__(self, folder: LeRobotV3Folder, episodes: Sequence[int], family: str, *,
                 max_side: int | None = 512, with_actions: bool = False):
        if episodes is None or isinstance(episodes, (int, range)) or not list(episodes):
            raise ValueError("LeRobotV3Source needs an explicit, non-empty list of episode indices")
        self.folder = folder
        self.family = family
        self.max_side = max_side
        self.with_actions = with_actions
        self._indices = [int(i) for i in episodes]
        self._decoded: dict[tuple[int, str], CameraFrames] = {}

    def __len__(self) -> int:
        return len(self._indices)

    def camera_frames(self, idx: int, camera: str) -> CameraFrames:
        key = (int(idx), camera)
        if key not in self._decoded:
            jpegs, report = self.folder.decode(idx, camera, max_side=self.max_side)
            self._decoded[key] = CameraFrames(jpegs, report)
        return self._decoded[key]

    def release(self, idx: int) -> None:
        """Drop the decoded frames of one episode (free memory between episodes)."""
        for key in [k for k in self._decoded if k[0] == int(idx)]:
            del self._decoded[key]

    def external_cameras(self) -> list[str]:
        return [c for c in self.folder.cameras if not is_wrist_camera(c)]

    def wrist_cameras(self) -> list[str]:
        return [c for c in self.folder.cameras if is_wrist_camera(c)]

    def episode(self, idx: int) -> Episode:
        f = self.folder
        arrays = f.arrays(idx)
        n = f.num_frames(idx)
        externals = self.external_cameras() or list(f.cameras)
        first = externals[0]

        def getter(camera: str):
            return lambda i, _c=camera: self.camera_frames(idx, _c).frame(i)

        cameras = {c: getter(c) for c in f.cameras}
        return Episode(
            episode_id=f"{self.family}/{idx}",
            num_frames=n,
            fps=f.fps,
            task=f.task(idx),
            get_frame=cameras[first],
            actions=arrays["action"] if self.with_actions else None,
            camera_key=first,
            extra={
                "family": self.family,
                "episode_index": int(idx),
                "cameras": cameras,
                "camera_order": list(f.cameras),
                "external_cameras": externals,
                "wrist_cameras": self.wrist_cameras(),
                "camera_sizes": {c: [int(f.info["features"][c]["shape"][1]), int(f.info["features"][c]["shape"][0])]
                                 for c in f.cameras},
                "state": arrays["state"],
                "action": arrays["action"],
                "feature_names": {"observation.state": f.feature_names("observation.state"),
                                  "action": f.feature_names("action")},
                "repo_id": f.repo_id,
                "revision": f.revision,
                "source": self,
            },
        )

    def __iter__(self) -> Iterator[Episode]:
        for idx in self._indices:
            yield self.episode(idx)
