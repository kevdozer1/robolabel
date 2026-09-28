"""Clip folders (robolabel v1.1): one short video per folder, read as one episode with one camera.

Layout::

    <root>/<clip id>/clip.mp4        the clip (required)
    <root>/<clip id>/task.txt        the task text (optional; its first non-empty line)
    <root>/<clip id>/source.json     provenance (optional; only its "task" string is read, when there
                                     is no task.txt)

Every other file in the folder (for example evaluation truth) is never read here, so nothing from it
can reach a prompt. Each clip becomes one :class:`~robolabel.episode.Episode` with key ``C/<clip id>``
(or a key the caller maps it to, for a clip cut from a benchmark episode, such as ``F3/1821``), one camera
named ``video``, and the frame rate, size and frame count of the file itself.

Decode only: frames are decoded with PyAV (the optional ``video`` extra) in presentation order, frame k
being the k-th decoded frame, downscaled to a long side of at most ``max_side`` px and kept as JPEG bytes
in memory (as the LeRobot v3 reader does). The frame count is the number of video packets, read without
decoding; a decode that yields fewer frames repeats the last one and says so in the decode report. The
package never encodes video: :meth:`ClipFolderSource.video_part` hands out the file's own bytes for a
native-video coarse call when the file is an H.264 mp4 of at most 60 s, and None otherwise (the caller
then encodes one itself).
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..episode import Episode, EpisodeSource

CLIP_FILE = "clip.mp4"
TASK_FILE = "task.txt"
SOURCE_FILE = "source.json"
CAMERA = "video"
CLIP_FAMILY = "C"
VIDEO_PART_MAX_S = 60.0
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def clip_key(clip_id: str) -> str:
    """The episode key of a clip: ``C/<clip id>``."""
    return f"{CLIP_FAMILY}/{clip_id}"


def _need_av():
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("reading clip folders needs PyAV: pip install 'robolabel[video]' (decode only)") from exc
    return av


def probe_clip(path: str | Path) -> dict[str, Any]:
    """Frame rate, size, frame count (video packets, no decode), duration and codec of an mp4 file."""
    av = _need_av()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = float(rate) if rate else 0.0
        n = sum(1 for packet in container.demux(stream) if packet.size > 0)
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration) / 1_000_000.0
        else:
            duration = n / fps if fps > 0 else 0.0
        return {"fps": fps, "width": int(stream.width), "height": int(stream.height), "num_frames": int(n),
                "duration_s": round(duration, 6), "codec": str(stream.codec_context.name),
                "format": str(container.format.name)}


class ClipFrames:
    """Lazy random access to the frames of one clip file (decoded once, on first use, as JPEG bytes)."""

    def __init__(self, path: Path, num_frames: int, *, max_side: int | None = 512, jpeg_quality: int = 92):
        self.path = Path(path)
        self.num_frames = int(num_frames)
        self.max_side = max_side
        self.jpeg_quality = int(jpeg_quality)
        self._jpegs: list[bytes] | None = None
        self.report: dict[str, Any] = {}

    def _load(self) -> list[bytes]:
        if self._jpegs is None:
            from .lerobot_v3 import resize_long_side

            av = _need_av()
            kept: list[bytes] = []
            extra = 0
            with av.open(str(self.path)) as container:
                stream = container.streams.video[0]
                decoder = str(stream.codec_context.name)
                size = None
                for frame in container.decode(stream):
                    if len(kept) >= self.num_frames:
                        extra += 1  # more frames than packets: counted, not kept
                        continue
                    img = resize_long_side(frame.to_image(), self.max_side)
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=self.jpeg_quality)
                    kept.append(buf.getvalue())
                    size = img.size
            if not kept:
                raise RuntimeError(f"{self.path.name}: no frame decoded")
            decoded = len(kept) + extra
            missing = self.num_frames - len(kept)
            kept += [kept[-1]] * missing
            self.report = {"camera": CAMERA, "expected": self.num_frames, "decoded": decoded, "missing": missing,
                           "extra": extra, "decoder": decoder, "stored_size": list(size) if size else None,
                           "exact": decoded == self.num_frames}
            self._jpegs = kept
        return self._jpegs

    def jpeg(self, i: int) -> bytes:
        jpegs = self._load()
        return jpegs[int(min(max(int(i), 0), len(jpegs) - 1))]

    def frame(self, i: int):
        from .lerobot_v3 import decode_jpeg

        return decode_jpeg(self.jpeg(i))

    def release(self) -> None:
        self._jpegs = None


def _read_task(folder: Path) -> str | None:
    task_file = folder / TASK_FILE
    if task_file.is_file():
        for line in task_file.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                return line.strip()
        return None
    source = folder / SOURCE_FILE
    if source.is_file():
        try:
            doc = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return None
        task = doc.get("task") if isinstance(doc, dict) else None
        if isinstance(task, str) and task.strip():
            return task.strip()
    return None


class ClipFolderSource(EpisodeSource):
    """Clip folders under ``root`` as episodes (see the module text).

    ``clip_ids`` picks clips and their order (default: every folder with a ``clip.mp4``, sorted by id).
    ``keys`` maps a clip id to another episode key (for example ``{"robot_f3_1821": "F3/1821"}``); every
    other clip gets ``C/<clip id>``. Construction reads no video; :meth:`episode` probes the file (no
    decode) and the frames are decoded on first access.
    """

    name = "clip_folder"

    def __init__(self, root: str | Path, clip_ids: Sequence[str] | None = None, *,
                 keys: Mapping[str, str] | None = None, max_side: int | None = 512, jpeg_quality: int = 92):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"clip folder root not found: {self.root.name}")
        if clip_ids is None:
            ids = sorted(p.name for p in self.root.iterdir() if (p / CLIP_FILE).is_file())
        else:
            if isinstance(clip_ids, (str, bytes)):
                raise ValueError("clip_ids must be a list of clip ids, not a single string")
            ids = [str(c) for c in clip_ids]
        for cid in ids:
            if not _ID_RE.match(cid):
                raise ValueError(f"malformed clip id {cid!r}: letters, digits, '_', '-' and '.' only")
        if len(set(ids)) != len(ids):
            raise ValueError("clip_ids repeats a clip")
        self.clip_ids = ids
        self.keys = {str(k): str(v) for k, v in (keys or {}).items()}
        self.max_side = max_side
        self.jpeg_quality = int(jpeg_quality)
        self._frames: dict[str, ClipFrames] = {}

    def __len__(self) -> int:
        return len(self.clip_ids)

    def folder(self, clip_id: str) -> Path:
        return self.root / str(clip_id)

    def clip_path(self, clip_id: str) -> Path:
        return self.folder(clip_id) / CLIP_FILE

    def ready(self, clip_id: str) -> bool:
        """True when the clip's ``clip.mp4`` exists."""
        return self.clip_path(clip_id).is_file()

    def key(self, clip_id: str) -> str:
        return self.keys.get(str(clip_id), clip_key(str(clip_id)))

    def episode_ids(self) -> list[str]:
        """The episode keys, without reading any video."""
        return [self.key(c) for c in self.clip_ids]

    def task(self, clip_id: str) -> str | None:
        return _read_task(self.folder(clip_id))

    def episode(self, clip_id: str) -> Episode:
        cid = str(clip_id)
        path = self.clip_path(cid)
        if not path.is_file():
            raise FileNotFoundError(f"clip {cid!r} has no {CLIP_FILE}")
        info = probe_clip(path)
        if info["num_frames"] < 1 or info["fps"] <= 0:
            raise ValueError(f"clip {cid!r}: no video frames or no frame rate in {CLIP_FILE}")
        frames = ClipFrames(path, info["num_frames"], max_side=self.max_side, jpeg_quality=self.jpeg_quality)
        self._frames[cid] = frames
        key = self.key(cid)
        getter = frames.frame
        return Episode(
            episode_id=key, num_frames=info["num_frames"], fps=info["fps"], task=self.task(cid), get_frame=getter,
            camera_key=CAMERA,
            extra={"family": key.split("/", 1)[0], "clip_id": cid, "cameras": {CAMERA: getter},
                   "camera_order": [CAMERA], "external_cameras": [CAMERA], "wrist_cameras": [],
                   "camera_sizes": {CAMERA: [info["width"], info["height"]]}, "video_info": info,
                   "clip_frames": frames, "source": self},
        )

    def video_part(self, clip_id: str, *, max_seconds: float = VIDEO_PART_MAX_S):
        """A :class:`~robolabel.providers.base.VideoPart` with the clip file's own bytes when the file is an
        H.264 mp4 of at most ``max_seconds``; None otherwise (the caller then supplies an encoded clip)."""
        from ..providers.base import VideoPart

        path = self.clip_path(clip_id)
        info = probe_clip(path)
        if info["codec"] != "h264" or "mp4" not in info["format"].split(","):
            return None
        if not 0 < info["duration_s"] <= float(max_seconds):
            return None
        return VideoPart(data=path.read_bytes(), mime="video/mp4", label=f"{self.key(clip_id)} {CLIP_FILE}",
                         seconds=float(info["duration_s"]))

    def release(self, clip_id: str) -> None:
        """Drop a clip's decoded frames from memory."""
        frames = self._frames.pop(str(clip_id), None)
        if frames is not None:
            frames.release()

    def __iter__(self) -> Iterator[Episode]:
        for cid in self.clip_ids:
            yield self.episode(cid)
