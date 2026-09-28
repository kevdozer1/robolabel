"""Frame selection and image parts for the model layers (V_LITE L2 to L4).

Images for models: long side at most 448 px (never upscaled), JPEG quality 85. Each image is preceded by
a text part such as ``frame 160 of 303 (5.33 s), camera up``. Camera labels are short: the part after
``observation.images.``, except LIBERO's ``image`` and ``image2``, which read ``scene`` and ``wrist``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..providers.base import ImagePart, TextPart

MODEL_MAX_SIDE = 448
MODEL_JPEG_QUALITY = 85
_LABELS = {"image": "scene", "image2": "wrist"}


def camera_label(camera: str) -> str:
    short = camera.split(".")[-1]
    return _LABELS.get(short, short)


def frame_line(frame: int, num_frames: int, fps: float, camera: str) -> str:
    return f"frame {int(frame)} of {int(num_frames)} ({int(frame) / float(fps):.2f} s), camera {camera_label(camera)}"


def model_jpeg(get_frame: Callable[[int], np.ndarray], frame: int) -> bytes:
    from ..adapters.lerobot_v3 import encode_jpeg

    return encode_jpeg(get_frame(int(frame)), quality=MODEL_JPEG_QUALITY, max_side=MODEL_MAX_SIDE)


def image_parts(episode: Any, items: list[tuple[int, str]]) -> tuple[list[Any], list[dict[str, Any]]]:
    """(parts, manifest) for (frame, camera) items: a text line then the image, in the given order."""
    cams = episode.extra["cameras"]
    parts: list[Any] = []
    manifest: list[dict[str, Any]] = []
    for frame, cam in items:
        parts.append(TextPart(frame_line(frame, episode.num_frames, episode.fps, cam)))
        parts.append(ImagePart(model_jpeg(cams[cam], frame), f"{cam}@{frame}"))
        manifest.append({"frame": int(frame), "camera": cam})
    return parts, manifest


def even_frames(n: int, k: int) -> list[int]:
    if n <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, min(k, n))})


def segment_frame_plan(num_frames: int, fps: float, l1: dict[str, Any], *, n_even: int = 20,
                       max_external: int = 28, max_wrist: int = 4) -> tuple[list[int], list[int]]:
    """(external camera frames, wrist frames) for the segment call (V_LITE L3).

    20 evenly spaced frames plus, for each L1 event, the frames 0.5 s before and after its onset; at most
    28 in total, sorted. Event pairs are added by priority (attempt closings and the openings of released
    attempts first, then other events in time order) until the budget is used. Wrist: 0.25 s after each
    closing offset of an attempt, at most 4.
    """
    last = num_frames - 1
    half = max(1, int(round(0.5 * fps)))
    q = max(1, int(round(0.25 * fps)))
    base = even_frames(num_frames, n_even)
    chosen = set(base)
    attempts = l1.get("attempts", [])
    key_onsets: list[int] = []
    for a in attempts:
        key_onsets.append(int(a["closing_onset"]))
        if a.get("opening_onset") is not None:
            key_onsets.append(int(a["opening_onset"]))
    others = [int(e["onset"]) for e in l1.get("events", []) if int(e["onset"]) not in key_onsets]
    for onset in key_onsets + sorted(others):
        pair = {max(0, onset - half), min(last, onset + half)}
        new = pair - chosen
        if len(chosen) + len(new) > max_external:
            continue
        chosen |= new
    wrist = []
    for a in attempts:
        f = min(last, int(a["closing_offset"]) + q)
        if f not in wrist:
            wrist.append(f)
    return sorted(chosen), sorted(wrist)[:max_wrist]
