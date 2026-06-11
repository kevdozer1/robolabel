"""VLM provider abstraction.

A :class:`VLMProvider` answers a question about a small set of keyframes and
returns a :class:`ProviderResponse` carrying the parsed answer, a raw-response
receipt (written to disk for provenance), the provider/model identity, latency,
and an optional cost estimate. The base class also implements the two-stage
``observe -> label`` flow that every labeler uses.

Adding a provider is exactly one new file in this package that defines a
``VLMProvider`` subclass and calls :func:`register_provider`.
"""

from __future__ import annotations

import base64
import io
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class ProviderResponse:
    """One VLM call's result plus its provenance receipt."""

    answer: str
    raw: dict[str, Any]
    provider: str
    model: str
    elapsed_seconds: float
    estimated_cost_usd: float | None = None


@dataclass
class TwoStageResult:
    """Output of the observe -> label flow."""

    observe: ProviderResponse
    observations: Any
    label: ProviderResponse
    extra_calls: list[ProviderResponse] = field(default_factory=list)

    @property
    def all_calls(self) -> list[ProviderResponse]:
        return [self.observe, self.label, *self.extra_calls]


class VLMProvider(ABC):
    """Answer questions about keyframes. One concrete subclass per provider."""

    name: str = "vlm"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def ask(
        self,
        frames: list[np.ndarray],
        frame_labels: list[int],
        question: str,
        receipt_path: Path,
        *,
        frame_captions: list[str] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Send a contact sheet of ``frames`` and ``question``; write a receipt.

        ``frame_labels`` are the original frame indices, drawn on the sheet so
        the model can refer to them. Implementations must always write a receipt
        to ``receipt_path`` (even on error) and never include raw image bytes in
        it.

        ``frame_captions`` optionally overrides the per-thumbnail caption (the
        annotation-strategy layer uses this to print index + timestamp on each
        frame). ``temperature`` optionally overrides the provider's default
        decoding temperature (used by the self-consistency strategy to draw
        varied samples); ``None`` keeps the provider's deterministic default.
        """

    def observe_then_label(
        self,
        frames: list[np.ndarray],
        frame_labels: list[int],
        observe_question: str,
        build_label_question: Callable[[Any], str],
        observe_receipt: Path,
        label_receipt: Path,
        *,
        frame_captions: list[str] | None = None,
        temperature: float | None = None,
    ) -> TwoStageResult:
        """Two-stage flow: first elicit physical observations, then labels.

        Stage one asks only for visible physical evidence (objects, gripper
        state, motion). Stage two passes those observations back in and asks for
        the actual annotation. Grounding the label in a prior observation pass is
        what reduces hallucinated, ungrounded labels.
        """
        observed = self.ask(frames, frame_labels, observe_question, observe_receipt,
                            frame_captions=frame_captions)
        observations = try_extract_json(observed.answer)
        label_question = build_label_question(observations)
        labeled = self.ask(frames, frame_labels, label_question, label_receipt,
                           frame_captions=frame_captions, temperature=temperature)
        return TwoStageResult(observe=observed, observations=observations, label=labeled)


# --------------------------------------------------------------------------- #
# Registry — providers self-register so the CLI can build one by name.
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[VLMProvider]] = {}


def register_provider(name: str, cls: type[VLMProvider]) -> None:
    _REGISTRY[name.strip().lower()] = cls


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def build_provider(name: str | None = None, model: str | None = None) -> VLMProvider:
    """Construct a provider by name (defaults to ``$ROBOVID_PROVIDER`` or mock)."""
    # Import side-effect: ensure built-in providers have registered themselves.
    from . import gemini, mock, openai  # noqa: F401

    resolved = (name or os.environ.get("ROBOVID_PROVIDER") or "mock").strip().lower()
    if resolved not in _REGISTRY:
        raise ValueError(
            f"Unknown provider {resolved!r}. Available: {', '.join(available_providers())}. "
            "Set --provider or $ROBOVID_PROVIDER."
        )
    return _REGISTRY[resolved](model=model)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def make_contact_sheet(
    frames: list[np.ndarray],
    frame_labels: list[int],
    thumb_width: int = 224,
    captions: list[str] | None = None,
) -> Image.Image:
    """Tile keyframes into a labeled contact sheet for a single VLM call.

    By default each thumbnail is captioned ``frame {label}``. ``captions`` (one
    string per frame) overrides that — the strategy layer uses it to stamp the
    frame index *and* timestamp so the model can return boundaries as concrete
    frame indices.
    """
    images: list[Image.Image] = []
    for i, (frame, label) in enumerate(zip(frames, frame_labels, strict=False)):
        img = Image.fromarray(np.asarray(frame).astype("uint8")).convert("RGB")
        ratio = thumb_width / img.width
        img = img.resize((thumb_width, max(1, int(img.height * ratio))))
        canvas = Image.new("RGB", (img.width, img.height + 24), "white")
        canvas.paste(img, (0, 24))
        caption = captions[i] if captions is not None and i < len(captions) else f"frame {label}"
        ImageDraw.Draw(canvas).text((6, 4), caption, fill=(0, 0, 0))
        images.append(canvas)
    if not images:
        return Image.new("RGB", (thumb_width, thumb_width), "white")
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * images[0].width, rows * images[0].height), "white")
    for i, img in enumerate(images):
        sheet.paste(img, ((i % cols) * img.width, (i // cols) * img.height))
    return sheet


def image_to_data_url(image: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    """Persist a raw-response receipt (never include image bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class MissingCredentialError(RuntimeError):
    """Raised when a provider's API credential is not set; names the env var."""


def load_secret(env_vars: list[str], provider_label: str) -> str:
    """Return the first credential found in ``env_vars`` (then a local ``.env``).

    Raises :class:`MissingCredentialError` naming the exact variable to set. No
    OS-specific secret stores; Linux/macOS first.
    """
    for var in env_vars:
        value = _clean(os.environ.get(var))
        if value:
            return value
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            for var in env_vars:
                if line.strip().startswith(f"{var}="):
                    cleaned = _clean(line.split("=", 1)[1])
                    if cleaned:
                        return cleaned
    primary = env_vars[0]
    raise MissingCredentialError(
        f"{provider_label} credential not found. Set the {primary} environment variable "
        f"(e.g. `export {primary}=...`) or add a line `{primary}=...` to a local .env file."
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lstrip("﻿").strip().strip('"').strip("'")
    return cleaned or None


def try_extract_json(text: str) -> Any:
    """Best-effort parse of JSON possibly wrapped in prose or markdown fences."""
    try:
        return extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return None


def extract_json(text: str) -> Any:
    """Parse JSON from a model answer, tolerating ```json fences and prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [i for i in (cleaned.find("["), cleaned.find("{")) if i >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if end <= start:
            raise
        return json.loads(cleaned[start : end + 1])
