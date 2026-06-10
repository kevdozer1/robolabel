"""OpenAI provider (Responses API, multimodal).

Credential: ``OPENAI_API_KEY``. Default model ``gpt-4o``; override with
``--model`` or ``$ROBOVID_MODEL``. Token counts are stored raw in the receipt;
no dollar cost is asserted because per-model pricing changes frequently.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .base import (
    ProviderResponse,
    VLMProvider,
    image_to_data_url,
    load_secret,
    make_contact_sheet,
    register_provider,
    write_receipt,
)

RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(VLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, timeout_seconds: float = 120.0):
        super().__init__(model=model or os.environ.get("ROBOVID_MODEL") or DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds
        self.api_key = load_secret(["OPENAI_API_KEY"], "OpenAI")

    def ask(
        self,
        frames: list[np.ndarray],
        frame_labels: list[int],
        question: str,
        receipt_path: Path,
    ) -> ProviderResponse:
        image_url = image_to_data_url(make_contact_sheet(frames, frame_labels))
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": question},
                {"type": "input_image", "image_url": image_url, "detail": "high"},
            ]}],
        }
        receipt: dict[str, Any] = {
            "provider": self.name, "model": self.model, "endpoint": RESPONSES_URL,
            "question": question, "frame_labels": list(frame_labels),
            "request_image_note": "base64 omitted from receipt",
        }
        t0 = time.perf_counter()
        try:
            response = requests.post(
                RESPONSES_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload, timeout=self.timeout_seconds,
            )
            elapsed = time.perf_counter() - t0
            receipt["status_code"] = response.status_code
            receipt["elapsed_seconds"] = elapsed
            receipt["response_text"] = response.text
            response.raise_for_status()
            data = response.json()
            receipt["response_json"] = data
            return ProviderResponse(_extract_text(data), receipt, self.name, self.model, elapsed, None)
        finally:
            write_receipt(receipt_path, receipt)


def _extract_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    return "\n".join(chunks)


register_provider("openai", OpenAIProvider)
