"""Google Gemini provider (generateContent REST API).

Credential: ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``). Default model
``gemini-2.5-flash``; override with ``--model`` or ``$ROBOVID_MODEL``.
"""

from __future__ import annotations

import json
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

GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"
_RETRY_STATUS = {429, 500, 502, 503, 504}

# Published per-million-token prices (USD). Used only for a rough cost estimate;
# token counts are always stored raw in the receipt.
_PRICES = {
    "flash-lite": {"input": 0.10, "output": 0.40},
    "2.5-flash": {"input": 0.30, "output": 2.50},
}


class GeminiProvider(VLMProvider):
    name = "gemini"

    def __init__(self, model: str | None = None, timeout_seconds: float = 120.0, max_retries: int = 8,
                 use_cache: bool = True):
        super().__init__(model=model or os.environ.get("ROBOVID_MODEL") or DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.api_key = load_secret(["GEMINI_API_KEY", "GOOGLE_API_KEY"], "Gemini")

    def ask(
        self,
        frames: list[np.ndarray],
        frame_labels: list[int],
        question: str,
        receipt_path: Path,
    ) -> ProviderResponse:
        cached = self._cached(receipt_path) if self.use_cache else None
        if cached is not None:
            return cached
        image_data = image_to_data_url(make_contact_sheet(frames, frame_labels)).split(",", 1)[1]
        endpoint = GENERATE_URL.format(model=self.model)
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": question},
                {"inlineData": {"mimeType": "image/jpeg", "data": image_data}},
            ]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        receipt: dict[str, Any] = {
            "provider": self.name, "model": self.model, "endpoint": endpoint,
            "question": question, "frame_labels": list(frame_labels),
            "request_image_note": "base64 omitted from receipt", "attempts": [],
        }
        t0 = time.perf_counter()
        try:
            response = None
            for attempt in range(self.max_retries + 1):
                response = requests.post(
                    endpoint,
                    headers={"Content-Type": "application/json", "X-Goog-Api-Key": self.api_key},
                    json=payload, timeout=self.timeout_seconds,
                )
                receipt["attempts"].append({"attempt": attempt + 1, "status_code": response.status_code})
                if response.status_code not in _RETRY_STATUS or attempt >= self.max_retries:
                    break
                time.sleep(min(2.0 ** attempt, 20.0))
            elapsed = time.perf_counter() - t0
            receipt["status_code"] = response.status_code
            receipt["elapsed_seconds"] = elapsed
            receipt["response_text"] = response.text
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    f"Gemini request failed for model {self.model}: HTTP {response.status_code}; "
                    f"{response.text[:300]}"
                )
            data = response.json()
            receipt["response_json"] = data
            return ProviderResponse(
                _extract_text(data), receipt, self.name, self.model, elapsed, _estimate_cost(data, self.model)
            )
        finally:
            write_receipt(receipt_path, receipt)


    def _cached(self, receipt_path: Path) -> ProviderResponse | None:
        """Reuse a prior successful receipt at this path (free resume).

        A receipt is reused only if it was a 200 for the same model and parses to
        non-empty text. This makes re-running after a partial/failed run skip the
        episodes that already succeeded instead of paying for them again.
        """
        if not receipt_path.exists():
            return None
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if receipt.get("model") != self.model or int(receipt.get("status_code") or 0) != 200:
            return None
        data = receipt.get("response_json")
        if not isinstance(data, dict):
            return None
        answer = _extract_text(data)
        if not answer.strip():
            return None
        receipt["cache_hit"] = True
        return ProviderResponse(answer, receipt, self.name, self.model, 0.0, _estimate_cost(data, self.model))


def _extract_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("text"):
                chunks.append(str(part["text"]))
    return "\n".join(chunks)


def _estimate_cost(data: dict[str, Any], model: str) -> float | None:
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    prices = None
    low = model.lower()
    if "flash-lite" in low:
        prices = _PRICES["flash-lite"]
    elif "2.5-flash" in low:
        prices = _PRICES["2.5-flash"]
    if prices is None:
        return None
    inp = int(usage.get("promptTokenCount") or 0)
    out = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
    return inp / 1_000_000 * prices["input"] + out / 1_000_000 * prices["output"]


register_provider("gemini", GeminiProvider)
