"""Local Qwen2.5-VL provider (optional extra ``labelkit[qwen]``).

Runs a Qwen2.5-VL checkpoint locally via transformers — no API key, but heavy
(GPU strongly recommended). The torch/transformers imports are deferred to
construction time so importing this module (and registering the provider) is
cheap and does not require the extra to be installed.

Default model ``Qwen/Qwen2.5-VL-7B-Instruct``; override with ``--model``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import ProviderResponse, VLMProvider, make_contact_sheet, register_provider, write_receipt

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenProvider(VLMProvider):
    name = "qwen"

    def __init__(self, model: str | None = None, max_new_tokens: int = 768):
        super().__init__(model=model or os.environ.get("LABELKIT_MODEL") or DEFAULT_MODEL)
        self.max_new_tokens = max_new_tokens
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The local Qwen provider needs the 'qwen' extra. Install it with "
                "`pip install 'labelkit[qwen]'` (torch + transformers + accelerate)."
            ) from exc
        self._torch = __import__("torch")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model, torch_dtype="auto", device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(self.model)

    def ask(
        self,
        frames: list[np.ndarray],
        frame_labels: list[int],
        question: str,
        receipt_path: Path,
    ) -> ProviderResponse:
        sheet: Image.Image = make_contact_sheet(frames, frame_labels)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        receipt: dict[str, Any] = {
            "provider": self.name, "model": self.model, "question": question,
            "frame_labels": list(frame_labels), "local": True,
        }
        t0 = time.perf_counter()
        try:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[sheet], return_tensors="pt").to(self._model.device)
            with self._torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            answer = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            elapsed = time.perf_counter() - t0
            receipt["elapsed_seconds"] = elapsed
            receipt["response_text"] = answer
            return ProviderResponse(answer, receipt, self.name, self.model, elapsed, 0.0)
        finally:
            write_receipt(receipt_path, receipt)


register_provider("qwen", QwenProvider)
