"""Mock provider — offline, deterministic, and meaningless.

It exists for CI and the offline ``labelkit demo`` only. Its "labels" are
structural placeholders that always parse; they describe nothing about the
actual frames. Never use mock output as data, and never compute a reliability
number against it and believe it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .base import ProviderResponse, VLMProvider, register_provider, write_receipt


class MockProvider(VLMProvider):
    name = "mock"

    def __init__(self, model: str | None = None):
        super().__init__(model=model or "mock-vlm")

    def ask(
        self,
        frames: list[np.ndarray],
        frame_labels: list[int],
        question: str,
        receipt_path: Path,
    ) -> ProviderResponse:
        t0 = time.perf_counter()
        last = int(max(frame_labels)) if frame_labels else 0
        answer = _mock_answer(question, last)
        raw = {
            "provider": self.name,
            "model": self.model,
            "question": question,
            "frame_labels": list(frame_labels),
            "response_json": {"answer": answer},
            "warning": "MOCK OUTPUT — structurally valid, semantically meaningless.",
            "elapsed_seconds": 0.0,
        }
        write_receipt(receipt_path, raw)
        return ProviderResponse(answer, raw, self.name, self.model, time.perf_counter() - t0, 0.0)


def _mock_answer(question: str, last_frame: int) -> str:
    q = question.lower()
    if "observ" in q and "subtask" not in q and "quality" not in q:
        return json.dumps(
            {
                "observations": [
                    {"objects": ["object", "destination"], "gripper": "open then closed",
                     "motion": "arm moves toward object", "summary": "mock physical observation"}
                ],
                "episode_summary": "mock: robot appears to move an object toward a destination",
            }
        )
    if "subtask" in q or '"segments"' in q:
        thirds = _even_segments(last_frame, 3)
        return json.dumps(
            {
                "segments": [
                    {"segment_idx": i, "start_step": s, "end_step": e,
                     "subtask_text": txt}
                    for i, ((s, e), txt) in enumerate(
                        zip(thirds, ["approach and grasp object", "carry object to destination", "place object"])
                    )
                ]
            }
        )
    if "quality" in q or "mistake" in q:
        return json.dumps(
            {
                "task_success_quality": 4,
                "curation_quality": 4,
                "mistake": False,
                "boundary_clarity": "clear",
                "reason": "mock: placeholder reason, not derived from the frames",
            }
        )
    return json.dumps({"answer": "mock"})


def _even_segments(last_frame: int, n: int) -> list[tuple[int, int]]:
    last = max(last_frame, n - 1)
    edges = [round(i * last / n) for i in range(n + 1)]
    out = []
    for i in range(n):
        start = edges[i]
        end = max(start, edges[i + 1] - 1 if i < n - 1 else last)
        out.append((start, end))
    return out


register_provider("mock", MockProvider)
