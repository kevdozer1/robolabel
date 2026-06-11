"""Mock provider — offline, deterministic, and meaningless.

It exists for CI and the offline ``robolabel demo`` only. Its "labels" are
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
        *,
        frame_captions: list[str] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        t0 = time.perf_counter()
        labels = [int(x) for x in frame_labels] if frame_labels else [0]
        last = max(labels)
        answer = _mock_answer(question, last, labels)
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


_MOCK_PHASES = ["approach", "grasp", "transport", "retract"]


def _mock_answer(question: str, last_frame: int, labels: list[int] | None = None) -> str:
    q = question.lower()
    # Markers below are unique substrings of each prompt's requested JSON shape, so
    # the mock answers the right shape without colliding (e.g. the frame manifest
    # text "exact frame index" must NOT be read as a refinement request).
    #
    # Boundary refinement (S3): the refine prompt is the only one asking for a
    # "single integer frame index"; return the centre of the dense window.
    if "single integer frame index" in q:
        window = labels or [last_frame]
        return json.dumps({"frame": int(window[len(window) // 2])})
    # Grounded segmentation (S1+): the only prompt asking for per-segment end_frame
    # + phase together. Checked before the "events" marker, because the stage-one
    # events list is embedded into this prompt as the observations block.
    if "end_frame" in q and "phase" in q:
        quarters = _even_segments(last_frame, 4)
        return json.dumps(
            {
                "segments": [
                    {"segment_idx": i, "start_frame": s, "end_frame": e,
                     "phase": _MOCK_PHASES[i], "subtask_text": f"{_MOCK_PHASES[i]} the object",
                     "evidence": f"mock evidence: {_MOCK_PHASES[i]} visible near frame {e}"}
                    for i, (s, e) in enumerate(quarters)
                ]
            }
        )
    # Grounded stage-one (S1+): asks for an "events" list, no end_frame/phase.
    if '"events"' in q:
        events = [{"frame": int(last_frame * f), "evidence": f"mock {p} event"}
                  for p, f in zip(_MOCK_PHASES, (0.25, 0.5, 0.75, 1.0), strict=False)]
        return json.dumps({"events": events, "objects": ["object", "destination"]})
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
                        zip(thirds, ["approach and grasp object", "carry object to destination", "place object"], strict=False)
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
