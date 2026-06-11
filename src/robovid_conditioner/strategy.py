"""Annotation strategies: the layer that controls *how* a VLM is asked to
segment an episode, sitting between the adapter (frames) and the provider (the
model call).

A :class:`StrategyConfig` controls four things:

1. **frame sampling** — how many frames are drawn into the contact sheet
   (``frame_count``) and at what thumbnail resolution (``resolution``);
2. **frame presentation** — whether each frame is captioned with its index *and*
   timestamp, and a textual frame manifest is included in the prompt
   (``caption_timestamps``);
3. **output schema** — whether boundaries must come back as frame indices with a
   per-boundary visual-evidence statement (``grounded``), constrained to a closed
   phase vocabulary (``closed_vocabulary``) with a minimum granularity
   (``enforce_min_segments``);
4. **post-passes** — a dense-window boundary-refinement pass
   (``refine_boundaries``) and self-consistency over ``self_consistency_k``
   samples with a per-boundary median.

The strategies are cumulative presets S0..S4. They are **off by default**: the
annotate pipeline uses S0 (the original baseline) unless a strategy is selected,
so S0 output is bit-for-bit reproducible. The resolved config is recorded in
``<out>/strategy.json`` for provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    """One annotation strategy. See module docstring for what each field controls."""

    name: str
    description: str
    # frame sampling
    frame_count: int = 6
    resolution: int = 224
    # frame presentation
    caption_timestamps: bool = False
    # output schema
    grounded: bool = False            # S1+: frame-index boundaries + per-boundary evidence
    closed_vocabulary: bool = False   # S2+: phase constrained to rubric vocabulary
    enforce_min_segments: bool = False  # S2+: apply a minimum-granularity floor
    # What to do when a grounded answer is below the floor. "warn" (default): accept it,
    # flag it as a single_segment_candidate (the human ep7 case — some episodes really are
    # one segment). "reject": re-prompt then fall back (the behavior the ablation was run
    # under; set this to reproduce STRATEGY_REPORT.md exactly).
    min_granularity_policy: str = "warn"
    max_label_attempts: int = 1       # re-prompts when grounded validation fails
    # post-passes
    refine_boundaries: bool = False   # S3+: dense-window per-boundary refinement
    refine_window: int = 15           # ±frames sent to the refinement call
    refine_max_frames: int = 25       # cap on frames per refinement contact sheet
    self_consistency_k: int = 1       # S4: number of label samples (median-combined)
    temperature: float = 0.4          # decoding temperature for k>1 samples

    @property
    def is_baseline(self) -> bool:
        """True for S0 — the original non-grounded path, left untouched."""
        return not self.grounded

    def provenance(self) -> dict[str, Any]:
        return {"strategy": asdict(self)}


# Cumulative presets. Each strategy adds exactly one capability to the previous.
PRESETS: dict[str, StrategyConfig] = {
    "S0": StrategyConfig(
        name="S0",
        description="baseline: evenly-spaced keyframes, free-text segments (current default)",
        frame_count=6,
    ),
    "S1": StrategyConfig(
        name="S1",
        description="frame-indexed grounding: boundaries as frame indices + per-boundary evidence",
        frame_count=12,
        caption_timestamps=True,
        grounded=True,
        max_label_attempts=2,
    ),
    "S2": StrategyConfig(
        name="S2",
        description="S1 + closed phase vocabulary + minimum granularity (reject single-segment)",
        frame_count=12,
        caption_timestamps=True,
        grounded=True,
        closed_vocabulary=True,
        enforce_min_segments=True,
        max_label_attempts=3,
    ),
    "S3": StrategyConfig(
        name="S3",
        description="S2 + dense-window boundary refinement pass",
        frame_count=12,
        caption_timestamps=True,
        grounded=True,
        closed_vocabulary=True,
        enforce_min_segments=True,
        max_label_attempts=3,
        refine_boundaries=True,
        refine_window=15,
    ),
    "S4": StrategyConfig(
        name="S4",
        description="S3 + self-consistency (k=3 samples, per-boundary median)",
        frame_count=12,
        caption_timestamps=True,
        grounded=True,
        closed_vocabulary=True,
        enforce_min_segments=True,
        max_label_attempts=3,
        refine_boundaries=True,
        refine_window=15,
        self_consistency_k=3,
        temperature=0.4,
    ),
}

DEFAULT_STRATEGY = "S0"


def available_strategies() -> list[str]:
    return list(PRESETS)


def load_strategy(name_or_path: str | StrategyConfig | None = None) -> StrategyConfig:
    """Resolve a strategy by preset name (``S0``..``S4``), a JSON file, or pass-through.

    A JSON file may set any :class:`StrategyConfig` field; unknown keys are
    ignored. It may also set ``"base": "S2"`` to start from a preset and override
    only some fields.
    """
    if name_or_path is None:
        return PRESETS[DEFAULT_STRATEGY]
    if isinstance(name_or_path, StrategyConfig):
        return name_or_path
    key = str(name_or_path).strip()
    if key.upper() in PRESETS:
        return PRESETS[key.upper()]
    path = Path(key)
    if path.exists():
        return _from_dict(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(
        f"Unknown strategy {name_or_path!r}. Use one of {', '.join(PRESETS)} or a path to a strategy JSON."
    )


def _from_dict(data: dict[str, Any]) -> StrategyConfig:
    base = PRESETS.get(str(data.get("base", "")).upper(), PRESETS[DEFAULT_STRATEGY])
    fields = {f for f in StrategyConfig.__dataclass_fields__}  # noqa: C416
    overrides = {k: v for k, v in data.items() if k in fields}
    overrides.setdefault("name", data.get("name", base.name))
    overrides.setdefault("description", data.get("description", base.description))
    return replace(base, **overrides)
