"""Cost accounting from the per-call receipts and the sidecar.

``robolabel cost`` sums the per-episode cost estimates recorded in
``annotations.parquet`` and counts the raw receipts on disk. Cost estimates are
only as good as the provider's pricing table (OpenAI reports none, so its cost is
null); the raw token counts are always in the receipts for an exact audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .schema import read_annotations


def cost_summary(annotations_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(annotations_dir)
    df = read_annotations(out_dir)
    meta = df[df["record_type"] == "episode_metadata"]
    costs = pd.to_numeric(meta["cost_usd"], errors="coerce").dropna()
    episodes = int(df["episode_id"].nunique())
    receipts_root = (out_dir if out_dir.is_dir() else out_dir.parent) / "raw_receipts"
    receipt_count = sum(1 for _ in receipts_root.rglob("*.json")) if receipts_root.exists() else 0
    total = float(costs.sum()) if not costs.empty else None
    return {
        "episodes": episodes,
        "episodes_with_cost": int(len(costs)),
        "estimated_cost_usd_total": round(total, 6) if total is not None else None,
        "estimated_cost_usd_per_episode": round(total / episodes, 6) if total and episodes else None,
        "raw_receipt_files": receipt_count,
        "provider": str(meta["provider"].iloc[0]) if not meta.empty else None,
        "model": str(meta["model"].iloc[0]) if not meta.empty else None,
        "note": "Cost is an estimate from the provider pricing table; raw token counts are in raw_receipts/.",
    }
