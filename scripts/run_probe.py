"""Cross-task generalization probe (Phase B): grounded-Flash, closed-S2 vs open-vocab,
on the SAME pour/fold episodes. Gold-free. Hard spend guard at $5.50 (ceiling $6).

    python scripts/run_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robolabel.adapters import build_source  # noqa: E402
from robolabel.annotate import annotate_source  # noqa: E402
from robolabel.cost import cost_summary  # noqa: E402
from robolabel.providers.gemini import GeminiProvider  # noqa: E402
from robolabel.strategy import load_strategy  # noqa: E402

N_EPISODES = 8
SPEND_CEILING = 5.50  # abort before the $6 mission ceiling
TASKS = {
    "pour": ("Ishah8840/so101_pouring", "observation.images.front"),
    "fold": ("the-sam-uel/bi-so101-fold-horizontal-set-1", "observation.images.overhead"),
}
CONDITIONS = [("s2_closed", "S2"), ("s2_open", "S2-open")]


def total_spend() -> float:
    total = 0.0
    for task in TASKS:
        for cond, _ in CONDITIONS:
            d = Path(f"probe_{task}") / cond
            if (d / "annotations.parquet").exists():
                try:
                    total += float(cost_summary(d).get("estimated_cost_usd_total") or 0.0)
                except Exception:  # noqa: BLE001
                    pass
    return total


def main() -> int:
    provider = GeminiProvider(model="gemini-2.5-flash", timeout_seconds=180.0)
    for task, (repo, cam) in TASKS.items():
        source = build_source("lerobot", repo, camera_key=cam, episodes=list(range(N_EPISODES)))
        for cond, strat in CONDITIONS:
            out = f"probe_{task}/{cond}"
            spent = total_spend()
            if spent > SPEND_CEILING:
                print(f"STOP: spend ${spent:.2f} over ceiling ${SPEND_CEILING}; not running {task}/{cond}",
                      file=sys.stderr)
                return 2
            print(f"\n=== {task}/{cond} (strategy {strat}); spend so far ${spent:.3f} ===", file=sys.stderr)
            annotate_source(
                source, out, provider=provider, strategy=load_strategy(strat),
                extract_images=False, limit=N_EPISODES, resume=True,
                progress=lambda i, n, e: print(f"  [{i}/{n}] {e}", file=sys.stderr),
            )
    print(f"\nDONE. total estimated spend = ${total_spend():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
