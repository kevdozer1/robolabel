"""Regenerate the frozen tune/test split over a gold file's episodes.

The split is seeded, so this reproduces ``eval/so101_split.json`` byte-for-byte.
Strategies are tuned on ``tune`` only; ``test`` is scored once with the chosen
strategy (see ``scripts/eval_strategies.py`` and ``STRATEGY_REPORT.md``).

    python scripts/make_split.py --gold path/to/gold.json --out eval/so101_split.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20260607
N_TUNE = 30


def make_split(gold_path: str, out_path: str, *, seed: int = SEED, n_tune: int = N_TUNE) -> dict:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    ids = sorted((str(e["episode_id"]) for e in gold["episodes"]), key=_as_int)
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    tune = sorted(shuffled[:n_tune], key=_as_int)
    test = sorted(shuffled[n_tune:], key=_as_int)
    split = {
        "dataset": "lerobot/svla_so101_pickplace",
        "provider_of_gold": "human (50-episode full review)",
        "seed": seed,
        "n_total": len(ids),
        "n_tune": len(tune),
        "n_test": len(test),
        "note": (
            "Frozen split. Strategies are iterated on `tune` only; `test` is scored once "
            "with the chosen strategy. Regenerate identically with scripts/make_split.py."
        ),
        "tune": tune,
        "test": test,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split


def _as_int(x: str) -> int:
    try:
        return int(x)
    except ValueError:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="eval/so101_split.json")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-tune", type=int, default=N_TUNE)
    args = ap.parse_args()
    split = make_split(args.gold, args.out, seed=args.seed, n_tune=args.n_tune)
    print(f"wrote {args.out}: {split['n_tune']} tune / {split['n_test']} test (seed {split['seed']})")


if __name__ == "__main__":
    main()
