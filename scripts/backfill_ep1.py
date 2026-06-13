"""One-off: backfill the single fresh-stacking episode (ep 1) that repeatedly hit
the 120s read timeout, using a longer per-call timeout. resume=True means only the
missing episode is (re-)annotated; the 19 already in the sidecar are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robolabel.adapters import build_source  # noqa: E402
from robolabel.annotate import annotate_source  # noqa: E402
from robolabel.providers.gemini import GeminiProvider  # noqa: E402
from robolabel.strategy import load_strategy  # noqa: E402

OUT = "fresh_stacking/grounded_flash_v3"


def main() -> int:
    source = build_source("lerobot", "lerobot/svla_so100_stacking",
                          camera_key="observation.images.top")
    provider = GeminiProvider(model="gemini-2.5-flash", timeout_seconds=300.0)
    strategy = load_strategy("S2")
    made = annotate_source(source, OUT, provider=provider, strategy=strategy,
                           extract_images=False, limit=20, resume=True,
                           progress=lambda i, n, e: print(f"[{i}/{n}] {e}", file=sys.stderr))
    print(f"backfilled {len(made)} episode(s): {[a.episode_id for a in made]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
