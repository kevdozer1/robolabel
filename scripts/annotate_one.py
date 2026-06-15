"""Annotate a single episode by index (for the presentation GIF). Reports the spend.

    python scripts/annotate_one.py lerobot/svla_so101_pickplace observation.images.side 7 S2 probe_pickplace/s2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robolabel.adapters.lerobot import LeRobotAdapter  # noqa: E402
from robolabel.annotate import annotate_episode  # noqa: E402
from robolabel.cost import cost_summary  # noqa: E402
from robolabel.providers.gemini import GeminiProvider  # noqa: E402
from robolabel.rubric import load_rubric  # noqa: E402
from robolabel.schema import write_annotations  # noqa: E402
from robolabel.strategy import load_strategy  # noqa: E402


def main() -> int:
    repo, cam, ep, strat, out = sys.argv[1:6]
    ep = int(ep)
    # Load the contiguous prefix 0..ep so the global frame indices line up, but annotate ONLY
    # the target episode (1 episode of API calls).
    source = LeRobotAdapter(repo, camera_key=cam, episodes=list(range(ep + 1)))
    target = next(e for e in source if e.episode_id == str(ep))
    provider = GeminiProvider(model="gemini-2.5-flash", timeout_seconds=180.0)
    ann = annotate_episode(target, provider, load_rubric(), Path(out),
                           extract_images=False, strategy=load_strategy(strat))
    write_annotations([ann], out)
    print("spend:", cost_summary(out).get("estimated_cost_usd_total"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
