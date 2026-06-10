"""``labelkit`` command-line interface.

    labelkit annotate    run the VLM labelers over a dataset -> annotations.parquet
    labelkit review      open the Streamlit calibration GUI (human labels)
    labelkit reliability VLM-vs-human agreement report from a gold file
    labelkit gate        automatic red-flag check on an annotation set
    labelkit export      consolidated JSONL view of the sidecar
    labelkit cost        cost / receipt accounting
    labelkit demo        offline end-to-end demo (mock provider, no API key)

Heavy imports are deferred into each handler so unrelated subcommands stay light.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="labelkit", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("annotate", help="Run the VLM labelers over a dataset.")
    p.add_argument("--source", choices=["lerobot", "directory"], required=True)
    p.add_argument("--target", required=True, help="LeRobot repo id / local path, or a directory.")
    p.add_argument("--out", required=True, help="Output directory for annotations.parquet + receipts.")
    p.add_argument("--provider", default=None, help="Provider name (mock|gemini|openai|qwen).")
    p.add_argument("--model", default=None)
    p.add_argument("--rubric", default=None, help="Path to a rubric.yaml (defaults to bundled).")
    p.add_argument("--limit", type=int, default=None, help="Annotate at most N episodes.")
    p.add_argument("--camera-key", default=None, help="LeRobot camera key (defaults to the first).")
    p.add_argument("--fps", type=float, default=10.0, help="Directory adapter default fps.")
    p.add_argument("--no-images", action="store_true", help="Do not extract subgoal frame images.")

    p = sub.add_parser("review", help="Open the Streamlit calibration GUI.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--source", choices=["lerobot", "directory"], default=None, help="To show clip frames.")
    p.add_argument("--target", default=None)
    p.add_argument("--port", type=int, default=8501)

    p = sub.add_parser("reliability", help="VLM-vs-human agreement from a gold file.")
    p.add_argument("--gold", required=True)
    p.add_argument("--json", default=None, help="Also write the full report JSON here.")

    p = sub.add_parser("gate", help="Automatic red-flag check on an annotation set.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--rubric", default=None)

    p = sub.add_parser("export", help="Consolidated JSONL view of the sidecar.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("cost", help="Cost / receipt accounting.")
    p.add_argument("--annotations", required=True)

    p = sub.add_parser("demo", help="Offline end-to-end demo (mock provider).")
    p.add_argument("--out", default="demo_out")
    p.add_argument("--episodes", type=int, default=3)

    args = parser.parse_args(argv)
    return _DISPATCH[args.command](args)


def _annotate(args) -> int:
    from .adapters import build_source
    from .annotate import annotate_source
    from .providers.base import MissingCredentialError, build_provider
    from .rubric import load_rubric

    kwargs: dict = {}
    if args.source == "lerobot" and args.camera_key:
        kwargs["camera_key"] = args.camera_key
    if args.source == "directory":
        kwargs["fps"] = args.fps
    source = build_source(args.source, args.target, **kwargs)
    try:
        provider = build_provider(args.provider, args.model)
    except MissingCredentialError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rubric = load_rubric(args.rubric)

    def progress(i: int, total: int, episode_id: str) -> None:
        print(f"[{i + 1}/{total}] {episode_id}", file=sys.stderr)

    annotations = annotate_source(
        source, args.out, provider=provider, rubric=rubric,
        extract_images=not args.no_images, limit=args.limit, progress=progress,
    )
    print(json.dumps({
        "out_dir": str(args.out),
        "annotations_parquet": str(Path(args.out) / "annotations.parquet"),
        "episodes": len(annotations),
        "provider": provider.name, "model": provider.model,
    }, indent=2))
    return 0


def _review(args) -> int:
    import subprocess

    app = Path(__file__).with_name("review_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port), "--",
           "--annotations", args.annotations, "--gold", args.gold]
    if args.source and args.target:
        cmd += ["--source", args.source, "--target", args.target]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("Streamlit is not installed. Install the review extra: pip install 'labelkit[review]'.",
              file=sys.stderr)
        return 2


def _reliability(args) -> int:
    from .reliability import format_report, reliability_report

    report = reliability_report(args.gold)
    print(format_report(report))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nFull report: {args.json}")
    return 0


def _gate(args) -> int:
    from .gate import run_gate
    from .rubric import load_rubric

    report = run_gate(args.annotations, rubric=load_rubric(args.rubric))
    print(report.to_text())
    return 0 if report.passed else 1


def _export(args) -> int:
    from .schema import export_jsonl

    out = export_jsonl(args.annotations, args.out)
    print(f"Exported to {out}")
    return 0


def _cost(args) -> int:
    from .cost import cost_summary

    print(json.dumps(cost_summary(args.annotations), indent=2))
    return 0


def _demo(args) -> int:
    from .demo import run_demo

    print(json.dumps(run_demo(args.out, n_episodes=args.episodes), indent=2))
    return 0


_DISPATCH = {
    "annotate": _annotate,
    "review": _review,
    "reliability": _reliability,
    "gate": _gate,
    "export": _export,
    "cost": _cost,
    "demo": _demo,
}


if __name__ == "__main__":
    raise SystemExit(main())
