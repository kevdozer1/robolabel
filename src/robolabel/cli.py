"""``robolabel`` command-line interface.

    robolabel annotate    run the VLM labelers over a dataset -> annotations.parquet
    robolabel review      open the browser calibration GUI (human labels)
    robolabel reliability VLM-vs-human agreement report from a gold file
    robolabel gate        automatic red-flag check on an annotation set
    robolabel export      consolidated JSONL view of the sidecar
    robolabel cost        cost / receipt accounting
    robolabel demo        offline end-to-end demo (mock provider, no API key)

Heavy imports are deferred into each handler so unrelated subcommands stay light.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robolabel", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("annotate", help="Run the VLM labelers over a dataset.")
    p.add_argument("--source", choices=["lerobot", "directory"], required=True)
    p.add_argument("--target", required=True, help="LeRobot repo id / local path, or a directory.")
    p.add_argument("--out", required=True, help="Output directory for annotations.parquet + receipts.")
    p.add_argument("--provider", default=None, help="Provider name (mock|gemini|openai|qwen).")
    p.add_argument("--model", default=None)
    p.add_argument("--rubric", default=None, help="Path to a rubric.yaml (defaults to bundled).")
    p.add_argument("--strategy", default="S0",
                   help="Annotation strategy: S0 (baseline, default) .. S4, or a strategy JSON path.")
    p.add_argument("--limit", type=int, default=None, help="Annotate at most N episodes.")
    p.add_argument("--camera-key", default=None, help="LeRobot camera key (defaults to the first).")
    p.add_argument("--fps", type=float, default=10.0, help="Directory adapter default fps.")
    p.add_argument("--no-images", action="store_true", help="Do not extract subgoal frame images.")

    p = sub.add_parser("review", help="Open the browser calibration GUI (watch + scrub + correct).")
    p.add_argument("--annotations", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--source", choices=["lerobot", "directory"], default=None,
                   help="Show clip frames (scrubber). Without it you can still edit labels.")
    p.add_argument("--target", default=None)
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--no-browser", action="store_true")

    p = sub.add_parser("inspect", help="Verification viewer: multi-track timeline, evidence check, metrics.")
    p.add_argument("--data", required=True, help="inspect_data.json (scripts/build_inspect_data.py)")
    p.add_argument("--source", choices=["lerobot", "directory"], default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--camera-key", default=None)
    p.add_argument("--episodes", default=None,
                   help='limit the loaded source to these episodes, e.g. "0-7" (contiguous; '
                        "matches how the data was annotated — avoids downloading the whole dataset)")
    p.add_argument("--grades", default=None, help="(blind mode) JSON file to record grades into")
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--no-browser", action="store_true")

    p = sub.add_parser("query", help="Retrieve segments by phase -> contact sheet; or needs_review episodes.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--phase", default=None, help="Retrieve every segment with this phase (e.g. grasp).")
    p.add_argument("--needs-review", action="store_true", help="List gate needs_review episodes, worst first.")
    p.add_argument("--source", choices=["lerobot", "directory"], default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--camera-key", default=None)
    p.add_argument("--out", default=None, help="Write a contact-sheet PNG here (phase query).")
    p.add_argument("--limit", type=int, default=24)

    p = sub.add_parser("trial-report", help="Tally a blind-trial grades file into a markdown report.")
    p.add_argument("--grades", required=True)
    p.add_argument("--unblind", required=True, help="the *.unblind.json map (item_id->strategy) from build_inspect_data --blind")
    p.add_argument("--protocol", choices=["mark-failures-only", "mark-all"], default="mark-failures-only",
                   help="mark-failures-only (default): unmarked = pass over the known denominator.")
    p.add_argument("--out", default="FRESH_TRIAL_REPORT.md")

    p = sub.add_parser("reliability", help="VLM-vs-human agreement from a gold file.")
    p.add_argument("--gold", required=True)
    p.add_argument("--json", default=None, help="Also write the full report JSON here.")

    p = sub.add_parser("gate", help="Automatic red-flag check on an annotation set.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--rubric", default=None)

    p = sub.add_parser("export", help="Export the sidecar (JSONL, or the LeRobot subtask convention).")
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["jsonl", "lerobot"], default="jsonl",
                   help="jsonl = consolidated per-episode view; lerobot = meta/subtasks.parquet "
                        "+ per-episode subtask boundaries (the pinned-lerobot subtask convention).")
    p.add_argument("--subtask-field", default="subtask_text",
                   help="Which field becomes the LeRobot subtask string (subtask_text|phase).")

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
    from .strategy import load_strategy

    kwargs: dict = {}
    if args.source == "lerobot" and args.camera_key:
        kwargs["camera_key"] = args.camera_key
    if args.source == "lerobot" and args.limit:
        # Only fetch the episodes we will annotate (avoids downloading the whole
        # dataset's videos when --limit is small).
        kwargs["episodes"] = list(range(args.limit))
    if args.source == "directory":
        kwargs["fps"] = args.fps
    source = build_source(args.source, args.target, **kwargs)
    try:
        provider = build_provider(args.provider, args.model)
    except MissingCredentialError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rubric = load_rubric(args.rubric)
    strategy = load_strategy(args.strategy)

    def progress(i: int, total: int, episode_id: str) -> None:
        print(f"[{i + 1}/{total}] {episode_id}", file=sys.stderr)

    annotations = annotate_source(
        source, args.out, provider=provider, rubric=rubric,
        extract_images=not args.no_images, limit=args.limit, progress=progress,
        strategy=strategy,
    )
    print(json.dumps({
        "out_dir": str(args.out),
        "annotations_parquet": str(Path(args.out) / "annotations.parquet"),
        "episodes": len(annotations),
        "provider": provider.name, "model": provider.model,
        "strategy": strategy.name,
    }, indent=2))
    return 0


def _review(args) -> int:
    from .review_server import build_session, serve

    session = build_session(args.annotations, args.gold, args.source, args.target)
    serve(session, host="127.0.0.1", port=args.port, open_browser=not getattr(args, "no_browser", False))
    return 0


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
    if args.format == "lerobot":
        from .export_lerobot import export_lerobot_subtasks

        manifest = export_lerobot_subtasks(args.annotations, args.out, subtask_field=args.subtask_field)
        print(json.dumps(manifest, indent=2))
        return 0
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


def _inspect(args) -> int:
    from .inspect_server import build_session, serve

    session = build_session(args.data, args.source, args.target, args.grades, args.camera_key,
                            episodes=getattr(args, "episodes", None))
    serve(session, host="127.0.0.1", port=args.port, open_browser=not getattr(args, "no_browser", False))
    return 0


def _query(args) -> int:
    from .query import needs_review_episodes, phase_contact_sheet

    if args.needs_review:
        rows = needs_review_episodes(args.annotations)
        print(json.dumps(rows, indent=2))
        return 0
    if not args.phase:
        print("Pass --phase <name> or --needs-review.", file=sys.stderr)
        return 2
    source = None
    if args.source and args.target:
        from .adapters import build_source
        kwargs = {"camera_key": args.camera_key} if (args.source == "lerobot" and args.camera_key) else {}
        source = build_source(args.source, args.target, **kwargs)
    result = phase_contact_sheet(args.annotations, args.phase, source=source, out=args.out, limit=args.limit)
    print(json.dumps(result, indent=2))
    return 0


def _trial_report(args) -> int:
    import os

    from .trial_report import write_trial_report

    if not os.path.exists(args.grades):
        print(
            f"No grades file at {args.grades} yet — nothing has been graded.\n\n"
            "Grades are written only when you save items in the viewer's Grade tab, and only\n"
            "if the viewer was launched with --grades pointing at this file. Do this first:\n\n"
            f"  robolabel inspect --data fresh_stacking/blind.json --grades {args.grades} \\\n"
            "    --source lerobot --target lerobot/svla_so100_stacking --camera-key observation.images.top\n\n"
            "then open the 'Grade' tab, mark an item, and click 'Save grade & next' — that creates\n"
            f"{args.grades}. Re-run this command once you've graded at least one item.",
            file=sys.stderr,
        )
        return 2
    try:
        graded = json.loads(Path(args.grades).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        graded = {}
    if not graded:
        print(f"{args.grades} exists but is empty — grade at least one item in the viewer first.",
              file=sys.stderr)
        return 2
    out = write_trial_report(args.grades, args.unblind, args.out, protocol=args.protocol)
    print(f"wrote {out} ({len(graded)} item(s) graded, protocol={args.protocol})")
    return 0


_DISPATCH = {
    "annotate": _annotate,
    "review": _review,
    "inspect": _inspect,
    "query": _query,
    "trial-report": _trial_report,
    "reliability": _reliability,
    "gate": _gate,
    "export": _export,
    "cost": _cost,
    "demo": _demo,
}


if __name__ == "__main__":
    raise SystemExit(main())
