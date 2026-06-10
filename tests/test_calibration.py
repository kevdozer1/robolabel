from __future__ import annotations

from pathlib import Path

from labelkit.annotate import annotate_source
from labelkit.cost import cost_summary
from labelkit.demo import synthetic_source
from labelkit.gold import build_gold_template, load_or_sync_gold, update_episode_review
from labelkit.providers import build_provider
from labelkit.reliability import reliability_report
from labelkit.schema import export_jsonl


def _annotated(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    annotate_source(synthetic_source(3), out, provider=build_provider("mock"))
    return out


def test_gold_template_keeps_auto_and_empty_gold(tmp_path: Path):
    out = _annotated(tmp_path)
    template = build_gold_template(out)
    entry = template["episodes"][0]
    assert entry["auto"]["subtasks"]  # auto present
    assert entry["gold"]["metadata"]["quality"] is None  # gold empty
    # Human and VLM labels are in distinct blocks.
    assert "auto" in entry and "gold" in entry


def test_human_review_does_not_touch_auto(tmp_path: Path):
    out = _annotated(tmp_path)
    gold_path = tmp_path / "gold.json"
    load_or_sync_gold(out, gold_path)
    template = build_gold_template(out)
    ep_id = template["episodes"][0]["episode_id"]
    auto_quality_before = template["episodes"][0]["auto"]["metadata"]["quality"]

    update_episode_review(gold_path, ep_id, quality=1, mistake=True, reason="wrong object")

    import json
    gold = json.loads(gold_path.read_text())
    entry = next(e for e in gold["episodes"] if e["episode_id"] == ep_id)
    assert entry["gold"]["metadata"]["quality"] == 1
    assert entry["auto"]["metadata"]["quality"] == auto_quality_before  # auto untouched


def test_reliability_detects_quality_disagreement(tmp_path: Path):
    out = _annotated(tmp_path)
    gold_path = tmp_path / "gold.json"
    template = load_or_sync_gold(out, gold_path)
    # Human disagrees on quality for every episode (mock is always 4 -> set 2).
    for entry in template["episodes"]:
        update_episode_review(
            gold_path, entry["episode_id"], quality=2,
            gold_subtasks=[{"segment_idx": s["segment_idx"], "accept_auto": True} for s in entry["auto"]["subtasks"]],
            gold_subgoals=[{"segment_idx": s["segment_idx"], "frame_idx": s["frame_idx"]} for s in entry["auto"]["subgoals"]],
        )
    report = reliability_report(gold_path)
    assert report["reviewed_episode_count"] == 3
    assert report["quality_exact_agreement"] == 0.0  # 4 vs 2 everywhere
    assert report["subtask_boundary_temporal_iou_mean"] == 1.0  # boundaries accepted
    assert report["subgoal_frame_agreement"] == 1.0


def test_cost_and_export(tmp_path: Path):
    out = _annotated(tmp_path)
    summary = cost_summary(out)
    assert summary["episodes"] == 3
    assert summary["provider"] == "mock"
    jsonl = export_jsonl(out, tmp_path / "export.jsonl")
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
