"""Validate OST prediction coverage and delegate metrics to a local official evaluator.

Set CHRONOREPLAY_OST_EVALUATOR to the upstream evaluation/OST_evaluator.py.
The upstream metric implementation is not redistributed by this package.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

FORMAT = "iclr_spatial.ost_bench_official_score.v1"


@lru_cache(maxsize=1)
def upstream():
    value = os.environ.get("CHRONOREPLAY_OST_EVALUATOR")
    if not value:
        raise RuntimeError("Set CHRONOREPLAY_OST_EVALUATOR to the trusted local official OST_evaluator.py; see docs/evaluation.md")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("chronoreplay_external_ost_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("Judgement_evalution", "Estimation_evaluation", "Enumeration_evalution", "process_answer", "collect_results"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Official evaluator lacks {name}")
    return module


def process_answer(raw: Any) -> str:
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return upstream().process_answer(str(raw))


def judgement_metric(pred: Any, gt: str, options: Iterable[str]) -> float:
    return float(upstream().Judgement_evalution(pred, gt, list(options)))


def estimation_metric(pred: Any, gt: Any) -> float:
    # Preserve the existing adapter's zero-target rule.
    try:
        if float(gt) == 0:
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    return float(upstream().Estimation_evaluation(pred, gt))


def enumeration_metric(pred: Any, gt: Any) -> float:
    return float(upstream().Enumeration_evalution(pred, gt))


def item_metric(sample: Mapping[str, Any]) -> float | None:
    kind = str(sample.get("type") or "")
    if kind == "None" or sample.get("response") is None:
        return None
    pred = process_answer(sample["response"])
    if "Estimation" in kind:
        return estimation_metric(pred, sample["answer"])
    if "Judgement" in kind:
        return judgement_metric(pred, sample["answer"], sample["option"])
    if "Counting" in kind or "Temporal-loc" in kind:
        return enumeration_metric(pred, sample["answer"])
    return None


def collect_results(static: Mapping[str, float]) -> tuple[dict, dict]:
    return upstream().collect_results(dict(static))


def _score(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    sums: dict[str, float] = {}
    for record in records:
        kind = record["type"]
        totals[kind] = totals.get(kind, 0) + 1
        sums[kind] = sums.get(kind, 0.0) + record["metric"]
    static = {kind: sums[kind] / totals[kind] for kind in totals}
    full, overall = collect_results(static)
    return {
        "scored_items": len(records),
        "per_type_accuracy": {key: round(value, 6) for key, value in static.items()},
        "per_type_items": totals,
        "composite": {key: round(value, 6) for key, value in full.items()},
        "official": {key: round(100 * value, 4) for key, value in overall.items()},
        "overall_accuracy": round(100 * overall["Overall"], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "iclr_spatial.scene_disjoint_eval_manifest.v1":
        raise SystemExit("not a scene-disjoint evaluation manifest")
    admissible = {str(item) for item in manifest["admissible_scene_ids"]}

    expected: dict[tuple[str, int], str] = {}
    for row in json.loads(args.annotation.read_text(encoding="utf-8")):
        expected[(str(row["scan_id"]), int(row["turn_id"]))] = str(row["type"])
    expected_disjoint = {key for key in expected if key[0] in admissible}

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    empty_responses = 0
    for path in sorted(args.predictions.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scan_id = str(payload["scan_id"])
        for turn in payload["user_message"]:
            key = (scan_id, int(turn["turn_id"]))
            if key in seen:
                raise SystemExit(f"duplicate prediction for {key}")
            seen.add(key)
            if key not in expected or expected[key] != str(turn["type"]):
                raise SystemExit(f"prediction {key} does not match the annotation")
            if not str(turn.get("response") or "").strip():
                empty_responses += 1
            metric = item_metric(turn)
            if metric is None:
                continue
            records.append(
                {
                    "scan_id": scan_id,
                    "turn_id": int(turn["turn_id"]),
                    "type": str(turn["type"]),
                    "metric": float(metric),
                    "scene_disjoint": scan_id in admissible,
                }
            )

    disjoint = [record for record in records if record["scene_disjoint"]]
    missing_disjoint = expected_disjoint - seen
    if missing_disjoint:
        raise SystemExit(
            f"{len(missing_disjoint)} scene-disjoint turns are unpredicted; "
            "the reported slice must have exact coverage"
        )
    report = {
        "format": FORMAT,
        "official_evaluator_sha256": hashlib.sha256(
            Path(os.environ["CHRONOREPLAY_OST_EVALUATOR"]).read_bytes()
        ).hexdigest(),
        "model_path": str(args.model_path.resolve()),
        "annotation": str(args.annotation.resolve()),
        "annotation_rows": len(expected),
        "predicted_turns": len(seen),
        "coverage": round(len(seen) / len(expected), 6) if expected else 0.0,
        "full_coverage": len(seen) == len(expected),
        "empty_responses": empty_responses,
        "scene_disjoint_manifest": str(args.manifest.resolve()),
        "scene_disjoint_admissible_scenes": len(admissible),
        "scene_disjoint_expected_turns": len(expected_disjoint),
        "scene_disjoint_full_coverage": True,
        # The reported number: scans absent from every training source.
        "scene_disjoint": _score(disjoint),
        # Retained for completeness; contaminated by shared public scans.
        "full": _score(records) if len(seen) == len(expected) else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scene_disjoint_overall": report["scene_disjoint"]["overall_accuracy"],
                "scene_disjoint_items": report["scene_disjoint"]["scored_items"],
                "full_overall": (report["full"] or {}).get("overall_accuracy"),
                "coverage": report["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
