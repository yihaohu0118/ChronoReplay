#!/usr/bin/env python3
"""Freeze the scene-disjoint slice of a benchmark for a given training corpus.

Some benchmarks are built from the same public scans as our training sources
(OST-Bench draws ScanNet, Matterport3D, and ARKitScenes; VSI-Train-10k and the
VLM3R corrected-camera set draw ScanNet).  Reporting such a benchmark whole
would credit the model for scenes it trained on.

This stage emits the admissible item identifiers once, before any inference, so
the reported subset is fixed by scene identity rather than chosen after seeing
scores.  Benchmark answers, options, categories, predictions, and scores are
never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_pipeline.pure_vlm_utils import sha256
from data_pipeline.stages.build_protected_scene_registry import (
    _rows,
    _specification,
    scene_tokens,
)

FORMAT = "iclr_spatial.scene_disjoint_eval_manifest.v1"


def build(
    *,
    benchmark: tuple[str, Path, list[str]],
    training_sources: list[tuple[str, Path, list[str]]],
) -> dict[str, Any]:
    if not training_sources:
        raise ValueError("at least one training source is required")
    name, path, fields = benchmark
    if not path.is_file():
        raise FileNotFoundError(path)
    identifier = fields[0]

    trained: set[str] = set()
    source_reports: dict[str, Any] = {}
    for source_name, source_path, source_fields in training_sources:
        if source_name in source_reports:
            raise ValueError(f"duplicate training source: {source_name}")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        tokens: set[str] = set()
        for row in _rows(source_path, source_fields):
            for field in source_fields:
                tokens |= scene_tokens(row.get(field))
        if not tokens:
            raise ValueError(f"{source_name} produced no scene tokens")
        trained |= tokens
        source_reports[source_name] = {
            "path": str(source_path.resolve()),
            "sha256": sha256(source_path),
            "identifier_fields": source_fields,
            "unique_scene_tokens": len(tokens),
        }

    rows = _rows(path, [identifier])
    admissible: set[str] = set()
    excluded: set[str] = set()
    admissible_rows = excluded_rows = 0
    for row in rows:
        value = str(row.get(identifier) or "").strip()
        if not value:
            raise ValueError(f"{name} has a row without {identifier}")
        if scene_tokens(value) & trained:
            excluded.add(value)
            excluded_rows += 1
        else:
            admissible.add(value)
            admissible_rows += 1
    if not admissible_rows:
        raise ValueError(f"{name} has no scene-disjoint items left")

    return {
        "format": FORMAT,
        "ok": True,
        "benchmark": name,
        "benchmark_path": str(path.resolve()),
        "benchmark_sha256": sha256(path),
        "identifier_field": identifier,
        "total_rows": len(rows),
        "admissible_rows": admissible_rows,
        "excluded_rows": excluded_rows,
        "total_scenes": len(admissible | excluded),
        "admissible_scenes": len(admissible),
        "excluded_scenes": len(excluded),
        "admissible_scene_ids": sorted(admissible),
        "excluded_scene_ids": sorted(excluded),
        "training_sources": source_reports,
        "training_unique_scene_tokens": len(trained),
        "benchmark_answers_read": False,
        "benchmark_category_fields_read": False,
        "model_outputs_read": False,
        "evaluation_scores_read": False,
        "training_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", required=True, metavar="NAME=PATH=IDENTIFIER_FIELD",
    )
    parser.add_argument(
        "--training-source", action="append", default=[], required=True,
        metavar="NAME=PATH=FIELD[,FIELD...]",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        benchmark=_specification(args.benchmark),
        training_sources=[_specification(v) for v in args.training_source],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "benchmark", "total_rows", "admissible_rows", "excluded_rows",
                    "total_scenes", "admissible_scenes", "excluded_scenes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
