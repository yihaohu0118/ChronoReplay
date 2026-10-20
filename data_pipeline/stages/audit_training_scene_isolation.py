#!/usr/bin/env python3
"""Fail closed when training sources reach a protected benchmark scene.

Rendered training pools address their frames by content hash, so scene identity
only survives in the raw source annotations a builder consumed.  This audit
therefore compares the source annotations against the protected scene registry
produced by ``build_protected_scene_registry``.

The registry is identity-only: this stage never reads benchmark answers,
options, categories, predictions, or scores, and it never selects training rows.
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

FORMAT = "iclr_spatial.training_scene_isolation.v1"
REGISTRY_FORMAT = "iclr_spatial.protected_scene_registry.v1"


def _registry(path: Path) -> tuple[set[str], dict[str, set[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format") != REGISTRY_FORMAT
        or payload.get("ok") is not True
        or payload.get("answers_read") is not False
        or payload.get("categories_read") is not False
        or payload.get("scores_read") is not False
    ):
        raise ValueError("protected scene registry is not identity-only")
    tokens = {str(item) for item in payload.get("scene_tokens") or []}
    if not tokens:
        raise ValueError("protected scene registry is empty")
    per_benchmark = {
        str(name): {str(item) for item in value.get("scene_tokens") or []}
        for name, value in (payload.get("benchmarks") or {}).items()
    }
    return tokens, per_benchmark


def audit(
    *,
    registry_path: Path,
    sources: list[tuple[str, Path, list[str]]],
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one training source is required")
    protected, per_benchmark = _registry(registry_path)
    failures: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for name, path, fields in sources:
        if name in reports:
            raise ValueError(f"duplicate source name: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _rows(path, fields)
        tokens: set[str] = set()
        for row in rows:
            for field in fields:
                tokens |= scene_tokens(row.get(field))
        if not tokens:
            raise ValueError(f"{name} produced no scene tokens from {fields}")
        overlap = tokens & protected
        attribution = {
            benchmark: sorted(overlap & values)[:20]
            for benchmark, values in per_benchmark.items()
            if overlap & values
        }
        if overlap:
            failures[f"{name}:protected_scene_overlap"] = {
                "count": len(overlap),
                "preview": sorted(overlap)[:20],
                "benchmarks": attribution,
            }
        reports[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(rows),
            "identifier_fields": fields,
            "unique_scene_tokens": len(tokens),
            "protected_scene_overlap": len(overlap),
            "protected_scene_overlap_preview": sorted(overlap)[:20],
            "protected_scene_overlap_by_benchmark": attribution,
        }
    return {
        "format": FORMAT,
        "ok": not failures,
        "failures": failures,
        "registry": {
            "path": str(registry_path.resolve()),
            "sha256": sha256(registry_path),
            "unique_scene_tokens": len(protected),
            "benchmarks": sorted(per_benchmark),
        },
        "sources": reports,
        "total_protected_scene_overlap": sum(
            value["protected_scene_overlap"] for value in reports.values()
        ),
        "benchmark_answers_read": False,
        "benchmark_category_fields_read": False,
        "model_outputs_read": False,
        "evaluation_scores_read": False,
        "training_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--source", action="append", default=[], required=True,
        metavar="NAME=PATH=FIELD[,FIELD...]",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        registry_path=args.registry,
        sources=[_specification(value) for value in args.source],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
