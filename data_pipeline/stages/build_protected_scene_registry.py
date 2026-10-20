#!/usr/bin/env python3
"""Collect scene identities that training data must never contain.

The existing evaluation denylist hashes each benchmark's full media path, so a
training row that reaches the same physical scene through a different source
release (``scannet_videos_128f/train/scene0221_01_128f.mp4`` versus
``VSI-Bench/scannet/scene0221_01.mp4``) passes it unnoticed.  This stage emits
the scene tokens themselves so the identity survives re-encoding, clipping, and
per-dataset path conventions.

Only media/scene identifier fields are read.  Benchmark answers, options,
categories, predictions, and scores are never opened, and the registry selects
no training rows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from data_pipeline.pure_vlm_utils import builtin, sha256

FORMAT = "iclr_spatial.protected_scene_registry.v1"

# Source releases wrap the same scene in per-dataset packaging suffixes.  They
# carry no identity of their own and must be removed before comparison.
PACKAGING_SUFFIXES = (
    "_128f",
    "_compressed",
    "_full",
    "_frames",
    "_video",
    "_scene",
)


def scene_tokens(value: Any) -> set[str]:
    """Return comparable scene identities for one media or scan reference.

    A token is kept only when it is long enough and carries a digit.  Every
    scene identifier in the four benchmarks satisfies this (ScanNet
    ``scene0221_01``, ARKitScenes ``45261121``, ScanNet++ ``578511c8a9``,
    Matterport3D ``17DRP5sb8fy``, Ego4D UUIDs), while packaging words such as
    ``train`` or ``clips`` do not, which keeps the registry from blocking
    unrelated sources.
    """

    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return set()
    stem = raw.rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    candidates = {stem}
    changed = True
    while changed:
        changed = False
        for item in sorted(candidates):
            for suffix in PACKAGING_SUFFIXES:
                if len(item) > len(suffix) and item.casefold().endswith(suffix):
                    trimmed = item[: -len(suffix)]
                    if trimmed not in candidates:
                        candidates.add(trimmed)
                        changed = True
    return {
        item.casefold()
        for item in candidates
        if len(item) >= 6 and re.search(r"\d", item)
    }


def _rows(path: Path, fields: list[str]) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".parquet":
        available = set(pq.ParquetFile(path).schema_arrow.names)
        selected = [name for name in fields if name in available]
        if not selected:
            raise ValueError(f"none of {fields} exist in {path}")
        return [
            {str(key): builtin(value) for key, value in row.items()}
            for row in pq.read_table(path, columns=selected).to_pylist()
        ]
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON row list")
        records: Iterable[Mapping[str, Any]] = payload
    else:
        records = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    output = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{path} contains a non-object row")
        output.append({key: record.get(key) for key in fields})
    return output


def _specification(value: str) -> tuple[str, Path, list[str]]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("benchmark spec must be NAME=PATH=FIELD[,FIELD...]")
    fields = [item.strip() for item in parts[2].split(",") if item.strip()]
    if not fields:
        raise ValueError("at least one identifier field is required")
    return parts[0], Path(parts[1]), fields


def build(benchmarks: list[tuple[str, Path, list[str]]]) -> dict[str, Any]:
    if not benchmarks:
        raise ValueError("at least one benchmark is required")
    registry: dict[str, Any] = {}
    tokens: set[str] = set()
    for name, path, fields in benchmarks:
        if name in registry:
            raise ValueError(f"duplicate benchmark name: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = _rows(path, fields)
        local: set[str] = set()
        for row in rows:
            for field in fields:
                local |= scene_tokens(row.get(field))
        if not local:
            raise ValueError(f"{name} produced no scene tokens from {fields}")
        registry[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "rows": len(rows),
            "identifier_fields": fields,
            "unique_scene_tokens": len(local),
            "scene_tokens": sorted(local),
        }
        tokens |= local
    return {
        "format": FORMAT,
        "ok": True,
        "benchmarks": registry,
        "unique_scene_tokens": len(tokens),
        "scene_tokens": sorted(tokens),
        "answers_read": False,
        "categories_read": False,
        "options_read": False,
        "predictions_read": False,
        "scores_read": False,
        "training_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", action="append", default=[], required=True,
        metavar="NAME=PATH=FIELD[,FIELD...]",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build([_specification(value) for value in args.benchmark])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "unique_scene_tokens": report["unique_scene_tokens"],
        "benchmarks": {
            name: value["unique_scene_tokens"]
            for name, value in report["benchmarks"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
