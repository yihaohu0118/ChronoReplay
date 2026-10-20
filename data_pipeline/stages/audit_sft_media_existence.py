#!/usr/bin/env python3
"""Require every model-visible SFT image to exist at training time."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from data_pipeline.pure_vlm_utils import builtin, sha256


FORMAT = "iclr_spatial.sft_media_existence_audit.v1"


def image_paths(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    messages = builtin(row.get("messages") or [])
    for message in messages if isinstance(messages, list) else []:
        for part in builtin(message.get("content") or []):
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            path = str(part.get("image") or "")
            if not path:
                raise ValueError("image content block has no path")
            values.append(path)
    return values


def iter_rows(train: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    """Yield Python rows without materialising the full nested Parquet table."""

    parquet = pq.ParquetFile(train)
    available = set(parquet.schema_arrow.names)
    required = {"messages", "num_images"}
    missing = required - available
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    columns = [
        name for name in ("messages", "num_images", "uid", "component")
        if name in available
    ]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            yield {str(key): builtin(value) for key, value in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not args.train.is_file():
        raise FileNotFoundError(args.train)
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    paths: set[str] = set()
    references = 0
    rows = 0
    for row in iter_rows(args.train, args.batch_size):
        current = image_paths(row)
        if len(current) != int(row.get("num_images") or 0):
            raise ValueError(f"num_images disagrees with messages: {row.get('uid')}")
        paths.update(current)
        references += len(current)
        rows += 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        exists = dict(zip(paths, pool.map(os.path.isfile, paths), strict=True))
    missing = {path for path, ok in exists.items() if not ok}

    component_rows: Counter[str] = Counter()
    component_missing_rows: Counter[str] = Counter()
    component_image_references: Counter[str] = Counter()
    component_missing_references: Counter[str] = Counter()
    missing_prefixes: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    audited_rows = 0
    for row in iter_rows(args.train, args.batch_size):
        component = str(row.get("component") or "")
        current = image_paths(row)
        absent = [path for path in current if path in missing]
        component_rows[component] += 1
        component_image_references[component] += len(current)
        if absent:
            component_missing_rows[component] += 1
            component_missing_references[component] += len(absent)
            if len(examples[component]) < 5:
                examples[component].append(
                    {"uid": str(row.get("uid") or ""), "path": absent[0]}
                )
        for path in set(absent):
            parts = Path(path).parts
            missing_prefixes[str(Path(*parts[:7]))] += 1
        audited_rows += 1
    if audited_rows != rows:
        raise RuntimeError(f"Parquet row count changed during audit: {rows} -> {audited_rows}")

    result = {
        "format": FORMAT,
        "ok": not missing,
        "train": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "selection": "all_rows_all_model_visible_images",
        "rows": rows,
        "image_references": references,
        "unique_images": len(paths),
        "missing_unique_images": len(missing),
        "rows_with_missing_images": sum(component_missing_rows.values()),
        "missing_image_references": sum(component_missing_references.values()),
        "components": {
            component: {
                "rows": component_rows[component],
                "image_references": component_image_references[component],
                "rows_with_missing_images": component_missing_rows[component],
                "missing_image_references": component_missing_references[component],
            }
            for component in sorted(component_rows)
        },
        "missing_prefixes": dict(missing_prefixes.most_common(30)),
        "examples": dict(sorted(examples.items())),
        "benchmark_fields_read": False,
        "evaluation_results_read": False,
        "training_selection": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
