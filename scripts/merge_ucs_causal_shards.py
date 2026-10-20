#!/usr/bin/env python3
"""Merge disjoint causal-VLM UCS shards without changing predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORMAT = "iclr_spatial.ucs_causal_predictions.v1"


def merge(paths: list[Path], expected_rows: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    summaries = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results")
        if not isinstance(rows, list):
            raise ValueError(f"missing result list: {path}")
        results.extend(dict(row) for row in rows)
        summaries.append(dict(payload.get("summary") or {}))
    ids = [str(row.get("id") or "") for row in results]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("UCS shards contain empty or duplicate question IDs")
    if len(results) != expected_rows:
        raise ValueError(
            f"incomplete UCS shard coverage: {len(results)} != {expected_rows}"
        )
    correct = sum(int(row.get("correct") is True) for row in results)
    body = {
        "format": FORMAT,
        "full_coverage": True,
        "expected_total": expected_rows,
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "source_shards": [str(path.resolve()) for path in paths],
        "source_summaries": summaries,
        "results": sorted(results, key=lambda row: str(row["id"])),
    }
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_rows <= 0:
        raise SystemExit("expected rows must be positive")
    payload = merge(args.shard, args.expected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("total", "correct", "accuracy")}, sort_keys=True))


if __name__ == "__main__":
    main()
