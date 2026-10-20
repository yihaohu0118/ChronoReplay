#!/usr/bin/env python3
"""Exact Qwen processor-length gate over all rows or a high-cost SFT tail."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.pure_vlm_utils import builtin, canonical, sha256
from data_pipeline.processor_contract import configure_processor_pixel_budget
from synthesis.sft.media import encode_processor_item, normalize_messages


FORMAT = "iclr_spatial.sft_processor_length_tail_audit.v1"


def _optional_count(value: Any) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    return int(value)


def row_cost(row: dict[str, Any]) -> tuple[int, int, int, str]:
    uid = str(row.get("uid") or "")
    visual_units = max(
        _optional_count(row.get("num_images")),
        _optional_count(row.get("num_actor_frames")),
    )
    return (
        visual_units,
        _optional_count(row.get("num_turns")),
        len(canonical(row.get("messages") or []).encode()),
        hashlib.sha256(uid.encode()).hexdigest(),
    )


def select_probe_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 1 or len(rows) < count:
        raise ValueError("invalid processor probe size")
    # A global cost tail can entirely miss a smaller component when another
    # component has more than ``count`` equally image-heavy rows.  That is not
    # a safe train-ready gate: a newly added family may contain the only token
    # overflow even though its visual-unit count is lower.  Reserve the
    # highest-cost row from every source-defined component, then fill the
    # remaining probe with the global tail.  This remains independent of any
    # benchmark metadata, prediction, or score.
    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        component = str(row.get("component") or "")
        current = representatives.get(component)
        if current is None or row_cost(row) > row_cost(current):
            representatives[component] = row
    if len(representatives) > count:
        raise ValueError(
            "processor probe is smaller than the number of source components: "
            f"probe={count} components={len(representatives)}"
        )
    selected_by_uid = {
        str(row.get("uid") or ""): row for row in representatives.values()
    }
    for row in sorted(rows, key=row_cost, reverse=True):
        if len(selected_by_uid) >= count:
            break
        selected_by_uid.setdefault(str(row.get("uid") or ""), row)
    selected = list(selected_by_uid.values())
    if len(selected) != count:
        raise ValueError("processor probe could not select unique rows")
    return sorted(
        selected,
        key=lambda row: hashlib.sha256(
            f"processor-probe:{row.get('uid')}".encode()
        ).hexdigest(),
    )


def summarize_lengths(
    values: list[tuple[str, int]], *, max_length: int
) -> dict[str, Any]:
    if not values:
        raise ValueError("processor-length audit is empty")
    violations = [
        {"uid": uid, "tokens": length}
        for uid, length in values
        if length > max_length
    ]
    return {
        "ok": not violations,
        "rows": len(values),
        "minimum_tokens": min(length for _, length in values),
        "maximum_tokens": max(length for _, length in values),
        "max_length": max_length,
        "violations": violations[:20],
        "top_lengths": [
            {"uid": uid, "tokens": length}
            for uid, length in sorted(
                values, key=lambda item: (item[1], item[0]), reverse=True
            )[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-rows", type=int, default=64)
    parser.add_argument(
        "--all-rows", action="store_true",
        help="process every row instead of selecting a proxy high-cost tail",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument(
        "--include-uid",
        action="append",
        default=[],
        help="force a data-derived regression UID into the processor audit",
    )
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    if args.probe_rows < 1 or args.max_length < 1:
        raise ValueError("probe rows and max length must be positive")
    rows = [
        {str(key): builtin(value) for key, value in row.items()}
        for row in pd.read_parquet(args.train).to_dict("records")
    ]
    selected = rows if args.all_rows else select_probe_rows(
        rows, min(args.probe_rows, len(rows))
    )
    by_uid = {str(row.get("uid") or ""): row for row in rows}
    missing_forced = sorted(set(args.include_uid) - set(by_uid))
    if missing_forced:
        raise ValueError(f"forced processor UIDs are absent: {missing_forced}")
    selected_by_uid = {str(row.get("uid") or ""): row for row in selected}
    for uid in args.include_uid:
        selected_by_uid[uid] = by_uid[uid]
    selected = list(selected_by_uid.values())

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    processor_image_size = configure_processor_pixel_budget(
        processor, args.max_pixels
    )
    lengths: list[tuple[str, int]] = []
    for index, row in enumerate(selected, start=1):
        messages = normalize_messages(
            row.get("messages"), base_dir=args.train.resolve().parent
        )
        encoded = encode_processor_item(processor, messages)
        lengths.append((str(row.get("uid") or f"row_{index}"), int(
            encoded["input_ids"].numel()
        )))
        del encoded, messages
        if index % 8 == 0:
            gc.collect()
            print(f"[processor-length] {index}/{len(selected)}", flush=True)

    result = {
        "format": FORMAT,
        "selection": "all_rows" if args.all_rows else "deterministic_high_cost_tail",
        "selection_key": None if args.all_rows else (
            "component_max_then_global_tail:"
            "max(num_images,num_actor_frames),num_turns,"
            "serialized_message_bytes,uid_hash"
        ),
        "selection_reads_benchmark_metadata": False,
        "selection_reads_predictions": False,
        "train": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "source_rows": len(rows),
        "model_processor": str(args.model.resolve()),
        "max_pixels": args.max_pixels,
        "processor_image_size": processor_image_size,
        "forced_uids": sorted(set(args.include_uid)),
        "minimum_selected_proxy": list(min(map(row_cost, selected))[:3]),
        "maximum_selected_proxy": list(max(map(row_cost, selected))[:3]),
        **summarize_lengths(lengths, max_length=args.max_length),
    }
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
