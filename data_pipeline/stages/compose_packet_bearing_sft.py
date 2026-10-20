#!/usr/bin/env python3
"""Compose the frozen 40339 packet with ARKit persistent-object bearing.

The parent stays in order. The patch is the source-verified egocentric
bearing task built from official ARKit extrinsics and aligned 3D boxes.
Benchmark questions, labels, categories, predictions, and scores are
never read.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.pure_vlm_utils import PARQUET_ENCODING, builtin, canonical, sha256


FORMAT = "iclr_spatial.packet_bearing_sft.v1"
READY_FORMAT = "iclr_spatial.packet_bearing_data_ready.v1"
PARENT_FORMATS = {
    "iclr_spatial.state_reader_writer_sft_pool.v1",
}
PATCH_FORMAT = "iclr_spatial.arkit_persistent_object_bearing_sft.v1"
PATCH_COMPONENTS = {
    "source_verified_persistent_object_bearing_evidence_bridge",
    "source_verified_persistent_object_bearing_global_context",
}


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        {str(key): builtin(value) for key, value in row.items()}
        for row in pd.read_parquet(path).to_dict("records")
    ]


def content_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(row.get("messages") or []).encode()).hexdigest()


def absolute_media(row: dict[str, Any], root: Path) -> dict[str, Any]:
    output = copy.deepcopy(row)
    for message in output.get("messages") or []:
        for item in message.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            raw = str(item.get("image") or "").removeprefix("file://")
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = (root / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            item["image"] = str(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-train", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--patch-train", type=Path, required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.parent_train,
        args.parent_report,
        args.patch_train,
        args.patch_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(path)

    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if (
        parent_report.get("ok") is not True
        or parent_report.get("format") not in PARENT_FORMATS
        or parent_report.get("release_starts_from_official_base") is not True
        or parent_report.get("train_sha256") != sha256(args.parent_train)
    ):
        raise ValueError("parent SFT contract is not passing")
    patch_rows = rows(args.patch_train)
    patch_report = json.loads(args.patch_report.read_text(encoding="utf-8"))
    logical_sources = int(patch_report.get("logical_sources") or -1)
    views = int(patch_report.get("views_per_source") or -1)
    if (
        patch_report.get("ok") is not True
        or patch_report.get("format") != PATCH_FORMAT
        or patch_report.get("source_gt_answers") is not True
        or patch_report.get("evaluation_results_read") is not False
        or patch_report.get("benchmark_fields_read") is not False
        or patch_report.get("benchmark_category_selection") is not False
        or patch_report.get("train_sha256") != sha256(args.patch_train)
        or int(patch_report.get("rows") or -1) != len(patch_rows)
        or logical_sources < 1
        or views != 2
        or logical_sources * views != len(patch_rows)
        or (patch_report.get("target_contracts") or {}).get("letter_only") != len(patch_rows)
    ):
        raise ValueError("persistent-object-bearing patch contract is not passing")

    parent_root = args.parent_train.resolve().parent
    patch_root = args.patch_train.resolve().parent
    parent_input = [absolute_media(row, parent_root) for row in rows(args.parent_train)]
    patch_input = [absolute_media(row, patch_root) for row in patch_rows]
    if any(str(row.get("component") or "") not in PATCH_COMPONENTS for row in patch_input):
        raise ValueError("patch contains a non-bearing component")
    if set(str(row.get("component") or "") for row in patch_input) != PATCH_COMPONENTS:
        raise ValueError("patch is missing a required bearing view")

    parent: list[dict[str, Any]] = []
    parent_seen: set[str] = set()
    parent_duplicates = 0
    for row in parent_input:
        key = content_key(row)
        if key in parent_seen:
            parent_duplicates += 1
            continue
        parent_seen.add(key)
        parent.append(row)

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for row in [*parent, *patch_input]:
        key = content_key(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        row["sample_loss_weight"] = 1.0
        output.append(row)
    if (
        len(output) < len(parent)
        or [content_key(row) for row in output[: len(parent)]]
        != [content_key(row) for row in parent]
    ):
        raise ValueError("unique frozen parent content was not preserved in order")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(output).to_parquet(
        args.output, index=False, **PARQUET_ENCODING
    )
    contracts = Counter(str(row.get("target_contract") or "") for row in output)
    components = Counter(str(row.get("component") or "") for row in output)
    report = {
        "format": FORMAT,
        "ok": True,
        "train": str(args.output.resolve()),
        "train_sha256": sha256(args.output),
        "rows": len(output),
        "parent_input_rows": len(parent_input),
        "parent_rows": len(parent),
        "parent_exact_content_duplicates_removed": parent_duplicates,
        "added_rows": len(output) - len(parent),
        "patch_rows": len(patch_input),
        "patch_report_sha256": sha256(args.patch_report),
        "exact_content_duplicates_removed": parent_duplicates + duplicates,
        "target_contracts": dict(sorted(contracts.items())),
        "components": dict(sorted(components.items())),
        "release_starts_from_official_base": True,
        "one_stage_training": True,
        "curriculum": False,
        "checkpoint_continuation": False,
        "agent_trajectories": False,
        "unit_row_weights": True,
        "sample_reweighting": False,
        "teacher_used_for_targets": False,
        "teacher_text": False,
        "distillation": False,
        "benchmark_inputs_read": False,
        "benchmark_predictions_or_scores_read": False,
        "source_gt_targets_preserved": True,
        "media_paths_absolute": True,
        "construction": "frozen_40339_plus_arkit_persistent_object_bearing",
        "promotion_status": "requires_matched_mechanism_and_cheap_score_gate",
        "used_for_training": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ready = {
        "format": READY_FORMAT,
        "ok": True,
        "train": str(args.output.resolve()),
        "report": str(args.report.resolve()),
        "train_sha256": report["train_sha256"],
        "rows": report["rows"],
        "added_rows": report["added_rows"],
    }
    (args.output.parent / "DATA_READY").write_text(
        json.dumps(ready, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(output),
                "parent_rows": len(parent),
                "added_rows": len(output) - len(parent),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
