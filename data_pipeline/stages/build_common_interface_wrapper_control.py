#!/usr/bin/env python3
"""Build an information-preserving common-answer-interface control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.pure_vlm_utils import PARQUET_ENCODING, builtin, sha256


FORMAT = "iclr_spatial.common_interface_wrapper_control.v1"
PARENT_FORMAT = "iclr_spatial.ost_official_train_release.v2"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def wrapped(value: str) -> str:
    return canonical({"value": str(value)})


def transform_messages(raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    messages = builtin(raw)
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    output = copy.deepcopy(messages)
    original_targets: list[str] = []
    for message in output:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError("assistant content must be a list")
        text_items = [
            item for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if len(text_items) != 1:
            raise ValueError("each assistant turn must have exactly one text item")
        target = str(text_items[0].get("text", ""))
        original_targets.append(target)
        text_items[0]["text"] = wrapped(target)
    if not original_targets:
        raise ValueError("row has no assistant target")
    return output, original_targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    parent_train = args.parent_dir / "train.parquet"
    parent_report_path = args.parent_dir / "report.json"
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    if not (
        parent_report.get("format") == PARENT_FORMAT
        and parent_report.get("ok") is True
        and parent_report.get("rows") == 39561
        and parent_report.get("base_rows") == 33180
        and parent_report.get("ost_scene_rows") == 6381
        and parent_report.get("ost_supervised_turns") == 45006
        and abs(float(parent_report.get("ost_mass_ratio_to_parent")) - 0.5) < 1e-12
        and parent_report.get("causal_multi_round") is True
        and parent_report.get("future_frames_visible") is False
        and parent_report.get("train_sha256") == sha256(parent_train)
        and (args.parent_dir / "DATA_READY").is_file()
        and (args.parent_dir / "MEDIA_DECODE_READY").is_file()
    ):
        raise ValueError("parent ChronoReplay contract drifted")

    frame = pd.read_parquet(parent_train)
    required = {
        "uid", "messages", "sample_loss_weight", "supervised_assistant_ordinals"
    }
    if len(frame) != 39561 or not required.issubset(frame.columns):
        raise ValueError("parent row schema drifted")

    output_rows = []
    semantic_targets: list[list[str]] = []
    assistant_turns = 0
    for raw in frame.to_dict("records"):
        row = {str(key): builtin(value) for key, value in raw.items()}
        messages, targets = transform_messages(row["messages"])
        row["messages"] = messages
        row["common_interface_wrapper"] = True
        row["common_interface_format"] = "canonical_json_single_value_v1"
        output_rows.append(row)
        semantic_targets.append(targets)
        assistant_turns += len(targets)

    if len(output_rows) != len(frame):
        raise RuntimeError("row count changed")
    before_weights = [float(value) for value in frame["sample_loss_weight"].tolist()]
    after_weights = [float(row["sample_loss_weight"]) for row in output_rows]
    if before_weights != after_weights:
        raise RuntimeError("sample weights changed")

    for source, target in zip(frame.to_dict("records"), output_rows, strict=True):
        if builtin(source["supervised_assistant_ordinals"]) != builtin(
            target["supervised_assistant_ordinals"]
        ):
            raise RuntimeError("supervised ordinals changed")
        if str(source["uid"]) != str(target["uid"]):
            raise RuntimeError("row identity changed")

    args.output_dir.mkdir(parents=True)
    train = args.output_dir / "train.parquet"
    pd.DataFrame.from_records(output_rows).to_parquet(
        train, index=False, **PARQUET_ENCODING
    )
    semantic_sha = hashlib.sha256(canonical(semantic_targets).encode()).hexdigest()
    report = {
        "format": FORMAT,
        "ok": True,
        "scientific_question_id": "SQ2",
        "rows": len(output_rows),
        "assistant_turns": assistant_turns,
        "base_rows": int(parent_report["base_rows"]),
        "trajectory_rows": int(parent_report["ost_scene_rows"]),
        "supervised_transitions": int(parent_report["ost_supervised_turns"]),
        "trajectory_mass_ratio_to_parent": float(
            parent_report["ost_mass_ratio_to_parent"]
        ),
        "sample_loss_weight_mean": sum(after_weights) / len(after_weights),
        "parent_train": str(parent_train.resolve()),
        "parent_train_sha256": sha256(parent_train),
        "train": str(train.resolve()),
        "train_sha256": sha256(train),
        "semantic_target_sequence_sha256": semantic_sha,
        "canonical_interface": "canonical_json_single_value_v1",
        "target_values_preserved_exactly": True,
        "row_order_media_user_system_and_weights_unchanged": True,
        "only_assistant_surface_changed": True,
        "benchmark_questions_answers_categories_predictions_scores_read": False,
        "release_starts_from_official_base": True,
        "one_stage_training": True,
        "checkpoint_continuation": False,
        "weighted_model_fusion": False,
        "falsifiable_prediction": (
            "a common wrapper reduces native cross-interface transfer relative "
            "to ChronoReplay under matched content and family mass"
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("media_decode_audit.json", "MEDIA_DECODE_READY"):
        source = args.parent_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.output_dir / name)
    (args.output_dir / "DATA_READY").write_text("DATA_READY=1\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
