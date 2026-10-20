#!/usr/bin/env python3
"""Build an answer-equivalent, temporally destroyed trajectory control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


FORMAT = "iclr_spatial.causal_state_trajectory_order_control.v1"
SOURCE_FORMAT = "iclr_spatial.ost_official_train_release.v2"
OST_COMPONENT = "official_ost_train_multiturn_scene"


def builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return builtin(value.tolist())
    if isinstance(value, dict):
        return {str(key): builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def message_blocks(messages: Any) -> tuple[list[dict], list[list[dict]]]:
    values = builtin(messages)
    if not isinstance(values, list):
        raise ValueError("messages must be a list")
    prefix = []
    while values and values[0].get("role") == "system":
        prefix.append(values.pop(0))
    if len(values) < 2 or len(values) % 2:
        raise ValueError("OST control requires user/assistant turn blocks")
    blocks = []
    for index in range(0, len(values), 2):
        block = values[index : index + 2]
        if [item.get("role") for item in block] != ["user", "assistant"]:
            raise ValueError("OST turn block is not user then assistant")
        blocks.append(block)
    return prefix, blocks


def nonidentity_order(uid: str, count: int) -> list[int]:
    if count < 2:
        raise ValueError("trajectory control needs at least two turns")
    digest = int(hashlib.sha256(uid.encode()).hexdigest()[:16], 16)
    shift = 1 + digest % (count - 1)
    return list(range(shift, count)) + list(range(shift))


def reorder_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    prefix, blocks = message_blocks(output["messages"])
    if len(blocks) < 2:
        output["trajectory_order_control"] = False
        output["trajectory_original_turn_indices"] = [0]
        return output
    order = nonidentity_order(str(output.get("uid") or output.get("source_uid")), len(blocks))
    output["messages"] = prefix + [message for index in order for message in blocks[index]]
    supervised = builtin(output.get("supervised_assistant_ordinals"))
    if supervised is None or (isinstance(supervised, float) and math.isnan(supervised)):
        supervised = list(range(len(blocks)))
    supervised = [int(index) for index in supervised]
    if len(set(supervised)) != len(supervised) or any(
        index < 0 or index >= len(blocks) for index in supervised
    ):
        raise ValueError("supervised assistant ordinal coverage drift")
    weight_by_original: dict[int, float] | None = None
    weights = builtin(output.get("assistant_turn_loss_weights"))
    if weights is not None and not (isinstance(weights, float) and math.isnan(weights)):
        if not isinstance(weights, list) or len(weights) != len(supervised):
            raise ValueError("assistant turn weight coverage drift")
        weight_by_original = dict(zip(supervised, map(float, weights), strict=True))
    new_supervised = [
        new_index
        for new_index, original_index in enumerate(order)
        if original_index in set(supervised)
    ]
    output["supervised_assistant_ordinals"] = new_supervised
    if weight_by_original is not None:
        output["assistant_turn_loss_weights"] = [
            weight_by_original[order[new_index]] for new_index in new_supervised
        ]
    output["trajectory_order_control"] = True
    output["trajectory_original_turn_indices"] = order
    return output


def write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-train", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = json.loads(args.input_report.read_text(encoding="utf-8"))
    if not (
        source.get("format") == SOURCE_FORMAT
        and source.get("ok") is True
        and source.get("rows") == 39561
        and abs(float(source.get("ost_mass_ratio_to_parent")) - 0.25) < 1e-12
        and source.get("causal_multi_round") is True
        and source.get("future_frames_visible") is False
    ):
        raise ValueError("source mass0.25 contract drifted")
    frame = pd.read_parquet(args.input_train)
    if len(frame) != 39561 or "sample_loss_weight" not in frame.columns:
        raise ValueError("source row contract drifted")
    rows = []
    shuffled = 0
    single_turn = 0
    turn_blocks = 0
    supervised_turns = 0
    for raw in frame.to_dict("records"):
        row = {key: builtin(value) for key, value in raw.items()}
        is_ost = str(row.get("component")) == OST_COMPONENT
        # Exact analytical conversion from the normalized mass0.25 cell to
        # the normalized mass0.50 cell: double OST raw mass, then apply the
        # common 5/6 renormalization factor.
        row["sample_loss_weight"] = float(row["sample_loss_weight"]) * (
            5.0 / 3.0 if is_ost else 5.0 / 6.0
        )
        if is_ost:
            before_prefix, before_blocks = message_blocks(row["messages"])
            before_multiset = sorted(
                json.dumps(block, sort_keys=True, separators=(",", ":"))
                for block in before_blocks
            )
            row = reorder_trajectory(row)
            after_prefix, after_blocks = message_blocks(row["messages"])
            after_multiset = sorted(
                json.dumps(block, sort_keys=True, separators=(",", ":"))
                for block in after_blocks
            )
            if before_prefix != after_prefix or before_multiset != after_multiset:
                raise RuntimeError("trajectory content changed under order control")
            if len(before_blocks) >= 2:
                if row["trajectory_original_turn_indices"] == list(range(len(before_blocks))):
                    raise RuntimeError("trajectory permutation is the identity")
                shuffled += 1
            else:
                if row["trajectory_order_control"] is not False:
                    raise RuntimeError("single-turn control must remain unchanged")
                single_turn += 1
            turn_blocks += len(before_blocks)
            supervised = builtin(row.get("supervised_assistant_ordinals"))
            supervised_turns += (
                len(before_blocks)
                if supervised is None or (isinstance(supervised, float) and math.isnan(supervised))
                else len(supervised)
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    if (
        shuffled + single_turn != 6381
        or turn_blocks != 47086
        or supervised_turns != 45006
    ):
        raise RuntimeError(
            "OST trajectory coverage drift: "
            f"shuffled={shuffled} single={single_turn} turns={turn_blocks} "
            f"supervised={supervised_turns}"
        )
    mean_weight = float(output["sample_loss_weight"].mean())
    if not math.isclose(mean_weight, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(f"sample mass drift: {mean_weight}")

    args.output_dir.mkdir(parents=True)
    train = args.output_dir / "train.parquet"
    output.to_parquet(train, index=False)
    report = {
        "format": FORMAT,
        "ok": True,
        "scientific_question_id": "SQ2",
        "rows": len(output),
        "base_rows": len(output) - shuffled - single_turn,
        "ost_scene_rows": shuffled + single_turn,
        "ost_total_turn_blocks": turn_blocks,
        "ost_supervised_turns": supervised_turns,
        "ost_mass_ratio_to_parent": 0.5,
        "sample_loss_weight_mean": mean_weight,
        "all_images_text_and_targets_preserved": True,
        "assistant_turn_weights_reordered_with_targets": True,
        "trajectory_membership_preserved": True,
        "nonidentity_permuted_trajectories": shuffled,
        "single_turn_unpermuted_trajectories": single_turn,
        "chronological_causality": False,
        "contains_noncausal_order_by_design": True,
        "negative_control_only": True,
        "release_candidate": False,
        "benchmark_questions_answers_categories_predictions_scores_read": False,
        "release_starts_from_official_base": True,
        "one_stage_training": True,
        "checkpoint_continuation": False,
        "weighted_model_fusion": False,
        "source_train": str(args.input_train.resolve()),
        "source_train_sha256": sha256(args.input_train),
        "train": str(train.resolve()),
        "train_sha256": sha256(train),
    }
    write_atomic_json(args.output_dir / "report.json", report)
    (args.output_dir / "DATA_READY").write_text("ok\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
