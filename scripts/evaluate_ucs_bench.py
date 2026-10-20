#!/usr/bin/env python3
"""Score a complete UCS-Bench prediction artifact under a causal protocol.

This adapter intentionally does not implement a second inference harness.  It
normalizes the public DirectMe evaluator outputs, recomputes exact multiple-
choice correctness from labels, and fails closed on incomplete coverage.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FORMAT = "iclr_spatial.ucs_bench_official_score.v1"
PROTOCOLS = {
    "timestamp_causal_model_only",
    "timestamp_causal_structured_memory",
}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = (
                value.get("predictions")
                or value.get("results")
                or value.get("details")
                or []
            )
        else:
            rows = []
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"no prediction rows found in {path}")
    return rows


def _identity(row: dict[str, Any]) -> str:
    for key in ("qid", "q_id", "question_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    video = row.get("video_uid") or row.get("video_id")
    timestamp = row.get("query_timestamp", row.get("question_timestamp"))
    question = row.get("question")
    if video is None or timestamp is None or not question:
        raise ValueError("row lacks both a question id and a video/timestamp/question key")
    return f"{video}\t{float(timestamp):.6f}\t{str(question).strip()}"


def _label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and 0 <= value < 5:
        return "ABCDE"[value]
    text = str(value).strip().upper()
    if text.isdigit() and 0 <= int(text) < 5:
        return "ABCDE"[int(text)]
    match = re.search(r"(?:^|\b)([A-E])(?:\b|$)", text)
    return match.group(1) if match else None


def _reference(row: dict[str, Any]) -> str | None:
    for key in ("answer_label", "ground_truth", "gt", "label"):
        label = _label(row.get(key))
        if label is not None:
            return label
    for key in ("answer_idx", "correct_idx", "gt_idx"):
        label = _label(row.get(key))
        if label is not None:
            return label
    return None


def _prediction(row: dict[str, Any]) -> str | None:
    for key in (
        "predicted_label",
        "predicted_letter",
        "pred",
        "prediction",
        "predicted_answer",
        "raw_answer",
    ):
        label = _label(row.get(key))
        if label is not None:
            return label
    return None


def _question_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_rows(path)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _identity(row)
        if key in index:
            raise ValueError(f"duplicate question identity: {key}")
        index[key] = row
    return index


def _selected_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if path.suffix.lower() == ".jsonl":
        values: Any = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = values.get("ids") or values.get("question_ids") or []
    if not isinstance(values, list) or not values:
        raise ValueError(f"no selected UCS identities found in {path}")
    output: set[str] = set()
    for value in values:
        key = _identity(value) if isinstance(value, dict) else str(value).strip()
        if not key or key in output:
            raise ValueError(f"invalid or duplicate selected UCS identity: {key!r}")
        output.add(key)
    return output


def _stats(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        name = str(row["group"])
        counts[name][0] += 1
        counts[name][1] += int(row["correct"])
    return {
        name: {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
        }
        for name, (total, correct) in sorted(counts.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--id-file", type=Path)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-total", type=int, default=8114)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = _read_rows(args.predictions)
    questions = _question_index(args.questions)
    selected = _selected_ids(args.id_file)
    if selected is not None:
        missing_questions = selected - set(questions)
        if questions and missing_questions:
            raise ValueError(
                f"selected UCS identities absent from questions: {len(missing_questions)}"
            )
        questions = {
            key: value for key, value in questions.items() if key in selected
        }
        if len(selected) != args.expected_total:
            raise ValueError(
                "selected UCS coverage does not match --expected-total: "
                f"{len(selected)} != {args.expected_total}"
            )
    seen: set[str] = set()
    scored: list[dict[str, Any]] = []
    for prediction in predictions:
        key = _identity(prediction)
        if selected is not None and key not in selected:
            raise ValueError(f"prediction is outside selected UCS identities: {key}")
        if key in seen:
            raise ValueError(f"duplicate prediction identity: {key}")
        seen.add(key)
        question = questions.get(key, {})
        reference = _reference(prediction) or _reference(question)
        predicted = _prediction(prediction)
        if reference is None or predicted is None:
            raise ValueError(f"unscorable prediction: {key}")
        category = prediction.get(
            "category", question.get("category", question.get("dimension", "unknown"))
        )
        subcategory = prediction.get(
            "subcategory", question.get("subcategory", "unknown")
        )
        scored.append(
            {
                "id": key,
                "reference": reference,
                "prediction": predicted,
                "correct": predicted == reference,
                "category": str(category),
                "subcategory": str(subcategory),
            }
        )

    if len(scored) != args.expected_total:
        raise ValueError(
            f"incomplete UCS-Bench coverage: {len(scored)} != {args.expected_total}"
        )
    if questions and set(questions) != seen:
        missing = len(set(questions) - seen)
        extra = len(seen - set(questions))
        raise ValueError(f"question/prediction identity mismatch: missing={missing} extra={extra}")

    correct = sum(int(row["correct"]) for row in scored)
    by_category = _stats({**row, "group": row["category"]} for row in scored)
    by_subcategory = _stats({**row, "group": row["subcategory"]} for row in scored)
    report = {
        "format": FORMAT,
        "protocol": args.protocol,
        "model": args.model,
        "full_coverage": True,
        "expected_total": args.expected_total,
        "total": len(scored),
        "correct": correct,
        "official_overall_accuracy": correct / len(scored),
        "overall_accuracy": correct / len(scored),
        "overall_percent": 100.0 * correct / len(scored),
        "by_category": by_category,
        "by_subcategory": by_subcategory,
        "predictions_path": str(args.predictions.resolve()),
        "questions_path": str(args.questions.resolve()) if args.questions else None,
        "id_file": str(args.id_file.resolve()) if args.id_file else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("total", "correct", "overall_percent")}, sort_keys=True))


if __name__ == "__main__":
    main()
