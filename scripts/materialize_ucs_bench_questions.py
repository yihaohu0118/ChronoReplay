#!/usr/bin/env python3
"""Flatten the public UCS-Bench QA files into one timestamp-causal manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FORMAT = "iclr_spatial.ucs_bench_questions.v1"


def timestamp_seconds(value: Any) -> float:
    text = str(value).strip()
    # Six public rows contain a single trailing colon. Treat it as punctuation
    # and record the repair in the release report rather than dropping a row.
    text = text[:-1] if text.endswith(":") else text
    parts = [float(part) for part in text.split(":")]
    if not 1 <= len(parts) <= 3 or any(part < 0 for part in parts):
        raise ValueError(f"invalid UCS timestamp: {value!r}")
    total = 0.0
    for part in parts:
        total = total * 60.0 + part
    return total


def _evidence(raw: Any) -> tuple[list[dict[str, Any]], int, int]:
    normalized = []
    trailing_colon_repairs = 0
    reversed_interval_repairs = 0
    for item in raw or []:
        times = item.get("timestamps") or {}
        start_raw = times.get("start", 0)
        end_raw = times.get("end", start_raw)
        trailing_colon_repairs += int(str(start_raw).strip().endswith(":"))
        trailing_colon_repairs += int(str(end_raw).strip().endswith(":"))
        start = timestamp_seconds(start_raw)
        end = timestamp_seconds(end_raw)
        if end < start:
            start, end = end, start
            reversed_interval_repairs += 1
        normalized.append(
            {
                "description": str(item.get("description") or "").strip(),
                "start_seconds": start,
                "end_seconds": end,
            }
        )
    return normalized, trailing_colon_repairs, reversed_interval_repairs


def materialize(
    qa_dir: Path, *, expected_files: int = 532, expected_rows: int = 8114
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = sorted(qa_dir.glob("*-mcq.json"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    future_evidence = 0
    trailing_colon_repairs = 0
    reversed_interval_repairs = 0
    option_counts: Counter[int] = Counter()
    categories: Counter[str] = Counter()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"UCS QA file is not a nonempty list: {path}")
        file_video_uid = path.name.removesuffix("-mcq.json")
        for raw in payload:
            qid = str(raw.get("qid") or raw.get("q_id") or "").strip()
            if not qid or qid in seen:
                raise ValueError(f"missing or duplicate UCS qid: {qid!r}")
            seen.add(qid)
            video_uid = str(raw.get("video_uid") or "").strip()
            if not video_uid or video_uid != file_video_uid:
                raise ValueError(
                    f"video identity mismatch in {path.name}: {video_uid!r}"
                )
            options = {str(k).upper(): str(v) for k, v in raw["options"].items()}
            labels = tuple("ABCDE"[: len(options)])
            if tuple(options) != labels or len(options) not in (2, 5):
                raise ValueError(f"invalid UCS options for {qid}: {tuple(options)}")
            answer = str(raw.get("answer_label") or "").strip().upper()
            if answer not in options:
                raise ValueError(f"invalid UCS answer label for {qid}: {answer!r}")
            query_time = timestamp_seconds(raw["question_timestamps"])
            evidence, colon_repairs, interval_repairs = _evidence(raw.get("evidence"))
            trailing_colon_repairs += colon_repairs
            reversed_interval_repairs += interval_repairs
            is_causal = all(item["end_seconds"] <= query_time for item in evidence)
            future_evidence += int(not is_causal)
            category = str(raw.get("category") or "unknown")
            option_counts[len(options)] += 1
            categories[category] += 1
            rows.append(
                {
                    "id": qid,
                    "qid": qid,
                    "video_uid": video_uid,
                    "video_filename": f"{video_uid}.mp4",
                    # Generic causal-VLM evaluator aliases.  These contain no
                    # benchmark label or routing rule: they only bind the
                    # media path, causal boundary, and reference label.
                    "video_path": f"{video_uid}.mp4",
                    "query_timestamp": query_time,
                    "query_times": [query_time],
                    "question": str(raw["question"]).strip(),
                    "options": options,
                    "answer_label": answer,
                    "answers": [answer],
                    "category": category,
                    "subcategory": str(raw.get("subcategory") or "unknown"),
                    "task_subcategory": str(
                        raw.get("subcategory") or "unknown"
                    ),
                    "evidence": evidence,
                    "all_evidence_causal": is_causal,
                }
            )
    report = {
        "format": FORMAT,
        "ok": len(files) == expected_files and len(rows) == expected_rows,
        "files": len(files),
        "rows": len(rows),
        "unique_ids": len(seen),
        "unique_videos": len({row["video_uid"] for row in rows}),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "category_distribution": dict(sorted(categories.items())),
        "rows_with_post_query_evidence_annotation": future_evidence,
        "trailing_colon_timestamp_repairs": trailing_colon_repairs,
        "reversed_evidence_interval_repairs": reversed_interval_repairs,
        "causal_protocol": "pixels_and_memory_at_or_before_query_timestamp",
        "future_evidence_is_never_actor_visible": True,
    }
    if not report["ok"]:
        raise ValueError(
            f"incomplete UCS release: files={len(files)} rows={len(rows)}"
        )
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, report = materialize(args.qa_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
