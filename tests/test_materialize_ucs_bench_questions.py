from pathlib import Path
import json

import pytest

from scripts.materialize_ucs_bench_questions import materialize, timestamp_seconds


def test_timestamp_seconds() -> None:
    assert timestamp_seconds("00:02:20") == 140
    assert timestamp_seconds("1:02:03.5") == 3723.5
    assert timestamp_seconds("00:02:20:") == 140
    with pytest.raises(ValueError):
        timestamp_seconds("1:2:3:4")


def test_materialize_rejects_incomplete_release_but_validates_rows(tmp_path: Path) -> None:
    row = {
        "video_uid": "clip",
        "qid": "q1",
        "question_timestamps": "00:00:10",
        "question": "Where is it?",
        "options": {"A": "left", "B": "right"},
        "answer_label": "A",
        "category": "position",
        "subcategory": "bearing",
        "evidence": [
            {"description": "visible", "timestamps": {"start": "8", "end": "9"}}
        ],
    }
    (tmp_path / "clip-mcq.json").write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="incomplete UCS release"):
        materialize(tmp_path)


def test_materialized_row_exposes_generic_causal_vlm_schema(tmp_path: Path) -> None:
    row = {
        "video_uid": "clip",
        "qid": "q1",
        "question_timestamps": "00:00:10",
        "question": "Where is it?",
        "options": {"A": "left", "B": "right"},
        "answer_label": "A",
        "category": "position",
        "subcategory": "bearing",
        "evidence": [],
    }
    path = tmp_path / "clip-mcq.json"
    path.write_text(json.dumps([row]))

    rows, report = materialize(tmp_path, expected_files=1, expected_rows=1)
    assert report["ok"] is True
    assert rows[0]["video_path"] == "clip.mp4"
    assert rows[0]["query_times"] == [10.0]
    assert rows[0]["answers"] == ["A"]
    assert rows[0]["task_subcategory"] == "bearing"
