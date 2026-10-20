import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_ucs_bench.py"


def test_direct_launcher_writes_nonempty_complete_receipt() -> None:
    source = (ROOT / "scripts" / "evaluate_ucs_bench_direct.sh").read_text()
    assert "printf 'ok\\n' >\"$OUTPUT/complete\"" in source
    assert 'UCS_CACHE_READY=' in source
    assert 'export OVOS_FRAME_CACHE_ONLY=1' in source
    assert 'export OVOS_FRAME_CACHE_RESIZE_FALLBACK=0' in source
    assert 'all_cache_entries_present' in source
    assert 'UCS_FRAME_POLICY="${UCS_FRAME_POLICY:-uniform}"' in source
    assert 'UCS_SUPPORT_READY_ROOT' in source
    assert '--frame-policy "$UCS_FRAME_POLICY"' in source
    assert 'EVAL_MAX_NEW_TOKENS_OVERRIDE' in source
    assert 'EVAL_ANSWER_PARSER' in source
    assert 'EVAL_SAVE_RAW_REPLY' in source
    assert '--answer-parser "$EVAL_ANSWER_PARSER"' in source
    assert '--save-raw-reply' in source


def test_complete_ucs_predictions_are_rescored(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "qid": "q1",
                        "predicted_letter": "A",
                        "answer_label": "A",
                        "category": "position",
                    },
                    {
                        "qid": "q2",
                        "predicted_letter": "B",
                        "answer_label": "C",
                        "category": "position",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "score.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--predictions",
            str(predictions),
            "--protocol",
            "timestamp_causal_model_only",
            "--model",
            "test-model",
            "--expected-total",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    score = json.loads(output.read_text())
    assert score["full_coverage"] is True
    assert score["overall_accuracy"] == 0.5
    assert score["by_category"]["position"]["correct"] == 1


def test_incomplete_ucs_predictions_fail_closed(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [{"qid": "q1", "predicted_label": "A", "answer_label": "A"}]
        ),
        encoding="utf-8",
    )
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--predictions",
            str(predictions),
            "--protocol",
            "timestamp_causal_model_only",
            "--model",
            "test-model",
            "--expected-total",
            "2",
            "--output",
            str(tmp_path / "score.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "incomplete UCS-Bench coverage" in process.stderr


def test_generic_causal_vlm_result_schema_is_rescored(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "q1",
                "qid": "q1",
                "answer_label": "B",
                "category": "memory",
                "subcategory": "last_seen",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps({"results": [{"id": "q1", "pred": "B", "gt": "B"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "score.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--predictions",
            str(predictions),
            "--questions",
            str(questions),
            "--protocol",
            "timestamp_causal_model_only",
            "--model",
            "test-model",
            "--expected-total",
            "1",
            "--output",
            str(output),
        ],
        check=True,
    )
    score = json.loads(output.read_text())
    assert score["overall_accuracy"] == 1.0
    assert score["by_category"]["memory"]["correct"] == 1


def test_subset_score_filters_question_contract_by_opaque_ids(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join(
            json.dumps(
                {"id": name, "answer_label": answer, "category": "memory"}
            )
            for name, answer in (("q1", "A"), ("q2", "B"), ("q3", "C"))
        )
        + "\n",
        encoding="utf-8",
    )
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"ids": ["q1", "q3"]}), encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {"results": [{"id": "q1", "pred": "A"}, {"id": "q3", "pred": "B"}]}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "score.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--predictions",
            str(predictions),
            "--questions",
            str(questions),
            "--id-file",
            str(ids),
            "--protocol",
            "timestamp_causal_structured_memory",
            "--model",
            "test-model",
            "--expected-total",
            "2",
            "--output",
            str(output),
        ],
        check=True,
    )
    score = json.loads(output.read_text())
    assert score["total"] == 2
    assert score["correct"] == 1
    assert score["full_coverage"] is True
