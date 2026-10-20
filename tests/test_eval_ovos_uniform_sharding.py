import pytest
import json

from pathlib import Path

import synthesis.eval_ovos_uniform as evaluator
from synthesis.eval_ovos_uniform import (
    _candidate_prediction_map,
    _interleaved_shard,
    _prepare_example,
)


ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_evidence_redecodes_writer_indices_without_frame_cache() -> None:
    text = (ROOT / "synthesis/eval_ovos_uniform.py").read_text()
    assert "from benchmark.ovo.utils.frame_utils import _extract_frames_raw" in text
    assert "_extract_at_indices" not in text.split("if snapshot_store is not None:", 1)[1].split("item.video_frames = state_frames", 1)[0]
    assert "video_uid={uid} query_timestamp={query_timestamp}" in text


def test_interleaved_shards_are_disjoint_complete_and_ordered() -> None:
    rows = [{"id": str(index)} for index in range(11)]
    shards = [
        _interleaved_shard(rows, rank=rank, world_size=4)
        for rank in range(4)
    ]
    assert [[row["id"] for row in shard] for shard in shards] == [
        ["0", "4", "8"],
        ["1", "5", "9"],
        ["2", "6", "10"],
        ["3", "7"],
    ]
    observed = [row["id"] for shard in shards for row in shard]
    assert sorted(observed, key=int) == [row["id"] for row in rows]
    assert len(observed) == len(set(observed))


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(-1, 2), (2, 2), (0, 0)],
)
def test_interleaved_shards_fail_closed(rank: int, world_size: int) -> None:
    with pytest.raises(ValueError, match="invalid interleaved shard"):
        _interleaved_shard([], rank=rank, world_size=world_size)


def test_candidate_prediction_map_reads_only_original_label_predictions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "row-1",
                        "pred": "B",
                        "prediction_mapped_to_original_labels": True,
                        "gt": "A",
                        "correct": False,
                        "task": "forbidden-to-verifier",
                    }
                ]
            }
        )
    )
    assert _candidate_prediction_map(path) == {"row-1": "B"}


def test_candidate_verifier_prompt_is_taskblind_and_records_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "extract_item_frames",
        lambda *args, **kwargs: ([], "BASE PROMPT", []),
    )
    prepared = _prepare_example(
        {
            "id": "row-1",
            "video_path": "unused.mp4",
            "question": "Where is the chair?",
            "options": {"A": "left", "B": "right"},
            "query_times": [1.0],
            "answers": ["A"],
        },
        video_base=Path("."),
        nframes=8,
        frame_size=448,
        prompt_style="default",
        verifier_candidates_by_id={"row-1": ("A", "B")},
    )
    assert prepared.verifier_candidates == ("A", "B")
    assert "proposed choices A and B" in prepared.prompt
    assert "task" not in prepared.prompt.lower()
