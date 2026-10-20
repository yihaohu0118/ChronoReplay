from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_ucs_causal_shards import merge


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"summary": {"total": len(rows)}, "results": rows}))
    return path


def test_merge_ucs_shards_requires_unique_complete_ids(tmp_path: Path) -> None:
    one = _write(tmp_path / "one.json", [{"id": "q1", "correct": True}])
    two = _write(tmp_path / "two.json", [{"id": "q2", "correct": False}])
    payload = merge([one, two], 2)
    assert payload["full_coverage"] is True
    assert payload["correct"] == 1
    assert [row["id"] for row in payload["results"]] == ["q1", "q2"]
    with pytest.raises(ValueError, match="duplicate"):
        merge([one, one], 2)
