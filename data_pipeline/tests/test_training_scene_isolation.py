from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.stages.audit_training_scene_isolation import audit
from data_pipeline.stages.build_protected_scene_registry import build, scene_tokens
from data_pipeline.stages.build_scene_disjoint_eval_manifest import (
    build as build_manifest,
)


def test_packaging_suffixes_do_not_hide_a_scene() -> None:
    assert "scene0221_01" in scene_tokens(
        "scannet_videos_128f/train/scene0221_01_128f.mp4"
    )
    assert "scene0221_01" in scene_tokens("VSI-Bench/scannet/scene0221_01.mp4")
    assert "6c75a5290e" in scene_tokens("scannetpp2_compressed/6c75a5290e_compressed.mp4")
    assert "45261121" in scene_tokens("VSI-Bench/arkitscenes/45261121.mp4")


def test_packaging_words_are_not_scene_identities() -> None:
    assert scene_tokens("clips/train.mp4") == set()
    assert scene_tokens("") == set()
    assert scene_tokens("videos/scene.mp4") == set()


def _registry(tmp_path: Path, video_path: str) -> Path:
    benchmark = tmp_path / "ovo_s.parquet"
    pd.DataFrame([{"video_path": video_path}]).to_parquet(benchmark, index=False)
    report = build([("ovo_s", benchmark, ["video_path"])])
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _source(tmp_path: Path, video: str) -> Path:
    path = tmp_path / "source.parquet"
    pd.DataFrame([{"video": video}]).to_parquet(path, index=False)
    return path


def test_repackaged_benchmark_scene_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path, "VSI-Bench/scannet/scene0221_01.mp4")
    source = _source(tmp_path, "scannet_videos_128f/train/scene0221_01_128f.mp4")
    report = audit(registry_path=registry, sources=[("vsi_train", source, ["video"])])
    assert report["ok"] is False
    assert report["total_protected_scene_overlap"] == 1
    failure = report["failures"]["vsi_train:protected_scene_overlap"]
    assert failure["preview"] == ["scene0221_01"]
    assert failure["benchmarks"] == {"ovo_s": ["scene0221_01"]}


def test_disjoint_scene_passes(tmp_path: Path) -> None:
    registry = _registry(tmp_path, "VSI-Bench/scannet/scene0221_01.mp4")
    source = _source(tmp_path, "scannet_videos_128f/train/scene0335_02_128f.mp4")
    report = audit(registry_path=registry, sources=[("vsi_train", source, ["video"])])
    assert report["ok"] is True
    assert report["total_protected_scene_overlap"] == 0
    assert report["training_selection"] is False


def test_manifest_drops_only_trained_scenes(tmp_path: Path) -> None:
    benchmark = tmp_path / "ost.json"
    benchmark.write_text(
        json.dumps(
            [
                {"scan_id": "scene0006_00"},
                {"scan_id": "scene0006_00"},
                {"scan_id": "1mp3d_0004_region0"},
                {"scan_id": "41069025"},
            ]
        ),
        encoding="utf-8",
    )
    source = _source(tmp_path, "scannet_videos_128f/train/scene0006_00_128f.mp4")
    report = build_manifest(
        benchmark=("ost_bench", benchmark, ["scan_id"]),
        training_sources=[("vsi_train", source, ["video"])],
    )
    assert report["total_rows"] == 4
    assert report["excluded_rows"] == 2
    assert report["admissible_rows"] == 2
    assert report["excluded_scene_ids"] == ["scene0006_00"]
    assert report["admissible_scene_ids"] == ["1mp3d_0004_region0", "41069025"]


def test_manifest_rejects_a_fully_contaminated_benchmark(tmp_path: Path) -> None:
    benchmark = tmp_path / "ost.json"
    benchmark.write_text(json.dumps([{"scan_id": "scene0006_00"}]), encoding="utf-8")
    source = _source(tmp_path, "scannet_videos_128f/train/scene0006_00_128f.mp4")
    with pytest.raises(ValueError, match="no scene-disjoint items"):
        build_manifest(
            benchmark=("ost_bench", benchmark, ["scan_id"]),
            training_sources=[("vsi_train", source, ["video"])],
        )


def test_registry_must_be_identity_only(tmp_path: Path) -> None:
    registry = _registry(tmp_path, "VSI-Bench/scannet/scene0221_01.mp4")
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["answers_read"] = True
    registry.write_text(json.dumps(payload), encoding="utf-8")
    source = _source(tmp_path, "scannet_videos_128f/train/scene0335_02_128f.mp4")
    with pytest.raises(ValueError, match="identity-only"):
        audit(registry_path=registry, sources=[("vsi_train", source, ["video"])])
