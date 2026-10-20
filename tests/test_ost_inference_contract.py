from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ost_inference_uses_protocol_symmetric_truncated_image_decode() -> None:
    source = (ROOT / "benchmark/ost/inference.py").read_text()
    assert "from PIL import Image, ImageFile" in source
    assert "ImageFile.LOAD_TRUNCATED_IMAGES = True" in source
    assert "empty observation files" in source
    assert "for scan_id in scan_ids" in source
    assert "--classical-trajectory" in source
    assert "ObservationChunkPoseTimeline" in source
    assert 'turn["type"]' not in source
    assert 'turn["answer"]' not in source
    assert "--operator-grounded-candidate-choice" in source
    assert "compile_spatial_query(visible_question)" in source
    launcher = (ROOT / "scripts/evaluate_ost_bench_direct.sh").read_text()
    assert 'CLASSICAL_TRAJECTORY="${CLASSICAL_TRAJECTORY:-0}"' in launcher
    assert "--classical-trajectory" in launcher
    assert "trap fail_closed EXIT" in launcher
    assert '"$OUTPUT_ROOT/failed"' in launcher
    assert "OPERATOR_GROUNDED_CANDIDATE_CHOICE" in launcher
