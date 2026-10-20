from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.prepare_data import build, local_image, read_rows
from scripts.smoke_test import fixtures
from scripts.train import command
from scripts.materialize_frozen import select, text_sha256

ROOT = Path(__file__).resolve().parents[1]


def inputs(tmp_path):
    fixtures(tmp_path)
    return read_rows(tmp_path / "preservation.jsonl"), read_rows(tmp_path / "ost_train.json")


def test_balancing_preserves_semantic_mass_and_context_mask(tmp_path):
    preservation, scenes = inputs(tmp_path)
    rows, report, _ = build(preservation, scenes, tmp_path, {"heldout-example"})
    assert report["rows"] == 5
    assert report["mean_sample_weight"] == pytest.approx(1)
    masses = report["family_loss_mass"]
    assert masses["L"] == pytest.approx(masses["N"])
    assert masses["N"] == pytest.approx(masses["OST"])
    assert rows[-1]["supervised_assistant_ordinals"] == [1]
    assert len(rows[-1]["messages"]) == 4
    assert rows[-1]["messages"][1]["content"][0]["text"] == "Observed."
    assert sum(r["sample_loss_weight"] for r in rows[:3]) == pytest.approx(rows[3]["sample_loss_weight"])
    more = deepcopy(scenes)
    more[0]["user_message"].append(deepcopy(more[0]["user_message"][-1]))
    _, new, _ = build(preservation, more, tmp_path, {"heldout-example"})
    assert new["family_loss_mass"] == pytest.approx(masses)


def test_rejects_heldout_scenes_and_inconsistent_multiview_answers(tmp_path):
    preservation, scenes = inputs(tmp_path)
    with pytest.raises(ValueError, match="Held-out"):
        build(preservation, scenes, tmp_path, {"train-a"})
    with pytest.raises(ValueError, match="held-out"):
        build(preservation, scenes, tmp_path, {"train-b"})
    preservation[1]["messages"][-1]["content"][0]["text"] = "B"
    with pytest.raises(ValueError, match="same source target"):
        build(preservation, scenes, tmp_path, {"heldout-example"})


def test_local_media_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        local_image("../outside.png", tmp_path)
    with pytest.raises(ValueError, match="URLs"):
        local_image("https://example.invalid/image.png", tmp_path)


def test_published_training_budget_is_explicit_for_both_topologies():
    config = json.loads((ROOT / "configs/chronoreplay_train.json").read_text())
    for n, accum in [(8, "2"), (4, "4")]:
        cmd = command(Path("model"), Path("data"), Path("output"), n, config)
        for flag, value in [("--gradient-accumulation-steps", accum), ("--max-steps", "-1"),
                            ("--max-length", "32768"), ("--seed", "42"), ("--trainer-seed", "42")]:
            assert cmd[cmd.index(flag) + 1] == value
        assert "--sample-normalized-loss" in cmd
        assert "--no-dataloader-pin-memory" in cmd
        assert "--train-vision-encoder" not in cmd


def test_smoke_pipeline_writes_decoded_data(tmp_path):
    destination = tmp_path / "fresh"
    p = subprocess.run([sys.executable, "-B", "-m", "scripts.smoke_test",
                        "--output", str(destination)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "CPU_SMOKE_OK" in p.stdout
    report = json.loads((destination / "prepared/report.json").read_text())
    assert report["all_media_decoded"]
    assert (destination / "prepared/MEDIA_DECODE_READY").is_file()
    assert not report["paper_membership_verified"]


@pytest.mark.parametrize("benchmark", ["ovo", "ost", "ucs"])
@pytest.mark.parametrize("stage", ["infer", "score"])
def test_evaluation_dry_run_needs_no_weights(benchmark, stage):
    args = [sys.executable, "-B", "-m", "scripts.evaluate",
            "--benchmark", benchmark, "--stage", stage,
            "--model", "not-downloaded", "--annotation", "not-downloaded.json",
            "--media-root", "missing-media", "--output", "not-created",
            "--dry-run"]
    if benchmark == "ost":
        args += ["--manifest", "not-downloaded-manifest.json"]
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert not (ROOT / "not-created").exists()
    if (benchmark, stage) == ("ucs", "score"):
        assert "scripts.merge_ucs_causal_shards" in p.stdout


def test_training_launcher_flags_exist_in_actual_trainer():
    tree = ast.parse((ROOT / "scripts/train_qwen3vl_sft.py").read_text())
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")]
            allowed.update(flags)
            if any(k.arg == "action" and isinstance(k.value, ast.Attribute) and k.value.attr == "BooleanOptionalAction" for k in node.keywords):
                allowed.update("--no-" + f[2:] for f in flags)
    config = json.loads((ROOT / "configs/chronoreplay_train.json").read_text())
    cmd = command(Path("model"), Path("data"), Path("output"), 8, config)
    actual = cmd[cmd.index(str(ROOT / "scripts/train_qwen3vl_sft.py")) + 1:]
    assert {v for v in actual if v.startswith("--")} <= allowed


def test_frozen_materializer_keeps_order_and_rejects_changed_targets(tmp_path):
    preservation, scenes = inputs(tmp_path)
    rows, _, _ = build(preservation, scenes, tmp_path, {"heldout-example"})
    target = []
    for i, r in enumerate(rows):
        target.append({**{k: r.get(k) for k in (
            "uid", "num_images", "component", "target_contract", "workfirst_family",
            "supervised_assistant_ordinals", "sample_loss_weight")},
            "row_index": i, "text_sha256": text_sha256(r["messages"])})
    selection = {"format": "chronoreplay.frozen_training_selection.v1", "rows": target}
    recovered = select(list(reversed(rows)), selection)
    assert [r["uid"] for r in recovered] == [r["uid"] for r in rows]
    with pytest.raises(ValueError, match="Missing"):
        select(rows[:-1], selection)
    changed = deepcopy(rows)
    changed[0]["messages"][-1]["content"][0]["text"] = "changed target"
    with pytest.raises(ValueError, match="text mismatch"):
        select(changed, selection)


def test_train_rejects_changed_data_before_starting_workers(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"modified-data")
    (data / "report.json").write_text(json.dumps({"ok": True, "rows": 39561, "train_sha256": "0" * 64}))
    for marker in ("DATA_READY", "MEDIA_DECODE_READY"):
        (data / marker).write_text("ok\n")
    output = tmp_path / "never-started"
    p = subprocess.run([
        sys.executable, "-B", "-m", "scripts.train", "--model", str(model),
        "--data", str(data), "--output", str(output),
    ], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0
    assert "changed since" in p.stderr
    assert not output.exists()
