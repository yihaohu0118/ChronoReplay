"""Run a tiny synthetic data/weighting pipeline on CPU; no model or dataset download."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


def fixtures(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        Image.new("RGB", (32, 32), (40 * i, 30, 80)).save(root / f"frame-{i}.png")
    rows = []
    for i, view in enumerate(("dense", "context", "long")):
        rows.append({
            "uid": f"cat-{view}", "source_id": "synthetic-example", "scene_id": "train-a",
            "semantic_id": "cat-question", "family": "categorical", "view": view,
            "source_split": "train", "messages": [
                {"role": "user", "content": [{"type": "image", "image": f"frame-{i}.png"},
                    {"type": "text", "text": "Synthetic format example. Choose A or B."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
            ],
        })
    rows.append({
        "uid": "numeric-1", "source_id": "synthetic-example", "scene_id": "train-a",
        "semantic_id": "numeric-question", "family": "numeric", "view": "single",
        "source_split": "train", "messages": [
            {"role": "user", "content": [{"type": "image", "image": "frame-3.png"},
                {"type": "text", "text": "Synthetic numeric target format."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "1.0 m"}]},
        ],
    })
    scenes = [{
        "scan_id": "train-b", "system_prompt": "Synthetic episode for software checks.",
        "user_message": [
            {"image_paths": ["frame-0.png"], "prompt": "Observe.", "answer": "Observed.", "type": "None"},
            {"image_paths": ["frame-4.png"], "prompt": "Synthetic event turn.", "answer": "1", "type": "temporal"},
        ],
    }]
    (root / "preservation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "ost_train.json").write_text(json.dumps(scenes))
    (root / "heldout_scenes.json").write_text(json.dumps(["heldout-example"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Choose a new output directory.")
    root = a.output.resolve()
    fixtures(root / "inputs")
    subprocess.run([
        sys.executable, "-m", "scripts.prepare_data",
        "--preservation", str(root / "inputs/preservation.jsonl"),
        "--ost-train", str(root / "inputs/ost_train.json"),
        "--heldout-scenes", str(root / "inputs/heldout_scenes.json"),
        "--media-root", str(root / "inputs"), "--output", str(root / "prepared"),
    ], check=True)
    report = json.loads((root / "prepared/report.json").read_text())
    assert report["rows"] == 5 and report["trajectory_supervised_turns"] == 1
    assert abs(report["mean_sample_weight"] - 1) < 1e-12
    masses = report["family_loss_mass"]
    assert abs(masses["L"] - masses["N"]) < 1e-12
    assert abs(masses["N"] - masses["OST"]) < 1e-12
    print("CPU_SMOKE_OK: synthetic inputs -> decoded media -> weighted train.parquet")


if __name__ == "__main__":
    main()
