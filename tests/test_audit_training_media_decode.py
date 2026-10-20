from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_training_media_decode.py"


def message(path: Path):
    return np.array(
        [
            {
                "role": "user",
                "content": np.array([{"type": "image", "image": str(path)}]),
            }
        ]
    )


def test_decode_audit_publishes_marker_only_for_decodable_media(tmp_path: Path):
    good = tmp_path / "good.jpg"
    Image.new("RGB", (8, 8), color="red").save(good)
    train = tmp_path / "train.parquet"
    pd.DataFrame({"messages": [message(good)]}).to_parquet(train, index=False)
    output = tmp_path / "audit.json"
    marker = tmp_path / "MEDIA_DECODE_READY"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train",
            str(train),
            "--output",
            str(output),
            "--ready-marker",
            str(marker),
            "--workers",
            "2",
        ],
        check=False,
    )
    assert completed.returncode == 0
    assert marker.is_file()
    report = json.loads(output.read_text())
    assert report["ok"] is True
    assert report["unique_media"] == 1

    good.write_bytes(b"")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train",
            str(train),
            "--output",
            str(output),
            "--ready-marker",
            str(marker),
            "--workers",
            "2",
        ],
        check=False,
    )
    assert completed.returncode == 2
    assert not marker.exists()
    report = json.loads(output.read_text())
    assert report["ok"] is False
    assert report["bad_count"] == 1
