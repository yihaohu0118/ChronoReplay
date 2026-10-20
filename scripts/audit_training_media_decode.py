#!/usr/bin/env python3
"""Decode every unique image referenced by an SFT parquet before GPU launch."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def collect_image_paths(value: Any, paths: set[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                collect_image_paths(json.loads(text), paths)
                return
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if text.startswith("/") and text.lower().endswith(IMAGE_SUFFIXES):
            paths.add(text)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            collect_image_paths(item, paths)
        return
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            collect_image_paths(item, paths)


def decode_image(path: str) -> tuple[str, str] | None:
    try:
        if os.path.getsize(path) <= 0:
            return path, "empty"
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return None
    except Exception as error:  # The report must retain codec/filesystem errors.
        return path, f"{type(error).__name__}:{error}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-reported-errors", type=int, default=500)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    started = time.time()
    args.ready_marker.unlink(missing_ok=True)
    frame = pd.read_parquet(args.train, columns=["messages"])
    paths: set[str] = set()
    collect_image_paths(frame["messages"].to_numpy(), paths)
    if not paths:
        raise ValueError("training parquet contains no absolute image paths")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        failures = [
            result
            for result in pool.map(decode_image, sorted(paths), chunksize=32)
            if result is not None
        ]

    report = {
        "format": "iclr_spatial.media_decode_audit.v1",
        "train": str(args.train.resolve()),
        "rows": len(frame),
        "unique_media": len(paths),
        "bad_count": len(failures),
        "bad": failures[: args.max_reported_errors],
        "elapsed_seconds": time.time() - started,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if failures:
        return 2
    args.ready_marker.write_text("ok\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
