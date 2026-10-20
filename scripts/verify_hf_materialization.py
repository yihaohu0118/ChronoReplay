#!/usr/bin/env python3
"""Verify that a merged HF checkpoint is complete without loading its tensors."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from data_pipeline.pure_vlm_utils import (
    CONTENT_SHA256_FORMAT,
    parquet_content_sha256,
    sha256,
)


def bind_sft_train_content_fingerprint(
    root: Path, *, write_contract: bool = False
) -> dict[str, object] | None:
    """Verify logical training rows and optionally persist their binding.

    Evaluation preflights call this verifier after ``SHA256SUMS`` has been
    frozen, so their default path must be strictly read-only.  The final model
    materialization step opts into the one permitted contract rewrite and
    regenerates the manifest immediately afterwards.
    """

    path = root / "sft_contract.json"
    if not path.is_file():
        return None
    contract = json.loads(path.read_text(encoding="utf-8"))
    train_data = contract.get("train_data")
    if not isinstance(train_data, dict):
        raise ValueError(f"SFT contract lacks train_data: {path}")
    train = Path(str(train_data.get("path") or ""))
    if not train.is_file():
        raise ValueError(f"SFT contract train parquet is unavailable: {train}")
    physical = sha256(train)
    if train_data.get("sha256") != physical:
        raise ValueError(
            f"SFT contract physical train hash mismatch: {path}: {physical}"
        )
    content = parquet_content_sha256(train)
    expected = {
        "canonical_rows_sha256": content,
        "canonical_rows_sha256_format": CONTENT_SHA256_FORMAT,
    }
    for key, value in expected.items():
        declared = train_data.get(key)
        if declared is not None and declared != value:
            raise ValueError(
                f"SFT contract logical train hash mismatch: {path}: {key}"
            )
    if write_contract:
        train_data.update(expected)
        contract["train_data"] = train_data
        fd, temporary = tempfile.mkstemp(prefix=".sft-contract.", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(contract, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {
        "physical_sha256": physical,
        "canonical_rows_sha256": content,
        "canonical_rows_sha256_format": CONTENT_SHA256_FORMAT,
        "rows": train_data.get("rows"),
    }


def safetensors_header(path: Path) -> int:
    before = path.stat()
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_size = int.from_bytes(raw, "little")
        if not 2 <= header_size <= 100_000_000:
            raise ValueError(f"invalid safetensors header size: {path}: {header_size}")
        header_raw = handle.read(header_size)
        if len(header_raw) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
        header = json.loads(header_raw)
        if not isinstance(header, dict) or not header:
            raise ValueError(f"empty safetensors header: {path}")

    intervals: list[tuple[int, int]] = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"invalid safetensors tensor entry: {path}: {name}")
        offsets = value.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
        ):
            raise ValueError(f"invalid safetensors offsets: {path}: {name}")
        start, end = offsets
        if start < 0 or end <= start:
            raise ValueError(f"invalid safetensors interval: {path}: {name}")
        intervals.append((start, end))
    if not intervals:
        raise ValueError(f"safetensors file contains no tensors: {path}")
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if start != cursor:
            raise ValueError(
                f"safetensors data is overlapping or not fully covered: {path}"
            )
        cursor = end

    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"safetensors shard changed during verification: {path}")
    expected_size = 8 + header_size + cursor
    if after.st_size != expected_size:
        raise ValueError(
            f"safetensors byte coverage mismatch: {path}: "
            f"expected={expected_size} actual={after.st_size}"
        )
    return cursor


def validate_materialization(
    root: Path, *, min_total_bytes: int = 1_000_000_000
) -> dict[str, object]:
    """Return an auditable report only for a complete, stable HF model."""

    if not (root / "config.json").is_file():
        raise ValueError(f"missing config.json: {root}")

    index_path = root / "model.safetensors.index.json"
    expected_tensor_bytes: int | None = None
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set((index.get("weight_map") or {}).values()))
        if not names:
            raise ValueError(f"empty safetensors index: {index_path}")
        shards = [root / name for name in names]
        metadata = index.get("metadata") or {}
        if "total_size" in metadata:
            expected_tensor_bytes = int(metadata["total_size"])
            if expected_tensor_bytes <= 0:
                raise ValueError(f"invalid index total_size: {index_path}")
    else:
        shards = [root / "model.safetensors"]
    missing = [
        str(path)
        for path in shards
        if not path.is_file() or path.stat().st_size <= 1_000_000
    ]
    if missing:
        raise ValueError(f"missing/truncated shards: {missing}")
    if sum(path.stat().st_size for path in shards) <= min_total_bytes:
        raise ValueError("materialized checkpoint is implausibly small")

    tensor_bytes = [safetensors_header(path) for path in shards]
    if expected_tensor_bytes is not None and sum(tensor_bytes) != expected_tensor_bytes:
        raise ValueError(
            "safetensors tensor-byte total disagrees with index metadata: "
            f"expected={expected_tensor_bytes} actual={sum(tensor_bytes)}"
        )
    return {
        "format": "iclr_spatial.hf_materialization_complete.v2",
        "config": str((root / "config.json").resolve()),
        "total_bytes": sum(path.stat().st_size for path in shards),
        "tensor_bytes": sum(tensor_bytes),
        "shards": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "tensor_bytes": data_bytes,
            }
            # Keep the standalone verifier runnable on Python 3.9 hosts; both
            # lists are created from the same shard list immediately above.
            for path, data_bytes in zip(shards, tensor_bytes)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("--write-marker", action="store_true")
    args = parser.parse_args()
    root = args.hf_dir
    try:
        report = validate_materialization(root)
        sft_data = bind_sft_train_content_fingerprint(
            root, write_contract=args.write_marker
        )
        if sft_data is not None:
            report["sft_train_data"] = sft_data
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if args.write_marker:
        marker = root / "MATERIALIZATION_COMPLETE.json"
        fd, temporary = tempfile.mkstemp(prefix=".materialization.", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, marker)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
