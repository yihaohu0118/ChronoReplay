"""Small deterministic utilities shared only by the pure-VLM release path."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


PARQUET_ENCODING = {
    "engine": "pyarrow",
    "compression": "zstd",
    "use_dictionary": False,
    "write_statistics": False,
}
CONTENT_SHA256_FORMAT = "canonical_jsonl_sorted_keys_v2_null_normalized"


def builtin(value: Any) -> Any:
    """Convert Arrow/NumPy containers into stable JSON-compatible values."""

    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        value = value.tolist()
    # The same Arrow null is surfaced as ``None`` by some pandas versions and
    # floating ``NaN`` by others. Logical data hashes must be host independent.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    if isinstance(value, Mapping):
        return {str(key): builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash logical rows independently of host-specific Parquet metadata."""

    digest = hashlib.sha256()
    for row in rows:
        rendered = json.dumps(
            builtin(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update((rendered + "\n").encode())
    return digest.hexdigest()


def parquet_content_sha256(path: Path) -> str:
    """Hash ordered logical rows independently of Parquet serialization."""

    import pandas as pd

    frame = pd.read_parquet(path)
    return content_sha256(
        {str(key): builtin(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    )


def canonical(value: Any) -> str:
    """Serialize a value using the release's stable JSON representation."""

    return json.dumps(
        builtin(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def text_parts(message: Mapping[str, Any]) -> list[str]:
    """Extract text blocks from one normalized chat message."""

    content = message.get("content") or []
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        str(item.get("text") or "")
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    ]


def image_paths(messages: Any) -> list[str]:
    """Extract image paths from normalized chat messages."""

    paths: list[str] = []
    for message in builtin(messages or []):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content") or []
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "image":
                continue
            value = item.get("image")
            if isinstance(value, str):
                paths.append(value[7:] if value.startswith("file://") else value)
    return paths
