"""Local Qwen3-VL media dataset and processor helpers.

These utilities intentionally live in this repository so the released SFT
entry point does not import code from a sibling checkout.  Parquet rows use
OpenAI-style multimodal ``messages``.  Images resolve to RGB images and a
video resolves to an ordered RGB frame list before processor encoding.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _plain_path(value: str) -> str:
    return value[7:] if value.startswith("file://") else value


@lru_cache(maxsize=1)
def _media_prefix_map() -> tuple[tuple[Path, Path], ...]:
    """Return the optional node-local media mirror map.

    Released parquet rows retain their source-verifiable absolute media paths.
    A compute node may materialize the same immutable media beneath a different
    prefix.  ``SFT_MEDIA_PREFIX_MAP_JSON`` maps source prefixes to those
    node-local mirrors without rewriting the training rows.  The original path
    always wins when it exists, and malformed or non-absolute mappings fail
    closed before any sample is read.
    """

    raw = os.environ.get("SFT_MEDIA_PREFIX_MAP_JSON", "").strip()
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SFT_MEDIA_PREFIX_MAP_JSON is not valid JSON") from exc
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("SFT_MEDIA_PREFIX_MAP_JSON must be a non-empty object")
    mappings: list[tuple[Path, Path]] = []
    for source, mirror in payload.items():
        source_path = Path(str(source))
        mirror_path = Path(str(mirror))
        if not source_path.is_absolute() or not mirror_path.is_absolute():
            raise RuntimeError("SFT media prefix mappings must be absolute paths")
        mappings.append((source_path, mirror_path))
    return tuple(sorted(mappings, key=lambda pair: len(pair[0].parts), reverse=True))


def _resolve_path(value: str, base_dir: Path | None) -> Path:
    path = Path(_plain_path(value))
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    prefer_mirror = os.environ.get("SFT_MEDIA_PREFER_MIRROR", "").strip() == "1"
    if path.is_absolute() and prefer_mirror:
        for source, mirror in _media_prefix_map():
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            mirrored = mirror / relative
            if not mirrored.exists():
                raise FileNotFoundError(
                    f"preferred SFT media mirror is incomplete: {mirrored}"
                )
            return mirrored
    if path.is_absolute() and not path.exists():
        for source, mirror in _media_prefix_map():
            try:
                relative = path.relative_to(source)
            except ValueError:
                continue
            path = mirror / relative
            break
    return path


def resolve_media_path(value: str, base_dir: Path | None = None) -> Path:
    """Resolve source media, including an explicitly configured node mirror."""

    return _resolve_path(value, base_dir)


def _load_video(value: Any, base_dir: Path | None) -> str | list[Image.Image]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        frames: list[Image.Image] = []
        for raw in value:
            if isinstance(raw, Image.Image):
                frames.append(raw.convert("RGB"))
            elif isinstance(raw, str):
                frames.append(Image.open(_resolve_path(raw, base_dir)).convert("RGB"))
            else:
                raise TypeError(f"unsupported video frame payload: {type(raw)!r}")
        if not frames:
            raise ValueError("video frame list must be non-empty")
        return frames
    if isinstance(value, str):
        path = _resolve_path(value, base_dir)
        if not path.is_file():
            raise FileNotFoundError(path)
        # Keep encoded videos as paths. qwen_vl_utils then applies its native
        # nframes/fps sampling instead of decoding every source frame here.
        return str(path)
    raise TypeError(f"unsupported video payload: {type(value)!r}")


def normalize_messages(
    messages: Any,
    *,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert parquet/Arrow message values into processor-ready objects."""

    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    output: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return output
    for raw_message in messages:
        if hasattr(raw_message, "as_py"):
            raw_message = raw_message.as_py()
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role") or "user")
        content = raw_message.get("content")
        if content is None:
            content = []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif hasattr(content, "tolist"):
            content = content.tolist()
        normalized_content: list[dict[str, Any]] = []
        for raw_item in content if isinstance(content, list) else []:
            if hasattr(raw_item, "as_py"):
                raw_item = raw_item.as_py()
            if not isinstance(raw_item, dict):
                continue
            item_type = str(raw_item.get("type") or "")
            if item_type == "text":
                normalized_content.append(
                    {"type": "text", "text": str(raw_item.get("text") or "")}
                )
            elif item_type == "image" and raw_item.get("image"):
                image = raw_item["image"]
                if isinstance(image, str):
                    image_path = _resolve_path(image, base_dir)
                    image = Image.open(image_path).convert("RGB")
                elif isinstance(image, Image.Image):
                    image = image.convert("RGB")
                else:
                    raise TypeError(f"unsupported image payload: {type(image)!r}")
                normalized_image: dict[str, Any] = {"type": "image", "image": image}
                # The clock ablation changes video/image serialization while
                # retaining the native video's per-frame spatial budget.
                # Historical image rows remain unchanged because this branch
                # is enabled only by the explicit source-certified flag.
                if raw_item.get("clock_ablation_match_video_preprocess") is True:
                    for key in ("min_pixels", "max_pixels"):
                        value = raw_item.get(key)
                        if value is None:
                            raise ValueError(
                                "clock ablation image is missing its video pixel budget"
                            )
                        normalized_image[key] = int(value)
                normalized_content.append(normalized_image)
            elif item_type == "video" and raw_item.get("video") is not None:
                video = {
                    "type": "video",
                    "video": _load_video(raw_item["video"], base_dir),
                }
                for key in ("fps", "nframes", "min_pixels", "max_pixels"):
                    value = raw_item.get(key)
                    if value is not None:
                        video[key] = value
                normalized_content.append(video)
        output.append({"role": role, "content": normalized_content})
    return output


class MultimodalSFTDataset(Dataset):
    """Read audited multimodal SFT rows without an external code import."""

    def __init__(self, path: Path, processor: Any, *, max_samples: int = 0) -> None:
        frame = pd.read_parquet(path)
        if max_samples > 0:
            frame = frame.head(max_samples)
        self.rows = frame.to_dict("records")
        self.processor = processor
        self.base_dir = path.resolve().parent

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        raw_weight = row.get("sample_loss_weight")
        sample_loss_weight = float(1.0 if raw_weight is None else raw_weight)
        if not math.isfinite(sample_loss_weight) or sample_loss_weight <= 0.0:
            raise RuntimeError(
                "train-ready contract violated: "
                f"uid={row.get('uid') or f'row_{index}'} "
                f"sample_loss_weight={sample_loss_weight}"
            )
        return {
            "uid": str(row.get("uid") or f"row_{index}"),
            "messages": normalize_messages(
                row.get("messages"),
                base_dir=self.base_dir,
            ),
            "sample_loss_weight": sample_loss_weight,
        }


def load_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for message in messages:
        for item in message.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            value = item.get("image")
            if isinstance(value, Image.Image):
                images.append(value.convert("RGB"))
            elif value:
                images.append(Image.open(_plain_path(str(value))).convert("RGB"))
    return images


def encode_processor_item(
    processor: Any,
    messages: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    from qwen_vl_utils import process_vision_info

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadata = None
    if videos is not None:
        videos, video_metadata = zip(*videos)
        videos, video_metadata = list(videos), list(video_metadata)
    image_slots = rendered.count("<|image_pad|>")
    video_slots = rendered.count("<|video_pad|>")
    if image_slots != len(images or []):
        raise RuntimeError(
            "strict media contract violated: "
            f"chat template has {image_slots} image slots but messages contain "
            f"{len(images or [])} images"
        )
    if video_slots != len(videos or []):
        raise RuntimeError(
            "strict media contract violated: "
            f"chat template has {video_slots} video slots but messages contain "
            f"{len(videos or [])} videos"
        )
    kwargs: dict[str, Any] = {
        "text": rendered,
        "return_tensors": "pt",
    }
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
        kwargs["video_metadata"] = video_metadata
        kwargs.update(video_kwargs)
    item = processor(**kwargs)
    return {
        key: (
            value.squeeze(0)
            if hasattr(value, "dim")
            and value.dim() > 1
            and key in {"input_ids", "attention_mask"}
            else value
        )
        for key, value in item.items()
    }


def find_subsequence(
    haystack: list[int],
    needle: list[int],
    start: int,
) -> int:
    if not needle:
        return -1
    for index in range(max(start, 0), len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1
