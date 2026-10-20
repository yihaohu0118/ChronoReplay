"""Shared frame extraction for synthetic items (benchmark protocol)."""

from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import List, Optional, Tuple

from PIL import Image

from benchmark.ovo.option_utils import (
    has_image_options, split_options, build_image_options_instruction,
)
from benchmark.ovo.prompts import build_prompt
from synthesis.schema import SynthItem


def video_path_for(item: SynthItem, video_base: str) -> Path:
    return Path(video_base) / item.video_path


def referenced_time_from_visible_question(question: str, query_time: float) -> float:
    """Parse exactly one causal timestamp from model-visible question text."""

    matches = re.findall(
        r"\bat\s+(\d+(?:\.\d+)?)\s*seconds?\b",
        str(question or ""),
        flags=re.IGNORECASE,
    )
    valid = [float(value) for value in matches if 0 <= float(value) <= query_time]
    if len(valid) != 1:
        raise ValueError(
            "referenced_pair requires exactly one actor-visible causal timestamp"
        )
    return valid[0]


def extract_item_frames(
    item: SynthItem,
    video_base: str,
    nframes: int = 64,
    frame_size: int = 448,
    prompt_style: str = "default",
    frame_policy: str = "uniform",
    recent_window_fps: float = 4.0,
    hybrid_recent_nframes: int = 16,
) -> Tuple[List[Image.Image], str, Optional[List[Tuple[str, Image.Image]]]]:
    """Return (video_frames, text_prompt, option_images_or_None).

    Follows the benchmark eval protocol: uniform frames from [0, t_q],
    default prompt template, image options appended after video frames.
    """
    from benchmark.ovo.utils.frame_utils import (
        extract_causal_spatial_pyramid,
        extract_frames_fixed_count,
        extract_hybrid_global_recent,
        extract_log_decay,
        extract_uniform_subsample,
        extract_referenced_pair,
        extract_recent_window,
    )

    t_q = max(item.query_times)
    vp = video_path_for(item, video_base)
    if frame_policy in {"uniform", "cyclic"}:
        frames = extract_frames_fixed_count(
            str(vp), query_time=t_q, nframes=nframes, frame_size=frame_size
        )
        if frame_policy == "cyclic" and len(frames) > 1:
            key = str(getattr(item, "id", "") or getattr(item, "uid", ""))
            digest = int.from_bytes(
                hashlib.sha256(f"video-causal-origin-v1|{key}".encode()).digest()[:8],
                "big",
            )
            shift = 1 + digest % (len(frames) - 1)
            frames = frames[shift:] + frames[:shift]
    elif frame_policy == "uniform_subsample128":
        frames = extract_uniform_subsample(
            str(vp), query_time=t_q, nframes=nframes,
            source_nframes=128, frame_size=frame_size,
        )
    elif frame_policy == "recent":
        frames = extract_recent_window(
            str(vp),
            query_time=t_q,
            nframes=nframes,
            window_fps=recent_window_fps,
            frame_size=frame_size,
        )
    elif frame_policy == "log_decay":
        frames = extract_log_decay(
            str(vp), query_time=t_q, nframes=nframes, frame_size=frame_size
        )
    elif frame_policy == "causal_pyramid":
        frames = extract_causal_spatial_pyramid(
            str(vp),
            query_time=t_q,
            nframes=nframes,
            recent_window_fps=recent_window_fps,
            frame_size=frame_size,
        )
    elif frame_policy == "referenced_pair":
        reference_time = referenced_time_from_visible_question(item.question, t_q)
        frames = extract_referenced_pair(
            str(vp),
            query_time=t_q,
            reference_time=reference_time,
            nframes=nframes,
            window_fps=recent_window_fps,
            frame_size=frame_size,
        )
    elif frame_policy == "hybrid_global_recent":
        frames = extract_hybrid_global_recent(
            str(vp), query_time=t_q, nframes=nframes,
            recent_frames=hybrid_recent_nframes, window_fps=recent_window_fps,
            frame_size=frame_size, history_frame_size=224,
        )
    else:
        raise ValueError(f"unsupported frame policy: {frame_policy}")

    if has_image_options(item.options):
        text_options, option_images = split_options(item.options, frame_size=frame_size)
        prompt = build_prompt(item.question, text_options, style=prompt_style)
        prompt += "\n\n" + build_image_options_instruction(option_images)
        return frames, prompt, option_images

    prompt = build_prompt(item.question, item.options, style=prompt_style)
    return frames, prompt, None


def all_images_for_teacher(
    frames: List[Image.Image],
    option_images: Optional[List[Tuple[str, Image.Image]]] = None,
) -> List[Image.Image]:
    if not option_images:
        return frames
    return frames + [img for _, img in option_images]
