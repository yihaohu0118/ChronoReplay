"""Lightweight shared processor-budget contracts for data audits and training."""

from __future__ import annotations

from typing import Any


def configure_processor_pixel_budget(
    processor: Any, max_pixels: int
) -> dict[str, int]:
    """Apply a runtime pixel ceiling without mutating the frozen base files."""

    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if image_processor is None or not isinstance(size, dict):
        raise RuntimeError("processor has no configurable image size contract")
    shortest = int(size.get("shortest_edge", 0))
    longest = int(size.get("longest_edge", 0))
    if shortest <= 0 or longest <= 0 or shortest > longest:
        raise RuntimeError("processor image size contract is invalid")
    requested = longest if max_pixels == 0 else int(max_pixels)
    if requested < shortest:
        raise ValueError(
            f"max_pixels={requested} is below shortest_edge={shortest}"
        )
    image_processor.size = {
        "shortest_edge": shortest,
        "longest_edge": requested,
    }
    return dict(image_processor.size)
