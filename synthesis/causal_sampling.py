"""Task-blind causal frame allocation shared by training and evaluation."""

from __future__ import annotations

from typing import List

import numpy as np


def causal_spatial_pyramid_indices(
    end_frame: int,
    video_fps: float,
    nframes: int = 128,
    recent_window_fps: float = 4.0,
) -> List[int]:
    """Build one deterministic multiscale frame set for a causal prefix.

    Half of the budget preserves the full prefix, one quarter samples three
    logarithmic history bands, and one quarter densely covers the recent
    state. The interface deliberately contains no question, task, answer,
    evidence annotation, or benchmark identifier.
    """

    if nframes < 1:
        raise ValueError("nframes must be positive")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    if recent_window_fps <= 0:
        raise ValueError("recent_window_fps must be positive")

    end_frame = max(0, int(end_frame))
    budget = min(int(nframes), end_frame + 1)
    global_budget = budget // 2
    history_budget = budget // 4
    recent_budget = budget - global_budget - history_budget

    targets: set[int] = set()
    if global_budget:
        targets.update(
            np.linspace(0, end_frame, global_budget, dtype=int).tolist()
        )

    near_start = max(0, end_frame - int(round(30 * video_fps)))
    mid_start = max(0, end_frame - int(round(300 * video_fps)))
    history_alloc = [
        int(round(history_budget * 0.60)),
        int(round(history_budget * 0.30)),
    ]
    history_alloc.append(history_budget - sum(history_alloc))
    history_bands = [
        (near_start, end_frame),
        (mid_start, near_start - 1),
        (0, mid_start - 1),
    ]
    for (start, stop), count in zip(
        history_bands, history_alloc, strict=True
    ):
        if count > 0 and stop >= start:
            targets.update(
                np.linspace(start, stop, count, dtype=int).tolist()
            )

    if recent_budget:
        step = max(1, int(round(video_fps / recent_window_fps)))
        recent_start = max(0, end_frame - (recent_budget - 1) * step)
        recent = list(range(recent_start, end_frame + 1, step))
        targets.update(recent[-recent_budget:])

    # Component overlap is expected. A fixed uniform top-up keeps the exact
    # budget without introducing a hidden routing rule.
    for index in np.linspace(0, end_frame, budget, dtype=int).tolist():
        if len(targets) >= budget:
            break
        targets.add(int(index))
    if len(targets) != budget:
        for index in range(end_frame + 1):
            if len(targets) >= budget:
                break
            targets.add(index)

    result = sorted(targets)
    if len(result) != budget:
        raise RuntimeError(
            f"causal pyramid produced {len(result)} indices for budget {budget}"
        )
    return result
