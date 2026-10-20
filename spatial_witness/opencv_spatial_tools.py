"""Answer-blind OpenCV motion providers for the spatial agent runtime.

These providers add executable information rather than another language-model
turn.  They operate only on a caller-selected causal interval, expose image-
plane motion (not uncalibrated metric pose), and fail closed when tracking or
RANSAC support is insufficient.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from spatial_witness.causal_state_builder import pixel_sha256
from spatial_witness.causal_workspace import EvidenceRef, WorkspaceError
from spatial_witness.question_aware_agent_runtime import (
    LocalSpatialToolProvider,
    ProviderContext,
    ProviderResult,
)


def _number(value: Any, name: str, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkspaceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise WorkspaceError(f"{name} must be in [{lower}, {upper}]")
    return result


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 16:
        raise WorkspaceError("max_frames must be an integer in [2, 16]")
    return value


def _interval(arguments: Mapping[str, Any], *, with_box: bool) -> tuple[float, float, int, tuple[float, float, float, float] | None]:
    expected = {"start_fraction", "end_fraction", "max_frames"}
    if with_box:
        expected.add("box")
    if set(arguments) != expected:
        raise WorkspaceError("motion provider received invalid arguments")
    start = _number(arguments["start_fraction"], "start_fraction", lower=0.0, upper=1.0)
    end = _number(arguments["end_fraction"], "end_fraction", lower=0.0, upper=1.0)
    if start >= end:
        raise WorkspaceError("motion interval must be increasing")
    maximum = _count(arguments["max_frames"])
    if not with_box:
        return start, end, maximum, None
    raw_box = arguments["box"]
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise WorkspaceError("box must contain four normalized coordinates")
    box = tuple(
        _number(value, f"box[{index}]", lower=0.0, upper=1.0)
        for index, value in enumerate(raw_box)
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        raise WorkspaceError("box must have positive area")
    return start, end, maximum, box


def _sample(context: ProviderContext, start: float, end: float, maximum: int):
    frames = context.causal_frames
    first = int(round(start * (len(frames) - 1)))
    last = int(round(end * (len(frames) - 1)))
    if last <= first:
        last = min(len(frames) - 1, first + 1)
    count = min(maximum, last - first + 1)
    indices = np.linspace(first, last, num=count, dtype=np.int64).tolist()
    return [frames[index] for index in dict.fromkeys(indices)]


def _gray(frame) -> np.ndarray:
    return _gray_image(frame)


def _refs(context: ProviderContext, frames) -> tuple[EvidenceRef, ...]:
    return tuple(
        EvidenceRef(
            context.source_id,
            frame.frame_index,
            float(frame.timestamp),
            pixel_sha256(frame.image),
        )
        for frame in frames
    )


def estimate_image_similarity(
    previous: Any,
    current: Any,
    *,
    minimum_tracks: int = 8,
    minimum_inliers: int = 6,
) -> dict[str, Any] | None:
    """Estimate one uncalibrated image-plane similarity from two RGB frames.

    The result is a 2-D warp in normalized image coordinates. It is not a
    metric camera pose and never reads a question or option.
    """

    previous_gray = _gray_image(previous)
    current_gray = _gray_image(current)
    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=256,
        qualityLevel=0.01,
        minDistance=5,
        mask=None,
        blockSize=5,
    )
    if points is None or len(points) < minimum_tracks:
        return None
    try:
        next_points, status, _error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
    except cv2.error:
        return None
    if next_points is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    source = points.reshape(-1, 2)[valid]
    target = next_points.reshape(-1, 2)[valid]
    if len(source) < minimum_inliers:
        return None
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if affine is None or inlier_mask is None or int(inlier_mask.sum()) < minimum_inliers:
        return None
    height, width = current_gray.shape
    inliers = int(inlier_mask.sum())
    return {
        "image_dx_fraction": float(affine[0, 2]) / max(1, width),
        "image_dy_fraction": float(affine[1, 2]) / max(1, height),
        "image_rotation_radians": math.atan2(float(affine[1, 0]), float(affine[0, 0])),
        "ransac_inlier_fraction": inliers / max(1, len(source)),
        "track_count": int(len(source)),
        "metric_pose_claimed": False,
    }


def _gray_image(frame) -> np.ndarray:
    image = frame if isinstance(frame, np.ndarray) else np.asarray(
        frame.convert("RGB") if hasattr(frame, "convert") else frame.image.convert("RGB")
    )
    return _gray_array(image)


def _gray_array(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, 512.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _motion(frames, box: tuple[float, float, float, float] | None) -> dict[str, Any] | None:
    if len(frames) < 2:
        return None
    grays = [_gray(frame) for frame in frames]
    first = grays[0]
    mask = None
    if box is not None:
        height, width = first.shape
        x0, y0, x1, y1 = (
            int(round(box[0] * (width - 1))),
            int(round(box[1] * (height - 1))),
            int(round(box[2] * (width - 1))),
            int(round(box[3] * (height - 1))),
        )
        mask = np.zeros_like(first)
        mask[y0 : y1 + 1, x0 : x1 + 1] = 255
    points = cv2.goodFeaturesToTrack(
        first,
        maxCorners=256,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=5,
    )
    if points is None or len(points) < 8:
        return None
    cumulative_dx = cumulative_dy = cumulative_rotation = 0.0
    inliers_total = tracks_total = valid_steps = 0
    previous = first
    current_points = points
    for current in grays[1:]:
        next_points, status, _error = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            current_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            break
        valid = status.reshape(-1).astype(bool)
        source, target = current_points.reshape(-1, 2)[valid], next_points.reshape(-1, 2)[valid]
        if len(source) < 6:
            break
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
        )
        if affine is None or inlier_mask is None or int(inlier_mask.sum()) < 6:
            break
        height, width = current.shape
        cumulative_dx += float(affine[0, 2]) / max(1, width)
        cumulative_dy += float(affine[1, 2]) / max(1, height)
        cumulative_rotation += math.atan2(float(affine[1, 0]), float(affine[0, 0]))
        inliers_total += int(inlier_mask.sum())
        tracks_total += len(source)
        valid_steps += 1
        kept = inlier_mask.reshape(-1).astype(bool)
        current_points = target[kept].reshape(-1, 1, 2).astype(np.float32)
        previous = current
        if len(current_points) < 8:
            break
    if valid_steps == 0:
        return None
    magnitude = math.hypot(cumulative_dx, cumulative_dy)
    if abs(cumulative_rotation) > 0.08 and abs(cumulative_rotation) > 1.5 * magnitude:
        motion = "counterclockwise" if cumulative_rotation > 0 else "clockwise"
    elif abs(cumulative_dx) >= abs(cumulative_dy):
        motion = "image_right" if cumulative_dx > 0 else "image_left"
    else:
        motion = "image_down" if cumulative_dy > 0 else "image_up"
    return {
        "image_dx_fraction": cumulative_dx,
        "image_dy_fraction": cumulative_dy,
        "image_rotation_radians": cumulative_rotation,
        "dominant_image_motion": motion,
        "valid_pair_steps": valid_steps,
        "ransac_inlier_fraction": inliers_total / max(1, tracks_total),
        "metric_pose_claimed": False,
    }


def _execute(arguments: Mapping[str, Any], context: ProviderContext, *, with_box: bool) -> ProviderResult:
    start, end, maximum, box = _interval(arguments, with_box=with_box)
    selected = _sample(context, start, end, maximum)
    result = _motion(selected, box)
    references = _refs(context, selected)
    if result is None:
        return ProviderResult("unavailable", (), references, "insufficient_track_support")
    result["start_frame"] = selected[0].frame_index
    result["end_frame"] = selected[-1].frame_index
    return ProviderResult("ok", (result,), references)


def opencv_spatial_tool_providers() -> tuple[LocalSpatialToolProvider, ...]:
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return (
        LocalSpatialToolProvider(
            "estimate_frame_motion",
            implementation_hash,
            lambda arguments, context: _execute(arguments, context, with_box=False),
        ),
        LocalSpatialToolProvider(
            "track_region_motion",
            implementation_hash,
            lambda arguments, context: _execute(arguments, context, with_box=True),
        ),
    )

