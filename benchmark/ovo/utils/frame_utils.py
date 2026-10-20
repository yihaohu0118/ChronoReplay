"""
Frame extraction utilities for OVO-S evaluation.

Strategy:
  1. On first access, extract target frames and cache as JPEG files.
  2. On subsequent accesses, load directly from cache (~0.1s vs ~35s).
  3. Uses decord for extraction (falls back to OpenCV).

Cache layout:
  {cache_dir}/{video_id}/{frame_idx}.jpg
"""

import os
import cv2
import hashlib
import json
import threading
import numpy as np
from functools import lru_cache
from PIL import Image
from typing import List, Optional, Tuple
from pathlib import Path

from synthesis.causal_sampling import causal_spatial_pyramid_indices

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

# Default cache directory (can be overridden via set_cache_dir)
_CACHE_DIR: Optional[Path] = None


def set_cache_dir(path: str):
    """Set the frame cache directory."""
    global _CACHE_DIR
    _CACHE_DIR = Path(path)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_dir() -> Optional[Path]:
    """Get cache dir, auto-initialize if not set."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        # Shared evaluation code is often staged read-only while workers run
        # as a separate experiment account.  Let the launcher bind a writable
        # cache without changing sampling semantics or copying the code tree.
        configured = os.getenv("OVOS_FRAME_CACHE_DIR", "").strip()
        default = (
            Path(configured).expanduser()
            if configured
            else Path(__file__).parent.parent / ".frame_cache"
        )
        default.mkdir(parents=True, exist_ok=True)
        _CACHE_DIR = default
    return _CACHE_DIR


def _compute_target_frames(
    query_time: float, video_fps: float, max_frames: int, fps: float
) -> List[int]:
    """Compute sorted list of target frame indices to extract."""
    end_frame = int(query_time * video_fps)
    if end_frame <= 0:
        end_frame = 1

    fps_sample_count = int(query_time * fps) + 1

    if fps_sample_count < max_frames:
        sample_interval = video_fps / fps
        target_frames = []
        frame_pos = 0.0
        while frame_pos <= end_frame:
            target_frames.append(int(frame_pos))
            frame_pos += sample_interval
        if target_frames[-1] != end_frame:
            target_frames.append(end_frame)
    else:
        if max_frames == 1:
            target_frames = [end_frame]
        else:
            target_frames = np.linspace(
                0, end_frame, max_frames, dtype=int
            ).tolist()

    return sorted(set(target_frames))


def _resize_frame(frame_rgb: np.ndarray, frame_size: int) -> np.ndarray:
    """Resize frame so max dimension <= frame_size."""
    h, w = frame_rgb.shape[:2]
    if max(h, w) > frame_size:
        scale = frame_size / max(h, w)
        frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
    return frame_rgb


def _cache_key(video_path: str, target_frames: List[int], frame_size: int) -> str:
    """Build a unique cache filename for this exact extraction request."""
    canonical_path = str(Path(video_path).expanduser().resolve())
    vkey = hashlib.md5(canonical_path.encode()).hexdigest()[:12]
    fkey = hashlib.md5(
        f"{sorted(target_frames)}_{frame_size}".encode()
    ).hexdigest()[:8]
    return f"{vkey}_{fkey}.npy"


def _legacy_cache_key(
    video_path: str, target_frames: List[int], frame_size: int
) -> str:
    """Return the pre-canonicalization cache key for compatibility."""

    vkey = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
    fkey = hashlib.md5(
        f"{sorted(target_frames)}_{frame_size}".encode()
    ).hexdigest()[:8]
    return f"{vkey}_{fkey}.npy"


def _frame_cache_only() -> bool:
    """Return whether decoding is forbidden and every request must hit cache."""

    return os.getenv("OVOS_FRAME_CACHE_ONLY", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _existing_cache_path(
    video_path: str, target_frames: List[int], frame_size: int
) -> Optional[Path]:
    """Find a canonical or legacy cache entry and promote legacy in place."""

    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return None
    canonical = cache_dir / _cache_key(video_path, target_frames, frame_size)
    if canonical.is_file():
        return canonical
    legacy = cache_dir / _legacy_cache_key(video_path, target_frames, frame_size)
    if legacy == canonical or not legacy.is_file():
        return None
    # A hard link upgrades an existing multi-gigabyte cache without copying
    # frame bytes. Concurrent workers may safely race this promotion.
    try:
        os.link(legacy, canonical)
    except FileExistsError:
        pass
    except OSError:
        return legacy
    return canonical if canonical.is_file() else legacy


def _try_load_from_cache(
    video_path: str, target_frames: List[int], frame_size: int
) -> Optional[List[Image.Image]]:
    """Load all frames from a single .npy cache file (uncompressed, fast)."""
    fpath = _existing_cache_path(video_path, target_frames, frame_size)
    if fpath is None:
        return None
    try:
        # Load fully into memory (not mmap): mmap regions accumulate in RSS
        # because PIL Images keep references to the underlying buffer, and the
        # cgroup OOM-kills us at ~6 GB after a few dozen queries.
        # np.array(...) forces a copy so PIL Images don't hold the np.load buffer.
        arr = np.load(fpath)
        return [Image.fromarray(np.ascontiguousarray(arr[i])) for i in range(arr.shape[0])]
    except Exception:
        return None
    return frames


def _resize_cache_fallback_enabled() -> bool:
    value = os.getenv("OVOS_FRAME_CACHE_RESIZE_FALLBACK", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disable", "disabled"}


def _candidate_source_sizes(frame_size: int) -> List[int]:
    raw = os.getenv("OVOS_FRAME_CACHE_SOURCE_SIZES", "512,768,1024")
    sizes = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            size = int(item)
        except ValueError:
            continue
        if size > frame_size and size not in sizes:
            sizes.append(size)
    # Prefer the nearest larger cached size to minimize unnecessary resizing.
    return sorted(sizes)


def _try_derive_from_larger_cache(
    video_path: str, target_frames: List[int], frame_size: int
) -> Optional[List[Image.Image]]:
    """Create this cache entry by resizing an existing larger-frame cache."""
    if not _resize_cache_fallback_enabled():
        return None

    for source_size in _candidate_source_sizes(frame_size):
        source_frames = _try_load_from_cache(video_path, target_frames, source_size)
        if source_frames is None:
            continue

        resized_frames = []
        for img in source_frames:
            frame_rgb = np.array(img)
            frame_rgb = _resize_frame(frame_rgb, frame_size)
            resized_frames.append(Image.fromarray(np.ascontiguousarray(frame_rgb)))
        _save_to_cache(video_path, target_frames, resized_frames, frame_size)
        if os.getenv("OVOS_FRAME_CACHE_VERBOSE"):
            print(
                "Derived frame cache "
                f"frame_size={frame_size} from frame_size={source_size}: {video_path}"
            )
        return resized_frames

    return None


def _save_to_cache(
    video_path: str, target_frames: List[int],
    frames: List[Image.Image], frame_size: int
):
    """Atomically publish frames as one uncompressed ``.npy`` cache entry."""
    cache_dir = _get_cache_dir()
    if cache_dir is None:
        return
    fpath = cache_dir / _cache_key(video_path, target_frames, frame_size)
    if _existing_cache_path(video_path, target_frames, frame_size) is not None:
        return
    try:
        # Stack into (N, H, W, 3) — pad to uniform size if needed
        arrays = [np.array(img) for img in frames]
        # All frames should be same size after resize, but be safe
        max_h = max(a.shape[0] for a in arrays)
        max_w = max(a.shape[1] for a in arrays)
        padded = np.zeros((len(arrays), max_h, max_w, 3), dtype=np.uint8)
        for i, a in enumerate(arrays):
            padded[i, :a.shape[0], :a.shape[1]] = a
        # Do not expose a partially written npy file to parallel evaluators.
        # ``link`` is an atomic create-if-absent operation, so concurrent
        # writers can safely race without replacing a complete cache entry.
        tmp = fpath.with_name(f"{fpath.name}.{os.getpid()}.tmp.npy")
        try:
            np.save(tmp, padded)
            try:
                os.link(tmp, fpath)
            except FileExistsError:
                pass
        finally:
            tmp.unlink(missing_ok=True)
    except Exception:
        pass


def _extract_frames_raw(
    video_path: str, target_frames: List[int], frame_size: int
) -> List[Image.Image]:
    """Extract frames from video (decord preferred, OpenCV fallback)."""
    if os.getenv("OVOS_FRAME_CACHE_ONLY", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        raise FileNotFoundError(
            f"predecoded-only evaluation encountered a frame-cache miss: {video_path}"
        )
    extract_backend = os.getenv("OVOS_FRAME_EXTRACT_BACKEND", "").lower()
    use_decord = HAS_DECORD and extract_backend not in {"opencv", "cv2"}
    if use_decord:
        try:
            # Four decoder threads trigger FFmpeg EAGAIN/threaded-decoder
            # failures on a subset of Ego4D MP4s with decord 0.6.0.  Those
            # failures silently fall back to 128 independent OpenCV seeks and
            # dominate evaluation time.  Two threads retain parallel decode
            # while remaining stable on the frozen OVO-S media corpus.
            decord_threads = max(
                1, int(os.getenv("OVOS_DECORD_THREADS", "2"))
            )
            vr = VideoReader(
                str(video_path), ctx=cpu(0), num_threads=decord_threads
            )
            total = len(vr)
            if total <= 0:
                return []
            indices = sorted({min(max(int(f), 0), total - 1) for f in target_frames})
            batch = vr.get_batch(indices).asnumpy()
            frames = []
            for frame_rgb in batch:
                frame_rgb = _resize_frame(frame_rgb, frame_size)
                frames.append(Image.fromarray(frame_rgb))
            return frames
        except Exception as e:
            print(f"Warning: decord failed ({e}), falling back to OpenCV")

    # OpenCV fallback
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            target_frames = sorted(
                {min(max(int(fidx), 0), total - 1) for fidx in target_frames}
            )
        frames = []
        for fidx in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_frame(frame_rgb, frame_size)
            frames.append(Image.fromarray(frame_rgb))
        return frames
    finally:
        cap.release()


@lru_cache(maxsize=4096)
def _get_video_fps(video_path: str) -> float:
    """Get video FPS using decord or OpenCV."""
    source = Path(video_path).expanduser().resolve()
    configured = os.getenv("OVOS_VIDEO_METADATA_CACHE_DIR", "").strip()
    metadata_root = (
        Path(configured).expanduser()
        if configured
        else _get_cache_dir() / ".video_metadata"
    )
    metadata_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(source).encode()).hexdigest()
    metadata_path = metadata_root / f"{key}.json"
    try:
        source_stat = source.stat()
    except OSError:
        source_stat = None
    try:
        if metadata_path.is_file():
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_matches_source = source_stat is not None and (
                int(value.get("size", -1)) == source_stat.st_size
                and int(value.get("mtime_ns", -1)) == source_stat.st_mtime_ns
            )
            metadata_matches_cache_only_path = (
                source_stat is None and _frame_cache_only()
            )
            if (
                value.get("format") == "iclr_spatial.video_fps_cache.v1"
                and value.get("canonical_path") == str(source)
                and (metadata_matches_source or metadata_matches_cache_only_path)
                and float(value.get("fps", 0.0)) > 0
            ):
                return float(value["fps"])
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    fps_backend = os.getenv("OVOS_FRAME_FPS_BACKEND", "").lower()
    use_decord = HAS_DECORD and fps_backend not in {"opencv", "cv2"}
    fps = 0.0
    if use_decord:
        try:
            vr = VideoReader(str(source), ctx=cpu(0), num_threads=1)
            fps = float(vr.get_avg_fps())
            del vr
        except Exception:
            fps = 0.0
    if fps <= 0:
        cap = cv2.VideoCapture(str(source))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()

    if fps > 0 and metadata_path is not None and source_stat is not None:
        payload = {
            "format": "iclr_spatial.video_fps_cache.v1",
            "canonical_path": str(source),
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "fps": fps,
        }
        temporary = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
    return fps


@lru_cache(maxsize=4096)
def _get_video_frame_count(video_path: str) -> int:
    """Return the decoder-visible frame count used to clamp exact plans."""

    extract_backend = os.getenv("OVOS_FRAME_EXTRACT_BACKEND", "").lower()
    if HAS_DECORD and extract_backend not in {"opencv", "cv2"}:
        try:
            return int(len(VideoReader(str(video_path), ctx=cpu(0), num_threads=1)))
        except Exception:
            pass
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def extract_frames_for_query(
    video_path: str,
    query_time: float,
    max_frames: int = 64,
    fps: float = 1.0,
    frame_size: int = 512
) -> List[Image.Image]:
    """
    Extract frames for a single query point.

    First checks disk cache; on miss, extracts via decord/OpenCV and
    caches the result. Subsequent calls for the same video+frames
    return in ~0.1s instead of ~35s.
    """
    if not Path(video_path).exists() and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return []

    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        print(f"Warning: Invalid FPS for video: {video_path}")
        return []

    targets = _compute_target_frames(query_time, video_fps, max_frames, fps)

    # Try cache first
    cached = _try_load_from_cache(video_path, targets, frame_size)
    if cached is not None:
        return cached

    derived = _try_derive_from_larger_cache(video_path, targets, frame_size)
    if derived is not None:
        return derived

    # Cache miss — extract and save whatever frames are readable.  Some
    # videos cannot seek all requested indices; caching the partial result
    # avoids repeatedly live-decoding the same problematic segment.
    frames = _extract_frames_raw(video_path, targets, frame_size)
    if frames:
        _save_to_cache(video_path, targets, frames, frame_size)

    return frames


def extract_frames_fixed_count(
    video_path: str,
    query_time: float,
    nframes: int = 128,
    frame_size: int = 512,
    start_time: float = 0.0,
) -> List[Image.Image]:
    """
    Extract a fixed number of frames uniformly from [start_time, query_time].

    Uses np.linspace to sample exactly `nframes` frames.  Designed for
    video-type model input where a fixed frame count is required.

    Args:
        video_path: Path to the video file.
        query_time: End time in seconds.
        nframes:    Exact number of frames to extract.
        frame_size: Max dimension for frame resize.
        start_time: Segment start in seconds (default 0 = whole prefix);
                    used by persistent-buffer streaming eval to only ingest
                    frames since the previous query.

    Returns:
        List of PIL Image frames (length == nframes, or fewer on error).
    """
    if not Path(video_path).exists() and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return []

    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        print(f"Warning: Invalid FPS for video: {video_path}")
        return []

    cap_raw = os.getenv("OVOS_FRAME_LONG_QUERY_CAP_SEC", "").strip()
    if cap_raw:
        try:
            cap_sec = float(cap_raw)
        except ValueError:
            cap_sec = 0.0
        path_substr = os.getenv("OVOS_FRAME_LONG_QUERY_CAP_PATH_SUBSTR", "").strip()
        if cap_sec > 0 and query_time > cap_sec and (
            not path_substr or path_substr in str(video_path)
        ):
            print(
                "Warning: capping frame extraction query_time "
                f"from {query_time:.1f}s to {cap_sec:.1f}s for {video_path}",
                flush=True,
            )
            query_time = cap_sec

    end_frame = int(query_time * video_fps)
    if end_frame <= 0:
        end_frame = 1
    start_frame = max(0, int(start_time * video_fps))
    if start_frame >= end_frame:
        start_frame = max(0, end_frame - 1)

    target_frames = sorted(set(
        np.linspace(start_frame, end_frame, nframes, dtype=int).tolist()))

    # Try cache first
    cached = _try_load_from_cache(video_path, target_frames, frame_size)
    if cached is not None:
        return cached

    derived = _try_derive_from_larger_cache(video_path, target_frames, frame_size)
    if derived is not None:
        return derived

    # Cache miss — extract and save whatever frames are readable.  Some
    # videos cannot seek all requested indices; caching the partial result
    # avoids repeatedly live-decoding the same problematic segment.
    frames = _extract_frames_raw(video_path, target_frames, frame_size)
    if frames:
        _save_to_cache(video_path, target_frames, frames, frame_size)

    return frames


def uniform_subsample_fixed_indices(
    end_frame: int,
    *,
    source_nframes: int = 128,
    target_nframes: int = 81,
) -> List[int]:
    """Select a fixed-budget subset of the canonical uniform source plan.

    The target indices are always a subset of the exact ``source_nframes``
    uniform plan, enabling cache-only equal-budget controls without decoding
    new source pixels.
    """

    if end_frame < 1 or source_nframes < 1 or target_nframes < 1:
        raise ValueError("frame indices and budgets must be positive")
    source = sorted(set(
        np.linspace(0, end_frame, source_nframes, dtype=int).tolist()
    ))
    if len(source) <= target_nframes:
        return source
    positions = sorted(set(
        np.linspace(0, len(source) - 1, target_nframes, dtype=int).tolist()
    ))
    return [source[index] for index in positions]


def extract_uniform_subsample(
    video_path: str,
    query_time: float,
    nframes: int = 81,
    frame_size: int = 512,
    source_nframes: int = 128,
    **_kwargs,
) -> List[Image.Image]:
    """Cache-derivable fixed-budget subset of canonical uniform-128."""

    if not Path(video_path).exists() and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    end_frame = max(1, int(query_time * video_fps))
    targets = uniform_subsample_fixed_indices(
        end_frame,
        source_nframes=source_nframes,
        target_nframes=nframes,
    )
    return _extract_at_indices(video_path, targets, frame_size)


def fixed_count_interval_plan(
    video_path: str,
    *,
    start_time_exclusive: Optional[float],
    query_time: float,
    nframes: int = 16,
) -> Optional[dict]:
    """Plan bounded observations from one previously unconsumed interval.

    Unlike prefix sampling, the lower bound is exclusive.  Consecutive calls
    therefore never send the same decoder-visible frame to a persistent state
    writer twice.  This plan uses only media metadata and causal timestamps;
    questions, options, labels, categories, and evidence annotations are not
    inputs.
    """

    source = Path(video_path).expanduser()
    if not source.exists() or nframes <= 0 or query_time < 0:
        return None
    if start_time_exclusive is not None and (
        start_time_exclusive < 0 or query_time <= start_time_exclusive
    ):
        raise ValueError("causal interval timestamps must advance strictly")
    video_fps = _get_video_fps(str(source))
    frame_count = _get_video_frame_count(str(source))
    if video_fps <= 0 or frame_count <= 0:
        return None
    end_frame = min(frame_count - 1, max(0, int(query_time * video_fps)))
    start_frame = (
        0
        if start_time_exclusive is None
        else max(0, int(start_time_exclusive * video_fps) + 1)
    )
    if start_frame > end_frame:
        targets: List[int] = []
    elif end_frame - start_frame + 1 <= nframes:
        targets = list(range(start_frame, end_frame + 1))
    else:
        targets = sorted(
            set(np.linspace(start_frame, end_frame, nframes, dtype=int).tolist())
        )
    return {
        "format": "iclr_spatial.causal_interval_frame_plan.v1",
        "video_path": str(source.resolve()),
        "start_time_exclusive": start_time_exclusive,
        "query_time": float(query_time),
        "video_fps": float(video_fps),
        "video_frame_count": int(frame_count),
        "frame_indices": targets,
        "frame_timestamps": [index / video_fps for index in targets],
        "maximum_frame_budget": int(nframes),
        "lower_bound_exclusive": True,
        "causal_prefix_verified": all(index <= end_frame for index in targets),
        "question_or_options_used": False,
    }


def extract_fixed_count_interval(
    video_path: str,
    *,
    start_time_exclusive: Optional[float],
    query_time: float,
    nframes: int = 16,
    frame_size: int = 448,
) -> Tuple[List[Image.Image], Optional[dict]]:
    """Extract the exact pixels declared by :func:`fixed_count_interval_plan`."""

    plan = fixed_count_interval_plan(
        video_path,
        start_time_exclusive=start_time_exclusive,
        query_time=query_time,
        nframes=nframes,
    )
    if plan is None:
        return [], None
    frames = _extract_at_indices(video_path, plan["frame_indices"], frame_size)
    if len(frames) != len(plan["frame_indices"]):
        raise RuntimeError("causal interval plan and decoded frame count differ")
    return frames, plan


def extract_causal_spatial_pyramid(
    video_path: str,
    query_time: float,
    nframes: int = 128,
    recent_window_fps: float = 4.0,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """Extract the fixed causal spatial pyramid used by training and eval."""

    if not Path(video_path).exists() and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        print(f"Warning: Invalid FPS for video: {video_path}")
        return []
    targets = causal_spatial_pyramid_indices(
        max(0, int(query_time * video_fps)),
        video_fps,
        nframes=nframes,
        recent_window_fps=recent_window_fps,
    )
    return _extract_at_indices(video_path, targets, frame_size)


def _extract_at_indices(
    video_path: str,
    target_frames: List[int],
    frame_size: int,
) -> List[Image.Image]:
    """Shared cache→derive→raw→save path used by all fixed-set samplers."""
    target_frames = sorted(set(int(t) for t in target_frames))
    cached = _try_load_from_cache(video_path, target_frames, frame_size)
    if cached is not None:
        return cached
    derived = _try_derive_from_larger_cache(video_path, target_frames, frame_size)
    if derived is not None:
        return derived
    frames = _extract_frames_raw(video_path, target_frames, frame_size)
    if frames:
        _save_to_cache(video_path, target_frames, frames, frame_size)
    return frames


def extract_single_frame_at_query(
    video_path: str,
    query_time: float,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """§4.3.2 policy `single@query`: a single frame closest to query_time."""
    if not Path(video_path).exists() and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    idx = max(0, int(query_time * video_fps))
    return _extract_at_indices(video_path, [idx], frame_size)


def extract_recent_window(
    video_path: str,
    query_time: float,
    nframes: int = 16,
    window_fps: float = 4.0,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """§4.3.2 policy `nearest-Nf@Kfps`: N frames at K fps ending at query_time.

    Default 16 frames at 4 fps → covers the last ~4 seconds before the query.
    Always anchors the last sample at query_time; pads to the start if the
    requested window extends earlier than the video itself.
    """
    if not Path(video_path).exists() and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    end_frame = max(0, int(query_time * video_fps))
    step_frames = max(1, int(round(video_fps / window_fps)))
    start_frame = max(0, end_frame - (nframes - 1) * step_frames)
    target_frames = list(range(start_frame, end_frame + 1, step_frames))
    if len(target_frames) > nframes:
        target_frames = target_frames[-nframes:]
    # Smaller recent budgets are exact suffixes of the frozen recent-8
    # deployment policy. Reuse that already-materialized cache when present;
    # this changes I/O only and preserves every requested frame index.
    superset_nframes = 8
    if len(target_frames) < superset_nframes:
        superset_start = max(
            0, end_frame - (superset_nframes - 1) * step_frames
        )
        superset_targets = list(
            range(superset_start, end_frame + 1, step_frames)
        )
        if len(superset_targets) > superset_nframes:
            superset_targets = superset_targets[-superset_nframes:]
        cached_superset = _try_load_from_cache(
            video_path, superset_targets, frame_size
        )
        if cached_superset is not None and len(cached_superset) == len(
            superset_targets
        ):
            by_index = dict(zip(superset_targets, cached_superset, strict=True))
            if all(index in by_index for index in target_frames):
                subset = [by_index[index] for index in target_frames]
                _save_to_cache(video_path, target_frames, subset, frame_size)
                return subset
    return _extract_at_indices(video_path, target_frames, frame_size)


def extract_hybrid_global_recent(
    video_path: str,
    query_time: float,
    nframes: int = 128,
    recent_frames: int = 16,
    window_fps: float = 4.0,
    frame_size: int = 448,
    history_frame_size: int = 224,
    **_kwargs,
) -> List[Image.Image]:
    """Fixed task-blind low-resolution history plus exact recent pixels.

    The historical channel uniformly covers the causal prefix strictly before
    the recent window. Recent frames remain at source evaluation resolution.
    The schedule depends only on query time and the public global budget.
    """

    if (not Path(video_path).exists() and not _frame_cache_only()) or not 1 <= recent_frames < nframes:
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0 or window_fps <= 0:
        return []
    end_frame = max(0, int(query_time * video_fps))
    step = max(1, int(round(video_fps / window_fps)))
    recent_start = max(0, end_frame - (recent_frames - 1) * step)
    recent_targets = list(range(recent_start, end_frame + 1, step))[-recent_frames:]
    history_budget = nframes - len(recent_targets)
    history_end = max(0, recent_start - 1)
    history_targets = sorted(
        set(np.linspace(0, history_end, history_budget, dtype=int).tolist())
    )
    history = _extract_at_indices(video_path, history_targets, history_frame_size)
    recent = _extract_at_indices(video_path, recent_targets, frame_size)
    return history + recent


def extract_referenced_pair(
    video_path: str,
    query_time: float,
    reference_time: float,
    nframes: int = 16,
    window_fps: float = 4.0,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """Extract dense windows around one visible reference time and query time.

    The timestamp comes from the user-visible question, never from benchmark
    evidence metadata.  Half the budget is centered on the referenced state;
    the other half is a causal window ending at the query boundary.
    """

    plan = referenced_pair_plan(
        video_path,
        query_time=query_time,
        reference_time=reference_time,
        nframes=nframes,
        window_fps=window_fps,
    )
    if plan is None:
        return []
    return _extract_at_indices(video_path, plan["extracted_frame_indices"], frame_size)


def referenced_pair_plan(
    video_path: str,
    query_time: float,
    reference_time: float,
    nframes: int = 16,
    window_fps: float = 4.0,
) -> Optional[dict]:
    """Return the exact causal timestamp-index lookup used by the executor.

    This is model-external state.  It contains no benchmark category, answer,
    evidence annotation, prediction, or score and deterministically generates
    the same frame indices consumed by :func:`extract_referenced_pair`.
    """

    source = Path(video_path).expanduser()
    source_exists = source.exists()
    if not source_exists and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return None
    video_fps = _get_video_fps(video_path)
    # A frozen cache-only evaluator may intentionally omit the original MP4.
    # The referenced-pair plan never reads beyond the causal query boundary,
    # so query_index + 1 is an exact sufficient frame-count clamp.  FPS is
    # loaded from the immutable metadata cache populated with the frame cache.
    frame_count = (
        _get_video_frame_count(video_path)
        if source_exists
        else max(1, int(query_time * video_fps) + 1)
    )
    if video_fps <= 0 or frame_count <= 0 or nframes < 2 or window_fps <= 0:
        return None
    if not 0 <= reference_time <= query_time:
        raise ValueError("reference time must lie inside the causal prefix")
    reference_budget = nframes // 2
    current_budget = nframes - reference_budget
    step = max(1, int(round(video_fps / window_fps)))
    # The query boundary is a hard causal cutoff. Rounding can select the first
    # frame after a fractional query time (e.g. 164.3 s at 3 fps -> frame 493,
    # timestamp 164.333 s). Floor it exactly as the other causal extractors do.
    query_index = min(frame_count - 1, max(0, int(query_time * video_fps)))
    reference_index = min(query_index, max(0, int(round(reference_time * video_fps))))

    before = (reference_budget - 1) // 2
    reference_targets = [
        min(query_index, max(0, reference_index + (offset - before) * step))
        for offset in range(reference_budget)
    ]
    current_start = max(0, query_index - (current_budget - 1) * step)
    current_targets = [current_start + offset * step for offset in range(current_budget)]
    current_targets[-1] = query_index
    targets = sorted(set(reference_targets + current_targets))
    if not targets or max(targets) > query_index:
        raise RuntimeError("temporal dereference escaped the causal prefix")
    resolved_source = source.resolve()
    canonical = str(resolved_source)
    source_stat = resolved_source.stat() if source_exists else None
    return {
        "format": "iclr_spatial.temporal_dereference_state.v1",
        "executor": "explicit_temporal_dereference",
        "status": "ok",
        "canonical_video_path_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "video_size_bytes": int(source_stat.st_size) if source_stat is not None else None,
        "video_mtime_ns": int(source_stat.st_mtime_ns) if source_stat is not None else None,
        "source_metadata_from_cache_only": not source_exists,
        "video_fps": float(video_fps),
        "video_frame_count": int(frame_count),
        "query_time_seconds": float(query_time),
        "reference_time_seconds": float(reference_time),
        "window_fps": float(window_fps),
        "requested_frame_budget": int(nframes),
        "reference_frame_indices": reference_targets,
        "current_frame_indices": current_targets,
        "extracted_frame_indices": targets,
        "extracted_frame_timestamps_seconds": [index / video_fps for index in targets],
        "causal_prefix_verified": True,
        "source_fields": ["visible_question_timestamp", "causal_video_prefix"],
    }


def extract_evidence_only(
    video_path: str,
    query_time: float,
    evidence_times: Optional[List[List[float]]] = None,
    nframes: int = 128,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """§4.3.2 policy `oracle-evidence`: nframes uniformly sampled inside the
    annotated evidence interval(s). Falls back to uniform-N over [0, query_time]
    if no evidence_times are provided (so the policy degrades gracefully on
    annotations missing evidence).
    """
    if not Path(video_path).exists() and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    end_frame = max(1, int(query_time * video_fps))

    # Build a sorted union of evidence intervals (in frame indices). Treat any
    # malformed entry as empty.
    spans: List[Tuple[int, int]] = []
    for span in (evidence_times or []):
        try:
            s_t, e_t = float(span[0]), float(span[1])
        except Exception:
            continue
        s_f = max(0, int(s_t * video_fps))
        e_f = max(s_f, min(end_frame, int(e_t * video_fps)))
        if e_f > s_f:
            spans.append((s_f, e_f))
    if not spans:
        # No evidence available — degrade to uniform-N over full prefix.
        targets = sorted(set(np.linspace(0, end_frame, nframes, dtype=int).tolist()))
        return _extract_at_indices(video_path, targets, frame_size)

    # Allocate the nframes budget proportional to each span's length, but give
    # every span at least 1 frame so very short evidence windows still appear.
    lengths = [e - s for s, e in spans]
    total = sum(lengths) or 1
    raw_alloc = [max(1, round(nframes * L / total)) for L in lengths]
    # Trim back to exactly nframes, removing from the longest span first.
    diff = sum(raw_alloc) - nframes
    while diff > 0:
        i = max(range(len(raw_alloc)), key=lambda k: raw_alloc[k])
        if raw_alloc[i] > 1:
            raw_alloc[i] -= 1
            diff -= 1
        else:
            break

    targets: List[int] = []
    for (s_f, e_f), n in zip(spans, raw_alloc):
        n = max(1, n)
        targets += np.linspace(s_f, e_f, n, dtype=int).tolist()
    return _extract_at_indices(video_path, sorted(set(targets)), frame_size)


def extract_log_decay(
    video_path: str,
    query_time: float,
    nframes: int = 128,
    frame_size: int = 512,
    **_kwargs,
) -> List[Image.Image]:
    """§4.3.2 policy `log-decay-N`: 60% of the budget within last 30 s before
    query_time, 30% in the 30 s – 5 min window, 10% earlier. Each band is
    sampled uniformly in its own range. Falls back gracefully to the longer
    band when the prefix is shorter than the band boundary.
    """
    if not Path(video_path).exists() and not _frame_cache_only():
        print(f"Warning: Video not found: {video_path}")
        return []
    video_fps = _get_video_fps(video_path)
    if video_fps <= 0:
        return []
    end_frame = max(1, int(query_time * video_fps))
    near_start = max(0, end_frame - int(30 * video_fps))      # last 30 s
    mid_start = max(0, end_frame - int(300 * video_fps))      # last 5 min
    far_start = 0

    bands = [
        (near_start, end_frame, 0.60),
        (mid_start, near_start, 0.30),
        (far_start, mid_start, 0.10),
    ]
    targets: List[int] = []
    leftover = nframes
    for i, (s_f, e_f, frac) in enumerate(bands):
        if e_f <= s_f:
            continue
        n_band = nframes - len(targets) if i == len(bands) - 1 else int(round(nframes * frac))
        n_band = min(n_band, leftover)
        if n_band <= 0:
            continue
        targets += np.linspace(s_f, e_f - 1, n_band, dtype=int).tolist()
        leftover -= n_band
        if leftover <= 0:
            break
    targets = sorted(set(targets))
    if not targets:
        targets = [0]
    return _extract_at_indices(video_path, targets, frame_size)


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


# Legacy function for backward compatibility
def extract_frames_for_annotation(
    video_path: str,
    query_times: List[float],
    evidence_times: List[List[float]] = None,
    max_frames: int = 8,
    fps: float = 1.0,
    frame_size: int = 512,
    **kwargs
) -> List[Image.Image]:
    """Legacy wrapper — extracts frames for the first query time only."""
    if not query_times:
        return []
    return extract_frames_for_query(
        video_path, query_times[0], max_frames, fps, frame_size
    )
