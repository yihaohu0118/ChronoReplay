#!/usr/bin/env python3
"""Task-blind, causal frame evaluation for OVO-S.

The model-facing path consumes only the video prefix, question, answer options,
and a frozen visible-language temporal policy. Benchmark level and subcategory
metadata are attached to result rows only after prediction for official
reporting; they never select frames, prompts, token budgets, or answer logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from benchmark.ovo.annotation_utils import load_annotations
from benchmark.ovo.option_utils import cyclically_relabel_options
from benchmark.ovo.score import extract_answer, extract_canonical_json_value
from synthesis.frames import (
    extract_item_frames,
    referenced_time_from_visible_question,
    video_path_for,
)
from synthesis.schema import SynthItem
from synthesis.semantic_frame_policy import (
    select_adaptive_causal_frame_policy,
    select_causal_frame_policy,
    select_temporal_dereference_frame_policy,
    select_uniform_temporal_dereference_frame_policy,
)
from synthesis.spatial_query_compiler import compile_spatial_query
from spatial_witness.causal_workspace import EvidenceRef, WorkspaceError
from spatial_witness.causal_state_snapshots import CausalStateSnapshotStore
from spatial_witness.question_blind_ledger import (
    augment_answer_prompt,
    augment_confident_entity_prompt,
    confident_entity_memory,
    parse as parse_spatial_ledger,
    prompt as spatial_ledger_prompt,
    unavailable as unavailable_spatial_ledger,
)
from spatial_witness.incremental_spatial_ledger import compile_incrementally
from scripts.build_ucs_causal_state_snapshots import writer_implementation_sha256
from synthesis.vlm_inference import (
    build_generate_inputs,
    generate_from_inputs,
    load_model,
    normalize_gt,
    predict,
    predict_batch,
    predict_native_video,
    predict_option_logprobs_batch,
)


@dataclass
class PreparedExample:
    """CPU-prepared model input with post-hoc scoring metadata attached."""

    row: dict[str, Any]
    answers: Any
    images: list[Any]
    prompt: str
    video_frames: list[Any] | None = None
    option_images: list[Any] | None = None
    frame_policy: str = "uniform"
    nframes: int = 0
    frame_policy_reason: str = "fixed_policy"
    new_to_original_option: dict[str, str] | None = None
    option_label_shift: int = 0
    native_video_sample_fps: float | None = None
    harness_state: dict[str, Any] | None = None
    verifier_candidates: tuple[str, str] | None = None
    generate_inputs: dict[str, Any] | None = None


def _load_state_snapshot_store(path: Path) -> CausalStateSnapshotStore:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") in {
        "iclr_spatial.ucs_query_independent_state_builder.v1",
        "iclr_spatial.ucs_query_independent_state_release.v1",
    }:
        value = value.get("snapshot_store")
    if not isinstance(value, Mapping):
        raise WorkspaceError("state snapshot artifact has no snapshot store")
    return CausalStateSnapshotStore.from_manifest(value)


def _extract_writer_update_evidence_frames(
    snapshot_store: CausalStateSnapshotStore,
    *,
    uid: str,
    query_timestamp: float,
    evidence_refs: tuple[EvidenceRef, ...] | list[EvidenceRef],
    source_path: str,
    frame_size: int,
) -> list[Any]:
    """Reload evidence using each writer update's original index set.

    The skip-writer reader must hit the same frame-cache keys the snapshot
    writer used. Those keys are one interval update (up to 16 frames), not a
    re-chunk of the bounded recent packet.
    """

    from bisect import bisect_right

    from benchmark.ovo.utils.frame_utils import _extract_at_indices
    from spatial_witness.causal_workspace import EvidenceRef

    needed = {(ref.frame_index, ref.pixel_sha256) for ref in evidence_refs}
    timestamps = snapshot_store._timestamps.get(uid, [])
    end = bisect_right(timestamps, query_timestamp)
    decoded: dict[tuple[int, str], Any] = {}
    for timestamp in timestamps[:end]:
        snapshot = snapshot_store._snapshots[(uid, timestamp)]
        if not snapshot.writer_invoked or not snapshot.evidence_provenance:
            continue
        group_refs = [
            raw if isinstance(raw, EvidenceRef) else EvidenceRef(**dict(raw))
            for raw in snapshot.evidence_provenance
        ]
        if not any((ref.frame_index, ref.pixel_sha256) in needed for ref in group_refs):
            continue
        indices = [ref.frame_index for ref in group_refs]
        frames = _extract_at_indices(source_path, indices, frame_size)
        unique = sorted(set(indices))
        if len(frames) != len(unique):
            raise WorkspaceError(
                "persistent state writer-update decode is incomplete "
                f"video_uid={uid} query_timestamp={query_timestamp} "
                f"update_timestamp={timestamp}"
            )
        by_index = dict(zip(unique, frames, strict=True))
        for ref in group_refs:
            decoded[(ref.frame_index, ref.pixel_sha256)] = by_index[ref.frame_index]
    frames = []
    for ref in evidence_refs:
        image = decoded.get((ref.frame_index, ref.pixel_sha256))
        if image is None:
            raise WorkspaceError(
                "persistent state evidence missing writer-update pixels "
                f"video_uid={uid} query_timestamp={query_timestamp} "
                f"frame_index={ref.frame_index}"
            )
        frames.append(image)
    return frames


def _snapshot_key(row: Mapping[str, Any]) -> tuple[str, float]:
    uid = str(row.get("video_uid") or row.get("video_id") or "").strip()
    if not uid:
        uid = Path(str(row.get("video_path") or "")).stem
    raw_times = row.get("query_times")
    if hasattr(raw_times, "tolist"):
        raw_times = raw_times.tolist()
    if not raw_times:
        raw_times = [row.get("query_timestamp", row.get("question_timestamp"))]
    if not uid or any(value is None for value in raw_times):
        raise WorkspaceError("evaluation row has no causal snapshot key")
    return uid, max(float(value) for value in raw_times)


def _plain_options(value: Any) -> dict[str, Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return value
    return dict(value or [])


def _result_payload(
    results: list[dict[str, Any]], skipped: int, *, frame_policy: str
) -> dict[str, Any]:
    correct = sum(int(row["correct"]) for row in results)
    policy_counts: dict[str, int] = defaultdict(int)
    policy_budget_counts: dict[str, int] = defaultdict(int)
    media_serialization_counts: dict[str, int] = defaultdict(int)
    for row in results:
        resolved = str(row.get("frame_policy") or frame_policy)
        policy_counts[resolved] += 1
        policy_budget_counts[f"{resolved}:{int(row.get('nframes') or 0)}"] += 1
        media_serialization_counts[
            str(row.get("media_serialization") or "ordered_image_blocks")
        ] += 1
    return {
        "summary": {
            "correct": correct,
            "total": len(results),
            "skipped_missing_video": skipped,
            "accuracy": correct / max(len(results), 1),
            "frame_sampling_protocol": f"{frame_policy}_fixed_causal",
            "frame_policy": frame_policy,
            "resolved_frame_policy_counts": dict(sorted(policy_counts.items())),
            "resolved_policy_budget_counts": dict(
                sorted(policy_budget_counts.items())
            ),
            "actor_metadata_fields": [],
            "observed_media_serializations": dict(
                sorted(media_serialization_counts.items())
            ),
        },
        "results": results,
    }


def _evaluation_ids(path: Path) -> list[str]:
    """Read an explicit diagnostic subset without exposing IDs to the actor."""

    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(value, Mapping):
        if isinstance(value.get("results"), list):
            value = [row.get("id") for row in value["results"]]
        else:
            value = value.get("ids") or value.get("evaluation_ids")
    if not isinstance(value, list) or not value:
        raise ValueError("evaluation ID file must contain a non-empty list")
    ids = [str(item or "").strip() for item in value]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("evaluation ID file contains empty or duplicate IDs")
    return ids


def _select_evaluation_ids(
    rows: list[dict[str, Any]], ids: list[str]
) -> list[dict[str, Any]]:
    by_id = {str(row.get("id") or ""): row for row in rows}
    missing = [uid for uid in ids if uid not in by_id]
    if missing:
        raise ValueError(f"evaluation IDs absent from annotation: {missing[:5]}")
    return [by_id[uid] for uid in ids]


def _candidate_prediction_map(path: Path) -> dict[str, str]:
    """Load only original-label predictions from one frozen actor run."""

    files = [path] if path.is_file() else sorted(path.glob("lane*/predictions.json"))
    if not files and path.is_dir():
        files = sorted(path.glob("predictions.json"))
    if not files:
        raise ValueError(f"candidate run has no prediction payload: {path}")
    output: dict[str, str] = {}
    for filename in files:
        value = json.loads(filename.read_text(encoding="utf-8"))
        rows = value.get("results") if isinstance(value, Mapping) else value
        if not isinstance(rows, list):
            raise ValueError(f"candidate payload has no result rows: {filename}")
        for row in rows:
            uid = str(row.get("id") or "").strip()
            prediction = str(row.get("pred") or "").strip().upper()
            if not uid or prediction not in set("ABCDEFG"):
                raise ValueError(f"invalid candidate prediction in {filename}: {uid}")
            if row.get("prediction_mapped_to_original_labels") is not True:
                raise ValueError(f"candidate prediction is not in original labels: {uid}")
            if uid in output and output[uid] != prediction:
                raise ValueError(f"conflicting duplicate candidate prediction: {uid}")
            output[uid] = prediction
    return output


def _interleaved_shard(
    rows: list[dict[str, Any]], *, rank: int, world_size: int
) -> list[dict[str, Any]]:
    """Partition an ordered evaluation set without contiguous cost skew.

    Long-video evaluation cost is strongly row-dependent.  Contiguous shards
    can therefore leave one GPU idle while another finishes a cluster of long
    prefixes.  A deterministic round-robin partition preserves the exact row
    set and per-row protocol while distributing neighbouring costs across
    workers.
    """

    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid interleaved shard rank/world size")
    return rows[rank::world_size]


def _prepare_example(
    row: dict[str, Any],
    *,
    video_base: Path,
    nframes: int,
    frame_size: int,
    prompt_style: str,
    frame_policy: str = "uniform",
    recent_window_fps: float = 4.0,
    semantic_uniform_nframes: int = 128,
    semantic_recent_nframes: int = 16,
    semantic_reference_nframes: int = 8,
    hybrid_recent_nframes: int = 16,
    option_label_shift: int = 0,
    verifier_candidates_by_id: Mapping[str, tuple[str, str]] | None = None,
) -> PreparedExample:
    """Load one example's exact frozen frame set on CPU."""

    original_options = _plain_options(row["options"])
    options, new_to_original = cyclically_relabel_options(
        original_options, option_label_shift
    )
    query_times = row["query_times"]
    if hasattr(query_times, "tolist"):
        query_times = query_times.tolist()
    query_time = max(float(value) for value in query_times)
    answers = row["answers"]
    if hasattr(answers, "tolist"):
        answers = answers.tolist()
    item = SynthItem(
        video_path=str(row["video_path"]),
        question=str(row["question"]),
        options=options,
        query_times=list(query_times),
        answers=answers,
    )
    resolved_policy = frame_policy
    resolved_reason = "fixed_policy"
    effective_nframes = nframes
    harness_state = None
    harness_executor_started = None
    if frame_policy in {
        "semantic_current",
        "semantic_adaptive",
        "semantic_dereference",
        "uniform_temporal_dereference",
    }:
        visible_question = str(row.get("question") or "")
        if frame_policy in {
            "semantic_dereference", "uniform_temporal_dereference"
        }:
            compiled = compile_spatial_query(visible_question, original_options)
            selector = (
                select_uniform_temporal_dereference_frame_policy
                if frame_policy == "uniform_temporal_dereference"
                else select_temporal_dereference_frame_policy
            )
            resolved_policy, resolved_reason = selector(
                visible_question,
                {str(key): str(value) for key, value in original_options.items()},
            )
            harness_state = {
                "format": "iclr_spatial.causal_state_harness_trace.v1",
                "compiled_query": compiled.to_dict(),
                "executor_status": "fallback_no_promoted_executor",
                "executor_result": None,
                "benchmark_metadata_actor_visible": False,
                "external_generative_model_or_api": False,
            }
        else:
            selector = (
                select_adaptive_causal_frame_policy
                if frame_policy == "semantic_adaptive"
                else select_causal_frame_policy
            )
            resolved_policy, resolved_reason = selector(visible_question)
        effective_nframes = {
            "recent": semantic_recent_nframes,
            "referenced_pair": semantic_reference_nframes,
        }.get(resolved_policy, semantic_uniform_nframes)
        if (
            frame_policy in {
                "semantic_dereference", "uniform_temporal_dereference"
            }
            and resolved_policy == "referenced_pair"
        ):
            from benchmark.ovo.utils.frame_utils import referenced_pair_plan

            harness_executor_started = time.perf_counter()
            reference_time = referenced_time_from_visible_question(
                visible_question, query_time
            )
            executor_result = referenced_pair_plan(
                str(video_path_for(item, str(video_base))),
                query_time=query_time,
                reference_time=reference_time,
                nframes=effective_nframes,
                window_fps=recent_window_fps,
            )
            if executor_result is None:
                raise RuntimeError("promoted temporal dereference executor unavailable")
            harness_state["executor_status"] = "ok"
            harness_state["executor_result"] = executor_result
    frames, prompt, option_images = extract_item_frames(
        item,
        str(video_base),
        nframes=effective_nframes,
        frame_size=frame_size,
        prompt_style=prompt_style,
        frame_policy=resolved_policy,
        recent_window_fps=recent_window_fps,
        hybrid_recent_nframes=hybrid_recent_nframes,
    )
    verifier_candidates = None
    if verifier_candidates_by_id is not None:
        uid = str(row.get("id") or "")
        if uid not in verifier_candidates_by_id:
            raise ValueError(f"candidate verifier lacks predictions for {uid}")
        verifier_candidates = verifier_candidates_by_id[uid]
        first, second = verifier_candidates
        prompt += (
            "\n\nTwo independent causal-evidence inspections proposed choices "
            f"{first} and {second}. Treat both as untrusted candidates. "
            "Re-check the visible evidence, reject unsupported suggestions, "
            "and respond with only the single best option letter."
        )
    if (
        harness_state is not None
        and harness_state.get("executor_status") == "ok"
        and isinstance(harness_state.get("executor_result"), dict)
    ):
        frame_hashes = []
        for image in frames:
            digest = hashlib.sha256()
            digest.update(str(getattr(image, "mode", "")).encode())
            digest.update(str(getattr(image, "size", "")).encode())
            digest.update(image.tobytes())
            frame_hashes.append(digest.hexdigest())
        state = harness_state["executor_result"]
        indices = state.get("extracted_frame_indices")
        if not isinstance(indices, list) or len(indices) != len(frame_hashes):
            raise RuntimeError("executor plan and extracted evidence frame count differ")
        state["evidence_frame_sha256"] = frame_hashes
        state["evidence_hash_protocol"] = "sha256(pil_mode||pil_size||pixel_bytes)"
        if harness_executor_started is None:
            raise RuntimeError("executor timing origin is missing")
        state["executor_wall_time_ms"] = 1000.0 * (
            time.perf_counter() - harness_executor_started
        )
    images = frames + (
        [image for _, image in option_images] if option_images else []
    )
    native_video_sample_fps = None
    if resolved_policy == "uniform" and query_time > 0 and effective_nframes > 1:
        # The in-memory native-video path exposes sequential frame indices
        # 0..N-1. Qwen3-VL renders timestamp i / raw_fps, hence (N-1)/T maps
        # the last uniformly sampled causal frame back to the query boundary.
        native_video_sample_fps = (effective_nframes - 1) / query_time
    elif resolved_policy in {"recent", "referenced_pair"}:
        native_video_sample_fps = float(recent_window_fps)
    return PreparedExample(
        row=row,
        answers=answers,
        images=images,
        prompt=prompt,
        video_frames=frames,
        option_images=(
            [image for _, image in option_images] if option_images else []
        ),
        frame_policy=resolved_policy,
        nframes=effective_nframes,
        frame_policy_reason=resolved_reason,
        new_to_original_option=new_to_original,
        option_label_shift=option_label_shift % len(options),
        native_video_sample_fps=native_video_sample_fps,
        harness_state=harness_state,
        verifier_candidates=verifier_candidates,
    )


def _prefetched_examples(
    rows: list[dict[str, Any]],
    *,
    video_base: Path,
    nframes: int,
    frame_size: int,
    prompt_style: str,
    frame_policy: str = "uniform",
    recent_window_fps: float = 4.0,
    semantic_uniform_nframes: int = 128,
    semantic_recent_nframes: int = 16,
    semantic_reference_nframes: int = 8,
    hybrid_recent_nframes: int = 16,
    option_label_shift: int = 0,
    verifier_candidates_by_id: Mapping[str, tuple[str, str]] | None = None,
) -> Iterator[tuple[int, PreparedExample]]:
    """Overlap bounded CPU frame I/O with current GPU inference.

    The worker count and bounded look-ahead depth are controlled by
    ``OVOS_EVAL_FRAME_PREFETCH_WORKERS`` and
    ``OVOS_EVAL_FRAME_PREFETCH_DEPTH``.  Futures are consumed in input order,
    so this changes scheduling only—not selected frames, prompts, or scores.
    """

    if not rows:
        return

    def submit(pool: ThreadPoolExecutor, row: dict[str, Any]):
        return pool.submit(
            _prepare_example,
            row,
            video_base=video_base,
            nframes=nframes,
            frame_size=frame_size,
            prompt_style=prompt_style,
            frame_policy=frame_policy,
            recent_window_fps=recent_window_fps,
            semantic_uniform_nframes=semantic_uniform_nframes,
            semantic_recent_nframes=semantic_recent_nframes,
            semantic_reference_nframes=semantic_reference_nframes,
            hybrid_recent_nframes=hybrid_recent_nframes,
            option_label_shift=option_label_shift,
            verifier_candidates_by_id=verifier_candidates_by_id,
        )

    workers = max(1, int(os.getenv("OVOS_EVAL_FRAME_PREFETCH_WORKERS", "1")))
    depth = max(
        workers,
        int(os.getenv("OVOS_EVAL_FRAME_PREFETCH_DEPTH", str(workers))),
    )
    depth = min(depth, len(rows))
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="ovo-frame-prefetch"
    ) as pool:
        pending = {
            row_index: submit(pool, rows[row_index - 1])
            for row_index in range(1, depth + 1)
        }
        next_submit = depth + 1
        for row_index in range(1, len(rows) + 1):
            prepared = pending.pop(row_index).result()
            if next_submit <= len(rows):
                pending[next_submit] = submit(pool, rows[next_submit - 1])
                next_submit += 1
            yield row_index, prepared


def _prepared_examples(
    rows: list[dict[str, Any]],
    *,
    video_base: Path,
    nframes: int,
    frame_size: int,
    prompt_style: str,
    prefetch: bool,
    frame_policy: str = "uniform",
    recent_window_fps: float = 4.0,
    semantic_uniform_nframes: int = 128,
    semantic_recent_nframes: int = 16,
    semantic_reference_nframes: int = 8,
    hybrid_recent_nframes: int = 16,
    option_label_shift: int = 0,
    verifier_candidates_by_id: Mapping[str, tuple[str, str]] | None = None,
) -> Iterator[tuple[int, PreparedExample]]:
    """Select the overlap path while retaining an exact serial fallback."""

    kwargs = {
        "video_base": video_base,
        "nframes": nframes,
        "frame_size": frame_size,
        "prompt_style": prompt_style,
        "frame_policy": frame_policy,
        "recent_window_fps": recent_window_fps,
        "semantic_uniform_nframes": semantic_uniform_nframes,
        "semantic_recent_nframes": semantic_recent_nframes,
        "semantic_reference_nframes": semantic_reference_nframes,
        "hybrid_recent_nframes": hybrid_recent_nframes,
        "option_label_shift": option_label_shift,
        "verifier_candidates_by_id": verifier_candidates_by_id,
    }
    if prefetch:
        yield from _prefetched_examples(rows, **kwargs)
        return
    for row_index, row in enumerate(rows, 1):
        yield row_index, _prepare_example(row, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--processor",
        default=None,
        help=(
            "processor/tokenizer source; frozen benchmark launchers pin this "
            "to the official base independently of checkpoint metadata"
        ),
    )
    parser.add_argument("--adapter")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--video-base", required=True)
    parser.add_argument("--nframes", type=int, default=128)
    parser.add_argument("--frame-size", type=int, default=448)
    parser.add_argument(
        "--media-serialization",
        choices=("ordered_image_blocks", "native_video"),
        default="ordered_image_blocks",
        help=(
            "Fixed serialization for every query. native_video matches the "
            "upstream Qwen OVO-S wrapper without changing selected pixels, "
            "prompt text, or answer logic."
        ),
    )
    parser.add_argument("--native-video-fps", type=float, default=2.0)
    parser.add_argument(
        "--frame-policy",
        choices=(
            "uniform",
            "uniform_subsample128",
            "cyclic",
            "recent",
            "log_decay",
            "causal_pyramid",
            "referenced_pair",
            "semantic_current",
            "semantic_adaptive",
            "semantic_dereference",
            "uniform_temporal_dereference",
            "hybrid_global_recent",
        ),
        default="uniform",
        help=(
            "Causal frame allocation policy. causal_pyramid is one fixed, "
            "question-blind mix of global, history, and recent frames."
        ),
    )
    parser.add_argument(
        "--recent-window-fps",
        type=float,
        default=4.0,
        help="Sampling rate used only by --frame-policy recent.",
    )
    parser.add_argument(
        "--semantic-uniform-nframes",
        type=int,
        default=128,
        help="Global-prefix budget for semantic_current questions.",
    )
    parser.add_argument(
        "--semantic-recent-nframes",
        type=int,
        default=16,
        help="Recent-window budget for semantic_current questions.",
    )
    parser.add_argument(
        "--semantic-reference-nframes",
        type=int,
        default=8,
        help="Per-pair frame budget for the temporal-dereference executor.",
    )
    parser.add_argument(
        "--hybrid-recent-nframes",
        type=int,
        default=16,
        help=(
            "Question-independent exact-recent allocation inside the fixed "
            "hybrid global/recent budget."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        choices=("default", "verbose", "cot", "spatial_state", "l4_map"),
        default="default",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--state-compiler",
        choices=(
            "none",
            "question_blind_ledger",
            "question_blind_incremental_ledger",
        ),
        default="none",
        help="Optional same-checkpoint two-pass state compiler; default is model-only.",
    )
    parser.add_argument("--state-compiler-max-new-tokens", type=int, default=512)
    parser.add_argument("--state-compiler-chunk-frames", type=int, default=16)
    parser.add_argument("--state-compiler-overlap-frames", type=int, default=4)
    parser.add_argument(
        "--state-snapshot-manifest",
        type=Path,
        help=(
            "Query-independent persistent state built once per video prefix. "
            "This is mutually exclusive with per-question state compilation."
        ),
    )
    parser.add_argument(
        "--state-packet-mode",
        choices=(
            "evidence_only",
            "state_and_evidence",
            "confident_entity_memory",
        ),
        default="state_and_evidence",
        help="Matched snapshot packet arm; both modes receive identical pixels.",
    )
    parser.add_argument(
        "--state-evidence-frames",
        type=int,
        default=32,
        help="Recent cumulative writer-provenance frames exposed in either arm.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size; batch=1 retains the reference path.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--shard-rank",
        type=int,
        default=0,
        help="Deterministic interleaved shard rank, applied before offset/limit.",
    )
    parser.add_argument(
        "--shard-world-size",
        type=int,
        default=1,
        help="Number of deterministic interleaved evaluator shards.",
    )
    parser.add_argument(
        "--id-file",
        type=Path,
        help=(
            "Optional explicit diagnostic subset. Accepts a JSON ID list, an "
            "evaluation result JSON, or one ID per line. IDs affect only row "
            "selection and are never included in the model prompt."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--save-raw-reply", action="store_true")
    parser.add_argument(
        "--answer-parser",
        choices=("native_letter", "canonical_json_value"),
        default="native_letter",
        help=(
            "Frozen output reader. canonical_json_value accepts only the "
            "single-field JSON ABI used by the common-interface control."
        ),
    )
    parser.add_argument(
        "--decode-mode",
        choices=("generate", "option_logprob"),
        default="generate",
        help=(
            "Greedy text generation, or answer-blind first-token scoring over "
            "the visible option labels."
        ),
    )
    parser.add_argument(
        "--option-label-shift",
        type=int,
        default=0,
        help="Fixed cyclic option-content shift; predictions are mapped back.",
    )
    parser.add_argument(
        "--verifier-candidate-run",
        action="append",
        type=Path,
        default=[],
        help=(
            "Exactly two same-checkpoint prediction files/directories. The actor "
            "sees only their original-label candidate answers, never correctness, "
            "task IDs, benchmark categories, or scores."
        ),
    )
    args = parser.parse_args()
    if len(args.verifier_candidate_run) not in {0, 2}:
        raise SystemExit("candidate verifier requires exactly two actor runs")
    if args.verifier_candidate_run and (
        args.decode_mode != "generate" or args.option_label_shift != 0
    ):
        raise SystemExit(
            "candidate verifier requires generate decoding and original option labels"
        )
    if min(
        args.nframes,
        args.max_new_tokens,
        args.state_compiler_max_new_tokens,
        args.state_compiler_chunk_frames,
        args.state_evidence_frames,
        args.batch_size,
        args.recent_window_fps,
        args.semantic_uniform_nframes,
        args.semantic_recent_nframes,
        args.semantic_reference_nframes,
        args.hybrid_recent_nframes,
    ) < 1:
        raise SystemExit(
            "all frame budgets, token budgets, and batch size must be positive"
        )
    if not 0 <= args.state_compiler_overlap_frames < args.state_compiler_chunk_frames:
        raise SystemExit(
            "state compiler overlap must be nonnegative and smaller than its chunk"
        )
    if args.native_video_fps < 0:
        raise SystemExit("native-video fps must be nonnegative (0 means causal-span auto)")
    if (
        args.shard_world_size < 1
        or args.shard_rank < 0
        or args.shard_rank >= args.shard_world_size
    ):
        raise SystemExit("invalid --shard-rank/--shard-world-size")
    if (
        args.media_serialization == "native_video"
        and args.native_video_fps == 0
        and args.frame_policy != "uniform"
    ):
        raise SystemExit(
            "causal-span native fps currently requires the global uniform frame policy"
        )
    if args.media_serialization == "native_video" and args.decode_mode != "generate":
        raise SystemExit("native-video serialization currently requires generate decoding")
    if args.state_compiler != "none" and (
        args.media_serialization != "ordered_image_blocks"
        or args.decode_mode != "generate"
        or args.batch_size != 1
    ):
        raise SystemExit(
            "question-blind state compilation requires ordered images, generate decoding, and batch size 1"
        )
    if args.state_snapshot_manifest is not None and args.state_compiler != "none":
        raise SystemExit(
            "persistent state snapshots and per-question state compilation are mutually exclusive"
        )
    if args.state_snapshot_manifest is not None and (
        args.media_serialization != "ordered_image_blocks"
        or args.decode_mode != "generate"
    ):
        raise SystemExit(
            "persistent state snapshots require ordered images and generate decoding"
        )
    cpu_threads = max(1, int(os.getenv("OVOS_EVAL_CPU_THREADS", "8")))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        import cv2

        cv2.setNumThreads(1)
    except (ImportError, AttributeError):
        pass

    rows = load_annotations(args.annotation)
    if args.id_file is not None:
        rows = _select_evaluation_ids(rows, _evaluation_ids(args.id_file))
    verifier_candidates_by_id = None
    if args.verifier_candidate_run:
        first = _candidate_prediction_map(args.verifier_candidate_run[0])
        second = _candidate_prediction_map(args.verifier_candidate_run[1])
        requested_ids = {str(row.get("id") or "") for row in rows}
        missing_first = sorted(requested_ids - set(first))
        missing_second = sorted(requested_ids - set(second))
        if missing_first or missing_second:
            raise ValueError(
                "candidate verifier coverage mismatch: "
                f"first={missing_first[:5]} second={missing_second[:5]}"
            )
        verifier_candidates_by_id = {
            uid: (first[uid], second[uid]) for uid in requested_ids
        }
    rows = _interleaved_shard(
        rows, rank=args.shard_rank, world_size=args.shard_world_size
    )
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[: args.limit]

    snapshot_store = None
    if args.state_snapshot_manifest is not None:
        snapshot_store = _load_state_snapshot_store(args.state_snapshot_manifest)
        expected_writer = writer_implementation_sha256(
            args.model, args.processor or args.model
        )
        if snapshot_store.writer_implementation_sha256 != expected_writer:
            raise SystemExit(
                "persistent state snapshots were not written by this model/processor"
            )

    video_base = Path(args.video_base)
    out_path = Path(args.out)
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    results: list[dict[str, Any]] = []
    skipped = 0
    if args.resume and partial_path.is_file():
        previous = json.loads(partial_path.read_text())
        results = list(previous.get("results") or [])
        skipped = int((previous.get("summary") or {}).get("skipped_missing_video", 0))
        completed = {str(row["id"]) for row in results}
        rows = [row for row in rows if str(row.get("id")) not in completed]

    def write_partial() -> None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = partial_path.with_suffix(partial_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                _result_payload(results, skipped, frame_policy=args.frame_policy)
            )
        )
        tmp.replace(partial_path)

    processor_path = str(Path(args.processor or args.model).resolve())
    model, processor = load_model(
        args.model,
        args.adapter,
        processor_path=processor_path,
    )
    by_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    prefetch_enabled = os.getenv("OVOS_EVAL_FRAME_PREFETCH", "1").strip().lower() \
        not in {"0", "false", "no", "off", "disabled"}
    prepared_iterator = _prepared_examples(
        rows,
        video_base=video_base,
        nframes=args.nframes,
        frame_size=args.frame_size,
        prompt_style=args.prompt_style,
        prefetch=prefetch_enabled,
        frame_policy=args.frame_policy,
        recent_window_fps=args.recent_window_fps,
        semantic_uniform_nframes=args.semantic_uniform_nframes,
        semantic_recent_nframes=args.semantic_recent_nframes,
        semantic_reference_nframes=args.semantic_reference_nframes,
        hybrid_recent_nframes=args.hybrid_recent_nframes,
        option_label_shift=args.option_label_shift,
        verifier_candidates_by_id=verifier_candidates_by_id,
    )
    batch: list[tuple[int, PreparedExample]] = []

    def run_batch(values: list[tuple[int, PreparedExample]]) -> None:
        nonlocal skipped
        started = time.time()
        valid = [
            (index, item)
            for index, item in values
            if (
                item.video_frames
                if args.media_serialization == "native_video"
                else item.images
            )
        ]
        skipped += len(values) - len(valid)
        if not valid:
            return
        option_score_rows: list[dict[str, float]] | None = None
        if snapshot_store is not None:
            from benchmark.ovo.utils.frame_utils import _extract_frames_raw

            prompts = []
            for _, item in valid:
                uid, query_timestamp = _snapshot_key(item.row)
                snapshot = snapshot_store.read_exact(
                    video_uid=uid, query_timestamp=query_timestamp
                )
                evidence_refs = snapshot_store.evidence_history(
                    video_uid=uid,
                    at_or_before=query_timestamp,
                    maximum_refs=args.state_evidence_frames,
                )
                if not evidence_refs:
                    raise WorkspaceError("persistent state packet has no evidence")
                evidence_refs = tuple(sorted(
                    evidence_refs, key=lambda value: value.frame_index
                ))
                source_path = video_path_for(
                    SynthItem(video_path=str(item.row["video_path"])),
                    str(video_base),
                )
                # Re-decode the writer indices. Skip-writer UCS must reuse each
                # original interval-update cache key; a re-chunk of the recent
                # packet misses those keys on short or multi-update prefixes.
                if os.getenv("UCS_MATCH_WRITER_FRAME_CACHE", "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    state_frames = _extract_writer_update_evidence_frames(
                        snapshot_store,
                        uid=uid,
                        query_timestamp=query_timestamp,
                        evidence_refs=evidence_refs,
                        source_path=str(source_path),
                        frame_size=args.frame_size,
                    )
                else:
                    state_frames = _extract_frames_raw(
                        str(source_path),
                        [value.frame_index for value in evidence_refs],
                        args.frame_size,
                    )
                if len(state_frames) != len(evidence_refs):
                    raise WorkspaceError(
                        "persistent state evidence packet decode is incomplete "
                        f"video_uid={uid} query_timestamp={query_timestamp}"
                    )
                observed_hashes = []
                for image in state_frames:
                    digest = hashlib.sha256()
                    digest.update(str(getattr(image, "mode", "")).encode())
                    digest.update(str(getattr(image, "size", "")).encode())
                    digest.update(image.tobytes())
                    observed_hashes.append(digest.hexdigest())
                expected_hashes = [value.pixel_sha256 for value in evidence_refs]
                if observed_hashes != expected_hashes:
                    raise WorkspaceError(
                        "persistent state evidence pixels changed after snapshot write "
                        f"video_uid={uid} query_timestamp={query_timestamp} "
                        f"n={len(expected_hashes)}"
                    )
                item.video_frames = state_frames
                item.images = state_frames + list(item.option_images or [])
                item.nframes = len(state_frames)
                item.frame_policy = "persistent_state_provenance"
                item.frame_policy_reason = "bounded_recent_writer_evidence"
                reader_state = (
                    confident_entity_memory(snapshot.state)
                    if args.state_packet_mode == "confident_entity_memory"
                    else snapshot.state
                )
                item.harness_state = {
                    "format": "iclr_spatial.persistent_state_reader_trace.v1",
                    "state_compiler": "query_independent_persistent_snapshots",
                    "same_checkpoint": True,
                    "answer_model_calls": 1,
                    "per_question_writer_calls": 0,
                    "question_or_options_visible_to_writer": False,
                    "external_generative_model_or_api": False,
                    "snapshot_key": [uid, query_timestamp],
                    "snapshot_state_version": snapshot.state_version,
                    "snapshot_state_sha256": snapshot.state_sha256,
                    "snapshot_prior_state_sha256": snapshot.prior_state_sha256,
                    "snapshot_writer_invoked": snapshot.writer_invoked,
                    "packet_mode": args.state_packet_mode,
                    "matched_evidence_frame_count": len(evidence_refs),
                    "matched_evidence_provenance": [
                        value.to_dict() for value in evidence_refs
                    ],
                    "ledger": reader_state if args.state_packet_mode != "evidence_only" else None,
                    "state_projection_applied": (
                        args.state_packet_mode != "confident_entity_memory"
                        or reader_state.get("status") == "available"
                    ),
                }
                prompts.append(
                    augment_answer_prompt(item.prompt, snapshot.state)
                    if args.state_packet_mode == "state_and_evidence"
                    else augment_confident_entity_prompt(item.prompt, reader_state)
                    if args.state_packet_mode == "confident_entity_memory"
                    else item.prompt
                )
            if len(valid) == 1:
                replies = [
                    predict(
                        model,
                        processor,
                        valid[0][1].images,
                        prompts[0],
                        max_new_tokens=args.max_new_tokens,
                    )
                ]
            else:
                replies = predict_batch(
                    model,
                    processor,
                    [item.images for _, item in valid],
                    prompts,
                    max_new_tokens=args.max_new_tokens,
                )
        elif args.state_compiler in {
            "question_blind_ledger",
            "question_blind_incremental_ledger",
        }:
            _, item = valid[0]
            visual_frames = item.video_frames or []
            state_started = time.perf_counter()
            incremental_trace = None
            if args.state_compiler == "question_blind_incremental_ledger":
                incremental_trace = compile_incrementally(
                    visual_frames,
                    lambda chunk, compiler_prompt: predict(
                        model,
                        processor,
                        list(chunk),
                        compiler_prompt,
                        max_new_tokens=args.state_compiler_max_new_tokens,
                    ),
                    chunk_frames=args.state_compiler_chunk_frames,
                    overlap_frames=args.state_compiler_overlap_frames,
                )
                ledger = incremental_trace["ledger"]
                compiler_calls = int(incremental_trace["compiler_calls"])
            else:
                raw_state = predict(
                    model,
                    processor,
                    visual_frames,
                    spatial_ledger_prompt(len(visual_frames)),
                    max_new_tokens=args.state_compiler_max_new_tokens,
                )
                try:
                    ledger = parse_spatial_ledger(
                        raw_state, frame_count=len(visual_frames)
                    )
                except WorkspaceError as error:
                    ledger = unavailable_spatial_ledger(
                        raw_state, f"invalid_model_output:{type(error).__name__}"
                    )
                compiler_calls = 1
            evidence_hashes = []
            for image in visual_frames:
                digest = hashlib.sha256()
                digest.update(str(getattr(image, "mode", "")).encode())
                digest.update(str(getattr(image, "size", "")).encode())
                digest.update(image.tobytes())
                evidence_hashes.append(digest.hexdigest())
            item.harness_state = {
                "format": "iclr_spatial.same_checkpoint_state_compiler_trace.v1",
                "state_compiler": args.state_compiler,
                "same_checkpoint": True,
                "total_model_calls": compiler_calls + 1,
                "compiler_model_calls": compiler_calls,
                "answer_model_calls": 1,
                "compiler_max_new_tokens": args.state_compiler_max_new_tokens,
                "compiler_chunk_frames": (
                    args.state_compiler_chunk_frames
                    if incremental_trace is not None else None
                ),
                "compiler_overlap_frames": (
                    args.state_compiler_overlap_frames
                    if incremental_trace is not None else None
                ),
                "answer_max_new_tokens": args.max_new_tokens,
                "question_or_options_visible_to_compiler": False,
                "external_generative_model_or_api": False,
                "frame_policy": item.frame_policy,
                "evidence_frames": len(visual_frames),
                "evidence_frame_sha256": evidence_hashes,
                "evidence_hash_protocol": "sha256(pil_mode||pil_size||pixel_bytes)",
                "ledger": ledger,
                "incremental_trace": incremental_trace,
                "compiler_wall_time_ms": 1000.0 * (
                    time.perf_counter() - state_started
                ),
            }
            replies = [
                predict(
                    model,
                    processor,
                    item.images,
                    augment_answer_prompt(item.prompt, ledger),
                    max_new_tokens=args.max_new_tokens,
                )
            ]
        elif args.media_serialization == "native_video":
            native_fps_values = []
            for _, item in valid:
                fps = (
                    float(args.native_video_fps)
                    if args.native_video_fps > 0
                    else item.native_video_sample_fps
                )
                if fps is None or fps <= 0:
                    raise RuntimeError(
                        "native-video causal-span FPS is undefined for the resolved policy"
                    )
                native_fps_values.append(float(fps))
            replies = [
                predict_native_video(
                    model,
                    processor,
                    item.video_frames or [],
                    item.prompt,
                    option_images=item.option_images,
                    fps=fps,
                    max_new_tokens=args.max_new_tokens,
                )
                for ((_, item), fps) in zip(valid, native_fps_values, strict=True)
            ]
        elif args.decode_mode == "option_logprob":
            option_score_rows = predict_option_logprobs_batch(
                model,
                processor,
                [item.images for _, item in valid],
                [item.prompt for _, item in valid],
                [
                    sorted((item.new_to_original_option or {}).keys())
                    for _, item in valid
                ],
            )
            replies = [max(scores, key=scores.get) for scores in option_score_rows]
        elif all(item.generate_inputs is not None for _, item in valid):
            replies = [
                generate_from_inputs(
                    model,
                    processor,
                    item.generate_inputs,
                    max_new_tokens=args.max_new_tokens,
                )
                for _, item in valid
            ]
        elif len(valid) == 1:
            replies = [
                predict(
                    model,
                    processor,
                    valid[0][1].images,
                    valid[0][1].prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            ]
        else:
            replies = predict_batch(
                model,
                processor,
                [item.images for _, item in valid],
                [item.prompt for _, item in valid],
                max_new_tokens=args.max_new_tokens,
            )
        for batch_index, ((row_index, prepared), reply) in enumerate(
            zip(valid, replies, strict=True)
        ):
            row = prepared.row
            answers = prepared.answers
            permuted_prediction = (
                extract_canonical_json_value(reply)
                if args.answer_parser == "canonical_json_value"
                else extract_answer(reply)
            )
            prediction = (prepared.new_to_original_option or {}).get(
                permuted_prediction, permuted_prediction
            )
            raw_gt = answers[0] if isinstance(answers, (list, tuple)) else answers
            gt = normalize_gt(raw_gt)
            task = str(row.get("task_subcategory") or "")
            correct = prediction == gt
            result = {
                "id": row.get("id"),
                "task": task,
                "level": int(row.get("level") or 0),
                "gt": gt,
                "pred": prediction,
                "correct": correct,
                "nframes": int(prepared.nframes),
                "frame_policy": prepared.frame_policy,
                "frame_policy_reason": prepared.frame_policy_reason,
                "requested_frame_policy": args.frame_policy,
                "hybrid_recent_nframes": (
                    args.hybrid_recent_nframes
                    if args.frame_policy == "hybrid_global_recent" else None
                ),
                "prompt_style": (
                    f"taskblind_{args.prompt_style}_candidate_verifier"
                    if prepared.verifier_candidates is not None
                    else f"taskblind_{args.prompt_style}"
                ),
                "prediction_source": (
                    "same_checkpoint_candidate_verifier"
                    if prepared.verifier_candidates is not None
                    else
                    f"same_checkpoint_persistent_{args.state_packet_mode}"
                    if snapshot_store is not None
                    else "same_checkpoint_question_blind_state_compiler"
                    if args.state_compiler != "none" else "model"
                ),
                "processor_path": processor_path,
                "media_serialization": args.media_serialization,
                "native_video_fps_protocol": (
                    "effective_causal_span"
                    if args.media_serialization == "native_video"
                    and args.native_video_fps == 0
                    else "fixed_config"
                    if args.media_serialization == "native_video"
                    else "not_applicable"
                ),
                "native_video_sample_fps": (
                    native_fps_values[batch_index]
                    if args.media_serialization == "native_video"
                    else None
                ),
                "permuted_pred": permuted_prediction,
                "option_label_shift": prepared.option_label_shift,
                "prediction_mapped_to_original_labels": True,
                "decode_mode": args.decode_mode,
                "answer_parser": args.answer_parser,
                "retrieval_status": "not_requested",
            }
            if prepared.verifier_candidates is not None:
                result["verifier_candidates"] = list(prepared.verifier_candidates)
                result["verifier_inputs"] = [
                    "candidate_option_letters",
                    "visible_question_and_options",
                    "causal_visual_evidence",
                ]
                result["verifier_reads_gold_task_category_or_score"] = False
            if prepared.harness_state is not None:
                result["harness_state"] = prepared.harness_state
            if option_score_rows is not None:
                permuted_scores = option_score_rows[batch_index]
                result["permuted_option_logprobs"] = permuted_scores
                result["original_option_logprobs"] = {
                    (prepared.new_to_original_option or {}).get(label, label): score
                    for label, score in permuted_scores.items()
                }
            if args.save_raw_reply:
                result["raw_reply"] = reply
            results.append(result)
            by_task[task][0] += int(correct)
            by_task[task][1] += 1
            if row_index % 10 == 0:
                correct_total = sum(int(item["correct"]) for item in results)
                print(
                    f"[eval] {len(results)} items "
                    f"acc={correct_total / max(len(results), 1):.1%} "
                    f"(skipped={skipped}, row={row_index}, "
                    f"batch={len(valid)}, last_batch={time.time() - started:.1f}s)",
                    flush=True,
                )
            if args.checkpoint_every and row_index % args.checkpoint_every == 0:
                write_partial()

    processor_prefetch_enabled = os.getenv(
        "OVOS_EVAL_PROCESSOR_PREFETCH", "1"
    ).strip().lower() not in {"0", "false", "no", "off", "disabled"}
    processor_overlap = (
        processor_prefetch_enabled
        and snapshot_store is None
        and args.state_compiler == "none"
        and args.media_serialization == "ordered_image_blocks"
        and args.decode_mode == "generate"
    )

    def attach_generate_inputs(
        values: list[tuple[int, PreparedExample]],
    ) -> list[tuple[int, PreparedExample]]:
        for _, item in values:
            if item.images:
                item.generate_inputs = build_generate_inputs(
                    processor, item.images, item.prompt
                )
        return values

    pending_tensorize = None
    tensorize_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="ovo-processor")
        if processor_overlap
        else None
    )

    def flush_batch(values: list[tuple[int, PreparedExample]]) -> None:
        nonlocal pending_tensorize
        if tensorize_pool is None:
            run_batch(values)
            return
        if pending_tensorize is not None:
            run_batch(pending_tensorize.result())
        pending_tensorize = tensorize_pool.submit(attach_generate_inputs, values)

    for value in prepared_iterator:
        batch.append(value)
        if len(batch) == args.batch_size:
            flush_batch(batch)
            batch = []
    if batch:
        flush_batch(batch)
    if pending_tensorize is not None:
        run_batch(pending_tensorize.result())
    if tensorize_pool is not None:
        tensorize_pool.shutdown(wait=True)

    payload = _result_payload(results, skipped, frame_policy=args.frame_policy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    partial_path.unlink(missing_ok=True)
    summary = payload["summary"]
    print(
        f"{summary['frame_policy']} causal overall: "
        f"{summary['correct']}/{summary['total']} = {summary['accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()
