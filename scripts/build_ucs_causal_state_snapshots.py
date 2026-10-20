#!/usr/bin/env python3
"""Build UCS causal state once per video prefix, without question visibility.

Only ``video_uid``, ``video_path``, and ``query_timestamp`` survive schedule
construction.  The model sees new pixels, the previous typed ledger, and an
update instruction; it never sees questions, options, labels, categories, or
scores.  Questions sharing a prefix later reuse the same immutable snapshot.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from benchmark.ovo.annotation_utils import load_annotations
from benchmark.ovo.utils.frame_utils import extract_fixed_count_interval
from spatial_witness.causal_state_snapshots import CausalStateSnapshotStore
from spatial_witness.causal_workspace import EvidenceRef, WorkspaceError
from spatial_witness.incremental_spatial_ledger import (
    empty_ledger,
    merge_ledgers,
    update_prompt,
)
from spatial_witness.question_blind_ledger import parse as parse_local_ledger


FORMAT = "iclr_spatial.ucs_query_independent_state_builder.v1"
IMPLEMENTATION = "ucs_interval_ledger_writer_v1"


def writer_schedule(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project benchmark rows onto the complete writer-visible schedule."""

    videos: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row.get("video_uid") or row.get("video_id") or "").strip()
        path = str(row.get("video_path") or row.get("video_filename") or "").strip()
        raw_times = row.get("query_times")
        if hasattr(raw_times, "tolist"):
            raw_times = raw_times.tolist()
        if not raw_times:
            value = row.get("query_timestamp", row.get("question_timestamp"))
            raw_times = [value]
        timestamps = [float(value) for value in raw_times]
        if not uid or not path or any(value < 0 for value in timestamps):
            raise ValueError("UCS writer schedule row lacks causal media identity")
        item = videos.setdefault(
            uid, {"video_uid": uid, "video_path": path, "query_timestamps": set()}
        )
        if item["video_path"] != path:
            raise ValueError(f"video path changed within writer schedule: {uid}")
        item["query_timestamps"].update(timestamps)
    return [
        {
            "video_uid": uid,
            "video_path": str(item["video_path"]),
            "query_timestamps": sorted(item["query_timestamps"]),
        }
        for uid, item in sorted(videos.items())
    ]


def balanced_video_shard(
    schedule: list[dict[str, Any]], *, rank: int, world_size: int
) -> list[dict[str, Any]]:
    """Greedily balance unique prefix updates while keeping videos intact."""

    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid writer shard rank/world size")
    loads = [0] * world_size
    shards: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    ordered = sorted(
        schedule,
        key=lambda row: (-len(row["query_timestamps"]), row["video_uid"]),
    )
    for item in ordered:
        target = min(range(world_size), key=lambda index: (loads[index], index))
        shards[target].append(item)
        loads[target] += len(item["query_timestamps"])
    return sorted(shards[rank], key=lambda row: row["video_uid"])


def selected_rows(
    rows: list[dict[str, Any]], id_file: Path | None
) -> list[dict[str, Any]]:
    """Select an opaque diagnostic subset before projecting writer fields."""

    if id_file is None:
        return rows
    value = json.loads(id_file.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("ids") or value.get("question_ids")
    if not isinstance(value, list) or not value:
        raise ValueError("writer ID file must contain a nonempty opaque ID list")
    ids = [str(item).strip() for item in value]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("writer ID file has empty or duplicate identities")
    by_id = {
        str(row.get("id") or row.get("qid") or "").strip(): row for row in rows
    }
    missing = [identity for identity in ids if identity not in by_id]
    if missing:
        raise ValueError(f"writer IDs absent from annotation: {missing[:3]}")
    return [by_id[identity] for identity in ids]


def _pixel_sha256(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(getattr(image, "mode", "")).encode())
    digest.update(str(getattr(image, "size", "")).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def writer_implementation_sha256(model: str, processor: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "implementation": IMPLEMENTATION,
                "model": str(Path(model).resolve()),
                "processor": str(Path(processor).resolve()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _write_checkpoint(
    path: Path,
    store: CausalStateSnapshotStore,
    trace: list[dict[str, Any]],
    report: Mapping[str, Any],
) -> None:
    payload = {
        "format": FORMAT,
        "snapshot_store": store.to_manifest(),
        "trace": trace,
        "report": dict(report),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--id-file", type=Path)
    parser.add_argument("--video-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-world-size", type=int, default=1)
    parser.add_argument("--frames-per-update", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    if min(
        args.frames_per_update,
        args.frame_size,
        args.max_new_tokens,
        args.checkpoint_every,
    ) < 1:
        raise SystemExit("writer budgets and checkpoint interval must be positive")
    if not args.annotation.is_file() or not args.video_base.is_dir():
        raise FileNotFoundError("UCS annotation or video root is unavailable")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)

    processor_path = args.processor or args.model
    implementation_sha256 = writer_implementation_sha256(args.model, processor_path)
    if args.resume and args.output.is_file():
        completed = json.loads(args.output.read_text(encoding="utf-8"))
        if completed.get("format") != FORMAT:
            raise WorkspaceError("completed UCS writer artifact format changed")
        completed_store = CausalStateSnapshotStore.from_manifest(
            completed["snapshot_store"]
        )
        if completed_store.writer_implementation_sha256 != implementation_sha256:
            raise WorkspaceError("completed UCS writer implementation changed")
        print(json.dumps(completed.get("report") or {}, sort_keys=True))
        return
    schedule = balanced_video_shard(
        writer_schedule(selected_rows(
            load_annotations(str(args.annotation)), args.id_file
        )),
        rank=args.shard_rank,
        world_size=args.shard_world_size,
    )
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    trace: list[dict[str, Any]] = []
    if args.resume and partial.is_file():
        checkpoint = json.loads(partial.read_text(encoding="utf-8"))
        if checkpoint.get("format") != FORMAT:
            raise WorkspaceError("UCS writer checkpoint format changed")
        store = CausalStateSnapshotStore.from_manifest(
            checkpoint["snapshot_store"]
        )
        if store.writer_implementation_sha256 != implementation_sha256:
            raise WorkspaceError("UCS writer checkpoint implementation changed")
        trace = list(checkpoint.get("trace") or [])
    else:
        store = CausalStateSnapshotStore(
            writer_implementation_sha256=implementation_sha256,
            initial_state=empty_ledger(),
        )

    from synthesis.vlm_inference import load_model, predict

    model, processor = load_model(
        args.model, processor_path=processor_path
    )
    updates = 0
    started = time.perf_counter()
    for video_index, item in enumerate(schedule, 1):
        uid = item["video_uid"]
        video_path = args.video_base / item["video_path"]
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        previous_timestamp = None
        observation_count = 0
        try:
            latest = store.read_latest(video_uid=uid)
        except WorkspaceError:
            latest = None
        if latest is not None:
            previous_timestamp = latest.query_timestamp
            observation_count = sum(
                len(row.get("evidence_provenance") or [])
                for row in store.to_manifest()["snapshots"]
                if row["video_uid"] == uid and row["writer_invoked"]
            )
        for timestamp in item["query_timestamps"]:
            try:
                store.read_exact(video_uid=uid, query_timestamp=timestamp)
                previous_timestamp = timestamp
                continue
            except WorkspaceError:
                pass
            frames, plan = extract_fixed_count_interval(
                str(video_path),
                start_time_exclusive=previous_timestamp,
                query_time=timestamp,
                nframes=args.frames_per_update,
                frame_size=args.frame_size,
            )
            if plan is None:
                raise RuntimeError(f"unable to plan causal frames for {uid}")
            if not frames:
                snapshot = store.advance_horizon(
                    video_uid=uid, query_timestamp=timestamp
                )
                trace.append({
                    "video_uid": uid,
                    "query_timestamp": timestamp,
                    "status": "no_new_decoder_frame",
                    "writer_invoked": False,
                    "state_sha256": snapshot.state_sha256,
                })
                previous_timestamp = timestamp
                continue
            evidence = [
                EvidenceRef(
                    source_id=uid,
                    frame_index=int(frame_index),
                    timestamp=float(frame_timestamp),
                    pixel_sha256=_pixel_sha256(image),
                )
                for image, frame_index, frame_timestamp in zip(
                    frames,
                    plan["frame_indices"],
                    plan["frame_timestamps"],
                    strict=True,
                )
            ]
            prior = (
                store.read_latest(video_uid=uid).state
                if previous_timestamp is not None
                else empty_ledger()
            )
            prompt = update_prompt(
                frame_start=observation_count,
                frame_count=len(frames),
                prior=prior,
                context_frame_count=0,
            )
            update_started = time.perf_counter()
            raw = predict(
                model, processor, frames, prompt,
                max_new_tokens=args.max_new_tokens,
            )
            try:
                local = parse_local_ledger(raw, frame_count=len(frames))
                merged = merge_ledgers(
                    prior, local, frame_start=observation_count
                )
                status = "merged"
            except WorkspaceError as error:
                merged = prior
                status = f"invalid_model_update:{type(error).__name__}"
            observations = [
                {
                    "frame_index": ref.frame_index,
                    "timestamp": ref.timestamp,
                    "pixel_sha256": ref.pixel_sha256,
                }
                for ref in evidence
            ]
            snapshot = store.advance(
                video_uid=uid,
                query_timestamp=timestamp,
                observations=observations,
                evidence_provenance=evidence,
                update_state=lambda _prior, _observations, _evidence, value=merged: value,
            )
            observation_count += len(frames)
            updates += 1
            trace.append({
                "video_uid": uid,
                "query_timestamp": timestamp,
                "status": status,
                "writer_invoked": True,
                "new_frame_count": len(frames),
                "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "state_sha256": snapshot.state_sha256,
                "writer_wall_time_ms": 1000.0 * (
                    time.perf_counter() - update_started
                ),
                "question_or_options_visible_to_writer": False,
            })
            previous_timestamp = timestamp
            if updates % args.checkpoint_every == 0:
                _write_checkpoint(
                    partial,
                    store,
                    trace,
                    {
                        "shard_rank": args.shard_rank,
                        "shard_world_size": args.shard_world_size,
                        "updates_completed": updates,
                        "videos_completed_or_active": video_index,
                    },
                )
                print(
                    f"[writer] videos={video_index}/{len(schedule)} "
                    f"updates={updates} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    report = {
        "shard_rank": args.shard_rank,
        "shard_world_size": args.shard_world_size,
        "videos": len(schedule),
        "unique_prefixes": sum(len(row["query_timestamps"]) for row in schedule),
        "writer_calls": store.writer_calls,
        "question_or_options_visible_to_writer": False,
        "writer_visible_schedule_fields": [
            "video_uid", "video_path", "query_timestamp"
        ],
    }
    _write_checkpoint(args.output, store, trace, report)
    partial.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
