"""Question-blind incremental compilation of a causal spatial ledger.

One model call over a long frame packet asks a small VLM to solve recognition,
temporal binding, and persistent memory simultaneously.  This module instead
exposes the streaming recurrence explicitly: a bounded frame chunk produces a
local update, and deterministic code merges that update into persistent state.
The compiler callback never receives a question, options, task metadata, or an
answer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from typing import Any

from spatial_witness.causal_workspace import WorkspaceError
from spatial_witness.question_blind_ledger import parse as parse_local_ledger


FORMAT = "spatial_witness.incremental_question_blind_ledger.v1"
Compiler = Callable[[Sequence[Any], str], str | Mapping[str, Any]]


def empty_ledger() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "status": "available",
        "places": [],
        "objects": [],
        "events": [],
    }


def update_prompt(
    *,
    frame_start: int,
    frame_count: int,
    prior: Mapping[str, Any],
    context_frame_count: int = 0,
) -> str:
    if (
        frame_start < 0
        or frame_count <= 0
        or context_frame_count < 0
        or context_frame_count >= frame_count
        or context_frame_count > frame_start
    ):
        raise WorkspaceError("incremental ledger frame interval is invalid")
    prior_names = {
        "places": sorted({str(row["name"]) for row in prior.get("places", [])}),
        "objects": sorted({str(row["name"]) for row in prior.get("objects", [])}),
    }
    recent_snapshot = {
        "places": sorted(
            (dict(row) for row in prior.get("places", [])),
            key=lambda row: int(row["last_observation"]),
        )[-16:],
        "objects": sorted(
            (dict(row) for row in prior.get("objects", [])),
            key=lambda row: int(row["last_observation"]),
        )[-32:],
        "events": sorted(
            (dict(row) for row in prior.get("events", [])),
            key=lambda row: int(row["observation"]),
        )[-32:],
    }
    global_offset = frame_start - context_frame_count
    first_new = context_frame_count
    overlap_instruction = (
        "There are no overlap-only observations in this first chunk."
        if context_frame_count == 0
        else (
            f"Indices 0 through {first_new - 1} are overlap context for identity "
            f"matching only. Emit places, objects, and events only for new "
            f"indices {first_new} through {frame_count - 1}."
        )
    )
    return f"""Update a persistent spatial memory from {frame_count} ordered causal video observations. The future question and answer options are unavailable.
These chunk-local observation indices run from 0 to {frame_count - 1}; the deterministic runtime will shift them by global offset {global_offset}.
{overlap_instruction}
Previously established canonical names are: {json.dumps(prior_names, ensure_ascii=False, sort_keys=True)}.
The bounded recent persistent state is: {json.dumps(recent_snapshot, ensure_ascii=False, sort_keys=True)}.
Reuse a previous name only when the pixels support the identity; otherwise use a short new canonical name. Report only facts visible in this chunk. Return exactly one JSON object and no markdown:
{{"status":"available","places":[{{"name":"short canonical place","first_observation":0,"last_observation":0,"visit_count":1}}],"objects":[{{"name":"short canonical object","last_observation":0,"place":"short canonical place or unknown","relation":"short visible spatial relation or unknown"}}],"events":[{{"kind":"enter|leave|revisit|turn_left|turn_right|object_seen","observation":0,"subject":"short name"}}]}}
Use empty lists when uncertain. Never infer metric pose, never answer a question, and never emit an option letter."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _shift(local: Mapping[str, Any], offset: int) -> dict[str, Any]:
    shifted = empty_ledger()
    shifted["places"] = [
        {
            **dict(row),
            "first_observation": int(row["first_observation"]) + offset,
            "last_observation": int(row["last_observation"]) + offset,
        }
        for row in local.get("places", [])
    ]
    shifted["objects"] = [
        {**dict(row), "last_observation": int(row["last_observation"]) + offset}
        for row in local.get("objects", [])
    ]
    shifted["events"] = [
        {**dict(row), "observation": int(row["observation"]) + offset}
        for row in local.get("events", [])
    ]
    return shifted


def _commit_new_observations(
    local: Mapping[str, Any], *, first_new_observation: int
) -> dict[str, Any]:
    """Remove overlap-only facts before deterministic persistent-state merge."""

    if first_new_observation <= 0:
        return dict(local)
    committed = {
        "format": local.get("format"),
        "status": local.get("status"),
        "places": [],
        "objects": [],
        "events": [],
    }
    for source in local.get("places", []):
        row = dict(source)
        if int(row["last_observation"]) < first_new_observation:
            continue
        crossed_boundary = int(row["first_observation"]) < first_new_observation
        row["first_observation"] = max(
            int(row["first_observation"]), first_new_observation
        )
        # The overlap is identity evidence, not a new visit. If a model spans
        # the boundary, commit one continuing visit instead of double-counting
        # a visit that was already present in the prior ledger.
        if crossed_boundary:
            row["visit_count"] = 1
        committed["places"].append(row)
    committed["objects"] = [
        dict(row) for row in local.get("objects", [])
        if int(row["last_observation"]) >= first_new_observation
    ]
    committed["events"] = [
        dict(row) for row in local.get("events", [])
        if int(row["observation"]) >= first_new_observation
    ]
    return committed


def merge_ledgers(
    prior: Mapping[str, Any], local: Mapping[str, Any], *, frame_start: int
) -> dict[str, Any]:
    """Merge one parsed local update without asking the model to remember it."""

    if prior.get("status") != "available" or local.get("status") != "available":
        raise WorkspaceError("only available typed ledgers may be merged")
    update = _shift(local, frame_start)

    places = {str(row["name"]): dict(row) for row in prior.get("places", [])}
    for row in update["places"]:
        name = str(row["name"])
        previous = places.get(name)
        if previous is None:
            places[name] = dict(row)
            continue
        contiguous = int(previous["last_observation"]) + 1 >= int(
            row["first_observation"]
        )
        places[name] = {
            "name": name,
            "first_observation": min(
                int(previous["first_observation"]), int(row["first_observation"])
            ),
            "last_observation": max(
                int(previous["last_observation"]), int(row["last_observation"])
            ),
            "visit_count": (
                int(previous["visit_count"])
                + int(row["visit_count"])
                - int(contiguous)
            ),
        }

    objects = {str(row["name"]): dict(row) for row in prior.get("objects", [])}
    for row in update["objects"]:
        name = str(row["name"])
        previous = objects.get(name)
        if previous is None or int(row["last_observation"]) >= int(
            previous["last_observation"]
        ):
            objects[name] = dict(row)

    events = {
        (str(row["kind"]), int(row["observation"]), str(row["subject"])): dict(row)
        for row in prior.get("events", [])
    }
    for row in update["events"]:
        events[(str(row["kind"]), int(row["observation"]), str(row["subject"]))] = dict(row)
    if len(places) > 64 or len(objects) > 128 or len(events) > 256:
        raise WorkspaceError("incremental ledger exceeded its public state budget")
    merged: dict[str, Any] = {
        "format": FORMAT,
        "status": "available",
        "places": sorted(places.values(), key=lambda row: str(row["name"])),
        "objects": sorted(objects.values(), key=lambda row: str(row["name"])),
        "events": sorted(
            events.values(),
            key=lambda row: (
                int(row["observation"]), str(row["kind"]), str(row["subject"])
            ),
        ),
    }
    merged["ledger_sha256"] = _canonical_hash(merged)
    return merged


def compile_incrementally(
    frames: Sequence[Any],
    compiler: Compiler,
    *,
    chunk_frames: int,
    overlap_frames: int = 2,
) -> dict[str, Any]:
    if (
        not frames
        or chunk_frames <= 0
        or overlap_frames < 0
        or overlap_frames >= chunk_frames
    ):
        raise WorkspaceError("incremental compiler requires frames and a chunk budget")
    ledger = empty_ledger()
    updates: list[dict[str, Any]] = []
    successful = 0
    for start in range(0, len(frames), chunk_frames):
        end = min(len(frames), start + chunk_frames)
        context_count = min(overlap_frames, start)
        context_start = start - context_count
        chunk = frames[context_start:end]
        text = update_prompt(
            frame_start=start,
            frame_count=len(chunk),
            prior=ledger,
            context_frame_count=context_count,
        )
        raw = compiler(chunk, text)
        try:
            local = parse_local_ledger(raw, frame_count=len(chunk))
            committed = _commit_new_observations(
                local, first_new_observation=context_count
            )
            ledger = merge_ledgers(
                ledger, committed, frame_start=context_start
            )
            status = "merged"
            successful += 1
        except WorkspaceError as error:
            status = f"unavailable:{type(error).__name__}"
        updates.append(
            {
                "frame_start": start,
                "frame_count": end - start,
                "context_frame_start": context_start,
                "context_frame_count": context_count,
                "status": status,
                "raw_output_sha256": hashlib.sha256(str(raw).encode()).hexdigest(),
                "state_sha256": _canonical_hash(ledger),
            }
        )
    if successful == 0:
        ledger = {
            **empty_ledger(),
            "status": "unavailable",
            "reason": "all_incremental_updates_invalid",
        }
        ledger["ledger_sha256"] = _canonical_hash(ledger)
    body = {
        "format": FORMAT,
        "status": "available" if successful else "unavailable",
        "frame_count": len(frames),
        "chunk_frames": chunk_frames,
        "overlap_frames": overlap_frames,
        "compiler_calls": math.ceil(len(frames) / chunk_frames),
        "successful_updates": successful,
        "question_or_options_visible_to_compiler": False,
        "ledger": ledger,
        "updates": updates,
    }
    return {**body, "trace_sha256": _canonical_hash(body)}
