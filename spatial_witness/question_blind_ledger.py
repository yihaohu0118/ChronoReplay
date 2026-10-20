"""Question-blind same-checkpoint compilation of one causal spatial ledger."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from spatial_witness.causal_workspace import WorkspaceError
from spatial_witness.qwen_agent_adapter import extract_json_object


FORMAT = "spatial_witness.question_blind_spatial_ledger.v1"
CONFIDENT_ENTITY_FORMAT = "spatial_witness.confident_entity_memory.v1"
_ALLOWED_EVENTS = frozenset(
    {"enter", "leave", "revisit", "turn_left", "turn_right", "object_seen"}
)


def prompt(frame_count: int) -> str:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise WorkspaceError("ledger frame count must be positive")
    return f"""Compile a compact spatial memory from these {frame_count} ordered causal video observations. The future question and answer options are unavailable.
Observation indices run from 0 to {frame_count - 1}. Use only visible pixels and temporal order. Return exactly one JSON object and no markdown:
{{"status":"available","places":[{{"name":"short canonical place","first_observation":0,"last_observation":0,"visit_count":1}}],"objects":[{{"name":"short canonical object","last_observation":0,"place":"short canonical place or unknown","relation":"short visible spatial relation or unknown"}}],"events":[{{"kind":"enter|leave|revisit|turn_left|turn_right|object_seen","observation":0,"subject":"short name"}}]}}
Use empty lists when uncertain. Never infer metric pose, never answer a question, and never emit an option letter."""


def _index(value: Any, frame_count: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkspaceError(f"ledger {name} must be an integer")
    if not 0 <= value < frame_count:
        raise WorkspaceError(f"ledger {name} is outside causal observations")
    return value


def _name(value: Any, name: str) -> str:
    result = " ".join(str(value or "").strip().casefold().split())
    if not result or len(result) > 96:
        raise WorkspaceError(f"ledger {name} is invalid")
    return result


def parse(raw: str | Mapping[str, Any], *, frame_count: int) -> dict[str, Any]:
    payload = extract_json_object(raw) if isinstance(raw, str) else dict(raw)
    if set(payload) != {"status", "places", "objects", "events"}:
        raise WorkspaceError("ledger object has missing or hidden fields")
    if payload["status"] != "available":
        raise WorkspaceError("ledger status must be available")
    output: dict[str, Any] = {
        "format": FORMAT,
        "status": "available",
        "places": [],
        "objects": [],
        "events": [],
    }
    for key, limit in (("places", 64), ("objects", 128), ("events", 256)):
        value = payload[key]
        if not isinstance(value, list) or len(value) > limit:
            raise WorkspaceError(f"ledger {key} must be a bounded list")
    for item in payload["places"]:
        if not isinstance(item, Mapping) or set(item) != {
            "name", "first_observation", "last_observation", "visit_count"
        }:
            raise WorkspaceError("ledger place has missing or hidden fields")
        first = _index(item["first_observation"], frame_count, "first observation")
        last = _index(item["last_observation"], frame_count, "last observation")
        visits = item["visit_count"]
        if first > last or isinstance(visits, bool) or not isinstance(visits, int) or visits <= 0:
            raise WorkspaceError("ledger place interval or visit count is invalid")
        output["places"].append(
            {"name": _name(item["name"], "place name"), "first_observation": first,
             "last_observation": last, "visit_count": visits}
        )
    for item in payload["objects"]:
        if not isinstance(item, Mapping) or set(item) != {
            "name", "last_observation", "place", "relation"
        }:
            raise WorkspaceError("ledger object entry has missing or hidden fields")
        output["objects"].append(
            {"name": _name(item["name"], "object name"),
             "last_observation": _index(item["last_observation"], frame_count, "last observation"),
             "place": _name(item["place"], "object place"),
             "relation": _name(item["relation"], "object relation")}
        )
    for item in payload["events"]:
        if not isinstance(item, Mapping) or set(item) != {"kind", "observation", "subject"}:
            raise WorkspaceError("ledger event has missing or hidden fields")
        kind = str(item["kind"])
        if kind not in _ALLOWED_EVENTS:
            raise WorkspaceError("ledger event kind is unsupported")
        output["events"].append(
            {"kind": kind,
             "observation": _index(item["observation"], frame_count, "event observation"),
             "subject": _name(item["subject"], "event subject")}
        )
    output["ledger_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return output


def unavailable(raw: str, reason: str) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "status": "unavailable",
        "reason": reason,
        "raw_output_sha256": hashlib.sha256(str(raw).encode()).hexdigest(),
        "places": [],
        "objects": [],
        "events": [],
    }


def augment_answer_prompt(original: str, ledger: Mapping[str, Any]) -> str:
    return (
        original
        + "\n\nQuestion-blind spatial memory compiled from the same causal frames:\n"
        + json.dumps(dict(ledger), ensure_ascii=False, sort_keys=True)
        + "\nUse this memory only when supported by the visible frames. "
          "Respond under the original answer-format instruction."
    )


def confident_entity_memory(
    ledger: Mapping[str, Any], *, maximum_entities: int = 24
) -> dict[str, Any]:
    """Project a noisy generated ledger onto its safest reader-visible facts.

    The current writer has no calibrated metric-pose, topology, transition, or
    revisit verifier.  Exposing those fields as facts makes an answer model
    over-trust hallucinated geometry.  Entity identity, last observation, and
    the writer's explicit place binding are retained; every unsupported field
    is deliberately absent.  The projection is question blind and therefore
    cannot route on benchmark categories or answer options.
    """

    if (
        isinstance(maximum_entities, bool)
        or not isinstance(maximum_entities, int)
        or maximum_entities <= 0
    ):
        raise WorkspaceError("maximum confident entities must be positive")
    if ledger.get("status") != "available":
        return {
            "format": CONFIDENT_ENTITY_FORMAT,
            "status": "abstain",
            "entities": [],
        }
    candidates = []
    for source in ledger.get("objects") or []:
        if not isinstance(source, Mapping):
            continue
        try:
            name = _name(source.get("name"), "object name")
            observation = int(source.get("last_observation"))
        except (TypeError, ValueError, WorkspaceError):
            continue
        if observation < 0:
            continue
        place = " ".join(str(source.get("place") or "").strip().casefold().split())
        row = {"name": name, "last_observation": observation}
        if place and place != "unknown" and len(place) <= 96:
            row["place"] = place
        candidates.append(row)

    # One canonical entry per entity.  Recency wins deterministically.
    by_name: dict[str, dict[str, Any]] = {}
    for row in candidates:
        previous = by_name.get(row["name"])
        if previous is None or row["last_observation"] > previous["last_observation"]:
            by_name[row["name"]] = row
    entities = sorted(
        by_name.values(),
        key=lambda row: (-int(row["last_observation"]), str(row["name"])),
    )[:maximum_entities]
    return {
        "format": CONFIDENT_ENTITY_FORMAT,
        "status": "available" if entities else "abstain",
        "entities": entities,
    }


def augment_confident_entity_prompt(
    original: str, memory: Mapping[str, Any]
) -> str:
    """Inject a compact entity memory, or preserve the exact control prompt."""

    if memory.get("status") != "available" or not memory.get("entities"):
        return original
    return (
        original
        + "\n\nQuestion-blind persistent entity memory from earlier causal observations:\n"
        + json.dumps(dict(memory), ensure_ascii=False, sort_keys=True)
        + "\nThis memory establishes only entity names, last-observation order, and "
          "explicit place bindings. It does not establish direction, metric pose, "
          "topology, motion, or visit counts. Use it only when relevant and answer "
          "under the original format instruction."
    )
