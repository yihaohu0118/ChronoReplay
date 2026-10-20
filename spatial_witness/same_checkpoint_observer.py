"""Question-blind state extraction with the released answer checkpoint.

The observer is an orchestration boundary, not a second model.  A caller
injects a callback backed by the exact checkpoint and decoding contract used
by the answer agent.  The callback sees one causal RGB frame and emits a
strict entity/place update.  It never receives the question, options,
benchmark metadata, predictions, or scores.

This module deliberately excludes pose deltas.  Numeric ego-motion requires a
separately validated geometric extractor; accepting free-form VLM pose output
would turn an auditable state service into hallucinated geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from PIL import Image

from spatial_witness.causal_state_builder import (
    ExtractorIdentity,
    QuestionBlindStateBuilder,
    parse_frame_state_update,
    pixel_sha256,
)
from spatial_witness.causal_workspace import EvidenceRef, WorkspaceError


FORMAT = "spatial_witness.same_checkpoint_state_observer.v1"
CACHE_FORMAT = "spatial_witness.same_checkpoint_observer_cache.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UNAVAILABLE_REASONS = frozenset(
    {"blurred", "low_confidence", "no_spatial_content", "unreadable"}
)


def observer_prompt(maximum_entities: int) -> str:
    """Return the single authoritative writer prompt used by SFT and runtime."""

    if (
        isinstance(maximum_entities, bool)
        or not isinstance(maximum_entities, int)
        or maximum_entities <= 0
    ):
        raise WorkspaceError("maximum observer entities must be positive")
    return f"""Inspect only this single causal video frame. The future question is unavailable.
Return exactly one JSON object and no markdown.
If the frame is unreadable, return {{"status":"unavailable","reason":"low_confidence"}}.
Otherwise return {{"status":"available","entities":[{{"name":"short canonical object name","attributes":{{"visible_relation":"short visible spatial relation"}}}}],"place":"short canonical room or place type"}}.
Use only visible pixels. List at most {maximum_entities} entities. Do not estimate camera pose. Do not answer any question."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SameCheckpointObserverConfig:
    checkpoint_sha256: str
    decoding_contract_sha256: str
    frame_stride: int = 15
    maximum_entities_per_frame: int = 16

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.checkpoint_sha256):
            raise WorkspaceError("observer checkpoint identity must be a SHA-256")
        if not _SHA256.fullmatch(self.decoding_contract_sha256):
            raise WorkspaceError("observer decoding contract must be a SHA-256")
        for name, value in (
            ("frame_stride", self.frame_stride),
            ("maximum_entities_per_frame", self.maximum_entities_per_frame),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WorkspaceError(f"{name} must be a positive integer")

    def extractor_identity(self) -> ExtractorIdentity:
        public = {
            "format": FORMAT,
            "checkpoint_sha256": self.checkpoint_sha256,
            "decoding_contract_sha256": self.decoding_contract_sha256,
            "frame_stride": self.frame_stride,
            "maximum_entities_per_frame": self.maximum_entities_per_frame,
            "question_blind": True,
            "pose_output_allowed": False,
        }
        return ExtractorIdentity(
            name="same-checkpoint-entity-place-observer",
            version="1",
            kind="same_model_observer",
            configuration_sha256=_canonical_sha256(public),
        )


@dataclass(frozen=True)
class ObserverDecision:
    status: str
    update: dict[str, Any] | None = None
    reason: str | None = None
    coalesced_duplicate_entities: int = 0


def _coalesce_duplicate_entities(value: Any) -> tuple[Any, int]:
    """Merge same-name instances without discarding visible relations.

    The causal workspace indexes semantic names.  A VLM may legitimately emit
    two ``truck`` observations in one frame; treating that as a fatal protocol
    error turns ordinary perception into missing benchmark predictions.  This
    generic normalization keeps one semantic entity and joins conflicting
    scalar attributes deterministically.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return value, 0
    output: list[Any] = []
    by_name: dict[str, int] = {}
    coalesced = 0
    for raw in value:
        if not isinstance(raw, Mapping):
            output.append(raw)
            continue
        item = dict(raw)
        name = " ".join(str(item.get("name") or "").strip().casefold().split())
        attributes = item.get("attributes") or {}
        if not name or not isinstance(attributes, Mapping) or name not in by_name:
            if name:
                by_name[name] = len(output)
            output.append(item)
            continue
        existing = output[by_name[name]]
        if not isinstance(existing, dict):
            output.append(item)
            continue
        merged = dict(existing.get("attributes") or {})
        for key, value_item in attributes.items():
            if key not in merged:
                merged[key] = value_item
            elif str(merged[key]) != str(value_item):
                values = [part.strip() for part in str(merged[key]).split(" | ")]
                if str(value_item) not in values:
                    merged[key] = " | ".join([*values, str(value_item)])
        existing["attributes"] = merged
        coalesced += 1
    return output, coalesced


def parse_observer_decision(
    value: str | Mapping[str, Any], *, maximum_entities: int
) -> ObserverDecision:
    """Validate one observer decision before any workspace mutation."""

    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise WorkspaceError("observer decision is not valid JSON") from error
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise WorkspaceError("observer decision must be one JSON object")
    if not isinstance(payload, dict):
        raise WorkspaceError("observer decision must be one JSON object")

    status = str(payload.get("status") or "")
    if status == "unavailable":
        if set(payload) != {"status", "reason"}:
            raise WorkspaceError("unavailable observer decision has hidden fields")
        reason = str(payload.get("reason") or "")
        if reason not in _UNAVAILABLE_REASONS:
            raise WorkspaceError("unsupported observer unavailable reason")
        return ObserverDecision(status=status, reason=reason)
    if status != "available":
        raise WorkspaceError(f"unsupported observer status: {status!r}")
    if set(payload) - {"status", "entities", "place"}:
        raise WorkspaceError("available observer decision has hidden fields")

    entities, coalesced = _coalesce_duplicate_entities(
        payload.get("entities") or []
    )
    update = {
        "entities": entities,
        "place": payload.get("place"),
    }
    parsed = parse_frame_state_update(update)
    if parsed.pose_delta is not None:
        raise WorkspaceError("same-checkpoint observer cannot emit pose deltas")
    if any(item.world_position is not None for item in parsed.entities):
        raise WorkspaceError(
            "same-checkpoint observer cannot emit metric entity positions"
        )
    if len(parsed.entities) > maximum_entities:
        raise WorkspaceError("observer emitted too many entities")
    # Return the normalized, JSON-serializable representation consumed by the
    # ordinary state builder.  No private parser objects enter the audit.
    normalized: dict[str, Any] = {
        "entities": [
            {"name": item.name, "attributes": dict(item.attributes)}
            for item in parsed.entities
        ]
    }
    if parsed.place is not None:
        normalized["place"] = parsed.place
    return ObserverDecision(
        status=status,
        update=normalized,
        coalesced_duplicate_entities=coalesced,
    )


ObserverStep = Callable[[dict[str, Any]], str | Mapping[str, Any]]


class ObserverDecisionCache:
    """Immutable cache for question-blind same-checkpoint observations.

    The key binds every model-facing input that can change an observer result.
    It intentionally excludes the downstream question and options because the
    observer never receives them.  Files are published with an atomic hard
    link, so concurrent workers cannot expose a partially written decision.
    """

    def __init__(self, root: Path | str, *, require_hit: bool = False) -> None:
        self.root = Path(root)
        self.require_hit = bool(require_hit)

    @staticmethod
    def key(config: SameCheckpointObserverConfig, pixel_hash: str) -> str:
        if not _SHA256.fullmatch(pixel_hash):
            raise WorkspaceError("observer cache pixel identity must be a SHA-256")
        return _canonical_sha256(
            {
                "format": CACHE_FORMAT,
                "checkpoint_sha256": config.checkpoint_sha256,
                "decoding_contract_sha256": config.decoding_contract_sha256,
                "frame_stride": config.frame_stride,
                "maximum_entities_per_frame": config.maximum_entities_per_frame,
                "pixel_sha256": pixel_hash,
            }
        )

    def _path(self, key: str) -> Path:
        if not _SHA256.fullmatch(key):
            raise WorkspaceError("observer cache key must be a SHA-256")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | Mapping[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            if self.require_hit:
                raise WorkspaceError(f"required observer cache entry is missing: {key}")
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkspaceError(f"invalid observer cache entry: {path}") from error
        if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
            raise WorkspaceError(f"invalid observer cache contract: {path}")
        if payload.get("key") != key or "raw_decision" not in payload:
            raise WorkspaceError(f"observer cache identity mismatch: {path}")
        return payload["raw_decision"]

    def put(self, key: str, raw: str | Mapping[str, Any]) -> None:
        # Validate JSON serializability before touching the cache directory.
        payload = {"format": CACHE_FORMAT, "key": key, "raw_decision": raw}
        encoded = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode()
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{key}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.get(key)
                if _canonical_sha256(existing) != _canonical_sha256(raw):
                    raise WorkspaceError("observer cache contains a conflicting decision")
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class SameCheckpointStateObserver:
    """Populate one causal workspace using a fixed question-blind schedule."""

    def __init__(
        self,
        source_id: str,
        config: SameCheckpointObserverConfig,
        *,
        answer_checkpoint_sha256: str,
        answer_decoding_contract_sha256: str,
        decision_cache: ObserverDecisionCache | None = None,
    ) -> None:
        if config.checkpoint_sha256 != answer_checkpoint_sha256:
            raise WorkspaceError("observer and answerer checkpoints differ")
        if config.decoding_contract_sha256 != answer_decoding_contract_sha256:
            raise WorkspaceError("observer and answerer decoding contracts differ")
        self.config = config
        self.decision_cache = decision_cache
        self.builder = QuestionBlindStateBuilder(
            source_id, config.extractor_identity()
        )
        self.model_calls = 0
        self.cache_hits = 0
        self.observer_requests = 0
        self.unavailable = 0
        self.coalesced_duplicate_entities = 0
        self.observed_frame_indices: list[int] = []
        self.raw_decision_sha256: list[str] = []
        self.input_pixel_sha256: list[str] = []

    def should_observe(self, frame_index: int) -> bool:
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise WorkspaceError("frame_index must be an integer")
        if frame_index < 0:
            raise WorkspaceError("frame_index must be nonnegative")
        return frame_index % self.config.frame_stride == 0

    def _context(
        self, image: Image.Image, *, frame_index: int, timestamp: float
    ) -> dict[str, Any]:
        return {
            "protocol": FORMAT,
            "frame_index": frame_index,
            "timestamp": float(timestamp),
            "model_visible_observation": image,
            "allowed_output": {
                "available": {
                    "status": "available",
                    "entities": [
                        {"name": "visible entity", "attributes": {}}
                    ],
                    "place": "stable visible place name or null",
                },
                "unavailable": {
                    "status": "unavailable",
                    "reason": sorted(_UNAVAILABLE_REASONS),
                },
            },
            "maximum_entities": self.config.maximum_entities_per_frame,
            "pose_output_allowed": False,
        }

    def observe_frame(
        self,
        image: Image.Image,
        *,
        frame_index: int,
        timestamp: float,
        model_step: ObserverStep,
    ) -> EvidenceRef | None:
        """Observe one scheduled frame and atomically apply its state update."""

        if not self.should_observe(frame_index):
            raise WorkspaceError("frame is outside the fixed observer schedule")
        if not isinstance(image, Image.Image):
            raise WorkspaceError("observer input must be a PIL image")
        context = self._context(image, frame_index=frame_index, timestamp=timestamp)
        input_pixel_hash = pixel_sha256(image)
        cache_key = ObserverDecisionCache.key(self.config, input_pixel_hash)
        raw = None if self.decision_cache is None else self.decision_cache.get(cache_key)
        self.observer_requests += 1
        if raw is None:
            raw = model_step(context)
            self.model_calls += 1
            if self.decision_cache is not None:
                self.decision_cache.put(cache_key, raw)
        else:
            self.cache_hits += 1
        decision = parse_observer_decision(
            raw, maximum_entities=self.config.maximum_entities_per_frame
        )
        self.coalesced_duplicate_entities += decision.coalesced_duplicate_entities
        self.observed_frame_indices.append(frame_index)
        self.input_pixel_sha256.append(input_pixel_hash)
        self.raw_decision_sha256.append(_canonical_sha256(raw))
        if decision.status == "unavailable":
            self.unavailable += 1
            return None
        assert decision.update is not None
        return self.builder.ingest(
            image,
            frame_index=frame_index,
            timestamp=timestamp,
            update=decision.update,
        )

    def observe_stream(
        self,
        frames: Sequence[tuple[int, float, Image.Image]],
        *,
        model_step: ObserverStep,
    ) -> list[EvidenceRef]:
        """Apply the fixed sampling schedule to an already causal frame stream."""

        output = []
        previous_index = -1
        previous_time = -1.0
        for frame_index, timestamp, image in frames:
            if frame_index <= previous_index or float(timestamp) < previous_time:
                raise WorkspaceError("observer stream must be chronological")
            previous_index, previous_time = frame_index, float(timestamp)
            if not self.should_observe(frame_index):
                continue
            reference = self.observe_frame(
                image,
                frame_index=frame_index,
                timestamp=timestamp,
                model_step=model_step,
            )
            if reference is not None:
                output.append(reference)
        return output

    def audit(self) -> dict[str, Any]:
        body = {
            "format": FORMAT,
            "config": asdict(self.config),
            "extractor": asdict(self.builder.extractor),
            "observer_requests": self.observer_requests,
            "model_calls": self.model_calls,
            "cache_hits": self.cache_hits,
            "question_blind_cache_enabled": self.decision_cache is not None,
            "observer_cache_requires_hit": (
                self.decision_cache.require_hit
                if self.decision_cache is not None
                else False
            ),
            "question_blind_cache_format": (
                CACHE_FORMAT if self.decision_cache is not None else None
            ),
            "unavailable": self.unavailable,
            "coalesced_duplicate_entities": self.coalesced_duplicate_entities,
            "observed_frame_indices": self.observed_frame_indices,
            "input_pixel_sha256": self.input_pixel_sha256,
            "raw_decision_sha256": self.raw_decision_sha256,
            "workspace": self.builder.audit(),
            "same_checkpoint_observer_and_answerer": True,
            "question_or_options_read": False,
            "benchmark_metadata_answers_predictions_or_scores_read": False,
            "external_generative_model_calls": 0,
            "pose_output_allowed": False,
        }
        return {**body, "audit_sha256": _canonical_sha256(body)}
