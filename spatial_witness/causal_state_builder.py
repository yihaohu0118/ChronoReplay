"""Question-blind raw-pixel adapter for the causal spatial workspace.

The adapter deliberately has no question, option, task, prediction, or answer
argument. A declared visual extractor may turn one causal RGB frame into
entity, place, and local-motion observations; this module validates the
extractor output, binds it to the exact pixels, and applies it chronologically
to :mod:`spatial_witness.causal_workspace`.

This is an inference substrate, not an oracle-data loader. Benchmark
annotations and source-provided hidden geometry are forbidden extractor
inputs. A concrete extractor remains experimental until its own identity and
complete paired benchmark result are recorded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from PIL import Image

from spatial_witness.causal_workspace import (
    CausalSpatialWorkspace,
    EvidenceRef,
    WorkspaceError,
)


FORMAT = "spatial_witness.question_blind_state_builder.v1"
PIXEL_HASH_PROTOCOL = "sha256(pil_mode||pil_size||pixel_bytes)"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_EXTRACTOR_KINDS = frozenset(
    {"same_model_observer", "classical_vision", "frozen_geometric"}
)


@dataclass(frozen=True)
class ExtractorIdentity:
    name: str
    version: str
    kind: str
    configuration_sha256: str
    raw_video_pixels_only: bool = True
    external_generative_model_or_api: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip():
            raise WorkspaceError("extractor name and version must be nonempty")
        if self.kind not in _ALLOWED_EXTRACTOR_KINDS:
            raise WorkspaceError(f"unsupported extractor kind: {self.kind!r}")
        if not _SHA256.fullmatch(self.configuration_sha256):
            raise WorkspaceError("extractor configuration must have a SHA-256")
        if self.raw_video_pixels_only is not True:
            raise WorkspaceError("state extractor must consume raw video pixels only")
        if self.external_generative_model_or_api is not False:
            raise WorkspaceError("external generative extractor is forbidden")


@dataclass(frozen=True)
class EntityUpdate:
    name: str
    attributes: tuple[tuple[str, str], ...]
    world_position: tuple[float, float] | None = None
    place: str | None = None


@dataclass(frozen=True)
class PoseDelta:
    local_dx: float
    local_dy: float
    delta_yaw: float


@dataclass(frozen=True)
class FrameStateUpdate:
    entities: tuple[EntityUpdate, ...] = ()
    place: str | None = None
    pose_delta: PoseDelta | None = None


def pixel_sha256(image: Image.Image) -> str:
    """Hash exact PIL pixels using the release evaluator's protocol."""

    digest = hashlib.sha256()
    digest.update(str(image.mode).encode())
    digest.update(str(image.size).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def _key(value: Any, name: str) -> str:
    result = " ".join(str(value or "").strip().casefold().split())
    if not result:
        raise WorkspaceError(f"{name} must be nonempty")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkspaceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WorkspaceError(f"{name} must be finite")
    return result


def parse_frame_state_update(value: Mapping[str, Any]) -> FrameStateUpdate:
    """Validate one extractor result before any workspace mutation."""

    if not isinstance(value, Mapping):
        raise WorkspaceError("frame state update must be an object")
    payload = dict(value)
    allowed = {"entities", "place", "pose_delta"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise WorkspaceError(f"unknown frame state fields: {unknown}")

    raw_entities = payload.get("entities") or []
    if isinstance(raw_entities, (str, bytes)) or not isinstance(
        raw_entities, Sequence
    ):
        raise WorkspaceError("entities must be a list")
    entities: list[EntityUpdate] = []
    seen_entities: set[str] = set()
    for raw in raw_entities:
        if not isinstance(raw, Mapping):
            raise WorkspaceError("entity update must be an object")
        item = dict(raw)
        extras = sorted(
            set(item) - {"name", "attributes", "world_position", "place"}
        )
        if extras:
            raise WorkspaceError(f"unknown entity fields: {extras}")
        name = _key(item.get("name"), "entity name")
        if name in seen_entities:
            raise WorkspaceError(f"duplicate entity in one frame: {name}")
        seen_entities.add(name)
        attributes = item.get("attributes") or {}
        if not isinstance(attributes, Mapping):
            raise WorkspaceError("entity attributes must be an object")
        normalized: list[tuple[str, str]] = []
        for key, raw_value in attributes.items():
            if isinstance(raw_value, (dict, list, tuple, set)):
                raise WorkspaceError("entity attribute values must be scalar")
            normalized.append((_key(key, "attribute name"), str(raw_value)))
        position = None
        raw_position = item.get("world_position")
        if raw_position is not None:
            if (
                isinstance(raw_position, (str, bytes))
                or not isinstance(raw_position, Sequence)
                or len(raw_position) != 2
            ):
                raise WorkspaceError("entity world_position must contain x and y")
            position = (
                _finite(raw_position[0], "entity world x"),
                _finite(raw_position[1], "entity world y"),
            )
        entity_place = (
            _key(item["place"], "entity place")
            if item.get("place") is not None else None
        )
        entities.append(
            EntityUpdate(name, tuple(sorted(normalized)), position, entity_place)
        )

    place = (
        _key(payload["place"], "place")
        if payload.get("place") is not None
        else None
    )
    pose = None
    if payload.get("pose_delta") is not None:
        raw_pose = payload["pose_delta"]
        if not isinstance(raw_pose, Mapping):
            raise WorkspaceError("pose_delta must be an object")
        pose_payload = dict(raw_pose)
        expected = {"local_dx", "local_dy", "delta_yaw"}
        if set(pose_payload) != expected:
            raise WorkspaceError(
                f"pose_delta must contain exactly {sorted(expected)}"
            )
        pose = PoseDelta(
            _finite(pose_payload["local_dx"], "local_dx"),
            _finite(pose_payload["local_dy"], "local_dy"),
            _finite(pose_payload["delta_yaw"], "delta_yaw"),
        )
    if not entities and place is None and pose is None:
        raise WorkspaceError("frame state update contains no observations")
    return FrameStateUpdate(tuple(entities), place, pose)


class QuestionBlindStateBuilder:
    """Apply validated causal frame observations to one workspace."""

    def __init__(self, source_id: str, extractor: ExtractorIdentity) -> None:
        self.source_id = _key(source_id, "source_id")
        self.extractor = extractor
        self.workspace = CausalSpatialWorkspace()
        self._frames: set[int] = set()
        self._pose_initialized = False
        self.frame_updates = 0
        self.entity_observations = 0
        self.place_observations = 0
        self.pose_observations = 0
        self.metric_entity_observations = 0

    def ingest(
        self,
        image: Image.Image,
        *,
        frame_index: int,
        timestamp: float,
        update: Mapping[str, Any],
    ) -> EvidenceRef:
        """Validate a complete update, then bind and apply it chronologically."""

        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise WorkspaceError("frame_index must be a nonnegative integer")
        if frame_index in self._frames:
            raise WorkspaceError(f"duplicate state-builder frame: {frame_index}")
        frame_time = _finite(timestamp, "timestamp")
        if frame_time < 0 or frame_time < self.workspace.current_time:
            raise WorkspaceError("state-builder frames must be chronological")
        parsed = parse_frame_state_update(update)
        if parsed.pose_delta is not None and not self._pose_initialized:
            values = asdict(parsed.pose_delta).values()
            if any(abs(float(item)) > 1e-12 for item in values):
                raise WorkspaceError("the first extracted pose must initialize the origin")
        ref = EvidenceRef(
            self.source_id, frame_index, frame_time, pixel_sha256(image)
        )

        for entity in parsed.entities:
            self.workspace.observe_entity(
                entity.name,
                ref,
                attributes=dict(entity.attributes),
                world_position=entity.world_position,
                place=entity.place,
            )
        if parsed.place is not None:
            self.workspace.observe_place(parsed.place, ref)
        if parsed.pose_delta is not None:
            self.workspace.integrate_pose(
                ref,
                local_dx=parsed.pose_delta.local_dx,
                local_dy=parsed.pose_delta.local_dy,
                delta_yaw=parsed.pose_delta.delta_yaw,
            )
            self._pose_initialized = True
        self._frames.add(frame_index)
        self.frame_updates += 1
        self.entity_observations += len(parsed.entities)
        self.metric_entity_observations += sum(
            item.world_position is not None for item in parsed.entities
        )
        self.place_observations += int(parsed.place is not None)
        self.pose_observations += int(parsed.pose_delta is not None)
        return ref

    def seal_causal_horizon(self, timestamp: float) -> None:
        """Record the query-visible prefix boundary without adding a fact."""

        self.workspace.advance_causal_horizon(timestamp)

    def audit(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "format": FORMAT,
            "source_id": self.source_id,
            "extractor": asdict(self.extractor),
            "frame_updates": self.frame_updates,
            "entity_observations": self.entity_observations,
            "metric_entity_observations": self.metric_entity_observations,
            "place_observations": self.place_observations,
            "pose_observations": self.pose_observations,
            "state_version": self.workspace.state_version,
            "causal_horizon": self.workspace.current_time,
            "pixel_hash_protocol": PIXEL_HASH_PROTOCOL,
            "question_or_options_read": False,
            "benchmark_metadata_read": False,
            "answers_predictions_or_scores_read": False,
        }
        encoded = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return {**body, "audit_sha256": hashlib.sha256(encoded).hexdigest()}
